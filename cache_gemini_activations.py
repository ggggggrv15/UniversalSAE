#!/usr/bin/env python3
"""
Script to pre-cache activations from Gemini model for the Universal SAE project.
This script can be run independently and will resume if interrupted.

Usage:
    python cache_gemini_activations.py --config_path path/to/config.json
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#os.environ['HF_TOKEN'] = 
import h5py
import json
import torch
import argparse
import tqdm
import time
import logging
from datetime import datetime
from pathlib import Path

# Import Gemma-specific functions and classes
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import EurlexDataset
from eurlex_dataset import EurlexDataset
from config import get_default_cfg

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('cache_gemini_activations')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Cache Gemini model activations")
    parser.add_argument('--config_path', type=str, help='Path to config JSON file')
    parser.add_argument('--output_dir', type=str, default='cached_activations',
                        help='Directory to store cached activations')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for processing')
    parser.add_argument('--max_docs', type=int, default=None,
                        help='Maximum number of documents to process (for testing)')
    parser.add_argument('--test_mode', action='store_true',
                        help='Run in test mode with a small subset of data')
    return parser.parse_args()

def load_config(config_path=None):
    """Load configuration from file or use default."""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            logger.info(f"Loading config from {config_path}")
            cfg = json.load(f)
    else:
        logger.info("Using default config")
        cfg = get_default_cfg()
        
    return cfg

def load_gemma_model(cfg):
    """Load Gemma model, downloading if necessary."""
    try:
        # Use Gemma 2B which is open access instead of 1B which requires authentication
        model_repo = "google/gemma-2b"
        local_path = cfg.get("gemma_local_path", "models/gemma-2b")
        local_path = local_path.replace("gemma-1b", "gemma-2b")
        
        logger.info(f"Using {model_repo} model")
        
        # Check if model exists locally
        if os.path.exists(local_path) and os.path.isfile(os.path.join(local_path, "pytorch_model.bin")):
            logger.info(f"Loading Gemma model from {local_path}")
            model_path = local_path
        else:
            logger.info(f"Downloading Gemma model to {local_path}")
            # Create directory if it doesn't exist
            os.makedirs(local_path, exist_ok=True)
            
            try:
                # Download model from Hugging Face Hub
                model_path = snapshot_download(
                    repo_id=model_repo,
                    local_dir=local_path,
                    token=os.environ.get("HF_TOKEN"),  # Try with token if available
                )
            except Exception as e:
                # If authentication is the issue, provide clear instructions
                error_message = str(e).lower()
                if "authentication" in error_message or "invalid username or password" in error_message or "401" in error_message:
                    logger.error("\n" + "="*80)
                    logger.error("AUTHENTICATION ERROR: Gemma models require Hugging Face authentication.")
                    logger.error("\nTo fix this issue:")
                    logger.error("1. Create an account at https://huggingface.co/ if you don't have one")
                    logger.error("2. Accept the Gemma model license at https://huggingface.co/google/gemma-2b")
                    logger.error("3. Create an access token at https://huggingface.co/settings/tokens")
                    logger.error("4. Run: export HF_TOKEN=your_token_here")
                    logger.error("5. Restart the script")
                    logger.error("="*80 + "\n")
                raise RuntimeError(f"Failed to download model: {str(e)}")
        
        try:
            # Load model with output_hidden_states=True to access middle layers
            logger.info(f"Loading model from {model_path}")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if cfg["device"] != "cpu" else torch.float32,
                device_map=cfg["device"],
                output_hidden_states=True,
                return_dict=True,
                token=os.environ.get("HF_TOKEN"),
            )
            model.eval()  # Set to evaluation mode
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path, token=os.environ.get("HF_TOKEN"))
            # Attach tokenizer to model for convenience
            model.tokenizer = tokenizer
            
            return model
        except Exception as e:
            # If loading fails, provide helpful error message
            logger.error("\n" + "="*80)
            logger.error(f"ERROR LOADING MODEL: {str(e)}")
            logger.error("\nTroubleshooting tips:")
            logger.error("1. Make sure you have enough disk space")
            logger.error("2. Make sure you have set up proper authentication")
            logger.error("3. If you already downloaded the model files, check if they're corrupted")
            logger.error("="*80 + "\n")
            raise
    except ImportError:
        logger.error("Please install transformers and huggingface_hub: pip install transformers huggingface_hub")
        raise

def get_cache_file_path(output_dir, cfg, suffix=None):
    """Generate cache file path based on configuration."""
    # Create a descriptive filename
    layer = cfg.get("layer", 9)
    model_name = cfg["model_name"].replace("/", "_")
    timestamp = datetime.now().strftime("%Y%m%d")
    
    # Add suffix for creating temporary files
    suffix_str = f"_{suffix}" if suffix else ""
    
    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    return os.path.join(output_dir, f"{model_name}_layer{layer}{suffix_str}_{timestamp}.h5")

def get_processed_docs(cache_file_path):
    """Get set of document IDs that have already been processed."""
    processed_docs = set()
    
    if os.path.exists(cache_file_path):
        try:
            with h5py.File(cache_file_path, 'r') as f:
                processed_docs = set(f.keys())
            logger.info(f"Found {len(processed_docs)} previously processed documents")
        except Exception as e:
            logger.warning(f"Could not read existing cache file: {e}")
    
    return processed_docs

def process_document_batch(model, dataset, doc_ids, cfg, temp_file, seq_len, batch_size=4):
    """Process a batch of documents and cache their activations."""
    # Group documents into mini-batches for efficient processing
    mini_batches = [doc_ids[i:i + batch_size] for i in range(0, len(doc_ids), batch_size)]
    
    for mini_batch in mini_batches:
        # Get all available languages for each document
        for doc_id in mini_batch:
            # Get document to find available languages
            doc = dataset.get_document(doc_id)
            if not doc:
                continue
                
            # Get all available languages for this document
            available_languages = doc[0]['included']
            
            # Create group for document
            if doc_id not in temp_file:
                temp_file.create_group(doc_id)
                
            # Process each available language
            for lang in available_languages:
                # Get text for this language
                text = dataset.get_translation(doc_id, lang)
                if not text:
                    continue
                    
                # Tokenize text
                encoded = model.tokenizer(
                    text,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=seq_len,
                    return_attention_mask=True
                )
                
                # Move to device
                input_ids = encoded.input_ids.to(cfg["device"])
                attention_mask = encoded.attention_mask.to(cfg["device"])
                
                # Process tokens to get activations
                with torch.no_grad():
                    try:
                        outputs = model(
                            input_ids,
                            attention_mask=attention_mask,
                            output_hidden_states=True,
                            return_dict=True
                        )
                        
                        # Extract activations from specified layer
                        layer_outputs = outputs.hidden_states[cfg["layer"]]
                        
                        # Get final non-padding token's activation
                        last_token_pos = attention_mask.sum().item() - 1
                        final_activation = layer_outputs[0, last_token_pos].cpu().numpy()
                        
                        # Store in HDF5 file
                        temp_file[doc_id].create_dataset(lang, data=final_activation)
                        
                    except Exception as e:
                        logger.error(f"Error processing {doc_id}/{lang}: {e}")
                        continue
            
            # Clear GPU cache
            torch.cuda.empty_cache()

def merge_cache_files(temp_file_path, final_file_path, processed_docs):
    """Merge temporary cache file into final cache file."""
    # If final file doesn't exist, just rename temp file
    if not os.path.exists(final_file_path):
        os.rename(temp_file_path, final_file_path)
        logger.info(f"Created new cache file: {final_file_path}")
        return
        
    # Otherwise, copy data from temp to final
    with h5py.File(temp_file_path, 'r') as temp_file, h5py.File(final_file_path, 'a') as final_file:
        # Copy each document group
        for doc_id in tqdm.tqdm(temp_file.keys(), desc="Merging files"):
            if doc_id in final_file:
                # Skip if already in final file
                continue
                
            # Create group in final file
            final_file.create_group(doc_id)
            
            # Copy all language datasets
            for lang in temp_file[doc_id].keys():
                data = temp_file[doc_id][lang][()]
                final_file[doc_id].create_dataset(lang, data=data)
                
            # Add to processed docs
            processed_docs.add(doc_id)
    
    # Remove temporary file
    os.remove(temp_file_path)
    logger.info(f"Merged data into {final_file_path}")

def cache_gemini_activations(cfg, output_dir, max_docs=None, test_mode=False):
    """Main function to cache Gemini model activations for all available translations."""
    logger.info(f"Caching activations for all available translations")
    
    # Load dataset
    dataset = EurlexDataset(cfg["dataset_path"])
    
    # Get all document IDs
    all_docs = list(dataset.data.keys())
    logger.info(f"Found {len(all_docs)} documents in dataset")
    
    # Limit if needed
    if max_docs:
        all_docs = all_docs[:max_docs]
        logger.info(f"Limited to {max_docs} documents for processing")
    
    if test_mode:
        # Override with small test set
        all_docs = all_docs[:20]
        logger.info(f"TEST MODE: Processing only 20 documents")
    
    # Generate file paths
    cache_file_path = get_cache_file_path(output_dir, cfg)
    temp_file_path = get_cache_file_path(output_dir, cfg, suffix="temp")
    
    # Check for already processed documents
    processed_docs = get_processed_docs(cache_file_path)
    
    # Filter out already processed documents
    docs_to_process = [doc for doc in all_docs if doc not in processed_docs]
    logger.info(f"Documents to process: {len(docs_to_process)} (skipping {len(processed_docs)} already processed)")
    
    if not docs_to_process:
        logger.info("All documents already processed. Nothing to do.")
        return cache_file_path
    
    # Load Gemma model
    model = load_gemma_model(cfg)
    
    # Set sequence length for tokenization
    seq_len = min(cfg.get("seq_len", 1024), model.config.max_position_embeddings)
    print(f"Sequence length: {seq_len}")
    
    # Process in batches to allow for resume
    batch_size_docs = 200  # Process 100 documents at a time before saving
    doc_batches = [docs_to_process[i:i + batch_size_docs] for i in range(0, len(docs_to_process), batch_size_docs)]
    
    for batch_idx, doc_batch in enumerate(doc_batches):
        logger.info(f"Processing batch {batch_idx+1}/{len(doc_batches)} ({len(doc_batch)} documents)")
        
        # Create temporary file for this batch
        with h5py.File(temp_file_path, 'w') as temp_file:
            # Process the batch of documents with all their translations
            process_document_batch(model, dataset, doc_batch, cfg, temp_file, seq_len, batch_size=4)
        
        # Merge temporary file into main cache file
        merge_cache_files(temp_file_path, cache_file_path, processed_docs)
        
        # Report progress
        logger.info(f"Progress: {len(processed_docs)}/{len(all_docs)} documents processed")
    
    
    logger.info(f"Caching complete! Activations stored in {cache_file_path}")
    logger.info(f"Total documents processed: {len(processed_docs)}")
    
    return cache_file_path

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    cfg = load_config(args.config_path)
    
    # Set output directory
    output_dir = args.output_dir
    
    # Start caching
    start_time = time.time()
    
    try:
        cache_file_path = cache_gemini_activations(
            cfg, 
            output_dir, 
            max_docs=args.max_docs,
            test_mode=args.test_mode
        )
        
        logger.info(f"Cache file created: {cache_file_path}")
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user. Progress has been saved.")
    except Exception as e:
        logger.error(f"Error during caching: {e}", exc_info=True)
    
    # Report total time
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info(f"Total time: {int(hours)}h {int(minutes)}m {int(seconds)}s") 