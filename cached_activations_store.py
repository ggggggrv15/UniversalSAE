"""
CachedActivationsStore for loading pre-cached activations from the Gemini model.
This follows the same interface as GemmaActivationsStore but loads activations from H5 files
instead of generating them on-the-fly.
"""

import torch
import h5py
import random
import os
import logging
from torch.utils.data import DataLoader, TensorDataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('cached_activations_store')

class CachedActivationsStore:
    """Loads and manages pre-cached activations from Gemini model."""
    
    def __init__(self, cfg):
        """Initialize the CachedActivationsStore with configuration."""
        # Determine cache file path
        cache_dir = cfg.get("activations_cache_dir", "cached_activations")
        cache_path = cfg.get("activations_cache_path", None)
        
        if cache_path is None:
            # Auto-detect the most recent cache file if not specified
            if not os.path.exists(cache_dir):
                raise FileNotFoundError(f"Cache directory {cache_dir} not found")
                
            cache_files = [f for f in os.listdir(cache_dir) if f.endswith('.h5') and not f.endswith('_temp.h5')]
            if not cache_files:
                raise FileNotFoundError(f"No cache files found in {cache_dir}")
                
            # Sort by modification time, newest first
            cache_files.sort(key=lambda x: os.path.getmtime(os.path.join(cache_dir, x)), reverse=True)
            cache_path = os.path.join(cache_dir, cache_files[0])
            logger.info(f"Auto-detected most recent cache file: {cache_path}")
        
        # Ensure cache file exists
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Cache file {cache_path} not found")
            
        # Open H5 file in read-only mode
        self.h5_file = h5py.File(cache_path, 'r')
        logger.info(f"Loaded activation cache from {cache_path}")
        
        # Get all document IDs
        self.doc_ids = list(self.h5_file.keys())
        logger.info(f"Found {len(self.doc_ids)} documents in cache")
        
        # Store configuration
        self.cfg = cfg
        self.device = cfg["device"]
        
        # Extract language information
        self.is_multilingual = "languages" in cfg and len(cfg["languages"]) > 1
        self.languages = cfg["languages"] if self.is_multilingual else [cfg["language"]]
        logger.info(f"Using languages: {self.languages}")
        
        # Validate languages in cache
        self._validate_languages()
        
        # Split into train/test sets (90/10 split)
        random.seed(cfg.get("seed", 42))
        random.shuffle(self.doc_ids)
        split_idx = int(len(self.doc_ids) * 0.9)
        self.train_docs = self.doc_ids[:split_idx]
        self.test_docs = self.doc_ids[split_idx:]
        logger.info(f"Split into {len(self.train_docs)} training and {len(self.test_docs)} test documents")
        
        # Initialize indices
        self.train_idx = 0
        self.test_idx = 0
        
        # For multilingual, we'll track indices separately for each language
        if self.is_multilingual:
            self.lang_train_idx = {lang: 0 for lang in self.languages}
            self.lang_test_idx = {lang: 0 for lang in self.languages}
        
        # Set up buffer parameters
        self.num_batches_in_buffer = cfg["num_batches_in_buffer"]
        self.model_batch_size = cfg["model_batch_size"]
        
        # Initialize buffer
        self.activation_buffer = self._fill_buffer()
        self._setup_dataloader()
        self.dataloader_iter = iter(self.dataloader)
    
    def _validate_languages(self):
        """Verify that the cache contains the required languages."""
        valid_docs = []
        
        # Check a sample of documents
        sample_size = min(100, len(self.doc_ids))
        sample_docs = random.sample(self.doc_ids, sample_size)
        
        for doc_id in sample_docs:
            # Check if document has all required languages
            if all(lang in self.h5_file[doc_id] for lang in self.languages):
                valid_docs.append(doc_id)
        
        if not valid_docs:
            raise ValueError(f"No documents found with all required languages: {self.languages}")
        
        logger.info(f"Validation: {len(valid_docs)}/{sample_size} sampled documents have all required languages")
    
    def _fill_buffer(self):
        """Fill the activation buffer with data from cache."""
        # For universal SAE, store paired activations
        if self.is_multilingual and self.cfg.get("sae_type") == "universal":
            paired_activations = {lang: [] for lang in self.languages}
            
            # Fill the buffer with activations from multiple documents
            for _ in range(self.num_batches_in_buffer):
                batch_activations = self._get_multilingual_batch(is_test=False)
                
                # Accumulate by language
                for lang in self.languages:
                    paired_activations[lang].append(batch_activations[lang])
            
            # Concatenate all activations by language
            for lang in self.languages:
                paired_activations[lang] = torch.cat(paired_activations[lang], dim=0)
                
            return paired_activations
        else:
            # Standard single-language buffer filling
            all_activations = []
            
            for _ in range(self.num_batches_in_buffer):
                batch = self._get_single_batch(is_test=False)
                all_activations.append(batch)
                
            return torch.cat(all_activations, dim=0)
    
    def _get_single_batch(self, is_test=False, language=None):
        """Get a batch of activations for a single language."""
        # Determine which language to use
        lang = language if language else self.languages[0]
        
        # Use appropriate document set and index
        docs = self.test_docs if is_test else self.train_docs
        idx = self.test_idx if is_test else self.train_idx
        
        # Initialize list to store activations
        activations = []
        
        # Collect activations until the batch size is met
        while len(activations) < self.model_batch_size:
            # Get next document
            if idx >= len(docs):
                idx = 0
                if not is_test:  # Only shuffle training docs
                    random.shuffle(docs)
            
            doc_id = docs[idx]
            idx += 1
            
            # Check if document has the required language
            if doc_id in self.h5_file and lang in self.h5_file[doc_id]:
                # Get activation tensor
                act = torch.tensor(
                    self.h5_file[doc_id][lang][()],
                    device=self.device,
                    dtype=torch.float32
                )
                activations.append(act)
        
        # Update appropriate index
        if is_test:
            self.test_idx = idx
        else:
            self.train_idx = idx
            
        # Stack tensors into a batch
        return torch.stack(activations[:self.model_batch_size])
    
    def _get_multilingual_batch(self, is_test=False):
        """Get a batch with activations from all languages."""
        # Use appropriate document set and index
        docs = self.test_docs if is_test else self.train_docs
        idx = self.test_idx if is_test else self.train_idx
        
        # Dictionary to store activations for each language
        lang_activations = {lang: [] for lang in self.languages}
        
        # Collect activations until we have enough for each language
        while min(len(acts) for acts in lang_activations.values()) < self.model_batch_size:
            # Get next document
            if idx >= len(docs):
                idx = 0
                if not is_test:  # Only shuffle training docs
                    random.shuffle(docs)
            
            doc_id = docs[idx]
            idx += 1
            
            # Check if document has all required languages
            has_all_langs = True
            for lang in self.languages:
                if doc_id not in self.h5_file or lang not in self.h5_file[doc_id]:
                    has_all_langs = False
                    break
            
            if has_all_langs:
                # Get activation for each language
                for lang in self.languages:
                    act = torch.tensor(
                        self.h5_file[doc_id][lang][()],
                        device=self.device,
                        dtype=torch.float32
                    )
                    lang_activations[lang].append(act)
        
        # Update appropriate index
        if is_test:
            self.test_idx = idx
        else:
            self.train_idx = idx
            
        # Stack tensors for each language
        return {
            lang: torch.stack(acts[:self.model_batch_size]) 
            for lang, acts in lang_activations.items()
        }
    
    def get_test_batch(self, language=None):
        """Get a batch of test activations for a single language."""
        if self.is_multilingual and not language:
            # If multilingual but no language specified, return parallel batches
            return self.get_multilingual_test_batch()
        
        # Get test batch for specified language
        batch = self._get_single_batch(is_test=True, language=language)
        return batch.reshape(-1, self.cfg["act_size"])
    
    def get_multilingual_test_batch(self):
        """Get a test batch with activations from all languages."""
        batch = self._get_multilingual_batch(is_test=True)
        return {
            lang: acts.reshape(-1, self.cfg["act_size"]) 
            for lang, acts in batch.items()
        }
    
    def _setup_dataloader(self):
        """Create a DataLoader for the activation buffer."""
        # For universal SAE, create a dataloader with paired language data
        if self.is_multilingual and self.cfg.get("sae_type") == "universal":
            # Create tensors for each language
            tensors = [self.activation_buffer[lang] for lang in self.languages]
            self.dataloader = DataLoader(
                TensorDataset(*tensors),
                batch_size=self.cfg["batch_size"], 
                shuffle=True
            )
        else:
            # Standard single tensor dataloader
            self.dataloader = DataLoader(
                TensorDataset(self.activation_buffer),
                batch_size=self.cfg["batch_size"], 
                shuffle=True
            )
    
    def next_batch(self):
        """Get the next batch of activations."""
        try:
            # Get the next batch from dataloader
            batch = next(self.dataloader_iter)
            
            # For universal SAE, return language tensors
            if self.is_multilingual and self.cfg.get("sae_type") == "universal":
                return batch  # This contains a tensor for each language
            else:
                # Standard return for single language
                return batch[0]
        except (StopIteration, AttributeError):
            # If iterator is exhausted or not initialized, refill buffer
            self.activation_buffer = self._fill_buffer()
            self._setup_dataloader()
            self.dataloader_iter = iter(self.dataloader)
            return self.next_batch()
    
    def __del__(self):
        """Clean up resources when object is destroyed."""
        # Close the H5 file if it's open
        if hasattr(self, 'h5_file') and self.h5_file:
            self.h5_file.close() 