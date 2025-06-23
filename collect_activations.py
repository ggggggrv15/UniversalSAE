import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WORLD_SIZE"] = "1"
import json
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer
from eurlex_dataset import EurlexDataset
import os
from typing import Dict, List, Tuple, Optional, Any
import gc

class ActivationCollector:
    def __init__(
        self,
        model_name: str = "gpt2-small",
        hook_point: str = "blocks.8.hook_resid_pre",
        batch_size: int = 8,
        device: str = "cuda:0",
        input_path: str = "/home/aidanm/scratch/DL/Thesis/processed/eurlex_processed.json",
        output_path: str = "/scratch/aidanm/eurlex_activations.json",
        save_frequency: int = 1000  # Save after processing this many documents
    ):
        # Initialize parameters
        self.model_name = model_name
        self.hook_point = hook_point
        self.batch_size = batch_size
        self.device = device
        self.input_path = input_path
        self.output_path = output_path
        self.save_frequency = save_frequency
        
        # Load or initialize the output database
        self.output_db = self._load_or_init_output_db()
        
        # Load the model
        print(f"Loading {model_name}...")
        self.model = HookedTransformer.from_pretrained(
            model_name,
            device=device
        )
        
        # Load the dataset
        self.dataset = EurlexDataset(input_path)
        
        # Track progress
        self.docs_processed = 0
    
    def _load_or_init_output_db(self) -> Dict:
        """Load existing output database or initialize new one"""
        if os.path.exists(self.output_path):
            print(f"Loading existing output database from {self.output_path}")
            with open(self.output_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _save_output_db(self):
        """Save current state of output database"""
        print(f"\nSaving output database to {self.output_path}")
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(self.output_db, f)
    
    def _get_final_activation(self, text: str) -> Optional[torch.Tensor]:
        """Get the final activation vector for a text string"""
        if not text:
            return None
            
        # Tokenize the text
        tokens = self.model.to_tokens(text, truncate=True)
        
        # Get activations using hooks
        _, cache = self.model.run_with_cache(
            tokens,
            names_filter=[self.hook_point],
        )
        
        # Get the last activation vector
        final_activation = cache[self.hook_point][0, -1].cpu().numpy().tolist()
        
        # Clear cache
        del cache
        torch.cuda.empty_cache()
        
        return final_activation
    
    def _process_batch(
        self,
        batch: List[Tuple[str, str, str]]  # List of (celex_id, lang, text)
    ) -> List[Tuple[str, str, Optional[List[float]]]]:
        """Process a batch of texts and return their final activations"""
        results = []
        valid_texts = [(i, text) for i, (_, _, text) in enumerate(batch) if text]
        
        if not valid_texts:
            return [(cid, lang, None) for cid, lang, _ in batch]
        
        try:
            # Prepare batch of valid texts
            indices, texts = zip(*valid_texts)
            
            # Add max length truncation
            tokens = self.model.to_tokens(list(texts), truncate=True)
            
            # Clear GPU memory before running model
            torch.cuda.empty_cache()
            
            # Get activations using hooks
            _, cache = self.model.run_with_cache(
                tokens,
                names_filter=[self.hook_point],
            )
            
            # Get final activations and immediately move to CPU
            final_activations = cache[self.hook_point][:, -1].cpu().numpy().tolist()
            
            # Clear cache immediately
            del cache
            torch.cuda.empty_cache()
            
            # Create mapping of results
            activation_map = {i: act for i, act in zip(indices, final_activations)}
            
            # Build results list maintaining original order
            for i, (cid, lang, text) in enumerate(batch):
                act = activation_map.get(i, None) if text else None
                results.append((cid, lang, act))
            
            return results
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\nOOM error with batch size {len(valid_texts)}. Trying one at a time...")
                torch.cuda.empty_cache()
                
                # Process one text at a time
                results = []
                for i, (cid, lang, text) in enumerate(batch):
                    if not text:
                        results.append((cid, lang, None))
                        continue
                        
                    try:
                        tokens = self.model.to_tokens([text], truncate=True)
                        _, cache = self.model.run_with_cache(
                            tokens,
                            names_filter=[self.hook_point],
                        )
                        activation = cache[self.hook_point][0, -1].cpu().numpy().tolist()
                        del cache
                        torch.cuda.empty_cache()
                        results.append((cid, lang, activation))
                    except:
                        print(f"\nFailed to process text for {cid}, {lang}")
                        results.append((cid, lang, None))
                        
                return results
            else:
                raise e
    
    def process_all(self):
        """Process all documents in the dataset"""
        # Get all document IDs
        all_docs = list(self.dataset.data.keys())
        
        # Create progress bar
        pbar = tqdm(total=len(all_docs), desc="Processing documents")
        
        current_batch = []
        
        for celex_id in all_docs:
            doc = self.dataset.get_document(celex_id)
            if not doc:
                continue
                
            # Skip if document is already fully processed
            if celex_id in self.output_db:
                all_processed = all(
                    isinstance(trans.get(lang), tuple) 
                    for trans in self.output_db[celex_id][1:]
                    for lang in trans
                )
                if all_processed:
                    pbar.update(1)
                    continue
            
            # Initialize document in output DB if needed
            if celex_id not in self.output_db:
                self.output_db[celex_id] = [doc[0]]  # Copy languages list
            
            # Process each translation
            for trans_dict in doc[1:]:
                for lang, text in trans_dict.items():
                    # Skip if already processed
                    if celex_id in self.output_db and \
                       any(lang in d and isinstance(d[lang], tuple) 
                           for d in self.output_db[celex_id][1:]):
                        continue
                    
                    # Add to current batch
                    current_batch.append((celex_id, lang, text))
                    
                    # Process batch if full
                    if len(current_batch) >= self.batch_size:
                        results = self._process_batch(current_batch)
                        
                        # Update output database
                        for (cid, lang, activation) in results:
                            if activation is not None:
                                # Find or create translation dict
                                found = False
                                for trans in self.output_db[cid][1:]:
                                    if lang in trans:
                                        trans[lang] = (text, activation)
                                        found = True
                                        break
                                if not found:
                                    self.output_db[cid].append({lang: (text, activation)})
                        
                        current_batch = []
                        gc.collect()
            
            self.docs_processed += 1
            pbar.update(1)
            
            # Save periodically
            if self.docs_processed % self.save_frequency == 0:
                self._save_output_db()
        
        # Process remaining batch
        if current_batch:
            results = self._process_batch(current_batch)
            for (cid, lang, activation) in results:
                if activation is not None:
                    # Find or create translation dict
                    found = False
                    for trans in self.output_db[cid][1:]:
                        if lang in trans:
                            trans[lang] = (text, activation)
                            found = True
                            break
                    if not found:
                        self.output_db[cid].append({lang: (text, activation)})
        
        # Final save
        self._save_output_db()
        pbar.close()

if __name__ == "__main__":
    collector = ActivationCollector(
        model_name="gpt2-small",
        hook_point="blocks.10.hook_resid_pre",
        batch_size=8,
        device="cuda:0",
        save_frequency=100
    )
    collector.process_all() 