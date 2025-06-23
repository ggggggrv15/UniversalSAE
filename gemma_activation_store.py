import torch
from activation_store import ActivationsStore
import random

class GemmaActivationsStore(ActivationsStore):
    """Adaptation of ActivationsStore for the Gemma model architecture."""
    
    def __init__(self, model, cfg):
        """Initialize the GemmaActivationsStore with a Gemma model."""
        # Don't call parent's __init__, implement our own initialization
        # to avoid the model.cfg.n_ctx access that causes the error
        
        # Store the model and configuration
        self.model = model
        # Load the Eurlex dataset (import here to avoid circular import)
        from eurlex_dataset import EurlexDataset
        self.dataset = EurlexDataset(cfg["dataset_path"])
        
        # Check if we're using multiple languages for Universal SAE
        self.is_multilingual = "languages" in cfg and len(cfg["languages"]) > 1
        
        if self.is_multilingual:
            self.languages = cfg["languages"]
            # Get documents available in all specified languages
            common_docs = self.dataset.get_common_documents(self.languages)
            all_docs = list(common_docs)
            random.shuffle(all_docs)
            
            # Store language -> docs mapping for single-language access
            self.lang_docs = {}
            for lang in self.languages:
                lang_docs = list(self.dataset.get_documents_by_language(lang))
                random.shuffle(lang_docs)
                self.lang_docs[lang] = lang_docs
        else:
            # Single language mode (backward compatibility)
            self.languages = [cfg["language"]]
            all_docs = list(self.dataset.get_documents_by_language(cfg["language"]))
            random.shuffle(all_docs)
        
        # Split into train/test sets (90/10 split)
        split_idx = int(len(all_docs) * 0.9)
        self.train_docs = all_docs[:split_idx]
        self.test_docs = all_docs[split_idx:]
        
        # For Gemma models, we access the context length differently
        self.context_size = min(cfg["seq_len"], model.config.max_position_embeddings)
        
        # Store the Gemma model specifically
        self.gemma_model = model
        # Track the target layer for extraction
        self.gemma_layer = cfg.get("layer", 8)
        
        # Set other configuration parameters
        self.model_batch_size = cfg["model_batch_size"]
        self.device = cfg["device"]
        self.num_batches_in_buffer = cfg["num_batches_in_buffer"]
        self.cfg = cfg
        self.tokenizer = model.tokenizer if hasattr(model, 'tokenizer') else model.config.tokenizer
        
        # Track current position in dataset
        self.train_idx = 0
        self.test_idx = 0
        # For multilingual, track indices per language
        if self.is_multilingual:
            self.lang_train_idx = {lang: 0 for lang in self.languages}
            self.lang_test_idx = {lang: 0 for lang in self.languages}
    
    def get_activations(self, batch_tokens: torch.Tensor):
        """Extract activations from the Gemma model.
        
        Args:
            batch_tokens: Tensor of token ids [batch_size, seq_len]
            
        Returns:
            Tensor of activations from final tokens [batch_size, act_size]
        """
        # Disable gradient computation for efficiency
        with torch.no_grad():
            # Process tokens through Gemma model to get hidden states
            # Set return_dict=True to ensure hidden_states are accessible
            outputs = self.model(
                batch_tokens, 
                output_hidden_states=True,
                return_dict=True
            )
            
            # Extract the specified middle layer's hidden states
            # Note: Gemma's hidden states are indexed from 0 to num_layers
            layer_outputs = outputs.hidden_states[self.gemma_layer]
            
            # CRITICAL: Only keep the activation from the final token in each sequence
            # Shape: [batch_size, hidden_size]
            final_token_activations = layer_outputs[:, -1, :]
            
        return final_token_activations
    
    def get_batch_tokens(self, is_test=False, language=None):
        """Get batch tokens adapted for Gemma tokenization.
        
        This overrides the parent method to use Gemma's tokenizer.
        """
        # Initialize a list to store tokens
        all_tokens = []
        
        # If language is specified, use it; otherwise use the default language
        lang = language if language else self.languages[0]
        
        # Use appropriate document set and index
        if self.is_multilingual and language:
            # Using single language from multilingual setup
            docs = self.lang_docs[lang]
            idx = self.lang_test_idx[lang] if is_test else self.lang_train_idx[lang]
        else:
            # Standard setup (common docs or single language mode)
            docs = self.test_docs if is_test else self.train_docs
            idx = self.test_idx if is_test else self.train_idx
        
        # Collect tokens until the required batch size is met
        while len(all_tokens) < self.model_batch_size * self.context_size:
            # Get next document
            if idx >= len(docs):
                idx = 0
                if not is_test:  # Only shuffle training docs
                    random.shuffle(docs)
            
            celex_id = docs[idx]
            text = self.dataset.get_translation(celex_id, lang)
            idx += 1
            
            if text:
                # Use Gemma tokenizer instead of HookedTransformer tokenizer
                tokens = self.model.tokenizer(
                    text, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=self.context_size
                ).input_ids.to(self.device)
                
                # Add the tokens to the list
                all_tokens.extend(tokens.flatten().tolist())
        
        # Update appropriate index
        if self.is_multilingual and language:
            if is_test:
                self.lang_test_idx[lang] = idx
            else:
                self.lang_train_idx[lang] = idx
        else:
            if is_test:
                self.test_idx = idx
            else:
                self.train_idx = idx
            
        # Convert the list of tokens to a tensor and reshape it
        token_tensor = torch.tensor(all_tokens, dtype=torch.long, device=self.device)[:self.model_batch_size * self.context_size]
        return token_tensor.view(self.model_batch_size, self.context_size)
        
    # Reimplement other necessary methods from ActivationsStore
    # that might rely on model.cfg or other attributes
    
    def _fill_buffer(self):
        # Initialize a list to store activations
        all_activations = []
        
        # For multilingual SAE, store paired activations
        if self.is_multilingual and self.cfg.get("sae_type") == "universal":
            paired_activations = {lang: [] for lang in self.languages}
            
            # Fill the buffer with activations from multiple batches
            for _ in range(self.num_batches_in_buffer):
                # Get tokens and activations for all languages
                batch_tokens = self.get_parallel_batch_tokens()
                batch_activations = self.get_parallel_activations(batch_tokens)
                torch.cuda.empty_cache()
                
                # Accumulate by language
                for lang, act in batch_activations.items():
                    paired_activations[lang].append(act)
            
            # Concatenate all activations by language
            for lang in self.languages:
                paired_activations[lang] = torch.cat(paired_activations[lang], dim=0)
                
            # Return the dictionary of language activations
            return paired_activations
        else:
            # Standard single-language buffer filling
            for _ in range(self.num_batches_in_buffer):
                # Get tokens for the current batch
                batch_tokens = self.get_batch_tokens()
                # Get activations for the current batch and reshape them
                activations = self.get_activations(batch_tokens).reshape(-1, self.cfg["act_size"])
                # Add the activations to the list
                all_activations.append(activations)
            # Concatenate all activations into a single tensor
            return torch.cat(all_activations, dim=0)
    
    def get_parallel_batch_tokens(self, is_test=False):
        """Get paired batch tokens for multiple languages (for Universal SAE)"""
        if not self.is_multilingual:
            raise ValueError("This method requires multiple languages in config")
            
        # Use test or train docs
        docs = self.test_docs if is_test else self.train_docs
        idx = self.test_idx if is_test else self.train_idx
        
        # Dictionary to store tokens for each language
        lang_tokens = {lang: [] for lang in self.languages}
        
        # Collect tokens until we have enough for each language
        while min(len(tokens) for tokens in lang_tokens.values()) < self.model_batch_size * self.context_size:
            # Get next document with all required translations
            if idx >= len(docs):
                idx = 0
                if not is_test:  # Only shuffle training docs
                    random.shuffle(docs)
                    
            celex_id = docs[idx]
            idx += 1
            
            # Get text for all languages
            all_langs_available = True
            texts = {}
            
            for lang in self.languages:
                text = self.dataset.get_translation(celex_id, lang)
                if not text:
                    all_langs_available = False
                    break
                texts[lang] = text
                
            if all_langs_available:
                # Process tokens for each language
                for lang, text in texts.items():
                    tokens = self.model.tokenizer(
                        text, 
                        return_tensors="pt", 
                        truncation=True, 
                        max_length=self.context_size
                    ).input_ids.to(self.device)
                    
                    lang_tokens[lang].extend(tokens.flatten().tolist())
        
        # Update index
        if is_test:
            self.test_idx = idx
        else:
            self.train_idx = idx
            
        # Convert token lists to tensors
        batch_tokens = {}
        for lang, tokens in lang_tokens.items():
            token_tensor = torch.tensor(tokens, dtype=torch.long, device=self.device)[:self.model_batch_size * self.context_size]
            batch_tokens[lang] = token_tensor.view(self.model_batch_size, self.context_size)
            
        return batch_tokens

    def get_parallel_activations(self, batch_tokens_dict):
        """Get activations for multiple languages in parallel"""
        activations = {}
        for lang, tokens in batch_tokens_dict.items():
            activations[lang] = self.get_activations(tokens).reshape(-1, self.cfg["act_size"])
        return activations

    def get_multilingual_test_batch(self):
        """Get a test batch with activations from all languages"""
        batch_tokens = self.get_parallel_batch_tokens(is_test=True)
        return self.get_parallel_activations(batch_tokens)

    def get_test_batch(self, language=None):
        """Get a batch of test activations for a single language"""
        if self.is_multilingual and not language:
            # If multilingual but no language specified, return parallel batches
            return self.get_multilingual_test_batch()
            
        batch_tokens = self.get_batch_tokens(is_test=True, language=language)
        activations = self.get_activations(batch_tokens)
        return activations.reshape(-1, self.cfg["act_size"])
        
    def next_batch(self):
        try:
            # Get the next batch
            batch = next(self.dataloader_iter)
            
            # For universal SAE, return a tuple of language tensors
            if self.is_multilingual and self.cfg.get("sae_type") == "universal":
                return batch  # This contains a tensor for each language
            else:
                # Standard return for single language
                return batch[0]
                
        except (StopIteration, AttributeError):
            # If the iterator is exhausted or not initialized, refill the buffer
            self.activation_buffer = self._fill_buffer()
            # Use the utility method to get a DataLoader
            self._setup_dataloader()
            # Initialize the iterator for the DataLoader
            self.dataloader_iter = iter(self.dataloader)
            # Return the first batch from the new iterator
            return self.next_batch()
            
    def _setup_dataloader(self):
        """Create a DataLoader for the activation buffer"""
        from torch.utils.data import DataLoader, TensorDataset
        
        # For universal SAE, create a dataloader with paired language data
        if self.is_multilingual and self.cfg.get("sae_type") == "universal":
            # Create tensors for each language
            tensors = [self.activation_buffer[lang] for lang in self.languages]
            self.dataloader = DataLoader(
                TensorDataset(*tensors),  # Unpack tensors as separate arguments
                batch_size=self.cfg["batch_size"], 
                shuffle=True
            )
        else:
            # Standard single tensor dataloader
            self.dataloader = DataLoader(
                TensorDataset(self.activation_buffer),  # Pass single tensor directly
                batch_size=self.cfg["batch_size"], 
                shuffle=True
            ) 