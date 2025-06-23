import torch
from torch.utils.data import DataLoader, TensorDataset
from transformer_lens.hook_points import HookedRootModule
from eurlex_dataset import EurlexDataset
import random

class ActivationsStore:
    
    def __init__(self,model: HookedRootModule, cfg: dict,):
        self.model = model
        self.dataset = EurlexDataset(cfg["dataset_path"])
        self.is_multilingual = "languages" in cfg and len(cfg["languages"]) > 1
        
        if self.is_multilingual:
            self.languages = cfg["languages"]
            common_docs = self.dataset.get_common_documents(self.languages)
            all_docs = list(common_docs)
            random.shuffle(all_docs)
            self.lang_docs = {}
            for lang in self.languages:
                lang_docs = list(self.dataset.get_documents_by_language(lang))
                random.shuffle(lang_docs)
                self.lang_docs[lang] = lang_docs
        else:
            self.languages = [cfg["language"]]
            all_docs = list(self.dataset.get_documents_by_language(cfg["language"]))
            random.shuffle(all_docs)
        
        split_idx = int(len(all_docs) * 0.9)
        self.train_docs = all_docs[:split_idx]
        self.test_docs = all_docs[split_idx:]
        self.hook_point = cfg["hook_point"]
        self.context_size = min(cfg["seq_len"], model.cfg.n_ctx)
        print(f"Context size: {self.context_size}")
        self.model_batch_size = cfg["model_batch_size"]
        self.device = cfg["device"]
        self.num_batches_in_buffer = cfg["num_batches_in_buffer"]
        self.cfg = cfg
        self.tokenizer = model.tokenizer
        self.train_idx = 0
        self.test_idx = 0
        if self.is_multilingual:
            self.lang_train_idx = {lang: 0 for lang in self.languages}
            self.lang_test_idx = {lang: 0 for lang in self.languages}

    def get_batch_tokens(self, is_test=False, language=None):
        all_tokens = []
        lang = language if language else self.languages[0]
        
        if self.is_multilingual and language:
            docs = self.lang_docs[lang]
            idx = self.lang_test_idx[lang] if is_test else self.lang_train_idx[lang]
        else:
            docs = self.test_docs if is_test else self.train_docs
            idx = self.test_idx if is_test else self.train_idx
        
        while len(all_tokens) < self.model_batch_size * self.context_size:
            if idx >= len(docs):
                idx = 0
                if not is_test:
                    random.shuffle(docs)
        
            celex_id = docs[idx]
            text = self.dataset.get_translation(celex_id, lang)
            idx += 1

            if text:
                tokens = self.model.to_tokens(text, truncate=False, move_to_device=True, prepend_bos=True).squeeze(0)
                
                # For longer documents, take the last 'context_size' tokens,
                # so we capture the final token's activation.
                if tokens.size(0) > self.context_size:
                    tokens = tokens[-self.context_size:]
                all_tokens.extend(tokens.flatten().tolist())

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
        
        token_tensor = torch.tensor(all_tokens, dtype=torch.long, device=self.device)[:self.model_batch_size * self.context_size]
        return token_tensor.view(self.model_batch_size, self.context_size)

    def get_activations(self, batch_tokens: torch.Tensor):
        with torch.no_grad():
            _, cache = self.model.run_with_cache(
                batch_tokens,
                names_filter=[self.hook_point],
                stop_at_layer=self.cfg["layer"] + 1,
            )
        return cache[self.hook_point]

    def get_parallel_batch_tokens(self, is_test=False):
    
        docs = self.test_docs if is_test else self.train_docs
        idx = self.test_idx if is_test else self.train_idx
        lang_tokens = {lang: [] for lang in self.languages}
        
        while min(len(tokens) for tokens in lang_tokens.values()) < self.model_batch_size * self.context_size:
            if idx >= len(docs):
                idx = 0
                if not is_test:
                    random.shuffle(docs)
    
            celex_id = docs[idx]
            idx += 1
            
            all_langs_available = True
            texts = {}
            
            for lang in self.languages:
                text = self.dataset.get_translation(celex_id, lang)
                if not text:
                    all_langs_available = False
                    break
                texts[lang] = text
                
            if all_langs_available:
                for lang, text in texts.items():
                    tokens = self.model.to_tokens(text, truncate=False, move_to_device=True, prepend_bos=True).squeeze(0)
                    if tokens.size(0) > self.context_size:
                        tokens = tokens[-self.context_size:]
                    lang_tokens[lang].extend(tokens.flatten().tolist())
        
        if is_test:
            self.test_idx = idx
        else:
            self.train_idx = idx
            
        batch_tokens = {}
        for lang, tokens in lang_tokens.items():
            token_tensor = torch.tensor(tokens, dtype=torch.long, device=self.device)[:self.model_batch_size * self.context_size]
            batch_tokens[lang] = token_tensor.view(self.model_batch_size, self.context_size)
            
        return batch_tokens

    def get_parallel_activations(self, batch_tokens_dict):
        activations = {}
        for lang, tokens in batch_tokens_dict.items():
            activations[lang] = self.get_activations(tokens).reshape(-1, self.cfg["act_size"])
        return activations

    def get_multilingual_test_batch(self):
        batch_tokens = self.get_parallel_batch_tokens(is_test=True)
        return self.get_parallel_activations(batch_tokens)

    def get_test_batch(self, language=None):
        if self.is_multilingual and not language:
            return self.get_multilingual_test_batch()

        batch_tokens = self.get_batch_tokens(is_test=True, language=language)
        activations = self.get_activations(batch_tokens)
        return activations.reshape(-1, self.cfg["act_size"])

    def _fill_buffer(self):
        all_activations = []
        
        if self.is_multilingual and self.cfg.get("sae_type") == "universal":
            paired_activations = {lang: [] for lang in self.languages}
            
            for _ in range(self.num_batches_in_buffer):
                batch_tokens = self.get_parallel_batch_tokens()
                batch_activations = self.get_parallel_activations(batch_tokens)
                
                for lang, act in batch_activations.items():
                    paired_activations[lang].append(act)
            
            for lang in self.languages:
                paired_activations[lang] = torch.cat(paired_activations[lang], dim=0)
                
            return paired_activations
        
        else:
            for _ in range(self.num_batches_in_buffer):
                batch_tokens = self.get_batch_tokens()
                activations = self.get_activations(batch_tokens).reshape(-1, self.cfg["act_size"])
                all_activations.append(activations)
            return torch.cat(all_activations, dim=0)

    def _get_dataloader(self):
        if self.is_multilingual and self.cfg.get("sae_type") == "universal":
            tensors = [self.activation_buffer[lang] for lang in self.languages]
            return DataLoader(TensorDataset(*tensors), batch_size=self.cfg["batch_size"], shuffle=True)
        else:
            return DataLoader(TensorDataset(self.activation_buffer), batch_size=self.cfg["batch_size"], shuffle=True)

    def next_batch(self):
        try:
            batch = next(self.dataloader_iter)
            if self.is_multilingual and self.cfg.get("sae_type") == "universal":
                return batch  # Contains a tensor for each language
            else:
                return batch[0]
                
        except (StopIteration, AttributeError):
            self.activation_buffer = self._fill_buffer()
            self.dataloader = self._get_dataloader()
            self.dataloader_iter = iter(self.dataloader)
            return self.next_batch()