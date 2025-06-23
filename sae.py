import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

class BaseAutoencoder(nn.Module):
    """Base class for autoencoder models."""

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        torch.manual_seed(self.cfg["seed"])

        self.b_dec = nn.Parameter(torch.zeros(self.cfg["act_size"]))
        self.b_enc = nn.Parameter(torch.zeros(self.cfg["dict_size"]))

        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.cfg["act_size"], self.cfg["dict_size"])
            )
        )
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.cfg["dict_size"], self.cfg["act_size"])
            )
        )
        self.W_dec.data[:] = self.W_enc.t().data
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

        self.num_batches_not_active = torch.zeros((self.cfg["dict_size"],)).to(
            cfg["device"]
        )

        self.to(cfg["dtype"]).to(cfg["device"])

    def preprocess_input(self, x):
        if self.cfg.get("input_unit_norm", False):
            x_mean = x.mean(dim=-1, keepdim=True)
            x = x - x_mean
            x_std = x.std(dim=-1, keepdim=True)
            x = x / (x_std + 1e-5)
            return x, x_mean, x_std
        else:
            return x, None, None

    def postprocess_output(self, x_reconstruct, x_mean, x_std):
        if self.cfg.get("input_unit_norm", False):
            x_reconstruct = x_reconstruct * x_std + x_mean
        return x_reconstruct

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(
            -1, keepdim=True
        ) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
        self.W_dec.data = W_dec_normed

    def update_inactive_features(self, acts):
        self.num_batches_not_active += (acts.sum(0) == 0).float()
        self.num_batches_not_active[acts.sum(0) > 0] = 0


class UniversalSAE(BaseAutoencoder):
    """
    UniversalSAE can encode and decode convergent cross-linguistic representations. 
    It uses a shared encoding space and separate decoders for each language.
    It can be thought of as a special case of the BatchTopKSAE, where USAE is a
    pair of BatchTopKSAEs, one for each language, except that they both share a single encoder,
    encodings of activations from either language can be used to reconstruct activations 
    from either decoder, and encodings of activations from one language are are similar to the
    the encodings of the activations of the translations of the same text in the other language.
    These two languages are specified in the config file, but the notation here
    arbitrarily assumes wolog that they are English (en) and Spanish (es).
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        device = cfg["device"]        
        self.bias = cfg["ratio"]
        self.bilingual_exposure = cfg["bilingual_exposure"]
        del self.W_dec, self.b_dec
        
        # Language 1 decoder
        self.W_dec_en = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.cfg["dict_size"], self.cfg["act_size"], device=device)
            )
        )
        self.b_dec_en = nn.Parameter(torch.zeros(self.cfg["act_size"], device=device))
        
        # Language 2 decoder
        self.W_dec_es = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.cfg["dict_size"], self.cfg["act_size"], device=device)
            )
        )
        self.b_dec_es = nn.Parameter(torch.zeros(self.cfg["act_size"], device=device))
        
        self.W_dec_en.data[:] = self.W_enc.t().data
        self.W_dec_en.data[:] = self.W_dec_en / self.W_dec_en.norm(dim=-1, keepdim=True)
        self.W_dec_es.data[:] = self.W_enc.t().data
        self.W_dec_es.data[:] = self.W_dec_es / self.W_dec_es.norm(dim=-1, keepdim=True)
        
        self.lang_usage = torch.zeros(2, device=device)

    def select_language(self):
        """
        If bias=0.2, Language 1 (lang_idx=0) will be sampled 80% of the time
        and Language 2 (lang_idx=1) will be sampled 20% of the time.
        """
        bias_val = float(self.bias)
        if bias_val <= 0:
            lang_idx = 0
        elif bias_val >= 1:
            lang_idx = 1
        else:
            lang_idx = 1 if torch.rand(1, device=self.cfg["device"]).item() < bias_val else 0
        # For logging
        self.lang_usage[lang_idx] += 1
        return lang_idx

    def encode(self, x):
        
        # Shared encoder, with batch-wise top-K selection (Bussmann et al., 2024, line 351)
        
        x = x.to(self.W_enc.device, non_blocking=True)
        x, x_mean, x_std = self.preprocess_input(x)
        pre_acts = x @ self.W_enc + self.b_enc
        acts = F.relu(pre_acts)
        
        batch_size = x.shape[0]
        total_k = self.cfg["top_k"] * batch_size
        flat_acts = acts.flatten()
        topk_values, topk_indices = torch.topk(flat_acts, total_k, dim=0)
        acts_topk_flat = torch.zeros_like(flat_acts)
        acts_topk_flat.scatter_(0, topk_indices, topk_values)
        acts_topk = acts_topk_flat.reshape(acts.shape)
        
        del pre_acts, flat_acts, topk_values, topk_indices, acts_topk_flat
        torch.cuda.empty_cache()
        
        return acts_topk, x, x_mean, x_std, acts

    def decode(self, acts):
        
        x_reconstruct_en = acts @ self.W_dec_en + self.b_dec_en
        x_reconstruct_es = acts @ self.W_dec_es + self.b_dec_es
        
        return x_reconstruct_en, x_reconstruct_es

    def get_auxiliary_loss(self, x_en, x_es, x_reconstruct_en, x_reconstruct_es, acts, lang_idx):
        """
        Encourages the model to use dead features (neurons which haven't fired in a while) to reconstruct 
        the residual between the input and current reconstruction in both languages.
        """
        dead_features = self.num_batches_not_active >= self.cfg["n_batches_to_dead"]
        dead_features_sum = dead_features.sum().item()
        
        if dead_features_sum > 0:
            residual_en = x_en.float() - x_reconstruct_en.float()
            residual_es = x_es.float() - x_reconstruct_es.float()
            
            acts_topk_aux = torch.topk(
                acts[:, dead_features],
                min(self.cfg["top_k_aux"], dead_features_sum),
                dim=-1,
            )
            
            acts_aux = torch.zeros_like(acts[:, dead_features]).scatter(
                -1, acts_topk_aux.indices, acts_topk_aux.values
            )
            
            x_reconstruct_aux_en = acts_aux @ self.W_dec_en[dead_features]
            x_reconstruct_aux_es = acts_aux @ self.W_dec_es[dead_features]
            
            l2_loss_aux_en = (x_reconstruct_aux_en.float() - residual_en.float()).pow(2).mean()
            l2_loss_aux_es = (x_reconstruct_aux_es.float() - residual_es.float()).pow(2).mean()
            
            # The same weighting scheme as in the main loss.
            if lang_idx == 0:
                en_weight = 1 - self.bilingual_exposure
                es_weight =     self.bilingual_exposure
            else:
                en_weight =     self.bilingual_exposure
                es_weight = 1 - self.bilingual_exposure
                
            l2_loss_aux_en_weighted = en_weight * l2_loss_aux_en
            l2_loss_aux_es_weighted = es_weight * l2_loss_aux_es
            l2_loss_aux = self.cfg["aux_penalty"] * (l2_loss_aux_en_weighted + l2_loss_aux_es_weighted)
                        
            return l2_loss_aux

        else:
            return torch.tensor(0.0, dtype=x_en.dtype, device=x_en.device)

    def get_loss_dict(self, lang_idx,
                      x_en, x_es, x_reconstruct_en, x_reconstruct_es, acts, acts_original, x_mean, x_std, 
                      x_prime_reconstruct_en, x_prime_reconstruct_es, acts_prime, acts_prime_original):

        device = self.W_enc.device
        
        x_en = x_en.to(device, non_blocking=True) # These are the original 
        x_es = x_es.to(device, non_blocking=True)
        
        x_reconstruct_en = x_reconstruct_en.to(device, non_blocking=True)
        x_reconstruct_es = x_reconstruct_es.to(device, non_blocking=True)
        acts = acts.to(device, non_blocking=True)

        x_prime_reconstruct_en = x_prime_reconstruct_en.to(device, non_blocking=True)
        x_prime_reconstruct_es = x_prime_reconstruct_es.to(device, non_blocking=True)
        acts_prime = acts_prime.to(device, non_blocking=True)

        l2_loss_en = (x_reconstruct_en.float() - x_en.float()).pow(2).mean()
        l2_loss_es = (x_reconstruct_es.float() - x_es.float()).pow(2).mean()
        
        l2_loss_en_prime = (x_prime_reconstruct_en.float() - x_en.float()).pow(2).mean()
        l2_loss_es_prime = (x_prime_reconstruct_es.float() - x_es.float()).pow(2).mean()
        
        num_dead_features = (self.num_batches_not_active > self.cfg["n_batches_to_dead"]).sum().item() / self.cfg["dict_size"]
        activation_variance = ((((acts > 0).float().sum(-1) - self.cfg["top_k"])**2).mean() + (((acts_prime > 0).float().sum(-1) - self.cfg["top_k"])**2).mean()) / 2
        acts_norm = torch.nn.functional.normalize(acts.float(), p=2, dim=-1)                                    # The above is the variance of magnitude in the number of features that activate in a batch. Higher values mean inconsistent sparsity the batch.
        acts_prime_norm = torch.nn.functional.normalize(acts_prime.float(), p=2, dim=-1)
        acts_norm_original = torch.nn.functional.normalize(acts_original.float(), p=2, dim=-1)
        acts_prime_norm_original = torch.nn.functional.normalize(acts_prime_original.float(), p=2, dim=-1)
        
        mse = (acts - acts_prime).pow(2).mean()                                                                 # Only calculated for logging.
        mse_original = (acts_original - acts_prime_original).pow(2).mean()
        cos_sim = (acts_norm * acts_prime_norm).sum(dim=-1).mean()                                          
        cos_sim_original = (acts_norm_original * acts_prime_norm_original).sum(dim=-1).mean()
                                                                                                                # This is meant to give a (weak) incentive to make the encoding vectors similar. The convergence suddenly dropping during training is not explainable by this.
        convergence = (1 - cos_sim) * self.cfg["convergence_coeff"]                                             # This gives 0 when the encoding vectors are identical, 1 when orthogonal, and 2 when opposite (scaled by the convergence_coeff)
                                                                                                                # If the current batch is in English, then weight the English decoder's reconstruction by how much we want it 
        if lang_idx == 0:                                                                                       # and weigh the Spanish decoder's reconstruction by how often the network has seen Spanish activations.
            en_weight = 1 - self.bilingual_exposure                                                             # Scale these values by how "bilingual" we want each update to be. This is equivalent to 
            es_weight =     self.bilingual_exposure                                                             # the hyperparameter lambda prime in my thesis. 
        else:                                                                                                   # This is important because certain languages are simply better represented by the LLM in the first place.
            en_weight =     self.bilingual_exposure                                                             # So, we want a hyperparameter to capture this pre-existing bias in the activation training data,
            es_weight = 1 - self.bilingual_exposure                                                             # which we can then control for when calculating the loss.
                                                                                                                # Maximizing the amount of exposure (or rather, minimizing (bilingual_exposure - 0.5)**2) with a proper Lagrangian would also be an interesting follow-up.
        l2_loss_en_weighted = en_weight * l2_loss_en                                                            
        l2_loss_es_weighted = es_weight * l2_loss_es                                                            
        l2_diff = l2_loss_en - l2_loss_es
        l2_loss = l2_loss_en_weighted + l2_loss_es_weighted
        l2_loss_en_prime_weighted = en_weight * l2_loss_en_prime
        l2_loss_es_prime_weighted = es_weight * l2_loss_es_prime
        l2_diff_prime = l2_loss_en_prime - l2_loss_es_prime
        l2_loss_prime = l2_loss_en_prime_weighted + l2_loss_es_prime_weighted
        l2 = (l2_loss + l2_loss_prime) / 2
        
        l1_norm_per_feature_prime = acts_prime.float().abs().sum() / (acts_prime > 0).float().sum().clamp(min=1.0)
        l1_norm_per_feature = acts.float().abs().sum() / (acts > 0).float().sum().clamp(min=1.0)                # Here [we [calculate [how strongly [the [top k features] that activate at all] activate] for logging]].
        l1_norm = acts.float().abs().sum(-1).mean()                                                             # No weighting here, since we only have the one encoder and activation space.
        l1_norm_prime = acts_prime.float().abs().sum(-1).mean()
        l1_loss = self.cfg["l1_coeff"] * l1_norm                                                                # The coefficient is implicitly defined in relation to our learning rate and top_k hyperparameters.
        l1_loss_prime = self.cfg["l1_coeff"] * l1_norm_prime
        l1 = (l1_loss + l1_loss_prime) / 2
    

        l0_norm_epsilon = (acts > self.cfg["l0_epsilon"]).float().sum(-1).mean()          # This is the average number of features that activate at all on an input across a batch, but only above a certain threshold (used for logging).
        l0_norm_epsilon_prime = (acts_prime > self.cfg["l0_epsilon"]).float().sum(-1).mean()
        l0_norm = (acts > 0).float().sum(-1).mean()                                        # This is the average number of features that activate at all on an input across a batch divided by how many features are available.
        l0_norm_prime = (acts_prime > 0).float().sum(-1).mean()
        l0 = (l0_norm + l0_norm_prime) / 2
                                                                                                                # This is the auxiliary loss term, which encourages the model to use dead features to
        aux_loss = self.get_auxiliary_loss(x_en, x_es, x_reconstruct_en, x_reconstruct_es, acts_original, lang_idx)          # reconstruct the residual between the input and current reconstruction,
        aux_loss_prime = self.get_auxiliary_loss(x_en, x_es, x_prime_reconstruct_en, x_prime_reconstruct_es, acts_prime_original, lang_idx)  # so we pass the original activations before the top-k selection.
        aux = (aux_loss + aux_loss_prime) / 2
        
        loss = l2 + l1 + aux + convergence
        
        sae_out = self.postprocess_output(x_reconstruct_en if lang_idx == 0 else x_reconstruct_es, x_mean, x_std)
        output = {
            "sae_out": sae_out,
            "feature_acts": acts,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l1_loss": l1_loss,
            "l1_loss_prime": l1_loss_prime,
            "l1_normalized": l1_loss / self.cfg["l1_coeff"],
            "l1_normalized_prime": l1_loss_prime / self.cfg["l1_coeff"],
            "l2_loss": l2_loss,
            "l2_loss_prime": l2_loss_prime,
            "l2_loss_en": l2_loss_en_weighted,
            "l2_loss_en_prime": l2_loss_en_prime_weighted,
            "l2_loss_es": l2_loss_es_weighted,
            "l2_loss_es_prime": l2_loss_es_prime_weighted,
            "l2_norm_diff": l2_diff,
            "l2_norm_diff_prime": l2_diff_prime,
            "convergence_loss": convergence,
            "convergence": 1 - cos_sim, # This is for logging since it's not scaled by our choice of coefficient, unlike convergence_loss.
            "convergence_original": 1 - cos_sim_original,
            "mse_convergence": mse,
            "mse_original": mse_original,
            "cos_convergence": 1 - cos_sim,
            "l0_norm": l0,
            "l0_norm_prime": l0_norm_prime,
            "l0_norm_epsilon": l0_norm_epsilon,
            "l0_norm_epsilon_prime": l0_norm_epsilon_prime,
            "l1_norm": l1_norm,
            "l1_norm_prime": l1_norm_prime,
            "aux_loss": aux_loss,
            "aux_loss_prime": aux_loss_prime,
            "aux_normalized": aux_loss / self.cfg["aux_penalty"],
            "aux_normalized_prime": aux_loss_prime / self.cfg["aux_penalty"],
            "selected_lang": lang_idx,
            "ratio": self.bias,
            "l1_norm_per_feature": l1_norm_per_feature,
            "l1_norm_per_feature_prime": l1_norm_per_feature_prime,
            "activation_variance": activation_variance,
            "conv": "last",
        }
        return output

    def forward(self, x_en, x_es):

        x_en = x_en.to(self.W_enc.device, non_blocking=True)
        x_es = x_es.to(self.W_enc.device, non_blocking=True)
        
        lang_idx = self.select_language()
        
        if lang_idx == 0:
            x_input = x_en
            x_prime = x_es
        else:
            x_input = x_es
            x_prime = x_en
        
        acts, x, x_mean, x_std, acts_original = self.encode(x_input)
        self.update_inactive_features(acts)
        x_reconstruct_en, x_reconstruct_es = self.decode(acts)
        
        acts_prime, x_prime, x_prime_mean, x_prime_std, acts_prime_original = self.encode(x_prime)
        self.update_inactive_features(acts_prime)
        x_prime_reconstruct_en, x_prime_reconstruct_es = self.decode(acts_prime)
        
        output = self.get_loss_dict(lang_idx,
                                    x_en, x_es, x_reconstruct_en, x_reconstruct_es, acts, acts_original, x_mean, x_std,
                                    x_prime_reconstruct_en, x_prime_reconstruct_es, acts_prime, acts_prime_original)
                
        return output
    
    # This is a crucial part of the setup that basically ensures the autoencoder
    # isn't able to use its weights to memorize a lookup table defined over activation magnitude,
    # which needed to be re-implemented since we have two decoders.
    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        for W_dec in [self.W_dec_en, self.W_dec_es]:
            W_dec_normed = W_dec / W_dec.norm(dim=-1, keepdim=True)
            W_dec_grad_proj = (W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
            W_dec.grad -= W_dec_grad_proj
            W_dec.data = W_dec_normed
        torch.cuda.empty_cache()


        """
Leaving these here, which can be evaluated as well:
        """

class BatchTopKSAE(BaseAutoencoder):
    def __init__(self, cfg):
        super().__init__(cfg)

    def forward(self, x):
        x, x_mean, x_std = self.preprocess_input(x)

        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc)

        acts_topk = torch.topk(acts.flatten(), self.cfg["top_k"] * x.shape[0], dim=-1)
        acts_topk = (
            torch.zeros_like(acts.flatten())
            .scatter(-1, acts_topk.indices, acts_topk.values)
            .reshape(acts.shape)
        )

        x_reconstruct = acts_topk @ self.W_dec + self.b_dec

        self.update_inactive_features(acts_topk)
        output = self.get_loss_dict(x, x_reconstruct, acts, acts_topk, x_mean, x_std)
        return output

    def get_loss_dict(self, x, x_reconstruct, acts, acts_topk, x_mean, x_std):
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()

        l1_norm = acts_topk.float().abs().sum(-1).mean()
        l1_loss = self.cfg["l1_coeff"] * l1_norm

        l0_norm = (acts_topk > 0).float().sum(-1).mean()

        aux_loss = self.get_auxiliary_loss(x, x_reconstruct, acts)

        loss = l2_loss + l1_loss + aux_loss

        num_dead_features = (
            self.num_batches_not_active > self.cfg["n_batches_to_dead"]
        ).sum() / self.cfg["dict_size"] # Quotient edited by Aidan to also be a percentage like my metric.

        sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std)
        
        # Added by Aidan to compare to my metrics.
        activation_variance = (((acts_topk > 0).float().sum(-1) - self.cfg["top_k"])**2).mean()
        l1_norm_per_feature = acts_topk.float().abs().sum() / (acts_topk > 0).float().sum().clamp(min=1.0)
        
        output = {
            "sae_out": sae_out,
            "feature_acts": acts_topk,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l1_loss": l1_loss,
            "l2_loss": l2_loss,
            "l0_norm": l0_norm,
            "l1_norm": l1_norm,
            "aux_loss": aux_loss,
            "activation_variance": activation_variance,
            "l1_norm_per_feature": l1_norm_per_feature,
        }
        return output

    def get_auxiliary_loss(self, x, x_reconstruct, acts):
        dead_features = self.num_batches_not_active >= self.cfg["n_batches_to_dead"]
        if dead_features.sum() > 0:
            residual = x.float() - x_reconstruct.float()
            acts_topk_aux = torch.topk(
                acts[:, dead_features],
                min(self.cfg["top_k_aux"], dead_features.sum()),
                dim=-1,
            )
            acts_aux = torch.zeros_like(acts[:, dead_features]).scatter(
                -1, acts_topk_aux.indices, acts_topk_aux.values
            )
            x_reconstruct_aux = acts_aux @ self.W_dec[dead_features]
            l2_loss_aux = (
                self.cfg["aux_penalty"]
                * (x_reconstruct_aux.float() - residual.float()).pow(2).mean()
            )
            return l2_loss_aux
        else:
            return torch.tensor(0, dtype=x.dtype, device=x.device)


class TopKSAE(BaseAutoencoder):
    def __init__(self, cfg):
        super().__init__(cfg)

    def forward(self, x):
        x, x_mean, x_std = self.preprocess_input(x)

        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc)

        acts_topk = torch.topk(acts, self.cfg["top_k"], dim=-1)
        acts_topk = torch.zeros_like(acts).scatter(
            -1, acts_topk.indices, acts_topk.values
        )

        x_reconstruct = acts_topk @ self.W_dec + self.b_dec

        self.update_inactive_features(acts_topk)
        output = self.get_loss_dict(x, x_reconstruct, acts, acts_topk, x_mean, x_std)
        return output

    def get_loss_dict(self, x, x_reconstruct, acts, acts_topk, x_mean, x_std):
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()

        l1_norm = acts_topk.float().abs().sum(-1).mean()
        l1_loss = self.cfg["l1_coeff"] * l1_norm

        l0_norm = (acts_topk > 0).float().sum(-1).mean()

        aux_loss = self.get_auxiliary_loss(x, x_reconstruct, acts)

        loss = l2_loss + l1_loss + aux_loss

        num_dead_features = (
            self.num_batches_not_active > self.cfg["n_batches_to_dead"]
        ).sum()

        sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std)
        output = {
            "sae_out": sae_out,
            "feature_acts": acts_topk,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l1_loss": l1_loss,
            "l2_loss": l2_loss,
            "l0_norm": l0_norm,
            "l1_norm": l1_norm,
            "aux_loss": aux_loss,
        }
        return output

    def get_auxiliary_loss(self, x, x_reconstruct, acts):
        dead_features = self.num_batches_not_active >= self.cfg["n_batches_to_dead"]
        if dead_features.sum() > 0:
            residual = x.float() - x_reconstruct.float()
            acts_topk_aux = torch.topk(
                acts[:, dead_features],
                min(self.cfg["top_k_aux"], dead_features.sum()),
                dim=-1,
            )
            acts_aux = torch.zeros_like(acts[:, dead_features]).scatter(
                -1, acts_topk_aux.indices, acts_topk_aux.values
            )
            x_reconstruct_aux = acts_aux @ self.W_dec[dead_features]
            l2_loss_aux = (
                self.cfg["aux_penalty"]
                * (x_reconstruct_aux.float() - residual.float()).pow(2).mean()
            )
            return l2_loss_aux
        else:
            return torch.tensor(0, dtype=x.dtype, device=x.device)


class VanillaSAE(BaseAutoencoder):
    def __init__(self, cfg):
        super().__init__(cfg)

    def forward(self, x):
        x, x_mean, x_std = self.preprocess_input(x)

        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc + self.b_enc)

        x_reconstruct = acts @ self.W_dec + self.b_dec

        self.update_inactive_features(acts)
        output = self.get_loss_dict(x, x_reconstruct, acts, x_mean, x_std)
        return output

    def get_loss_dict(self, x, x_reconstruct, acts, x_mean, x_std):
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()

        l1_norm = acts.float().abs().sum(-1).mean()
        l1_loss = self.cfg["l1_coeff"] * l1_norm

        l0_norm = (acts > 0).float().sum(-1).mean()
        # Added new metrics here too
        l0_norm_epsilon = (acts > self.cfg["l0_epsilon"]).float().sum(-1).mean()
        l1_norm_per_feature = acts.float().abs().sum() / (acts > 0).float().sum().clamp(min=1.0)
        activation_variance = (((acts > 0).float().sum(-1) - l0_norm.mean())**2).mean()

        loss = l2_loss + l1_loss

        num_dead_features = (
            self.num_batches_not_active > self.cfg["n_batches_to_dead"]
        ).sum() / self.cfg["dict_size"]  # Also edited to be a percentage like my metric.

        sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std)
        output = {
            "sae_out": sae_out,
            "feature_acts": acts,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l1_loss": l1_loss,
            "l2_loss": l2_loss,
            "l0_norm": l0_norm,
            "l0_norm_epsilon": l0_norm_epsilon,
            "l1_norm": l1_norm,
            "l1_norm_per_feature": l1_norm_per_feature,
            "activation_variance": activation_variance,
        }
        return output

class RectangleFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return ((x > -0.5) & (x < 0.5)).float()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[(x <= -0.5) | (x >= 0.5)] = 0
        return grad_input

class JumpReLUFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x, log_threshold, bandwidth):
        ctx.save_for_backward(x, log_threshold, torch.tensor(bandwidth))
        threshold = torch.exp(log_threshold)
        return x * (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, log_threshold, bandwidth_tensor = ctx.saved_tensors
        bandwidth = bandwidth_tensor.item()
        threshold = torch.exp(log_threshold)
        x_grad = (x > threshold).float() * grad_output
        threshold_grad = (
            -(threshold / bandwidth)
            * RectangleFunction.apply((x - threshold) / bandwidth)
            * grad_output
        )
        return x_grad, threshold_grad, None

class JumpReLU(nn.Module):
    def __init__(self, feature_size, bandwidth, device='cpu'):
        super(JumpReLU, self).__init__()
        self.log_threshold = nn.Parameter(torch.zeros(feature_size, device=device))
        self.bandwidth = bandwidth

    def forward(self, x):
        return JumpReLUFunction.apply(x, self.log_threshold, self.bandwidth)

class StepFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x, log_threshold, bandwidth):
        ctx.save_for_backward(x, log_threshold, torch.tensor(bandwidth))
        threshold = torch.exp(log_threshold)
        return (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, log_threshold, bandwidth_tensor = ctx.saved_tensors
        bandwidth = bandwidth_tensor.item()
        threshold = torch.exp(log_threshold)
        x_grad = torch.zeros_like(x)
        threshold_grad = (
            -(1.0 / bandwidth)
            * RectangleFunction.apply((x - threshold) / bandwidth)
            * grad_output
        )
        return x_grad, threshold_grad, None

class JumpReLUSAE(BaseAutoencoder):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.jumprelu = JumpReLU(feature_size=cfg["dict_size"], bandwidth=cfg["bandwidth"], device=cfg["device"])

    def forward(self, x, use_pre_enc_bias=False):
        x, x_mean, x_std = self.preprocess_input(x)

        if use_pre_enc_bias:
            x = x - self.b_dec

        pre_activations = torch.relu(x @ self.W_enc + self.b_enc)
        feature_magnitudes = self.jumprelu(pre_activations)

        x_reconstructed = feature_magnitudes @ self.W_dec + self.b_dec

        return self.get_loss_dict(x, x_reconstructed, feature_magnitudes, x_mean, x_std)

    def get_loss_dict(self, x, x_reconstruct, acts, x_mean, x_std):
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()

        l0 = StepFunction.apply(acts, self.jumprelu.log_threshold, self.cfg["bandwidth"]).sum(dim=-1).mean()
        l0_loss = self.cfg["l1_coeff"] * l0
        l1_loss = l0_loss

        loss = l2_loss + l1_loss

        num_dead_features = (
            self.num_batches_not_active > self.cfg["n_batches_to_dead"]
        ).sum()

        sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std)
        output = {
            "sae_out": sae_out,
            "feature_acts": acts,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l1_loss": l1_loss,
            "l2_loss": l2_loss,
            "l0_norm": l0,
            "l1_norm": l0,
        }
        return output