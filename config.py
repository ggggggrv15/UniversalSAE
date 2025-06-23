from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformer_lens.utils as utils
import torch
import os 

# The language model under investigation:
MODEL = "gemma-2b"

# Its activation dimensions:
if MODEL == "gemma-2b":
    DIM = 2**11
elif MODEL == "gpt2-small":
    DIM = 6 * 2**7

LEARNING_RATES = [0.0000006]      #fave 6e-7                  # These specific hyperparameters can be configured in a list, which will be swept upon running main.py.
DICT_SIZES = [DIM * 2**5]   #6                           # They're currently set to the values I used for the gemma-2b model, but you can change them to whatever.
TOP_K_VALUES = [2**4, 2**5]                                  # The only thing you'll have to remember is that the product of the top_k and dict_size/dim factor must be a power of two.
BILINGUAL_EXPOSURE = [0.95]                             # Each of them is explained a bit more below.
L1_COEFFS = [0.0002] # fave 0.00007
TOKENS = int(8e6) #8e6
CACHED_ACTIVATIONS = True
BIAS_VALUES = [0.5]   
CONVERGENCE_COEFF = [1.5] 
BATCH_SIZE = 2 ** 10 #7
MODEL_BATCH_SIZE = 2 ** 4
BUFFER_SIZE = 2 ** 9
ARCHS = ["universal","vanilla"]
DATASET_PATH = "/home/aidanm/scratch/DL/Thesis/processed/eurlex_processed.json"
WANDB_PROJECT = "USAE_corrected_loss"
LANGUAGES = ["en", "es"]
AUX_PENALTY = [1/2**4] # 1/32

def get_default_cfg():
    default_cfg = {
        "seed": 49,                                     # For reproducibility
        "use_cached_activations": CACHED_ACTIVATIONS,   # This selects whether we're actively running inference on an LLM or using pre-computed activations.
        "batch_size": BATCH_SIZE,                       # You'll have to reconfigure my HF authorization if you want online inference.
        "activations_cache_dir": "cached_activations",  # Directory for cached activations, we can also specify a specific path if we like:
        "activations_cache_path": "/home/aidanm/scratch/DL/Thesis/SAE/BatchTopK_paper/cached_activations/gemma-2b_layer9_20250426.h5",
        "lr": LEARNING_RATES[0],
        "num_tokens": TOKENS,                           # This is the number of tokens' activations that we'll process, from which we derive the number of batches.
        "convergence_coeff": CONVERGENCE_COEFF[0],      # This is the coefficient for the convergence loss term, which encourages the model to make the activations of the two languages similar.
        "l1_coeff": L1_COEFFS[0],                       # This is the L1 regularization coefficient, which controls the sparsity of the activations.
        "l0_epsilon": 0.05,                                # For the sake of logging, instead of measuring exactly how many features are dead on a given input,
        "beta1": 0.9,                                   # we can measure how many are nearly dead.
        "beta2": 0.99,
        "max_grad_norm": 100000,
        "seq_len": 2048,                               # Sequence length for input data when using non-cached activations.
        "dtype": torch.float32,
        "model_name": MODEL,                            # "gpt2-small" also available, but not recommended since it is quite small and is predominantly trained on English data.
        "site": "resid_pre",                            # Site in the LLM layer from which to extract activations
        "layer": 9,                                     # Layer of the model to extract activations from
        "act_size": DIM,
        "dict_size": DICT_SIZES[0],                     # This is the (maximum) number of concepts our model will learn to encode.
        "bilingual_exposure": BILINGUAL_EXPOSURE,       # This indicates the amount of incentive, from (0 to 1), for the model's encoding of an activation from one language to be able to be accurately decoded into the other language. This is the hyperparameter lambda prime in my thesis.
        "sae_type": "universal",                        # A value of 1 means that the model is only incentivized to be able to *convert* activations from one language to the other, regardless of how well it can reconstruct the activation from the original language.
        "num_batches_in_buffer": BUFFER_SIZE,           # to reconstruct the activation from the opposite language as the language the current batch's activations are from.
        "wandb_project": WANDB_PROJECT,                 # We would like to find whichever value for this hyperparameter that maximizes performance on both languages.
        "device": "cuda:0",                             # You'll probably need to change this.
        "model_batch_size": MODEL_BATCH_SIZE,           # Batch size for LLM
        "dataset_path": DATASET_PATH,                   # Path to (Eurlex) dataset.
        "language": "en",                               # Language for training monolingual SAEs
        "input_unit_norm": True,                        # Whether to normalize input
        "n_batches_to_dead": 5,                         # Number of batches after which to consider a feature dead
        "top_k": TOP_K_VALUES[0]*(DICT_SIZES[0]//DIM),  # The product of K and the dictionary_size/dimension factor (the power of 2 factor in the definition of DICT_SIZES above)
        "num_test_batches": 4,                          # is the mean number of activations passed to the decoders for each input. Certain values in each bach may have more or
        "perf_log_freq": 1000,                          # less than K activations, but each bach will have exactly K * batch_size activations passed to the decoders.    E
        "test_freq": 100,                       
        "ratio": BIAS_VALUES[0],                        # This defines the amount of training data from one language versus the other. A value of 0.9 for example means we'd train on 10% of the data from Language 1, and 90% from Language 2.
        "checkpoint_freq": 10000,
        "original_top_k": TOP_K_VALUES[0],              # Relation to dict size for logging purposes
        "top_k_aux": DIM // 2,                          # Most people use 512 but the original authors used DIM // 2, which works better for me (They specify that it should be a power of 2 near DIM, which is convenient since this model has a power of 2 dimension).
        "aux_penalty": AUX_PENALTY[0],                  # Penalty for the aux loss term which encourages the model to use dead features to reconstruct in the Universal, BatchTopK and TopK SAEs
        "bandwidth": 0.001,                             # Bandwidth parameter for JumpReLU SAE
        "checkpoint_dir": "checkpoints",                # This is a path to the local Gemma LLM:
        "gemma_local_path": os.path.expanduser("~/scratch/DL/Thesis/models/gemma-2b"),
        "training_bias": BIAS_VALUES[0] - 0.5,         # This logging makes it easier to search for correlations between the training bias and the performance of the SAE during the fine-tuning process.
        "training_bias_magnitude": (BIAS_VALUES[0] - 0.5)**2,
        "conv": "pls",
    }
    default_cfg = post_init_cfg(default_cfg)
    return default_cfg

def get_architecture_cfg(arch_type, 
                         dict_size=None, 
                         l1_coeff=None, 
                         top_k=None, 
                         language=None, 
                         lr=None, bias=None, 
                         bilingual_exposure=None, 
                         original_top_k=None, 
                         convergence_coeff=None, 
                         aux_penalty=None):
    
    cfg = get_default_cfg()
    cfg["sae_type"] = arch_type
    if dict_size is not None:
        cfg["dict_size"] = dict_size
    if l1_coeff is not None:
        cfg["l1_coeff"] = l1_coeff
    if top_k is not None:
        cfg["top_k"] = top_k
    if lr is not None:
        cfg["lr"] = lr
    if convergence_coeff is not None:
        cfg["convergence_coeff"] = convergence_coeff
    if aux_penalty is not None:
        cfg["aux_penalty"] = aux_penalty
    if bias is not None and arch_type == "universal":
        cfg["ratio"] = bias
        cfg["training_bias"] = bias - 0.5
        cfg["training_bias_magnitude"] = (bias - 0.5) **2
    if bilingual_exposure is not None:
        cfg["bilingual_exposure"] = bilingual_exposure
    if original_top_k is not None:
        cfg["original_top_k"] = original_top_k
    if arch_type == "universal":
        cfg["languages"] = LANGUAGES
    elif language is not None:
        cfg["language"] = language
    if arch_type == "universal":
        cfg["name"] = f"{cfg['name']}_bilingual"
    elif language is not None:
        cfg["name"] = f"{cfg['name']}_{language}"
    return post_init_cfg(cfg)

def load_gemma_model(cfg):
    model_repo = "google/gemma-2b"
    local_path = cfg.get("gemma_local_path")
    if os.path.exists(local_path) and os.path.isfile(os.path.join(local_path, "pytorch_model.bin")):
        print(f"Loading Gemma model from {local_path}")
        model_path = local_path
    else:
        print(f"Downloading Gemma model to {local_path}")
        os.makedirs(local_path, exist_ok=True)
        model_path = snapshot_download(
            repo_id=model_repo,
            local_dir=local_path,
            token=os.environ.get("HF_TOKEN"),
        )
    # Load with output_hidden_states=True
    print(f"Loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if cfg["device"] != "cpu" else torch.float32,
        device_map=cfg["device"],
        output_hidden_states=True,
        return_dict=True,
        token=os.environ.get("HF_TOKEN"),
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=os.environ.get("HF_TOKEN"))
    model.tokenizer = tokenizer
    return model

def post_init_cfg(cfg):
    cfg["hook_point"] = utils.get_act_name(cfg["site"], cfg["layer"])
    if "name" not in cfg:
        cfg["name"] = f"{cfg['model_name']}_{cfg['hook_point']}_{cfg['dict_size']}_{cfg['sae_type']}_{cfg['top_k']}_{cfg['lr']}"
    return cfg