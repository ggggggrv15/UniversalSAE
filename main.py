import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["WANDB__SERVICE_WAIT"] = "300"
#os.environ['HF_TOKEN'] = 
from training import train_sae
from sae import VanillaSAE, TopKSAE, BatchTopKSAE, JumpReLUSAE, UniversalSAE
from gemma_activation_store import GemmaActivationsStore
from cached_activations_store import CachedActivationsStore
from config import get_architecture_cfg, load_gemma_model, DICT_SIZES, L1_COEFFS, TOP_K_VALUES, LANGUAGES, LEARNING_RATES, ARCHS, BIAS_VALUES, BILINGUAL_EXPOSURE, DIM, TOKENS, BATCH_SIZE, WANDB_PROJECT, CONVERGENCE_COEFF, AUX_PENALTY
import torch
from tqdm import tqdm
import numpy as np
import wandb
import sys
import random
from logs import clean_wandb_artifacts_cache
import wandb

BATCHES_PER_RUN = TOKENS // BATCH_SIZE
resuming_sweep = False

if resuming_sweep: # For when a sweep crashes
    
    api = wandb.Api()
    project_path = WANDB_PROJECT
    try:
        entity = api.default_entity
        if entity:
            project_path = f"{entity}/{WANDB_PROJECT}"
        existing_runs = api.runs(project_path)
    except:
        existing_runs = api.runs(WANDB_PROJECT)

    non_pre_signatures = set()
    pre_signatures = set()
    runs_by_signature = {}
    
    for run in existing_runs:
        if run.state != "finished":
            continue
    config = run.config
    arch_type = config.get('sae_type')
    dict_size = config.get('dict_size')
    l1_coeff = config.get('l1_coeff')
    top_k = config.get('top_k')  # This is already the adjusted value
    lr = config.get('lr')
    bias = config.get('ratio') if arch_type == 'universal' else None
    bilingual_exposure = config.get('bilingual_exposure') if arch_type == 'universal' else 0
    language = config.get('language') if arch_type != 'universal' else 'both'
    convergence_coeff = config.get('convergence_coeff') if arch_type == 'universal' else None
    signature = (arch_type, dict_size, l1_coeff, top_k, lr, bias, bilingual_exposure, language, convergence_coeff)
    
    is_pre_run = False
    if hasattr(run, "group") and run.group is not None and "pre" in run.group:
        is_pre_run = True
    elif "pre" in run.name.lower():
        is_pre_run = True
        
    if is_pre_run:
        pre_signatures.add(signature)
    else:
        non_pre_signatures.add(signature)
        print(f"Found regular run with signature: {signature}")
    
    runs_by_signature[signature] = run.name
    completed_signatures = non_pre_signatures

total_iterations = 0
planned_iterations = 0

for arch_type in ARCHS:
    relevant_top_k_values = [TOP_K_VALUES[0]] if arch_type in ["vanilla", "jumprelu"] else TOP_K_VALUES
    lang_list = ["both"] if arch_type == "universal" else LANGUAGES
    relevant_bias_values = BIAS_VALUES if arch_type == "universal" else [None]
    relevant_convergence_coeff_values = CONVERGENCE_COEFF if arch_type == "universal" else [None]
    for lr in LEARNING_RATES:
        for top_k in relevant_top_k_values:
            for l1_coeff in L1_COEFFS:
                for dict_size in DICT_SIZES:
                    top_k_adjusted = (dict_size // DIM) * top_k
                    for bias in relevant_bias_values:
                        for bilingual_exposure in BILINGUAL_EXPOSURE:
                            for lang in lang_list:
                                for convergence_coeff in relevant_convergence_coeff_values:
                                    language = None if lang == "both" else lang
                                    run_signature = (arch_type, dict_size, l1_coeff, top_k_adjusted, 
                                                lr, bias, bilingual_exposure, lang, convergence_coeff)      
                                    planned_iterations += 1
                                    if resuming_sweep and run_signature in completed_signatures:
                                        print(f"Already completed run: {runs_by_signature.get(run_signature, run_signature)}")
                                        continue
                                    total_iterations += BATCHES_PER_RUN

if resuming_sweep:
    print(f"Planned: {planned_iterations}, already completed: {len(completed_signatures)}, remaining: {planned_iterations - len(completed_signatures)}")

main_progress_bar = tqdm(total=total_iterations, desc="Overall progress", position=0)

"""
SWEEEEEEEP 🧹
"""
first_run = True
for arch_type in ARCHS:
    print(f"\n--- Training {arch_type.upper()} architecture ---")
    relevant_top_k_values = [TOP_K_VALUES[0]] if arch_type in ["vanilla", "jumprelu"] else TOP_K_VALUES
    lang_list = ["both"] if arch_type == "universal" else LANGUAGES
    relevant_bias_values = BIAS_VALUES if arch_type == "universal" else [None]
    relevant_convergence_coeff_values = CONVERGENCE_COEFF if arch_type == "universal" else [None]
    for lr in LEARNING_RATES:
        for top_k in relevant_top_k_values:
            for l1_coeff in L1_COEFFS:
                for dict_size in DICT_SIZES:
                    top_k_adjusted = (dict_size // DIM) * top_k
                    for bias in relevant_bias_values:
                        for bilingual_exposure in BILINGUAL_EXPOSURE:
                            for convergence_coeff in relevant_convergence_coeff_values:
                                for lang in lang_list: # (both for universal)
                                    run_signature = (arch_type, dict_size, l1_coeff, top_k_adjusted, lr, bias, bilingual_exposure, lang, convergence_coeff)
                                    if resuming_sweep and run_signature in completed_signatures:
                                        print(f"Skipping already completed run: {runs_by_signature.get(run_signature, run_signature)}")
                                        continue
                    
                                    language = None if lang == "both" else lang
                                    cfg = get_architecture_cfg(arch_type, dict_size, l1_coeff, top_k_adjusted, language, lr, bias, bilingual_exposure, original_top_k=top_k, convergence_coeff=convergence_coeff)
                                    
                                    if arch_type == "universal":
                                        sae = UniversalSAE(cfg)
                                    elif arch_type == "vanilla":
                                        sae = VanillaSAE(cfg)
                                    elif arch_type == "topk":
                                        sae = TopKSAE(cfg)
                                    elif arch_type == "batchtopk":
                                        sae = BatchTopKSAE(cfg)
                                    elif arch_type == "jumprelu":
                                        sae = JumpReLUSAE(cfg)
                                    else:
                                        raise ValueError(f"I think you typed the wrong arch name.")

                                    model = load_gemma_model(cfg)
                                    use_cached = cfg.get("use_cached_activations", False)
                                    
                                    if not first_run:
                                        sys.stdout.write('\033[2J\033[H')
                                        sys.stdout.flush()
                                    else:
                                        first_run = False
                                    main_progress_bar.set_description(f"LR{lr}-K{top_k_adjusted}-L1{l1_coeff}-D{dict_size}-B{bias}-B{bilingual_exposure}")
                                    
                                    if use_cached:
                                        print(f"Using cached activations from {cfg.get('activations_cache_path', 'auto-detected cache')}")
                                        activations_store = CachedActivationsStore(cfg)
                                        train_sae(sae, activations_store, None, cfg, main_progress_bar)
                                    else:
                                        activations_store = GemmaActivationsStore(model, cfg)
                                        train_sae(sae, activations_store, model, cfg, main_progress_bar)

                                    torch.cuda.empty_cache()
                                    clean_wandb_artifacts_cache()

main_progress_bar.close()