import wandb  # Weights & Biases for experiment tracking
import torch  # PyTorch for tensor operations and model building
from functools import partial  # Partial function application
import os  # Operating system interface for file operations
import json  # JSON serialization and deserialization
from datetime import datetime
import tempfile
import shutil
import glob
import filelock
import time
import random  # For random probability in cache cleaning decisions

def init_wandb(cfg):
    # If there's an active run, finish it first
    if wandb.run is not None:
        wandb.finish()
    
    # Import datetime module for adding timestamps
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a unique name for this run based on architecture and parameters
    run_name = f"{cfg['sae_type']}_{cfg['dict_size']}_{cfg['top_k']}"
    
    # Add language info to run name
    if cfg['sae_type'] == "universal":
        run_name += "_bilingual"
    elif "language" in cfg:
        run_name += f"_{cfg['language']}"
    
    # Add L1 coefficient to run name
    run_name += f"_l1_{cfg['l1_coeff']}"
    
    # Add timestamp at the end
    run_name += f"_{current_time}"
    
    # Configure Weights & Biases and create a new run
    wandb.init(
        project=cfg["wandb_project"],
        config=cfg,
        name=run_name,
        reinit=True,  # Force reinitialize a new run
    )
    wandb_run = wandb.run
    return wandb_run

def log_wandb(output, step, wandb_run, index=None):
    # Prepare a dictionary for logging
    log_dict = {}
    # Log each metric in the output dictionary
    for k, v in output.items():
        # Skip special keys that are not metrics
        if k in ["sae_out", "feature_acts"]:
            continue
        # Add the metric to the log dictionary with an optional index
        if index is not None:
            log_dict[f"{k}_{index}"] = v.item() if hasattr(v, "item") else v
        else:
            log_dict[k] = v.item() if hasattr(v, "item") else v
    # Log the metrics to Weights & Biases
    if wandb_run is not None:
        wandb_run.log(log_dict, step=step)

# Hook function to return the SAE output for reconstruction for model performance evaluation
def reconstr_hook(activation, hook, sae_out):
    return sae_out

# Hook function to zero out activations
def zero_abl_hook(activation, hook):
    return torch.zeros_like(activation)

# Hook function to replace activations with their mean
def mean_abl_hook(activation, hook):
    return activation.mean([0, 1]).expand_as(activation)

@torch.no_grad()  # Disable gradient tracking for performance logging
def log_model_performance(wandb_run, step, model, activation_store, sae, index=None, batch_tokens=None, is_universal=False, language=None):
    # Skip model performance logging if no wandb_run
    if wandb_run is None:
        return
        
    # If model is None (using cached activations), log basic metrics only
    if model is None:
        log_dict = {
            "cached_activations": True,
            "model_metrics_available": False
        }
        
        if index is not None:
            log_dict[f"using_cached_activations_{index}"] = True
        else:
            log_dict["using_cached_activations"] = True
            
        # Log the metrics to Weights & Biases
        wandb_run.log(log_dict, step=step)
        return
    
    # If batch tokens are not provided, generate new tokens
    if batch_tokens is None:
        if language:
            batch_tokens = activation_store.get_batch_tokens(is_test=True, language=language)
        else:
            batch_tokens = activation_store.get_batch_tokens(is_test=True)
    
    with torch.no_grad():
        # Check if this is a Gemma model (doesn't have run_with_cache)
        is_gemma_model = hasattr(model, 'config') and not hasattr(model, 'run_with_cache')
        
        if is_gemma_model:
            # For Gemma models, use the activation_store to get activations directly
            # This bypasses the need for run_with_cache
            batch = activation_store.get_activations(batch_tokens)
            
            # Get tokens before and after for accuracy calculation
            # Since Gemma tokenizer might work differently, get these directly from batch_tokens
            tokens_before = batch_tokens[:, :-1]
            tokens_after = batch_tokens[:, 1:]
            
            # Handle forward pass for both standard and universal SAE
            if is_universal:
                # For UniversalSAE, use single-input compatible forward pass
                sae_out = sae(batch.reshape(-1, activation_store.cfg["act_size"]))["sae_out"].reshape(batch.shape)
            else:
                # Standard forward pass for regular SAE models
                sae_out = sae(batch.reshape(-1, activation_store.cfg["act_size"]))["sae_out"].reshape(batch.shape)
        
            # For Gemma models, we can't easily compute next token prediction accuracy
            # since we don't have direct access to the embedding/unembed layers
            # Instead, just log that we're using a Gemma model
            log_dict = {}
            log_dict["model_type"] = "gemma"
            
            if index is not None:
                # Still log the basic info but without accuracy metrics
                log_dict[f"batch_size_{index}"] = batch.shape[0]
                log_dict[f"seq_len_{index}"] = batch.shape[1] if len(batch.shape) > 1 else 1
            else:
                log_dict["batch_size"] = batch.shape[0]
                log_dict["seq_len"] = batch.shape[1] if len(batch.shape) > 1 else 1
                
            # Log additional info for universal SAE
            if is_universal:
                log_dict["language_used"] = language
            
            # Log the metrics to Weights & Biases
            wandb_run.log(log_dict, step=step)
            
            # Clear CUDA cache before returning
            torch.cuda.empty_cache()
            return
            
        # Original HookedTransformer implementation
        # Pass tokens through the model to get activations
        _, cache = model.run_with_cache(
            batch_tokens,
            names_filter=[activation_store.hook_point],
            stop_at_layer=activation_store.cfg["layer"] + 1,
        )
        # Get batch activations
        batch = cache[activation_store.hook_point]
        # Get tokens before and after (for performance comparison)
        tokens_before = batch_tokens[:, :-1]
        tokens_after = batch_tokens[:, 1:]
        
        # Handle forward pass for both standard and universal SAE
        if is_universal:
            # For UniversalSAE, use a single-input compatible forward pass
            # The UniversalSAE class should handle this with its single-input fallback
            sae_out = sae(batch.reshape(-1, activation_store.cfg["act_size"]))["sae_out"].reshape(batch.shape)
        else:
            # Standard forward pass for regular SAE models
            sae_out = sae(batch.reshape(-1, activation_store.cfg["act_size"]))["sae_out"].reshape(batch.shape)
            
        # Compute next token prediction accuracy
        next_token_logits_sae = model.unembed(sae_out)
        next_token_logits = model.unembed(batch)
        next_token_pred_sae = next_token_logits_sae.argmax(-1)
        next_token_pred = next_token_logits.argmax(-1)
        sae_accuracy = (next_token_pred_sae[:, :-1] == tokens_after).float().mean()
        base_accuracy = (next_token_pred[:, :-1] == tokens_after).float().mean()
        
        # Create dictionary for logging metrics
        log_dict = {}
        log_dict["model_type"] = "hooked_transformer"
        
        if index is not None:
            log_dict[f"sae_next_token_accuracy_{index}"] = sae_accuracy.item()
            log_dict[f"base_next_token_accuracy_{index}"] = base_accuracy.item()
            log_dict[f"sae_to_base_accuracy_ratio_{index}"] = (sae_accuracy / base_accuracy).item()
        else:
            log_dict["sae_next_token_accuracy"] = sae_accuracy.item()
            log_dict["base_next_token_accuracy"] = base_accuracy.item()
            log_dict["sae_to_base_accuracy_ratio"] = (sae_accuracy / base_accuracy).item()
        
        # Log additional info for universal SAE
        if is_universal:
            log_dict["language_used"] = language
        
        # Log the metrics to Weights & Biases
        wandb_run.log(log_dict, step=step)
        
        # Clear up memory
        del sae_out, next_token_logits_sae, next_token_logits, next_token_pred_sae, next_token_pred, batch, cache
        torch.cuda.empty_cache()

def clean_wandb_artifacts_cache():
    """Clean the wandb artifacts cache directory to save disk space.
    Safe for concurrent runs with file locking and conservative deletion strategy."""
    
    if wandb.run is None:
        return
    
    try:
        # Get the current run ID
        current_run_id = wandb.run.id
        
        # Determine the wandb cache directory location
        cache_dir = os.path.expanduser("~/.cache/wandb/artifacts")
        
        if not os.path.exists(cache_dir):
            return
            
        # Create a lock file to prevent concurrent access
        lock_file = os.path.join(os.path.dirname(cache_dir), "wandb_cache_clean.lock")
        
        # Add a small random delay to reduce contention
        time.sleep(random.uniform(0.1, 0.5))
        
        # Try to acquire the lock with a timeout
        try:
            with filelock.FileLock(lock_file, timeout=2):
                print(f"Cleaning wandb artifacts cache directory: {cache_dir}")
                
                # Get the current run's artifact directory pattern
                current_run_pattern = f"*{current_run_id}*"
                
                # Get all artifact directories
                all_artifact_dirs = []
                for root, dirs, files in os.walk(cache_dir):
                    # Skip deep directories to avoid removing individual files
                    # Only process directories at most 2 levels deep
                    if root.count(os.sep) - cache_dir.count(os.sep) <= 2:
                        for dir_name in dirs:
                            dir_path = os.path.join(root, dir_name)
                            all_artifact_dirs.append(dir_path)
                
                # Keep track of how much space we free
                bytes_freed = 0
                
                # Only remove top-level directories that don't contain the current run ID
                # and have been created more than 10 minutes ago
                current_time = time.time()
                for dir_path in all_artifact_dirs:
                    # Skip directories containing the current run ID
                    if current_run_id in dir_path:
                        continue
                        
                    try:
                        # Check if directory exists before attempting to get stats
                        if not os.path.exists(dir_path):
                            continue
                            
                        # Check if the directory is older than 10 minutes to avoid cleaning recent files
                        dir_mtime = os.path.getmtime(dir_path)
                        if current_time - dir_mtime < 600:  # 600 seconds = 10 minutes
                            continue
                        
                        # Get directory size before removing
                        dir_size = 0
                        for dirpath, _, filenames in os.walk(dir_path):
                            for filename in filenames:
                                file_path = os.path.join(dirpath, filename)
                                if os.path.exists(file_path):
                                    try:
                                        dir_size += os.path.getsize(file_path)
                                    except (OSError, FileNotFoundError):
                                        pass
                        
                        # Remove the directory
                        if os.path.exists(dir_path):
                            shutil.rmtree(dir_path, ignore_errors=True)
                            bytes_freed += dir_size
                        
                    except (OSError, FileNotFoundError) as e:
                        # Just log the error and continue - don't crash the run
                        print(f"Skipping artifact directory {dir_path}: {e}")
                
                # Convert bytes to a human-readable format
                def human_readable_size(size_bytes):
                    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
                        if size_bytes < 1024.0:
                            return f"{size_bytes:.2f} {unit}"
                        size_bytes /= 1024.0
                    return f"{size_bytes:.2f} PB"
                
                print(f"Freed {human_readable_size(bytes_freed)} from wandb artifacts cache")
        
        except filelock.Timeout:
            print("Cache cleaning skipped - another process is already cleaning the cache")
            
    except Exception as e:
        print(f"Error cleaning wandb artifacts cache (non-fatal): {e}")
        # Don't propagate the exception - cleaning the cache should never crash the run

def save_checkpoint(wandb_run, sae, cfg, i, is_final=False, log_artifact=True):
    """Save model checkpoint to wandb without creating local files.
    
    Args:
        wandb_run: The wandb run to log the artifact to
        sae: The SAE model to save
        cfg: The configuration dictionary
        i: The iteration number
        is_final: Whether this is the final checkpoint (default: False). 
                  If False, only saves checkpoint if checkpoint_freq is divisible by i.
    """
    # Only proceed if wandb is active
    if wandb_run is None:
        return
    
    # Skip non-final checkpoints based on configuration
    checkpoint_freq = cfg.get("checkpoint_freq", 10000)
    if not is_final and i % checkpoint_freq != 0:
        return
    
    if log_artifact:
        # Create a wandb artifact for the model
        artifact_name = "final_model" if is_final else f"checkpoint_{i}"
        artifact = wandb.Artifact(artifact_name, type="model")
        
        # Save model directly to a temporary buffer in memory
        with tempfile.NamedTemporaryFile(suffix='.pt') as tmp_file:
            torch.save(sae.state_dict(), tmp_file.name)
            tmp_file.flush()  # Ensure all data is written
            
            # Add the temporary file to the artifact
            artifact.add_file(tmp_file.name)
            
            # Log the artifact to wandb
            wandb_run.log_artifact(artifact)
    
    # For final checkpoints, we'll perform cache cleanup with 100% probability
    # For intermediate checkpoints, we'll only do it with 10% probability to avoid conflicts
    # during concurrent runs, and only for very infrequent checkpoints
    if is_final:
        # Always clean up after the final checkpoint
        clean_wandb_artifacts_cache()
    elif i > 0 and i % checkpoint_freq == 0 and random.random() < 0.3:
        # Very rarely clean up during training (1/10 chance, every 10 checkpoint intervals)
        try:
            clean_wandb_artifacts_cache()
        except Exception as e:
            print(f"Intermediate cache cleaning error (non-fatal): {e}")