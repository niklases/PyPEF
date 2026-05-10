# PyPEF - Pythonic Protein Engineering Framework
# https://github.com/niklases/PyPEF

import os
import numpy as np
import torch
import torch.nn.functional as F
import platform
import random
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import set_seed, AutoConfig
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers.utils import logging as ts_logging
ts_logging.set_verbosity_error()

import logging
logger = logging.getLogger('pypef.plm.utils')


def _set_seeds(seed: int, use_deterministic_algorithms: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    if use_deterministic_algorithms:
        # For cross-machine consistency, before run:
        # export CUBLAS_WORKSPACE_CONFIG=:4096:8
        # or
        # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" 
        rw = False
        try:
            if not os.environ["CUBLAS_WORKSPACE_CONFIG"]:
                rw = True
        except KeyError:
            rw = True
        if rw:
            pass
            #warnings.warn(
            #    "'CUBLAS_WORKSPACE_CONFIG' not set, "
            #    "will likely face a torch RuntimeError. "
            #    "Make sure to e.g. run 'export CUBLAS_WORKSPACE_CONFIG=:4096:8' (Linux/Mac) "
            #    "or '$env:CUBLAS_WORKSPACE_CONFIG=\":4096:8\"' (Windows PowerShell) "
            #    "before running with set seeds and determinism."
            #)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def hybrid_corr_mse_loss(
    y_true: torch.Tensor, 
    y_pred: torch.Tensor, 
    method: str = "spearman", 
    tau: float = 0.1, 
    alpha: float| None = None
) -> torch.Tensor:
    """
    Hybrid differentiable loss combining correlation (Spearman/Pearson) and MSE.
    
    Args:
        y_true: Ground truth tensor.
        y_pred: Predicted tensor.
        method: "spearman" (uses soft-ranking) or "pearson".
        tau: Temperature for soft-rank approximation.
        alpha: Weight for correlation loss. (1 - alpha) is weight for MSE:
        alpha=0.0: only consider MSE, in between: hybrid loss, alpha=1.0: only 
        consider Spearman correlation/ranking or Pearson correlation).
    """
    if alpha is None:
        if method in ["spearman", "pearson"]:
            alpha=1.0
        elif method in ["spearman-hybrid", "pearson-hybrid"]:
            alpha=0.5
        else:
            raise RuntimeError(
                "Alpha parameter for loss function is not defined. Define alpha or a method "
                "from within ['spearman', 'pearson', 'spearman-hybrid', 'pearson-hybrid']."
            )
    # Calculate Correlation Component
    if method.startswith("spearman"):
        # Soft rank approximation helper
        def get_soft_ranks(z, t):
            # z: (batch, n) -> (batch, n, 1)
            z_expanded = z.unsqueeze(-1)
            # pairwise differences: (batch, n, n)
            diff = z_expanded - z_expanded.transpose(-1, -2)
            # sigmoid approximation of indicator function
            P = torch.sigmoid(diff / t)
            return P.sum(dim=-1) + 0.5
        
        rx = get_soft_ranks(y_true, tau)
        ry = get_soft_ranks(y_pred, tau)
    elif method.startswith("pearson"):
        rx = y_true
        ry = y_pred
    else:
        raise ValueError(f"Method {method} not supported.")

    # Centering and Normalizing for Cosine Similarity (Correlation)
    rx_c = rx - rx.mean(dim=-1, keepdim=True)
    ry_c = ry - ry.mean(dim=-1, keepdim=True)
    
    # Cosine similarity of centered vectors = Correlation
    corr = F.cosine_similarity(rx_c, ry_c, dim=-1).mean()
    loss_corr = -corr  # We want to maximize correlation, so minimize negative
    loss_mse = F.mse_loss(y_pred, y_true)
    return (alpha * loss_corr) + ((1 - alpha) * loss_mse)


def get_batches(a, dtype, batch_size=5,
                keep_remaining=False, verbose: bool = False
                ) -> list | list[np.ndarray]:
    a = np.asarray(a, dtype=dtype)
    a_remaining = None
    orig_shape = np.shape(a)
    remaining = len(a) % batch_size
    if remaining != 0:
        if len(a) > batch_size:
            a = a[:-remaining]
            a_remaining = a[-remaining:]
        else:
            logger.info(f"Batch size greater than or equal to total array length: "
                        f"returning full array (of shape: {np.shape(a)})...")
            if keep_remaining:
                return a.tolist()
    if len(orig_shape) == 2:
        a = a.reshape(np.shape(a)[0] // batch_size, batch_size, np.shape(a)[1])
    else: # elif len(orig_shape) == 1:
        a = a.reshape(np.shape(a)[0] // batch_size, batch_size)
    new_shape = np.shape(a)
    if verbose:
        logger.info(f'Batch reshape:{orig_shape} -> {new_shape} (dropped {remaining})')
    if keep_remaining: 
        if a_remaining is not None:
            logger.info(f'Appending remaining to collected batches as last batch '
                        f'(the resulting inhomogenous list shape is '
                        f'{np.shape(a)} + {np.shape(a_remaining)} = ('
                        f'{np.shape(a)[0] + 1}, *, {np.shape(a)[-1]}))...')
            a = a.tolist()
            a.append(a_remaining)
            a = [np.asarray(it) for it in a]
    return a


def get_default_cache_dir():
    """
    Detect OS and set Hugging Face transformers cache directory accordingly
    """
    system = platform.system()
    if system == "Windows":
        return os.path.join(
            os.environ.get("USERPROFILE", ""), ".cache",
            "huggingface", "hub"
        )
    elif system == "Darwin":
        return os.path.expanduser("~/.cache/huggingface/hub")
    else: # Assume Linux or other Unix-like systems
        return os.path.expanduser("~/.cache/huggingface/hub")


def is_model_cached(repo_id: str, cache_dir: str):
    """
    Check if the required model and tokenizer files are cached locally.
    """
    snapshot_dir = None
    ref_file = None
    if os.path.isdir(cache_dir):
        ref_file = os.path.join(
            cache_dir, f'models--{repo_id.replace("/", "--")}', 'refs', 'main'
        )
        if os.path.isfile(ref_file):
            with open(ref_file, 'r') as fh:
                t = fh.readlines()  # Getting hash contents
            ref = t[0].strip()
        else:
            return False, snapshot_dir, ref_file
        snapshot_dir = os.path.join(
            cache_dir, f'models--{repo_id.replace("/", "--")}', 'snapshots', ref
        )
        if os.path.isdir(snapshot_dir):
            return True, snapshot_dir, ref_file
        else:
            return False, None, ref_file
    else:
        return False, snapshot_dir, ref_file


def load_model_and_tokenizer(
    model_name: str, 
    cache_dir: str | os.PathLike | None = None, 
    model_loader=None, 
    tokenizer_loader=None,
    revision: str | None = None
):
    """
    Enhanced loader that bypasses broken Windows symlinks by manually 
    injecting weights from the HF blob store if ProSST is detected.
    """
    if cache_dir is None:
        # Assuming you have a helper for this, or use default
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        
    if model_loader is None:
        model_loader = AutoModelForMaskedLM
    if tokenizer_loader is None:
        tokenizer_loader = AutoTokenizer

    # Check if model is cached locally
    # Note: Even if exists=True, Windows symlinks might be broken pointers
    exists, snapshot_dir, _ = is_model_cached(model_name, cache_dir)
    if exists:
        logger.info(f"Model snapshot extists at {snapshot_dir}...")
    is_windows = platform.system().lower() == "windows"

        # Loading the model
    logger.info(f"Loading model architecture for {model_name}...")
    
    # We first try a standard load. 
    # On Windows, we use the model_name (repo_id) rather than snapshot_dir 
    # to let HF attempt its internal resolution.
    load_path = model_name if is_windows else (snapshot_dir if exists else model_name)

    config = AutoConfig.from_pretrained(
        load_path, 
        trust_remote_code=True, 
        revision=revision, 
        cache_dir=cache_dir
    )

    # Force the architecture to create a separate decoder layer
    config.tie_word_embeddings = False

    # Common loading arguments
    load_kwargs = {
        "cache_dir": cache_dir,
        "trust_remote_code": True,
        "revision": revision,
        "local_files_only": exists
    }

    # Add the config to your load_kwargs
    load_kwargs["config"] = config
    
    try:
        model = model_loader.from_pretrained(
            load_path,
            use_safetensors=True,
            **load_kwargs
        )
    except Exception as e:
        # Warning appears too oftne for ESM as no safetensor exists (respectively existed back then)
        # logger.warning(f"Standard load failed, trying without safetensors: {e}")
        model = model_loader.from_pretrained(
            load_path,
            use_safetensors=False,
            **load_kwargs
        )

    # THE WINDOWS SYMLINK BYPASS (Specific for ProSST)
    # If the weights didn't load, they are forced here
    if is_windows and "prosst" in model_name.lower():
        try:
            logger.info("Windows detected: Forcing manual weight injection from blobs...")
            
            # This identifies the actual large binary file in the blobs folder
            real_weight_path = hf_hub_download(
                repo_id=model_name,
                filename="model.safetensors",
                cache_dir=cache_dir,
                local_files_only=True
            )
            
            # Load weights manually
            state_dict = load_file(real_weight_path)

            # Prepare the state_dict (Inject the missing key if it's not there)
            if 'cls.predictions.decoder.weight' not in state_dict:
                logger.info("Injecting cloned embedding weights into model state dictionary for decoder.")
                state_dict['cls.predictions.decoder.weight'] = state_dict['prosst.embeddings.word_embeddings.weight'].clone()
        
            # Apply the weights to the model
            # Use strict=False so it doesn't crash on minor metadata mismatches
            msg = model.load_state_dict(state_dict, strict=False) 
            #assert (model.cls.predictions.decoder.weight.sum().item() == 
            #        model.prosst.embeddings.word_embeddings.weight.sum().item())
            
            if len(msg.missing_keys) > 0:
                logger.warning(f"Weights injected, but some keys still missing: {msg.missing_keys}")
                
        except Exception as e:
            logger.error(f"Manual weight injection failed: {e}")

    # GLOBAL PROSST PARITY (Runs on Linux/GitHub and Windows)
    # This ensures that even on Linux, the decoder is a bit-perfect clone of the embeddings.
    if "prosst" in model_name.lower():
        logger.info("Enforcing cross-platform weight parity for ProSST...")
        
        # Ensure the decoder layer exists (if not loaded via state_dict)
        if not hasattr(model.cls.predictions, 'decoder'):
            logger.info("Manually attaching linear decoder head...")
            model.cls.predictions.decoder = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        with torch.no_grad():
            embedding_weight = model.prosst.embeddings.word_embeddings.weight
            model.cls.predictions.decoder.weight.copy_(embedding_weight)
            
        # Do NOT call model.tie_weights() here, or PyTorch will turn them back into pointers

    # Loading the tokenizer
    logger.info(f"Loading tokenizer for {model_name}...")
    try:
        tokenizer = tokenizer_loader.from_pretrained(
            load_path,
            use_fast=False, # Avoids the sentencepiece/tiktoken conversion error
            **load_kwargs
        )
    except Exception as e:
        logger.warning(f"Tokenizer load failed with use_fast=False, trying default: {e}")
        tokenizer = tokenizer_loader.from_pretrained(
            load_path,
            **load_kwargs
        )

    logger.info(f"Successfully finished loading {model_name}.")
    return model, tokenizer


def load_model_and_tokenizer__(
        model_name: str, 
        cache_dir: str | os.PathLike | None = None, 
        model_loader=None, 
        tokenizer_loader=None,
        revision: str | None = None
):
    """
    Load the model and tokenizer from cache directory. Downloads to cache if not present.
    """
    if cache_dir is None:
        cache_dir = get_default_cache_dir()
    if model_loader is None:
        model_loader = AutoModelForMaskedLM
    if tokenizer_loader is None:
        tokenizer_loader = AutoTokenizer
    exists, snapshot_dir, ref_file = is_model_cached(model_name, cache_dir)

    # Check for Windows to avoid the 'Snapshot Path' trap
    is_windows = platform.system().lower() == "windows"
    if exists and not is_windows:
        try:
            logger.info(f"Loading model and tokenizer from cache {snapshot_dir}...")
            model = model_loader.from_pretrained(
                snapshot_dir, trust_remote_code=True, revision=revision, #use_safetensors=True
            )
            tokenizer = tokenizer_loader.from_pretrained(
                snapshot_dir, trust_remote_code=True, revision=revision, #use_safetensors=True
            )
        except OSError as e:
            logger.warning(f"Snapshot load failed: {e}. Falling back to repo_id load.")
            exists = False # Trigger the fallback below
    if not exists or is_windows:
        logger.info(f"Loading via repo_id '{model_name}' (safe mode for Windows/missing cache)...")
        # Using the model_name instead of snapshot_dir allows HF to resolve 
        # the actual weight blobs even if the symlink structure is broken.
        model = model_loader.from_pretrained(
            model_name, 
            cache_dir=cache_dir, 
            trust_remote_code=True, 
            revision=revision,
            local_files_only=exists, # Use cache if it's there, but let HF handle the mapping
            #use_safetensors=True
        )
        tokenizer = tokenizer_loader.from_pretrained(
            model_name, 
            cache_dir=cache_dir, 
            trust_remote_code=True, 
            revision=revision,
            local_files_only=exists,
            #use_safetensors=True
        )
    logger.info("Model and tokenizer loaded successfully...")
    return model, tokenizer
