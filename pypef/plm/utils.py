# PyPEF - Pythonic Protein Engineering Framework
# https://github.com/niklases/PyPEF

import numpy as np
import torch
import os
import platform
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers.utils import logging as ts_logging
ts_logging.set_verbosity_error()

import logging
logger = logging.getLogger('pypef.plm.utils')


def hybrid_corr_mse_loss(y_true, y_pred, method="spearman", tau=0.1, alpha=0.5):
    """
    Hybrid differentiable loss combining Spearman correlation and MSE.
    """
    # Differentiable Spearman or Pearson
    loss_rank = correlation_loss(y_true, y_pred, method=method, tau=tau)
    # MSE
    loss_value = torch.mean((y_pred - y_true)**2)
    # Combine
    return alpha * loss_rank + (1 - alpha) * loss_value


def correlation_loss(
    y_true: torch.Tensor, 
    y_pred: torch.Tensor, 
    method: str = "spearman", 
    tau: float = 0.1
) -> torch.Tensor:
    """
    Differentiable correlation loss for PyTorch.
    
    Args:
        y_true: Tensor of shape (..., n) or (batch, n)
        y_pred: Tensor of same shape as y_true
        method: "pearson" or "spearman"
        tau: temperature for soft-rank approximation (used if method="spearman")
        
    Returns:
        Scalar tensor representing the loss (to minimize)
    """
    if method == "spearman":
        # Soft rank approximation
        x = y_true
        y = y_pred

        def soft_rank(x, tau):
            x = x.unsqueeze(-1)
            diff = x - x.transpose(-1, -2)
            P = torch.sigmoid(diff / tau)
            return P.sum(dim=-1) + 0.5

        rx = soft_rank(x, tau)
        ry = soft_rank(y, tau)
    elif method == "pearson":
        rx = y_true
        ry = y_pred
    else:
        raise ValueError(f"Unsupported method: {method}. Choose 'pearson' or 'spearman'.")

    # Centering
    rx_c = rx - rx.mean(dim=-1, keepdim=True)
    ry_c = ry - ry.mean(dim=-1, keepdim=True)

    # Normalize (like dividing by std)
    rx_n = rx_c / (rx_c.norm(dim=-1, keepdim=True) + 1e-8)
    ry_n = ry_c / (ry_c.norm(dim=-1, keepdim=True) + 1e-8)

    # Compute correlation
    corr = (rx_n * ry_n).sum(dim=-1)

    # Return scalar loss (to minimize, so negative correlation)
    return -corr.mean()


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
        tokenizer_loader=None
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
    if exists:
        try:
            logger.info(f"Loading model and tokenizer from cache {snapshot_dir}...")
            model = model_loader.from_pretrained(
                snapshot_dir, trust_remote_code=True
            )
            tokenizer = tokenizer_loader.from_pretrained(
                snapshot_dir, trust_remote_code=True
            )
        except OSError as e:
            logger.info(f"Faced error \"{e}\": Trying to load with regular cache load path...")
            model = model_loader.from_pretrained(
                model_name, cache_dir=cache_dir, trust_remote_code=True
            )
            tokenizer = tokenizer_loader.from_pretrained(
                model_name, cache_dir=cache_dir, trust_remote_code=True
            )
    else:
        logger.info(f"Did not find model {model_name} and associated tokenizer in cache directory "
                    f"(checked for model snapshot reference file {ref_file}), downloading model and tokenizer "
                    f"from the internet and storing in cache {cache_dir}...")
        model = model_loader.from_pretrained(
            model_name, cache_dir=cache_dir, trust_remote_code=True
        )
        tokenizer = tokenizer_loader.from_pretrained(
            model_name, cache_dir=cache_dir, trust_remote_code=True
        )
    logger.info("Model and tokenizer loaded successfully...")
    return model, tokenizer
