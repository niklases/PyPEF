# PyPEF - Pythonic Protein Engineering Framework
# https://github.com/niklases/PyPEF

import os
import numpy as np
import torch
import torch.nn.functional as F
import platform
import random
import warnings
from transformers import set_seed
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
                warnings.warn(
                    "'CUBLAS_WORKSPACE_CONFIG' not set, "
                    "will likely face a torch RuntimeError. "
                    "Make sure to e.g. run 'export CUBLAS_WORKSPACE_CONFIG=:4096:8' (Linux/Mac) "
                    "or '$env:CUBLAS_WORKSPACE_CONFIG=\":4096:8\"' (Windows PowerShell) "
                    "before running with set seeds and determinism."
                )
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
        alpha=0.0: only consider MSE, in between: hybrid loss).
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
    # 1. Calculate Correlation Component
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

    # 2. Calculate MSE Component
    loss_mse = F.mse_loss(y_pred, y_true)

    # 3. Combine
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
