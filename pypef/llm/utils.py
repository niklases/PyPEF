# PyPEF - Pythonic Protein Engineering Framework
# https://github.com/niklases/PyPEF

import numpy as np
import torch
import os
import platform
from huggingface_hub import try_to_load_from_cache, _CACHED_NO_EXIST
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers.utils import logging as ts_logging
ts_logging.set_verbosity_error()

import logging
logger = logging.getLogger('pypef.llm.utils')


def corr_loss(y_true: torch.Tensor, y_pred: torch.Tensor):
    res_true = y_true - torch.mean(y_true)
    res_pred = y_pred - torch.mean(y_pred)
    cov = torch.mean(res_true * res_pred)
    var_true = torch.mean(res_true**2)
    var_pred = torch.mean(res_pred**2)
    sigma_true = torch.sqrt(var_true)
    sigma_pred = torch.sqrt(var_pred)
    return - cov / (sigma_true * sigma_pred)


def get_batches(a, dtype, batch_size=5,
                keep_numpy: bool = False, keep_remaining=False, verbose: bool = False):
    a = np.asarray(a, dtype=dtype)
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
                return list(a)
            else:
                return a
    if len(orig_shape) == 2:
        a = a.reshape(np.shape(a)[0] // batch_size, batch_size, np.shape(a)[1])
    else: # elif len(orig_shape) == 1:
        a = a.reshape(np.shape(a)[0] // batch_size, batch_size)
    new_shape = np.shape(a)
    if verbose:
        logger.info(f'{orig_shape} -> {new_shape} (dropped {remaining})')
    if keep_remaining: # Returning a list
        a = list(a)
        logger.info('Adding dropped back to batches as last batch...')
        a.append(a_remaining)
        return a
    if keep_numpy:
        return a
    return torch.Tensor(a).to(dtype)


def get_default_cache_dir():
    """
    Detect OS and set Hugging Face transformers cache directory accordingly
    """
    system = platform.system()
    if system == "Windows":
        return os.path.join(os.environ.get("USERPROFILE", ""), ".cache", "huggingface", "transformers")
    elif system == "Darwin":
        return os.path.expanduser("~/.cache/huggingface/transformers")
    else: # Assume Linux or other Unix-like systems
        return os.path.expanduser("~/.cache/huggingface/transformers")


def is_model_cached(repo_id: str, cache_dir: str) -> bool:
    """
    Check if the required model and tokenizer files are cached locally.
    """

    filepath = try_to_load_from_cache(repo_id=repo_id, filename='model.safetensors', cache_dir=cache_dir)
    if isinstance(filepath, str):
        return True # file is cached
    elif filepath is _CACHED_NO_EXIST:
        return False # non-existence of file is cached
    else:
        return False # file is not cached and not in non-existance cache


def load_model_and_tokenizer(model_name, cache_dir: str | os.PathLike | None = None):
    """
    Load the model and tokenizer from cache directory. Downloads to cache if not present.
    """
    if cache_dir is None:
        cache_dir = get_default_cache_dir()
    if is_model_cached(model_name, cache_dir):
        logger.info(f"Loading model and tokenizer from cache {cache_dir}...")
    else:
        logger.info(f"Did not find model and tokenizer in cache directory, downloading model "
                    f"and tokenizer from the internet and storing in cache {cache_dir}...")
    model = AutoModelForMaskedLM.from_pretrained(
        model_name, cache_dir=cache_dir, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=cache_dir, trust_remote_code=True
    )
    return model, tokenizer
