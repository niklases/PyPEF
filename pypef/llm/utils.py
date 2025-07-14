# PyPEF - Pythonic Protein Engineering Framework
# https://github.com/niklases/PyPEF

import numpy as np
import torch

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
    else:  # elif len(orig_shape) == 1:
        a = a.reshape(np.shape(a)[0] // batch_size, batch_size)
    new_shape = np.shape(a)
    if verbose:
        logger.info(f'{orig_shape} -> {new_shape}  (dropped {remaining})')
    if keep_remaining: # Returning a list
        a = list(a)
        logger.info('Adding dropped back to batches as last batch...')
        a.append(a_remaining)
        return a
    if keep_numpy:
        return a
    return torch.Tensor(a).to(dtype)
