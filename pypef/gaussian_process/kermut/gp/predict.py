# PyPEF - Pythonic Protein Engineering Framework
# https://github.com/niklases/PyPEF

# Using adapted Kermut code published under MIT License; 
# available at https://github.com/petergroth/kermut

from typing import List, Tuple
import numpy as np
import pandas as pd
import torch
import gpytorch
from gpytorch.likelihoods import GaussianLikelihood

from pypef.gaussian_process.kermut.gp.kermut_gp import KermutGP


def predict(
    gp: KermutGP,
    likelihood: GaussianLikelihood,
    test_inputs: Tuple[torch.Tensor, ...],
    *args, **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """Makes predictions using a trained Gaussian Process.
    
    Returns raw predictions instead of mutating external DataFrames.
    """
    gp.eval()
    likelihood.eval()

    # Cast all floating-point test inputs to float32 to match the model's parameters
    x_test = tuple(
        x.float() if (isinstance(x, torch.Tensor) and torch.is_floating_point(x)) else x 
        for x in test_inputs
    )

    # Wrap in standard GPyTorch inference managers
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # Pass the sanitized float32 test elements into the GP
        y_preds_dist = likelihood(gp(*x_test))
        
        # Extract means and variances
        test_means = y_preds_dist.mean
        test_variances = y_preds_dist.variance

    return test_means, test_variances
