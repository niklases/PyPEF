# PyPEF - Pythonic Protein Engineering Framework
# https://github.com/niklases/PyPEF

# Using adapted Kermut code published under MIT License; 
# available at https://github.com/petergroth/kermut


from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import torch
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.priors import HalfCauchyPrior
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood

from pypef.gaussian_process.kermut.tokenizer import Tokenizer



def prepare_kermut_inputs(
    wt_seq: str,
    seqs_train: List[str],
    seqs_test: List[str],
    x_embed_train: torch.Tensor,
    x_embed_test: torch.Tensor,
    x_zero_shot_train: torch.Tensor,
    x_zero_shot_test: torch.Tensor,
) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
    """Generates the properly routed and structured tensor tuples for training and testing.
    
    Handles internal structural alignment constraints using Kermut's specific Tokenizer.
    """
    tokenizer = Tokenizer()
    
    x_kermut_toks_train = torch.stack([tokenizer(seq) for seq in seqs_train]).float().cpu()
    x_kermut_toks_test = torch.stack([tokenizer(seq) for seq in seqs_test]).float().cpu()

    train_inputs = (x_kermut_toks_train, x_embed_train.cpu(), x_zero_shot_train.cpu())
    test_inputs = (x_kermut_toks_test, x_embed_test.cpu(), x_zero_shot_test.cpu())

    return train_inputs, test_inputs