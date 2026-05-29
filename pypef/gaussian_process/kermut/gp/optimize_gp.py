# PyPEF - Pythonic Protein Engineering Framework
# https://github.com/niklases/PyPEF

# Using adapted Kermut code published under MIT License; 
# available at https://github.com/petergroth/kermut


from typing import Tuple
import torch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from tqdm import trange


def optimize_gp(
    gp: ExactGP,
    likelihood: GaussianLikelihood,
    train_inputs: Tuple[torch.Tensor, ...],
    train_targets: torch.Tensor,
    lr: float = 0.05,
    n_steps: int = 150,
    progress_bar: bool = True,
) -> Tuple[ExactGP, GaussianLikelihood]:
    """Optimizes a Gaussian Process using marginal likelihood maximization."""
    gp.train()
    likelihood.train()
    mll = ExactMarginalLogLikelihood(likelihood, gp)

    optimizer = torch.optim.AdamW(gp.parameters(), lr=lr)

    x_train = tuple(
        x.float() if (isinstance(x, torch.Tensor) and torch.is_floating_point(x)) else x 
        for x in train_inputs
    )
    y_train = train_targets.float()

    gp.set_train_data(inputs=x_train, targets=y_train, strict=True)

    for _ in trange(n_steps, disable=not progress_bar):
        optimizer.zero_grad()
        output = gp(*x_train)
        loss = -mll(output, y_train)
        loss.backward()
        optimizer.step()
        
    return gp, likelihood