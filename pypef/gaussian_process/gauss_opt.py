# PyPEF - Pythonic Protein Engineering Framework
# https://github.com/niklases/PyPEF

# Using adapted Kermut code published under MIT License

"""
Gaussian process optimization similar (but less sophisticated compared) to 
Kermut: Composite kernel regression for protein variant effects
Peter Mørch Groth, Mads Herbert Kerrn, Lars Olsen, Jesper Salomon, Wouter Boomsma
2024, 38th Conference on Neural Information Processing Systems (NeurIPS 2024).
TL;DR: Gaussian process regression model with a novel composite kernel, Kermut, achieves 
state-of-the-art variant effect prediction while providing meaningful uncertainties.
Literature: https://openreview.net/forum?id=jM9atrvUii.
Used under MIT license; code available at https://github.com/petergroth/kermut.
"""


from sklearn.model_selection import train_test_split
import torch
import gpytorch
import pandas as pd
from tqdm import tqdm

from pypef.plm.esm_lora_tune import get_esm_models
from pypef.plm.inference import plm_inference, tokenize_sequences
from pypef.plm.prosst_lora_tune import get_prosst_models, get_structure_quantizied
from pypef.plm.utils import spearman_soft, correlation_loss, hybrid_corr_mse_loss, pearson_loss
from pypef.utils.variant_data import get_wt_sequence



class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class HellingerRBFKernel(gpytorch.kernels.Kernel):
    has_lengthscale = True  # GPyTorch handles log-lengthscale automatically

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Amplitude/variance parameter
        self.register_parameter(
            name="raw_variance",
            parameter=torch.nn.Parameter(torch.tensor(0.0))
        )
        self.register_constraint("raw_variance", gpytorch.constraints.Positive())

    @property
    def variance(self):
        return self.raw_variance_constraint.transform(self.raw_variance)

    @variance.setter
    def variance(self, value):
        self._set_variance(value)

    def _set_variance(self, value):
        # Properly set raw_variance via inverse transform
        self.raw_variance.data = self.raw_variance_constraint.inverse_transform(value)

    def forward(self, x1, x2, **params):
        """
        x1: [n1, d] (probabilities)
        x2: [n2, d]
        Returns: covariance matrix [n1, n2]
        """
        # Ensure probabilities
        x1 = torch.clamp(x1, min=0)
        x2 = torch.clamp(x2, min=0)
        x1 = x1 / x1.sum(dim=1, keepdim=True)
        x2 = x2 / x2.sum(dim=1, keepdim=True)

        # Hellinger distance
        x1_sqrt = torch.sqrt(x1)
        x2_sqrt = torch.sqrt(x2)
        diff2 = (x1_sqrt.unsqueeze(1) - x2_sqrt.unsqueeze(0))**2
        H2 = 0.5 * diff2.sum(dim=2)  # [n1, n2]

        # RBF-like kernel
        K = self.variance * torch.exp(-H2 / (2 * self.lengthscale ** 2))
        return K


class CombinedKernel(gpytorch.kernels.Kernel):
    """
    Combine two kernels: K_seq + K_struct
    Input X is a single concatenated tensor: [seq | struct]
    """

    def __init__(self, kernel_seq, kernel_struct, d_seq):
        super().__init__()
        self.kernel_seq = kernel_seq
        self.kernel_struct = kernel_struct
        self.d_seq = d_seq  # number of sequence dimensions

    def forward(self, X1, X2, **params):
        X1_seq, X1_struct = X1[:, :self.d_seq], X1[:, self.d_seq:]
        X2_seq, X2_struct = X2[:, :self.d_seq], X2[:, self.d_seq:]

        K_seq = self.kernel_seq(X1_seq, X2_seq)
        K_struct = self.kernel_struct(X1_struct, X2_struct)

        return K_seq + K_struct  # could also use product or weighted sum


class MultiInputGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super().__init__(train_x, train_y, likelihood)
        self.likelihood = likelihood
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = kernel

    def forward(self, X):
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X, X)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def get_gp_kernel_model(X_combined, y_train, train: bool = False):
    # Define kernels and model
    d_seq = X_combined.shape[1]  # TODO: Check
    seq_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    struct_kernel = HellingerRBFKernel()
    combined_kernel = CombinedKernel(seq_kernel, struct_kernel, d_seq=d_seq)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = MultiInputGP(X_combined, y_train, likelihood, combined_kernel)

    # Train
    # -----------------------------
    if train:
        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        pbar = tqdm(range(100), desc='Training')
        for i in pbar:
            optimizer.zero_grad()
            output = model(X_combined)
            loss = -mll(output, y_train)
            loss.backward()
            optimizer.step()
            pbar.set_description(f"Training (step {i+1}/{100}, loss: {loss:.4f})")
    return model
