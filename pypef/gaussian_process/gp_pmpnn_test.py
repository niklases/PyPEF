import torch
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
import gpytorch
import torch.nn.functional as F
import os
from proteinmpnn.protein_mpnn_utils import ProteinMPNN  # pip install proteinmpnn

import torch
import gpytorch
import torch.nn.functional as F


"""
pip install proteinmpnn

python -m proteinmpnn.protein_mpnn_run \
       --pdb-path "example_data/blat_ecolx/BLAT_ECOLX.pdb" \
       --save-score 1 \
       --conditional-probs-only 1 \
       --num-seq-per-target 10 \
       --batch-size 1 \
       --out-folder "pmpnn_out" \
       --seed 37
"""

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


class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = kernel   # <- Kermut kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x, x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def get_probs_from_pmlnn_npz(npz_file="pmpnn_out/conditional_probs_only/BLAT_ECOLX.npz"):
    print(f"Getting PMPNN amino acid probs from NPZ file: {os.path.abspath(npz_file)}...")
    data = np.load(npz_file)
    data_dict = {k: data[k] for k in data.files}
    data.close()
    log_ps = data_dict["log_p"][..., :20]

    mean_probs = np.mean(log_ps, axis=0)
    mean_probs = torch.tensor(mean_probs)         # [L, 20]
    probs = F.softmax(mean_probs, dim=-1)         # [L, 20]
    return probs


def get_probs_from_mutations(mutations, probs=None):
    if probs is None:
        probs = get_probs_from_pmlnn_npz()
    x_list = []
    for m in mutations:
        pos = int(m[1:-1])  # Only single muts for now
        x_list.append(probs[pos - 1])
    return torch.stack(x_list)
    
# For now, running with CLI...
# https://github.com/petergroth/kermut/blob/main/example_scripts/conditional_probabilities_single.sh
"""
python -m proteinmpnn.protein_mpnn_run \
       --pdb-path "example_data/blat_ecolx/BLAT_ECOLX.pdb" \
       --save-score 1 \
       --conditional-probs-only 1 \
       --num-seq-per-target 10 \
       --batch-size 1 \
       --out-folder "pmpnn_out" \
       --seed 37
"""


if __name__ == '__main__':
    data = np.load("pmpnn_out/conditional_probs_only/BLAT_ECOLX.npz")
    data_dict = {k: data[k] for k in data.files}
    data.close()

    for k, v in data_dict.items():
        print(f"K:{k}\nv:{v}\n{np.shape(v)}\n\n")


    import matplotlib.pyplot as plt
    from matplotlib.cm import get_cmap
    from cycler import cycler
    cmap = get_cmap('rainbow')
    amino_acids = list('ACDEFGHIKLMNPQRSTVWY') # Excluded: X
    colors = [cmap(i / len(amino_acids)) for i in range(len(amino_acids))]
    plt.rcParams['axes.prop_cycle'] = cycler(color=colors)
    log_ps = data_dict["log_p"][..., :20]

    mean_probs = np.mean(log_ps, axis=0)

    print(np.shape(mean_probs))
    #plt.figure(figsize=(20,5))
    #plt.plot(mean_probs, label=amino_acids, linewidth=0.5)
    # Annotate above peaks
    #for pos, (aa, val) in enumerate(zip(top_aa, top_val)):
    #    if pos % 1 == 0:
    #        plt.text(pos, val + 0.02, aa, ha='center', va='bottom', fontsize=4, color='black')

    #plt.legend(ncol=7)
    #plt.show()

    mean_probs = torch.tensor(mean_probs)         # [L, 20]
    probs = F.softmax(mean_probs, dim=-1)         # [L, 20]


    df = pd.read_csv('example_data/blat_ecolx/BLAT_ECOLX_Stiffler_2015.csv')
    print(df.columns)
    mutants = df['mutant'].to_list()
    sequences = df['mutated_sequence'].to_list()
    y = df['DMS_score'].to_list()

    m_train, m_test, s_train, s_test, y_train, y_test = train_test_split(
        mutants, sequences, y, test_size=0.33, random_state=42)  # train_size=100, test_size=200,

    X_train = get_probs_from_mutations(m_train)  # shape [N, 20] 

    y_train = torch.tensor(y_train)    # shape [N]

    # Initialize kernel
    kernel = HellingerRBFKernel()
    kernel.lengthscale = torch.tensor(0.5)  # optional manual override
    kernel.variance = torch.tensor(1.0)

    # Compute covariance matrix
    K = kernel(probs, probs)
    print(K.shape)  # [286, 286]

    # Visualize
    #plt.figure(figsize=(8, 6))
    #plt.imshow(K.detach().numpy(), cmap='viridis')
    #plt.colorbar(label='Kernel value')
    #plt.title('Hellinger RBF Kernel Matrix')
    #plt.xlabel('Sequence position')
    #plt.ylabel('Sequence position')
    #plt.show()


    # Training
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPModel(X_train, y_train, likelihood, kernel)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in tqdm(range(100)):
        optimizer.zero_grad()
        output = model(X_train)
        loss = -mll(output, y_train)
        loss.backward()
        optimizer.step()

    X_test = get_probs_from_mutations(m_test)
    model.eval()
    likelihood.eval()

    with torch.no_grad():
        pred_train = likelihood(model(X_train))
        y_pred_train = pred_train.mean.cpu().numpy()

    with torch.no_grad():
        pred_test = likelihood(model(X_test))
        y_pred_test = pred_test.mean.cpu().numpy()

    from scipy.stats import spearmanr
    rho, p = spearmanr(y_train.numpy(), y_pred_train)
    print("Spearman rho TRAIN:", rho)
    print("p-value TRAIN:", p)
    rho, p = spearmanr(y_test, y_pred_test)
    print("Spearman rho:", rho)
    print("p-value:", p)