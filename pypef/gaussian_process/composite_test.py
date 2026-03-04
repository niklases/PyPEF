from typing import Literal, Tuple
from sklearn.model_selection import train_test_split
import torch

import gpytorch
from gpytorch.kernels import ScaleKernel
import pandas as pd
from tqdm import tqdm

from pypef.gaussian_process.gp_esm2_test import extract_esm_embeddings
from pypef.gaussian_process.gp_pmpnn_test import HellingerRBFKernel, get_probs_from_mutations
from pypef.gaussian_process.gp_prosst_test import (extract_prosst_embeddings, get_prosst_models, 
                            get_structure_quantizied, read_fasta_biopython)
from pypef.plm.utils import spearman_soft, correlation_loss, hybrid_corr_mse_loss, pearson_loss


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
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = kernel

    def forward(self, X):
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X, X)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# -----------------------------
# Load and preprocess data
# -----------------------------
df = pd.read_csv('datasets/BLAT_ECOLX/BLAT_ECOLX_Stiffler_2015.csv')

print(df.columns)
mutants = df['mutant'].to_list()
sequences = df['mutated_sequence'].to_list()
y = df['DMS_score'].to_list()

m_train, m_test, s_train, s_test, y_train, y_test = train_test_split(
    mutants, sequences, y, train_size=100, test_size=100, random_state=42
)

#X_struct = get_probs_from_mutations(m_train)        # [N, 20]


print("Getting ProSST models")
pdb = 'datasets/BLAT_ECOLX/BLAT_ECOLX.pdb'
wt_seq = list(read_fasta_biopython('datasets/BLAT_ECOLX/blat_ecolx_wt.fasta').values())[0]
prosst_base_model, prosst_lora_model, prosst_tokenizer, prosst_optimizer = get_prosst_models()
prosst_vocab = prosst_tokenizer.get_vocab()
prosst_base_model = prosst_base_model.to("cuda")

input_ids, prosst_attention_mask, structure_input_ids = get_structure_quantizied(
        pdb, prosst_tokenizer, wt_seq, verbose=True
)
wt_structure_input_ids = structure_input_ids[0, 1:-1].tolist()  # Remove CLS/EOS
#X_seq = torch.tensor(extract_esm_embeddings(s_train)).float()  # [N, d_seq]
X_seq = torch.tensor(extract_prosst_embeddings(
    prosst_base_model, prosst_tokenizer, s_train, wt_structure_input_ids
))
y_train = torch.tensor(y_train).float()
y_test = torch.tensor(y_test).float()

# Concatenate features
X_combined = torch.cat([X_seq, X_seq], dim=-1)  # Concenation is necessary as GPkernel does not accept a tuple as input 
d_seq = X_seq.shape[1]

# -----------------------------
# Define kernels and model
# -----------------------------
seq_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
struct_kernel = HellingerRBFKernel()
combined_kernel = CombinedKernel(seq_kernel, struct_kernel, d_seq=d_seq)

likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = MultiInputGP(X_combined, y_train, likelihood, combined_kernel)

# -----------------------------
# Train
# -----------------------------
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
    pbar.set_description(f"Training (loss: {loss:.4f})")

# -----------------------------
# Test
# -----------------------------
#X_struct_test = get_probs_from_mutations(m_test)
#X_seq_test = torch.tensor(extract_esm_embeddings(s_test)).float()
X_seq_test = torch.tensor(extract_prosst_embeddings(prosst_base_model, prosst_tokenizer, s_test, wt_structure_input_ids))
X_test_combined = torch.cat([X_seq_test, X_seq_test], dim=-1)

model.eval()
likelihood.eval()


with torch.no_grad(), gpytorch.settings.fast_pred_var():
    pred_train = likelihood(model(X_combined))
    y_pred_train = pred_train.mean.cpu().numpy()

    pred = likelihood(model(X_test_combined))
    y_pred = pred.mean.cpu().numpy()


from scipy.stats import spearmanr

rho, p = spearmanr(y_train, y_pred_train)
print("Spearman rho SciPy           TRAIN:", rho)
print("Spearman soft                TRAIN:", spearman_soft(y_train, torch.from_numpy(y_pred_train)).item())
print("Correlation loss Spearman    TRAIN:", correlation_loss(y_train, torch.from_numpy(y_pred_train), method="spearman"))
print("Correlation hybrid MSE loss Spearman    TRAIN:", hybrid_corr_mse_loss(y_train, torch.from_numpy(y_pred_train)))
print("Correlation loss Pearson     TRAIN:", correlation_loss(y_train, torch.from_numpy(y_pred_train), method="pearson"))
print("Correlation loss Pearson 2   TRAIN:", pearson_loss(y_train, torch.from_numpy(y_pred_train)))
y_train_t  = y_train.float().unsqueeze(0)       # shape (1, n)
y_pred_train_t  = torch.from_numpy(y_pred_train).float().unsqueeze(0)    # shape (1, n)
#print("Spearman corr diff (ChatGPT) TRAIN:", spearman_corr_differentiable(y_train_t, y_pred_train_t).item())
#print("Spearman2 torchsort          TRAIN:", spearmanr2(y_train_t, y_pred_train_t).item())

rho, p = spearmanr(y_test, y_pred)
print("Spearman rho SciPy           TEST:", rho)
print("Spearman soft                TEST:", spearman_soft(y_test, torch.from_numpy(y_pred)).item())
print("Correlation loss Spearman    TEST:", correlation_loss(y_test, torch.from_numpy(y_pred), method="spearman"))
print("Correlation hybrid MSE loss Spearman    TEST:", hybrid_corr_mse_loss(y_test, torch.from_numpy(y_pred)))
print("Correlation loss Pearson     TEST:", correlation_loss(y_test, torch.from_numpy(y_pred), method="pearson"))
print("Correlation loss Pearson 2   TEST:", pearson_loss(y_test, torch.from_numpy(y_pred)))
y_test_t  = y_test.float().unsqueeze(0)       # shape (1, n)
y_pred_t  = torch.from_numpy(y_pred).float().unsqueeze(0)    # shape (1, n)
#print("Spearman corr diff (ChatGPT) TEST:", spearman_corr_differentiable(y_test_t, y_pred_t).item())
#print("Spearman2 torchsort          TEST:", spearmanr2(y_test_t, y_pred_t).item())
