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
    mutants, sequences, y, test_size=0.80, random_state=42
)

#X_struct = get_probs_from_mutations(m_train)        # [N, 20]


print("Getting ProSST models")
pdb = 'datasets/BLAT_ECOLX/BLAT_ECOLX.pdb'
wt_seq = get_wt_sequence('datasets/BLAT_ECOLX/blat_ecolx_wt.fasta')
prosst_base_model, prosst_lora_model, prosst_tokenizer, prosst_optimizer = get_prosst_models()
prosst_vocab = prosst_tokenizer.get_vocab()
prosst_base_model = prosst_base_model.to("cuda")

esm_base_model, esm_lora_model, esm_tokenizer, esm_optimizer = get_esm_models()

wt_prosst_input_ids, prosst_attention_mask, wt_structure_input_ids = get_structure_quantizied(
        pdb, prosst_tokenizer, wt_seq, verbose=True
)

wt_esm_input_ids, esm_attention_mask_2 = tokenize_sequences([wt_seq], esm_tokenizer)
wt_esm_input_ids = torch.tensor( wt_esm_input_ids[0], dtype=torch.long)  # shape (L,)

#wt_structure_input_ids = wt_structure_input_ids[0, 1:-1].tolist()  # Remove CLS/EOS
#X_seq = torch.tensor(extract_esm_embeddings(s_train)).float()  # [N, d_seq]
#X_seq = torch.tensor(extract_prosst_embeddings(
#    prosst_base_model, prosst_tokenizer, s_train, wt_structure_input_ids
#))

x_prosst_tok_train, prosst_attention_mask_2 = tokenize_sequences(s_train, prosst_tokenizer)
x_prosst_emb_train = plm_inference(x_prosst_tok_train, wt_prosst_input_ids, prosst_attention_mask, prosst_base_model, 
                                   extract_emb=True, wt_structure_input_ids=wt_structure_input_ids).cpu()

x_esm_tok_train, esm_attention_mask = tokenize_sequences(s_train, esm_tokenizer)
x_esm_emb_train = plm_inference(x_esm_tok_train, wt_esm_input_ids, esm_attention_mask, 
                                esm_base_model, extract_emb=True).cpu()

y_train = torch.tensor(y_train).float()
y_test = torch.tensor(y_test).float()

# Concatenate features
X_combined = torch.cat([x_prosst_emb_train, x_esm_emb_train], dim=-1)  # Concenation is necessary as GPkernel does not accept a tuple as input 
d_seq = x_prosst_emb_train.shape[1]  # TODO: Check

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
#X_seq_test = torch.tensor(extract_prosst_embeddings(prosst_base_model, prosst_tokenizer, s_test, wt_structure_input_ids))
x_prosst_tok_test, prosst_attention_mask_2 = tokenize_sequences(s_test, prosst_tokenizer)
x_prosst_emb_test = plm_inference(x_prosst_tok_test, wt_prosst_input_ids, prosst_attention_mask, prosst_base_model, 
                                  extract_emb=True, wt_structure_input_ids=wt_structure_input_ids).cpu()

x_esm_tok_test, esm_attention_mask = tokenize_sequences(s_test, esm_tokenizer)
x_esm_emb_test = plm_inference(x_esm_tok_test, wt_esm_input_ids, esm_attention_mask, 
                                esm_base_model, extract_emb=True).cpu()

X_test_combined = torch.cat([x_prosst_emb_test, x_esm_emb_test], dim=-1)

model.eval()
likelihood.eval()


with torch.no_grad(), gpytorch.settings.fast_pred_var():
    pred_train = likelihood(model(X_combined))
    y_pred_train = pred_train.mean.cpu().numpy()

    pred = likelihood(model(X_test_combined))
    y_pred = pred.mean.cpu().numpy()


from scipy.stats import spearmanr

rho, p = spearmanr(y_train, y_pred_train)
print("Spearman rho SciPy TRAIN:                   ", rho)
print("Spearman soft TRAIN:                        ", spearman_soft(y_train, torch.from_numpy(y_pred_train)).item())
print("Correlation loss Spearman TRAIN:            ", correlation_loss(y_train, torch.from_numpy(y_pred_train), method="spearman"))
print("Correlation hybrid MSE loss Spearman TRAIN: ", hybrid_corr_mse_loss(y_train, torch.from_numpy(y_pred_train)))
print("Correlation loss Pearson     TRAIN:         ", correlation_loss(y_train, torch.from_numpy(y_pred_train), method="pearson"))
print("Correlation loss Pearson 2   TRAIN:         ", pearson_loss(y_train, torch.from_numpy(y_pred_train)))
y_train_t  = y_train.float().unsqueeze(0)       # shape (1, n)
y_pred_train_t  = torch.from_numpy(y_pred_train).float().unsqueeze(0)    # shape (1, n)
#print("Spearman corr diff (ChatGPT) TRAIN:", spearman_corr_differentiable(y_train_t, y_pred_train_t).item())
#print("Spearman2 torchsort          TRAIN:", spearmanr2(y_train_t, y_pred_train_t).item())
print()
rho, p = spearmanr(y_test, y_pred)
print("Spearman rho SciPy TEST:                   ", rho)
print("Spearman soft TEST:                        ", spearman_soft(y_test, torch.from_numpy(y_pred)).item())
print("Correlation loss Spearman TEST:            ", correlation_loss(y_test, torch.from_numpy(y_pred), method="spearman"))
print("Correlation hybrid MSE loss Spearman TEST: ", hybrid_corr_mse_loss(y_test, torch.from_numpy(y_pred)))
print("Correlation loss Pearson TEST:             ", correlation_loss(y_test, torch.from_numpy(y_pred), method="pearson"))
print("Correlation loss Pearson 2 TEST:           ", pearson_loss(y_test, torch.from_numpy(y_pred)))
y_test_t  = y_test.float().unsqueeze(0)       # shape (1, n)
y_pred_t  = torch.from_numpy(y_pred).float().unsqueeze(0)    # shape (1, n)
#print("Spearman corr diff (ChatGPT) TEST:", spearman_corr_differentiable(y_test_t, y_pred_t).item())
#print("Spearman2 torchsort          TEST:", spearmanr2(y_test_t, y_pred_t).item())
