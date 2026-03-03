
import torch
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
import gpytorch

# --- Step 1: Load a pretrained ESM plm_model ---
from esm import pretrained  # pip install fair-esm

from pypef.plm.inference import plm_inference, tokenize_sequences
from pypef.utils.variant_data import get_wt_sequence
from pypef.plm.esm_lora_tune import get_esm_models


"""
git clone https://github.com/facebookresearch/esm.git
cd esm
pip install .
"""


device = "cuda" if torch.cuda.is_available() else "cpu"

USE_SCIKIT_LEARN = False


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


def extract_esm_embeddings(sequences):
    embeddings = []

    for seq in tqdm(sequences, 'Embedding (ESM)'):
        data = [("protein", seq)]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)
        with torch.no_grad():
            results = plm_model(batch_tokens, repr_layers=[33], return_contacts=False)
        token_representations = results["representations"][33]
        # Mean-pool per-residue representations (excluding special tokens)
        seq_embedding = token_representations[0, 1:len(seq)+1].mean(0)
        embeddings.append(seq_embedding.cpu().numpy())

    X = np.vstack(embeddings)

    return X


def extract_esm_emb(sequences, wt_seq):
    base_model, _lora_model, tokenizer, _optimizer = get_esm_models()
    xs, attn_mask = tokenize_sequences(sequences, tokenizer=tokenizer)
    wt_tokens = tokenize_sequences([wt_seq], tokenizer=tokenizer)
    wt_tokens = torch.tensor(wt_tokens[0], dtype=torch.long)  # shape (L,)
    X = plm_inference(xs, wt_tokens, attn_mask, base_model, extract_emb=True)
    return X

#self.scoress = inference(
#            np.array(self.variant_sequencess).flatten(), llm=self.model, pdb_file=self.pdb, wt_seq=self.wt_seq
#).numpy()



plm_model, alphabet = pretrained.esm1v_t33_650M_UR90S_3()  #esm2_t33_650M_UR50D()
plm_model = plm_model.to(device)
batch_converter = alphabet.get_batch_converter()
plm_model.eval()  # disable dropout

if __name__ == '__main__':
    # Load ESM-2 (you can choose different sizes: 35M, 150M, 650M, 3B)

    # --- Example dataset ---
    # sequences: list of amino acid strings
    # y: list/array of experimental fitness values

    df = pd.read_csv('datasets/BLAT_ECOLX/BLAT_ECOLX_Stiffler_2015.csv')
    sequences = df['mutated_sequence'].to_list()
    y = df['DMS_score'].to_list()
    wt_seq_file_blat_ecolx = 'datasets/BLAT_ECOLX/blat_ecolx_wt.fasta'
    wt_seq = get_wt_sequence(wt_seq_file_blat_ecolx)

    s_train, s_test, y_train, y_test = train_test_split(
        sequences, y, test_size=0.33, random_state=42)  # train_size=100, test_size=200,

    # --- Step 2: Extract ESM embeddings ---
    X2 = extract_esm_embeddings(s_train)
    print("Embedding extraction done")
    print(np.shape(X2))

    X = extract_esm_emb(sequences=s_train, wt_seq=wt_seq)
    X_cpu = X.cpu().numpy()
    print(np.shape(X_cpu))
    assert np.shape(X_cpu) == np.shape(X2), f"{np.shape(X)}\n  !=\n{np.shape(X2)}"
    print("Shape X :", X_cpu.shape)
    print("Shape X2:", X2.shape)

    print("Max abs diff:", np.max(np.abs(X_cpu[0] - X2[0])))
    print("Mean abs diff:", np.mean(np.abs(X_cpu[0] - X2[0])))
    cos_sim = np.dot(X_cpu[0], X2[0]) / (np.linalg.norm(X_cpu[0]) * np.linalg.norm(X2[0]))
    print("Cosine sim:", cos_sim)
    print(f"{X_cpu[0]}\n\n{X2[0]}")
    assert cos_sim > 0.99
    #assert X_cpu[0] == X2[0], f"{X_cpu[0]}\n  !=\n{X2[0]}"
    # --- Step 3: Build and fit a Gaussian Process ---

    if USE_SCIKIT_LEARN:
        # RBF kernel + WhiteKernel (noise term)
        kernel = 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)
        gpr.fit(X, y_train)

    else: # GPYTORCH
        # Likelihood
        # Suppose X: [num_sequences, embedding_dim], y: [num_sequences]
        X = X.to(torch.float32).to(device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        gp_model = ExactGPModel(X, y_train, likelihood).to(device)

        gp_model.train()
        likelihood.train()

        optimizer = torch.optim.Adam(gp_model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)

        training_iter = 100
        for i in range(training_iter):
            optimizer.zero_grad()
            output = gp_model(X)
            loss = -mll(output, y_train)
            loss.backward()
            print(f"Iter {i+1}/{training_iter} - Loss: {loss.item():.3f}")
            optimizer.step()

    # --- Step 4: Predict on new sequences ---
    test_embeddings = []

    for seq in tqdm(s_test):
        data = [("protein", seq)]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)
        with torch.no_grad():
            results = plm_model(batch_tokens, repr_layers=[33], return_contacts=False)
        seq_embedding = results["representations"][33][0, 1:len(seq)+1].mean(0)
        test_embeddings.append(seq_embedding.cpu().numpy())

    X_test2 = np.array(test_embeddings)  # or np.vstack
    print("Test embeddings shape:", X_test2.shape)
    X_test = extract_esm_emb(s_test, wt_seq)

    if USE_SCIKIT_LEARN:
        y_mean, y_std = gpr.predict(X_test2, return_std=True)
    else:  # GPYTORCH
        for X_ in [X_test2, X_test]:
            X_ = torch.tensor(X_, dtype=torch.float32).to(device)
            gp_model.eval()
            likelihood.eval()

            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                # Suppose X_test: [num_test, embedding_dim]
                X_ = torch.tensor(X_, dtype=torch.float32).to(device)
                pred = likelihood(gp_model(X_))
                y_mean = pred.mean  # predicted mean
                lower, upper = pred.confidence_region()  # 95% confidence interval
                y_mean = y_mean.cpu().numpy()

                print("Predicted fitness:", y_mean)
                #print("Uncertainty (std):", y_std)

                print(spearmanr(y_test, y_mean))
