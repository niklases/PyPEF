

"""
Gaussian process optimization similar (but less sophisticated compared) to 
Kermut: Composite kernel regression for protein variant effects
Peter Mørch Groth, Mads Herbert Kerrn, Lars Olsen, Jesper Salomon, Wouter Boomsma
2024, 38th Conference on Neural Information Processing Systems (NeurIPS 2024).
TL;DR: Gaussian process regression model with a novel composite kernel, Kermut, achieves 
state-of-the-art variant effect prediction while providing meaningful uncertainties.
https://openreview.net/forum?id=jM9atrvUii
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from scipy.stats import spearmanr
from Bio import SeqIO
import gpytorch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from tqdm import tqdm

from pypef.plm.prosst_lora_tune import (
    get_prosst_models, get_structure_quantizied
)
from pypef.plm.inference import tokenize_sequences, plm_inference
from pypef.utils.helpers import get_device


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



def read_fasta_biopython(filepath):
    return {record.id: str(record.seq)
            for record in SeqIO.parse(filepath, "fasta")}


def extract_prosst_embeddings(
    model,
    prosst_tokenizer,
    sequences,
    structures,   # either List[List[int]] for each sequence, or a single List[int] (WT)
    device='cuda'
):
    """
    Extract ProSST embeddings for a list of sequences.

    Args:
        model: ProSST model
        prosst_tokenizer: tokenizer for sequences
        sequences: List[str] of sequences
        structures: Either
            - List[List[int]] of per-sequence structures
            - Single List[int] (WT) to be reused
        device: 'cpu' or 'cuda'

    Returns:
        X: np.ndarray of shape [num_sequences, embedding_dim]
    """

    embeddings = []
    model.eval()

    # Detect single WT structure and repeat for all sequences
    if isinstance(structures[0], int):
        print("Using \"WT\"/single reference structure for all sequences!")
        # single structure provided
        structures = [structures] * len(sequences)

    assert len(sequences) == len(structures), \
        (f"Number of sequences must match number of structures "
         f"{len(sequences)} != {len(structures)}")

    for seq, struct in tqdm(zip(sequences, structures),
                            total=len(sequences),
                            desc="Embedding (ProSST)"):
        # Tokenize sequence
        tokenized = prosst_tokenizer(
            [seq],
            max_length=len(seq) + 2,
            return_tensors="pt",
            padding=False,
            truncation=False
        )

        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)

        # Add special tokens to structure: [CLS] ... [EOS]
        structure_input_ids = torch.tensor(
            [1, *struct, 2],
            dtype=torch.long
        ).unsqueeze(0).to(device)

        # Safety check
        assert input_ids.shape == structure_input_ids.shape, \
            f"Shape mismatch: {input_ids.shape} vs {structure_input_ids.shape}"

        # Forward pass
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                ss_input_ids=structure_input_ids,
                output_hidden_states=True,
                return_dict=True
            )

        token_embeddings = outputs.hidden_states[-1]  # (1, L+2, D)

        # Mean pool over residues (exclude CLS/EOS)
        seq_embedding = token_embeddings[0, 1:-1].mean(dim=0)

        embeddings.append(seq_embedding.cpu().numpy())

    X = np.vstack(embeddings)
    print("Embedding matrix shape:", X.shape)
    return X


def gp_fit():
    pass



def gp_fit_sklearn():
    pass




if __name__ == '__main__':
    wt_seq = list(read_fasta_biopython('datasets/BLAT_ECOLX/blat_ecolx_wt.fasta').values())[0]
    pdb = 'datasets/BLAT_ECOLX/BLAT_ECOLX.pdb'
    device = get_device()
    print("Getting ProSST models")
    prosst_base_model, prosst_lora_model, prosst_tokenizer, prosst_optimizer = get_prosst_models()
    prosst_vocab = prosst_tokenizer.get_vocab()
    prosst_base_model = prosst_base_model.to(device)
    
    print(f"Getting structure tokens...")
    wt_input_ids, prosst_attention_mask, structure_input_ids = get_structure_quantizied(
        pdb, prosst_tokenizer, wt_seq, verbose=True
    )

    df = pd.read_csv('datasets/BLAT_ECOLX/BLAT_ECOLX_Stiffler_2015.csv')
    sequences = df['mutated_sequence'].to_list()
    y = df['DMS_score'].to_list()

    s_train, s_test, y_train, y_test = train_test_split(
        sequences, y, test_size=0.33, random_state=42)  # train_size=100, test_size=200,

    # --- Step 2: Extract ProSST embeddings ---
    print(structure_input_ids)
    print('np.shape(structure_input_ids):', np.shape(structure_input_ids))
    wt_structure_input_ids = structure_input_ids  

    X_emb_train = extract_prosst_embeddings(prosst_base_model, prosst_tokenizer, s_train, wt_structure_input_ids[0, 1:-1].tolist()) # Remove CLS/EOS

    x_train_2, prosst_attention_mask_2 = tokenize_sequences(s_train, prosst_tokenizer)
    assert len(prosst_attention_mask[0]) == len(prosst_attention_mask_2), f"{len(prosst_attention_mask[0])}\n  !=\n  {len(prosst_attention_mask_2)}"

    X_emb_train_2 = plm_inference(x_train_2, wt_input_ids, prosst_attention_mask, prosst_base_model, 
                                  extract_emb=True, wt_structure_input_ids=wt_structure_input_ids) #[0, 1:-1])

    print("Embedding extraction done")
    assert np.shape(X_emb_train) == np.shape(X_emb_train_2)

    # --- Step 3: Fit Gaussian Process ---
    if USE_SCIKIT_LEARN:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel

        kernel = 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)
        gpr.fit(X_emb_train, y_train)

    else:  # GPYTORCH
        import gpytorch

        X_train_t = torch.tensor(X_emb_train, dtype=torch.float32).to(device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        gp_model = ExactGPModel(X_train_t, y_train_t, likelihood).to(device)

        gp_model.train()
        likelihood.train()

        optimizer = torch.optim.Adam(gp_model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)

        for i in range(100):
            optimizer.zero_grad()
            output = gp_model(X_train_t)
            loss = -mll(output, y_train_t)
            loss.backward()
            print(f"Iter {i+1}/100 - Loss: {loss.item():.3f}")
            optimizer.step()

    # --- Step 4: Extract ProSST embeddings for test sequences ---
    print("Extracting ProSST embeddings for test sequences...")
    X_emb_test = extract_prosst_embeddings(
        model=prosst_base_model,
        prosst_tokenizer=prosst_tokenizer,
        sequences=s_test,
        structures=wt_structure_input_ids[0, 1:-1].tolist(),  # still using same WT structure
        device=device
    )
    print("Test embeddings shape:", X_emb_test.shape)

    x_test_2, prosst_attention_mask_2 = tokenize_sequences(s_test, prosst_tokenizer)
    X_emb_test_2 = plm_inference(x_test_2, wt_input_ids, prosst_attention_mask, prosst_base_model, 
                                  extract_emb=True, wt_structure_input_ids=wt_structure_input_ids)  #[0, 1:-1])

    # --- Step 5: Predict with Gaussian Process ---
    for x_t in [torch.tensor(X_emb_test).to(device), X_emb_test_2]:
        if USE_SCIKIT_LEARN:
            y_mean, y_std = gpr.predict(x_t, return_std=True)
        else:
            #X_test_t = torch.tensor(x_t, dtype=torch.float32).to(device)
            gp_model.eval()
            likelihood.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                pred = likelihood(gp_model(x_t))
                y_mean = pred.mean.cpu().numpy()
                lower, upper = pred.confidence_region()  # optional 95% CI

        print("Predicted fitness:", y_mean)
        print("Spearman correlation:", spearmanr(y_test, y_mean))  #  SignificanceResult(statistic=0.8617991109937434, pvalue=0.0)