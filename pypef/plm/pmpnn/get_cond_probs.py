
# Protein-MPNN autoregressive model inference

import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

from protein_mpnn_run import run_pmpnn

# ProteinMPNN 21-character alphabet mapping
ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'
AA_TO_IDX = {aa: i for i, aa in enumerate(ALPHABET)}


def run_protein_mpnn_conditional(pdb_path, out_folder="out", seed=13, num_samples=10):
    """Step 1: Run ProteinMPNN on structure and return conditional log-probs."""
    print('\nRunning ProteinMPNN (conditional_probs_only)...\n' + '-' * 80)
    
    cond_probs_dict = run_pmpnn(
        input_seqs=[],               
        pdb_path=pdb_path,
        pdb_path_chains="A",
        path_to_model_weights="",
        out_folder=out_folder,
        score_only=False,            
        ca_only=False,
        num_seq_per_target=num_samples, 
        sampling_temp=0.1,
        model_name="v_48_020",
        use_soluble_model=False,
        seed=seed,
        conditional_probs_only=1,
        conditional_probs_only_backbone=0,
    )
    return cond_probs_dict


def score_mutants_from_probs(cond_probs_input, df, wt_seq):
    """
    Step 2: Score all variants against the WT sequence.
    
    Parameters:
    -----------
    cond_probs_input : str, dict, or np.ndarray
        - File path string pointing to a .npz file
        - Dictionary returned by run_pmpnn()
        - NumPy array of shape [num_samples, L, 21]
    df : pd.DataFrame
        DataFrame containing 'mutated_sequence' column.
    wt_seq : str
        Wild-type amino acid sequence string.
    """
    # 1. Resolve input type (Path, Dict, or NumPy Array)
    if isinstance(cond_probs_input, (str, os.PathLike)):
        if not os.path.exists(cond_probs_input):
            raise FileNotFoundError(f"Could not find file at {cond_probs_input}")
        data = np.load(cond_probs_input)
        if 'conditional_probs' in data:
            log_p = data['conditional_probs']
        elif 'log_p' in data:
            log_p = data['log_p']
        else:
            log_p = data[list(data.keys())[0]]
            
    elif isinstance(cond_probs_input, dict):
        if 'conditional_probs' in cond_probs_input:
            log_p = cond_probs_input['conditional_probs']
        elif 'log_p' in cond_probs_input:
            log_p = cond_probs_input['log_p']
        else:
            raise KeyError("Dictionary must contain 'conditional_probs' or 'log_p' key.")
            
    elif isinstance(cond_probs_input, np.ndarray):
        log_p = cond_probs_input
    else:
        raise TypeError(f"Unsupported input type: {type(cond_probs_input)}")

    # 2. Average across decoding order draws -> Shape: [L, 21]
    mean_log_p = np.mean(log_p, axis=0)

    # 3. Compute Delta Log-Likelihood for each sequence
    predicted_fitness = []

    for _, row in df.iterrows():
        mut_seq = row['mutated_sequence']
        
        delta_ll = 0.0
        for i, (wt_aa, mut_aa) in enumerate(zip(wt_seq, mut_seq)):
            if wt_aa != mut_aa:
                wt_idx = AA_TO_IDX[wt_aa]
                mut_idx = AA_TO_IDX[mut_aa]
                delta_ll += (mean_log_p[i, mut_idx] - mean_log_p[i, wt_idx])
                
        predicted_fitness.append(delta_ll)

    return np.array(predicted_fitness)


if __name__ == "__main__":
    scipt_dirpath = os.path.dirname(__file__)
    pdb_path = f"{scipt_dirpath}/../../../datasets/BLAT_ECOLX/BLAT_ECOLX.pdb"
    csv_path = f"{scipt_dirpath}/../../../datasets/BLAT_ECOLX/BLAT_ECOLX_Stiffler_2015.csv"

    df = pd.read_csv(csv_path)
    true_scores = df['DMS_score'].values

    wt_seq = "MSIQHFRVALIPFFAAFCLPVFAHPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTFKVLLCGAVLSRVDAGQEQLGRRIHYSQNDLVEYSPVTEKHLTDGMTVRELCSAAITMSDNTAANLLLTTIGGPKELTAFLHNMGDHVTRLDRWEPELNEAIPNDERDTTMPAAMATTLRKLLTGELLTLASRQQLIDWMEADKVAGPLLRSALPAGWFIADKSGAGERGSRGIIAALGPDGKPSRIVVIYTTGSQATMDERNRQIAEIGASLIKHW"

    cond_probs_dict = run_protein_mpnn_conditional(pdb_path=pdb_path, out_folder="out", seed=37, num_samples=10)

    # Option A: In-Memory Evaluation
    pred_fitness_mem = score_mutants_from_probs(cond_probs_dict, df, wt_seq)
    rho_mem, _ = spearmanr(true_scores, pred_fitness_mem)
    print(f"\n[In-Memory] Spearman Correlation: {rho_mem:.4f}")

    # Option B: Loaded from Disk (.npz) Evaluation
    npz_path = "out/conditional_probs_only/BLAT_ECOLX.npz"
    if os.path.exists(npz_path):
        pred_fitness_file = score_mutants_from_probs(npz_path, df, wt_seq)
        rho_file, _ = spearmanr(true_scores, pred_fitness_file)
        print(f"[From .npz ] Spearman Correlation: {rho_file:.4f}")
        
        np.testing.assert_array_almost_equal(pred_fitness_mem, pred_fitness_file)
        print("-> Results match exactly between in-memory and loaded .npz!\n" + '=' * 120 + '\n')