# Protein-MPNN autoregressive model inference

import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

from pypef.plm.utils import resolve_variant_mutations
from pypef.plm.pmpnn.protein_mpnn_run import run_pmpnn

import logging

from pypef.utils.helpers import tqdm
logger = logging.getLogger(__name__)

# ProteinMPNN 21-character alphabet mapping
ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'
AA_TO_IDX = {aa: i for i, aa in enumerate(ALPHABET)}


def run_protein_mpnn_conditional(pdb_path, out_folder="out", seed=13, num_samples=10):
    """Step 1: Run ProteinMPNN on structure and return conditional log-probs."""
    logger.info('Running ProteinMPNN (conditional_probs_only)...')
    
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


def load_conditional_log_probs(cond_probs_input, wt_seq):
    """Load PMPNN log probabilities and average decoding draws to [L, 21]."""
    if isinstance(cond_probs_input, (str, os.PathLike)):
        with np.load(cond_probs_input) as data:
            if 'conditional_probs' in data:
                log_p = data['conditional_probs']
            elif 'log_p' in data:
                log_p = data['log_p']
            else:
                raise KeyError("NPZ must contain 'conditional_probs' or 'log_p'.")
    elif isinstance(cond_probs_input, dict):
        if 'conditional_probs' in cond_probs_input:
            log_p = cond_probs_input['conditional_probs']
        elif 'log_p' in cond_probs_input:
            log_p = cond_probs_input['log_p']
        else:
            raise KeyError("Dictionary must contain 'conditional_probs' or 'log_p'.")
    elif isinstance(cond_probs_input, np.ndarray):
        log_p = cond_probs_input
    else:
        raise TypeError(f"Unsupported input type: {type(cond_probs_input)}")
    log_p = np.asarray(log_p)
    if log_p.ndim == 3:
        if log_p.shape[0] == 0:
            raise ValueError("PMPNN requires at least one decoding draw.")
        log_p = log_p.mean(axis=0)
    if log_p.shape != (len(wt_seq), len(ALPHABET)):
        raise ValueError("PMPNN log probabilities must have shape [L, 21] or [N, L, 21] matching wt_seq.")
    if not np.isfinite(log_p).all():
        raise ValueError("PMPNN log probabilities must be finite.")
    if any(aa not in AA_TO_IDX for aa in wt_seq):
        raise ValueError("Wild-type sequence contains unsupported PMPNN amino acids.")
    return log_p


def score_sequences_from_probs(
        cond_probs_input, 
        sequences=None, 
        wt_seq=None,
        mutation_strings=None, 
        mutation_separator="/"
):
    """Sum mutant-minus-WT conditional log likelihoods over sequence positions."""
    if wt_seq is None:
        raise ValueError("Provide wt_seq.")
    if sequences is None and mutation_strings is None:
        raise ValueError("Provide sequences or mutation_strings.")
    if sequences is not None:
        sequences = list(sequences)
    if mutation_strings is not None:
        mutation_strings = list(mutation_strings)
        if sequences is not None and len(sequences) != len(mutation_strings):
            raise ValueError("Sequences and mutation_strings must have the same length.")
    count = len(sequences) if sequences is not None else len(mutation_strings)
    log_p = load_conditional_log_probs(cond_probs_input, wt_seq)
    wt_idx = [AA_TO_IDX[aa] for aa in wt_seq]
    scores = []
    for variant_index in tqdm(range(count), desc='PMPNN variant scoring using cond. probs.'):
        sequence, positions = resolve_variant_mutations(
            wt_seq, sequences[variant_index] if sequences is not None else None,
            mutation_strings[variant_index] if mutation_strings is not None else None,
            mutation_separator,
        )
        if any(aa not in AA_TO_IDX for aa in sequence):
            raise ValueError("Variant sequence contains unsupported PMPNN amino acids.")
        indices = [AA_TO_IDX[aa] for aa in sequence]
        # Preserve the original mutation-loop accumulation order and precision
        delta_ll = 0.0
        for i in positions:
            delta_ll += log_p[i, indices[i]] - log_p[i, wt_idx[i]]
        scores.append(delta_ll)
    return np.asarray(scores)


def score_mutants_from_probs(cond_probs_input, df, wt_seq):
    """Score a DataFrame containing a 'mutated_sequence' column."""
    return score_sequences_from_probs(cond_probs_input, df['mutated_sequence'], wt_seq)


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
    logger.info(f"\n[In-Memory] Spearman Correlation: {rho_mem:.4f}")

    # Option B: Loaded from Disk (.npz) Evaluation
    npz_path = "out/conditional_probs_only/BLAT_ECOLX.npz"
    if os.path.exists(npz_path):
        pred_fitness_file = score_mutants_from_probs(npz_path, df, wt_seq)
        rho_file, _ = spearmanr(true_scores, pred_fitness_file)
        logger.info(f"[From .npz ] Spearman Correlation: {rho_file:.4f}")
        
        np.testing.assert_array_almost_equal(pred_fitness_mem, pred_fitness_file)
        logger.info("-> Results match exactly between in-memory and loaded .npz!\n" + '=' * 120 + '\n')