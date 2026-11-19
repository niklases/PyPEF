"""Gaussian-process fitting and prediction for PMPNN hybrid components."""
import torch

from pypef.gaussian_process.kermut.tokenizer import Tokenizer
from pypef.gaussian_process.kermut.utils import prepare_kermut_inputs
from pypef.gaussian_process.kermut.gp.instantiate_gp import instantiate_gp
from pypef.gaussian_process.kermut.gp.optimize_gp import optimize_gp
from pypef.gaussian_process.kermut.gp.predict import predict
from pypef.utils.variant_data import extract_pdb_coords
from pypef.plm.pmpnn.get_cond_probs import score_sequences_from_probs


def prepare_pmpnn_gp_inputs(sequences, log_probs, wt_sequence, device, scores=None):
    """
    Inputs for the PMPNN GP: one-hot sequence embeddings and the PMPNN zero-shot scores.

    `scores` optionally supplies those zero-shot scores when the caller already has them (the
    hybrid model computes them for its own `PMPNN_base` ensemble member), which avoids scoring the
    same sequence set twice. They must be the scores of exactly `sequences`, in order; without
    them they are computed here, so the function also works standalone.
    """
    embeddings = torch.stack([Tokenizer()(seq) for seq in sequences]).float().to(device)
    if scores is None:
        scores = score_sequences_from_probs(log_probs, sequences, wt_sequence)
    elif len(scores) != len(sequences):
        raise ValueError(
            f"Precomputed PMPNN scores must align with sequences: got {len(scores)} "
            f"scores for {len(sequences)} sequences."
        )
    scores = torch.as_tensor(scores, dtype=torch.float32, device=device)
    return prepare_kermut_inputs(sequences, embeddings, scores, device=device)


def train_pmpnn_gp(sequences, targets, log_probs, aa_cond_probs, wt_sequence,
                   pdb_struct, device, n_steps=150, lr=0.05, scores=None):
    inputs = prepare_pmpnn_gp_inputs(sequences, log_probs, wt_sequence, device, scores=scores)
    targets = torch.as_tensor(targets, dtype=torch.float32, device=device)
    gp, likelihood = instantiate_gp(
        train_inputs=inputs, train_targets=targets,
        gp_inputs={"aa_cond_probs": aa_cond_probs,
                   "struct_coords": torch.as_tensor(extract_pdb_coords(
                       pdb_struct, target_len=len(wt_sequence)), dtype=torch.float32, device=device),
                   "wt_seq": wt_sequence},
        use_structure_kernel=True, use_sequence_kernel=True,
        use_zero_shot=True, device=device,
    )
    return optimize_gp(gp, likelihood, inputs, targets, lr=lr, n_steps=n_steps)


def predict_pmpnn_gp(gp, likelihood, sequences, log_probs, wt_sequence, device, scores=None):
    inputs = prepare_pmpnn_gp_inputs(sequences, log_probs, wt_sequence, device, scores=scores)
    mean, _ = predict(gp, likelihood, inputs)
    return mean.detach().cpu().numpy()
