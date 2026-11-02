"""Gaussian-process fitting and prediction for PMPNN hybrid components."""
import torch

from pypef.gaussian_process.kermut.tokenizer import Tokenizer
from pypef.gaussian_process.kermut.utils import prepare_kermut_inputs
from pypef.gaussian_process.kermut.gp.instantiate_gp import instantiate_gp
from pypef.gaussian_process.kermut.gp.optimize_gp import optimize_gp
from pypef.gaussian_process.kermut.gp.predict import predict
from pypef.utils.variant_data import extract_pdb_coords
from pypef.plm.pmpnn.get_cond_probs import score_sequences_from_probs


def prepare_pmpnn_gp_inputs(sequences, log_probs, wt_sequence, device):
    embeddings = torch.stack([Tokenizer()(seq) for seq in sequences]).float().to(device)
    scores = torch.as_tensor(score_sequences_from_probs(log_probs, sequences, wt_sequence),
                             dtype=torch.float32, device=device)
    return prepare_kermut_inputs(sequences, embeddings, scores, device=device)


def train_pmpnn_gp(sequences, targets, log_probs, aa_cond_probs, wt_sequence,
                   pdb_struct, device, n_steps=150):
    inputs = prepare_pmpnn_gp_inputs(sequences, log_probs, wt_sequence, device)
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
    return optimize_gp(gp, likelihood, inputs, targets, lr=0.05, n_steps=n_steps)


def predict_pmpnn_gp(gp, likelihood, sequences, log_probs, wt_sequence, device):
    inputs = prepare_pmpnn_gp_inputs(sequences, log_probs, wt_sequence, device)
    mean, _ = predict(gp, likelihood, inputs)
    return mean.detach().cpu().numpy()
