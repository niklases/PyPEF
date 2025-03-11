# Niklas Siedhoff
# PyPEF - Pythonic Protein Engineering Framework

# Using (training, testing/infering) ProSST model(s) published under 
# GNU GENERAL PUBLIC LICENSE: GPL-3.0 license
# https://github.com/ai4protein/ProSST
#import warnings
#warnings.filterwarnings('error')

from sys import path
import os
path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm

from transformers import AutoModelForMaskedLM, AutoTokenizer


from peft import LoraConfig, get_peft_model

from pypef.llm.esm_lora_tune import corr_loss, get_batches
from pypef.llm.prosst_structure.quantizer import PdbQuantizer
from pypef import __path__
pypef_path = __path__[0]


def get_logits_from_full_seqs(sequences, model, input_ids, attention_mask, structure_input_ids, train: bool = False, verbose: bool = True):
    if train:
        outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                ss_input_ids=structure_input_ids
        )
    else:
        with torch.no_grad():
            outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    ss_input_ids=structure_input_ids
            )
    vocab = tokenizer.get_vocab()
    logits = torch.log_softmax(outputs.logits[:, 1:-1], dim=-1).squeeze()
    for i_s, sequence in enumerate(tqdm(sequences, disable=not verbose, desc='Getting sequence logits')):
        for i_aa, aa in enumerate(sequence):
            if i_aa == 0:
                seq_log_probs = logits[i_aa, vocab[aa]].reshape(1)
            else:
                seq_log_probs = torch.cat((seq_log_probs, logits[i_aa, vocab[aa]].reshape(1)), 0)
        if i_s == 0:
            log_probs = torch.sum(torch.Tensor(seq_log_probs)).reshape(1)
        else:
            log_probs = torch.cat((log_probs, torch.sum(torch.Tensor(seq_log_probs)).reshape(1)), 0)
    return log_probs


def get_scores(sequences, y_true, pdb_path):
    structure_sequence = PdbQuantizer()(pdb_file=pdb_path)
    structure_sequence_offset = [i + 3 for i in structure_sequence]
    tokenized_res = tokenizer([wt_seq], return_tensors='pt')
    input_ids = tokenized_res['input_ids']
    attention_mask = tokenized_res['attention_mask']
    structure_input_ids = torch.tensor([1, *structure_sequence_offset, 2], dtype=torch.long).unsqueeze(0)

    pred_scores1 = get_logits_from_full_seqs(sequences, prosst_model, input_ids, attention_mask, structure_input_ids, train=False)
    print(spearmanr(y_true, pred_scores1.detach().cpu().numpy()))
    peft_config = LoraConfig(r=8, target_modules=["query", "value"])
    prosst_model_peft = get_peft_model(prosst_model, peft_config)
    optimizer = torch.optim.Adam(prosst_model_peft.parameters(), lr=0.01)
    pred_scores2 = get_logits_from_full_seqs(sequences, prosst_model, input_ids, attention_mask, structure_input_ids, train=True)
    loss_fn = corr_loss
    loss = loss_fn(torch.Tensor(y_true), pred_scores2)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    pred_scores3 = get_logits_from_full_seqs(sequences, prosst_model_peft, input_ids, attention_mask, structure_input_ids, train=False)
    print(spearmanr(y_true, pred_scores3))


def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)


def load_model(model, filename):
    print(f'Loading best model: {os.path.abspath(filename)}...')
    model.load_state_dict(torch.load(filename))


def prosst_train(sequences, scores, loss_fn, model, optimizer, pdb_path, n_epochs=3, device: str | None = None, seed: int | None = None):
    if seed is not None:
        torch.manual_seed(seed)
    if device is None:
        device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    print(f'Training using {device.upper()} device (N_Train={len(torch.flatten(scores))})...')

    structure_sequence = PdbQuantizer()(pdb_file=pdb_path)
    structure_sequence_offset = [i + 3 for i in structure_sequence]
    tokenized_res = tokenizer([wt_seq], return_tensors='pt')
    input_ids = tokenized_res['input_ids']
    attention_mask = tokenized_res['attention_mask']
    structure_input_ids = torch.tensor([1, *structure_sequence_offset, 2], dtype=torch.long).unsqueeze(0)

    y_preds_train = get_logits_from_full_seqs(sequences.flatten(), model, input_ids, attention_mask, structure_input_ids, train=False, verbose=False)
    print(f'Train-->Train UNTRAINED Performance (N={len(scores.flatten())}):', spearmanr(scores.flatten(), y_preds_train))
    pbar_epochs = tqdm(range(1, n_epochs + 1))
    epoch_spearman_1 = 0.0
    os.makedirs('model_saves', exist_ok=True)
    for epoch in pbar_epochs:
        if epoch == 0:
            pbar_epochs.set_description(f'Epoch {epoch}/{n_epochs}')
        model.train()
        y_preds_detached = []
        pbar_batches = tqdm(zip(sequences, scores), total=len(sequences), leave=False)
        for batch, (seqs_b, scores_b) in enumerate(pbar_batches):
            y_preds_b = get_logits_from_full_seqs(seqs_b, model, input_ids, attention_mask, structure_input_ids, train=True, verbose=False)
            y_preds_detached.append(y_preds_b.detach().cpu().numpy().flatten())
            loss = loss_fn(scores_b, y_preds_b)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            pbar_batches.set_description(
                f"Epoch: {epoch}. Loss: {loss.item():>1f}  "
                f"[batch: {batch+1}/{len(sequences)} | "
                f"sequence: {(batch + 1) * len(seqs_b):>5d}/{len(sequences) * len(seqs_b)}]  "
            )
        #print(np.shape(scores.cpu().numpy().flatten()), np.shape(np.array(y_preds_detached).flatten()))
        epoch_spearman_2 = spearmanr(scores.cpu().numpy().flatten(), np.array(y_preds_detached).flatten())[0]
        if epoch_spearman_2 > epoch_spearman_1:
            best_model_epoch = epoch
            best_model_perf = epoch_spearman_2
            best_model = f"model_saves/Epoch{epoch}-Ntrain{len(scores.cpu().numpy().flatten())}-SpearCorr{epoch_spearman_2:.3f}.pt"
            checkpoint(model, best_model)
            epoch_spearman_1 = epoch_spearman_2
            #print(f"Saved current best model as {best_model}")
        loss_total = loss_fn(torch.flatten(scores), torch.flatten(torch.Tensor(np.array(y_preds_detached).flatten())))
        pbar_epochs.set_description(
            f'Epoch {epoch}/{n_epochs} [SpearCorr: {epoch_spearman_2:.3f}, Loss: {loss_total:.3f}] '
            f'(Best epoch: {best_model_epoch}: {best_model_perf:.3f})')
    print(f"Loading best model as {best_model}...")
    load_model(model, best_model)
    y_preds_train = get_logits_from_full_seqs(sequences.flatten(), model, input_ids, attention_mask, structure_input_ids, train=False, verbose=False)
    print(f'Train-->Train Performance (N={len(scores.flatten())}):', spearmanr(scores.flatten(), y_preds_train))


def get_prosst_models():
    prosst_base_model = AutoModelForMaskedLM.from_pretrained("AI4Protein/ProSST-2048", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("AI4Protein/ProSST-2048", trust_remote_code=True)
    peft_config = LoraConfig(r=8, target_modules=["query", "value"])
    prosst_lora_model = get_peft_model(prosst_base_model, peft_config)
    optimizer = torch.optim.Adam(prosst_lora_model.parameters(), lr=0.01)
    return prosst_base_model, prosst_lora_model, tokenizer, optimizer


def get_structure_quantizied(pdb_file):
    structure_sequence = PdbQuantizer()(pdb_file=pdb_file)
    structure_sequence_offset = [i + 3 for i in structure_sequence]
    tokenized_res = tokenizer([wt_seq], return_tensors='pt')
    input_ids = tokenized_res['input_ids']
    attention_mask = tokenized_res['attention_mask']
    structure_input_ids = torch.tensor([1, *structure_sequence_offset, 2], dtype=torch.long).unsqueeze(0)
    return input_ids, attention_mask, structure_input_ids



if __name__ == '__main__':
    import pandas as pd
    import copy
    from sklearn.model_selection import train_test_split 
    # Test on dataset GRB2_HUMAN_Faure_2021: SignificanceResult(statistic=0.6997442598613315, pvalue=0.0)
    wt_seq = "MEAIAKYDFKATADDELSFKRGDILKVLNEECDQNWYKAELNGKDGFIPKNYIEMKPHPWFFGKIPRAKAEEMLSKQRHDGAFLIRESESAPGDFSLSVKFGNDVQHFKVLRDGAGKYFLWVVKFNSLNELVDYHRSTSVSRNQQIFLRDIEQVPQQPTYVQALFDFDPQEDGELGFRRGDFIHVMDNSDPNWWKGACHGQTGMFPRNYVTPVNRNV"
    grb2_folder = os.path.abspath(os.path.join(pypef_path, '..', 'datasets', 'GRB2'))
    pdb_file = os.path.join(grb2_folder, 'GRB2_HUMAN.pdb')
    csv_file = os.path.join(grb2_folder, 'GRB2_HUMAN_Faure_2021.csv')
    df = pd.read_csv(csv_file)
    print(df)
    prosst_base_model, prosst_lora_model, tokenizer, optimizer = get_prosst_models()
    structure_sequence = PdbQuantizer()(pdb_file=pdb_file)
    structure_sequence_offset = [i + 3 for i in structure_sequence]
    tokenized_res = tokenizer([wt_seq], return_tensors='pt')
    input_ids = tokenized_res['input_ids']
    attention_mask = tokenized_res['attention_mask']
    structure_input_ids = torch.tensor([1, *structure_sequence_offset, 2], dtype=torch.long).unsqueeze(0)
    #y_pred = get_logits_from_full_seqs(df['mutated_sequence'], prosst_model, input_ids, attention_mask, structure_input_ids, train=False)
    #print(spearmanr(df['DMS_score'], y_pred.detach().cpu().numpy()))  # SignificanceResult(statistic=np.float64(0.7216670719282277), pvalue=np.float64(0.0))
    
    for train_size in [100, 250, 1000]:
        prosst_model_peft_copy = copy.deepcopy(prosst_lora_model)
        print(f"\n=========\nTRAIN SIZE: {train_size}\n=========")
        y_pred = get_logits_from_full_seqs(df['mutated_sequence'], prosst_model_peft_copy, input_ids, attention_mask, structure_input_ids, train=False)
        print(f'Train-->Test UNTRAINED Performance (N={len(y_pred.flatten())}):',spearmanr(df['DMS_score'], y_pred.detach().cpu().numpy()))
        seqs_train, seqs_test, scores_train, scores_test = train_test_split(
            df['mutated_sequence'].to_numpy(), df['DMS_score'].to_numpy().astype(float), train_size=train_size
        )
        seqs_train = get_batches(seqs_train, batch_size=10, keep_numpy=True, verbose=True)
        scores_train = get_batches(scores_train, batch_size=10, verbose=True)
        prosst_train(seqs_train, scores_train, corr_loss, prosst_model_peft_copy, optimizer, pdb_file, n_epochs=50)
        
        y_pred = get_logits_from_full_seqs(df['mutated_sequence'], prosst_model_peft_copy, input_ids, attention_mask, structure_input_ids, train=False)
        print(f'Train-->Test Performance (N={len(y_pred.flatten())}):', spearmanr(df['DMS_score'], y_pred.detach().cpu().numpy()))
