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


def prosst_tokenize_sequences(sequences, vocab):
    sequences = np.atleast_1d(sequences).tolist()
    x_sequences = []
    for sequence in sequences:
        x_sequence = []
        for aa in sequence:
            x_sequence.append(vocab[aa])
        x_sequences.append(x_sequence)
    return torch.Tensor(x_sequences).to(torch.int)


def get_logits_from_full_seqs(
        xs, 
        model, 
        input_ids, 
        attention_mask, 
        structure_input_ids,
        train: bool = False,
        verbose: bool = True,
        device: str | None = None
):
    if device is None:
        device =  ("cuda" if torch.cuda.is_available()
                   else "mps" if torch.backends.mps.is_available()
                   else "cpu")
    model = model.to(device)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    structure_input_ids = structure_input_ids.to(device)
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

    logits = torch.log_softmax(outputs.logits[:, 1:-1], dim=-1).squeeze()
    for i_s, sequence in enumerate(tqdm(xs, disable=not verbose, desc='Getting ProSST sequence logits')):
        for i_aa, x_aa in enumerate(sequence):
            if i_aa == 0:
                seq_log_probs = logits[i_aa, x_aa].reshape(1)
            else:
                seq_log_probs = torch.cat((seq_log_probs, logits[i_aa, x_aa].reshape(1)), 0)
        if i_s == 0:
            log_probs = torch.sum(torch.Tensor(seq_log_probs)).reshape(1)
        else:
            log_probs = torch.cat((log_probs, torch.sum(torch.Tensor(seq_log_probs)).reshape(1)), 0)
    return log_probs





def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)


def load_model(model, filename):
    print(f'Loading best model: {os.path.abspath(filename)}...')
    model.load_state_dict(torch.load(filename, weights_only=True))


def prosst_train(
        x_sequence_batches, score_batches, loss_fn, model, optimizer,  
        input_ids, attention_mask, structure_input_ids,
        n_epochs=3, device: str | None = None, seed: int | None = None, early_stop: int = 50):
    if seed is not None:
        torch.manual_seed(seed)
    if device is None:
        device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f'ProSST training using {device.upper()} device (N_Train={len(torch.flatten(score_batches))})...')
    x_sequence_batches = x_sequence_batches.to(device)
    score_batches = score_batches.to(device)

    pbar_epochs = tqdm(range(1, n_epochs + 1))
    epoch_spearman_1 = 0.0
    did_not_improve_counter = 0
    best_model_epoch = np.nan
    best_model_perf = np.nan
    os.makedirs('model_saves', exist_ok=True)
    for epoch in pbar_epochs:
        if epoch == 0:
            pbar_epochs.set_description(f'Epoch {epoch}/{n_epochs}')
        model.train()
        y_preds_detached = []
        pbar_batches = tqdm(zip(x_sequence_batches, score_batches), total=len(x_sequence_batches), leave=False)
        for batch, (seqs_b, scores_b) in enumerate(pbar_batches):
            y_preds_b = get_logits_from_full_seqs(seqs_b, model, input_ids, attention_mask, structure_input_ids, train=True, verbose=False)
            y_preds_detached.append(y_preds_b.detach().cpu().numpy().flatten())
            loss = loss_fn(scores_b, y_preds_b)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            pbar_batches.set_description(
                f"Epoch: {epoch}. Loss: {loss.detach():>1f}  "
                f"[batch: {batch+1}/{len(x_sequence_batches)} | "
                f"sequence: {(batch + 1) * len(seqs_b):>5d}/{len(x_sequence_batches) * len(seqs_b)}]  "
            )
        epoch_spearman_2 = spearmanr(score_batches.cpu().numpy().flatten(), np.array(y_preds_detached).flatten())[0]
        if epoch_spearman_2 == np.nan:
            raise SystemError(
                f"No correlation between Y_true and Y_pred could be computed...\n"
                f"Y_true: {score_batches.cpu().numpy().flatten()}, "
                f"Y_pred: {np.array(y_preds_detached)}"
            )
        if epoch_spearman_2 > epoch_spearman_1:
            did_not_improve_counter = 0
            best_model_epoch = epoch
            best_model_perf = epoch_spearman_2
            best_model = f"model_saves/Epoch{epoch}-Ntrain{len(score_batches.cpu().numpy().flatten())}-SpearCorr{epoch_spearman_2:.3f}.pt"
            checkpoint(model, best_model)
            epoch_spearman_1 = epoch_spearman_2
            #print(f"Saved current best model as {best_model}")
        else:
            did_not_improve_counter += 1
            if did_not_improve_counter >= early_stop:
                print(f'\nEarly stop at epoch {epoch}...')
                break
        loss_total = loss_fn(
            torch.flatten(score_batches).to('cpu'), 
            torch.flatten(torch.Tensor(np.array(y_preds_detached).flatten()))
        )
        pbar_epochs.set_description(
            f'Epoch {epoch}/{n_epochs} [SpearCorr: {epoch_spearman_2:.3f}, Loss: {loss_total:.3f}] '
            f'(Best epoch: {best_model_epoch}: {best_model_perf:.3f})')
    try:
        print(f"Loading best model as {best_model}...")
    except UnboundLocalError:
        raise RuntimeError
    load_model(model, best_model)
    y_preds_train = get_logits_from_full_seqs(
        x_sequence_batches.flatten(start_dim=0, end_dim=1), 
        model, input_ids, attention_mask, structure_input_ids, train=False, verbose=False)
    print(f'Train-->Train Performance (N={len(score_batches.cpu().flatten())}):', spearmanr(score_batches.cpu().flatten(), y_preds_train.cpu()))
    return y_preds_train.cpu()


def get_prosst_models():
    prosst_base_model = AutoModelForMaskedLM.from_pretrained("AI4Protein/ProSST-2048", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("AI4Protein/ProSST-2048", trust_remote_code=True)
    peft_config = LoraConfig(r=8, target_modules=["query", "value"])
    prosst_lora_model = get_peft_model(prosst_base_model, peft_config)
    # TODO: Check: LoRa or base model parameters better for ProSST fine-tuning and learning rate?
    optimizer = torch.optim.Adam(prosst_lora_model.parameters(), lr=0.01)  
    return prosst_base_model, prosst_lora_model, tokenizer, optimizer


def get_structure_quantizied(pdb_file, tokenizer, wt_seq):
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
    import matplotlib.pyplot as plt
    # Test on dataset GRB2_HUMAN_Faure_2021: SignificanceResult(statistic=0.6997442598613315, pvalue=0.0)
    wt_seq = "MEAIAKYDFKATADDELSFKRGDILKVLNEECDQNWYKAELNGKDGFIPKNYIEMKPHPWFFGKIPRAKAEEMLSKQRHDGAFLIRESESAPGDFSLSVKFGNDVQHFKVLRDGAGKYFLWVVKFNSLNELVDYHRSTSVSRNQQIFLRDIEQVPQQPTYVQALFDFDPQEDGELGFRRGDFIHVMDNSDPNWWKGACHGQTGMFPRNYVTPVNRNV"
    grb2_folder = os.path.abspath(os.path.join(pypef_path, '..', 'datasets', 'GRB2'))
    pdb_file = os.path.join(grb2_folder, 'GRB2_HUMAN.pdb')
    csv_file = os.path.join(grb2_folder, 'GRB2_HUMAN_Faure_2021.csv')
    df = pd.read_csv(csv_file) #, nrows=120)
    print(df)
    prosst_base_model, prosst_lora_model, tokenizer, optimizer = get_prosst_models()
    vocab = tokenizer.get_vocab()
    structure_sequence = PdbQuantizer()(pdb_file=pdb_file)
    structure_sequence_offset = [i + 3 for i in structure_sequence]
    tokenized_res = tokenizer([wt_seq], return_tensors='pt')
    input_ids = tokenized_res['input_ids']
    attention_mask = tokenized_res['attention_mask']
    structure_input_ids = torch.tensor([1, *structure_sequence_offset, 2], dtype=torch.long).unsqueeze(0)
    #y_pred = get_logits_from_full_seqs(df['mutated_sequence'], prosst_model, input_ids, attention_mask, structure_input_ids, train=False)
    #print(spearmanr(df['DMS_score'], y_pred.detach().cpu().numpy()))  # SignificanceResult(statistic=np.float64(0.7216670719282277), pvalue=np.float64(0.0))
    x_sequences = prosst_tokenize_sequences(df['mutated_sequence'], vocab=vocab)
    for batch_size in [5, 10, 25, 50, 100]:
        train_perfs_unsup, test_perfs_unsup = [], []
        train_perfs, test_perfs = [], []
        for train_size in [200, 1000, 10000]:
            prosst_model_copy = copy.deepcopy(prosst_base_model)
            x_train, x_test, scores_train, scores_test = train_test_split(
                x_sequences, df['DMS_score'].to_numpy().astype(float), train_size=train_size, random_state=42
            )
            print(f"\n=========================\nTRAIN SIZE: {train_size} TEST SIZE: {len(x_test)} -- BATCH SIZE: {batch_size}\n=========================")

            y_pred = get_logits_from_full_seqs(
                x_test, prosst_model_copy, input_ids, attention_mask, structure_input_ids, train=False)
            print(f'Train-->Test UNTRAINED Performance (N={len(y_pred.flatten())}):',spearmanr(scores_test, y_pred.detach().cpu().numpy()))
            test_perfs_unsup.append(spearmanr(scores_test, y_pred.detach().cpu().numpy()))


            y_preds_train_unsup = get_logits_from_full_seqs(
                x_train, prosst_model_copy, input_ids, attention_mask, structure_input_ids, train=False, verbose=False)
            y_preds_train_unsup = y_preds_train_unsup.cpu().numpy()
            print(f'Train-->Train UNTRAINED Performance (N={len(y_preds_train_unsup)}):', spearmanr(scores_train, y_preds_train_unsup))
            train_perfs_unsup.append(spearmanr(scores_train, y_preds_train_unsup)[0])

            # TRAINING
            x_train_b = get_batches(x_train, dtype=int, batch_size=batch_size, verbose=True)
            scores_train_b = get_batches(scores_train, dtype=float, batch_size=batch_size, verbose=True)
            y_preds_train = prosst_train(x_train_b, scores_train_b, corr_loss, prosst_model_copy, optimizer, pdb_file, n_epochs=500)
            print(f'Train-->Train Performance (N={len(y_preds_train)}):', spearmanr(scores_train, y_preds_train))
            train_perfs.append(spearmanr(scores_train, y_preds_train)[0])

            y_pred = get_logits_from_full_seqs(
                x_test, prosst_model_copy, input_ids, attention_mask, structure_input_ids, train=False)
            print(f'Train-->Test Performance (N={len(y_pred.flatten())}):', spearmanr(scores_test, y_pred.detach().cpu().numpy()))
            test_perfs.append(spearmanr(scores_test, y_pred.detach().cpu().numpy())[0])
        for k in [train_perfs_unsup, train_perfs, test_perfs_unsup, test_perfs]:
            plt.plot(range(len(k)), k, label=f'Batch size: {batch_size}')
    plt.xticks(range(len(k)), [100, 200, 1000, 10000])
    plt.legend()
    plt.savefig('1.png')
