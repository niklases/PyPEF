# Niklas Siedhoff
# PyPEF - Pythonic Protein Engineering Framework

# Using (training, testing/infering) ESM model(s) (e.g. ESM1v) published under 
# MIT License
# Copyright (c) Meta Platforms, Inc. and affiliates.
# https://github.com/facebookresearch/esm

# Inspired by ConFit
# https://github.com/luo-group/ConFit

from __future__ import annotations

import torch
import numpy as np
from tqdm import tqdm
import logging

from peft import LoraConfig, get_peft_model
from transformers import EsmForMaskedLM, EsmTokenizer

logger = logging.getLogger('pypef.llm.esm_lora_tune')


def get_vram(verbose: bool = True):
    free = torch.cuda.mem_get_info()[0] / 1024 ** 3
    total = torch.cuda.mem_get_info()[1] / 1024 ** 3
    total_cubes = 24
    free_cubes = int(total_cubes * free / total)
    if verbose:
        logger.info(f'VRAM: {total - free:.2f}/{total:.2f}GB\t VRAM:[' + (
            total_cubes - free_cubes) * '▮' + free_cubes * '▯' + ']')
    return free, total


def get_esm_models():
    base_model = EsmForMaskedLM.from_pretrained(f'facebook/esm1v_t33_650M_UR90S_3')
    tokenizer = EsmTokenizer.from_pretrained(f'facebook/esm1v_t33_650M_UR90S_3')
    peft_config = LoraConfig(r=8, target_modules=["query", "value"])
    lora_model = get_peft_model(base_model, peft_config)
    optimizer = torch.optim.Adam(lora_model.parameters(), lr=0.01)
    return base_model, lora_model, tokenizer, optimizer


def get_encoded_seqs(sequences, tokenizer, max_length=104):
    encoded_sequences, attention_masks = tokenizer(
        sequences, 
        padding='max_length', 
        truncation=True, 
        max_length=max_length
    ).values()
    return encoded_sequences, attention_masks


def get_y_pred_scores(encoded_sequences, attention_masks, model, device: str | None = None):
    if device is None:
        device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    #logger.info(f'Getting scores (y_pred) using {device.upper()} device...')
    model = model.to(device)
    out = model(encoded_sequences.to(device), attention_masks.to(device), output_hidden_states=True)
    logits = out.logits
    token_probs = torch.log_softmax(logits, dim=-1)
    for i_s, sequence in enumerate(encoded_sequences):
        for i_aa, aa in enumerate(sequence):
            # alternative: use Tensor.index_select() function
            #logger.info('Target AA:', i_aa, aa, proteinseq_toks['toks'][aa], token_probs[i_s, i_aa, aa])
            if i_aa == 0:
                seq_log_probs = token_probs[i_s, i_aa, aa].reshape(1)
            else:
                seq_log_probs = torch.cat((seq_log_probs, token_probs[i_s, i_aa, aa].reshape(1)), 0)
        if i_s == 0:
            log_probs = torch.sum(torch.Tensor(seq_log_probs)).reshape(1)
        else:
            log_probs = torch.cat((log_probs, torch.sum(torch.Tensor(seq_log_probs)).reshape(1)), 0)
    return log_probs


def corr_loss(y_true: torch.Tensor, y_pred: torch.Tensor):
    res_true = y_true - torch.mean(y_true)
    res_pred = y_pred - torch.mean(y_pred)
    cov = torch.mean(res_true * res_pred)
    var_true = torch.mean(res_true**2)
    var_pred = torch.mean(res_pred**2)
    sigma_true = torch.sqrt(var_true)
    sigma_pred = torch.sqrt(var_pred)
    return - cov / (sigma_true * sigma_pred)


def get_batches(a, batch_size=5, verbose: bool = False):
    a = np.array(a)
    orig_shape = np.shape(a)
    remaining = len(a) % batch_size
    if remaining != 0:
        a = a[:-remaining]
    if len(orig_shape) == 2:
        a = a.reshape(np.shape(a)[0] // batch_size, batch_size, np.shape(a)[1])
    else:  # elif len(orig_shape) == 1:
        a = a.reshape(np.shape(a)[0] // batch_size, batch_size)
    new_shape = np.shape(a)
    if verbose:
        logger.info(f'{orig_shape} -> {new_shape}  (dropped {remaining})')
    return torch.Tensor(a)
    

def test(xs, attns, scores, loss_fn, model, device: str | None = None):
    if device is None:
        device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f'Infering model for testing using {device.upper()} device...')
    model = model.to(device)
    xs, attns, scores = xs.to(device), attns.to(device), scores.to(device) 
    for i ,(xs_b, attns_b) in enumerate(tqdm(zip(xs, attns), total=len(xs))):
        xs_b, attns_b = xs_b.to(torch.int64), attns_b.to(torch.int64)
        with torch.no_grad():
            y_preds = get_y_pred_scores(xs_b, attns_b, model, device)
            if i == 0:
                y_preds_total = y_preds
            else:
                y_preds_total = torch.cat((y_preds_total, y_preds))
    loss = loss_fn(
        torch.flatten(scores), 
        torch.flatten(y_preds_total)
    )
    logger.info(f'TESTING LOSS: {float(loss.cpu()):.3f}')
    return torch.flatten(scores).detach().cpu(), torch.flatten(y_preds_total).detach().cpu()


def infer(xs, attns, model, desc: None | str = None, device: str | None = None):
    if device is None:
        device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f'Infering model for predictions using {device.upper()} device...')
    for i ,(xs_b, attns_b) in enumerate(tqdm(zip(xs, attns), total=len(xs), desc=desc)):
        xs_b, attns_b = xs_b.to(torch.int64), attns_b.to(torch.int64)
        with torch.no_grad():
            y_preds = get_y_pred_scores(xs_b, attns_b, model, device)
            if i == 0:
                y_preds_total = y_preds
            else:
                y_preds_total = torch.cat((y_preds_total, y_preds))
    return torch.flatten(y_preds_total)


def train(xs, attns, scores, loss_fn, model, optimizer, n_epochs=3, device: str | None = None):
    if device is None:
        device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f'Training using {device.upper()} device (N_Train={len(scores)})...')
    model = model.to(device)
    xs, attns, scores = xs.to(device), attns.to(device), scores.to(device) 
    pbar_epochs = tqdm(range(1, n_epochs + 1))
    for epoch in pbar_epochs:
        #logger.info(f'VRAM CHECK --- EPOCH {epoch} --- CHECK VRAM')
        #get_vram()
        pbar_epochs.set_description(f'EPOCH {epoch}/{n_epochs}')
        model.train()
        pbar_batches = tqdm(zip(xs, attns, scores), total=len(xs), leave=False)
        for batch, (xs_b, attns_b, scores_b) in enumerate(pbar_batches):
            xs_b, attns_b = xs_b.to(torch.int64), attns_b.to(torch.int64)
            #logger.info(xs_b.size(), attns_b.size(), scores_b.size())
            y_preds = get_y_pred_scores(xs_b, attns_b, model, device=device)
            #scores_b = scores_b.to(device)
            loss = loss_fn(scores_b, y_preds)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            #saved_params = []
            #for i, (name, param) in enumerate(model.named_parameters()):  # 33 layers (0-32)
            #    if 'lora' in name:
            #        saved_params.append(torch.sum(param).clone())
            pbar_batches.set_description(
                f"EPOCH: {epoch}. Loss: {loss.item():>1f}  "
                f"[batch: {batch+1}/{len(xs)} | "
                f"sequence: {(batch + 1) * len(xs_b):>5d}/{len(xs) * len(xs_b)}]  "
                #f"(LoRA weight sum:{sum(saved_params):.3f})"
            )
    y_preds.detach()
    model.train(False)
