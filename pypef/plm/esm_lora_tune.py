# PyPEF - Pythonic Protein Engineering Framework
# https://github.com/niklases/PyPEF

# Using (training, testing/infering) ESM model(s) (e.g. ESM1v) published under 
# MIT License
# Copyright (c) Meta Platforms, Inc. and affiliates.
# https://github.com/facebookresearch/esm
# ESM1v model publication:
# Joshua Meier, Roshan Rao, Robert Verkuil, Jason Liu, Tom Sercu, Alexander Rives
# Language models enable zero-shot prediction of the effects of mutations on protein function
# bioRxiv 2021.07.09.450648; doi: https://doi.org/10.1101/2021.07.09.450648 

# Inspired by ConFit
# https://github.com/luo-group/ConFit


from __future__ import annotations

import logging

from pypef.plm.prosst_lora_tune import get_logits_from_full_seqs
logger = logging.getLogger('pypef.llm.esm_lora_tune')

import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm


from peft import LoraConfig, get_peft_model
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

from pypef.utils.helpers import get_device
from pypef.plm.utils import corr_loss, get_batches, load_model_and_tokenizer


def get_esm_models(model='facebook/esm1v_t33_650M_UR90S_3'):
    base_model, tokenizer = load_model_and_tokenizer(
        model
        # Just sticking to AutoModelForMaskedLM and AutoTokenizer 
        # instead to EsmForMaskedLM and EsmTokenizer
    )  
    peft_config = LoraConfig(r=8, target_modules=["query", "value"])
    lora_model = get_peft_model(base_model, peft_config)
    optimizer = torch.optim.Adam(lora_model.parameters(), lr=0.01)
    return base_model, lora_model, tokenizer, optimizer


def tokenize_sequences(sequences, tokenizer, max_length, verbose=True):
    tokenized_sequences = []
    for seq in tqdm(sequences, desc='Tokenizing sequences for ESM modeling', disable=not verbose):
        encoded_sequence, attention_mask = tokenizer(
            seq, 
            padding='max_length', 
            truncation=True,  # False for not uniform length distribution (truncation) 
            max_length=max_length
        ).values()
        tokenized_sequences.append(encoded_sequence)
    return tokenized_sequences, attention_mask


def get_y_pred_scores(encoded_sequences, attention_masks, 
                      model, device: str | None = None):
    if device is None:
        device = get_device()
    model = model.to(device)
    out = model(encoded_sequences.to(device), attention_masks.to(device), 
                output_hidden_states=True)
    logits = out.logits
    token_probs = torch.log_softmax(logits, dim=-1)
    for i_s, sequence in enumerate(encoded_sequences):
        for i_aa, aa in enumerate(sequence):
            # alternative: use Tensor.index_select() function
            if i_aa == 0:
                seq_log_probs = token_probs[i_s, i_aa, aa].reshape(1)
            else:
                seq_log_probs = torch.cat(
                    (seq_log_probs, token_probs[i_s, i_aa, aa].reshape(1)), 0)
        if i_s == 0:
            log_probs = torch.sum(torch.Tensor(seq_log_probs)).reshape(1)
        else:
            log_probs = torch.cat(
                (log_probs, torch.sum(torch.Tensor(seq_log_probs)).reshape(1)), 0)
    return log_probs


def esm_test(xs, attention_mask, scores, loss_fn, model, 
             device: str | None = None, verbose: bool = True):
    if device is None:
        device = get_device()
    attention_masks = torch.Tensor(np.full(
        shape=np.shape(xs), fill_value=attention_mask)).to(torch.int64)
    logger.info(f'Infering ESM model for testing using {device.upper()} device...')
    model = model.to(device)
    xs, attention_masks, scores = (
        torch.Tensor(xs).to(device), attention_masks.to(device), 
        torch.Tensor(scores).to(torch.float).to(device)
    )
    pbar_epochs = tqdm(zip(xs, attention_masks, scores), total=len(xs), disable=not verbose)
    for i ,(xs_b, attns_b, scores_b) in enumerate(pbar_epochs):
        xs_b, attns_b = xs_b.to(torch.int64), attns_b.to(torch.int64)
        with torch.no_grad():
            y_preds = get_y_pred_scores(xs_b, attns_b, model, device)
            if i == 0:
                y_preds_total = y_preds
                scores_total = scores_b
            else:
                y_preds_total = torch.cat((y_preds_total, y_preds))
                scores_total = torch.cat((scores_total, scores_b))
        batch_loss = loss_fn(scores_b, y_preds)
        total_loss = loss_fn(torch.flatten(scores_total), torch.flatten(y_preds_total))
        batch_scorr = spearmanr(scores_b.cpu(), y_preds.cpu())[0]
        total_scorr = spearmanr(scores_total.cpu(), y_preds_total.cpu())[0]
        pbar_epochs.set_description(
            f"Testing: Batch {i + 1}/{len(xs)} | Batch loss: {batch_loss:.4f} (SpearCorr: "
            f"{batch_scorr:.4f})| Total loss: {total_loss:.4f} (SpearCorr: {total_scorr:.4f})")
    logger.info(f"Test performance: Loss: {total_loss:.4f}, SpearCorr: {total_scorr:.4f} "
                f"({device.upper()})")
    return torch.flatten(scores).detach().cpu(), torch.flatten(y_preds_total).detach().cpu()


def esm_infer(xs, attention_mask, model, device: str | None = None, verbose=False):
    if device is None:
        device = get_device()
    attention_masks = torch.Tensor(np.full(
        shape=np.shape(xs), fill_value=attention_mask)).to(torch.int64)
    if verbose:
        logger.info(f'Infering ESM model for predictions using {device.upper()} device...')
    for i , (xs_b, am_b) in enumerate(tqdm(
        zip(xs, attention_masks), total=len(xs), 
        desc=f"ESM inference - processing sequences ({device.upper()})",
        disable=not verbose
    )):
        xs_b = xs_b.to(torch.int64)
        with torch.no_grad():
            y_preds = get_y_pred_scores(xs_b, am_b, model, device)
            if i == 0:
                y_preds_total = y_preds
            else:
                y_preds_total = torch.cat((y_preds_total, y_preds))
    return torch.flatten(y_preds_total)


def esm_unmasked_wt_score(
        tokenized_sequences, 
        attention_mask, 
        wt_input_ids,
        model, 
        train: bool = False,
        device=None, 
        **kwargs
    ):
    if device is None:
        device = get_device()
    if wt_input_ids.dim() == 1:
        wt_input_ids = wt_input_ids.unsqueeze(0)
    structure_input_ids = kwargs.get("structure_input_ids", None)
    attention_masks = torch.Tensor(np.full(
        shape=np.shape(wt_input_ids), fill_value=attention_mask)).to(torch.int64)
    if train:
        if structure_input_ids is not None:
            outputs = model(
                input_ids=wt_input_ids.to(device),
                attention_mask=attention_masks.to(device),
                ss_input_ids=structure_input_ids.to(device)
            )
        else:
            outputs = model(
                wt_input_ids.to(device), 
                attention_masks.to(device), 
                output_hidden_states=False
            )
    else:
        with torch.no_grad():
            if structure_input_ids is not None:
                outputs = model(
                        input_ids=wt_input_ids.to(device),
                        attention_mask=attention_masks.to(device),
                        ss_input_ids=structure_input_ids.to(device)
                )
            else:
                outputs = model(
                    wt_input_ids.to(device), 
                    attention_masks.to(device), 
                    output_hidden_states=False
                )

    logits = outputs.logits
    logits = logits.squeeze(0)   # remove batch dim
    #print('logits.shape:', logits.shape)
    # Better make sure that special tokens are always removed / masked 
    # and only pure amino acid sequence tokens are present / unmasked
    #logits = logits[1:-1]        # drop CLS/EOS
    token_probs = torch.log_softmax(logits, dim=-1)
    assert len(tokenized_sequences[0]) == token_probs.shape[0], f"{len(tokenized_sequences[0])} != {token_probs.shape[0]}"
    #print('token_probs.shape:', token_probs.shape)

    for i_s, tokenized_seq in enumerate(tokenized_sequences):
        for i_aa, aa in enumerate(tokenized_seq):
            # alternative: use Tensor.index_select() function
            if i_aa == 0:
                seq_log_probs = token_probs[i_aa, aa].reshape(1)
            else:
                seq_log_probs = torch.cat(
                    (seq_log_probs, token_probs[i_aa, aa].reshape(1)), 0)
        if i_s == 0:
            log_probs = torch.sum(torch.Tensor(seq_log_probs)).reshape(1)
        else:
            log_probs = torch.cat(
                (log_probs, torch.sum(torch.Tensor(seq_log_probs)).reshape(1)), 0)
    return log_probs


def esm_mutation_only_mutation_masked_pll(
    tokenized_sequences: torch.Tensor,        # (L,)
    wt_input_ids: torch.Tensor,     # (L,)
    attention_mask: torch.Tensor,   # (L,)
    model,
    mask_token_id: int,
    train: bool = False,
    device: str | None = None,
    verbose: bool = False,
    **kwargs
):
    """
    Correct mutation-only pseudo-log-likelihood for ONE sequence.
    """
    model.eval()

    tokenized_sequences = tokenized_sequences.to(device)
    wt_input_ids = wt_input_ids.to(device)
    attention_mask = attention_mask.to(device)
    plls = torch.empty(len(tokenized_sequences), device=device)
    for i, tokenized_seq in enumerate(tokenized_sequences):
        pll = 0.0

        # Identify mutated positions (exclude padding, CLS, EOS)
        diff = (tokenized_seq != wt_input_ids) & (attention_mask == 1)
        diff[0] = False
        diff[-1] = False

        mutated_positions = diff.nonzero(as_tuple=False).flatten()
        # Mutated positions: [int(m) - 1 for m in mutated_positions.cpu()]  # Remove CLS token position

        for pos in tqdm(
            mutated_positions,
            desc="Masked PLL (single sequence)",
            disable=not verbose
        ):
            masked_input_ids = tokenized_seq.clone()
            masked_input_ids[pos] = mask_token_id
            if train:
                outputs = model(
                    input_ids=masked_input_ids.unsqueeze(0),
                    attention_mask=attention_mask.unsqueeze(0),
                )
            else:
                with torch.no_grad():
                    outputs = model(
                        input_ids=masked_input_ids.unsqueeze(0),
                        attention_mask=attention_mask.unsqueeze(0),
                    )
            logits = outputs.logits  # (1, L, V)

            log_probs = F.log_softmax(logits[0, pos], dim=-1)
            true_token = tokenized_seq[pos]

            pll += log_probs[true_token].item()
        
        plls[i] = pll

    return plls


def esm_mutation_all_pos_masked_pll(
    tokenized_sequences: torch.Tensor,        # (L,)
    wt_input_ids: torch.Tensor,     # (L,)
    attention_mask: torch.Tensor,   # (L,)
    model,
    mask_token_id: int,
    train: bool = False,
    device: str | None = None,
    verbose: bool = False,
):
    """
    Correct mutation-only pseudo-log-likelihood for ONE sequence.
    """
    model.eval()

    tokenized_sequences = tokenized_sequences.to(device)
    wt_input_ids = wt_input_ids.to(device)
    attention_mask = attention_mask.to(device)
    plls = torch.empty(len(tokenized_sequences), device=device)
    for i, tokenized_seq in enumerate(tokenized_sequences):
        L = tokenized_seq.shape[0]
        pll = 0.0

        # Positions to score: all real tokens except CLS/EOS
        positions = (attention_mask == 1).nonzero(as_tuple=False).flatten()
        positions = positions[(positions != 0) & (positions != L - 1)]

        for pos in tqdm(
            positions,
            desc="Masked PLL (single sequence)",
            disable=not verbose
        ):
            masked_input_ids = tokenized_seq.clone()
            masked_input_ids[pos] = mask_token_id

            if train:
                outputs = model(
                    input_ids=masked_input_ids.unsqueeze(0),
                    attention_mask=attention_mask.unsqueeze(0),
                )
            else:
                with torch.no_grad():
                    outputs = model(
                        input_ids=masked_input_ids.unsqueeze(0),
                        attention_mask=attention_mask.unsqueeze(0),
                    )
            logits = outputs.logits  # (1, L, V)

            log_probs = F.log_softmax(logits[0, pos], dim=-1)
            true_token = tokenized_seq[pos]

            pll += log_probs[true_token].item()

        plls[i] = pll

    return plls


def esm_infer_pll(
    xs,
    wt_input_ids,
    attention_mask,
    model,
    mask_token_id,
    inference_type='unmasked',
    batch_size=5,
    train=False,
    device=None,
    verbose=False,
):
    if device is None:
        device = get_device()

    model = model.to(device)

    if not isinstance(xs, torch.Tensor):
        xs = torch.tensor(xs, dtype=torch.long)

    if not isinstance(attention_mask, torch.Tensor):
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
    wt_structure_input_ids = None
    if inference_type == 'mutation-masking':
        inference_function = esm_mutation_only_mutation_masked_pll
    elif inference_type in ['full-masking', 'all-pos-masking']:
        inference_function = esm_mutation_all_pos_masked_pll
    elif inference_type in ['unmasked', 'wt-marginals']:
        inference_function = esm_unmasked_wt_score
    elif inference_type == 'prosst':
        wt_input_ids, wt_structure_input_ids = wt_input_ids
        inference_function = esm_unmasked_wt_score
    else:
        raise SystemError("Choose between 'mutation_masking', 'unmasked', and 'full_masking'")

    scores = []

    xs_b = get_batches(xs, dtype=int, batch_size=batch_size, keep_remaining=True, verbose=True)
    desc = f"ESM inference: {inference_type} batch (size={batch_size}) processing ({device.upper()})'"

    pbar = tqdm(
        range(len(xs_b)),
        desc=desc,
        disable=not verbose
    )

    for i in pbar:
        pll = inference_function(
            tokenized_sequences=torch.tensor(xs_b[i]),
            wt_input_ids=wt_input_ids,
            structure_input_ids=wt_structure_input_ids,
            attention_mask=attention_mask,
            model=model,
            mask_token_id=mask_token_id,
            train=train,
            device=device,
            verbose=False
        )
        scores.append(pll)
    return torch.cat(scores)


def esm_train(
        xs, attention_mask, scores, loss_fn, model, optimizer, n_epochs=3, 
        device: str | None = None, seed: int | None = None, 
        n_batch_grad_accumulations: int = 1, verbose: bool = True,
        progress_cb=None, abort_cb=None
):
    if seed is not None:
        torch.manual_seed(seed)
    if device is None:
        device = get_device()
    print(f'Training ESM model using {device.upper()} device '
          f'(N_Train={len(torch.flatten(scores))})...')
    model = model.to(device)
    attention_masks = torch.Tensor(np.full(
        shape=np.shape(xs), fill_value=attention_mask)).to(torch.int64)
    xs, attention_masks, scores = xs.to(device), attention_masks.to(device), scores.to(device) 
    pbar_epochs = tqdm(range(1, n_epochs + 1), disable=not verbose)
    loss = np.nan
    for epoch in pbar_epochs:
        try:
            pbar_epochs.set_description(f'Epoch: {epoch}/{n_epochs}. Loss: {loss.detach():>1f}')
        except AttributeError:
            pbar_epochs.set_description(f'Epoch: {epoch}/{n_epochs}')
        model.train()
        pbar_batches = tqdm(
            zip(xs, attention_masks, scores), 
            total=len(xs), leave=False, disable=not verbose
        )
        for batch, (xs_b, attns_b, scores_b) in enumerate(pbar_batches):
            if abort_cb and abort_cb():
                return
            xs_b, attns_b = xs_b.to(torch.int64), attns_b.to(torch.int64)
            y_preds_b = get_y_pred_scores(xs_b, attns_b, model, device=device)
            loss = loss_fn(scores_b, y_preds_b) / n_batch_grad_accumulations
            if progress_cb:
                progress_cb(epoch - 1, batch + 1, len(pbar_epochs), len(pbar_batches), loss)
            loss.backward()
            if (batch + 1) % n_batch_grad_accumulations == 0 or (batch + 1) == len(pbar_batches):
                optimizer.step()
                optimizer.zero_grad()
            pbar_batches.set_description(
                f"Epoch: {epoch}. Loss: {loss.detach():>1f}  "
                f"[batch: {batch+1}/{len(xs)} | sequence: "
                f"{(batch + 1) * len(xs_b):>5d}/{len(xs) * len(xs_b)}] ({device.upper()})"
            )
    if progress_cb:
        progress_cb(epoch, batch + 1, len(pbar_epochs), len(pbar_batches), loss)
    y_preds_b = y_preds_b.detach()
    model.train(False)


def esm_setup(sequences, device: str | None = None, verbose: bool = True):
    esm_base_model, esm_lora_model, esm_tokenizer, esm_optimizer = get_esm_models()
    esm_base_model = esm_base_model.to(device)
    x_esm, esm_attention_mask = tokenize_sequences(
        sequences, esm_tokenizer, max_length=len(sequences[0]), verbose=verbose)
    llm_dict_esm = {
        'esm1v': {
            'llm_base_model': esm_base_model,
            'llm_model': esm_lora_model,
            'llm_optimizer': esm_optimizer,
            'llm_train_function': esm_train,
            'llm_inference_function': esm_infer,
            'llm_loss_function': corr_loss,
            'x_llm' : x_esm,
            'llm_attention_mask':  esm_attention_mask,
            'llm_tokenizer': esm_tokenizer
        }
    }
    return llm_dict_esm
