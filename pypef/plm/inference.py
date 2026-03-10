# PyPEF - Pythonic Protein Engineering Framework
# https://github.com/niklases/PyPEF

# Some helper functions for infernece of different models 
# based on simple/wrapping functions

import os
import inspect
import numpy as np
from scipy.stats import spearmanr
import torch
import torch.nn.functional as F
from tqdm import tqdm
from Bio import SeqIO

from pypef.plm.prosst_lora_tune import get_prosst_models, get_structure_quantizied
from pypef.utils.helpers import get_device
from pypef.plm.utils import pearson_loss, spearman_loss, get_batches
from pypef.plm.esm_lora_tune import get_esm_models


import logging
logger = logging.getLogger('pypef.llm.inference')


def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)


def load_model(model, filename):
    logger.info(f'Loading best model: {os.path.abspath(filename)}...')
    model.load_state_dict(torch.load(filename, weights_only=True))


def tokenize_sequences(sequences, tokenizer, max_length=None, verbose=True):
    if max_length is None:
        logger.info(f"Setting max. tokenized sequence length to {len(sequences[0]) + 2}...")
        max_length = len(sequences[0]) + 2
    tokenized_sequences = []
    for seq in tqdm(sequences, desc='Tokenizing sequences', disable=not verbose):
        encoded_sequence, attention_mask = tokenizer(
            seq, 
            padding='max_length', 
            truncation=True,  # False for not uniform length distribution (truncation) 
            max_length=max_length
        ).values()
        tokenized_sequences.append(encoded_sequence)
    return tokenized_sequences, attention_mask


def sequence_log_likelihood(
        tokenized_sequences, 
        attention_mask, 
        wt_input_ids,
        model, 
        scoring_mode: str = "wt-marginal",   # "wt-marginal" | "full-sequence"
        train: bool = False,
        cut_special_tokens: bool = True,  # assumption: cut first and last token
        device=None,
        verbose: bool = False,
        **model_kwargs
    ):
    """
    Unified scoring function.

    scoring_mode:
        - "wt-marginal": forward pass on WT only (fast, approximate)
        - "full-sequence": forward pass per sequence (exact PLL)

    Returns:
        torch.Tensor of shape [num_sequences]
    """
    extract_emb = model_kwargs.pop("extract_emb", False)
    assert scoring_mode in ["wt-marginal", "full-sequence"]
    if device is None:
        device = get_device()
    if wt_input_ids.dim() == 1:
        wt_input_ids = wt_input_ids.unsqueeze(0)
    wt_input_ids = wt_input_ids.to(device)
    log_probs = []
    #structure_input_ids = model_kwargs.get("structure_input_ids", None)
    if scoring_mode == "wt-marginal":
        attention_masks = torch.Tensor(np.full(
            shape=np.shape(wt_input_ids), fill_value=attention_mask)).to(torch.int64).to(device)
        try:
            if train:
                outputs = model(
                    input_ids=wt_input_ids,
                    attention_mask=attention_masks,
                    output_hidden_states=False,
                    return_dict=True,
                    **model_kwargs
                )

            else:
                with torch.no_grad():
                    outputs = model(
                        input_ids=wt_input_ids,
                        attention_mask=attention_masks,
                        output_hidden_states=False,
                        return_dict=True,
                        **model_kwargs
                    )
        except TypeError as e:
            logger.info(f"Did not find model input keyword arguments (kwargs: "
                  f"{model_kwargs.keys()}). Available kawrgs identified from "
                  f"model.forward function inspect:\n"
                  f"{inspect.signature(model.forward)}\nOriginal error:")
            raise e


        logits = outputs.logits
        logits = logits.squeeze(0)   # remove batch dim
        # Better make sure that special tokens are always removed / masked 
        # and only pure amino acid sequence tokens are present / unmasked
        tokenized_seq_len = tokenized_sequences.shape[1]
        if cut_special_tokens:
            logits = logits[1:-1]        # drop CLS/EOS
            tokenized_seq_len -= 2
        token_probs = torch.log_softmax(logits, dim=-1)
        assert tokenized_seq_len == token_probs.shape[0], (
            f"{tokenized_seq_len} != {token_probs.shape[0]}")

        for tokenized_seq in tokenized_sequences:
            if cut_special_tokens:
                tokenized_seq = tokenized_seq[1:-1]

            seq_lp = token_probs[
                torch.arange(tokenized_seq.shape[0], device=tokenized_seq.device),
                tokenized_seq
            ].sum(dtype=torch.float64)

            log_probs.append(seq_lp)


    elif scoring_mode == "full-sequence":
        if extract_emb:
            embeddings = []
        for tokenized_seq in tokenized_sequences:
            if tokenized_seq.dim() == 1:
                tokenized_seq = tokenized_seq.unsqueeze(0)

            attention_masks = torch.Tensor(np.full(
                shape=tokenized_seq.shape,
                fill_value=attention_mask)
            ).to(torch.int64).to(device)

            try:
                if train:
                    outputs = model(
                        input_ids=tokenized_seq,
                        attention_mask=attention_masks,
                        return_dict=True,
                        **model_kwargs
                    )
                else:
                    with torch.no_grad():
                        outputs = model(
                            input_ids=tokenized_seq,
                            attention_mask=attention_masks,
                            return_dict=True,
                            output_hidden_states=extract_emb,
                            **model_kwargs
                        )
                    if extract_emb:
                        token_embeddings = outputs.hidden_states[-1]  # (1, L+2, D)
                        # Mean pool over residues (exclude CLS/EOS)
                        seq_embedding = token_embeddings[0, 1:-1].mean(dim=0)
                        embeddings.append(seq_embedding)
                        continue

            except TypeError as e:
                logger.info(f"Did not find model input keyword arguments (kwargs: "
                    f"{model_kwargs.keys()}). Available kawrgs identified from "
                    f"model.forward function inspect:\n"
                    f"{inspect.signature(model.forward)}\nOriginal error:"
                )
                raise e

            logits = outputs.logits.squeeze(0)

            if cut_special_tokens:
                logits = logits[1:-1]
                target_tokens = tokenized_seq.squeeze(0)[1:-1]
            else:
                target_tokens = tokenized_seq.squeeze(0)

            token_log_probs = torch.log_softmax(logits, dim=-1)

            seq_lp = token_log_probs[
                torch.arange(target_tokens.shape[0], device=device),
                target_tokens
            ].sum(dtype=torch.float64)

            log_probs.append(seq_lp)
    if extract_emb:
        return torch.stack(embeddings)
    return torch.stack(log_probs)


def mutation_only_mutation_masked_pll(
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
    Correct mutation-only pseudo-log-likelihood for sequences.
    """
    tokenized_sequences = tokenized_sequences.to(device)
    structure_input_ids = kwargs.get("ss_input_ids", None)
    if structure_input_ids is not None:
        assert structure_input_ids.shape[1] == tokenized_sequences.shape[1], (
            f"{structure_input_ids.shape[1]} != {tokenized_sequences.shape[1]}")
        structure_input_ids = structure_input_ids.to(device)
    if wt_input_ids.dim() == 2 and wt_input_ids.shape[0] == 1:
        wt_input_ids = wt_input_ids.squeeze(0)
    wt_input_ids = wt_input_ids.to(device)
    if attention_mask.dim() == 2 and attention_mask.shape[0] == 1:
        attention_mask = attention_mask.squeeze(0)
    attention_mask = attention_mask.to(device)
    plls = torch.empty(len(tokenized_sequences), device=device)
    for i, tokenized_seq in enumerate(tokenized_sequences):
        assert tokenized_seq.dim() == 1
        assert wt_input_ids.dim() == 1
        assert attention_mask.dim() == 1
        assert tokenized_seq.shape == wt_input_ids.shape == attention_mask.shape
        pll = torch.tensor(0.0, device=device)

        # Identify mutated positions (exclude padding, CLS, EOS)
        diff = (tokenized_seq != wt_input_ids) & (attention_mask == 1)
        diff[0] = False
        diff[-1] = False

        mutated_positions = diff.nonzero(as_tuple=False).flatten()
        # n_mutations = (tokenized_seq != wt_input_ids).sum().item()
        # Mutated positions: [int(m) - 1 for m in mutated_positions.cpu()]  # Remove CLS token position

        for pos in tqdm(
            mutated_positions,
            desc="Masked PLL (single sequence)",
            disable=not verbose
        ):
            masked_input_ids = tokenized_seq.clone()
            masked_input_ids[pos] = mask_token_id
            if structure_input_ids is not None:
                masked_ss_input_ids = structure_input_ids.clone()
                masked_ss_input_ids[0, pos] = mask_token_id

            if train:
                if structure_input_ids is not None:
                    outputs = model(
                        input_ids=masked_input_ids.unsqueeze(0),
                        attention_mask=attention_mask.unsqueeze(0),
                        ss_input_ids=masked_ss_input_ids  # Check
                    )
                else:
                    outputs = model(
                        input_ids=masked_input_ids.unsqueeze(0),
                        attention_mask=attention_mask.unsqueeze(0),
                        output_hidden_states=False
                    )
            else:
                with torch.no_grad():
                    if structure_input_ids is not None:
                        outputs = model(
                            input_ids=masked_input_ids.unsqueeze(0),
                            attention_mask=attention_mask.unsqueeze(0),
                            ss_input_ids=masked_ss_input_ids  # Check
                        )
                    else:
                        outputs = model(
                            input_ids=masked_input_ids.unsqueeze(0),
                            attention_mask=attention_mask.unsqueeze(0),
                            output_hidden_states=False
                        )
            logits = outputs.logits  # (1, L, V)
            log_probs = F.log_softmax(logits[0, pos], dim=-1)
            true_token = tokenized_seq[pos]
            pll = pll + log_probs[true_token]
        
        plls[i] = pll

    return plls


def mutation_all_pos_masked_pll(
    tokenized_sequences: torch.Tensor,    # (L,)
    attention_mask: torch.Tensor,         # (L,)
    model,
    mask_token_id: int,
    train: bool = False,
    device: str | None = None,
    verbose: bool = False,
    **kwargs
):
    """
    Correct mutation-only pseudo-log-likelihood for sequences.
    """
    structure_input_ids = kwargs.get("ss_input_ids", None)
    if structure_input_ids is not None:
        assert structure_input_ids.shape[1] == tokenized_sequences.shape[1], (
            f"{structure_input_ids.shape[1]} != {tokenized_sequences.shape[1]}")
        structure_input_ids = structure_input_ids.to(device)
    tokenized_sequences = tokenized_sequences.to(device)
    if attention_mask.dim() == 2 and attention_mask.shape[0] == 1:
        attention_mask = attention_mask.squeeze(0)
    attention_mask = attention_mask.to(device)
    plls = torch.empty(len(tokenized_sequences), device=device)
    for i, tokenized_seq in enumerate(tokenized_sequences):
        L = tokenized_seq.shape[0]
        pll = torch.tensor(0.0, device=device)

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

            if structure_input_ids is not None:
                masked_ss_input_ids = structure_input_ids.clone()
                masked_ss_input_ids[0, pos] = mask_token_id

            if train:
                if structure_input_ids is not None:
                    outputs = model(
                        input_ids=masked_input_ids.unsqueeze(0),
                        attention_mask=attention_mask.unsqueeze(0),
                        ss_input_ids=masked_ss_input_ids  # Check 
                    )
                else:
                    outputs = model(
                        input_ids=masked_input_ids.unsqueeze(0), 
                        attention_mask=attention_mask.unsqueeze(0), 
                        output_hidden_states=False
                    )
            else:
                with torch.no_grad():
                    if structure_input_ids is not None:
                        outputs = model(
                                input_ids=masked_input_ids.unsqueeze(0),
                                attention_mask=attention_mask.unsqueeze(0),
                                ss_input_ids=masked_ss_input_ids  # Check
                        )
                    else:
                        outputs = model(
                            input_ids=masked_input_ids.unsqueeze(0), 
                            attention_mask=attention_mask.unsqueeze(0), 
                            output_hidden_states=False
                        )
            logits = outputs.logits  # (1, L, V)

            log_probs = F.log_softmax(logits[0, pos], dim=-1)
            true_token = tokenized_seq[pos]
            pll = pll + log_probs[true_token]

        plls[i] = pll

    return plls


def plm_inference(
    xs,
    wt_input_ids,
    attention_mask,
    model,
    mask_token_id = None,
    inference_type='wt-marginal-log-likelihood',
    extract_emb: bool = False,
    wt_structure_input_ids=None,
    batch_size: int | None = 5,
    train=False,
    device=None,
    verbose=False,
):
    if device is None:
        device = get_device()
    
    if train:
        keep_remaining = False
    else:
        keep_remaining = True

    model = model.to(device)

    kwargs = {}

    if not isinstance(attention_mask, torch.Tensor):
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
    scoring_mode = None
    if inference_type == 'mutation-masking':
        inference_function = mutation_only_mutation_masked_pll
    elif inference_type in ['full-masking', 'all-pos-masking']:
        inference_function = mutation_all_pos_masked_pll
    # Unmasked
    elif inference_type in ['wt-marginal', 'wt-marginal-log-likelihood']:
        inference_function = sequence_log_likelihood
        scoring_mode = "wt-marginal"
    elif inference_type in ['full-sequence', 'full-sequence-log-likelihood']:
        inference_function = sequence_log_likelihood
        scoring_mode = "full-sequence"
    else:
        raise SystemError(
            f"Choose between 'wt-marginal-log-likelihood', "
            f"'full-sequence-log-likelihood', 'mutation-masking', "
            f"and 'full-masking', got {inference_type}.")
    if extract_emb:
        logger.info(f"Extracting sequence embeddings using the 'full-sequence-log-likelihood' "
                    f"function. ")
        inference_function = sequence_log_likelihood
        scoring_mode = "full-sequence"
        kwargs["extract_emb"] = True
        
    scores = []
    if batch_size is None:
        xs_b = torch.atleast_2d(xs)
    else:
        logger.info(f"Splitting tokenized sequences into batches...")
        xs_b = get_batches(xs, dtype=int, batch_size=batch_size,
                           keep_remaining=keep_remaining, verbose=True)
        xs_b = [torch.from_numpy(x).to(device) for x in xs_b]
    if extract_emb:
        desc = (f"Inference: getting embeddings batch "
                f"(size={batch_size}) processing ({device.upper()})")
    else:
        desc = (f"Inference: {inference_type} batch (size={batch_size}) "
                f"processing ({device.upper()})'")


    if mask_token_id is not None:
        kwargs["mask_token_id"] = mask_token_id

    if wt_structure_input_ids is not None:
        kwargs["ss_input_ids"] = wt_structure_input_ids.to(device)

    pbar = tqdm(
        xs_b,
        desc=desc,
        disable=not verbose
    )

    for x in pbar:
        pll = inference_function(
            tokenized_sequences=x,
            wt_input_ids=wt_input_ids,
            attention_mask=attention_mask,
            model=model,
            train=train,
            scoring_mode=scoring_mode,
            device=device,
            verbose=False,
            **kwargs
        )
        scores.append(pll)
    return torch.cat(scores)


def plm_train(
        x_sequences, 
        scores, 
        loss_fn, 
        model, 
        optimizer,
        wt_input_ids, 
        attention_mask, 
        batch_size: int = 5,
        n_epochs=50, 
        device: str | None = None, 
        seed: int | None = None,
        early_stop: int = 50, 
        verbose: bool = True, 
        wt_structure_input_ids=None,
        n_batch_grad_accumulations: int = 1, 
        raise_error_on_train_fail: bool = True,
        progress_cb=None, 
        abort_cb=None
):
    """
    TODO: Wrapper function for `plm_inference()` for PLM training.
    """
    if seed is not None:
        torch.manual_seed(seed)
    if device is None:
        device = get_device()
    logger.info(f"Model training using {device.upper()} device "
          f"(N_Train={len(scores)})...")
    scores_batched = torch.from_numpy(
        get_batches(scores, dtype=float, batch_size=batch_size,
                    keep_remaining=False, verbose=True)
    )
    x_sequences_batched = torch.from_numpy(
        get_batches(x_sequences, dtype=int, batch_size=batch_size, 
                    keep_remaining=False, verbose=True)
    )
    x_sequences_batched = x_sequences_batched.to(device)
    scores_batched = scores_batched.to(device)
    pbar_epochs = tqdm(range(1, n_epochs + 1), disable=not verbose)
    epoch_spearman_1 = -1.0
    did_not_improve_counter = 0
    best_model = None
    best_model_epoch = np.nan
    best_model_perf = np.nan
    loss = np.nan
    os.makedirs('model_saves', exist_ok=True)
    for epoch in pbar_epochs:
        if epoch == 0:
            pbar_epochs.set_description(f'Epoch {epoch}/{n_epochs}')
        model.train()
        y_preds_detached = []
        pbar_batches = tqdm(
            zip(x_sequences_batched, scores_batched),
            total=len(x_sequences_batched), leave=False, disable=not verbose
        )
        for batch, (seqs_b, scores_b) in enumerate(pbar_batches):
            if abort_cb and abort_cb():
                return
            if seqs_b.dim() == 2:
                seqs_b = seqs_b.unsqueeze(0)  # e.g., (5, 400)  -> (1, 5 400)
            y_preds_b = plm_inference(
                xs=seqs_b, 
                wt_input_ids=wt_input_ids, 
                attention_mask=attention_mask,
                model=model, 
                train=True, 
                wt_structure_input_ids=wt_structure_input_ids, 
                batch_size=None, 
                device=device,
                verbose=False
            )
            y_preds_detached.append(y_preds_b.detach().cpu().numpy().flatten())
            loss = loss_fn(scores_b, y_preds_b) / n_batch_grad_accumulations
            if progress_cb:
                progress_cb(epoch - 1, batch + 1, len(pbar_epochs), len(pbar_batches), loss)
            loss.backward()
            if (batch + 1) % n_batch_grad_accumulations == 0 or (batch + 1) == len(pbar_batches):
                optimizer.step()
                optimizer.zero_grad()
            pbar_batches.set_description(
                f"Epoch: {epoch}. Loss: {loss.detach():>1f} "
                f"[batch: {batch + 1}/{len(x_sequences_batched)} | "
                f"sequence: {(batch + 1) * len(x_sequences_batched[0]):>5d}/"
                f"{len(x_sequences)}] ({device.upper()})"
            )
        epoch_spearman_2 = spearmanr(scores_batched.cpu().numpy().flatten(),
                                     np.array(y_preds_detached).flatten())[0]
        if epoch_spearman_2 == np.nan:
            raise SystemError(
                f"No correlation between Y_true and Y_pred could be computed...\n"
                f"Y_true: {scores_batched.cpu().numpy().flatten()}, "
                f"Y_pred: {np.array(y_preds_detached)}"
            )
        if epoch_spearman_2 > epoch_spearman_1 or epoch == 0:
            if best_model is not None:
                if os.path.isfile(best_model):
                    os.remove(best_model)
            did_not_improve_counter = 0
            best_model_epoch = epoch
            best_model_perf = epoch_spearman_2
            best_model = (
                f"model_saves/Epoch{epoch}-Ntrain{len(scores_batched.cpu().numpy().flatten())}"
                f"-SpearCorr{epoch_spearman_2:.3f}.pt"
            )
            checkpoint(model, best_model)
            epoch_spearman_1 = epoch_spearman_2
            logger.info(f"Saved current best model as {best_model}")
        else:
            did_not_improve_counter += 1
            if did_not_improve_counter >= early_stop:
                logger.info(f'\nEarly stop at epoch {epoch}...')
                break
        loss_total = loss_fn(
            torch.flatten(scores_batched).to('cpu'),
            torch.flatten(torch.Tensor(np.array(y_preds_detached).flatten()))
        )
        pbar_epochs.set_description(
            f'Epoch {epoch}/{n_epochs} [SpearCorr: {epoch_spearman_2:.3f}, Loss: {loss_total:.3f}] '
            f'(Best epoch: {best_model_epoch}: {best_model_perf:.3f})')
    if progress_cb:
        progress_cb(epoch, batch + 1, len(pbar_epochs), len(pbar_batches), loss)
    if best_model is None:
        msg = ("Failed to train a model (probably due to the input "
               "data characteristics and loss/correlation being NaN).")
        if raise_error_on_train_fail:
            raise RuntimeError(msg)
        else:
            logger.warning(f"{msg} Continuing nonetheless (using failed model "
                           f"and replacing NaN's with zeros)...")
            y_preds_train = plm_inference(
                x_sequences,#.flatten(start_dim=0, end_dim=1),
                wt_input_ids, 
                attention_mask, 
                model,
                wt_structure_input_ids=wt_structure_input_ids,
                train=False, 
                verbose=False
            )
            y_preds_train[torch.isnan(y_preds_train)] = 0.0
    else:        
        logger.info(f"Loading best model as {best_model}...")
        load_model(model, best_model)
        y_preds_train = plm_inference(
                x_sequences,#.flatten(start_dim=0, end_dim=1),
                wt_input_ids, 
                attention_mask, 
                model,
                wt_structure_input_ids=wt_structure_input_ids,
                train=False, 
                verbose=False
            )
    return y_preds_train.cpu()


def esm_setup(wt_seq, sequences, device: str | None = None, verbose: bool = True):
    esm_base_model, esm_lora_model, esm_tokenizer, esm_optimizer = get_esm_models()
    esm_base_model = esm_base_model.to(device)
    wt_tokens, _ = tokenize_sequences(
            [wt_seq],
            esm_tokenizer,
            max_length=len(wt_seq) + 2
    )
    x_esm, esm_attention_mask = tokenize_sequences(
        sequences, esm_tokenizer, max_length=len(wt_seq) + 2, verbose=verbose)
    llm_dict_esm = {
        'esm1v': {
            'llm_base_model': esm_base_model,
            'llm_model': esm_lora_model,
            'llm_optimizer': esm_optimizer,
            'llm_train_function': plm_train,
            'llm_inference_function': plm_inference,
            'llm_loss_function': spearman_loss,  # pearson_loss,
            'x_llm' : torch.tensor(x_esm),  # TODO: Not needed here?
            'llm_attention_mask':  torch.tensor(esm_attention_mask),  # TODO: Not needed here?
            'wt_input_ids': torch.tensor(wt_tokens),  # TODO: Not needed here?
            'llm_tokenizer': esm_tokenizer
        }
    }
    return llm_dict_esm


def prosst_setup(wt_seq, pdb_file, sequences, device: str | None = None, verbose: bool = True):
    if wt_seq is None:
        raise SystemError(
            "Running ProSST requires a wild-type sequence "
            "FASTA file input for embedding sequences! "
            "Specify a FASTA file with the --wt flag."
        )
    if pdb_file is None:
        raise SystemError(
            "Running ProSST requires a PDB file input "
            "for embedding sequences! Specify a PDB file "
            "with the --pdb flag."
        )

    pdb_seq = str(list(SeqIO.parse(pdb_file, "pdb-atom"))[0].seq)
    assert wt_seq == pdb_seq, (
        f"Wild-type sequence is not matching PDB-extracted sequence:"
        f"\nWT sequence:\n{wt_seq}\nPDB sequence:\n{pdb_seq}"
    )
    prosst_base_model, prosst_lora_model, prosst_tokenizer, prosst_optimizer = get_prosst_models()
    prosst_vocab = prosst_tokenizer.get_vocab()
    prosst_base_model = prosst_base_model.to(device)
    prosst_optimizer = torch.optim.Adam(prosst_lora_model.parameters(), lr=0.0001)
    input_ids, prosst_attention_mask, structure_input_ids = get_structure_quantizied(
        pdb_file, prosst_tokenizer, wt_seq, verbose=verbose
    )
    x_llm_train_prosst, _attention_mask = tokenize_sequences(
        sequences=sequences, tokenizer=prosst_tokenizer, 
        max_length=len(wt_seq) + 2, verbose=verbose
    )
    llm_dict_prosst = {
        'prosst': {
            'llm_base_model': prosst_base_model,
            'llm_model': prosst_lora_model,
            'llm_optimizer': prosst_optimizer,
            'llm_train_function': plm_train,
            'llm_inference_function': plm_inference,  # prosst_infer,
            'llm_loss_function': spearman_loss,  # pearson_loss,
            'x_llm' : x_llm_train_prosst,
            'llm_attention_mask': prosst_attention_mask,
            'llm_vocab': prosst_vocab,
            'wt_input_ids': input_ids,
            'structure_input_ids': structure_input_ids,
            'llm_tokenizer': prosst_tokenizer
        }
    }
    return llm_dict_prosst
