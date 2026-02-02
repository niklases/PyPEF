# PyPEF - Pythonic Protein Engineering Framework
# https://github.com/niklases/PyPEF

# Some helper functions for infernece of different models 
# based on simple/wrapping functions

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from pypef.utils.helpers import get_device
from pypef.plm.utils import get_batches
from pypef.plm.esm_lora_tune import esm_infer, esm_setup, tokenize_sequences
from pypef.plm.prosst_lora_tune import prosst_setup, prosst_simple_vocab_aa_tokenizer, prosst_infer

import logging
logger = logging.getLogger('pypef.llm.inference')


def tokenize_sequences(sequences, tokenizer, max_length, verbose=True):
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


def unmasked_wt_score(
        tokenized_sequences, 
        attention_mask, 
        wt_input_ids,
        model, 
        train: bool = False,
        cut_special_tokens: bool = True,  # assumption: cut first and last token
        device=None,
        verbose: bool = False,
        **model_kwargs
    ):
    if device is None:
        device = get_device()
    if wt_input_ids.dim() == 1:
        wt_input_ids = wt_input_ids.unsqueeze(0)
    #structure_input_ids = model_kwargs.get("structure_input_ids", None)

    attention_masks = torch.Tensor(np.full(
        shape=np.shape(wt_input_ids), fill_value=attention_mask)).to(torch.int64)
    if train:
        outputs = model(
            input_ids=wt_input_ids.to(device),
            attention_mask=attention_masks.to(device),
            **model_kwargs
        )

    else:
        with torch.no_grad():
            outputs = model(
                input_ids=wt_input_ids.to(device),
                attention_mask=attention_masks.to(device),
                **model_kwargs
            )

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

    log_probs = []
    for tokenized_seq in tokenized_sequences:
        if cut_special_tokens:
            tokenized_seq = tokenized_seq[1:-1]
    
        seq_lp = token_probs[
            torch.arange(tokenized_seq.shape[0], device=tokenized_seq.device),
            tokenized_seq
        ].sum(dtype=torch.float64)

        log_probs.append(seq_lp)
    
    log_probs = torch.stack(log_probs)
    return log_probs


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
    structure_input_ids = kwargs.get("structure_input_ids", None)
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
    tokenized_sequences: torch.Tensor,        # (L,)
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
    structure_input_ids = kwargs.get("structure_input_ids", None)
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
    inference_type='unmasked',
    wt_structure_input_ids=None,
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
    if inference_type == 'mutation-masking':
        inference_function = mutation_only_mutation_masked_pll
    elif inference_type in ['full-masking', 'all-pos-masking']:
        inference_function = mutation_all_pos_masked_pll
    elif inference_type in ['unmasked', 'wt-marginals']:
        inference_function = unmasked_wt_score
    else:
        raise SystemError("Choose between 'mutation-masking', 'unmasked', and 'full-masking'")

    scores = []

    xs_b = get_batches(xs, dtype=int, batch_size=batch_size, keep_remaining=True, verbose=True)
    desc = f"Inference: {inference_type} batch (size={batch_size}) processing ({device.upper()})'"

    kwargs = {}
    if mask_token_id is not None:
        kwargs["mask_token_id"] = mask_token_id

    if wt_structure_input_ids is not None:
        kwargs["structure_input_ids"] = wt_structure_input_ids

    pbar = tqdm(
        range(len(xs_b)),
        desc=desc,
        disable=not verbose
    )

    for i in pbar:
        pll = inference_function(
            tokenized_sequences=torch.tensor(xs_b[i]),
            wt_input_ids=wt_input_ids,
            attention_mask=attention_mask,
            model=model,
            train=train,
            device=device,
            verbose=False,
            **kwargs
        )
        scores.append(pll)
    return torch.cat(scores)


######################### Deprecated

def llm_tokenizer(llm_dict, seqs, verbose=True):
    try:
        np.shape(seqs)
    except ValueError:
        raise SystemError("Unequal input sequence length detected!")
    if list(llm_dict.keys())[0] == 'esm1v':
        x_llm_seqs, _attention_mask = tokenize_sequences(
            seqs, tokenizer=llm_dict['esm1v']['llm_tokenizer'], 
            max_length=len(seqs[0]) + 2, verbose=verbose
        )
    elif list(llm_dict.keys())[0] == 'prosst':
        x_llm_seqs, _attention_mask = tokenize_sequences(
            seqs, tokenizer=llm_dict['prosst']['llm_tokenizer'], 
            max_length=len(seqs[0]) + 2, verbose=verbose
        )
    else:
        raise SystemError(f"Unknown LLM dictionary input:\n{list(llm_dict.keys())[0]}")
    return x_llm_seqs


def inference(
        sequences,
        llm: str,
        pdb_file: str | None = None,
        wt_seq: str | None = None,
        device: str| None = None,
        model = None,
        verbose: bool = True
):
    """
    Inference of input or base model.
    """
    if device is None:
        device = get_device()
    if llm == 'esm':
        logger.info("Zero-shot LLM inference on test set using ESM1v...")
        llm_dict = esm_setup(wt_seq, sequences, verbose=verbose)
        if model is None:
            model = llm_dict['esm1v']['llm_base_model']
        x_llm_test = llm_tokenizer(llm_dict, sequences, verbose)
        y_test_pred = esm_infer(#llm_dict['esm1v']['llm_inference_function'](
            xs=torch.from_numpy(get_batches(x_llm_test, batch_size=1, dtype=int)), 
            attention_mask=llm_dict['esm1v']['llm_attention_mask'], 
            model=model, 
            device=device,
            verbose=verbose
        ).cpu()
        y_test_pred = plm_inference(
            xs=x_llm_test,
            wt_input_ids=torch.tensor(llm_dict['esm1v']['input_ids'][0], dtype=torch.long),
            attention_mask=llm_dict['esm1v']['llm_attention_mask'],
            model=model,
            mask_token_id=llm_dict['esm1v']['llm_tokenizer'].mask_token_id,
            inference_type='unmasked',
            batch_size=5,
            train=False,
            verbose=True
        ).cpu()

    elif llm == 'prosst':
        logger.info("Zero-shot LLM inference on test set using ProSST...")
        llm_dict = prosst_setup(
            wt_seq, pdb_file, sequences=sequences, verbose=verbose
        )
        if model is None:
            model = llm_dict['prosst']['llm_base_model']
        x_llm_test = llm_tokenizer(llm_dict, sequences, verbose)
        #y_test_pred = prosst_infer(#llm_dict['prosst']['llm_inference_function'](
        #    xs=x_llm_test, 
        #    model=model, 
        #    input_ids=llm_dict['prosst']['input_ids'], 
        #    attention_mask=llm_dict['prosst']['llm_attention_mask'], 
        #    structure_input_ids=llm_dict['prosst']['structure_input_ids'],
        #    verbose=verbose,
        #    device=device
        #).cpu()
        print('XXX:', np.shape(x_llm_test))
        y_test_pred = plm_inference(
            xs=x_llm_test,
            wt_input_ids=llm_dict['prosst']['input_ids'],
            attention_mask=llm_dict['prosst']['llm_attention_mask'],
            model=model,
            mask_token_id=llm_dict['prosst']['llm_tokenizer'].mask_token_id,
            inference_type='mutation-masking',
            wt_structure_input_ids=llm_dict['prosst']['structure_input_ids'],
            batch_size=5,
            train=False,
            verbose=True   
        ).cpu()
    else:
        raise RuntimeError("Unknown LLM option.")
    return y_test_pred
