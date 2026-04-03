# PyPEF - Pythonic Protein Engineering Framework
# https://github.com/niklases/PyPEF

# Using (training, testing/infering) ProSST model(s) published under
# GNU GENERAL PUBLIC LICENSE: GPL-3.0 license
# Code repository: https://github.com/ai4protein/ProSST
# Mingchen Li, Pan Tan, Xinzhu Ma, Bozitao Zhong, Huiqun Yu, Ziyi Zhou,
# Wanli Ouyang, Bingxin Zhou, Liang Hong, Yang Tan
# ProSST: Protein Language Modeling with Quantized Structure and Disentangled Attention
# bioRxiv 2024.04.15.589672; doi: https://doi.org/10.1101/2024.04.15.589672

import logging
logger = logging.getLogger('pypef.plm.prosst_lora_tune')

import warnings
import copy
import torch
import numpy as np
from tqdm import tqdm
from peft import LoraConfig, get_peft_model
from Bio import BiopythonParserWarning
warnings.filterwarnings(action='ignore', category=BiopythonParserWarning)

from pypef.plm.prosst_structure.quantizer import PdbQuantizer
from pypef.utils.helpers import get_device
from pypef.plm.utils import _set_seeds, load_model_and_tokenizer


def prosst_simple_vocab_aa_tokenizer(sequences, vocab, verbose=True):
    sequences = np.atleast_1d(sequences).tolist()
    x_sequences = []
    for sequence in tqdm(
        sequences, desc='Tokenizing sequences for ProSST modeling', 
        disable=not verbose
    ):
        x_sequence = [vocab['<cls>']]
        for aa in sequence:
            try:
                x_sequence.append(vocab[aa])
            except KeyError:
                x_sequence.append(vocab['<unk>'])
        x_sequence.append(vocab['<eos>'])
        x_sequences.append(x_sequence)
    return torch.Tensor(x_sequences).to(torch.int)


def get_logits_from_full_seqs(
        xs,
        model,
        input_ids,
        attention_mask,
        structure_input_ids,
        train: bool = False,
        verbose: bool = False,
        device: str | None = None,
        replace_nan_with_zeros: bool = False
):
    if device is None:
        device = get_device()
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
    for i_s, x_sequence in enumerate(
        tqdm(
            xs,
            desc=f'ProSST inference: getting sequence logits ({device.upper()})',
            disable=not verbose
        )
    ):
        x_sequence = x_sequence[1:-1] # if cls, eos tokens included
        for i_aa, x_aa in enumerate(x_sequence):
            if i_aa == 0:
                seq_log_probs = logits[i_aa, x_aa].reshape(1)
            else:
                seq_log_probs = torch.cat(
                    (seq_log_probs, logits[i_aa, x_aa].reshape(1)), 0)
        if i_s == 0:
            log_probs = torch.sum(torch.Tensor(seq_log_probs)).reshape(1)
        else:
            log_probs = torch.cat((
                log_probs,
                torch.sum(torch.Tensor(seq_log_probs)).reshape(1)
                ), 0
            )
    if replace_nan_with_zeros:
        logger.warning("Replacing NaN's with zeros in predictions...")
        log_probs[torch.isnan(log_probs)] = 0.0
    return log_probs


def prosst_infer(
        xs,
        model,
        input_ids,
        attention_mask,
        structure_input_ids,
        verbose: bool = False,
        device: str | None = None,
        replace_nan_with_zeros: bool = False
):
    return get_logits_from_full_seqs(
        xs,
        model,
        input_ids,
        attention_mask,
        structure_input_ids,
        train = False,
        verbose = verbose,
        device = device,
        replace_nan_with_zeros=replace_nan_with_zeros
    )


def get_prosst_models(seed: None | bool = None, revision: str | None = None):
    if seed is not None:
        _set_seeds(seed)
    prosst_base_model, tokenizer = load_model_and_tokenizer("AI4Protein/ProSST-2048", revision=revision)
    for param in prosst_base_model.parameters():
        param.requires_grad = False
    prosst_base_model.eval()
    prosst_base_model = copy.deepcopy(prosst_base_model)
    peft_config = LoraConfig(r=8, target_modules=["query", "value"])
    prosst_lora_model = get_peft_model(prosst_base_model, peft_config)
    optimizer = torch.optim.Adam(prosst_lora_model.parameters(), lr=0.01)
    return prosst_base_model, prosst_lora_model, tokenizer, optimizer


def get_structure_quantizied(pdb_file, tokenizer, wt_seq, device: None | str = None, verbose: bool = True):
    structure_sequence = PdbQuantizer(device=device, verbose=verbose)(pdb_file=pdb_file)
    structure_sequence_offset = [i + 3 for i in structure_sequence]
    tokenized_res = tokenizer([wt_seq], return_tensors='pt')
    input_ids = tokenized_res['input_ids']
    attention_mask = tokenized_res['attention_mask']
    structure_input_ids = torch.tensor(
        [1, *structure_sequence_offset, 2],
        dtype=torch.long
    ).unsqueeze(0)
    return input_ids, attention_mask, structure_input_ids
