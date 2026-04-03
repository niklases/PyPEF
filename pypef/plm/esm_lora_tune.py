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
logger = logging.getLogger('pypef.plm.esm_lora_tune')

import copy
import torch
from peft import LoraConfig, get_peft_model
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

from pypef.plm.utils import load_model_and_tokenizer
from pypef.plm.utils import _set_seeds


def get_esm_models(
        model='facebook/esm1v_t33_650M_UR90S_3', 
        seed: None | bool = None, 
        revision: str | None = None
):
    if seed is not None:
        _set_seeds(seed)
    base_model, tokenizer = load_model_and_tokenizer(
        model,
        revision=revision
        # Just sticking to AutoModelForMaskedLM and AutoTokenizer 
        # instead to EsmForMaskedLM and EsmTokenizer
    )  
    for param in base_model.parameters():
        param.requires_grad = False
    base_model.eval()
    base_model = copy.deepcopy(base_model)
    peft_config = LoraConfig(r=8, target_modules=["query", "value"])
    lora_model = get_peft_model(base_model, peft_config)
    optimizer = torch.optim.Adam(lora_model.parameters(), lr=0.01)
    return base_model, lora_model, tokenizer, optimizer
