# PyPEF - Pythonic Protein Engineering Framework
# https://github.com/niklases/PyPEF


import logging

from pypef.dca.gremlin_inference import GREMLIN
logger = logging.getLogger('pypef.utils.ssm')

import os
from typing import Literal, Union
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np

from pypef.llm.inference import inference


def get_ssm(wt_seq: str, aas: str = "ARNDCQEGHILKMFPSTWYV"):
    """
    Function to plot all predicted 19 amino acid substitution 
    effects at all predictable WT/input sequence positions; e.g.: 
    M1A, M1C, M1E, ..., D2A, D2C, D2E, ..., ..., T300V, T300W, T300Y
    """
    variantss, variant_sequencess = [], []
    logger.info("Predicting all SSM effects using the unsupervised GREMLIN model...")
    for i, aa_wt in enumerate(tqdm(wt_seq)):
        variants, variant_sequences = [], []
        for aa_sub in aas:
            variant = aa_wt + str(i + 1) + aa_sub
            variant_sequence = wt_seq[:i] + aa_sub + wt_seq[i + 1:]
            variants.append(variant)
            variant_sequences.append(variant_sequence)
        variantss.append(variants)
        variant_sequencess.append(variant_sequences)


def predict_ssm(
    variant_sequences: Union[list[list[str]]],
    model: Literal["dca", "esm", "prosst"] = "dca",
    pdb: str | os.PathLike | None = None,
    wt_seq: str | os.PathLike | None = None,
    gremlin: GREMLIN | None = None
):
    if model == "dca":
        wt_score = gremlin.get_wt_score()[0]
    variant_scores = []
    for seqs in variant_sequences:
        scores = []
        for seq in seqs:
            if model == "dca":
                scores.append(gremlin.get_scores(seq)[0] - wt_score)
            else:
                scores.append(inference(seq, llm=model, pdb_file=pdb, wt_seq=wt_seq))
                pass # TODO


def plot_ssm(
        wt_seq: str, 
        variants: Union[list[list[str]]], 
        variant_sequences: Union[list[list[str]]], 
        variant_scores: Union[list[list[float]]] | np.ndarray, 
        aas: str = "ARNDCQEGHILKMFPSTWYV"
    ):
    _fig, ax = plt.subplots(figsize=(2 * len(wt_seq) / len(aas), 3))
    ax.imshow(np.array(variant_scores).T)
    for i_vss, vss in enumerate(variant_scores):
        for i_vs, vs in enumerate(vss):
            ax.text(
                i_vss, i_vs, 
                f'{variants[i_vss][i_vs]}\n{round(vs, 1)}', 
                size=1.5, va='center', ha='center'
            )
    ax.set_xticks(
        range(len(wt_seq)), 
        [f'{aa}{i + 1}' for i, aa in enumerate(wt_seq)], 
        size=6, rotation=90
    )
    ax.set_yticks(range(len(aas)), aas, size=6)
    plt.tight_layout()
    plt.savefig('SSM_landscape.png', dpi=500)
    plt.clf()
    plt.close('all')
    pd.DataFrame(
        {
            'Variant': np.array(variants).flatten(),
            'Sequence': np.array(variant_sequences).flatten(),
            'Variant_Score': np.array(variant_scores).flatten()
        }
    ).to_csv('SSM_landscape.csv', sep=',')
    logger.info(f"Saved SSM landscape as {os.path.abspath('SSM_landscape.png')} "
                f"and CSV data as {os.path.abspath('SSM_landscape.csv')}...")
