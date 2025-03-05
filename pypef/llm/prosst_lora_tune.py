# Niklas Siedhoff
# PyPEF - Pythonic Protein Engineering Framework

# Using (training, testing/infering) ProSST model(s) published under 
# GNU GENERAL PUBLIC LICENSE: GPL-3.0 license
# https://github.com/ai4protein/ProSST
#import warnings
#warnings.filterwarnings('error')

import torch
from scipy.stats import spearmanr
from tqdm import tqdm

from transformers import AutoModelForMaskedLM, AutoTokenizer
prosst_model = AutoModelForMaskedLM.from_pretrained("AI4Protein/ProSST-2048", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("AI4Protein/ProSST-2048", trust_remote_code=True)

from peft import LoraConfig, get_peft_model

from pypef.utils.variant_data import get_seqs_from_var_name
from pypef.llm.esm_lora_tune import corr_loss
from pypef.llm.prosst_structure.quantizer import PdbQuantizer


def get_logits_from_full_seqs(sequences, model, input_ids, attention_mask, structure_input_ids):
    with torch.no_grad():
        outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                ss_input_ids=structure_input_ids
        )
        vocab = tokenizer.get_vocab()
        logits = torch.log_softmax(outputs.logits[:, 1:-1], dim=-1).squeeze()
        for i_s, sequence in enumerate(tqdm(sequences)):
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


def get_scores(wt_seq, variants, y_true, pdb_path):
    structure_sequence = PdbQuantizer()(pdb_file=pdb_path)
    print(len(wt_seq))
    print(structure_sequence)
    print(len(structure_sequence))
    structure_sequence_offset = [i + 3 for i in structure_sequence]
    tokenized_res = tokenizer([wt_seq], return_tensors='pt')
    input_ids = tokenized_res['input_ids']
    attention_mask = tokenized_res['attention_mask']
    structure_input_ids = torch.tensor([1, *structure_sequence_offset, 2], dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        outputs = prosst_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            ss_input_ids=structure_input_ids
        )
    logits = torch.log_softmax(outputs.logits[:, 1:-1], dim=-1).squeeze()

    vocab = tokenizer.get_vocab()
    pred_scores = []
    for mutant in variants:
        mutant_score = 0
        for sub_mutant in mutant.split(":"):
            wt, idx, mt = sub_mutant[0], int(sub_mutant[1:-1]) - 1, sub_mutant[-1]
            pred = logits[idx, vocab[mt]] - logits[idx, vocab[wt]]
            mutant_score += pred.item()
        pred_scores.append(mutant_score)
    print(spearmanr(y_true, pred_scores))
    peft_config = LoraConfig(r=8, target_modules=["query", "value"])
    prosst_model_peft = get_peft_model(prosst_model, peft_config)
    optimizer = torch.optim.Adam(prosst_model_peft.parameters(), lr=0.01)
    loss_fn = corr_loss
    loss = loss_fn(y_true, pred_scores)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

if __name__ == '__main__':
    import os
    import pandas as pd
    # Test on dataset GRB2_HUMAN_Faure_2021: SignificanceResult(statistic=0.6997442598613315, pvalue=0.0)
    wt_seq = "MEAIAKYDFKATADDELSFKRGDILKVLNEECDQNWYKAELNGKDGFIPKNYIEMKPHPWFFGKIPRAKAEEMLSKQRHDGAFLIRESESAPGDFSLSVKFGNDVQHFKVLRDGAGKYFLWVVKFNSLNELVDYHRSTSVSRNQQIFLRDIEQVPQQPTYVQALFDFDPQEDGELGFRRGDFIHVMDNSDPNWWKGACHGQTGMFPRNYVTPVNRNV"
    grb2_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'datasets', 'GRB2'))
    pdb_file = os.path.join(grb2_folder, 'GRB2_HUMAN.pdb')
    csv_file = os.path.join(grb2_folder, 'GRB2_HUMAN_Faure_2021.csv')
    df = pd.read_csv(csv_file)
    print(df)
    structure_sequence = PdbQuantizer()(pdb_file=pdb_file)
    structure_sequence_offset = [i + 3 for i in structure_sequence]
    tokenized_res = tokenizer([wt_seq], return_tensors='pt')
    input_ids = tokenized_res['input_ids']
    attention_mask = tokenized_res['attention_mask']
    structure_input_ids = torch.tensor([1, *structure_sequence_offset, 2], dtype=torch.long).unsqueeze(0)
    y_pred = get_logits_from_full_seqs(df['mutated_sequence'], prosst_model, input_ids, attention_mask, structure_input_ids)
    print(spearmanr(df['DMS_score'], y_pred))  # SignificanceResult(statistic=np.float64(0.7216670719282277), pvalue=np.float64(0.0))
