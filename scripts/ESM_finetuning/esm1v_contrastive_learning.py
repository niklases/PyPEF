# Niklas Siedhoff, 17.01.2025
# Inspired from ConFit
# https://github.com/luo-group/ConFit


import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy import stats
import pandas as pd
from peft import PeftModel, PeftConfig, LoraConfig, get_peft_model
from peft.utils.other import fsdp_auto_wrap_policy
from transformers import EsmForMaskedLM, EsmTokenizer, EsmConfig
from scipy.stats import spearmanr
import os
import gc
import matplotlib.pyplot as plt
from matplotlib.colors import XKCD_COLORS
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
#device="cpu"
# https://github.com/facebookresearch/esm/blob/2b369911bb5b4b0dda914521b9475cad1656b2ac/esm/data.py#L91 -->
# https://github.com/facebookresearch/esm/blob/main/esm/constants.py#L7
#proteinseq_toks = {
#    'toks': ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O', '.', '-']
#}
## self.tok_to_idx = {tok: i for i, tok in enumerate(self.all_toks)}
proteinseq_toks = {
    'toks': ['<null_0>', '<pad>', '<eos>', '<unk>',
    'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O', '.', '-',
    '<cls>', '<mask>', '<sep>']
}
#print(len(proteinseq_toks['toks']))

"""

seqs_, attention_mask = tokenizer(
    seqs, 
    padding='max_length', 
    truncation=True, 
    max_length=len(wt_seq)
    ).values()
print(seqs_)
print(np.shape(seqs_), np.shape(attention_mask))
print(muts)
log_scores = []
wt_score = 0.0
with torch.no_grad():
    wt_sequence_distribution = []
    for i, (seq_, attn_msk) in enumerate(tqdm(zip(seqs_, attention_mask), total=len(seqs_))):
        out = model(torch.tensor([seq_]), torch.tensor([attn_msk]), output_hidden_states=True)  # batch of 1 
        logits = out.logits
        #print(i,'/',len(seqs_), logits.shape)
        log_probs = torch.log_softmax(logits, dim=-1)
        #print(log_probs, log_probs.shape)
        if i == 0:
            for i_pos, aa_pos in enumerate(log_probs[0]):  # is of shape x,y,1 so one can simply use [0] 
                #print(f'Seq. {i}, AA pos {i_pos+2}, log Probs.: {aa_pos}')
                canonical_positional_aa_distribution = []
                for i_aa_prob, aa_prob in enumerate(aa_pos):
                    #print(f"AA {i_aa_prob} = {proteinseq_toks['toks'][i_aa_prob]} : {aa_prob}")
                    if 4 <= i_aa_prob <= 23:
                        canonical_positional_aa_distribution.append(aa_prob)
                wt_sequence_distribution.append(canonical_positional_aa_distribution)
                #print(np.shape(wt_sequence_distribution))
        if i == 0:
            wt_score = torch.sum(log_probs)
        else:
            log_scores.append(float(torch.sum(log_probs).cpu()))
            #print(torch.sum(log_probs))

print('WT score:', wt_score)
df['predicted_log_score_ESM1v'] = log_scores
df.to_csv('esm1_pred.csv')
print(stats.spearmanr(scores, log_scores))
print('np.shape(wt_sequence_distribution):',np.shape(wt_sequence_distribution))

var_y_trues = dict(zip(muts, scores))
var_y_preds = dict(zip(muts, log_scores))


fig, ax = plt.subplots(figsize=(30, 6))
k = 0
x_tick_poses, labels = [], []
for i, aa_distr in enumerate(wt_sequence_distribution):
    x_tick_pos = np.array(range(len(aa_distr))) + k
    for aa in proteinseq_toks['toks'][4:24]:
        labels.append(f"{i+2}{aa}")
    if i == 0:
        plt.bar(x_tick_pos, aa_distr, label=proteinseq_toks['toks'][4:24], color=XKCD_COLORS)
    else:
        plt.bar(x_tick_pos, aa_distr, color=XKCD_COLORS)
    x_tick_poses.append(x_tick_pos)
    k += len(aa_distr) + 1
plt.legend()
plt.xticks(np.array(x_tick_poses).flatten(), labels, size=1, rotation=45)
plt.margins(0.01)
plt.tight_layout()
plt.savefig('aa_esm1v_probability_distribution.png', dpi=300)


#yt_vs_sorted, yt_fs_sorted, yp_vs_sorted_according_to_ytfs, yp_fs_sorted_according_to_ytfs = sort_var_fits(var_y_trues, var_y_preds)
#get_loss(yp_fs_sorted_according_to_ytfs)
"""


def sort_var_fits(var_fits_true, var_fits_pred):

    def get_kvs(d):
        variants, fitnesses = [], []
        for k, v in d.items():
            variants.append(k)
            fitnesses.append(v)
        return variants, fitnesses
    
    yt_vs, yt_fs = get_kvs(var_fits_true)
    yp_vs, yp_fs = get_kvs(var_fits_pred)
    assert len(yt_vs) == len(yt_fs) == len(yp_vs) == len(yp_fs)
    if not yt_vs == yp_vs:
        yp_vs_temp, yp_fs_temp = [], []
        for vart, _yt in zip(yt_vs, yt_fs):
            for varp, yp in zip(yp_vs, yp_fs):
                if vart == varp:
                    yp_vs_temp.append(vart)
                    yp_fs_temp.append(yp)
        yp_vs, yp_fs = yp_vs_temp, yp_fs_temp
        assert yt_vs == yp_vs
    (
        yt_vs_sorted, yt_fs_sorted, 
        yp_vs_sorted_according_to_ytfs, yp_fs_sorted_according_to_ytfs
    ) = [list(l) for l in zip(*sorted(zip(yt_vs, yt_fs, yp_vs, yp_fs), key=lambda x: x[1]))]

    assert yt_vs_sorted == yp_vs_sorted_according_to_ytfs

    return yt_vs_sorted, yt_fs_sorted, yp_vs_sorted_according_to_ytfs, yp_fs_sorted_according_to_ytfs


def get_encoded_seqs(sequences, tokenizer, max_length=104):
    encoded_sequences, attention_masks = tokenizer(
        sequences, 
        padding='max_length', 
        truncation=True, 
        max_length=max_length
    ).values()
    return encoded_sequences, attention_masks

def get_y_pred_scores(encoded_sequences, attention_masks, model):
    out = model(encoded_sequences.to(device), attention_masks.to(device), output_hidden_states=True)
    logits = out.logits
    token_probs = torch.log_softmax(logits, dim=-1)
    log_probs = []
    for i_s, sequence in enumerate(encoded_sequences):
        seq_log_probs = []
        for i_aa, aa in enumerate(sequence):
            #print('Target AA:', i_aa, aa, proteinseq_toks['toks'][aa], token_probs[i_s, i_aa, aa])
            if i_aa == 0:
                seq_log_probs = token_probs[i_s, i_aa, aa].reshape(1)  # or better just use Tensor.index_select() function!
                #print(seq_log_probs)
            else:

                seq_log_probs = torch.cat((seq_log_probs, token_probs[i_s, i_aa, aa].reshape(1)), 0)
            #seq_log_probs.append(token_probs[i_s, i_aa, aa])
        if i_s == 0:
            log_probs = torch.sum(torch.Tensor(seq_log_probs)).reshape(1)
        else:
            #print(i_s, log_probs2)
            log_probs = torch.cat((log_probs, torch.sum(torch.Tensor(seq_log_probs)).reshape(1)), 0)
    #print(log_probs2)
    return log_probs


def _get_ranks(x: torch.Tensor) -> torch.Tensor:
    tmp = x.argsort()
    ranks = torch.zeros_like(tmp)
    ranks[tmp] = torch.arange(len(x))
    return ranks

def spearman_correlation(x: torch.Tensor, y: torch.Tensor):
    """Compute correlation between 2 1-D vectors
    Args:
        x: Shape (N, )
        y: Shape (N, )
    """
    x_rank = _get_ranks(x)
    y_rank = _get_ranks(y)
    
    n = x.size(0)
    upper = 6 * torch.sum((x_rank - y_rank).pow(2))
    down = n * (n ** 2 - 1.0)
    return 1.0 - (upper / down)


# yt_vs_sorted, yt_fs_sorted, yp_vs_sorted_according_to_ytfs, yp_fs_sorted_according_to_ytfs = sort_var_fits(var_y_trues, var_y_preds)

# TODO:
##############################################################################
## Adapt ESM1v model LoRA parameters based on BT loss to low N observations (fitness values) 


def corr_loss(y_true, y_pred):
    res_true = y_true - torch.mean(y_true)
    res_pred = y_pred - torch.mean(y_pred)
    #res_true = res_true.to(device)
    #res_pred = res_pred.to(device)
    cov = torch.mean(res_true * res_pred)
    var_true = torch.mean(res_true**2)
    var_pred = torch.mean(res_pred**2)
    sigma_true = torch.sqrt(var_true)
    sigma_pred = torch.sqrt(var_pred)
    return - cov / (sigma_true * sigma_pred)


#print(encoded_seqs)
#y_preds = get_y_pred_scores(encoded_seqs, attention_masks)

def get_batches(a, batch_size=5):
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
    print(f'{orig_shape} -> {new_shape}  (dropped {remaining})')
    return torch.Tensor(a).to(device)
    

def test(xs, attns, scores, loss_fn, model):
    print('TESTING...')
    for i ,(xs_b, attns_b) in enumerate(tqdm(zip(xs, attns), total=len(xs))):
        xs_b, attns_b = xs_b.to(torch.int64), attns_b.to(torch.int64)
        with torch.no_grad():
            y_preds = get_y_pred_scores(xs_b, attns_b, model)
            if i == 0:
                y_preds_total = y_preds
            else:
                y_preds_total = torch.cat((y_preds_total, y_preds))
    loss = loss_fn(
        torch.flatten(scores), 
        torch.flatten(y_preds_total)
    )
    print(f'TESTING LOSS: {float(loss.cpu()):.3f}')
    return torch.flatten(scores), torch.flatten(y_preds_total)


def infer(xs, attns, model):
    for i ,(xs_b, attns_b) in enumerate(tqdm(zip(xs, attns), total=len(xs))):
        xs_b, attns_b = xs_b.to(torch.int64), attns_b.to(torch.int64)
        with torch.no_grad():
            y_preds = get_y_pred_scores(xs_b, attns_b, model)
            if i == 0:
                y_preds_total = y_preds
            else:
                y_preds_total = torch.cat((y_preds_total, y_preds))
    return torch.flatten(y_preds_total)


def train(xs, attns, scores, loss_fn, model, optimizer, n_epochs=3):
    for epoch in range(n_epochs):
        model.train()
        pbar = tqdm(zip(xs, attns, scores), total=len(xs))
        for batch, (xs_b, attns_b, scores_b) in enumerate(pbar):
            xs_b, attns_b = xs_b.to(torch.int64), attns_b.to(torch.int64)
            y_preds = get_y_pred_scores(xs_b, attns_b, model)
            scores_b = scores_b.to(device)
            loss = loss_fn(scores_b, y_preds)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            #saved_params = []
            #for i, (name, param) in enumerate(model.named_parameters()):  # 33 layers (0-32)
            #    if 'lora' in name:
            #        saved_params.append(torch.sum(param).clone())
            pbar.set_description(
                f"EPOCH: {epoch + 1}. Loss: {loss.item():>1f}  [batch: {batch+1}/{len(xs)}: {(batch + 1) * len(xs_b):>5d}/{len(xs)*len(xs_b)}]  "
                #f"(LoRA weight sum:{sum(saved_params):.3f})"
            )


def plot_true_preds(y_true, y_pred, muts):
    fig, ax = plt.subplots()
    plt.scatter(y_true, y_pred)
    for yt, yp, m in zip(y_true, y_pred, muts):
        if yt >= 1.0 * max(y_true):
            plt.text(yt, yp, m)
    plt.text(max(y_true), max(y_pred), f'{spearmanr(y_true, y_pred)[0]:.3f}')
    plt.show()
    plt.clf()


if __name__ == '__main__':
    df = pd.read_csv('CBPA2_HUMAN_Tsuboyama_2023_1O6X.csv', sep=',')
    print(df)
    muts_, seqs, scores = df['mutant'], df['mutated_sequence'], df['DMS_score']
    wt_seq = "VGDQVLEIVPSNEEQIKNLLQLEAQEHLQLDFWKSPTTPGETAHVRVPFVNVQAVKVFLESQGIAYSIMIED"
    seqs = seqs.to_list() #+ [wt_seq]
    scores_ = scores.to_list()  #+ [1.0] 
    print(len(seqs), len(scores))

    print(f"Using {device} device")

    basemodel = EsmForMaskedLM.from_pretrained(f'facebook/esm1v_t33_650M_UR90S_3')
    model_reg = EsmForMaskedLM.from_pretrained(f'facebook/esm1v_t33_650M_UR90S_3')
    tokenizer = EsmTokenizer.from_pretrained(f'facebook/esm1v_t33_650M_UR90S_3')

    peft_config = LoraConfig(r=8, target_modules=["query", "value"])
    model = get_peft_model(basemodel, peft_config)
    model = model.to(device)

    encoded_seqs_, attention_masks_ = get_encoded_seqs(seqs, tokenizer, max_length=len(wt_seq))
    #encoded_seqs, attention_masks, scores = encoded_seqs_[200:350], attention_masks_[200:350], scores_[200:350]
    #encoded_seqs_test, attention_masks_test, scores_test = encoded_seqs_[350:400], attention_masks_[350:400], scores_[350:400]
    encoded_seqs, encoded_seqs_test, attention_masks, attention_masks_test, scores, scores_test, muts, muts_test = train_test_split(
        encoded_seqs_, attention_masks_, scores_, muts_, train_size=0.2, shuffle = True, random_state=42)
    print('\n' + '-' * 60 + f'\nTrain size: {len(encoded_seqs)}, test size: {len(encoded_seqs_test)}')
    xs, attns, scores = get_batches(encoded_seqs), get_batches(attention_masks), get_batches(scores)
    xs_test, attns_test, scores_test = get_batches(encoded_seqs_test), get_batches(attention_masks_test), get_batches(scores_test)

    print('SHAPES:', np.shape(xs), np.shape(attns), np.shape(scores))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    #loss_fn = torch.nn.MSELoss()
    loss_fn = corr_loss


    print('\n\nPRE-TRAIN-PERFORMANCE')
    # INITIAL TEST
    y_true, y_pred = test(xs=xs, attns=attns, scores=scores, loss_fn=loss_fn, model=model)
    y_true_test, y_pred_test = test(xs_test, attns_test, scores_test, loss_fn, model)
    plot_true_preds(y_true_test.cpu(), y_pred_test.cpu(), muts_test)


    print('\n\nRE-TRAINING ESM1v...)')
    # TRAIN
    # https://stackoverflow.com/questions/56360644/pytorch-runtimeerror-expected-tensor-for-argument-1-indices-to-have-scalar-t
    train(xs, attns, scores, loss_fn, model, optimizer, n_epochs=3)


    # TEST
    print('\nPOST-TRAIN-PERFORMANCE')
    y_true, y_pred = test(xs=xs, attns=attns, scores=scores, loss_fn=loss_fn, model=model)
    y_true_test, y_pred_test = test(xs_test, attns_test, scores_test, loss_fn, model)
    plot_true_preds(y_true_test.cpu(), y_pred_test.cpu(), muts_test)

