
# https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
# https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
# https://neptune.ai/blog/pytorch-loss-functions

# https://github.com/luo-group/ConFit


import torch
import numpy as np
import pandas as pd
from peft import LoraConfig, get_peft_model
from transformers import EsmForMaskedLM, EsmTokenizer
from scipy.stats import spearmanr

import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def get_encoded_seqs(sequences, tokenizer, max_length=104):
    encoded_sequences, attention_masks = tokenizer(
        sequences, 
        padding='max_length', 
        truncation=True, 
        max_length=max_length
    ).values()
    return encoded_sequences, attention_masks

def get_y_pred_scores(encoded_sequences, attention_masks, model):
    device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available()else "cpu")
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


def get_batches(a, batch_size=5, verbose: bool = False):
    device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
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
    return torch.flatten(scores).detach().cpu(), torch.flatten(y_preds_total).detach().cpu()


def infer(xs, attns, model, desc: None | str = None):
    for i ,(xs_b, attns_b) in enumerate(tqdm(zip(xs, attns), total=len(xs), desc=desc)):
        xs_b, attns_b = xs_b.to(torch.int64), attns_b.to(torch.int64)
        with torch.no_grad():
            y_preds = get_y_pred_scores(xs_b, attns_b, model)
            if i == 0:
                y_preds_total = y_preds
            else:
                y_preds_total = torch.cat((y_preds_total, y_preds))
    return torch.flatten(y_preds_total)


def train(xs, attns, scores, loss_fn, model, optimizer, n_epochs=3):
    device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    pbar_epochs = tqdm(range(1, n_epochs + 1))
    for epoch in pbar_epochs:
        pbar_epochs.set_description(f'EPOCH {epoch}/{n_epochs}')
        model.train()
        pbar_batches = tqdm(zip(xs, attns, scores), total=len(xs), leave=False)
        for batch, (xs_b, attns_b, scores_b) in enumerate(pbar_batches):
            xs_b, attns_b = xs_b.to(torch.int64), attns_b.to(torch.int64)
            #print(xs_b.size(), attns_b.size(), scores_b.size())
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
            pbar_batches.set_description(
                f"EPOCH: {epoch}. Loss: {loss.item():>1f}  "
                f"[batch: {batch+1}/{len(xs)}: {(batch + 1) * len(xs_b):>5d}/{len(xs) * len(xs_b)}]  "
                #f"(LoRA weight sum:{sum(saved_params):.3f})"
            )
            #scores_b.detach().cpu()
        

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
    # Get cpu, gpu or mps device for training.
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"esm1v_contrastive_learning.py: Using {device} device")
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

