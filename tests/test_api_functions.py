
# Linux
#     Run me from parent dir:
#         export PYTHONPATH="${PYTHONPATH}:${PWD}" && python -m pytest tests/

import os.path

import numpy as np
from scipy.stats import spearmanr
import torch

from pypef.ml.regression import AAIndexEncoding, full_aaidx_txt_path, get_regressor_performances
from pypef.dca.gremlin_inference import GREMLIN
from pypef.utils.variant_data import get_sequences_from_file
from pypef.llm.esm_lora_tune import get_esm_models, get_encoded_seqs, corr_loss, get_batches, esm_test
from pypef.hybrid.hybrid_model import DCAESMHybridModel



msa_file_avgfp = os.path.abspath(
    os.path.join(
        os.path.abspath(__file__), 
        '../../datasets/AVGFP/uref100_avgfp_jhmmer_119.a2m'
    )
)


msa_file_aneh = os.path.abspath(
    os.path.join(
        os.path.abspath(__file__), 
        '../../datasets/ANEH/ANEH_jhmmer.a2m'
    )
)

ls_b = os.path.abspath(
    os.path.join(
        os.path.abspath(__file__), 
        '../../datasets/ANEH/LS_B.fasl'
    )
)

ts_b = os.path.abspath(
    os.path.join(
        os.path.abspath(__file__), 
        '../../datasets/ANEH/TS_B.fasl'
    )
)


train_seqs, _train_vars, train_ys = get_sequences_from_file(ls_b)
test_seqs, _test_vars, test_ys = get_sequences_from_file(ts_b)


def test_gremlin():
    g = GREMLIN(
        alignment=msa_file_avgfp,
        char_alphabet="ARNDCQEGHILKMFPSTWYV-",
        wt_seq=None,
        optimize=True,
        gap_cutoff=0.5,
        eff_cutoff=0.8,
        opt_iter=100
    )
    wt_score = g.get_wt_score()  # only 1 decimal place for TensorFlow result
    np.testing.assert_almost_equal(wt_score, 952.1, decimal=1)
    y_pred = g.get_scores(np.append(train_seqs, test_seqs))
    np.testing.assert_almost_equal(
        spearmanr(np.append(train_ys, test_ys), y_pred)[0], 
        0.4516502675400598, 
        decimal=3
    )


def test_hybrid_model():
    g = GREMLIN(
        alignment=msa_file_aneh,
        char_alphabet="ARNDCQEGHILKMFPSTWYV-",
        wt_seq=None,
        optimize=True,
        gap_cutoff=0.5,
        eff_cutoff=0.8,
        opt_iter=100
    )
    x_dca_train = g.get_scores(train_seqs, encode=True)
    print(spearmanr(
        train_ys,
        np.sum(x_dca_train, axis=1)
    ), len(train_ys))

    print(len(train_seqs[0]), train_seqs[0])
    assert len(train_seqs[0]) == len(g.wt_seq)
    base_model, lora_model, tokenizer, optimizer = get_esm_models()
    encoded_seqs_train, attention_masks_train = get_encoded_seqs(list(train_seqs), tokenizer, max_length=len(train_seqs[0]))
    x_esm_b, attention_masks_b = get_batches(encoded_seqs_train), get_batches(attention_masks_train)
    y_true, y_pred_esm = esm_test(x_esm_b, attention_masks_b, torch.Tensor(train_ys), loss_fn=corr_loss, model=base_model)
    print(spearmanr(
        y_true,
        y_pred_esm
    ), len(y_true))

    hm = DCAESMHybridModel(
        x_train_dca=np.array(x_dca_train), 
        x_train_esm=np.array(encoded_seqs_train), 
        x_train_esm_attention_masks=np.array(attention_masks_train), 
        y_train=train_ys,
        esm_model=lora_model,
        esm_base_model=base_model,
        esm_optimizer=optimizer,
        x_wt=g.x_wt,
        seed=42
    )

    x_dca_test = g.get_scores(test_seqs, encode=True)
    encoded_seqs_test, attention_masks_test = get_encoded_seqs(list(test_seqs), tokenizer, max_length=len(test_seqs[0]))
    y_pred_test = hm.hybrid_prediction(x_dca=x_dca_test, x_esm=encoded_seqs_test, attns_esm=attention_masks_test)
    print(spearmanr(test_ys, y_pred_test), len(test_ys))
    # Torch reproducibility documentation: https://pytorch.org/docs/stable/notes/randomness.html
    assert 0.70 < spearmanr(test_ys, y_pred_test)[0]  # Nondeterministic behavior, should be about ~ 0.8


def test_dataset_b_results():
    aaindex = "WOLR810101.txt"
    x_fft_train, _ = AAIndexEncoding(full_aaidx_txt_path(aaindex), train_seqs).collect_encoded_sequences()
    x_fft_test, _ = AAIndexEncoding(full_aaidx_txt_path(aaindex), test_seqs).collect_encoded_sequences()
    performances = get_regressor_performances(
        x_learn=x_fft_train, 
        x_test=x_fft_test, 
        y_learn=train_ys, 
        y_test=test_ys, 
        regressor='pls_loocv'
    )  
    # Dataset B PLS_LOOCV results: R², RMSE, NRMSE, Pearson's r, Spearman's rho 
    # RMSE, in Python 3.10 14.669 and from Python 3.11 on 14.17:
    np.testing.assert_almost_equal(performances[1], 14.48, decimal=0)
    # R²
    np.testing.assert_almost_equal(performances[0], 0.72, decimal=2)
    #  NRMSE, Pearson's r, Spearman's rho
    np.testing.assert_almost_equal(performances[2:5], [0.52, 0.86, 0.89], decimal=2)
