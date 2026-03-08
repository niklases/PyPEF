
# Run me from parent dir:
#   Linux
#       export PYTHONPATH="${PYTHONPATH}:${PWD}" && python -m pytest ./tests/
#   Windows
#       $env:PYTHONPATH = "${PWD};${env:PYTHONPATH}";python -m pytest .\tests\


import os.path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
import torch
import gpytorch
from pypef.plm.utils import correlation_loss, hybrid_corr_mse_loss, pearson_loss, spearman_soft
import pytest

from pypef.ml.regression import AAIndexEncoding, full_aaidx_txt_path, get_regressor_performances
from pypef.dca.gremlin_inference import GREMLIN
from pypef.utils.variant_data import get_sequences_from_file, get_wt_sequence
from pypef.plm.inference import plm_inference, esm_setup, prosst_setup, tokenize_sequences
from pypef.hybrid.hybrid_model import DCALLMHybridModel
from pypef.plm.esm_lora_tune import get_esm_models
from pypef.plm.prosst_lora_tune import (
    get_prosst_models, get_structure_quantizied, 
    prosst_simple_vocab_aa_tokenizer
)
from pypef.gaussian_process.gauss_opt import get_gp_kernel_model
from pypef.utils.helpers import get_device

device = "cpu"  # get_device()
torch.manual_seed(42)
torch.cuda.manual_seed(42)
#torch.use_deterministic_algorithms(True)
np.random.seed(42)

msa_file_avgfp = os.path.abspath(os.path.join(
    __file__, '../../datasets/AVGFP/uref100_avgfp_jhmmer_119.a2m'
))

msa_file_aneh = os.path.abspath(
    os.path.join(__file__, '../../datasets/ANEH/ANEH_jhmmer.a2m'
))
pdb_file_aneh = os.path.abspath(os.path.join(
    __file__, '../../datasets/ANEH/AF-Q9UR30-F1-model_v4.pdb'
))
wt_seq_file_aneh = os.path.abspath(os.path.join(
    __file__, '../../datasets/ANEH/Sequence_WT_ANEH.fasta'
))
ls_b = os.path.abspath(os.path.join(
    __file__, '../../datasets/ANEH/LS_B.fasl'
))
ts_b = os.path.abspath(
    os.path.join(__file__, '../../datasets/ANEH/TS_B.fasl'
))

csv_blat_ecolx_stiffler2015 = os.path.abspath(
    os.path.join(__file__, '../../datasets/BLAT_ECOLX/BLAT_ECOLX_Stiffler_2015.csv'
))
pdb_blat_ecolx = os.path.abspath(
    os.path.join(__file__, '../../datasets/BLAT_ECOLX/BLAT_ECOLX.pdb'
))
wt_seq_file_blat_ecolx = os.path.abspath(
    os.path.join(__file__, '../../datasets/BLAT_ECOLX/blat_ecolx_wt.fasta'
))


train_seqs_aneh, _train_vars_aneh, train_ys_aneh = get_sequences_from_file(ls_b)
test_seqs_aneh, _test_vars_aneh, test_ys_aneh = get_sequences_from_file(ts_b)


def test_gremlin_avgfp():
    print("test_gremlin_avgfp()...")
    g = GREMLIN(
        alignment=msa_file_avgfp,
        char_alphabet="ARNDCQEGHILKMFPSTWYV-",
        wt_seq=None,
        optimize=True,
        gap_cutoff=0.5,
        eff_cutoff=0.8,
        opt_iter=100
    )
    wt_score = g.get_wt_score()  
    np.testing.assert_almost_equal(wt_score, 952.1102220697624, decimal=1)
    assert wt_score == g.wt_score == np.sum(g.x_wt)


def test_hybrid_model_dca_llm():
    print("test_hybrid_model_dca_llm()...")
    g = GREMLIN(
        alignment=msa_file_aneh,
        char_alphabet="ARNDCQEGHILKMFPSTWYV-",
        wt_seq=None,
        optimize=True,
        gap_cutoff=0.5,
        eff_cutoff=0.8,
        opt_iter=100
    )
    wt_score = g.get_wt_score()
    np.testing.assert_almost_equal(wt_score, 1743.2087199198131, decimal=1)
    assert wt_score == g.wt_score == np.sum(g.x_wt)
    y_pred = g.get_scores(np.append(train_seqs_aneh, test_seqs_aneh))
    np.testing.assert_almost_equal(
        spearmanr(np.append(train_ys_aneh, test_ys_aneh), y_pred)[0], 
        -0.5528510930046211, 
        decimal=7
    )
    x_dca_train = g.get_scores(train_seqs_aneh, encode=True)
    np.testing.assert_almost_equal(
        spearmanr(train_ys_aneh, np.sum(x_dca_train, axis=1))[0],
        -0.5556053466180598,
        decimal=7
    )
    assert len(train_seqs_aneh[0]) == len(g.wt_seq)
    aneh_wt_seq = get_wt_sequence(wt_seq_file_aneh)
    #y_pred_esm = inference(train_seqs_aneh, 'esm', wt_seq=aneh_wt_seq)
    print('len(aneh_wt_seq)', len(aneh_wt_seq))

    esm_base_model, _esm_lora_model, esm_tokenizer, _esm_optimizer = get_esm_models(
        model='facebook/esm1v_t33_650M_UR90S_3')
    esm_base_model.eval()
    esm_base_model = esm_base_model.to(device)
    x_esm, esm_attention_mask = tokenize_sequences(
        train_seqs_aneh, esm_tokenizer, max_length=len(aneh_wt_seq) + 2)
    # Tokenize WT sequence once
    wt_tokens, _ = tokenize_sequences(
            [aneh_wt_seq],
            esm_tokenizer,
            max_length=len(aneh_wt_seq) + 2
    )
    wt_tokens = torch.tensor(wt_tokens[0], dtype=torch.long)  # shape (L,)
    y_pred_esm = plm_inference(xs=x_esm, wt_input_ids=wt_tokens, 
                               attention_mask=esm_attention_mask, model=esm_base_model,
                               device=device).cpu()
    np.testing.assert_almost_equal(
        spearmanr(train_ys_aneh, y_pred_esm)[0], 
         -0.713214007088901, 
        decimal=7
    )

    #y_pred_prosst = inference(
    #    train_seqs_aneh, 'prosst', 
    #    pdb_file=pdb_file_aneh, wt_seq=aneh_wt_seq
    #)
    prosst_base_model, prosst_lora_model, prosst_tokenizer, prosst_optimizer = get_prosst_models()
    prosst_base_model.eval()
    prosst_vocab = prosst_tokenizer.get_vocab()
    prosst_base_model = prosst_base_model.to(device)
    wt_input_ids, prosst_attention_mask, wt_structure_input_ids = get_structure_quantizied(
        pdb_file_aneh, prosst_tokenizer, aneh_wt_seq, device=device)
    x_prosst, prosst_attention_mask_ = tokenize_sequences(
        sequences=train_seqs_aneh, 
        tokenizer=prosst_tokenizer, 
        max_length=len(aneh_wt_seq) + 2
    )
    y_pred_prosst = plm_inference(xs=x_prosst, wt_input_ids=wt_input_ids, 
                                  attention_mask=prosst_attention_mask, model=prosst_base_model, 
                                  wt_structure_input_ids=wt_structure_input_ids, device=device).cpu()

    # TODO: Check reproducibility on different devices and machines
    np.testing.assert_almost_equal(
        spearmanr(train_ys_aneh, y_pred_prosst)[0], 
        [-0.5022957688493356, -0.7425657069861902][0], 
        decimal=7
    )

    x_dca_test = g.get_scores(test_seqs_aneh, encode=True)
    for i, setup in enumerate([esm_setup, prosst_setup]):
        print(['~~~ ESM ~~~', '~~~ ProSST ~~~'][i])
        if setup == esm_setup:
            llm_dict = setup(sequences=train_seqs_aneh, wt_seq=aneh_wt_seq)
        else:  # elif setup == prosst_setup:
            llm_dict = setup(
                aneh_wt_seq, pdb_file_aneh, sequences=train_seqs_aneh)
        x_llm_test, _ = tokenize_sequences(test_seqs_aneh, llm_dict[['esm1v', 'prosst'][i]]['llm_tokenizer'])
        hm = DCALLMHybridModel(
            x_train_dca=np.array(x_dca_train), 
            y_train=train_ys_aneh,
            llm_model_input=llm_dict,
            x_wt=g.x_wt,
            seed=42,
            device=device
        )

        y_pred_test = hm.hybrid_prediction(x_dca=x_dca_test, x_llm=x_llm_test)
        print(hm.beta1, hm.beta2, hm.beta3, hm.beta4, hm.ridge_opt)
        print('hm.y_dca_ttest:', spearmanr(hm.y_ttest, hm.y_dca_ttest), len(hm.y_ttest))
        print('hm.y_dca_ridge_ttest:', spearmanr(hm.y_ttest, hm.y_dca_ridge_ttest), len(hm.y_ttest))
        print('hm.y_llm_ttest:', spearmanr(hm.y_ttest, hm.y_llm_ttest), len(hm.y_ttest))
        print('hm.y_llm_lora_ttest:', spearmanr(hm.y_ttest, hm.y_llm_lora_ttest), len(hm.y_ttest))
        print('Hybrid prediction:', spearmanr(test_ys_aneh, y_pred_test), len(test_ys_aneh))
        np.testing.assert_almost_equal(
            spearmanr(hm.y_ttest, hm.y_dca_ttest)[0], -0.5342743713116743, 
            decimal=7
        )
        np.testing.assert_almost_equal(
            spearmanr(hm.y_ttest, hm.y_dca_ridge_ttest)[0], 0.717333573331078, 
            decimal=7
        )
        np.testing.assert_almost_equal(
            spearmanr(hm.y_ttest, hm.y_llm_ttest)[0], 
            [-0.7704181041760417,        # TODO: Check on different machines (CPU vs CUDA)
             -0.6370803136561448][i],    # Use same loss function, e.g. Spearman!
            decimal=7
        )  
        # Nondeterministic behavior (without setting seed), should be about ~0.7 to ~0.9, 
        # but as sample size is so low the following is only checking if not NaN / >=-1.0 and <=1.0,
        # Torch reproducibility documentation: https://pytorch.org/docs/stable/notes/randomness.html
        # https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
        assert -1.0 <= spearmanr(hm.y_ttest, hm.y_llm_lora_ttest)[0] <= 1.0  
        assert -1.0 <= spearmanr(test_ys_aneh, y_pred_test)[0] <= 1.0
        #np.testing.assert_almost_equal(spearmanr(test_ys_aneh, y_pred_test)[0], 0.8403249605842074, decimal=7)  # 0.814064805565951
        # With seed 42 for numpy and torch for implemented LLM's and on local machine:
        if setup == esm_setup:
            continue  # TODO: Make new/overloaded pytest decorator function
            #try: # Different values on different machines (TODO) has to be investigated
            #    np.testing.assert_almost_equal(
            #        spearmanr(hm.y_ttest, hm.y_llm_lora_ttest)[0], 0.7772102863835341, decimal=7
            #    )
            #except AssertionError as ae1:
            #    try:
            #        np.testing.assert_almost_equal(
            #            spearmanr(hm.y_ttest, hm.y_llm_lora_ttest)[0], 0.7239938685054149, decimal=7
            #        )
            #    except AssertionError as ae2:
            #        raise AssertionError(
            #            f"Neither condition passed:\nFirst comparison failed:\n{ae1}\n"
            #            f"Second comparison failed:\n{ae2}"
            #        )
            #try:
            #    np.testing.assert_almost_equal(
            #        spearmanr(test_ys_aneh, y_pred_test)[0], 0.8004896406836318, decimal=7
            #    )
            #except AssertionError as ae1:
            #    try:
            #        np.testing.assert_almost_equal(
            #            spearmanr(test_ys_aneh, y_pred_test)[0], 0.8338711936729409, decimal=7
            #        )
            #    except AssertionError as ae2:
            #        raise AssertionError(
            #            f"Neither condition passed:\nFirst comparison failed:\n{ae1}\n"
            #            f"Second comparison failed:\n{ae2}"
            #        )

        elif setup == prosst_setup:
            continue
            #try:
            #    np.testing.assert_almost_equal(
            #        spearmanr(hm.y_ttest, hm.y_llm_lora_ttest)[0], 0.7770124558338013, decimal=7
            #    )
            #except AssertionError as ae1:
            #    try: 
            #        np.testing.assert_almost_equal(  
            #        spearmanr(hm.y_ttest, hm.y_llm_lora_ttest)[0], 0.7239938685054149, decimal=7
            #        )                                
            #    except AssertionError as ae2:
            #        raise AssertionError(
            #            f"Neither condition passed:\nFirst comparison failed:\n{ae1}\n"
            #            f"Second comparison failed:\n{ae2}"
            #        )
            #np.testing.assert_almost_equal(
            #    spearmanr(test_ys_aneh, y_pred_test)[0], 0.8291977762544377, decimal=7
            #)


def test_dataset_b_results():
    print("test_dataset_b_results()...")
    aaindex = "WOLR810101.txt"
    x_fft_train, _ = AAIndexEncoding(
        full_aaidx_txt_path(aaindex), train_seqs_aneh
    ).collect_encoded_sequences()
    x_fft_test, _ = AAIndexEncoding(
        full_aaidx_txt_path(aaindex), test_seqs_aneh
    ).collect_encoded_sequences()
    performances = get_regressor_performances(
        x_learn=x_fft_train,
        x_test=x_fft_test,
        y_learn=train_ys_aneh,
        y_test=test_ys_aneh,
        regressor='pls_loocv'
    )  
    # Dataset B PLS_LOOCV results: R², RMSE, NRMSE, Pearson's r, Spearman's rho 
    # RMSE, in Python 3.10 14.669 and from Python 3.11 on 14.17:
    np.testing.assert_almost_equal(performances[1], 14.48, decimal=0)
    # R²
    np.testing.assert_almost_equal(performances[0], 0.72, decimal=2)
    #  NRMSE, Pearson's r, Spearman's rho
    np.testing.assert_almost_equal(performances[2:5], [0.52, 0.86, 0.89], decimal=2)


@pytest.mark.requires_gpu
def test_plm_corr_blat_ecolx():
    print("test_plm_corr_blat_ecolx()...")
    print("Device", device)
    blat_ecolx_wt_seq = get_wt_sequence(wt_seq_file_blat_ecolx)
    prosst_base_model, prosst_lora_model, prosst_tokenizer, prosst_optimizer = get_prosst_models()
    prosst_vocab = prosst_tokenizer.get_vocab()
    prosst_base_model = prosst_base_model.to(device)
    df = pd.read_csv(csv_blat_ecolx_stiffler2015)
    sequences = df['mutated_sequence'].to_list()
    y_true = df['DMS_score'].to_list()
    for x in ['facebook/esm1v_t33_650M_UR90S_3']:
        esm_base_model, _esm_lora_model, esm_tokenizer, esm_optimizer = get_esm_models(model=x)
        esm_base_model = esm_base_model.to(device)
        x_esm, esm_attention_mask = tokenize_sequences(
            sequences, esm_tokenizer, max_length=len(blat_ecolx_wt_seq) + 2)
        # Tokenize WT sequence once
        wt_tokens, _ = tokenize_sequences(
            [blat_ecolx_wt_seq],
            esm_tokenizer,
            max_length=len(blat_ecolx_wt_seq) + 2
        )
        wt_tokens = torch.tensor(wt_tokens[0], dtype=torch.long)  # shape (L,)
        
        y_esm = plm_inference(
            xs=x_esm,
            wt_input_ids=wt_tokens,
            attention_mask=esm_attention_mask,
            model=esm_base_model,
            mask_token_id=esm_tokenizer.mask_token_id,
            inference_type='mutation-masking',
            batch_size=5,
            train=False,
            device=device,
            verbose=True
        )
        print(f'{x}: ESM1v (unsupervised performance mutation-masking): '  
              f'{spearmanr(y_true, y_esm.cpu())[0]}')
        np.testing.assert_almost_equal(spearmanr(y_true, y_esm.cpu())[0], 0.6367826285982324, decimal=6)
        
        y_esm = plm_inference(
            xs=x_esm,
            wt_input_ids=wt_tokens,
            attention_mask=esm_attention_mask,
            model=esm_base_model,
            mask_token_id=None, # do not define for unmasked
            inference_type='wt-marginal',
            batch_size=5,
            train=False,
            verbose=True
        )
        print(f'{x}: ESM1v (unsupervised performance wt-marginal): '  
              f'{spearmanr(y_true, y_esm.cpu())[0]}')
        np.testing.assert_almost_equal(spearmanr(y_true, y_esm.cpu())[0], 0.6498987261125897, decimal=6)

        y_esm = plm_inference(
            xs=x_esm,
            wt_input_ids=wt_tokens,
            attention_mask=esm_attention_mask,
            model=esm_base_model,
            mask_token_id=None, # do not define for unmasked
            inference_type='full-sequence',
            batch_size=5,
            train=False,
            verbose=True
        )
        print(f'{x}: ESM1v (unsupervised performance full-sequence): '  
              f'{spearmanr(y_true, y_esm.cpu())[0]}')
        np.testing.assert_almost_equal(spearmanr(y_true, y_esm.cpu())[0], 0.6400694954450116, decimal=6)
        
        #y_esm = plm_inference(
        #    xs=x_esm,
        #    wt_input_ids=wt_tokens,
        #    attention_mask=esm_attention_mask,
        #    model=esm_base_model,
        #    mask_token_id=esm_tokenizer.mask_token_id,
        #    inference_type='full-masking',
        #    batch_size=5,
        #    train=False,
        #    verbose=True
        #)
        #print(f'{x}: ESM1v (unsupervised performance): '  
        #      f'{spearmanr(y_true, y_esm.cpu())[0]}')
        #np.testing.assert_almost_equal(spearmanr(y_true, y_esm.cpu())[0], 0.666666666666666, decimal=6)
    wt_input_ids, prosst_attention_mask, wt_structure_input_ids = get_structure_quantizied(
        pdb_blat_ecolx, prosst_tokenizer, blat_ecolx_wt_seq, device=device)
    x_prosst2 = prosst_simple_vocab_aa_tokenizer(sequences, prosst_vocab)
    x_prosst, prosst_attention_mask_ = tokenize_sequences(
        sequences=sequences, 
        tokenizer=prosst_tokenizer, 
        max_length=len(blat_ecolx_wt_seq) + 2
    )
    assert x_prosst[0][1:-1] == x_prosst2.tolist()[0][1:-1], (
        f"{x_prosst[0][1:-1]} != {x_prosst2.tolist()[0][1:-1]}")
    assert prosst_attention_mask.tolist()[0] == prosst_attention_mask_, (
        f"{prosst_attention_mask.tolist()[0]} != {prosst_attention_mask_}")

    y_prosst = plm_inference(
            xs=x_prosst,
            wt_input_ids=wt_input_ids,
            attention_mask=prosst_attention_mask,
            model=prosst_base_model,
            mask_token_id=prosst_tokenizer.mask_token_id,
            inference_type='mutation-masking',
            wt_structure_input_ids=wt_structure_input_ids,
            batch_size=5,
            train=False,
            verbose=True   
    )
    print(f'ProSST (unsupervised performance mutation-masking): '  # ProSST not made/trained for MLM: 0.607137337377509
          f'{spearmanr(y_true, y_prosst.cpu())[0]}')
    np.testing.assert_almost_equal(spearmanr(y_true, y_prosst.cpu())[0], 0.607137337377509, decimal=6)

    y_prosst = plm_inference(
            xs=x_prosst,
            wt_input_ids=wt_input_ids,
            attention_mask=prosst_attention_mask,
            model=prosst_base_model,
            mask_token_id=None,  # do not define for unmasked
            inference_type='wt-marginal',
            wt_structure_input_ids=wt_structure_input_ids,
            batch_size=5,
            train=False,
            verbose=True        
    )
    print(f'ProSST (unsupervised performance wt-marginal): '  # ProteinGym: ProSST: 0.760
          f'{spearmanr(y_true, y_prosst.cpu())[0]}')
    np.testing.assert_almost_equal(spearmanr(y_true, y_prosst.cpu())[0], 0.7430279087189432, decimal=6)

    y_prosst = plm_inference(
            xs=x_prosst,
            wt_input_ids=wt_input_ids,
            attention_mask=prosst_attention_mask,
            model=prosst_base_model,
            mask_token_id=None,  # do not define for unmasked
            inference_type='full-sequence',
            wt_structure_input_ids=wt_structure_input_ids,
            batch_size=5,
            train=False,
            verbose=True        
    )
    print(f'ProSST (unsupervised performance full-sequence): '
          f'{spearmanr(y_true, y_prosst.cpu())[0]}')
    np.testing.assert_almost_equal(spearmanr(y_true, y_prosst.cpu())[0], 0.5656131250565296, decimal=6)

    #y_prosst = plm_inference(
    #        xs=x_prosst,
    #        wt_input_ids=wt_input_ids,
    #        attention_mask=prosst_attention_mask,
    #        model=prosst_base_model,
    #        mask_token_id=prosst_tokenizer.mask_token_id,
    #        inference_type='full-masking',
    #        wt_structure_input_ids=wt_structure_input_ids,
    #        batch_size=5,
    #        train=False,
    #        verbose=True        
    #)
    #print(f'ProSST (unsupervised performance): '  # ProteinGym: ProSST: 0.760
    #      f'{spearmanr(y_true, y_prosst.cpu())[0]}')


def test_gaussian_process_opt():
    print("test_gaussian_process_opt()...")
    df = pd.read_csv(csv_blat_ecolx_stiffler2015)
    mutants = df['mutant'].to_list()
    sequences = df['mutated_sequence'].to_list()
    y = df['DMS_score'].to_list()
    m_train, m_test, s_train, s_test, y_train, y_test = train_test_split(
        mutants, sequences, y, test_size=0.80, random_state=42
    )
    print("Getting ProSST models")
    pdb = 'datasets/BLAT_ECOLX/BLAT_ECOLX.pdb'
    wt_seq = get_wt_sequence('datasets/BLAT_ECOLX/blat_ecolx_wt.fasta')
    prosst_base_model, _prosst_lora_model, prosst_tokenizer, _prosst_optimizer = get_prosst_models()
    prosst_base_model = prosst_base_model.to(device)

    esm_base_model, _esm_lora_model, esm_tokenizer, _esm_optimizer = get_esm_models()

    wt_prosst_input_ids, prosst_attention_mask, wt_structure_input_ids = get_structure_quantizied(
            pdb, prosst_tokenizer, wt_seq, verbose=True
    )

    wt_esm_input_ids, _esm_attention_mask = tokenize_sequences([wt_seq], esm_tokenizer)
    wt_esm_input_ids = torch.tensor( wt_esm_input_ids[0], dtype=torch.long)  # shape (L,)

    x_prosst_tok_train, _prosst_attention_mask = tokenize_sequences(s_train, prosst_tokenizer)
    x_prosst_emb_train = plm_inference(x_prosst_tok_train, wt_prosst_input_ids, prosst_attention_mask, prosst_base_model, 
                                       extract_emb=True, wt_structure_input_ids=wt_structure_input_ids).cpu()

    x_esm_tok_train, esm_attention_mask = tokenize_sequences(s_train, esm_tokenizer)
    x_esm_emb_train = plm_inference(x_esm_tok_train, wt_esm_input_ids, esm_attention_mask, 
                                    esm_base_model, extract_emb=True).cpu()

    y_train = torch.tensor(y_train).float()
    y_test = torch.tensor(y_test).float()

    # Concatenate features, necessary as GPkernel does not accept a tuple as input
    x_combined_train = torch.cat([x_prosst_emb_train, x_esm_emb_train], dim=-1) 

    # Test
    # -----------------------------
    x_prosst_tok_test, _prosst_attention_mask = tokenize_sequences(s_test, prosst_tokenizer)
    x_prosst_emb_test = plm_inference(x_prosst_tok_test, wt_prosst_input_ids, prosst_attention_mask, prosst_base_model, 
                                      extract_emb=True, wt_structure_input_ids=wt_structure_input_ids).to(device)

    x_esm_tok_test, esm_attention_mask = tokenize_sequences(s_test, esm_tokenizer)
    x_esm_emb_test = plm_inference(x_esm_tok_test, wt_esm_input_ids, esm_attention_mask, 
                                    esm_base_model, extract_emb=True).to(device)

    x_combined_test = torch.cat([x_prosst_emb_test, x_esm_emb_test], dim=-1)

    esm_model = get_gp_kernel_model(x_esm_emb_train, y_train, train=True).to(device)
    prosst_model = get_gp_kernel_model(x_prosst_emb_train, y_train, train=True).to(device)
    comb_model = get_gp_kernel_model(x_combined_train, y_train, train=True).to(device)

    for i, (model, x_test) in enumerate(
        zip(
            [esm_model, prosst_model, comb_model], 
            [x_esm_emb_test, x_prosst_emb_test, x_combined_test]
        )
    ):
        print("~~~ " + ["ESM", "ProSST", "ESM + ProSST combined "][i] + "GP Test ~~~")
        likelihood = model.likelihood
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = likelihood(model(x_test))
            y_pred = pred.mean

        rho = spearmanr(y_test, y_pred.cpu().numpy())[0]
        print("Spearman rho SciPy TEST:                   ", rho)
        print("Spearman soft TEST:                        ", spearman_soft(y_test, y_pred).item())
        print("Correlation loss Spearman TEST:            ", correlation_loss(y_test, y_pred, method="spearman"))
        print("Correlation hybrid MSE loss Spearman TEST: ", hybrid_corr_mse_loss(y_test, y_pred))
        print("Correlation loss Pearson TEST:             ", correlation_loss(y_test, y_pred, method="pearson"))
        print("Correlation loss Pearson 2 TEST:           ", pearson_loss(y_test, y_pred))
        np.testing.assert_almost_equal(
            rho, 
            [0.8213561923449809,  0.7873173926713299, 0.8360338429235056][i], 
            decimal=3
        )


if __name__ == "__main__":
    #test_gremlin_avgfp()
    #test_hybrid_model_dca_llm()
    #test_dataset_b_results()
    #test_plm_corr_blat_ecolx()
    test_gaussian_process_opt()
    