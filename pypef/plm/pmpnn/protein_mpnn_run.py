import os.path
import json, time, os, sys
import numpy as np
import pandas as pd
import torch
import copy
import random
import os.path
import subprocess

from pypef.utils.helpers import tqdm
from pypef.utils.helpers import get_device
from pypef.plm.pmpnn.protein_mpnn_utils import (
    pmpnn_scores, pmpnn_aaidx_to_seq, tied_featurize, parse_pdb
)
from pypef.plm.pmpnn.protein_mpnn_utils import (
    StructureDataset, StructureDatasetPDB, ProteinMPNN
)

import logging
logger = logging.getLogger(__name__)


def run_pmpnn(
        input_seqs: str | os.PathLike,
        pdb_path: str | os.PathLike,
        pdb_path_chains: str,
        path_to_model_weights: str | os.PathLike,
        out_folder: str | os.PathLike,
        jsonl_path: str | os.PathLike = "",
        score_only: bool = True,
        ca_only: bool = False,
        save_npz_files: bool = False,
        num_seq_per_target: int = 5,
        sampling_temp: float = 0.1,
        model_name: str = "v_48_020",
        use_soluble_model: bool= False,  # str = "store_true",
        omit_AAs: str = 'X',
        seed: int | None = None,
        batch_size: int = 1,
        max_length: int = 20000,
        chain_id_jsonl: str = "",
        fixed_positions_jsonl: str = "",
        pssm_jsonl: str = "",
        omit_AA_jsonl: str = "",
        bias_AA_jsonl: str = "",
        bias_by_res_jsonl: str = "",
        tied_positions_jsonl: str = "",
        backbone_noise: float = 0.0,
        pssm_multi: float = 0.0,
        pssm_threshold: float = 0.0,
        pssm_log_odds_flag: int = 0,
        pssm_bias_flag: int = 0,
        conditional_probs_only: int = 0,
        conditional_probs_only_backbone: int = 0,
        unconditional_probs_only: int = 0,
        suppress_print: int = 1
):
    if not seed:
        seed=int(np.random.randint(0, high=999, size=1, dtype=int)[0])
    sampling_temp = str(sampling_temp)

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)   
    
    hidden_dim = 128
    num_layers = 3 

    result_dict = {}
  

    if path_to_model_weights:
        model_folder_path = os.path.abspath(path_to_model_weights)
    else: 
        dir_path = os.path.dirname(__file__)
        if ca_only:
            logger.info("Using CA-ProteinMPNN!")
            model_folder_path = os.path.join(dir_path, 'ca_model_weights')
            if use_soluble_model:
                logger.warning("WARNING: CA-SolubleMPNN is not available yet")
                sys.exit()
        else:
            if use_soluble_model:
                logger.info("Using ProteinMPNN trained on soluble proteins only!")
                model_folder_path = os.path.join(dir_path, 'soluble_model_weights')
            else:
                model_folder_path = os.path.join(dir_path, 'vanilla_model_weights')
    checkpoint_path = os.path.join(model_folder_path, f'{model_name}.pt')
    logger.info(f"Loading ProteinMPNN model from {checkpoint_path}")
    folder_for_outputs = out_folder
    
    NUM_BATCHES = num_seq_per_target // batch_size
    BATCH_COPIES = batch_size
    temperatures = [float(item) for item in sampling_temp.split()]
    omit_AAs_list = omit_AAs
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    alphabet_dict = dict(zip(alphabet, range(21)))    
    print_all = suppress_print == 0 
    omit_AAs_np = np.array([AA in omit_AAs_list for AA in alphabet]).astype(np.float32)
    device = get_device()
    logger.info(f"Device: {device.upper()}")
    if os.path.isfile(chain_id_jsonl):
        with open(chain_id_jsonl, 'r') as json_file:
            json_list = list(json_file)
        for json_str in json_list:
            chain_id_dict = json.loads(json_str)
    else:
        chain_id_dict = None
        if print_all:
            logger.info(40*'-')
            logger.info('chain_id_jsonl is NOT loaded')
        
    if os.path.isfile(fixed_positions_jsonl):
        with open(fixed_positions_jsonl, 'r') as json_file:
            json_list = list(json_file)
        for json_str in json_list:
            fixed_positions_dict = json.loads(json_str)
    else:
        if print_all:
            logger.info(40*'-')
            logger.info('fixed_positions_jsonl is NOT loaded')
        fixed_positions_dict = None
    
    
    if os.path.isfile(pssm_jsonl):
        with open(pssm_jsonl, 'r') as json_file:
            json_list = list(json_file)
        pssm_dict = {}
        for json_str in json_list:
            pssm_dict.update(json.loads(json_str))
    else:
        if print_all:
            logger.info(40*'-')
            logger.info('pssm_jsonl is NOT loaded')
        pssm_dict = None
    
    
    if os.path.isfile(omit_AA_jsonl):
        with open(omit_AA_jsonl, 'r') as json_file:
            json_list = list(json_file)
        for json_str in json_list:
            omit_AA_dict = json.loads(json_str)
    else:
        if print_all:
            logger.info(40*'-')
            logger.info('omit_AA_jsonl is NOT loaded')
        omit_AA_dict = None
    
    
    if os.path.isfile(bias_AA_jsonl):
        with open(bias_AA_jsonl, 'r') as json_file:
            json_list = list(json_file)
        for json_str in json_list:
            bias_AA_dict = json.loads(json_str)
    else:
        if print_all:
            logger.info(40*'-')
            logger.info('bias_AA_jsonl is NOT loaded')
        bias_AA_dict = None
    
    
    if os.path.isfile(tied_positions_jsonl):
        with open(tied_positions_jsonl, 'r') as json_file:
            json_list = list(json_file)
        for json_str in json_list:
            tied_positions_dict = json.loads(json_str)
    else:
        if print_all:
            logger.info(40*'-')
            logger.info('tied_positions_jsonl is NOT loaded')
        tied_positions_dict = None

    
    if os.path.isfile(bias_by_res_jsonl):
        with open(bias_by_res_jsonl, 'r') as json_file:
            json_list = list(json_file)
    
        for json_str in json_list:
            bias_by_res_dict = json.loads(json_str)
        if print_all:
            logger.info('bias by residue dictionary is loaded')
    else:
        if print_all:
            logger.info(40*'-')
            logger.info('bias by residue dictionary is not loaded, or not provided')
        bias_by_res_dict = None
   

    if print_all: 
        logger.info(40*'-')
    bias_AAs_np = np.zeros(len(alphabet))
    if bias_AA_dict:
            for n, AA in enumerate(alphabet):
                    if AA in list(bias_AA_dict.keys()):
                            bias_AAs_np[n] = bias_AA_dict[AA]
    
    if pdb_path:
        pdb_dict_list = parse_pdb(pdb_path, ca_only=ca_only)
        dataset_valid = StructureDatasetPDB(pdb_dict_list, truncate=None, max_length=max_length)
        all_chain_list = [item[-1:] for item in list(pdb_dict_list[0]) if item[:9]=='seq_chain'] #['A','B', 'C',...]
        if pdb_path_chains:
            designed_chain_list = [str(item) for item in pdb_path_chains.split()]
        else:
            designed_chain_list = all_chain_list
        fixed_chain_list = [letter for letter in all_chain_list if letter not in designed_chain_list]
        chain_id_dict = {}
        chain_id_dict[pdb_dict_list[0]['name']]= (designed_chain_list, fixed_chain_list)
    else:
        dataset_valid = StructureDataset(jsonl_path, truncate=None, max_length=max_length, verbose=print_all)

    checkpoint = torch.load(checkpoint_path, map_location=device) 
    noise_level_print = checkpoint['noise_level']
    model = ProteinMPNN(
        ca_only=ca_only, 
        num_letters=21, 
        node_features=hidden_dim, 
        edge_features=hidden_dim, 
        hidden_dim=hidden_dim, 
        num_encoder_layers=num_layers, 
        num_decoder_layers=num_layers, 
        augment_eps=backbone_noise, 
        k_neighbors=checkpoint['num_edges'])
    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    if print_all:
        logger.info(40*'-')
        logger.info('%s %s', 'Number of edges:', checkpoint['num_edges'])
        logger.info(f'Training noise level: {noise_level_print}A')
 
    # Build paths for experiment
    base_folder = folder_for_outputs
    if base_folder[-1] != '/':
        base_folder = base_folder + '/'
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    
    if not os.path.exists(base_folder + 'seqs'):
        os.makedirs(base_folder + 'seqs')
    
    if score_only:
        if not os.path.exists(base_folder + 'score_only'):
            os.makedirs(base_folder + 'score_only')
   
    if conditional_probs_only:
        if not os.path.exists(base_folder + 'conditional_probs_only'):
            os.makedirs(base_folder + 'conditional_probs_only')

    if unconditional_probs_only:
        if not os.path.exists(base_folder + 'unconditional_probs_only'):
            os.makedirs(base_folder + 'unconditional_probs_only')
 
    # Validation epoch
    with torch.no_grad():
        for protein in dataset_valid:
            score_list = []
            global_score_list = []
            all_probs_list = []
            all_log_probs_list = []
            S_sample_list = []
            batch_clones = [copy.deepcopy(protein) for i in range(BATCH_COPIES)]
            X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, \
                visible_list_list, masked_list_list, masked_chain_length_list_list, \
                    chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, \
                        tied_pos_list_of_lists_list, pssm_coef, pssm_bias, \
                            pssm_log_odds_all, bias_by_res_all, tied_beta = tied_featurize(
                                batch_clones, device, chain_id_dict, fixed_positions_dict, omit_AA_dict, 
                                tied_positions_dict, pssm_dict, bias_by_res_dict, ca_only=ca_only
            )
            pssm_log_odds_mask = (pssm_log_odds_all > pssm_threshold).float() #1.0 for true, 0.0 for false
            name_ = batch_clones[0]['name']
            if score_only:
                names, input_fasta_names, mean_scores, mean_scores_stds, mean_global_scores, mean_global_scores_stds = [], [], [], [], [], []
                loop_c = 0 
                
                if input_seqs:
                    loop_c = len(input_seqs)
                for fc in tqdm(range(loop_c), desc='PMPNN scoring'):
                    #structure_sequence_score_file = base_folder + '/score_only/' + batch_clones[0]['name'] + f'_fasta_{fc+1}'
                    native_score_list = []
                    global_native_score_list = []
                    input_seq_length = len(input_seqs[fc])
                    #if fc == 0:
                    #    logger.info('%s %s', 'Chain encoding all length:', list(chain_encoding_all.size())[1])
                    #    logger.info(f'Input sequence length: {input_seq_length}')
                    S_input = torch.tensor([alphabet_dict[AA] for AA in input_seqs[fc]], device=device)[None,:].repeat(X.shape[0], 1)
                    S[:,:input_seq_length] = S_input #assumes that S and S_input are alphabetically sorted for masked_chains
                    
                    for j in range(NUM_BATCHES):
                        randn_1 = torch.randn(chain_M.shape, device=X.device)
                        log_probs = model(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_1)
                        mask_for_loss = mask*chain_M*chain_M_pos
                        scores = pmpnn_scores(S, log_probs, mask_for_loss)
                        native_score = scores.cpu().data.numpy()
                        native_score_list.append(native_score)
                        global_scores = pmpnn_scores(S, log_probs, mask)
                        global_native_score = global_scores.cpu().data.numpy()
                        global_native_score_list.append(global_native_score)
                    native_score = np.concatenate(native_score_list, 0)
                    global_native_score = np.concatenate(global_native_score_list, 0)
                    ns_mean = native_score.mean()
                    ns_mean_print = np.format_float_positional(np.float32(ns_mean), unique=False, precision=4)
                    ns_std = native_score.std()
                    ns_std_print = np.format_float_positional(np.float32(ns_std), unique=False, precision=4)

                    global_ns_mean = global_native_score.mean()
                    global_ns_mean_print = np.format_float_positional(np.float32(global_ns_mean), unique=False, precision=4)
                    global_ns_std = global_native_score.std()
                    global_ns_std_print = np.format_float_positional(np.float32(global_ns_std), unique=False, precision=4)

                    #ns_sample_size = native_score.shape[0]
                    #seq_str = pmpnn_aaidx_to_seq(S[0,], chain_M[0,])
                    #if fc < 9:
                    #    logger.info(f'Score for {name_}_{fc+1} from FASTA, mean: {ns_mean_print}, std: {ns_std_print}, '
                    #          f'sample size: {ns_sample_size},  global score, mean: {global_ns_mean_print}, '
                    #          f'std: {global_ns_std_print}, sample size: {ns_sample_size}\n Silencing further score prints...')
                    names.append(f'{name_}_{fc+1}')
                    input_fasta_names.append(input_seqs[fc])
                    mean_scores.append(ns_mean_print)
                    mean_scores_stds.append(ns_std_print)
                    mean_global_scores.append(global_ns_mean_print)
                    mean_global_scores_stds.append(global_ns_std_print)
                score_df = pd.DataFrame({
                    'name': names, 
                    'mutant': input_fasta_names, 
                    'mean_score': mean_scores, 
                    'mean_score_std': mean_scores_stds, 
                    'mean_global_score': mean_global_scores, 
                    'mean_global_score_std': mean_global_scores_stds
                })
                logger.info('%s %s %s', 'Writing scores to', os.path.abspath(base_folder + 'pmpnn_mean_scores.csv'), '...')
                score_df.to_csv(os.path.abspath(base_folder + 'pmpnn_mean_scores.csv'), sep=',', index=False)
                result_dict.update({'mean_global_score': mean_global_scores})

            elif conditional_probs_only:
                if print_all:
                    logger.info(f'Calculating conditional probabilities for {name_}')
                conditional_probs_only_file = os.path.join(base_folder, 'conditional_probs_only', batch_clones[0]['name'])
                log_conditional_probs_list = []
                for j in tqdm(range(NUM_BATCHES), desc='Getting PMPNN cond. probs. (ensemble repetitions)'):
                    randn_1 = torch.randn(chain_M.shape, device=X.device)
                    log_conditional_probs = model.conditional_probs(
                        X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_1, conditional_probs_only_backbone
                        )
                    log_conditional_probs_list.append(log_conditional_probs.cpu().numpy())
                concat_log_p = np.concatenate(log_conditional_probs_list, 0) #[B, L, 21]
                mask_out = (chain_M*chain_M_pos*mask)[0,].cpu().numpy()
                if save_npz_files:
                    np.savez(conditional_probs_only_file, log_p=concat_log_p, S=S[0,].cpu().numpy(), mask=mask[0,].cpu().numpy(), design_mask=mask_out)
                result_dict.update({'conditional_probs': concat_log_p})
            elif unconditional_probs_only:
                if print_all:
                    logger.info(f'Calculating sequence unconditional probabilities for {name_}')
                unconditional_probs_only_file = os.path.join(base_folder, 'unconditional_probs_only', batch_clones[0]['name'])
                log_unconditional_probs_list = []
                for j in range(NUM_BATCHES):
                    log_unconditional_probs = model.unconditional_probs(X, mask, residue_idx, chain_encoding_all)
                    log_unconditional_probs_list.append(log_unconditional_probs.cpu().numpy())
                concat_log_p = np.concatenate(log_unconditional_probs_list, 0) #[B, L, 21]
                mask_out = (chain_M*chain_M_pos*mask)[0,].cpu().numpy()
                if save_npz_files:
                    np.savez(unconditional_probs_only_file, log_p=concat_log_p, S=S[0,].cpu().numpy(), mask=mask[0,].cpu().numpy(), design_mask=mask_out)
                result_dict.update({'unconditional_probs': concat_log_p})
            else:
                randn_1 = torch.randn(chain_M.shape, device=X.device)
                log_probs = model(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_1)
                mask_for_loss = mask*chain_M*chain_M_pos
                scores = pmpnn_scores(S, log_probs, mask_for_loss) #score only the redesigned part
                native_score = scores.cpu().data.numpy()
                global_scores = pmpnn_scores(S, log_probs, mask) #score the whole structure-sequence
                global_native_score = global_scores.cpu().data.numpy()
                # Generate some sequences
                ali_file = base_folder + '/seqs/' + batch_clones[0]['name'] + '.fa'
                score_file = base_folder + '/scores/' + batch_clones[0]['name'] + '.npz'
                probs_file = base_folder + '/probs/' + batch_clones[0]['name'] + '.npz'
                if print_all:
                    logger.info(f'Generating sequences for: {name_}')
                t0 = time.time()
                with open(ali_file, 'w') as f:
                    for temp in temperatures:
                        for j in range(NUM_BATCHES):
                            randn_2 = torch.randn(chain_M.shape, device=X.device)
                            if tied_positions_dict == None:
                                sample_dict = model.sample(
                                    X, randn_2, S, chain_M, chain_encoding_all, residue_idx, mask=mask, temperature=temp, 
                                    omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np, chain_M_pos=chain_M_pos, 
                                    omit_AA_mask=omit_AA_mask, pssm_coef=pssm_coef, pssm_bias=pssm_bias, pssm_multi=pssm_multi, 
                                    pssm_log_odds_flag=bool(pssm_log_odds_flag), pssm_log_odds_mask=pssm_log_odds_mask, 
                                    pssm_bias_flag=bool(pssm_bias_flag), bias_by_res=bias_by_res_all
                                )
                                S_sample = sample_dict["S"] 
                            else:
                                sample_dict = model.tied_sample(
                                    X, randn_2, S, chain_M, chain_encoding_all, residue_idx, mask=mask, temperature=temp, 
                                    omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np, chain_M_pos=chain_M_pos, omit_AA_mask=omit_AA_mask, 
                                    pssm_coef=pssm_coef, pssm_bias=pssm_bias, pssm_multi=pssm_multi, pssm_log_odds_flag=bool(pssm_log_odds_flag), 
                                    pssm_log_odds_mask=pssm_log_odds_mask, pssm_bias_flag=bool(pssm_bias_flag), 
                                    tied_pos=tied_pos_list_of_lists_list[0], tied_beta=tied_beta, bias_by_res=bias_by_res_all
                                )
                            # Compute scores
                                S_sample = sample_dict["S"]
                            log_probs = model(
                                X, S_sample, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_2, 
                                use_input_decoding_order=True, decoding_order=sample_dict["decoding_order"]
                                )
                            mask_for_loss = mask*chain_M*chain_M_pos
                            scores = pmpnn_scores(S_sample, log_probs, mask_for_loss)
                            scores = scores.cpu().data.numpy()
                            
                            global_scores = pmpnn_scores(S_sample, log_probs, mask)  # score the whole structure-sequence
                            global_scores = global_scores.cpu().data.numpy()
                            
                            all_probs_list.append(sample_dict["probs"].cpu().data.numpy())
                            all_log_probs_list.append(log_probs.cpu().data.numpy())
                            S_sample_list.append(S_sample.cpu().data.numpy())
                            for b_ix in range(BATCH_COPIES):
                                masked_chain_length_list = masked_chain_length_list_list[b_ix]
                                masked_list = masked_list_list[b_ix]
                                seq_recovery_rate = torch.sum(
                                    torch.sum(
                                        torch.nn.functional.one_hot(S[b_ix], 21) * 
                                        torch.nn.functional.one_hot(S_sample[b_ix], 21)
                                        , axis=-1
                                    ) * mask_for_loss[b_ix]
                                ) / torch.sum(mask_for_loss[b_ix])
                                seq = pmpnn_aaidx_to_seq(S_sample[b_ix], chain_M[b_ix])
                                score = scores[b_ix]
                                score_list.append(score)
                                global_score = global_scores[b_ix]
                                global_score_list.append(global_score)
                                native_seq = pmpnn_aaidx_to_seq(S[b_ix], chain_M[b_ix])
                                if b_ix == 0 and j==0 and temp==temperatures[0]:
                                    start = 0
                                    end = 0
                                    list_of_AAs = []
                                    for mask_l in masked_chain_length_list:
                                        end += mask_l
                                        list_of_AAs.append(native_seq[start:end])
                                        start = end
                                    native_seq = "".join(list(np.array(list_of_AAs)[np.argsort(masked_list)]))
                                    l0 = 0
                                    for mc_length in list(np.array(masked_chain_length_list)[np.argsort(masked_list)])[:-1]:
                                        l0 += mc_length
                                        native_seq = native_seq[:l0] + '/' + native_seq[l0:]
                                        l0 += 1
                                    sorted_masked_chain_letters = np.argsort(masked_list_list[0])
                                    print_masked_chains = [masked_list_list[0][i] for i in sorted_masked_chain_letters]
                                    sorted_visible_chain_letters = np.argsort(visible_list_list[0])
                                    print_visible_chains = [visible_list_list[0][i] for i in sorted_visible_chain_letters]
                                    native_score_print = np.format_float_positional(np.float32(native_score.mean()), unique=False, precision=4)
                                    global_native_score_print = np.format_float_positional(
                                        np.float32(global_native_score.mean()), unique=False, precision=4
                                    )
                                    script_dir = os.path.dirname(os.path.realpath(__file__))
                                    try:
                                        commit_str = subprocess.check_output(
                                            f'git --git-dir {script_dir}/.git rev-parse HEAD', 
                                            shell=True, stderr=subprocess.DEVNULL).decode().strip()
                                    except subprocess.CalledProcessError:
                                        commit_str = 'unknown'
                                    if ca_only:
                                        print_model_name = 'CA_model_name'
                                    else:
                                        print_model_name = 'model_name'
                                    f.write('>{}, score={}, global_score={}, fixed_chains={}, designed_chains={}, '
                                            '{}={}, git_hash={}, seed={}\n{}\n'.format(
                                                name_, native_score_print, global_native_score_print, print_visible_chains, 
                                                print_masked_chains, print_model_name, model_name, commit_str, seed, native_seq
                                            )
                                    ) #write the native sequence
                                start = 0
                                end = 0
                                list_of_AAs = []
                                for mask_l in masked_chain_length_list:
                                    end += mask_l
                                    list_of_AAs.append(seq[start:end])
                                    start = end
    
                                seq = "".join(list(np.array(list_of_AAs)[np.argsort(masked_list)]))
                                l0 = 0
                                for mc_length in list(np.array(masked_chain_length_list)[np.argsort(masked_list)])[:-1]:
                                    l0 += mc_length
                                    seq = seq[:l0] + '/' + seq[l0:]
                                    l0 += 1
                                score_print = np.format_float_positional(np.float32(score), unique=False, precision=4)
                                global_score_print = np.format_float_positional(np.float32(global_score), unique=False, precision=4)
                                seq_rec_print = np.format_float_positional(np.float32(seq_recovery_rate.detach().cpu().numpy()), unique=False, precision=4)
                                sample_number = j*BATCH_COPIES+b_ix+1
                                f.write('>T={}, sample={}, score={}, global_score={}, seq_recovery={}\n{}\n'.format(
                                    temp,sample_number,score_print,global_score_print,seq_rec_print,seq
                                    )
                                ) #write generated sequence
                if save_npz_files:
                    np.savez(score_file, score=np.array(score_list, np.float32), global_score=np.array(global_score_list, np.float32))
                if save_npz_files:
                    all_probs_concat = np.concatenate(all_probs_list)
                    all_log_probs_concat = np.concatenate(all_log_probs_list)
                    S_sample_concat = np.concatenate(S_sample_list)
                    np.savez(probs_file, probs=np.array(all_probs_concat, np.float32), log_probs=np.array(all_log_probs_concat, np.float32), S=np.array(S_sample_concat, np.int32), mask=mask_for_loss.cpu().data.numpy(), chain_order=chain_list_list)
                t1 = time.time()
                dt = round(float(t1-t0), 4)
                num_seqs = len(temperatures)*NUM_BATCHES*BATCH_COPIES
                total_length = X.shape[1]
                if print_all:
                    logger.info(f'{num_seqs} sequences of length {total_length} generated in {dt} seconds')
    return result_dict        