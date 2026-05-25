"""
Main benchmarking script to evaluate PyPEF hybrid PLM-DCA model on ProteinGym DMS assays.
Structured based on Kermut run script.
"""

from pathlib import Path
import hydra
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from omegaconf import DictConfig
from scipy.stats import spearmanr
from Bio import SeqIO, BiopythonParserWarning
import warnings
warnings.filterwarnings(action='ignore', category=BiopythonParserWarning)


from pypef.plm.esm_lora_tune import get_esm_models
from pypef.plm.prosst_lora_tune import get_prosst_models
from pypef.utils.variant_data import check_alignment, get_mismatches, get_seqs_from_var_name, shift_and_trim_vars_seqs
from pypef.dca.gremlin_inference import GREMLIN, get_delta_e_statistical_model
from pypef.hybrid.hybrid_model import DCALLMHybridModel
from pypef.plm.inference import esm_setup, prosst_setup, tokenize_sequences

import logging
package_logger = logging.getLogger('pypef')
package_logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
package_logger.addHandler(handler)

# Make sure to "export CUBLAS_WORKSPACE_CONFIG=:4096:8" first
@hydra.main(version_base=None, config_path="../configs", config_name="proteingym_data_setup")
def main(cfg: DictConfig) -> None:
    # Experiment settings
    split_method = cfg.split_method
    progress_bar = cfg.progress_bar
    llm = cfg.llm
    print("PLM:", llm)
    sequence_col, target_col = "mutated_sequence", "DMS_score"
    assert cfg.split_method in ["fold_random_5", "fold_modulo_5", "fold_contiguous_5", "fold_rand_multiples"]
    use_multiples = True if cfg.split_method == "fold_rand_multiples" else False

    # Reproducibility
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    seed = cfg.seed

    # Verify input paths
    if use_multiples:
        DMS_data_folder = Path(cfg.DMS_data_folder_multiples)
    else:
        DMS_data_folder = Path(cfg.DMS_data_folder_singles)

    DMS_reference_file_path = Path(cfg.DMS_reference_file_path)
    DMS_MSA_folder = Path(cfg.DMS_MSA_data_path)
    DMS_PDB_folder = Path(cfg.DMS_PDB_data_path)
    output_scores_folder = Path(cfg.output_scores_folder)

    # Load dataset
    DMS_idx = cfg.DMS_idx
    df_ref = pd.read_csv(DMS_reference_file_path)
    if use_multiples:
        df_ref = df_ref[df_ref['includes_multiple_mutants'] == True]
        df_ref['single_index'] = df_ref.index
        df_ref = df_ref.reset_index(drop=True)
    DMS_id = df_ref.loc[DMS_idx, "DMS_id"]
    DMS_msa = df_ref.loc[DMS_idx, "MSA_filename"]
    msa_file = (DMS_MSA_folder / DMS_msa).resolve()
    msa_start = df_ref.loc[DMS_idx, "MSA_start"]
    msa_end = df_ref.loc[DMS_idx, "MSA_end"]
    wt_msa_trimmed_sequence = df_ref.loc[DMS_idx, "target_seq"]
    DMS_pdb = df_ref.loc[DMS_idx, "pdb_file"]
    #pdb_range = df_ref.loc[DMS_idx, "pdb_range"]
    #pdb_start = int(pdb_range.split('-')[0])
    #pdb_end = int(pdb_range.split('-')[1])
    csv_substitutions_file = (DMS_data_folder / f"{DMS_id}.csv").resolve()
    pdb_file = (DMS_PDB_folder / DMS_pdb).resolve()
    output_path = output_scores_folder / f"{split_method}/pypef_hybrid/{llm}/{DMS_id}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print('CSV path:', csv_substitutions_file)
    print('MSA path:', msa_file)
    print('MSA start:', msa_start, '- MSA end:', msa_end)
    wt_msa_trimmed_sequence = wt_msa_trimmed_sequence[msa_start - 1:msa_end]
    print(f'WT sequence (trimmed from MSA start to MSA end), length='
          f'{len(wt_msa_trimmed_sequence)}:\n{wt_msa_trimmed_sequence}')

    if output_path.resolve().exists():
        if not cfg.overwrite:
            print(f"Output file already exists: {output_path.resolve()}")
            return   # Hydra could overwrite the 0 return later, so the Bash script still faces an error ('set -e')
        else:
            print(f"Overwriting existing output file: {output_path.resolve()}")
    else:
        print("Output does not yet exist")

    if "prosst" in llm.lower():
        _, _, prosst_tokenizer, _ = get_prosst_models(
            seed=seed, revision="e94ffee7846d7f55c1bf5efa8ec7372a336ac4b8"
        )
    if "esm" in llm.lower():
        _, _, esm_tokenizer, _ = get_esm_models(
            model="facebook/esm1v_t33_650M_UR90S_3", seed=seed, revision="0b00fd112e63f6b5e70a9cd8484d4e660312ce70"
        )
    df = pd.read_csv(csv_substitutions_file)
    print(df)
    variants = df['mutant']
    sequences = df['mutated_sequence']
    sequences_msa_trimmed = []
    for s in sequences:
        sequences_msa_trimmed.append(s[msa_start - 1:msa_end])
    pdb_seq = str(list(SeqIO.parse(pdb_file, "pdb-atom"))[0].seq)
    variants_split = []
    for variant in variants:
        # Split double and higher substituted variants to multiple single substitutions
        # e.g. separated by ':' or '/'
        variants_split.append(variant.split(':'))
    variants, _, _ = get_seqs_from_var_name(
        wt_msa_trimmed_sequence, variants_split, shift_pos=msa_start - 1)
    print(f"PDB sequence length: {len(pdb_seq)}")
    if pdb_seq != wt_msa_trimmed_sequence:
        print(f"Original WT sequence length: {len(wt_msa_trimmed_sequence)}")
        mapping = check_alignment(wt_msa_trimmed_sequence, pdb_seq)
        pdb_trimmed_common_sequence = mapping['common_seq']  # Use common sequence (WT (MSA-trimmed) seq trimmed to PDB seq)
        print(f"New WT sequence trimmed to common sequence/PDB sequence length): {len(pdb_trimmed_common_sequence)}")
        print(mapping['identity'])
        print(mapping['alignment_obj'])
        if mapping and mapping['identity']:
            # Perform the shift and trim
            pdb_vars, orig_vars, sequences_msa_trimmed, pdb_trimmed_seqs = shift_and_trim_vars_seqs(
                vars_list=variants, 
                seqs_list=sequences_msa_trimmed, 
                alignment_mapping=mapping,
                msa_start=msa_start
            )
            print(f'Shifted variant namings, i.e., mutation positions; e.g., "{variants[0]}" --> "{pdb_vars[0]}"')
            variants = pdb_vars
            
            print(f"Shifted variants relative to PDB start. New length: {len(pdb_trimmed_seqs[0])}")
            assert len(wt_msa_trimmed_sequence) == len(sequences_msa_trimmed[0]), (
                f"{len(wt_msa_trimmed_sequence)} != {len(sequences_msa_trimmed[0])}")
            assert len(pdb_trimmed_common_sequence) == len(pdb_trimmed_seqs[0]), (
                f"{len(pdb_trimmed_common_sequence)} != {len(pdb_trimmed_seqs[0])}")
            
        else:
            print(
                f"Wild-type sequence is not matching PDB-extracted sequence"
                f"\nWT sequence:\n{wt_msa_trimmed_sequence}\nPDB sequence:\n{pdb_seq}. TODO: Shifting "
                f"variants and trimming sequences. Skipping dataset..."
            )
            raise RuntimeError
    else:
        pdb_trimmed_common_sequence = wt_msa_trimmed_sequence
        pdb_trimmed_seqs = sequences_msa_trimmed
    
    
    #if DMS_id == "BRCA2_HUMAN_Erwood_2022_HEK293T":
    #    # Disable distance kernel due to sequenc length
    #    cfg.gp.mutation_kernel.use_distances = False
    #    cfg.gp.mutation_kernel.model._target_ = "kermut.model.kernel.Kermut_no_d"

    print(f"Output path: {output_path.resolve()}")

    print(
        f"Using {split_method} split on DMS idx {DMS_idx}: {DMS_id}",
        flush=True,
    )

    # Prepare output
    df_predictions = pd.DataFrame(columns=["fold", "mutant", "y", "y_pred", "y_var"])

    df = df.reset_index(drop=True)
    print('GREMLIN DCA (MSA optimization)...')
    gremlin = GREMLIN(
        alignment=msa_file, opt_iter=100, optimize=True
    )

    if "prosst" in llm.lower():
        assert gremlin.first_msa_seq.upper() == wt_msa_trimmed_sequence, (
            f"{gremlin.first_msa_seq.upper()}\n   !=\n{wt_msa_trimmed_sequence}")
        n_mismatches, mismatches = get_mismatches(
            wt_msa_trimmed_sequence, gremlin.first_msa_seq.upper())
        print(f'Ratio of mismatches: {n_mismatches / len(wt_msa_trimmed_sequence)}, '
              f'N={n_mismatches}, Mismatches="{mismatches}"')
        assert (n_mismatches / len(wt_msa_trimmed_sequence)) <= 0.05
    y_full = df[target_col].values
    #seq_full = df[sequence_col].values

    x_dca_full = gremlin.collect_encoded_sequences(sequences_msa_trimmed)
    x_dca_full = np.array(x_dca_full)
    y_pred_dca = get_delta_e_statistical_model(x_dca_full, gremlin.x_wt)
    print(f'DCA (unsupervised performance, Spear. corr.): {spearmanr(y_full, y_pred_dca)[0]:.3f}')  
    try:
        unique_folds = df[split_method].unique()
    except KeyError as e:
        raise RuntimeError(f"KeyError: {e}. Available columns: {df.columns.to_list()}")

    for i, test_fold in enumerate(tqdm(unique_folds, disable=not progress_bar)):
        print(f'Split {i+1}/{len(unique_folds)}...')
        # Assign splits
        train_idx = (df[split_method] != test_fold).tolist()
        test_idx = (df[split_method] == test_fold).tolist()
        v_train = np.asarray(variants)[train_idx]
        v_test = np.asarray(variants)[train_idx]
        s_train = np.asarray(pdb_trimmed_seqs)[train_idx]
        s_test =  np.asarray(pdb_trimmed_seqs)[test_idx]
        x_dca_train = x_dca_full[train_idx]
        x_dca_test = x_dca_full[test_idx]
        y_train = y_full[train_idx]
        y_test = y_full[test_idx]
        
        print(f"    Test fold: {test_fold}: N_Train={len(y_train)}, N_Test={len(y_test)} "
              f"Test proportion: {len(y_test) / (len(y_train) + len(y_test)):.3f}")
        
        llm_dict_train = {}
        if "esm" in llm.lower():
            llm_dict_esm = esm_setup(
                wt_seq=pdb_trimmed_common_sequence, sequences=s_train, 
                seed=seed, revision="0b00fd112e63f6b5e70a9cd8484d4e660312ce70", device="cuda", verbose=True
            )
            llm_dict_train.update(llm_dict_esm)
        if "prosst" in llm.lower():
            llm_dict_prosst = prosst_setup(
                wt_seq=pdb_trimmed_common_sequence, pdb_file=pdb_file, sequences=s_train, 
                seed=seed, revision="e94ffee7846d7f55c1bf5efa8ec7372a336ac4b8", device="cuda", verbose=True
            )
            llm_dict_train.update(llm_dict_prosst)
        print(f'Train: {len(np.array(y_train))} --> Test: {len(np.array(y_test))}')

        
        #if df.shape[0] >= 100000:  # Not CV-training the PLM on much data but just relying on DCA
        #    llm_kwargs = None      # Datasets: HIS7_YEAST_Pokusaeva_2019.csv
        #    x_llm_test = None
        #    print(f'\nSkipping LLM CV training for dataset {csv_substitutions_file} as it '
        #          f'would take up (too) much time...')
        hm = DCALLMHybridModel(
            x_train_dca=np.array(x_dca_train), 
            y_train=y_train,
            llm_model_input=llm_dict_train,
            x_wt=gremlin.x_wt,
            variants=v_train,
            lora_train=False,
            gauss_opt=True,
            n_epochs=50  # Only used if lora_train==True
        )

        x_llm_dict_test = {}
        if "esm" in llm.lower():
            x_test_esm, _esm_attention_mask = tokenize_sequences(
                sequences=s_test, 
                tokenizer=esm_tokenizer, 
                max_length=len(pdb_trimmed_common_sequence) + 2
            )
            x_llm_dict_test.update({'esm1v': np.asarray(x_test_esm)})

        if "prosst" in llm.lower():
            x_test_prosst, _prosst_attention_mask = tokenize_sequences(
                sequences=s_test, 
                tokenizer=prosst_tokenizer, 
                max_length=len(pdb_trimmed_common_sequence) + 2
            )
            x_llm_dict_test.update({'prosst': np.asarray(x_test_prosst)})

        y_test_pred = hm.hybrid_prediction(
            x_dca=np.array(x_dca_test), 
            x_llm_dict=x_llm_dict_test,
            variants=v_test

        )
        print(f"====> Performance (Spearman corr.): {spearmanr(y_test, y_test_pred)[0]:.3f}\n")

        df_pred_fold = pd.DataFrame(
            {
                "fold": test_fold,
                "mutant": df.loc[test_idx, "mutant"],
                "y": y_test,
                "y_pred": y_test_pred
            }
        )
        if df_predictions.empty:
            df_predictions = df_pred_fold
        else:
            df_predictions = pd.concat([df_predictions, df_pred_fold])

    df_predictions.to_csv(output_path, index=False)
    print(f"Saved prediction CSV to {output_path}.")


if __name__ == "__main__":
    # e.g. run with 
    #   python pgym_cv_benchmark.py split_method=fold_random_5 DMS_idx=3 llm=prosst overwrite=true
    main()
