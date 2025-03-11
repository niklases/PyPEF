
import os
import copy
import gc
import time
import json
import pandas as pd
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text

import sys  # Use local directory PyPEF files
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from pypef.dca.gremlin_inference import GREMLIN
from pypef.llm.esm_lora_tune import get_esm_models, get_encoded_seqs, get_batches, esm_train, esm_test, esm_infer, corr_loss
from pypef.llm.prosst_lora_tune import get_logits_from_full_seqs, get_prosst_models, get_structure_quantizied
from pypef.hybrid.hybrid_model import DCALLMHybridModel, reduce_by_batch_modulo, get_delta_e_statistical_model



def get_seqs_from_var_name(
        wt_seq: str,
        substitutions: list,
        fitness_values: None | list = None,
        shift_pos: int = 0
) -> tuple[list, list, list]:
    """
    Similar to function "get_sequences_from_file" but instead of getting 
    sequences from fasta file it directly gets them from wt sequence and
    variant specifiers.

    wt_seq: str
        Wild-type sequence as string
    substitutions: list
        List of amino acid substitutions of a single variant of the format:
            - Single substitution variant, e.g. variant A123C: ['A123C']
            - Higher variants, e.g. variant A123C/D234E/F345G: ['A123C', 'D234E, 'F345G']
            --> Full substitutions list, e.g.: [['A123C'], ['A123C', 'D234E, 'F345G']]
    fitness_values: list
        List of ints/floats of the variant fitness values, e.g. for two variants: [1.4, 0.8]
    """
    if fitness_values is None:
        fitness_values = np.zeros(len(substitutions)).tolist()
    variants, values, sequences = [], [], []
    for i, var in enumerate(substitutions):  # var are lists of (single or multiple) substitutions
        temp = list(wt_seq)
        name = ''
        separation = 0
        if var == ['WT']:
            name = 'WT'
        else:
            for single_var in var:  # single entries of substitution list
                position_index = int(str(single_var)[1:-1]) - 1 - shift_pos
                new_amino_acid = str(single_var)[-1]
                if str(single_var)[0].isalpha(): # Assertion only possible for format AaPosAa, e.g. A123C
                    assert str(single_var)[0] == temp[position_index], f"Input variant: "\
                        f"{str(single_var)[0]}{position_index + 1}{new_amino_acid}, WT amino "\
                        f"acid variant {temp[position_index]}{position_index + 1}{new_amino_acid}"
                temp[position_index] = new_amino_acid
                # checking if multiple entries are inside list
                if separation == 0:
                    name += single_var
                else:
                    name += '/' + single_var
                separation += 1
        variants.append(name)
        values.append(fitness_values[i])
        sequences.append(''.join(temp))

    return variants, values, sequences


def get_vram(verbose: bool = True):
    free = torch.cuda.mem_get_info()[0] / 1024 ** 3
    total = torch.cuda.mem_get_info()[1] / 1024 ** 3
    total_cubes = 24
    free_cubes = int(total_cubes * free / total)
    if verbose:
        print(f'VRAM: {total - free:.2f}/{total:.2f}GB\t VRAM:[' + (
            total_cubes - free_cubes) * '▮' + free_cubes * '▯' + ']')
    return free, total


# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")
get_vram()
#base_model, lora_model, tokenizer, optimizer = get_esm_models()
base_model, lora_model, tokenizer, optimizer = get_prosst_models()
base_model = base_model.to(device)
MAX_WT_SEQUENCE_LENGTH = 400
N_EPOCHS = 5
get_vram()


def compute_performances(mut_data, mut_sep=':', start_i: int = 0, already_tested_is: list = []):
    tested_dsets = []
    tested_dsets_only = []
    tested_ns = []
    dset_all_ns = []
    dset_dca_perfs = []
    dset_esm_perfs = []
    n_max_muts = []
    n_tested_datasets = 0
    plt.figure(figsize=(40, 12))
    numbers_of_datasets = [i + 1 for i in range(len(mut_data.keys()))]
    delta_times = []
    for i, (dset_key, dset_paths) in enumerate(mut_data.items()):
        if i >= start_i and i not in already_tested_is and i == 18 - 1:
            start_time = time.time()
            print(f'\n{i+1}/{len(mut_data.items())}\n'
                  f'===============================================================')
            csv_path = dset_paths['CSV_path']
            msa_path = dset_paths['MSA_path']
            wt_seq = dset_paths['WT_sequence']
            msa_start = dset_paths['MSA_start']
            msa_end = dset_paths['MSA_end']
            pdb = dset_paths['PDB_path']
            wt_seq = wt_seq[msa_start - 1:msa_end]
            print('CSV path:', csv_path)
            print('MSA path:', msa_path)
            print('MSA start:', msa_start, '- MSA end:', msa_end)
            print('WT sequence (trimmed from MSA start to MSA end):\n' + wt_seq)
            # Getting % usage of virtual_memory (3rd field)
            import psutil;print(f'RAM used: {round(psutil.virtual_memory()[3]/1E9, 3)} '
                  f'GB ({psutil.virtual_memory()[2]} %)')
            variant_fitness_data = pd.read_csv(csv_path, sep=',')
            print('N_variant-fitness-tuples:', np.shape(variant_fitness_data)[0])
            #if np.shape(variant_fitness_data)[0] > 400000:
            #    print('More than 400000 variant-fitness pairs which represents a '
            #          'potential out-of-memory risk, skipping dataset...')
            #    continue
            variants = variant_fitness_data['mutant'].to_numpy()  # [400:700]
            variants_orig = variants
            fitnesses = variant_fitness_data['DMS_score'].to_numpy()  # [400:700]
            if len(fitnesses) <= 50: # and len(fitnesses) >= 500:  # TODO: RESET TO 50
                print('Number of available variants <= 50, skipping dataset...')
                continue
            variants_split = []
            for variant in variants:
                # Split double and higher substituted variants to multiple single substitutions
                # e.g. separated by ':' or '/'
                variants_split.append(variant.split(mut_sep))
            variants, fitnesses, sequences = get_seqs_from_var_name(
                wt_seq, variants_split, fitnesses, shift_pos=msa_start - 1)
            # Only model sequences with length of max. 800 amino acids to avoid out of memory errors 
            print('Sequence length:', len(wt_seq))
            if len(wt_seq) > MAX_WT_SEQUENCE_LENGTH:
                print(f'Sequence length over {MAX_WT_SEQUENCE_LENGTH}, which represents a potential out-of-memory risk '
                      f'(when running on GPU, set threshold to length ~400 dependent on available VRAM), '
                      f'skipping dataset...')
                continue
            variants, variants_split, sequences, fitnesses = (
                reduce_by_batch_modulo(variants), 
                reduce_by_batch_modulo(variants_split), 
                reduce_by_batch_modulo(sequences), 
                reduce_by_batch_modulo(fitnesses)
            )
            count_gap_variants = 0
            n_muts = []
            for variant in variants_split:
                n_muts.append(len(variant))
            max_muts = max(n_muts)
            print(f'N max. (multiple) amino acid substitutions: {max_muts}')
            ratio_input_vars_at_gaps = count_gap_variants / len(variants)
            
            print(len(sequences))
            gremlin = GREMLIN(alignment=msa_path, opt_iter=100, optimize=True)
            x_dca = gremlin.collect_encoded_sequences(sequences)
            x_wt = gremlin.x_wt
            y_pred_dca = get_delta_e_statistical_model(x_dca, x_wt)
            print('DCA:', spearmanr(fitnesses, y_pred_dca), len(fitnesses)) 
            dca_unopt_perf = spearmanr(fitnesses, y_pred_dca)[0]
            # TF    10,000: DCA: SignificanceResult(statistic=np.float64(0.6486616550552755), pvalue=np.float64(3.647740047145113e-119))  989
            # Torch 10,000: DCA: SignificanceResult(statistic=np.float64(0.6799982280150232), pvalue=np.float64(3.583110693136881e-135)) 989

            #x_esm, attention_masks = get_encoded_seqs(sequences, tokenizer, max_length=len(wt_seq))
            input_ids, attention_mask, structure_input_ids = get_structure_quantizied(pdb_file=pdb)
            y_prosst = get_logits_from_full_seqs(sequences, base_model, input_ids, attention_mask, structure_input_ids, train=False)
            #x_esm_b, attention_masks_b, fitnesses_b = get_batches(x_esm), get_batches(attention_masks), get_batches(fitnesses)
            #y_true, y_pred_esm = esm_test(x_esm_b, attention_masks_b, fitnesses_b, loss_fn=corr_loss, model=base_model)
            #y_true.detach().cpu().numpy()
            #y_pred_esm.detach().cpu().numpy()
            #print('ESM1v:', spearmanr(y_true, y_pred_esm))
            print('ProSST:', spearmanr(fitnesses, y_prosst))
            esm_unopt_perf = spearmanr(y_true, y_pred_esm)[0]

            hybrid_perfs = []
            ns_y_test = [len(variants)]
            for i_t, train_size in enumerate([100, 200, 1000]):
                lora_model = copy.deepcopy(base_model)
                print('TRAIN SIZE:', train_size)
                get_vram()
                optimizer = torch.optim.Adam(lora_model.parameters(), lr=0.0001)
                try:
                    (
                        x_dca_train, x_dca_test, 
                        x_esm_train, x_esm_test, 
                        attns_train, attns_test, 
                        y_train, y_test
                    ) = train_test_split(
                        x_dca,
                        x_esm, 
                        attention_masks, 
                        fitnesses, 
                        train_size=train_size, 
                        random_state=42
                    )
                    (
                        x_dca_train, x_dca_test, 
                        x_esm_train, x_esm_test, 
                        attns_train, attns_test, 
                        y_train, y_test
                    ) = (
                        reduce_by_batch_modulo(x_dca_train), reduce_by_batch_modulo(x_dca_test), 
                        reduce_by_batch_modulo(x_esm_train), reduce_by_batch_modulo(x_esm_test), 
                        reduce_by_batch_modulo(attns_train), reduce_by_batch_modulo(attns_test), 
                        reduce_by_batch_modulo(y_train), reduce_by_batch_modulo(y_test)
                    )
                    print(f'Train: {len(np.array(y_train))} --> Test: {len(np.array(y_test))}')
                    if len(y_test) <= 50:
                        print(f'Only {len(fitnesses)} in total, splitting the data in N_Train = {len(y_train)} '
                              f'and N_Test = {len(y_test)} results in N_Test <= 50 variants - '
                              f'not getting performance for N_Train = {len(y_train)}...')
                        hybrid_perfs.append(np.nan)
                        ns_y_test.append(np.nan)
                        continue
                    get_vram()
                    hm = DCALLMHybridModel(
                        x_train_dca=np.array(x_dca_train), 
                        x_train_llm=np.array(x_esm_train), 
                        x_train_llm_attention_masks=np.array(attns_train), 
                        y_train=y_train,
                        llm_model=lora_model,
                        llm_base_model=base_model,
                        llm_optimizer=optimizer,
                        llm_train_function=esm_train,
                        llm_test_function=esm_test,
                        llm_inference_function=esm_infer,
                        llm_loss_function=corr_loss,
                        x_wt=x_wt
                    )

                    y_test = reduce_by_batch_modulo(y_test)

                    y_test_pred = hm.hybrid_prediction(
                        x_dca=np.array(x_dca_test), 
                        x_llm=np.array(x_esm_test), 
                        attns_llm=np.array(attns_test)
                    )

                    print(f'Hybrid perf.: {spearmanr(y_test, y_test_pred)[0]}')
                    hybrid_perfs.append(spearmanr(y_test, y_test_pred)[0])
                    ns_y_test.append(len(y_test_pred))
                except ValueError:
                    print(f'Only {len(y_true)} variant-fitness pairs in total, cannot split the data '
                          f'in N_Train = {train_size} and N_Test (N_Total - N_Train).')
                    hybrid_perfs.append(np.nan)
                    ns_y_test.append(np.nan)
                del lora_model 
                torch.cuda.empty_cache()
                gc.collect()
            dt = time.time() - start_time
            delta_times.append(dt)
            n_tested_datasets += 1
            tested_dsets.append(f'({n_tested_datasets}) {dset_key} '
                                f'({len(variants)}, {100.0 - (ratio_input_vars_at_gaps * 100):.2f}%, {max_muts})')
            tested_dsets_only.append(dset_key)
            dset_all_ns.append(len(variants))
            tested_ns.append(numbers_of_datasets[i])
            numbers_of_datasets[i]
            n_max_muts.append(max_muts)
            dset_dca_perfs.append(dca_unopt_perf)
            dset_esm_perfs.append(esm_unopt_perf)
            dset_hybrid_perfs_i = ''
            for hp in hybrid_perfs:
                dset_hybrid_perfs_i += f'{hp},'
            dset_ns_y_test_i = ''
            for ns_y_t in ns_y_test:
                dset_ns_y_test_i += f'{ns_y_t},'
            print(ns_y_test)
            print('\nREADME:\n', dset_hybrid_perfs_i, '\n', dset_ns_y_test_i, '\n')
            with open(out_results_csv, 'a') as fh:
                fh.write(
                    f'{numbers_of_datasets[i]},{dset_key},{len(variants_orig)},{max_muts},{dca_unopt_perf},'
                    f'{esm_unopt_perf},{dset_hybrid_perfs_i}{dset_ns_y_test_i}{dt}\n')
                

def plot_csv_data(csv, plot_name):
    train_test_size_texts = []
    df = pd.read_csv(csv, sep=',')  
    # No.,Dataset,N_Variants,N_Max_Muts,Untrained_Performance_DCA,Untrained_Performance_LLM,Hybrid_Trained_Performance_100,Hybrid_Trained_Performance_200,Hybrid_Trained_Performance_1000
    tested_dsets = df['No.']
    dset_dca_perfs = df['Untrained_Performance_DCA']
    dset_esm_perfs = df['Untrained_Performance_LLM']
    dset_hybrid_perfs_100 = df['Hybrid_Trained_Performance_100']
    dset_hybrid_perfs_200 = df['Hybrid_Trained_Performance_200']
    dset_hybrid_perfs_1000 = df['Hybrid_Trained_Performance_1000']

    plt.figure(figsize=(80, 12))
    #import gc;gc.collect()  # Potentially GC is needed to free some RAM (deallocated VRAM -> partly stored in RAM?) after each run
    plt.plot(range(len(tested_dsets)), dset_dca_perfs, 'o--', markersize=8, color='tab:blue', label='DCA (0)')
    plt.plot(range(len(tested_dsets) + 1), np.full(len(tested_dsets) + 1, np.nanmean(dset_dca_perfs)), color='tab:blue', linestyle='--')
    for i, (p, n_test) in enumerate(zip(dset_dca_perfs, df['N_Y_test'].astype('Int64').to_list())):
        plt.text(i, 0.970, i, color='tab:grey', size=2)
        plt.text(i, 0.975, f'0'  + r'$\rightarrow$' + f'{n_test}', color='tab:blue', size=2)
    train_test_size_texts.append(plt.text(len(tested_dsets), np.nanmean(dset_dca_perfs), f'{np.nanmean(dset_dca_perfs):.2f}', color='tab:blue'))

    plt.plot(range(len(tested_dsets)), dset_esm_perfs, 'o--', markersize=8, color='tab:grey', label='ESM (0)')
    plt.plot(range(len(tested_dsets) + 1), np.full(len(tested_dsets) + 1, np.nanmean(dset_esm_perfs)), color='tab:grey', linestyle='--')
    for i, (p, n_test) in enumerate(zip(dset_esm_perfs, df['N_Y_test'].astype('Int64').to_list())):
       plt.text(i, 0.980, f'0'  + r'$\rightarrow$' + f'{n_test}', color='tab:grey', size=2)
    train_test_size_texts.append(plt.text(len(tested_dsets), np.nanmean(dset_esm_perfs), f'{np.nanmean(dset_esm_perfs):.2f}', color='tab:grey'))

    plt.plot(range(len(tested_dsets)), dset_hybrid_perfs_100, 'o--', markersize=8, color='tab:orange', label='Hybrid (100)')
    plt.plot(range(len(tested_dsets) + 1), np.full(len(tested_dsets) + 1, np.nanmean(dset_hybrid_perfs_100)), color='tab:orange', linestyle='--')
    for i, (p, n_test) in enumerate(zip(dset_hybrid_perfs_100, df['N_Y_test_100'].astype('Int64').to_list())):
        plt.text(i, 0.985, f'100'  + r'$\rightarrow$' + f'{n_test}', color='tab:orange', size=2)
    train_test_size_texts.append(plt.text(len(tested_dsets), np.nanmean(dset_hybrid_perfs_100), f'{np.nanmean(dset_hybrid_perfs_100):.2f}', color='tab:orange'))

    plt.plot(range(len(tested_dsets)), dset_hybrid_perfs_200, 'o--', markersize=8, color='tab:green', label='Hybrid (200)')
    plt.plot(range(len(tested_dsets) + 1), np.full(len(tested_dsets) + 1, np.nanmean(dset_hybrid_perfs_200)), color='tab:green', linestyle='--')
    for i, (p, n_test) in enumerate(zip(dset_hybrid_perfs_200, df['N_Y_test_200'].astype('Int64').to_list())):
       plt.text(i, 0.990, f'200'  + r'$\rightarrow$' + f'{n_test}', color='tab:green', size=2)
    train_test_size_texts.append(plt.text(len(tested_dsets), np.nanmean(dset_hybrid_perfs_200), f'{np.nanmean(dset_hybrid_perfs_200):.2f}', color='tab:green'))

    plt.plot(range(len(tested_dsets)), dset_hybrid_perfs_1000, 'o--', markersize=8, color='tab:red',  label='Hybrid (1000)')
    plt.plot(range(len(tested_dsets) + 1), np.full(len(tested_dsets) + 1, np.nanmean(dset_hybrid_perfs_1000)), color='tab:red', linestyle='--')
    for i, (p, n_test) in enumerate(zip(dset_hybrid_perfs_1000, df['N_Y_test_1000'].astype('Int64').to_list())):
        plt.text(i, 0.995, f'1000'  + r'$\rightarrow$' + f'{n_test}', color='tab:red', size=2)
    train_test_size_texts.append(plt.text(len(tested_dsets), np.nanmean(dset_hybrid_perfs_1000), f'{np.nanmean(dset_hybrid_perfs_1000):.2f}', color='tab:red'))
    plt.grid(zorder=-1)
    plt.xticks(range(len(tested_dsets)), tested_dsets, rotation=45, ha='right')
    plt.margins(0.01)
    plt.legend()
    plt.tight_layout()
    plt.ylim(0.0, 1.0)
    plt.ylabel(r'|Spearman $\rho$|')
    adjust_text(train_test_size_texts, expand=(1.2, 2))
    plt.savefig(os.path.join(os.path.dirname(__file__), f'{plot_name}.png'), dpi=300)
    print('Saved file as ' + os.path.join(os.path.dirname(__file__),  f'{plot_name}.png') + '.')

    plt.clf()
    plt.figure(figsize=(16, 12))
    sns.set_style("whitegrid")
    df_ = df[[
        'Untrained_Performance_DCA',
        'Untrained_Performance_LLM',
        'Hybrid_Trained_Performance_100',
        'Hybrid_Trained_Performance_200',
        'Hybrid_Trained_Performance_1000'
        ]]
    print(df_)
    sns.violinplot(
        df_, saturation=0.4, 
        palette=['tab:blue', 'tab:grey', 'tab:orange', 'tab:green', 'tab:red']
    )
    plt.ylim(-0.09, 1.09)
    plt.ylabel(r'|Spearmanr $\rho$|')
    sns.swarmplot(df_, color='black')
    print(df.columns)
    dset_ns_y_test = [
        df['N_Y_test'].to_list(), 
        df['N_Y_test'].to_list(),
        df['N_Y_test_100'].to_list(), 
        df['N_Y_test_200'].to_list(), 
        df['N_Y_test_1000'].to_list()
    ]
    for n in range(0, len(dset_ns_y_test)):
        plt.text(
            n + 0.2, -0.075, 
            [
                r'$\overline{|\rho|}=$' + f'{np.nanmean(dset_dca_perfs):.2f}', 
                r'$\overline{|\rho|}=$' + f'{np.nanmean(dset_esm_perfs):.2f}', 
                r'$\overline{|\rho|}=$' + f'{np.nanmean(dset_hybrid_perfs_100):.2f}', 
                r'$\overline{|\rho|}=$' + f'{np.nanmean(dset_hybrid_perfs_200):.2f}', 
                r'$\overline{|\rho|}=$' + f'{np.nanmean(dset_hybrid_perfs_1000):.2f}'
            ][n]
        )
        plt.text(  # N_Y_test,N_Y_test_100,N_Y_test_200,N_Y_test_1000
            n + 0.2, -0.05, 
            r'$\overline{N_{Y_\mathrm{test}}}=$' + f'{int(np.nanmean(np.array(dset_ns_y_test)[n]))}'
        )
        plt.text(
            n + 0.2, -0.025,
            r'$N_\mathrm{Datasets}=$' + f'{np.count_nonzero(~np.isnan(np.array(dset_ns_y_test)[n]))}'
        )
    print(f'\n{np.shape(dset_ns_y_test)[1]} datasets tested with N_Test\'s at N_Train\'s =\n'
          f'  0    0    100  200  1000\n{np.nan_to_num(dset_ns_y_test).T.astype("int")}')
    print()
    plt.savefig(os.path.join(os.path.dirname(__file__), f'{plot_name}_violin.png'), dpi=300)
    print('Saved file as ' + os.path.join(os.path.dirname(__file__), f'{plot_name}_violin.png') + '.')


if __name__ == '__main__':
    single_point_mut_data = os.path.abspath(os.path.join(
        os.path.dirname(__file__), f"single_point_dms_mut_data.json"))
    higher_mut_data = os.path.abspath(os.path.join(
        os.path.dirname(__file__), f"higher_point_dms_mut_data.json"))
    with open(single_point_mut_data, 'r') as fh:
        s_mut_data = json.loads(fh.read())
    with open(higher_mut_data, 'r') as fh:
        h_mut_data = json.loads(fh.read())
    combined_mut_data = s_mut_data.copy()
    combined_mut_data.update(h_mut_data)

    os.makedirs('results', exist_ok=True)
    out_results_csv = os.path.join(os.path.dirname(__file__), 'results/dca_esm_and_hybrid_opt_results.csv')
    if os.path.exists(out_results_csv):
        print(f'\nReading existing file {out_results_csv}...')
        df = pd.read_csv(out_results_csv, sep=',')
        print(df)
        try:
            start_i = df['No.'].to_list()[-1]
            already_tested_is = [i - 1 for i in df['No.'].to_list()]
        except IndexError:
            start_i = 0
            already_tested_is = []
        print(list(combined_mut_data.keys())[start_i-1])
        print(f'Already tested datasets:')
        for i in already_tested_is:
            print(f'{i + 1} {list(combined_mut_data.keys())[i]}')
        try:
            print(f'\nContinuing getting model performances at {start_i + 1} {list(combined_mut_data.keys())[start_i]} '
                  f'(last tested dataset: {start_i}, {list(combined_mut_data.keys())[start_i - 1]})')
        except IndexError:
            print('\nComputed all results already?!')
    else:
        with open(out_results_csv, 'w') as fh:
            print(f'\nCreating new file {out_results_csv}...')
            fh.write(
                f'No.,Dataset,N_Variants,N_Max_Muts,Untrained_Performance_DCA,Untrained_Performance_LLM,'
                f'Hybrid_Trained_Performance_100,Hybrid_Trained_Performance_200,'
                f'Hybrid_Trained_Performance_1000,N_Y_test,N_Y_test_100,N_Y_test_200,N_Y_test_1000,Time_in_s\n'
            )
            start_i = 0
            already_tested_is = []


    compute_performances(
        mut_data=combined_mut_data, 
        start_i=start_i, 
        already_tested_is=already_tested_is
    )


    with open(out_results_csv, 'r') as fh:
        with open(os.path.join(os.path.dirname(__file__), 'results/dca_esm_and_hybrid_opt_results_clean.csv'), 'w') as fh2:
            for line in fh:
                if not line.split(',')[1].startswith('OOM') and not line.split(',')[1].startswith('X'):
                    fh2.write(line)
    
    plot_csv_data(csv=os.path.join(os.path.dirname(__file__), 'results/dca_esm_and_hybrid_opt_results_clean.csv'), plot_name='mut_performance')
