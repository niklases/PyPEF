
import os
import sys
import json
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
# Add local PyPEF path if not using pip-installed PyPEF version
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from pypef.dca.gremlin_inference import GREMLIN
from pypef.dca.hybrid_model import get_delta_e_statistical_model, remove_gap_pos
from pypef.utils.variant_data import get_seqs_from_var_name


single_point_mut_data = os.path.abspath(os.path.join(os.path.dirname(__file__), f"single_point_dms_mut_data.json"))
higher_mut_data = os.path.abspath(os.path.join(os.path.dirname(__file__), f"higher_point_dms_mut_data.json"))


def plot_performance(mut_data, plot_name, mut_sep=':'):
    tested_dsets = []
    dset_perfs = []
    for i, (dset_key, dset_paths) in enumerate(mut_data.items()):
            #if i < 3:
            #try:
            print('\n', i, '\n===============================================================')
            csv_path = dset_paths['CSV_path']
            msa_path = dset_paths['MSA_path']
            wt_seq = dset_paths['WT_sequence']
            print(msa_path)
            variant_fitness_data = pd.read_csv(csv_path, sep=',')
            variants = variant_fitness_data['mutant']
            fitnesses = variant_fitness_data['DMS_score']
            variants_split = []
            for variant in variants:
                # Split double and higher substituted variants to multiple single substitutions; 
                # e.g. separated by ':' or '/'
                variants_split.append(variant.split(mut_sep))
            variants, fitnesses, sequences = get_seqs_from_var_name(wt_seq, variants_split, fitnesses)
            # Only model sequences with length of max. 800 amino acids to avoid out of memory errors 
            if len(wt_seq) > 800:
                print('Sequence length over 800, continuing...')
                continue
            gremlin_new = GREMLIN(alignment=msa_path, wt_seq=wt_seq, max_msa_seqs=10000)
            #gaps = gremlin_new.gaps
            gaps_1_indexed = gremlin_new.gaps_1_indexed
            var_pos = [int(v[1:-1]) for variants in variants_split for v in variants]
            n_muts = []
            for vs in variants_split:
                n_muts.append(len(vs))
            max_muts = max(n_muts)
            c = 0
            for vp in var_pos:
                if vp in gaps_1_indexed:
                    c += 1
            print(f'N max. (multiple) amino acid substitutions: {max_muts}')
            c = c / max_muts
            ratio_input_vars_at_gaps = c / len(var_pos)
            if c > 0:
                print(f'{c} of {len(var_pos)} ({ratio_input_vars_at_gaps * 100:.2f}%) input variants to be predicted are variants with '
                      f'amino acid substitutions at gap positions (these variants will be predicted/labeled with a fitness of 0.0).')
            #variants, sequences, fitnesses = remove_gap_pos(gaps, variants, sequences, fitnesses)
            x_dca = gremlin_new.collect_encoded_sequences(sequences)
            x_wt = gremlin_new.x_wt
            # Statistical model performance
            y_pred_new = get_delta_e_statistical_model(x_dca, x_wt)
            print(f'Statistical DCA model performance on all datapoints; Spearman\'s rho: {abs(spearmanr(fitnesses, y_pred_new)[0]):.3f}')
            assert len(x_dca) == len(fitnesses) == len(variants) == len(sequences)
            #except SystemError:  # Check MSAs
            #   continue
            tested_dsets.append(f'{dset_key} ({100.0 - (ratio_input_vars_at_gaps * 100):.2f}%, ' 
                                + r'$N_\mathrm{AASubs:}$' + f'{max_muts})')
            dset_perfs.append(abs(spearmanr(fitnesses, y_pred_new)[0]))
    plt.figure(figsize=(20, 12))
    plt.plot(range(len(tested_dsets)), dset_perfs, 'o--', markersize=12)
    plt.plot(range(len(tested_dsets)), np.full(np.shape(tested_dsets), np.mean(dset_perfs)), 'k--')
    plt.text(len(tested_dsets) - 1, np.mean(dset_perfs), f'{np.mean(dset_perfs):.2f}')
    plt.xticks(range(len(tested_dsets)), tested_dsets, rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), f'{plot_name}.png'), dpi=300)
    print('Saved file as ' + os.path.join(os.path.dirname(__file__),  f'{plot_name}.png') + '.')


with open(single_point_mut_data, 'r') as fh:
    s_mut_data = json.loads(fh.read())
with open(higher_mut_data, 'r') as fh:
    h_mut_data = json.loads(fh.read())
plot_performance(mut_data=s_mut_data, plot_name='single_point_mut_performance')
plot_performance(mut_data=h_mut_data, plot_name='multi_point_mut_performance')
