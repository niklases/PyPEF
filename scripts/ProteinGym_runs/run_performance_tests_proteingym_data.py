

import os
import sys
import json
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
print(sys.path)
from pypef.dca.gremlin_inference import GREMLIN as GREMLIN_OLD
from pypef.dca.new_gremlin_inference import GREMLIN as GREMLIN_NEW
from pypef.dca.hybrid_model import remove_gap_pos, get_delta_e_statistical_model
from pypef.utils.variant_data import get_seqs_from_var_name


single_point_mut_data = os.path.abspath(os.path.join(os.path.dirname(__file__), f"single_point_dms_mut_data.json"))
higher_mut_data = os.path.abspath(os.path.join(os.path.dirname(__file__), f"higher_point_dms_mut_data.json"))




def plot_performance(mut_data, plot_name):
    tested_dsets = []
    dset_perfs = []
    for i, (dset_key, dset_paths) in enumerate(mut_data.items()):
            #if i <= 3:
            #try:
            print('\n', i, '\n===============================================================')
            print('#'*60 + ' OLD MODEL ' + '#'*60)
            csv_path = dset_paths['CSV_path']
            msa_path = dset_paths['MSA_path']
            wt_seq = dset_paths['WT_sequence']
            print(msa_path)
            variant_fitness_data = pd.read_csv(csv_path, sep=',')
            variants = variant_fitness_data['mutant']
            fitnesses = variant_fitness_data['DMS_score']
            variants_split = []
            for variant in variants:
                variants_split.append(variant.split('/'))
            variants, fitnesses, sequences = get_seqs_from_var_name(wt_seq, variants_split, fitnesses)
            ####
            if len(wt_seq) > 800:
                print('Sequence length over 800, continuing...')
                continue
            gremlin_old = GREMLIN_OLD(alignment=msa_path, wt_seq=wt_seq, max_msa_seqs=10000)
            gaps = gremlin_old.gaps
            variants, sequences, fitnesses = remove_gap_pos(gaps, variants, sequences, fitnesses)
            x_dca = gremlin_old.collect_encoded_sequences(sequences)
            print(f'N Variants remaining after excluding non-DCA-encodable positions: {len(x_dca)}')
            x_wt = gremlin_old.x_wt
            # Statistical model performance
            y_pred_old = get_delta_e_statistical_model(x_dca, x_wt)
            print(f'Statistical DCA model performance on all datapoints; Spearman\'s rho: {spearmanr(fitnesses, y_pred_old)[0]:.3f}')
            # Split double and higher substituted variants to multiple single substitutions separated by '/'
            assert len(x_dca) == len(fitnesses) == len(variants) == len(sequences)
            print('#'*60 + ' NEW MODEL ' + '#'*60)
            ####
            gremlin_new = GREMLIN_NEW(alignment=msa_path, wt_seq=wt_seq, max_msa_seqs=10000)
            gaps = gremlin_new.gaps
            #variants, sequences, fitnesses = remove_gap_pos(gaps, variants, sequences, fitnesses)
            x_dca = gremlin_new.collect_encoded_sequences(sequences)
            print(f'N Variants remaining after excluding non-DCA-encodable positions: {len(x_dca)}')
            x_wt = gremlin_new.x_wt
            # Statistical model performance
            y_pred_new = get_delta_e_statistical_model(x_dca, x_wt)
            print(f'Statistical DCA model performance on all datapoints; Spearman\'s rho: {spearmanr(fitnesses, y_pred_new)[0]:.3f}')
            # Split double and higher substituted variants to multiple single substitutions separated by '/'
            assert len(x_dca) == len(fitnesses) == len(variants) == len(sequences)
            np.testing.assert_almost_equal(spearmanr(fitnesses, y_pred_old)[0], spearmanr(fitnesses, y_pred_new)[0], decimal=3)
            #except SystemError:  # Check MSAs
            #   continue
            #exit()
            tested_dsets.append(dset_key)
            dset_perfs.append(spearmanr(fitnesses, y_pred_new)[0])
    plt.figure(figsize=(20, 12))
    plt.plot(range(len(tested_dsets)), dset_perfs)
    plt.xticks(range(len(tested_dsets)), tested_dsets, rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), f'{plot_name}.png'), dpi=300)
    print('Saved file as ' + os.path.join(os.path.dirname(__file__),  f'{plot_name}.png') + '.')


with open(single_point_mut_data, 'r') as fh:
    s_mut_data = json.loads(fh.read())
with open(higher_mut_data, 'r') as fh:
    h_mut_data = json.loads(fh.read())
plot_performance(mut_data=s_mut_data, plot_name='single_point_mut_performance')
#plot_performance(mut_data=h_mut_data, plot_name='multi_point_mut_performance')

