

import os
import sys
import json
import pandas as pd
from scipy.stats import spearmanr
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
print(sys.path)
from pypef.dca.gremlin_inference import GREMLIN as GREMLIN_OLD
from pypef.dca.new_gremlin_inference import GREMLIN as GREMLIN_NEW
from pypef.dca.hybrid_model import remove_gap_pos, get_delta_e_statistical_model
from pypef.utils.variant_data import get_seqs_from_var_name


single_point_mut_data = os.path.abspath(os.path.join(os.path.dirname(__file__), f"single_point_dms_mut_data.json"))
higher_mut_data = os.path.abspath(os.path.join(os.path.dirname(__file__), f"higher_point_dms_mut_data.json"))


with open(single_point_mut_data, 'r') as fh:
    mut_data = json.loads(fh.read())



for i, (dset_key, dset_paths) in enumerate(mut_data.items()):
        #try:
        print(i, '\n===============================================================')
        csv_path = dset_paths['CSV_path']
        msa_path = dset_paths['MSA_path']
        wt_seq = dset_paths['WT_sequence']
        variant_fitness_data = pd.read_csv(csv_path, sep=',')
        variants = variant_fitness_data['mutant']
        fitnesses = variant_fitness_data['DMS_score']
        variants_split = []
        for variant in variants:
            variants_split.append(variant.split('/'))
        variants, fitnesses, sequences = get_seqs_from_var_name(wt_seq, variants_split, fitnesses)
        ####
        with open(msa_path, 'r') as fh:
            cnt = 0
            for line in fh: 
                cnt += 1
        if cnt > 100000:
            print('Too big MSA, continuing...')
            continue
        gremlin_old = GREMLIN_OLD(alignment=msa_path, wt_seq=wt_seq)
        gaps = gremlin_old.gaps
        variants, sequences, fitnesses = remove_gap_pos(gaps, variants, sequences, fitnesses)
        x_dca = gremlin_old.collect_encoded_sequences(sequences)
        print(f'N Variants remaining after excluding non-DCA-encodable positions: {len(x_dca)}')
        x_wt = gremlin_old.x_wt
        # Statistical model performance
        y_pred = get_delta_e_statistical_model(x_dca, x_wt)
        print(f'Statistical DCA model performance on all datapoints; Spearman\'s rho: {spearmanr(fitnesses, y_pred)[0]:.3f}')
        # Split double and higher substituted variants to multiple single substitutions separated by '/'
        assert len(x_dca) == len(fitnesses) == len(variants) == len(sequences)
        ####
        gremlin_new = GREMLIN_NEW(alignment=msa_path, wt_seq=wt_seq, max_msa_seqs=None)
        gaps = gremlin_new.gaps
        variants, sequences, fitnesses = remove_gap_pos(gaps, variants, sequences, fitnesses)
        x_dca = gremlin_new.collect_encoded_sequences(sequences)
        print(f'N Variants remaining after excluding non-DCA-encodable positions: {len(x_dca)}')
        x_wt = gremlin_new.x_wt
        # Statistical model performance
        y_pred = get_delta_e_statistical_model(x_dca, x_wt)
        print(f'Statistical DCA model performance on all datapoints; Spearman\'s rho: {spearmanr(fitnesses, y_pred)[0]:.3f}')
        # Split double and higher substituted variants to multiple single substitutions separated by '/'
        assert len(x_dca) == len(fitnesses) == len(variants) == len(sequences)
        #except SystemError:  # Check MSAs
        #   continue

