from unittest.mock import patch

import numpy as np
import pytest

from pypef.hybrid.hybrid_model import predict_ps, performance_ls_ts
from pypef.plm.pmpnn.get_cond_probs import AA_TO_IDX


def conditional_probs():
    probs = np.zeros((2, 21))
    probs[0, AA_TO_IDX['C']] = 2.0
    return probs


def test_pmpnn_prediction_scores_and_ranks_sequences():
    with patch('pypef.hybrid.hybrid_model.get_sequences_from_file',
               return_value=(['AA', 'CA'], ['WT', 'A1C'], [])), \
         patch('pypef.hybrid.hybrid_model.run_protein_mpnn_conditional',
               return_value=conditional_probs()) as compute, \
         patch('pypef.hybrid.hybrid_model.predictions_out') as output:
        predict_ps({}, 1, '/', prediction_set='variants.fa', llm='pmpnn',
                   pdb_file='structure.pdb', wt_seq='AA', device='cpu')
    compute.assert_called_once_with('structure.pdb')
    assert output.call_args.kwargs['predictions'] == [(2.0, 'A1C'), (0.0, 'WT')]
    assert output.call_args.kwargs['model'] == 'PLM_PMPNN'


def test_pmpnn_zero_shot_test_does_not_require_learning_set():
    with patch('pypef.hybrid.hybrid_model.get_sequences_from_file',
               return_value=(['AA', 'CA'], ['WT', 'A1C'], [0.0, 1.0])), \
         patch('pypef.hybrid.hybrid_model.run_protein_mpnn_conditional',
               return_value=conditional_probs()) as compute, \
         patch('pypef.hybrid.hybrid_model.plot_y_true_vs_y_pred') as plot:
        performance_ls_ts(None, 'test.fasl', 1, None, llm='pmpnn',
                          pdb_file='structure.pdb', wt_seq='AA', device='cpu', seed=42)
    compute.assert_called_once_with('structure.pdb', seed=42)
    np.testing.assert_array_equal(plot.call_args.args[1], [0.0, 2.0])


@pytest.mark.parametrize('pdb,wt', [(None, 'AA'), ('structure.pdb', None)])
def test_pmpnn_prediction_requires_structure_and_wt(pdb, wt):
    with pytest.raises(ValueError, match='requires --pdb and --wt'):
        predict_ps({}, 1, '/', prediction_set='variants.fa', llm='pmpnn',
                   pdb_file=pdb, wt_seq=wt, device='cpu')
