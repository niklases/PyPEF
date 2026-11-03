from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from pypef.plm.pmpnn.get_cond_probs import score_sequences_from_probs
from pypef.utils.ssm import SSM


def test_pmpnn_ssm_matches_pmpnn_zero_shot_scoring(tmp_path, monkeypatch):
    wt = 'MKAC'
    log_p = np.random.default_rng(0).normal(size=(3, len(wt), 21))  # 3 decoding draws
    monkeypatch.chdir(tmp_path)  # SSM writes SSM_landscape.png/.csv to the cwd
    with patch('pypef.utils.ssm.run_protein_mpnn_conditional', return_value=log_p) as compute:
        ssm = SSM(wt_seq=wt, model='PMPNN', pdb='structure.pdb')
    compute.assert_called_once_with('structure.pdb', seed=42)
    sequences = np.array(ssm.variant_sequencess).flatten()
    expected = score_sequences_from_probs(log_p, sequences, wt)
    np.testing.assert_allclose(np.array(ssm.scoress).flatten(), expected)
    assert np.shape(ssm.scoress) == (len(wt), 20)
    # WT -> WT "substitutions" score zero
    for i, aa in enumerate(wt):
        assert ssm.scoress[i][ssm.aas.index(aa)] == 0
    df = pd.read_csv(tmp_path / 'SSM_landscape.csv')
    assert len(df) == len(wt) * 20
    assert (tmp_path / 'SSM_landscape.png').is_file()


@pytest.mark.parametrize('model', ['esm+prosst', 'esm,pmpnn', 'prosst pmpnn'])
def test_ssm_refuses_plm_combinations(model):
    with pytest.raises(ValueError, match='single model'):
        SSM(wt_seq='MKAC', model=model, pdb='structure.pdb')


@pytest.mark.parametrize('model', ['prosst', 'pmpnn'])
def test_structure_based_ssm_requires_pdb(model):
    with pytest.raises(ValueError, match='requires a PDB'):
        SSM(wt_seq='MKAC', model=model)
