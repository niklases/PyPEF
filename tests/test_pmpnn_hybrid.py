import pickle

import numpy as np
import pytest

from pypef.hybrid.hybrid_model import DCALLMHybridModel
from pypef.plm.pmpnn.get_cond_probs import (
    load_conditional_log_probs, score_sequences_from_probs,
)


def test_conditional_scores_and_input_formats(tmp_path):
    log_p = np.zeros((2, 2, 21))
    log_p[:, 0, 1] = [2, 4]  # A -> C
    log_p[:, 1, 2] = [4, 6]  # C -> D
    path = tmp_path / 'conditional.npz'
    np.savez(path, log_p=log_p)
    for source in (log_p, {'conditional_probs': log_p}, path, log_p.mean(0)):
        np.testing.assert_allclose(
            score_sequences_from_probs(source, ['AC', 'CC', 'AD', 'CD'], 'AC'),
            [0, 3, 5, 8],
        )
    with pytest.raises(ValueError, match='length'):
        score_sequences_from_probs(log_p, ['A'], 'AC')
    with pytest.raises(ValueError, match='shape'):
        load_conditional_log_probs(log_p, 'ACC')
    with pytest.raises(ValueError, match='unsupported'):
        score_sequences_from_probs(log_p, ['A-'], 'AC')


@pytest.mark.parametrize('ensemble_func', ['torch', 'de'])
def test_hybrid_pmpnn_multisplit_and_pickle(ensemble_func, monkeypatch):
    monkeypatch.setattr(DCALLMHybridModel, "_train_pmpnn_gp", lambda self: None)
    monkeypatch.setattr(DCALLMHybridModel, "_predict_pmpnn_gp",
                        lambda self, sequences: self._pmpnn_scores(sequences))
    rng = np.random.default_rng(8)
    sequences = ['AC', 'CC', 'AD', 'CD'] * 5
    log_p = np.zeros((2, 21))
    log_p[0, 1] = 3
    log_p[1, 2] = 5
    targets = score_sequences_from_probs(log_p, sequences, 'AC') + rng.normal(0, .1, 20)
    x = rng.normal(size=(20, 2))
    model = DCALLMHybridModel(
        x, targets, sequences=sequences, wt_sequence='AC',
        pmpnn_conditional_probs={'log_p': log_p}, x_dca_wt=np.zeros(2),
        pdb_struct='mock.pdb',
        alphas=np.array([1.]), batch_size=1, seed=8, device='cpu',
        n_ensemble_splits=2, ensemble_func=ensemble_func,
    )
    assert model.feature_names == ['y_dca', 'y_ridge', 'PMPNN_base', 'PMPNN_gp']
    assert len(model.all_betas) == 4
    np.testing.assert_allclose(model.pmpnn_aa_cond_probs.cpu().sum(-1), 1, atol=1e-6)
    predictions, components = model.hybrid_prediction(x, sequences=sequences)
    np.testing.assert_allclose(components['PMPNN_base'], score_sequences_from_probs(log_p, sequences, 'AC'))
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_allclose(restored.hybrid_prediction(x, sequences=sequences)[0], predictions)
    with pytest.raises(ValueError, match='requires variant sequences'):
        restored.hybrid_prediction(x)


def test_separate_pmpnn_gp(monkeypatch):
    import torch
    import pypef.plm.pmpnn.gp as hybrid

    optimize = hybrid.optimize_gp
    monkeypatch.setattr(hybrid, 'optimize_gp',
                        lambda gp, likelihood, inputs, targets, **kw:
                        optimize(gp, likelihood, inputs, targets, n_steps=2))
    monkeypatch.setattr(hybrid, 'extract_pdb_coords',
                        lambda *args, **kw: np.array([[0., 0., 0.], [1., 0., 0.]]))
    rng = np.random.default_rng(19)
    sequences = ['CC', 'AD', 'CD'] * 8
    x = rng.normal(size=(24, 2))
    log_p = rng.normal(size=(2, 21))
    model = DCALLMHybridModel(
        x, rng.normal(size=24), x_dca_wt=np.zeros(2),
        sequences=sequences, wt_sequence='AC', pmpnn_conditional_probs=log_p,
        pdb_struct='mock.pdb', device='cpu',
        alphas=np.array([1.]), batch_size=1, seed=19, n_ensemble_splits=2,
    )
    assert not model.gauss_opt
    assert model.feature_names == ['y_dca', 'y_ridge', 'PMPNN_base', 'PMPNN_gp']
    kernel = model.pmpnn_gp.covar_module.structure_kernel.base_kernel
    torch.testing.assert_close(kernel.k_p.conditional_probs, model.pmpnn_aa_cond_probs)
    prediction, components = model.hybrid_prediction(x, sequences=sequences)
    assert np.isfinite(components['PMPNN_gp']).all()
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_allclose(restored.hybrid_prediction(x, sequences=sequences)[0], prediction)


def test_pmpnn_gp_requires_inputs():
    with pytest.raises(ValueError, match='PMPNN GP requires'):
        DCALLMHybridModel(
            np.zeros((20, 2)), np.zeros(20), sequences=['AC'] * 20,
            wt_sequence='AC', pmpnn_conditional_probs=np.zeros((2, 21)),
        )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_prior_scoring_exact_equality(dtype):
    """Reference the original mean-and-mutation-loop calculation exactly."""
    from pypef.plm.pmpnn.get_cond_probs import ALPHABET
    rng = np.random.default_rng(37)
    wt = 'ACDEFGHIKLMNPQRSTVWY'
    log_p = rng.normal(size=(10, len(wt), 21)).astype(dtype)
    sequences = [wt, 'CCDEFGHIKLMNPQRSTVWY', 'ACDEYGHIKLMNPQRSTVWA',
                 'Y' * len(wt)]
    mean_log_p = np.mean(log_p, axis=0)
    original = []
    for sequence in sequences:
        delta_ll = 0.0
        for i, (wt_aa, mut_aa) in enumerate(zip(wt, sequence)):
            if wt_aa != mut_aa:
                delta_ll += mean_log_p[i, ALPHABET.index(mut_aa)] - mean_log_p[i, ALPHABET.index(wt_aa)]
        original.append(delta_ll)
    np.testing.assert_array_equal(score_sequences_from_probs(log_p, sequences, wt), original)


def test_cli_pmpnn_arguments():
    from docopt import docopt
    from pypef.main import __doc__, validate
    args = validate(docopt(__doc__, argv=[
        'hybrid', '--ls', 'train.fasta', '--ts', 'test.fasta', '--params', 'gremlin',
        '--pdb', 'protein.pdb', '--wt', 'wt.fasta', '--pmpnn',
        '--pmpnn_cond_probs', 'conditional.npz',
    ]))
    assert args['--pmpnn'] is True
    assert args['--pmpnn_cond_probs'] == 'conditional.npz'


def test_cli_execution_passes_pmpnn_inputs(monkeypatch):
    from docopt import docopt
    from pypef.main import __doc__, validate
    import pypef.hybrid.hybrid_run as runner
    args = validate(docopt(__doc__, argv=[
        'hybrid', '--ls', 'train.fasta', '--ts', 'test.fasta',
        '--pmpnn_cond_probs', 'conditional.npz', '--pdb', 'protein.pdb',
        '--wt', 'wt.fasta',
    ]))
    calls = []
    monkeypatch.setattr(runner, 'get_wt_sequence', lambda path: 'AC')
    monkeypatch.setattr(runner, 'performance_ls_ts', lambda **kwargs: calls.append(kwargs))
    runner.run_pypef_hybrid_modeling(args)
    assert calls[0]['pmpnn_conditional_probs'] == 'conditional.npz'
    assert calls[0]['pdb_file'] == 'protein.pdb'
    assert calls[0]['wt_seq'] == 'AC'


def test_plm_pmpnn_selects_in_memory_setup(monkeypatch):
    import pypef.hybrid.hybrid_model as hybrid
    import pypef.plm.pmpnn.get_cond_probs as pmpnn
    calls = []
    monkeypatch.setattr(hybrid, 'run_protein_mpnn_conditional',
                        lambda *args, **kwargs: calls.append((args, kwargs)) or {'log_p': np.zeros((2, 21))})
    # Stop after setup to avoid loading datasets or PLM weights.
    def stop(*args):
        raise RuntimeError('setup complete')
    monkeypatch.setattr(hybrid, 'get_sequences_from_file', stop)
    with pytest.raises(RuntimeError, match='setup complete'):
        hybrid.performance_ls_ts('train.fasta', 'test.fasta', 1, 'gremlin',
                                 llm='esm+prosst+pmpnn', pdb_file='protein.pdb',
                                 wt_seq='AC', device='cpu')
    assert calls[0][0] == ('protein.pdb',)


def test_mutation_sequence_inputs():
    from pypef.plm.utils import resolve_variant_mutations
    assert resolve_variant_mutations('ACD', mutation_string='D3A/A1C') == ('CCA', [0, 2])
    assert resolve_variant_mutations('ACD', 'CCA', 'A1C/D3A') == ('CCA', [0, 2])
    assert resolve_variant_mutations('ACD', mutation_string='WT') == ('ACD', [])
    for mutation in ('A0C', 'A4C', 'C1A', 'A1C/A1D', 'invalid'):
        with pytest.raises(ValueError):
            resolve_variant_mutations('ACD', mutation_string=mutation)
    with pytest.raises(ValueError, match='different variants'):
        resolve_variant_mutations('ACD', 'CCD', 'D3A')
    log_p = np.random.default_rng(2).normal(size=(4, 3, 21)).astype(np.float32)
    expected = score_sequences_from_probs(log_p, ['ACD', 'CCA'], 'ACD')
    actual = score_sequences_from_probs(log_p, wt_seq='ACD', mutation_strings=['WT', 'D3A/A1C'])
    np.testing.assert_array_equal(actual, expected)
