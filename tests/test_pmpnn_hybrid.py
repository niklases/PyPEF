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


@pytest.mark.parametrize('gauss_opt', [False, True])
@pytest.mark.parametrize('ensemble_func', ['torch', 'de'])
def test_hybrid_pmpnn_multisplit_and_pickle(ensemble_func, gauss_opt, monkeypatch):
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
        ridge_alphas=np.array([1.]), batch_size=1, seed=8, device='cpu',
        n_ensemble_splits=2, ensemble_func=ensemble_func, gauss_opt=gauss_opt,
    )
    # The PMPNN GP follows gauss_opt, like the ESM/ProSST GPs.
    features = ['y_dca', 'y_ridge', 'PMPNN_base'] + (['PMPNN_gp'] if gauss_opt else [])
    assert model.feature_names == features
    assert len(model.all_betas) == len(features)
    np.testing.assert_allclose(model.pmpnn_aa_cond_probs.cpu().sum(-1), 1, atol=1e-6)
    predictions, components = model.hybrid_prediction(x, sequences=sequences)
    assert ('PMPNN_gp' in components) == gauss_opt
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
        ridge_alphas=np.array([1.]), batch_size=1, seed=19, n_ensemble_splits=2,
        gauss_opt=True,
    )
    assert model.feature_names == ['y_dca', 'y_ridge', 'PMPNN_base', 'PMPNN_gp']
    kernel = model.pmpnn_gp.covar_module.structure_kernel.base_kernel
    torch.testing.assert_close(kernel.k_p.conditional_probs, model.pmpnn_aa_cond_probs)
    prediction, components = model.hybrid_prediction(x, sequences=sequences)
    assert np.isfinite(components['PMPNN_gp']).all()
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_allclose(restored.hybrid_prediction(x, sequences=sequences)[0], prediction)


def test_pmpnn_gp_requires_inputs():
    with pytest.raises(RuntimeError, match='requires .* structure file'):
        DCALLMHybridModel(
            np.zeros((20, 2)), np.zeros(20), sequences=['AC'] * 20,
            wt_sequence='AC', pmpnn_conditional_probs=np.zeros((2, 21)),
            gauss_opt=True,
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


@pytest.mark.parametrize('ensemble_func', ['torch', 'de'])
def test_hybrid_prediction_independent_of_batch(ensemble_func):
    """
    Sub-model predictions are standardized with statistics stored at fitting time,
    not with the statistics of the predicted batch: a variant's prediction must not
    depend on which other variants are predicted with it. Batch-wise standardization
    removed the mean level of each batch (e.g. of a held-out sequence region in
    contiguous CV splits), so that a batch of only low-fitness variants was shifted
    to the mean of the training data.
    """
    rng = np.random.default_rng(3)
    sequences = ['AC', 'CC', 'AD', 'CD'] * 6
    log_p = np.zeros((2, 21))
    log_p[0, 1] = 3
    log_p[1, 2] = 5
    targets = score_sequences_from_probs(log_p, sequences, 'AC') + rng.normal(0, .1, 24)
    x = rng.normal(size=(24, 2))
    model = DCALLMHybridModel(
        x, targets, sequences=sequences, wt_sequence='AC',
        pmpnn_conditional_probs={'log_p': log_p}, x_dca_wt=np.zeros(2),
        ridge_alphas=np.array([1.]), batch_size=1, seed=3, device='cpu',
        n_ensemble_splits=2, ensemble_func=ensemble_func,
    )
    assert model.feature_means.shape == model.feature_stds.shape == (len(model.feature_names),)
    full, _ = model.hybrid_prediction(x, sequences=sequences)
    low = [i for i, s in enumerate(sequences) if s == 'AC']  # lowest-fitness variants only
    for idx in (low, [0], list(range(5))):
        subset, _ = model.hybrid_prediction(x[idx], sequences=[sequences[i] for i in idx])
        np.testing.assert_allclose(subset, full[idx], rtol=1e-6, atol=1e-8)
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_allclose(restored.feature_means, model.feature_means)
    np.testing.assert_allclose(restored.hybrid_prediction(x[low], sequences=['AC'] * len(low))[0], full[low])


def test_refit_full_train_and_clipping(monkeypatch):
    """Refit of supervised sub-models on all training data keeps weights and statistics."""
    import pypef.plm.pmpnn.gp as pmpnn_gp
    optimize = pmpnn_gp.optimize_gp
    monkeypatch.setattr(pmpnn_gp, 'optimize_gp',
                        lambda gp, likelihood, inputs, targets, **kw:
                        optimize(gp, likelihood, inputs, targets, n_steps=2))
    monkeypatch.setattr(pmpnn_gp, 'extract_pdb_coords',
                        lambda *args, **kw: np.array([[0., 0., 0.], [1., 0., 0.]]))
    rng = np.random.default_rng(5)
    sequences = ['CC', 'AD', 'CD'] * 8
    x = rng.normal(size=(24, 2))
    y = rng.normal(size=24)
    kwargs = dict(
        x_dca_wt=np.zeros(2), sequences=sequences, wt_sequence='AC',
        pmpnn_conditional_probs=rng.normal(size=(2, 21)), pdb_struct='mock.pdb', device='cpu',
        ridge_alphas=np.array([1.]), batch_size=1, seed=5, gauss_opt=True,
    )
    base = DCALLMHybridModel(x, y, **kwargs)
    refit = DCALLMHybridModel(x, y, refit_full_train=True, **kwargs)
    np.testing.assert_allclose(refit.all_betas, base.all_betas)
    np.testing.assert_allclose(refit.feature_means, base.feature_means)
    # Sub-models were retrained on all 24 instead of the ttrain subsplit variants
    assert len(base.pmpnn_gp.train_targets) < len(refit.pmpnn_gp.train_targets) == 24
    assert not np.allclose(refit.ridge_opt.coef_, base.ridge_opt.coef_)
    pred, _ = refit.hybrid_prediction(x, sequences=sequences)
    assert np.isfinite(pred).all()
    # Clipping bounds the contribution of off-scale sub-model predictions
    x_far = x * 1e4
    unclipped, _ = base.hybrid_prediction(x_far, sequences=sequences)
    base.clip_std_preds = 4.0
    clipped, _ = base.hybrid_prediction(x_far, sequences=sequences)
    bound = abs(base.ols_slope) * 4.0 * np.sum(np.abs(base.all_betas)) + abs(base.ols_intercept)
    assert np.abs(clipped).max() <= bound + 1e-6
    assert np.abs(unclipped).max() >= np.abs(clipped).max()


def test_gp_standardized_targets_and_pmpnn_probs(monkeypatch):
    """GP targets are standardized with fixed statistics; PMPNN probabilities replace PLM ones."""
    import pypef.plm.pmpnn.gp as pmpnn_gp
    import torch
    seen = []
    optimize = pmpnn_gp.optimize_gp

    def optimize_spy(gp, likelihood, inputs, targets, **kw):
        seen.append(targets.detach().cpu().numpy())
        return optimize(gp, likelihood, inputs, targets, n_steps=2)
    monkeypatch.setattr(pmpnn_gp, 'optimize_gp', optimize_spy)
    monkeypatch.setattr(pmpnn_gp, 'extract_pdb_coords',
                        lambda *args, **kw: np.array([[0., 0., 0.], [1., 0., 0.]]))
    rng = np.random.default_rng(13)
    sequences = ['CC', 'AD', 'CD'] * 8
    x = rng.normal(size=(24, 2))
    y = 50 + 10 * rng.normal(size=24)  # far from zero mean / unit variance
    kwargs = dict(
        x_dca_wt=np.zeros(2), sequences=sequences, wt_sequence='AC',
        pmpnn_conditional_probs=rng.normal(size=(2, 21)), pdb_struct='mock.pdb', device='cpu',
        ridge_alphas=np.array([1.]), batch_size=1, seed=13, gauss_opt=True,
    )
    DCALLMHybridModel(x, y, **kwargs)
    assert abs(seen[-1].mean()) > 10  # raw targets by default
    model = DCALLMHybridModel(x, y, gp_standardize_targets=True, refit_full_train=True, **kwargs)
    ttrain, full = seen[-2], seen[-1]
    assert len(full) == 24 > len(ttrain)
    np.testing.assert_allclose([ttrain.mean(), ttrain.std()], [0, 1], atol=1e-5)
    mean, std = model._gp_target_stats  # statistics of the first fit are kept for the refit
    np.testing.assert_allclose(full, (y - mean) / std, rtol=1e-5)
    pred, _ = model.hybrid_prediction(x, sequences=sequences)
    assert np.isfinite(pred).all()
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_allclose(restored.hybrid_prediction(x, sequences=sequences)[0], pred)

    plm_probs = torch.full((2, 20), 0.05, dtype=torch.float64)
    assert model._struct_kernel_probs(plm_probs) is plm_probs
    model.gp_pmpnn_probs = True
    used = model._struct_kernel_probs(plm_probs)
    assert used.dtype == plm_probs.dtype
    np.testing.assert_allclose(used.numpy(), model.pmpnn_aa_cond_probs.numpy())
    assert model._struct_kernel_probs(plm_probs[:1]) is not model.pmpnn_aa_cond_probs  # shape fallback
    assert model._struct_kernel_probs(plm_probs[:1]).shape == (1, 20)


def test_gp_predictions_on_target_scale(monkeypatch):
    """With standardized GP targets, GP predictions are mapped back to the scale of the targets
    instead of staying in units of training standard deviations."""
    import pypef.plm.pmpnn.gp as pmpnn_gp
    monkeypatch.setattr(pmpnn_gp, 'extract_pdb_coords',
                        lambda *args, **kw: np.array([[0., 0., 0.], [1., 0., 0.]]))
    rng = np.random.default_rng(21)
    sequences = ['CC', 'AD', 'CD'] * 8
    x = rng.normal(size=(24, 2))
    y = 50 + 10 * rng.normal(size=24)  # far from zero mean / unit variance
    kwargs = dict(
        x_dca_wt=np.zeros(2), sequences=sequences, wt_sequence='AC',
        pmpnn_conditional_probs=rng.normal(size=(2, 21)), pdb_struct='mock.pdb', device='cpu',
        ridge_alphas=np.array([1.]), batch_size=1, seed=21, gauss_opt=True,
    )
    model = DCALLMHybridModel(x, y, gp_standardize_targets=True, **kwargs)
    pred, components = model.hybrid_prediction(x, sequences=sequences)
    gp_pred = components['PMPNN_gp']
    # On the target scale (mean ~ 50), not in units of training standard deviations (mean ~ 0)
    assert abs(np.mean(gp_pred) - np.mean(y)) < 3 * np.std(y)
    assert abs(np.mean(gp_pred)) > 1.0
    assert np.isfinite(pred).all()

    # Without the flag nothing is transformed, and the raw targets are already on their own scale
    plain = DCALLMHybridModel(x, y, **kwargs)
    assert plain._gp_untransform(np.arange(3.0)).tolist() == [0.0, 1.0, 2.0]
    plain_gp = plain.hybrid_prediction(x, sequences=sequences)[1]['PMPNN_gp']
    assert abs(np.mean(plain_gp) - np.mean(y)) < 3 * np.std(y)


def test_comb_all_zero_shot_matrix():
    """The combined GP's zero-shot matrix: extra columns, standardization with kept statistics."""
    import torch
    rng = np.random.default_rng(5)
    sequences = ['CC', 'AD', 'CD'] * 8
    x = rng.normal(size=(24, 2))
    y = 50 + 10 * rng.normal(size=24)
    model = DCALLMHybridModel(
        x, y, x_dca_wt=np.zeros(2), sequences=sequences, wt_sequence='AC',
        pmpnn_conditional_probs=rng.normal(size=(2, 21)), pdb_struct='mock.pdb', device='cpu',
        ridge_alphas=np.array([1.]), batch_size=1, seed=5,
    )
    plm_scores = [torch.arange(24, dtype=torch.float32), torch.zeros(24)]

    # Default: PLM scores only, unstandardized
    plain = model._comb_zs_matrix(plm_scores, x, sequences)
    assert plain.shape == (24, 2)
    np.testing.assert_allclose(plain[:, 0].numpy(), np.arange(24), atol=1e-5)
    assert model._zs_stats is None

    # Flag on: DCA and PMPNN columns appended, all columns standardized
    model.gp_comb_all_zero_shot = True
    full = model._comb_zs_matrix(plm_scores, x, sequences)
    assert full.shape == (24, 4)
    np.testing.assert_allclose(full[:, 0].mean().item(), 0, atol=1e-5)
    np.testing.assert_allclose(full[:, 0].std().item(), 1, atol=1e-2)
    np.testing.assert_allclose(full[:, 1].numpy(), 0, atol=1e-5)  # constant column stays finite
    assert np.isfinite(full.numpy()).all()

    # The statistics of the first call are reused (prediction batches keep the training scale)
    stats = model._zs_stats
    half = model._comb_zs_matrix([s[:12] for s in plm_scores], x[:12], sequences[:12])
    assert model._zs_stats is stats
    np.testing.assert_allclose(half.numpy(), full.numpy()[:12], rtol=1e-5)


def test_ridge_block_cv():
    """
    Position-blocked CV for the DCA-Ridge alpha: folds are contiguous regions of mutated
    positions, so one-hot-like features of held-out positions carry no training signal and
    a larger alpha is selected than by plain CV on shuffled data.
    With pure site one-hot features the blocked CV score is the same for every alpha (held-out
    positions are predicted by the intercept); such ties are resolved towards the largest alpha,
    independent of the platform's floating-point rounding.
    """
    rng = np.random.default_rng(7)
    length, aas = 40, 'CDEFG'
    wt = 'A' * length
    sequences, x, y = [], [], []
    site_effect = rng.normal(0, 2, size=(length, len(aas)))
    for pos in range(length):
        for j, aa in enumerate(aas):
            sequences.append(wt[:pos] + aa + wt[pos + 1:])
            onehot = np.zeros(length * len(aas))
            onehot[pos * len(aas) + j] = 1.0
            x.append(onehot)
            y.append(site_effect[pos, j])
    order = rng.permutation(len(y))  # shuffled like the random ttrain subsplit
    sequences = [sequences[i] for i in order]
    x, y = np.array(x)[order], np.array(y)[order]
    # Each site is seen twice with noise, so plain CV can learn the site effects
    sequences, x = sequences * 2, np.vstack([x, x])
    y = np.concatenate([y, y]) + rng.normal(0, .1, 2 * len(y))

    model = DCALLMHybridModel.__new__(DCALLMHybridModel)
    model.wt_sequence, model.seed = wt, 7
    model.ridge_alphas = np.logspace(-4, 4, 9)
    model.ridge_block_cv, model.ridge_block_cv_min_n = True, 150

    splits = model._block_cv_splits(sequences)
    assert len(splits) == 5
    positions = np.array([next(i for i, (a, b) in enumerate(zip(s, wt)) if a != b) for s in sequences])
    previous_max = -1
    for train, test in splits:
        assert len(train) + len(test) == len(sequences)
        assert set(positions[train]).isdisjoint(positions[test])  # whole positions held out
        assert positions[test].min() > previous_max               # contiguous, ordered blocks
        previous_max = positions[test].max()
        assert 60 <= len(test) <= 100                            # balanced (400 / 5 = 80)

    alpha_block = model.ridge_predictor(x, y, sequences).alpha
    model.ridge_block_cv = False
    alpha_plain = model.ridge_predictor(x, y, sequences).alpha
    assert alpha_block == model.ridge_alphas.max()  # tie between all alphas -> most regularized
    assert alpha_plain <= 1e-2                      # plain CV learns the site effects
    assert alpha_block >= 100 * alpha_plain

    # Fallbacks to plain CV: no/unaligned sequences, too few variants or positions
    model.ridge_block_cv = True
    assert model._block_cv_splits(None) is None
    assert model._block_cv_splits(sequences[:149]) is None
    assert model._block_cv_splits([s[:-1] for s in sequences]) is None
    assert model._block_cv_splits([s for s, p in zip(sequences, positions) if p < 5] * 4) is None
    assert model.ridge_predictor(x, y, None).alpha == alpha_plain
    model.wt_sequence = None
    assert model._block_cv_splits(sequences) is None


def test_ridge_block_cv_in_hybrid_model():
    """The flag is used for the initial fit and the refit, and survives pickling."""
    rng = np.random.default_rng(11)
    sequences = [('A' * i + 'C' + 'A' * (11 - i)) for i in range(12)] * 4
    x = rng.normal(size=(48, 3))
    y = rng.normal(size=48)
    kwargs = dict(
        x_dca_wt=np.zeros(3), sequences=sequences, wt_sequence='A' * 12,
        pmpnn_conditional_probs=rng.normal(size=(12, 21)), device='cpu',
        ridge_alphas=np.logspace(-2, 2, 5), batch_size=1, seed=11,
    )
    model = DCALLMHybridModel(
        x, y, ridge_block_cv=True, ridge_block_cv_min_n=20, refit_full_train=True, **kwargs
    )
    assert model._block_cv_splits(model.sequences_ttrain) is not None
    restored = pickle.loads(pickle.dumps(model))
    pred, _ = restored.hybrid_prediction(x, sequences=sequences)
    np.testing.assert_allclose(pred, model.hybrid_prediction(x, sequences=sequences)[0])
    assert np.isfinite(pred).all()
