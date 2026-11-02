"""
Saved model files:
  * A trained GREMLIN model holds no torch tensors anymore (they were pickled before:
    1.5 GB for ANEH, and a GPU-trained pickle could not be loaded without CUDA), and
    its predictions are unchanged after a pickle round trip.
  * Hybrid models with PMPNN carry "PMPNN" in their file name.
"""
import os
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pypef.dca.gremlin_inference import GREMLIN
from pypef.hybrid.hybrid_model import save_model_to_dict_pickle

AAS = 'ARNDCQEGHILKMFPSTWYV'


def tensors_in(obj, path='gremlin', depth=0):
    """Paths of all torch tensors in an object's attributes (and containers)."""
    if isinstance(obj, torch.Tensor):
        return [path]
    if depth > 3:
        return []
    if isinstance(obj, dict):
        items = obj.items()
    elif isinstance(obj, (list, tuple)):
        items = enumerate(obj)
    elif hasattr(obj, '__dict__') and not isinstance(obj, type):
        items = vars(obj).items()
    else:
        return []
    return [p for k, v in items for p in tensors_in(v, f'{path}.{k}', depth + 1)]


@pytest.fixture(scope='module')
def gremlin():
    rng = np.random.default_rng(0)
    wt = ''.join(rng.choice(list(AAS), 30))
    seqs = [wt] + [
        ''.join(aa if rng.random() > 0.2 else rng.choice(list(AAS)) for aa in wt)
        for _ in range(80)
    ]
    return GREMLIN(sequences=seqs, wt_seq=wt, opt_iter=5, optimize=True)


def test_trained_gremlin_holds_no_tensors(gremlin):
    # Device-independent: on a GPU machine these would otherwise be CUDA tensors
    assert tensors_in(gremlin) == []
    assert isinstance(gremlin.v, np.ndarray) and isinstance(gremlin.w, np.ndarray)
    assert isinstance(gremlin.msa_weights, np.ndarray)
    for training_only in ('oh_msa', 'msa', 'mt_w', 'vt_w', 'mt_v', 'vt_v'):
        assert not hasattr(gremlin, training_only), f'{training_only} would be pickled'


def test_gremlin_pickle_round_trip(gremlin, tmp_path):
    variants = [gremlin.wt_seq[:4] + aa + gremlin.wt_seq[5:] for aa in 'ACDE']
    scores = gremlin.get_scores(variants)
    path = tmp_path / 'GREMLIN'
    path.write_bytes(pickle.dumps(gremlin))
    loaded = pickle.loads(path.read_bytes())
    np.testing.assert_allclose(loaded.get_scores(variants), scores)
    # Pickle is dominated by the couplings w (L x 21 x L x 21) - nothing much else
    assert path.stat().st_size < 2 * gremlin.w.nbytes + 1_000_000


@pytest.mark.parametrize('pmpnn, expected', [(False, 'HYBRIDGREMLIN'), (True, 'HYBRIDGREMLINPMPNN')])
def test_hybrid_model_name_shows_pmpnn(tmp_path, monkeypatch, pmpnn, expected):
    monkeypatch.chdir(tmp_path)
    model = SimpleNamespace(
        llm_keys=None, llm_data={}, progress_cb=None, abort_cb=None,
        pmpnn_log_probs=np.zeros((5, 21)) if pmpnn else None,
    )
    save_model_to_dict_pickle(model, 'HybridGREMLIN')
    assert os.listdir(tmp_path / 'Pickles') == [expected]
