"""
Saved model files:
  * A trained GREMLIN model holds no torch tensors anymore (they were pickled before:
    1.5 GB for ANEH, and a GPU-trained pickle could not be loaded without CUDA), and
    its predictions are unchanged after a pickle round trip.
  * Hybrid models with PMPNN carry "PMPNN" in their file name.
  * Hybrid PLM models store only the LoRA adapter weights and the pretrained model's
    name and revision (not the full PLMs and optimizers: ~9 GB for ESM+ProSST), and
    are restored exactly, also from files in the old full-state-dict format.
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


def tiny_esm():
    """A small, deterministic stand-in for a pretrained ESM model and its LoRA copy."""
    import copy
    from peft import LoraConfig, get_peft_model
    from transformers import EsmConfig, EsmForMaskedLM
    torch.manual_seed(0)  # identical "pretrained" weights on every call
    config = EsmConfig(vocab_size=33, hidden_size=16, num_hidden_layers=2,
                       num_attention_heads=2, intermediate_size=32,
                       max_position_embeddings=40, pad_token_id=1, mask_token_id=32)
    base = EsmForMaskedLM(config).eval()
    base.config._name_or_path = 'facebook/esm2_tiny'
    base.config._commit_hash = 'abc123'
    for param in base.parameters():
        param.requires_grad = False
    lora = get_peft_model(copy.deepcopy(base), LoraConfig(r=2, target_modules=['query', 'value']))
    return base, lora, 'tokenizer', torch.optim.Adam(lora.parameters())


def hybrid_with_tiny_esm():
    from pypef.hybrid.hybrid_model import DCALLMHybridModel
    base, lora, tokenizer, optimizer = tiny_esm()
    torch.manual_seed(1)
    with torch.no_grad():  # "LoRA training": non-zero adapter weights
        for name, param in lora.named_parameters():
            if 'lora_' in name:
                param.copy_(torch.randn_like(param))
    model = DCALLMHybridModel.__new__(DCALLMHybridModel)
    model.llm_keys = ['ESM']
    model.llm_data = {'ESM': {
        'llm_base_model': base, 'llm_model': lora, 'llm_optimizer': optimizer,
        'llm_tokenizer': tokenizer, 'x_llm': torch.zeros(2, 5, dtype=torch.long),
    }}
    model.progress_cb = lambda *a: None
    model.abort_cb = lambda: False
    model.pmpnn_log_probs = None
    return model


def plm_outputs(model):
    tokens = torch.tensor([[0, 5, 6, 7, 2]])
    data = model.llm_data['ESM']
    with torch.no_grad():
        return (data['llm_base_model'](tokens).logits, data['llm_model'](tokens).logits)


@pytest.fixture
def reload_tiny_esm(monkeypatch):
    import pypef.hybrid.hybrid_model as hm
    calls = []
    def get_esm_models(**kwargs):
        calls.append(kwargs)
        return tiny_esm()
    monkeypatch.setattr(hm, 'get_esm_models', get_esm_models)
    return calls


def test_hybrid_pickle_stores_only_lora_adapters(tmp_path, monkeypatch, reload_tiny_esm):
    from pypef.hybrid.hybrid_model import get_model_and_type
    monkeypatch.chdir(tmp_path)
    model = hybrid_with_tiny_esm()
    expected = plm_outputs(model)
    save_model_to_dict_pickle(model, 'HybridGREMLIN')
    # The model itself is left intact (it was converted to state dicts before)
    data = model.llm_data['ESM']
    assert isinstance(data['llm_base_model'], torch.nn.Module)
    assert isinstance(data['llm_optimizer'], torch.optim.Adam)
    assert model.progress_cb is not None
    path = tmp_path / 'Pickles' / 'HYBRIDGREMLINESM'
    stored = pickle.loads(path.read_bytes())['model'].llm_data['ESM']
    assert stored['llm_base_model'] is None and stored['llm_optimizer'] is None
    assert stored['llm_storage'] == 'lora_adapter'
    assert stored['llm_source'] == {'model': 'facebook/esm2_tiny', 'revision': 'abc123'}
    assert stored['llm_model'] and all('lora_' in k for k in stored['llm_model'])
    base_bytes = sum(p.numel() * p.element_size() for p in data['llm_base_model'].parameters())
    assert path.stat().st_size < base_bytes  # i.e. not even one full model copy

    loaded, model_type = get_model_and_type(str(path))
    assert model_type == 'Hybrid_ESM'
    assert reload_tiny_esm == [{'model': 'facebook/esm2_tiny', 'revision': 'abc123'}]
    for before, after in zip(expected, plm_outputs(loaded)):
        torch.testing.assert_close(after, before, rtol=0, atol=0)


def test_hybrid_pickle_old_full_state_dict_format(tmp_path, monkeypatch, reload_tiny_esm):
    """Files saved before v0.5.1 store the full base and LoRA model state dicts."""
    from pypef.hybrid.hybrid_model import get_model_and_type
    monkeypatch.chdir(tmp_path)
    model = hybrid_with_tiny_esm()
    expected = plm_outputs(model)
    data = model.llm_data['ESM']
    data['llm_base_model'] = data['llm_base_model'].state_dict()
    data['llm_model'] = data['llm_model'].state_dict()
    model.progress_cb = model.abort_cb = None
    path = tmp_path / 'OLD'
    path.write_bytes(pickle.dumps({'model': model, 'model_type': 'HybridGREMLINESM'}))
    loaded, _ = get_model_and_type(str(path))
    assert reload_tiny_esm == [{'revision': None}]  # no source recorded: default model
    for before, after in zip(expected, plm_outputs(loaded)):
        torch.testing.assert_close(after, before, rtol=0, atol=0)
