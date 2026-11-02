"""
Kermut structure kernel with sequences that carry no substitution (the WT itself or
synonymous "mutants" such as L215L/R219R, e.g. present in datasets/ANEH/TS_B.fasl).
Such sequences have no mutated site; the kernel must still return one row/column
per sequence (formerly: 29 x 29 instead of 30 x 30 if the WT was the last sequence).
"""
import pytest
import torch

from pypef.gaussian_process.kermut.kernels.structure_kernel import StructureKernel

SEQ_LEN = 6
WT = [0, 1, 2, 3, 4, 5]  # amino acid indices per position


def one_hot(seqs):
    x = torch.zeros(len(seqs), SEQ_LEN, 20)
    for i, seq in enumerate(seqs):
        x[i, torch.arange(SEQ_LEN), torch.tensor(seq)] = 1
    return x.view(len(seqs), SEQ_LEN * 20)


def mutate(*subs):
    seq = list(WT)
    for pos, aa in subs:
        seq[pos] = aa
    return seq


@pytest.fixture(scope='module')
def kernel():
    torch.manual_seed(0)
    probs = torch.softmax(torch.randn(SEQ_LEN, 20), dim=-1)
    coords = torch.randn(SEQ_LEN, 3) * 5
    return StructureKernel(one_hot([WT])[0].long(), conditional_probs=probs, coords=coords)


MUTANTS = [mutate((0, 7)), mutate((1, 9), (3, 12)), mutate((2, 15)), mutate((4, 0), (5, 1))]


@pytest.mark.parametrize('wt_index', [0, 2, len(MUTANTS)])  # first, middle, last
def test_wt_like_sequences_keep_kernel_shape(kernel, wt_index):
    seqs = list(MUTANTS)
    seqs.insert(wt_index, list(WT))
    with torch.no_grad():
        k = kernel(one_hot(seqs), one_hot(seqs)).to_dense()
        k_mutants = kernel(one_hot(MUTANTS), one_hot(MUTANTS)).to_dense()
    assert k.shape == (len(seqs), len(seqs))
    # No mutated site -> zero structure-kernel similarity to everything ...
    assert torch.all(k[wt_index] == 0) and torch.all(k[:, wt_index] == 0)
    # ... and the mutant block is unaffected by the WT entry.
    keep = [i for i in range(len(seqs)) if i != wt_index]
    torch.testing.assert_close(k[keep][:, keep], k_mutants)


def test_only_wt_like_sequences(kernel):
    with torch.no_grad():
        k = kernel(one_hot([WT, WT]), one_hot([WT] + MUTANTS)).to_dense()
    assert k.shape == (2, len(MUTANTS) + 1)
    assert torch.all(k == 0)
