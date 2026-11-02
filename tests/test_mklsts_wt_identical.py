"""
MKLSTS notes variants that are identical to the WT, i.e. consist only of synonymous
"substitutions" such as L215L/R219R (e.g. the WT reference entry of a combinatorial
library, as in datasets/ANEH/TS_B.fasl). They are kept in the data.
"""
import logging
from pathlib import Path

import pytest

from pypef.main import run_main
from pypef.utils.learning_test_sets import wt_identical_variants

ANEH = Path(__file__).resolve().parent.parent / 'datasets' / 'ANEH'


@pytest.mark.parametrize('variants, expected', [
    (['L215L/R219R/F244F', 'L215F', 'L215L/R219D', 'WT'], ['L215L/R219R/F244F']),
    (['A1A', 'a2a', 'A1C'], ['A1A', 'a2a']),
    (['L215F/R219D', 'L215F'], []),
])
def test_wt_identical_variants(variants, expected):
    assert wt_identical_variants(variants) == expected


def test_mklsts_notes_and_keeps_wt_identical_variant(tmp_path, monkeypatch, caplog):
    monkeypatch.chdir(tmp_path)
    csv = tmp_path / 'variants.csv'
    rows = ['L215F;-1.5', 'R219S;-0.6', 'L249Y;0.2', 'C350V;0.8',
            'L215L/R219R/L249L/C350C;0.0']  # identical to the ANEH WT
    csv.write_text('Variants;ddG\n' + '\n'.join(rows) + '\n')
    with caplog.at_level(logging.INFO, logger='pypef'):
        run_main(argv=['mklsts', '--wt', str(ANEH / 'Sequence_WT_ANEH.fasta'),
                       '--input', str(csv), '--ls_proportion', '0.6'])
    notes = [r.getMessage() for r in caplog.records if 'identical to the WT' in r.getMessage()]
    assert len(notes) == 1 and 'L215L/R219R/L249L/C350C' in notes[0]
    written = (tmp_path / 'LS.fasl').read_text() + (tmp_path / 'TS.fasl').read_text()
    assert 'L215L/R219R/L249L/C350C' in written  # kept, not dropped


def test_mklsts_note_with_selected_csv_columns(tmp_path, monkeypatch, caplog):
    """Variant, fitness and sequence columns chosen via --variant_key / --fitness_key /
    --sequence_key (GUI: Mutant/Fitness/Sequence column fields), in a different order."""
    monkeypatch.chdir(tmp_path)
    wt = ''.join(l.strip() for l in (ANEH / 'Sequence_WT_ANEH.fasta').read_text().splitlines()
                 if not l.startswith('>'))

    def mutate(variant):
        seq = list(wt)
        for sub in variant.split('/'):
            assert seq[int(sub[1:-1]) - 1] == sub[0]
            seq[int(sub[1:-1]) - 1] = sub[-1]
        return ''.join(seq)

    variants = ['L215F', 'R219S', 'L249Y', 'C350V', 'L215L/R219R/L249L/C350C']
    lines = ['score;mutated_sequence;notes;mutant']
    lines += [f'{i * 0.5};{mutate(v)};x;{v}' for i, v in enumerate(variants)]
    csv = tmp_path / 'variants.csv'
    csv.write_text('\n'.join(lines) + '\n')
    with caplog.at_level(logging.INFO, logger='pypef'):
        run_main(argv=['mklsts', '--wt', str(ANEH / 'Sequence_WT_ANEH.fasta'), '--input', str(csv),
                       '--ls_proportion', '0.6', '--variant_key', 'mutant',
                       '--fitness_key', 'score', '--sequence_key', 'mutated_sequence'])
    notes = [r.getMessage() for r in caplog.records if 'identical to the WT' in r.getMessage()]
    assert len(notes) == 1 and 'L215L/R219R/L249L/C350C' in notes[0]
    assert 'x' not in notes[0].split(':', 1)[1].split()  # not taken from another column
