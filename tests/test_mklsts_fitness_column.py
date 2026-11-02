import pandas as pd
import pytest
from docopt import docopt

from pypef.main import __doc__ as cli_help
from pypef.utils.learning_test_sets import drop_rows

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'


@pytest.mark.parametrize('separator', [',', ';', '\t'])
def test_select_fitness_column(tmp_path, separator):
    csv = tmp_path / 'variants.csv'
    pd.DataFrame({'mutant': ['A1C', 'A1D'], 'mutated_sequence': ['CDE', 'DDE'],
                  'DMS_score': [0.5, -2.0]}).to_csv(csv, sep=separator, index=False)
    df = drop_rows(csv, AMINO_ACIDS, -1, fitness_key='DMS_score')
    assert list(df.columns) == ['mutant', 'DMS_score']
    assert df['mutant'].tolist() == ['A1C']
    assert df['DMS_score'].tolist() == [0.5]


def test_default_second_column(tmp_path):
    csv = tmp_path / 'variants.csv'
    csv.write_text('mutant,fitness,other\nA1C,0.5,ignored\n')
    df = drop_rows(csv, AMINO_ACIDS, -1)
    assert df.iloc[0].tolist() == ['A1C', 0.5]


@pytest.mark.parametrize('key', ['missing', 'mutant'])
def test_invalid_fitness_column(tmp_path, key):
    csv = tmp_path / 'variants.csv'
    csv.write_text('mutant,fitness\nA1C,0.5\n')
    with pytest.raises(ValueError):
        drop_rows(csv, AMINO_ACIDS, -1, fitness_key=key)


def test_mklsts_cli_accepts_fitness_key():
    args = docopt(cli_help, argv=['mklsts', '--wt', 'wt.fa', '--input', 'data.csv',
                                 '--fitness_key', 'DMS_score'])
    assert args['--fitness_key'] == 'DMS_score'


def test_select_variant_and_sequence_columns(tmp_path):
    csv = tmp_path / 'variants.csv'
    csv.write_text('other,score,identifier,sequence\nignored,0.5,A1C,CDE\n')
    df = drop_rows(csv, AMINO_ACIDS, -1, variant_key='identifier',
                   fitness_key='score', sequence_key='sequence')
    assert df.iloc[0].tolist() == ['A1C', 0.5, 'CDE']


def test_mklsts_cli_accepts_column_keys():
    args = docopt(cli_help, argv=['mklsts', '--wt', 'wt.fa', '--input', 'data.csv',
                                 '--variant_key', 'mutant', '--sequence_key', 'mutated_sequence',
                                 '--fitness_key', 'DMS_score'])
    assert args['--variant_key'] == 'mutant'
    assert args['--sequence_key'] == 'mutated_sequence'
