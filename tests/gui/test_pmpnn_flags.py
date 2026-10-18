"""Check hybrid command construction without launching training or GUI threads."""
from types import SimpleNamespace
from unittest.mock import Mock, patch
import pytest


RUNS = [
    'pypef_dca_hybrid_train',
    'pypef_dca_hybrid_train_test',
    'pypef_dca_llm_hybrid_train',
    'pypef_dca_llm_hybrid_train_test',
]


def make_window(plm):
    from pypef.gui.qt_window import MainWidget
    window = SimpleNamespace(
        win2=None, llm=plm,
        check_lora=Mock(isChecked=Mock(return_value=False)),
        check_gauss_opt=Mock(isChecked=Mock(return_value=False)),
        check_gauss_comb=Mock(isChecked=Mock(return_value=False)),
        version_text=Mock(), logTextBox=SimpleNamespace(widget=Mock()),
        start_process=Mock(), end_process=Mock(), start_main_thread=Mock(),
    )
    for suffix in ('train_dca', 'train_test_dca', 'train_dca_llm', 'train_test_dca_llm'):
        setattr(window, 'button_hybrid_' + suffix, Mock())
    for name in ('_llm_hybrid_flags', '_llm_needs_structure'):
        setattr(window, name, getattr(MainWidget, name).__get__(window))
    return window


@pytest.mark.pip_specific
@pytest.mark.parametrize('run', RUNS[2:])
@pytest.mark.parametrize('missing', ['WT FASTA', 'PDB'])
def test_cancelled_pmpnn_structure_does_not_start(run, missing):
    from pypef.gui.qt_window import MainWidget
    window = make_window('pmpnn')
    def select_file(parent, title, **kwargs):
        return ('', '') if missing in title else ('input_file', '')
    with patch('pypef.gui.qt_window.QFileDialog.getOpenFileName', side_effect=select_file):
        getattr(MainWidget, run)(window)
    window.start_main_thread.assert_not_called()
    window.end_process.assert_called_once()


@pytest.mark.pip_specific
@pytest.mark.parametrize('plm', ['PMPNN', 'ESM+PMPNN', 'ProSST+PMPNN', 'ESM+ProSST+PMPNN'])
@pytest.mark.parametrize('run', RUNS[2:])
def test_pmpnn_dropdown_training(run, plm):
    from pypef.gui.qt_window import MainWidget, PLM_OPTIONS
    window = make_window(None)
    MainWidget.selection_llm_model(window, PLM_OPTIONS.index(plm))
    assert window.llm == plm.lower()
    with patch('pypef.gui.qt_window.QFileDialog.getOpenFileName',
               side_effect=lambda *args, **kwargs: ('input_file', '')) as dialog:
        getattr(MainWidget, run)(window)
    window.start_main_thread.assert_called_once()
    assert f'--plm {plm.lower()}' in window.cmd
    assert '--wt input_file' in window.cmd
    assert '--pdb input_file' in window.cmd
    assert dialog.call_count == (5 if run.endswith('_test') else 4)


@pytest.mark.pip_specific
@pytest.mark.parametrize('custom', [False, True])
def test_mklsts_inline_column_fields(custom):
    from pypef.gui.qt_window import MainWidget
    window = make_window('esm')
    window.button_mklsts = Mock()
    window.ls_proportion = 0.8
    window.mklsts_cv_method = '--random'
    fields = [('input_variant_key', '--variant_key', 'mutant'),
              ('input_sequence_key', '--sequence_key', 'mutated_sequence'),
              ('input_fitness_key', '--fitness_key', 'DMS_score')]
    for attr, flag, key in fields:
        setattr(window, attr, Mock(text=Mock(return_value=key if custom else '')))
    with patch('pypef.gui.qt_window.QFileDialog.getOpenFileName',
               return_value=('input file', '')):
        MainWidget.pypef_mklsts(window)
    window.start_main_thread.assert_called_once()
    assert window.cmd[window.cmd.index('--input') + 1] == 'input file'
    for attr, flag, key in fields:
        if custom:
            assert window.cmd[window.cmd.index(flag) + 1] == key
        else:
            assert flag not in window.cmd


@pytest.mark.pip_specific
@pytest.mark.parametrize('action', ['pypef_llm_predict', 'pypef_llm_test'])
@pytest.mark.parametrize('cancel_structure', [False, True])
def test_pmpnn_zero_shot_gui(action, cancel_structure):
    from pypef.gui.qt_window import MainWidget
    window = make_window('pmpnn')
    window.button_llm_predict_zs = Mock()
    window.button_llm_test_zs = Mock()
    def select(parent, title, **kwargs):
        return ('', '') if cancel_structure and 'PDB' in title else ('input_file', '')
    with patch('pypef.gui.qt_window.QFileDialog.getOpenFileName', side_effect=select):
        getattr(MainWidget, action)(window)
    if cancel_structure:
        window.start_main_thread.assert_not_called()
        window.end_process.assert_called_once()
    else:
        window.start_main_thread.assert_called_once()
        assert '--plm pmpnn' in window.cmd
        assert '--pdb input_file' in window.cmd
        assert '--wt input_file' in window.cmd
