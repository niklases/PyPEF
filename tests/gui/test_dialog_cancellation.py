"""Cancelling any file prompt must stop the current action immediately."""
import inspect
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from pypef.gui.qt_window import MainWidget


ACTIONS = [name for name, method in inspect.getmembers(MainWidget, inspect.isfunction)
           if name.startswith('pypef_') and 'getOpenFileName' in method.__code__.co_names]


def make_window():
    window = SimpleNamespace(
        win2=None, llm='pmpnn', n_cores=1, regression_model='ridge',
        ls_proportion=0.8, mklsts_cv_method='',
        start_process=Mock(), end_process=Mock(), start_main_thread=Mock(),
        version_text=Mock(), logTextBox=SimpleNamespace(widget=Mock()),
        input_variant_key=Mock(text=Mock(return_value='')),
        input_sequence_key=Mock(text=Mock(return_value='')),
        input_fitness_key=Mock(text=Mock(return_value='')),
        _llm_hybrid_flags=Mock(return_value=''),
        _llm_needs_structure=Mock(return_value=True),
    )
    for name in ACTIONS:
        for attr in getattr(MainWidget, name).__code__.co_names:
            if attr.startswith('button_'):
                setattr(window, attr, Mock())
    return window


@pytest.mark.parametrize('action', ACTIONS)
def test_each_cancelled_file_prompt_stops_action(action):
    # Count prompts on this action's normal path, then cancel each one in turn.
    with patch('pypef.gui.qt_window.QFileDialog.getOpenFileName',
               return_value=('input_file', '')) as dialog:
        getattr(MainWidget, action)(make_window())
    prompt_count = dialog.call_count
    for cancelled_index in range(prompt_count):
        window = make_window()
        selections = [('input_file', '')] * cancelled_index + [('', '')]
        with patch('pypef.gui.qt_window.QFileDialog.getOpenFileName',
                   side_effect=selections) as dialog:
            getattr(MainWidget, action)(window)
        assert dialog.call_count == cancelled_index + 1
        window.start_main_thread.assert_not_called()
        window.end_process.assert_called_once()


def test_cancelled_working_directory_keeps_existing_directory():
    window = SimpleNamespace(win2=None, working_directory='existing')
    with patch('pypef.gui.qt_window.QFileDialog.getExistingDirectory', return_value=''), \
         patch('pypef.gui.qt_window.chdir') as change_directory:
        MainWidget.set_work_dir(window)
    assert window.working_directory == 'existing'
    change_directory.assert_not_called()
