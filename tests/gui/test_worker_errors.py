from types import SimpleNamespace
from unittest.mock import Mock, patch
import pytest


@pytest.mark.pip_specific
@pytest.mark.parametrize('error', [
    ValueError("CSV column 'fitness' not found. Available columns: mutant, DMS_score"),
    FileNotFoundError('Missing input file'),
    SystemExit('Invalid command'),
])
def test_worker_reports_failure_and_finishes(error):
    from pypef.gui.qt_window import Worker
    worker = Worker(3, ['mklsts'])
    errors, completed = [], []
    worker.sig_error.connect(errors.append)
    worker.sig_done.connect(completed.append)
    with patch('pypef.gui.qt_window.run_main', side_effect=error):
        worker.work()
    assert errors == [f'{type(error).__name__}: {error}']
    assert completed == [3]


@pytest.mark.pip_specific
@pytest.mark.parametrize('exit_code', [None, 0])
def test_successful_cli_exit_finishes_without_error(exit_code):
    from pypef.gui.qt_window import Worker
    worker = Worker(3, ['--help'])
    errors, completed = [], []
    worker.sig_error.connect(errors.append)
    worker.sig_done.connect(completed.append)
    with patch('pypef.gui.qt_window.run_main', side_effect=SystemExit(exit_code)):
        worker.work()
    assert not errors
    assert completed == [3]


@pytest.mark.pip_specific
def test_worker_success_finishes():
    from pypef.gui.qt_window import Worker
    worker = Worker(3, ['mklsts'])
    errors, completed = [], []
    worker.sig_error.connect(errors.append)
    worker.sig_done.connect(completed.append)
    with patch('pypef.gui.qt_window.run_main'):
        worker.work()
    assert not errors
    assert completed == [3]


@pytest.mark.pip_specific
def test_gui_displays_failure_and_cleans_up():
    from pypef.gui.qt_window import MainWidget
    window = SimpleNamespace(
        _run_failed=False,
        logTextBox=SimpleNamespace(widget=Mock()), textedit_out=Mock(),
        version_text=Mock(), end_process=Mock(),
        _MainWidget__workers_done=0, _MainWidget__threads=[(Mock(), Mock())],
    )
    MainWidget.on_worker_error(window, 'ValueError: missing fitness column')
    MainWidget.on_worker_done(window, 3)
    window.logTextBox.widget.appendPlainText.assert_called_once_with(
        'Job failed: ValueError: missing fitness column')
    window.end_process.assert_called_once()
    window.version_text.setText.assert_called_once_with('Failed — check the log')
    thread, _ = window._MainWidget__threads[0]
    thread.quit.assert_called_once()
    thread.wait.assert_called_once()
