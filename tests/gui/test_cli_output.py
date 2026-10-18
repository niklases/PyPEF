import io
import sys
import threading
from types import SimpleNamespace
from unittest.mock import patch
from tqdm import tqdm
import pytest


@pytest.mark.pip_specific
def test_stream_captures_only_own_thread():
    from pypef.gui.qt_window import WorkerOutputStream
    terminal, output = io.StringIO(), []
    stream = WorkerOutputStream(terminal, output.append)
    stream.write('job')
    other = threading.Thread(target=lambda: stream.write('other'))
    other.start()
    other.join()
    assert output == ['job']
    assert terminal.getvalue() == 'jobother'


@pytest.mark.pip_specific
def test_worker_captures_tqdm_and_restores_streams():
    from pypef.gui.qt_window import Worker
    worker = Worker(0, [])
    output = []
    worker.sig_output.connect(output.append)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    def run(**kwargs):
        print('Starting')
        for _ in tqdm(range(2), desc='GP training', mininterval=0):
            pass
        raise ValueError('Failed')
    with patch('pypef.gui.qt_window.run_main', side_effect=run):
        worker.work()
    assert sys.stdout is original_stdout
    assert sys.stderr is original_stderr
    assert 'Starting' in ''.join(output)
    assert 'GP training' in ''.join(output)
    assert '100%' in ''.join(output)


@pytest.mark.pip_specific
def test_progress_updates_replace_current_line():
    from PySide6.QtWidgets import QApplication, QTextEdit
    from pypef.gui.qt_window import MainWidget
    app = QApplication.instance() or QApplication([])
    widget = QTextEdit()
    left_widget = QTextEdit()
    left_widget.setPlainText('Job summary')
    window = SimpleNamespace(logTextBox=SimpleNamespace(widget=widget),
                             textedit_out=left_widget)
    for chunk in ('Starting\n', '\rGP training: 0%', '\rGP training: 50%',
                  '\r\x1b[32mGP training: 100%\x1b[0m', '\nFinished\n'):
        MainWidget.on_cli_output(window, chunk)
    assert widget.toPlainText() == 'Starting\nGP training: 100%\nFinished\n'
    assert left_widget.toPlainText() == 'Job summary'
    left_widget.close()
    widget.close()
    app.processEvents()
