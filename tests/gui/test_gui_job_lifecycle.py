"""
Run GUI jobs through the real QThread/Worker machinery with a fake run_main,
covering what otherwise needs manual clicking: button states while running,
live CLI output, training progress bars/labels, failures, the stop button,
and running several jobs one after another.
"""
import threading
import time

import pytest

pytestmark = pytest.mark.pip_specific

FILES = {'ls': '/data/LS.fasl', 'ts': '/data/TS.fasl', 'params': '/data/GREMLIN'}


@pytest.fixture
def fake_run_main(monkeypatch):
    """Install a fake run_main; returns a setter taking the fake's body."""
    from pypef.gui import qt_window
    calls = []

    def install(body):
        def fake(argv=None, progress_cb=None, abort_cb=None):
            calls.append({'argv': argv, 'thread': threading.current_thread()})
            return body(argv, progress_cb, abort_cb)
        monkeypatch.setattr(qt_window, 'run_main', fake)
        return calls
    return install


def test_successful_job(gui_window, gui, file_dialogs, fake_run_main):
    started = threading.Event()
    release = threading.Event()

    def body(argv, progress_cb, abort_cb):
        print('Training hybrid model')
        started.set()
        release.wait(10)
        for epoch in (1, 2):
            for batch in (1, 2, 3):
                progress_cb(epoch, batch, 2, 3, 0.5)

    calls = fake_run_main(body)
    file_dialogs(FILES)
    gui_window.button_supervised_train_test_onehot.click()
    gui.wait_until(started.is_set)
    # While running: all job buttons are disabled, the stop button is not.
    assert not any(gui.buttons_enabled(gui_window))
    assert gui_window.button_abort.isEnabled()
    assert gui_window.version_text.text() == 'Running...'
    release.set()
    gui.wait_until(lambda: all(gui.buttons_enabled(gui_window)))
    gui.wait_until(lambda: 'Training hybrid model' in gui_window.logTextBox.widget.toPlainText())

    assert calls[0]['thread'] is not threading.main_thread()
    assert calls[0]['argv'][:7] == ['ml', '--encoding', 'onehot', '--ls', '/data/LS.fasl',
                                    '--ts', '/data/TS.fasl']
    assert 'Executing command: ml --encoding onehot' in gui_window.textedit_out.toPlainText()
    assert gui_window.version_text.text() == 'Finished...'
    assert not gui_window._run_failed
    assert gui_window.epoch_progress_bar.value() == 0  # reset after the job
    assert gui_window.epoch_time_label.text().startswith('Epoch 2 / 2')


@pytest.mark.parametrize('error', [ValueError('missing fitness column'), SystemExit('usage')])
def test_failing_job(gui_window, gui, file_dialogs, fake_run_main, error):
    def body(argv, progress_cb, abort_cb):
        raise error

    fake_run_main(body)
    file_dialogs(FILES)
    log, failed = gui.run_job(gui_window, gui_window.button_hybrid_train_test_dca, timeout=10)
    assert failed
    assert f'Job failed: {type(error).__name__}: {error}' in log
    assert gui_window.version_text.text() == 'Failed — check the log'
    assert all(gui.buttons_enabled(gui_window))


def test_stop_button_aborts_running_job(gui_window, gui, file_dialogs, fake_run_main):
    started = threading.Event()
    aborted = []

    def body(argv, progress_cb, abort_cb):
        started.set()
        deadline = time.monotonic() + 10
        while not abort_cb():
            if time.monotonic() > deadline:
                raise TimeoutError('abort not received')
            time.sleep(0.01)
        aborted.append(True)

    fake_run_main(body)
    file_dialogs(FILES)
    gui_window.button_hybrid_train_dca.click()
    gui.wait_until(started.is_set)
    gui_window.button_abort.click()
    gui.wait_until(lambda: all(gui.buttons_enabled(gui_window)))
    assert aborted == [True]
    assert not gui_window._run_failed
    assert 'notified to abort' in gui_window.logTextBox.widget.toPlainText()
    # The finished worker is gone: stopping again must not crash.
    gui_window.abort_workers()
    assert 'No running job to stop.' in gui_window.logTextBox.widget.toPlainText()


def test_consecutive_jobs(gui_window, gui, file_dialogs, fake_run_main):
    calls = fake_run_main(lambda argv, progress_cb, abort_cb: print(f'job {argv[0]}'))
    file_dialogs(FILES)
    jobs_before = gui_window.c
    buttons = (gui_window.button_supervised_train_onehot, gui_window.button_hybrid_train_dca,
               gui_window.button_supervised_train_dca)
    for button in buttons:
        log, failed = gui.run_job(gui_window, button, timeout=10)
        assert not failed
    assert len(calls) == 3
    assert gui_window.c == jobs_before + 3
    assert gui_window.textedit_out.toPlainText().count('Executing command:') == 3


def test_finished_worker_is_not_destroyed_by_qt(gui_window, gui, file_dialogs, fake_run_main):
    """
    The Worker (a Python QObject subclass) and its QThread must not be destroyed by
    Qt (deleteLater) inside the job thread: PySide then needs the GIL while Qt holds
    internal locks, which deadlocked the GUI right after a job (seen after ~90 jobs).
    They stay alive until the next job replaces them, after their thread has ended.
    """
    from shiboken6 import isValid
    fake_run_main(lambda argv, progress_cb, abort_cb: print('job'))
    file_dialogs(FILES)
    gui.run_job(gui_window, gui_window.button_supervised_train_onehot, timeout=10)
    for _ in range(20):  # give pending deleteLater events a chance to run
        gui.wait_until(lambda: True)
    (thread, worker), = gui_window._MainWidget__threads
    assert isValid(worker) and isValid(thread), 'Qt destroyed the finished worker/thread'
    assert thread.isFinished()
    gui.run_job(gui_window, gui_window.button_supervised_train_onehot, timeout=10)
    (new_thread, new_worker), = gui_window._MainWidget__threads
    assert new_worker is not worker and new_thread.isFinished()
