"""
Shared fixtures for driving the real PyPEF Qt GUI (pypef.gui.qt_window.MainWidget)
in tests: a QApplication, a MainWidget, a fake QFileDialog that answers every file
prompt by the prompt's title, and helpers for running jobs to completion.

Run headless with QT_QPA_PLATFORM=offscreen (set automatically below if unset).
"""
import gc
import os
import time
from pathlib import Path

import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

DATASETS = Path(__file__).resolve().parents[2] / 'datasets'

# File prompts are identified by a (lower-case) substring of their dialog title.
# The first match wins. An unknown title makes the test fail on purpose, so a new
# GUI prompt has to be registered here (and thereby gets covered by the tests).
DIALOG_ROLES = (
    ('wt fasta', 'wt'),
    ('variant csv', 'csv'),
    ('pdb protein structure', 'pdb'),
    ('multiple sequence alignment', 'msa'),
    ('training set', 'ls'),
    ('test set', 'ts'),
    ('prediction set', 'ps'),
    ('hybrid model', 'hybrid_model'),
    ('ml model', 'ml_model'),
    ('onehot model', 'onehot_model'),
    ('parameter pickle', 'params'),
    ('gremlin pickle', 'params'),
)


def dialog_role(title):
    for key, role in DIALOG_ROLES:
        if key in title.lower():
            return role
    raise AssertionError(
        f'Unknown file dialog title {title!r}: register it in DIALOG_ROLES '
        f'(tests/gui/conftest.py).'
    )


class FakeFileDialogs:
    """
    Stand-in for QFileDialog.getOpenFileName: returns files[role] for each
    prompt and records the prompted roles. With cancel_at=i, the i-th prompt
    (0-based) is cancelled, i.e. returns an empty path like the real dialog.
    """
    def __init__(self, files, cancel_at=None):
        self.files = files
        self.cancel_at = cancel_at
        self.prompted = []

    def __call__(self, parent=None, title='', *args, **kwargs):
        role = dialog_role(title)
        index = len(self.prompted)
        self.prompted.append(role)
        if index == self.cancel_at:
            return '', ''
        assert role in self.files, f'No test file configured for prompt {title!r} ({role})'
        return str(self.files[role]), ''


@pytest.fixture(autouse=True)
def no_real_file_dialogs(monkeypatch):
    """A real (modal) dialog would block a headless test run forever."""
    try:
        from pypef.gui import qt_window
    except ImportError:  # CLI-only environment without PySide6
        return

    def refuse(*args, **kwargs):
        raise AssertionError(f'Unexpected real file dialog: {args[1:2]}')
    monkeypatch.setattr(qt_window.QFileDialog, 'getOpenFileName', refuse)
    monkeypatch.setattr(qt_window.QFileDialog, 'getExistingDirectory', refuse)


@pytest.fixture
def file_dialogs(monkeypatch, no_real_file_dialogs):
    """Factory: file_dialogs(files, cancel_at=None) installs a FakeFileDialogs."""
    from pypef.gui import qt_window

    def install(files, cancel_at=None):
        fake = FakeFileDialogs(files, cancel_at)
        monkeypatch.setattr(qt_window.QFileDialog, 'getOpenFileName', fake)
        return fake
    return install


@pytest.fixture(scope='session')
def qapp():
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


@pytest.fixture(scope='module')
def gui_window_module(qapp):
    """One real MainWidget per test module (creating it per test is slow)."""
    from pypef.gui.qt_window import MainWidget
    window = MainWidget()
    window.show()
    qapp.processEvents()
    yield window
    dispose_window(qapp, window)
    del window
    gc.collect()  # see dispose_window


def dispose_window(qapp, window):
    """
    Close a MainWidget and make sure it is destroyed here, in the main thread.
    A window is kept alive by reference cycles (signal connections to its bound
    methods), so it would otherwise be freed by Python's cyclic garbage collector,
    which may run inside the job thread of a later test; shiboken then defers the
    Qt deletion, which crashes (segfault) while that later test runs.
    """
    from pypef.main import logger
    window.close()
    logger.removeHandler(window.logTextBox)
    qapp.processEvents()


def reset_window(window):
    """Bring all GUI options back to their start-up defaults."""
    window.box_llm.setCurrentIndex(1)  # ESM
    window.box_regression_model.setCurrentIndex(0)  # PLS
    window.box_mklsts_cv.setCurrentIndex(0)  # None
    window.slider.setValue(80)
    for check in (window.check_lora, window.check_gauss_opt, window.check_gauss_comb):
        check.setChecked(False)
    for field in (window.input_variant_key, window.input_sequence_key, window.input_fitness_key):
        field.clear()
    window.toggle_buttons(True)
    window.logTextBox.widget.clear()
    window.textedit_out.clear()


@pytest.fixture
def gui_window(gui_window_module):
    reset_window(gui_window_module)
    yield gui_window_module
    reset_window(gui_window_module)


def buttons_enabled(window):
    return [b.isEnabled() for b in window.all_buttons]


def button_name(window, button):
    return next(n for n in vars(window) if n.startswith('button_') and getattr(window, n) is button)


def wait_until(qapp, condition, timeout, interval=0.02):
    """Process Qt events until condition() is true; fail after timeout seconds."""
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise TimeoutError(f'GUI condition not met within {timeout} s')
        qapp.processEvents()
        time.sleep(interval)
    qapp.processEvents()


def run_job(qapp, window, button, timeout=600):
    """
    Click a GUI button that starts a real background job and wait until it has
    finished (all buttons enabled again). Returns (log_text, failed).
    """
    button.click()
    qapp.processEvents()
    wait_until(qapp, lambda: all(buttons_enabled(window)), timeout)
    # Let queued log/output signals from the finished worker arrive.
    for _ in range(10):
        qapp.processEvents()
        time.sleep(0.01)
    return window.logTextBox.widget.toPlainText(), window._run_failed


@pytest.fixture
def gui(qapp):
    """
    Helper functions as a fixture (test modules cannot import from conftest
    under --import-mode=importlib, which the CI pip-installation run uses).
    """
    from types import SimpleNamespace
    return SimpleNamespace(
        datasets=DATASETS,
        reset=reset_window,
        buttons_enabled=buttons_enabled,
        button_name=button_name,
        wait_until=lambda condition, timeout=30: wait_until(qapp, condition, timeout),
        run_job=lambda window, button, timeout=600: run_job(qapp, window, button, timeout),
    )


def pytest_collection_modifyitems(config, items):
    """gui_slow tests (hours on CPU) only run when selected explicitly: -m gui_slow"""
    if 'gui_slow' in (config.getoption('markexpr') or ''):
        return
    skip = pytest.mark.skip(reason='long-running GUI run; select with -m gui_slow')
    for item in items:
        if 'gui_slow' in item.keywords:
            item.add_marker(skip)
