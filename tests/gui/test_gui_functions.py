

import sys
import tempfile
import pytest
from unittest.mock import patch
from PySide6.QtWidgets import QApplication

from pypef import __version__
from pypef.gui.qt_window import MainWidget


@pytest.fixture(scope="session")
def app():
    return QApplication(sys.argv)


@pytest.mark.pip_specific
def test_button_click_changes_label(app):
    window = MainWidget()
    window.show()

    assert window.version_text.text() == f"PyPEF v. {__version__}"

    import tempfile
    tmp_dir = tempfile.mkdtemp()
    with patch("PySide6.QtWidgets.QFileDialog.getExistingDirectory", return_value=str(tmp_dir)):
        window.button_work_dir.click()
        app.processEvents()

    assert hasattr(window, "working_directory")
    assert window.working_directory == str(tmp_dir)

