

import sys
import tempfile
import pytest
from unittest.mock import patch

from pypef import __version__



@pytest.fixture(scope="session")
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication(sys.argv)


@pytest.mark.pip_specific
def test_button_click_changes_label(app):
    from pypef.gui.qt_window import MainWidget
    window = MainWidget()
    window.show()

    assert window.version_text.text() == f"PyPEF v. {__version__}"

    tmp_dir = tempfile.mkdtemp()
    with patch("PySide6.QtWidgets.QFileDialog.getExistingDirectory", return_value=str(tmp_dir)):
        window.button_work_dir.click()
        app.processEvents()

    assert hasattr(window, "working_directory")
    assert window.working_directory == str(tmp_dir)

