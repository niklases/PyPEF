
# GUI created with PyQT/PySide
# PySide vs PyQT: https://www.pythonguis.com/faq/pyqt-vs-pyside/?gad_source=1&gclid=CjwKCAjwpbi4BhByEiwAMC8Jnfe7sYOqHjs5eOg_tYMD0iX3UDBduwykrF8qE5Y0IG66abhS6YXHvRoCg-kQAvD_BwE
# (If using PyQT, see: https://www.gnu.org/licenses/license-list.en.html#GPLCompatibleLicenses)

import sys
import os
from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtCore import QSize
# Up to now needs (pip) installed PyPEF version
from pypef import __version__

# https://stackoverflow.com/questions/67297494/redirect-console-output-to-pyqt5-gui-in-real-time
# sudo apt-get install -y libxcb-cursor-dev
pypef_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
#os.environ["PATH"] += os.pathsep + pypef_root


class MainWindow(QtWidgets.QWidget):
    def __init__(
            self, 
            pypef_root: str | None = None
    ):
        super().__init__()
        if pypef_root is None:
            self.pypef_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        else:
            self.pypef_root = pypef_root

        self.setMinimumSize(QSize(200, 100))
        self.setWindowTitle("PyPEF GUI")

        layout = QtWidgets.QVBoxLayout(self)
        self.text = QtWidgets.QLabel(f"PyPEF v. {__version__}", alignment=QtCore.Qt.AlignRight)
        self.textedit_out = QtWidgets.QTextEdit(readOnly=True)
        self.button_mklsts = QtWidgets.QPushButton("Run MKLSTS")
        self.button_mklsts.setMinimumWidth(80)
        #self.button_mklsts.setMaximumWidth(80)        
        self.button_mklsts.setToolTip("Create files for training and testing from variant-fitness CSV data")
        self.button_mklsts.clicked.connect(self.pypef_mklsts)

        self.button_dca_inference_gremlin = QtWidgets.QPushButton("Run GREMLIN")
        self.button_dca_inference_gremlin.setMinimumWidth(80)
        self.button_dca_inference_gremlin.setToolTip("To be implemented")

        self.setGeometry(100, 60, 1000, 800)
        layout.addWidget(self.text)
        layout.addWidget(self.button_mklsts)
        layout.addWidget(self.button_dca_inference_gremlin)
        layout.addWidget(self.textedit_out)

        self.process = QtCore.QProcess(self)
        #process_env = self.process.processEnvironment()
        #process_env.insert("PYTHONPATH", pypef_root) 
        #self.process.setProcessEnvironment(process_env)
        self.process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.on_readyReadStandardOutput)
        self.process.started.connect(lambda: self.button_mklsts.setEnabled(False))
        self.process.finished.connect(lambda: self.button_mklsts.setEnabled(True))

    def on_readyReadStandardOutput(self):
         text = self.process.readAllStandardOutput().data().decode()
         self.textedit_out.append(text)

    @QtCore.Slot()
    def pypef_mklsts(self):
        self.text.setText("Running MKLSTS...")
        wt_fasta_file = QtWidgets.QFileDialog.getOpenFileName(self, "Select WT FASTA File")[0]
        csv_variant_file = QtWidgets.QFileDialog.getOpenFileName(self, "Select variant CSV File")[0]
        self.exec_pypef(f'mklsts --wt {wt_fasta_file} --input {csv_variant_file}')       
    
    def exec_pypef(self, cmd):
        self.process.start(f'python', ['-u', f'{self.pypef_root}/pypef/main.py'] + cmd.split(' '))
        self.process.finished.connect(self.process_finished)

    def process_finished(self):
        self.text.setText("Finished...") 
        #self.process = None



if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    widget = MainWindow()
    widget.resize(800, 600)
    widget.show()
    sys.exit(app.exec())
