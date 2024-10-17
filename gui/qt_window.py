
# GUI created with PyQT/PySide
# PySide vs PyQT: https://www.pythonguis.com/faq/pyqt-vs-pyside/?gad_source=1&gclid=CjwKCAjwpbi4BhByEiwAMC8Jnfe7sYOqHjs5eOg_tYMD0iX3UDBduwykrF8qE5Y0IG66abhS6YXHvRoCg-kQAvD_BwE
# (If using PyQT, see: https://www.gnu.org/licenses/license-list.en.html#GPLCompatibleLicenses)

import re
import sys
import os
from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtCore import QSize
# Up to now needs (pip) installed PyPEF version
#from pypef import __version__

# https://stackoverflow.com/questions/67297494/redirect-console-output-to-pyqt5-gui-in-real-time
# sudo apt-get install -y libxcb-cursor-dev
pypef_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
with open(os.path.join(pypef_root, 'pypef', '__init__.py')) as fh:
    for line in fh:
        if line.startswith('__version__'):
            version = re.findall(r"[-+]?(?:\d*\.*\d.*\d+)", line)[0]
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
        self.c = 0
        self.setMinimumSize(QSize(200, 100))
        self.setWindowTitle("PyPEF GUI")

        # Texts #############################################################################
        layout = QtWidgets.QGridLayout(self)
        self.version_text = QtWidgets.QLabel(f"PyPEF v. {version}", alignment=QtCore.Qt.AlignRight)
        self.textedit_out = QtWidgets.QTextEdit(readOnly=True)
        self.textedit_out.setStyleSheet("font-family:DejaVu Sans Mono;font-size:12px;font-weight:normal;")
        self.utils_text = QtWidgets.QLabel("Utilities")

        self.supervised_text = QtWidgets.QLabel("Supervised")
        self.supervised_text_train_test = QtWidgets.QLabel("Train - Test")
        self.supervised_text_predict = QtWidgets.QLabel("Predict")

        self.unsupervised_text = QtWidgets.QLabel("Unsupervised (DCA)")

        # Buttons ###########################################################################
        self.button_mklsts = QtWidgets.QPushButton("Run MKLSTS")
        #self.button_mklsts.setMinimumWidth(80)
        #self.button_mklsts.setMaximumWidth(80)        
        self.button_mklsts.setToolTip("Create files for training and testing from variant-fitness CSV data")
        self.button_mklsts.clicked.connect(self.pypef_mklsts)

        self.button_dca_inference_gremlin = QtWidgets.QPushButton("GREMLIN (MSA optimization)")
        self.button_dca_inference_gremlin.setMinimumWidth(80)
        self.button_dca_inference_gremlin.setToolTip(
            "Generating DCA parameters using GREMLIN (\"MSA optimization\"), "
            "you have to provide an MSA in FASTA or A2M format"
        )
        self.button_dca_inference_gremlin.clicked.connect(self.pypef_gremlin)

        self.button_dca_test_gremlin = QtWidgets.QPushButton("Test GREMLIN")
        self.button_dca_test_gremlin.setMinimumWidth(80)
        self.button_dca_test_gremlin.setToolTip(
            "Test performance on any test dataset using the MSA-optimized GREMLIN model"
        )
        self.button_dca_test_gremlin.clicked.connect(self.pypef_gremlin_dca_test)

        self.button_dca_predict_gremlin = QtWidgets.QPushButton("Predict GREMLIN")
        self.button_dca_predict_gremlin.setMinimumWidth(80)
        self.button_dca_predict_gremlin.setToolTip(
            "Predict any dataset using the MSA-optimized GREMLIN model"
        )
        self.button_dca_predict_gremlin.clicked.connect(self.pypef_gremlin_dca_predict)

        self.button_supervised_train_gremlin = QtWidgets.QPushButton("Hybrid Training (GREMLIN)")
        self.button_supervised_train_gremlin.setMinimumWidth(80)
        self.button_supervised_train_gremlin.setToolTip(
            "Optimize the GREMLIN model by supervised training on variant-fitness labels"
        )
        self.button_supervised_train_gremlin.clicked.connect(self.pypef_gremlin_hybrid_train)

        self.button_supervised_train_test_gremlin = QtWidgets.QPushButton("Hybrid Train-Test (GREMLIN)")
        self.button_supervised_train_test_gremlin.setMinimumWidth(80)
        self.button_supervised_train_test_gremlin.setToolTip(
            "Optimize the GREMLIN model by supervised training on variant-fitness labels"
        )
        self.button_supervised_train_test_gremlin.clicked.connect(self.pypef_gremlin_hybrid_train_test)


        # Layout widgets ####################################################################
        # int fromRow, int fromColumn, int rowSpan, int columnSpan
        #self.setGeometry(100, 60, 1000, 800)
        layout.addWidget(self.version_text, 0, 0, 1, -1)

        layout.addWidget(self.utils_text, 1, 0, 1, 1)

        layout.addWidget(self.button_mklsts, 2, 0, 1, 1)

        layout.addWidget(self.unsupervised_text, 1, 1, 1, 1)
        layout.addWidget(self.button_dca_inference_gremlin, 2, 1, 1, 1)
        layout.addWidget(self.button_dca_test_gremlin, 3, 1, 1, 1)
        layout.addWidget(self.button_dca_predict_gremlin, 4, 1, 1, 1)

        layout.addWidget(self.supervised_text, 1, 2, 1, 1)
        layout.addWidget(self.button_supervised_train_gremlin, 2, 2, 1, 1)
        layout.addWidget(self.button_supervised_train_test_gremlin, 3, 2, 1, 1)

        layout.addWidget(self.textedit_out, 5, 0, 1, -1)


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
         self.c += 1
         self.textedit_out.append(text)

    @QtCore.Slot()
    def pypef_mklsts(self):
        self.text.setText("Running MKLSTS...")
        wt_fasta_file = QtWidgets.QFileDialog.getOpenFileName(self, "Select WT FASTA File")[0]
        csv_variant_file = QtWidgets.QFileDialog.getOpenFileName(self, "Select variant CSV File")[0]
        self.exec_pypef(f'mklsts --wt {wt_fasta_file} --input {csv_variant_file}')

    @QtCore.Slot()
    def pypef_gremlin(self):
        self.version_text.setText("Running GREMLIN (DCA) optimization on MSA...")
        wt_fasta_file = QtWidgets.QFileDialog.getOpenFileName(self, "Select WT FASTA File")[0]
        msa_file = QtWidgets.QFileDialog.getOpenFileName(self, "Select Multiple Sequence Alignment (MSA) file (in FASTA or A2M format)")[0]
        self.exec_pypef(f'param_inference --wt {wt_fasta_file} --msa {msa_file}')  # --opt_iter 100

    @QtCore.Slot()
    def pypef_gremlin_dca_test(self):
        self.version_text.setText("Testing GREMLIN (DCA) performance on provided test set...")
        test_set_file = QtWidgets.QFileDialog.getOpenFileName(self, "Select Test Set File in \"FASL\" format")[0]
        params_pkl_file = QtWidgets.QFileDialog.getOpenFileName(self, "GREMLIN parameter Pickle file")[0]
        self.exec_pypef(f'hybrid --ts {test_set_file} -m {params_pkl_file} --params {params_pkl_file}')  # --opt_iter 100

    @QtCore.Slot()
    def pypef_gremlin_dca_predict(self):
        self.version_text.setText("Predicting using the GREMLIN (DCA) model on provided prediction set...")
        prediction_file = QtWidgets.QFileDialog.getOpenFileName(self, "Select Prediction Set File in FASTA format")[0]
        params_pkl_file = QtWidgets.QFileDialog.getOpenFileName(self, "Select GREMLIN parameter Pickle file")[0]
        self.exec_pypef(f'hybrid --ps {prediction_file} -m {params_pkl_file} --params {params_pkl_file}')  # --opt_iter 100

    @QtCore.Slot()
    def pypef_gremlin_hybrid_train(self):
        self.version_text.setText("Hybrid (DCA-supervised) model training...")
        training_file = QtWidgets.QFileDialog.getOpenFileName(self, "Select Training Set File in \"FASL\" format")[0]
        params_pkl_file = QtWidgets.QFileDialog.getOpenFileName(self, "Select GREMLIN parameter Pickle file")[0]
        self.exec_pypef(f'hybrid --ls {training_file} --ts {training_file} -m {params_pkl_file} --params {params_pkl_file}')  # --opt_iter 100

    @QtCore.Slot()
    def pypef_gremlin_hybrid_train_test(self):
        self.version_text.setText("...")
        training_file = QtWidgets.QFileDialog.getOpenFileName(self, "Select Training Set File in \"FASL\" format")[0]
        test_file = QtWidgets.QFileDialog.getOpenFileName(self, "Select Test Set File in \"FASL\" format")[0]
        params_pkl_file = QtWidgets.QFileDialog.getOpenFileName(self, "Select GREMLIN parameter Pickle file")[0]
        self.exec_pypef(f'hybrid --ls {training_file} --ts {test_file} -m {params_pkl_file} --params {params_pkl_file}')  # --opt_iter 100
    
    def exec_pypef(self, cmd):
        self.process.start(f'python', ['-u', f'{self.pypef_root}/run.py'] + cmd.split(' '))
        self.process.finished.connect(self.process_finished)
        if self.c > 0:
            self.textedit_out.append("=" * 104 + "\n")

    def process_finished(self):
        self.version_text.setText("Finished...") 
        #self.process = None



if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    widget = MainWindow()
    widget.resize(800, 600)
    widget.show()
    sys.exit(app.exec())
