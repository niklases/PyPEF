#!/bin/bash
python -m pip install -U nuitka
conda install libpython-static
sudo apt install patchelf
python -m nuitka gui/qt_window.py \
    --mode=standalone \
    --enable-plugin=pyside6 \
    --include-data-dir=pypef/ml/AAindex=pypef/ml/. \
    #--include-distribution-metadata \
    #--onefile
