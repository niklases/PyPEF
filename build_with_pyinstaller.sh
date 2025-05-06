#!/bin/bash
pip install -e .
pyinstaller \
  --console \
  --collect-data torch \
  --collect-data biotite \
  --collect-all biotite \
  --collect-data torch_geometric \
  --collect-all torch_geometric \
  --hidden-import torch_geometric \
  --noconfirm \
  gui/qt_window.py
  #  --add-data=X/:X/. \