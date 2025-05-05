conda install cython
pip install -U pyinstaller
pip install -e .
pyinstaller^
  --console^
  --collect-all pypef^
  --collect-data torch^
  --collect-data biotite^
  --collect-all biotite^
  --collect-data torch_geometric^
  --collect-all torch_geometric^
  --hidden-import torch_geometric^
  --noconfirm^
  gui\qt_window.py
