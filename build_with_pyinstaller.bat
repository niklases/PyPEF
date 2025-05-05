REM Up to now pastes DLLs from local Python environment bin's to _internal...
REM alternative?: set PATH=%PATH%;c:\Users\nikla\miniconda3\envs\py312\Library\bin\;
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
  --add-binary=c:\Users\nikla\miniconda3\envs\py312\Library\bin\onedal_thread.3.dll:.^
  --add-binary=c:\Users\nikla\miniconda3\envs\py312\Library\bin\tbbbind.dll:.^
  --add-binary=c:\Users\nikla\miniconda3\envs\py312\Library\bin\tbbbind_2_0.dll:.^
  --add-binary=c:\Users\nikla\miniconda3\envs\py312\Library\bin\tbbbind_2_5.dll:.^
  --add-binary=c:\Users\nikla\miniconda3\envs\py312\Library\bin\tbbmalloc.dll:.^
  --add-binary=c:\Users\nikla\miniconda3\envs\py312\Library\bin\tbbmalloc_proxy.dll:.^
  --noconfirm^
  gui\qt_window.py
