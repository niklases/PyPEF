
(
    echo:
    echo import sys
    echo import os
    echo:
    echo sys.path.append(os.path.dirname^(os.path.abspath^(__file__^)^)^)
    echo:
    echo from pypef.main import run_main
    echo:
    echo:
    echo if __name__ == '__main__':
    echo     run_main^(^)
) > run.py

powershell -Command "python -m pip install -U pypef pyside6"

echo powershell -Command ^"python gui/qt_window.py^" > run_pypef_gui.bat

echo Created file run_pypef_gui.bat for future GUI starting.

.\run_pypef_gui
