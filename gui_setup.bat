@echo off
echo Installing PyPEF...

set "python_exe=python"

set /P AREYOUSURE=Install and use local Python version (Y/[N]) (downloads Python installer and installs Python locally in the current working directory)?
if /I "%AREYOUSURE%" NEQ "Y" if /I "%AREYOUSURE%" NEQ "y" goto NO_PYTHON

echo Installing Python...
powershell -Command "$ProgressPreference = 'SilentlyContinue';Invoke-WebRequest https://www.python.org/ftp/python/3.12.7/python-3.12.7-amd64.exe -OutFile python-3.12.7-amd64.exe"

.\python-3.12.7-amd64.exe /quiet TargetDir="%~dp0Python3127" Include_pip=1 Include_test=0 AssociateFiles=1 PrependPath=0 CompileAll=1 InstallAllUsers=0

REM Not removing Python installer EXE as it can be used for easy uninstall/repair
REM del /Q python-3.12.7-amd64.exe
set "python_exe=.\Python3127\python.exe"


:NO_PYTHON

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

powershell -Command "%python_exe% -m pip install -U pypef pyside6"

(
    echo @echo off
    echo:
    echo start /min cmd /c powershell -Command ^"%%~dp0%python_exe% %%~dp0gui\qt_window.py^"
 ) > run_pypef_gui.bat

echo Finished installation...
echo +++      Created file       +++ 
echo +++    run_pypef_gui.bat    +++
echo +++ for future GUI starting +++
echo You can close this window now.

REM call .\run_pypef_gui.bat 
start cmd /c ".\run_pypef_gui.bat"
cmd /k
