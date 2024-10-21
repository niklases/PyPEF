#!/bin/bash
set -e 

python -m pip install -U pypef pyside6

printf "
import sys\nimport os\n
sys.path.append(os.path.dirname(os.path.abspath(__file__)))\n
from pypef.main import run_main\n\n
if __name__ == '__main__':
    run_main()
" > run.py

printf "#!/bin/bash
python gui/qt_window.py
" > run_pypef_gui.sh

echo "Finished installation..."
echo "+++      Created file       +++"
echo "+++    run_pypef_gui.sh     +++"
echo "+++ for future GUI starting +++"

chmod a+x ./run_pypef_gui.sh
./run_pypef_gui.sh && exit
