#!/bin/bash
set -e 

# Only required for WSL(?):
# sudo apt-get install -y libxcb-cursor-dev

python -m pip install -U pypef pyside6

printf "
import sys\nimport os\n
sys.path.append(os.path.dirname(os.path.abspath(__file__)))\n
from pypef.main import run_main\n\n
if __name__ == '__main__':
    run_main()
" > run.py

printf "#!/bin/bash\n
SCRIPT_DIR=\$( cd -- \"\$( dirname -- \"\${BASH_SOURCE[0]}\" )\" &> /dev/null && pwd )\n
python "\${SCRIPT_DIR}/gui/qt_window.py"\n
" > run_pypef_gui.sh

echo "Finished installation..."
echo "+++      Created file       +++"
echo "+++    run_pypef_gui.sh     +++"
echo "+++ for future GUI starting +++"

chmod a+x ./run_pypef_gui.sh
./run_pypef_gui.sh && exit
