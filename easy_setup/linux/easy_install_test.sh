#!/bin/bash

# Run we with ./easy_install_test.sh

# Easy "installer" for downloading repository and running local Python files (not pip-installed).
# Testing a few PyPEF commands on downloaded example files.
# Requires conda, i.e. Anaconda3 or Miniconda3 [https://docs.conda.io/en/latest/miniconda.html].

# 1. wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.5.2-0-Linux-x86_64.sh
# 2. bash Miniconda3-py310_23.5.2-0-Linux-x86_64.sh


# Echo on
set -x  
# Exit on errors
set -e
# echo script line numbers
export PS4='+(Line ${LINENO}): '

# Alternatively to using conda, you can use Python 3.10 and install packages via "python3 -m pip install -r requirements.txt"
#conda update -n base -c defaults conda
conda env remove -n pypef
#conda env create -f PyPEF-main/linux_env.yml
conda create -n pypef python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate pypef
python3 -m pip install -U pypef

#export PYTHONPATH=${PYTHONPATH}:${PWD}/PyPEF-main
#pypef='python3 '${PWD}'/PyPEF-main/pypef/main.py'
pypef --version

while true; do
    read -p "Test PyPEF installation (runs sudo apt-get update && sudo apt-get install unzip, downloads PyPEF repository, and runs a PyPEF-Python script, ~ 1 h run time) [Y/N]? " yn
    case $yn in
        [Yy]* ) 
			wget https://github.com/niklases/PyPEF/archive/refs/heads/main.zip;
			sudo apt-get update
			sudo apt-get install unzip
			unzip main.zip
			cd 'PyPEF-main/workflow'
			python3 ./api_encoding_train_test.py
			break;;
        [Nn]* ) 
			break;;
        * ) 
			echo "Please answer yes or no.";;
    esac
done

echo 'Done!';


