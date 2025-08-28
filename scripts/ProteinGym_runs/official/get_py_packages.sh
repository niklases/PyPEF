#!/bin/bash

wget https://github.com/niklases/PyPEF/archive/refs/heads/dev-v0.4.3.zip
unzip dev-v0.4.3.zip && rm dev-v0.4.3.zip
cd PyPEF-dev-v0.4.3/
pip install -r requirements.txt
pip install hydra-core
pip install .


