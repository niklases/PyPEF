#!/bin/bash
set -e

# "fold_rand_multiples" "fold_random_5" "fold_modulo_5" "fold_contiguous_5"
split_methods=("fold_random_5" "fold_modulo_5" "fold_contiguous_5")
 
 cd data
#./download_data.sh
cd ..
 
#./get_py_packages.sh

cd benchmark_runs

for split_method in "${split_methods[@]}"; do
    ./run_over_all.sh split_method=$split_method # > output.log 2>&1 &
    done

python estimate_performance.py