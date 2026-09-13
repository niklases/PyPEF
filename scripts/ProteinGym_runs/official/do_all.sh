#!/bin/bash
set -e

export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Usage:
#   ./do_all.sh
#   ./do_all.sh --max-idx 10
#   ./do_all.sh --download --max-idx 10
#   ./do_all.sh --setup --max-idx 10
#   ./do_all.sh --help

DOWNLOAD_DATA=false
GET_PY_PACKAGES=false
MAX_IDX=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --download)
            DOWNLOAD_DATA=true
            shift
            ;;
        --packages)
            GET_PY_PACKAGES=true
            shift
            ;;
        --setup)
            DOWNLOAD_DATA=true
            GET_PY_PACKAGES=true
            shift
            ;;
        --max-idx)
            if [[ $# -lt 2 ]]; then
                echo "Error: --max-idx requires a value"
                exit 1
            fi
            MAX_IDX="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--download] [--packages] [--setup] [--max-idx N]"
            echo
            echo "Options:"
            echo "  --download      Download benchmark data"
            echo "  --packages      Install/get Python packages"
            echo "  --setup         Download data and install/get Python packages"
            echo "  --max-idx N     Override maximum DMS_idx for all split methods"
            echo "  --help, -h      Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use '$0 --help' for usage."
            exit 1
            ;;
    esac
done

# Validate max_idx if provided
if [[ -n "$MAX_IDX" && ! "$MAX_IDX" =~ ^[0-9]+$ ]]; then
    echo "Error: --max-idx must be a non-negative integer, got '$MAX_IDX'"
    exit 1
fi


# Download data if requested
if $DOWNLOAD_DATA; then
    echo "Downloading data..."
    cd data
    ./download_data.sh
    cd ..
else
    echo "Skipping data download..."
fi


# Get Python packages if requested
if $GET_PY_PACKAGES; then
    echo "Getting Python packages..."
    ./get_py_packages.sh
else
    echo "Skipping Python package setup..."
fi

# Pre-download models from HuggingFace if desired
# hf download facebook/esm2_t33_650M_UR50D
# hf download AI4Protein/ProSST-2048

split_methods=("fold_random_5" "fold_modulo_5" "fold_contiguous_5")

cd benchmark_runs

for split_method in "${split_methods[@]}"; do
    echo -e "\n-----------------------------------------------------"
    echo "Running benchmark with split method: $split_method"

    if [[ -n "$MAX_IDX" ]]; then
        ./run_over_all.sh \
            "split_method=$split_method" \
            "max_idx=$MAX_IDX"
    else
        ./run_over_all.sh \
            "split_method=$split_method"
    fi
done

python estimate_performance.py
