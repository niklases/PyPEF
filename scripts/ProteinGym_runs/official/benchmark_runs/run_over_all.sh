#!/bin/bash

set -e

# Exit if no arguments provided
if [ "$#" -eq 0 ]; then
    echo "Error: arguments are required (e.g. split_method=fold_random_5 max_idx=216)"
    exit 1
fi

# Initialize variables
split_method=""
max_idx=""

for arg in "$@"; do
    case $arg in
        split_method=*)
            split_method="${arg#*=}"
            ;;
        max_idx=*)
            max_idx="${arg#*=}"
            ;;
        *)
            echo "Unknown argument: $arg"
            exit 1
            ;;
    esac
done

# Check if split_method was set
if [ -z "$split_method" ]; then
    echo "Error: split_method not provided (use split_method=your_value)"
    exit 1
fi

# Define allowed split methods
allowed_split_methods=("fold_rand_multiples" "fold_random_5" "fold_modulo_5" "fold_contiguous_5")

# Check if split_method is valid
is_valid=false
for method in "${allowed_split_methods[@]}"; do
    if [ "$split_method" = "$method" ]; then
        is_valid=true
        break
    fi
done

if [ "$is_valid" = false ]; then
    echo "Error: Invalid split_method '$split_method'"
    echo "Allowed values are: ${allowed_split_methods[*]}"
    exit 1
fi

# Set default max_idx based on split_method if not explicitly provided
if [ -z "$max_idx" ]; then
    if [ "$split_method" = "fold_rand_multiples" ]; then
        max_idx=68
    else
        max_idx=216
    fi
fi

# Validate max_idx is a non-negative integer
if ! [[ "$max_idx" =~ ^[0-9]+$ ]]; then
    echo "Error: max_idx must be a non-negative integer, got '$max_idx'"
    exit 1
fi

for llm in prosst+esm; do

    # Set hybrid model split scheme based on split_method
    if [ "$split_method" = "fold_rand_multiples" ]; then
        hybrid_model_split_scheme="block-random"
    elif [ "$split_method" = "fold_random_5" ]; then
        hybrid_model_split_scheme="block-random"
    elif [ "$split_method" = "fold_modulo_5" ]; then
        hybrid_model_split_scheme="block-random" # or "modulo"
    elif [ "$split_method" = "fold_contiguous_5" ]; then
        hybrid_model_split_scheme="block-random" # or "contiguous"     
    else
        hybrid_model_split_scheme="block-random"
    fi

    echo "Using max_idx=$max_idx"

    for ((i=0; i<=max_idx; i++)); do
        echo -e "\n\nRunning DMS_idx=$i with llm=$llm and split_method=$split_method and hybrid_model_split_scheme=$hybrid_model_split_scheme"

        python pgym_cv_benchmark.py \
            split_method=$split_method \
            DMS_idx=$i \
            llm=$llm \
            hybrid_model_split_scheme=$hybrid_model_split_scheme \
            loss_method=listMLE \
            n_ensemble_splits=1

        find ./model_saves/ -type f -name '*.pt' -delete || true
    done
done
