#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPU_LIST=""
SCRIPT=""
SCRIPT_ARGS=()

# Parse options first
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)
            GPU_LIST="$2"
            shift 2
            ;;
        --*)
            echo "Unknown option: $1"
            exit 1
            ;;
        *)
            if [[ ! "$1" =~ \.py$ ]]; then
                echo "Error: Invalid usage. Expected format:"
                echo "  ./run_gpus.bash --gpus <gpu_list> <script.py> [script arguments...]"
                exit 1
            fi
            SCRIPT="$1"
            shift
            SCRIPT_ARGS=("$@")  # collect the rest as script args
            break
            ;;
    esac
done

# Require --gpus
if [[ -z "$GPU_LIST" ]]; then
    echo "Error: --gpus is required and must appear before the script"
    exit 1
fi

# Require script
if [[ -z "$SCRIPT" ]]; then
    echo "Usage: ./run_gpus.bash --gpus <gpu_list> <script.py> [args...]"
    exit 1
fi

# Set environment
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="$GPU_LIST"
if [[ ":$PYTHONPATH:" != *":$SCRIPT_DIR/../../:"* ]]; then
    export PYTHONPATH="$SCRIPT_DIR/../../:$PYTHONPATH"
fi

# Run Python script
python3 -O "$SCRIPT" "${SCRIPT_ARGS[@]}"
