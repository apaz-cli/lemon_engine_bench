#!/bin/bash
# Download model weights and convert to GGUF from source.
# All GGUF files are generated from the same HF weights to ensure identical
# underlying weights across engines (only quantization differs).
#
# Requires:
#   - Python venv activated (huggingface_hub installed)
#   - llama.cpp already built at setup/llama.cpp/build/bin/
#   - For Llama-3.2: run `huggingface-cli login` first (gated model)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$ROOT_DIR/models"
LLAMA_CPP_DIR="$SCRIPT_DIR/llama.cpp"
CONVERT="$LLAMA_CPP_DIR/convert_hf_to_gguf.py"
QUANTIZE="$LLAMA_CPP_DIR/build/bin/llama-quantize"

mkdir -p "$MODELS_DIR"

download_and_convert() {
    local repo_id="$1"
    local hf_dir="$2"
    local base_name="$3"

    echo "=== $repo_id ==="

    if [ ! -f "$hf_dir/config.json" ]; then
        echo "Downloading HF weights..."
        python -c "
from huggingface_hub import snapshot_download
snapshot_download('$repo_id', local_dir='$hf_dir')
"
    else
        echo "HF weights already present, skipping download."
    fi

    local f16="$MODELS_DIR/${base_name}-f16.gguf"
    local q4="$MODELS_DIR/${base_name}-Q4_K_M.gguf"

    if [ ! -f "$f16" ]; then
        echo "Converting to F16 GGUF..."
        python "$CONVERT" "$hf_dir" --outfile "$f16" --outtype f16
    else
        echo "F16 GGUF already present."
    fi

    if [ ! -f "$q4" ]; then
        echo "Quantizing to Q4_K_M..."
        "$QUANTIZE" "$f16" "$q4" Q4_K_M
    else
        echo "Q4_K_M GGUF already present."
    fi

    echo "Done: $repo_id"
    echo
}

download_and_convert \
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    "$MODELS_DIR/TinyLlama-1.1B-Chat-v1.0-HF" \
    "TinyLlama-1.1B-Chat-v1.0"

download_and_convert \
    "meta-llama/Llama-3.2-1B-Instruct" \
    "$MODELS_DIR/Llama-3.2-1B-Instruct-HF" \
    "Llama-3.2-1B-Instruct"

echo "All models ready in $MODELS_DIR"
