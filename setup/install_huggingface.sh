#!/bin/bash
set -e

echo "Installing HuggingFace transformers (ground truth)..."

# Install PyTorch with ROCm 6.2
pip install "torch==2.9.1+rocm6.3" --index-url https://download.pytorch.org/whl/rocm6.3

# Install transformers, datasets, accelerate
pip install transformers datasets accelerate sentencepiece protobuf

echo "HuggingFace installation complete."