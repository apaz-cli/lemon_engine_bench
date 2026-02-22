#!/bin/bash
set -e

echo "Installing vLLM for ROCm..."

# Install vLLM with ROCm support
pip install vllm --extra-index-url https://rocm.github.io/vllm/stable

echo "vLLM installation complete."