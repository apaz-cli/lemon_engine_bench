#!/bin/bash
set -e

echo "Installing SGLang for ROCm..."

# Option 1: Docker (recommended)
if command -v docker &> /dev/null; then
    echo "Docker detected. Pulling SGLang ROCm image..."
    docker pull lmsysorg/sglang:latest-rocm
else
    echo "Docker not found. Trying pip install..."
    pip install sglang[all]
fi

echo "SGLang installation complete."