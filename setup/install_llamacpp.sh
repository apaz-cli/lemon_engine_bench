#!/bin/bash
set -e

echo "Installing llama.cpp from source with HIP support..."

# Clone llama.cpp if not already present
if [ ! -d "llama.cpp" ]; then
    git clone https://github.com/ggerganov/llama.cpp.git
fi
cd llama.cpp

# Build with HIP
mkdir -p build
cd build
HIP_PLATFORM=amd cmake .. -DGGML_HIP=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_COMPILER_ROCM_ROOT=/opt/rocm
cmake --build . --config Release --parallel $(nproc)

# Install Python bindings
cd ..
pip install -e .

echo "llama.cpp installation complete."