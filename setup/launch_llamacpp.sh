#!/bin/bash
set -e

PORT=${LLAMACPP_PORT:-8003}
MODEL_PATH=${LLAMACPP_MODEL_PATH:-"models/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf"}

echo "Starting llama.cpp server on port $PORT with model $MODEL_PATH..."

./llama.cpp/build/bin/llama-server \
    -m "$MODEL_PATH" \
    -c 512 \
    --port "$PORT" \
    --host 0.0.0.0 \
    --log-disable \
    --n-gpu-layers -1