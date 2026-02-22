#!/bin/bash
set -e

PORT=${SGLANG_PORT:-8002}
MODEL_PATH=${SGLANG_MODEL_PATH:-"models/TinyLlama-1.1B-Chat-v1.0-HF"}
MODEL_NAME=${SGLANG_MODEL_NAME:-"TinyLlama-1.1B-Chat-v1.0"}

echo "Starting SGLang server on port $PORT with model $MODEL_NAME..."

# Docker mode
if command -v docker &> /dev/null && docker image ls | grep -q sglang; then
    docker run --rm -it \
        --network host \
        --device /dev/kfd --device /dev/dri \
        -v "$MODEL_PATH:/model" \
        lmsysorg/sglang:latest-rocm \
        python -m sglang.launch_server \
            --model-path "/model" \
            --port "$PORT" \
            --host "0.0.0.0"
else
    # Native mode
    python -m sglang.launch_server \
        --model-path "$MODEL_PATH" \
        --port "$PORT" \
        --host "0.0.0.0"
fi