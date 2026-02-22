#!/bin/bash
set -e

PORT=${VLLM_PORT:-8001}
MODEL_PATH=${VLLM_MODEL_PATH:-"models/gpt2"}
MODEL_NAME=${VLLM_MODEL_NAME:-"gpt2"}

echo "Starting vLLM server on port $PORT with model $MODEL_NAME..."
echo "Using Python from venv_vllm"

source venv_vllm/bin/activate

CUDA_VISIBLE_DEVICES="" VLLM_DEVICE=cpu python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --port "$PORT" \
    --api-key "dummy" \
    --served-model-name "$MODEL_NAME" \
    --max-model-len 4096 \
    --disable-log-stats \
    --log-level warning \
    --host 0.0.0.0 \
    --device cpu \
    --dtype float32