#!/bin/bash
set -e

PORT=${LEMONADE_PORT:-8000}
MODEL_PATH=${LEMONADE_MODEL_PATH:-"models/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf"}

echo "Starting Lemonade server (Python dev server) on port $PORT..."

# lemonade-server-dev is the pip-installed Python server.
# The C++ binary server is faster but requires a separate installer.
HSA_OVERRIDE_GFX_VERSION=11.0.0 lemonade-server-dev serve \
    --port "$PORT" \
    --host 0.0.0.0 \
    --llamacpp rocm \
    --log-level warning