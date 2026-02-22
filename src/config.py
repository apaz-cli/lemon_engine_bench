from typing import Dict, List, Optional

# Models to test
MODELS = {
    "TinyLlama-1.1B-Chat": {
        "hf": "models/TinyLlama-1.1B-Chat-v1.0-HF",
        "gguf_q4": "models/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf",
        "gguf_q8": None,
        "gguf_fp16": "models/TinyLlama-1.1B-Chat-v1.0-f16.gguf",
    },
    "Llama-3.2-1B-Instruct": {
        "hf": "models/Llama-3.2-1B-Instruct-HF",
        "gguf_q4": "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "gguf_q8": None,
        "gguf_fp16": None,
    },
}

# Engine endpoints
ENGINE_ENDPOINTS = {
    "huggingface": {"type": "huggingface", "port": None},
    "vllm":        {"type": "vllm",        "port": 8001, "base_url": "http://localhost:8001/v1"},
    # "sglang": {"type": "sglang", "port": 8002, "base_url": "http://localhost:8002"},  # requires gfx942+
    "llamacpp":    {"type": "llamacpp",    "port": None},  # uses Python binding, no HTTP server
    "lemonade":    {"type": "lemonade",    "port": 8000, "base_url": "http://localhost:8000/api/v1"},
}

# Experiment parameters
GREEDY_DECODE_PARAMS = {
    "max_tokens": 50,
    "temperature": 0.0,
    "top_logprobs": 5,
}

DISTRIBUTION_PARAMS = {
    "max_tokens": 1,
    "temperature": 0.0,
    "top_logprobs": 50,
}

PERPLEXITY_PARAMS = {
    "top_logprobs": 50,
}

# Dataset sizes
DATASET_SIZES = {
    "wikitext": 100,
    "humaneval": 50,
    "gsm8k": 50,
    "multilingual": 50,
}

# Paths
RESULTS_DIR = "./results"
RAW_DIR = f"{RESULTS_DIR}/raw"
PROCESSED_DIR = f"{RESULTS_DIR}/processed"
PLOTS_DIR = f"{RESULTS_DIR}/plots"