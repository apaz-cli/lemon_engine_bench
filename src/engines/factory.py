"""
Factory for creating engine adapters. Centralises model-ID resolution and
adapter construction so experiment modules don't duplicate this logic.
"""
import os
from typing import Optional, Dict, Any

from .huggingface import HuggingFaceAdapter
from .vllm_engine import VLLMAdapter
from .sglang_engine import SGLangAdapter
from .llamacpp_engine import LlamaCppAdapter
from .lemonade_engine import LemonadeAdapter

# Maps config model keys to the model-ID string Lemonade expects.
LEMONADE_MODEL_IDS: Dict[str, str] = {
    "TinyLlama-1.1B-Chat": "user.TinyLlama",
    "Llama-3.2-1B-Instruct": "extra.Llama-3.2-1B-Instruct-Q4_K_M.gguf",
}


def create_adapter(engine: str, model_key: str, model_info: Dict[str, Any]):
    """Create an EngineAdapter for the given engine and model.

    Args:
        engine:     Engine name: huggingface | vllm | sglang | llamacpp | lemonade
        model_key:  Key from the MODELS config (e.g. "TinyLlama-1.1B-Chat")
        model_info: Corresponding dict from the MODELS config

    Returns:
        An EngineAdapter instance, or None if the required model format is
        unavailable for this engine.
    """
    hf_path: Optional[str] = model_info.get("hf")
    gguf_path: Optional[str] = model_info.get("gguf_q4")

    if engine in ("huggingface", "sglang"):
        if hf_path is None:
            print(f"Skipping {engine} for {model_key}: HF model not available")
            return None
        model_path = hf_path
    elif engine == "vllm":
        # vLLM is launched with the GGUF and advertises the basename as its model ID.
        if gguf_path is None:
            print(f"Skipping vllm for {model_key}: GGUF model not available")
            return None
        model_path = os.path.basename(gguf_path)
    elif engine in ("llamacpp", "lemonade"):
        if gguf_path is None:
            print(f"Skipping {engine} for {model_key}: GGUF model not available")
            return None
        model_path = gguf_path
    else:
        print(f"Unknown engine: {engine}")
        return None

    if engine == "huggingface":
        return HuggingFaceAdapter(model_name=model_path)
    elif engine == "vllm":
        return VLLMAdapter(model_name=model_path, tokenizer_name=hf_path)
    elif engine == "sglang":
        return SGLangAdapter(model_name=model_path)
    elif engine == "llamacpp":
        return LlamaCppAdapter(model_name=model_path, tokenizer_name=hf_path)
    elif engine == "lemonade":
        if model_key in LEMONADE_MODEL_IDS:
            model_id = LEMONADE_MODEL_IDS[model_key]
        elif model_path.endswith(".gguf"):
            model_id = "extra." + os.path.basename(model_path)
        else:
            model_id = model_path
        return LemonadeAdapter(
            model_name=model_path,
            tokenizer_name=hf_path,
            model_id=model_id,
        )
    return None
