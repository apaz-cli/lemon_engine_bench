#!/usr/bin/env python3
"""Register locally-converted GGUF files with Lemonade's model registry.

Lemonade maintains a model registry at ~/.cache/lemonade/user_models.json.
This script writes entries that point to our own GGUF files (converted from
the same HF weights used by the HuggingFace adapter), so that all engines in
the benchmark use identical underlying weights.

Run once after download_models.sh, before launching the Lemonade server:

    python setup/register_models.py
"""
import json
import os
import sys
from pathlib import Path

LEMONADE_REGISTRY = Path.home() / ".cache" / "lemonade" / "user_models.json"

# Maps Lemonade model name → relative GGUF path inside the project.
# The project root is inferred from this script's location.
MODELS = {
    "TinyLlama": "models/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf",
    "Llama-3.2-1B-Instruct": "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    LEMONADE_REGISTRY.parent.mkdir(parents=True, exist_ok=True)

    if LEMONADE_REGISTRY.exists():
        with open(LEMONADE_REGISTRY) as f:
            registry = json.load(f)
    else:
        registry = {}

    registered = []
    skipped = []

    for model_name, rel_gguf in MODELS.items():
        gguf_path = PROJECT_ROOT / rel_gguf
        if not gguf_path.exists():
            skipped.append(f"  {model_name}: {gguf_path} not found, skipping")
            continue

        registry[model_name] = {
            "checkpoint": str(gguf_path),
            "labels": ["custom"],
            "recipe": "llamacpp",
            "source": "local_upload",
            "suggested": True,
        }
        registered.append(f"  {model_name} → {gguf_path}")

    with open(LEMONADE_REGISTRY, "w") as f:
        json.dump(registry, f, indent=2)

    if registered:
        print("Registered:")
        print("\n".join(registered))
    if skipped:
        print("Skipped (GGUF not found — run download_models.sh first):")
        print("\n".join(skipped))
    if not registered:
        print("Nothing registered.", file=sys.stderr)
        sys.exit(1)

    print(f"\nRegistry written to {LEMONADE_REGISTRY}")


if __name__ == "__main__":
    main()
