"""Experiment 3: Per-Token Prompt Logprobs (Perplexity).

Feeds complete texts to each engine's get_prompt_logprobs() method and
computes perplexity scores, enabling a direct comparison of how each engine
assigns probability to the same text.
"""
import asyncio
import numpy as np
from typing import List, Dict, Any
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import MODELS
from src.data.loader import load_all_datasets
from src.engines.factory import create_adapter


def compute_perplexity(logprobs: List[float]) -> float:
    """Compute perplexity from a list of per-token log-probabilities."""
    if not logprobs:
        return float("inf")
    return float(np.exp(-np.mean(logprobs)))


async def run_perplexity_experiment(
    model_key: str,
    texts: List[str],
    engines: List[str],
    top_logprobs: int = 50,
) -> Dict[str, Any]:
    """Run the perplexity experiment for one model.

    Args:
        model_key:   Key from config.MODELS (e.g. "TinyLlama-1.1B-Chat")
        texts:       List of full texts to score
        engines:     List of engine names to evaluate
        top_logprobs: Number of top logprobs to request per token position

    Returns:
        Dict with keys: model, engines, texts, results
    """
    model_info = MODELS.get(model_key)
    if model_info is None:
        raise ValueError(f"Unknown model key: {model_key!r}")

    adapters = {}
    for engine in engines:
        adapter = create_adapter(engine, model_key, model_info)
        if adapter is not None:
            adapters[engine] = adapter

    results: Dict[str, Dict] = {}
    for text in texts:
        text_results = {}
        for engine_name, adapter in adapters.items():
            try:
                prompt_logprobs = await adapter.get_prompt_logprobs(text, top_logprobs)
                token_logprobs = [pos.generated_token.logprob for pos in prompt_logprobs]
                text_results[engine_name] = {
                    "perplexity": compute_perplexity(token_logprobs),
                    "token_logprobs": token_logprobs,
                    "num_tokens": len(token_logprobs),
                }
            except Exception as e:
                print(f"Error with {engine_name}: {e}")
                text_results[engine_name] = {"error": str(e)}
        results[text] = text_results

    for adapter in adapters.values():
        await adapter.close()

    return {
        "model": model_key,
        "engines": engines,
        "texts": list(texts),
        "results": results,
    }


# ---------------------------------------------------------------------------
# Quick standalone test
# ---------------------------------------------------------------------------

async def _main():
    datasets = load_all_datasets()
    wikitext = datasets["wikitext"][:1]
    engines = ["huggingface", "llamacpp", "lemonade"]
    model_key = "TinyLlama-1.1B-Chat"
    print(f"Testing {model_key}")
    results = await run_perplexity_experiment(model_key, wikitext, engines)
    print(f"Completed {model_key}")
    import json
    with open(f"results/perplexity_{model_key}.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    asyncio.run(_main())
