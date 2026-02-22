"""Experiment 1: Greedy Decode Agreement.

Sends identical prompts to all engines at temperature=0, compares the output
token sequences pairwise, and reports exact-match rate, first-divergence
position, and longest-common-prefix length.
"""
import asyncio
from typing import List, Dict, Any
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import MODELS
from src.data.loader import load_all_datasets
from src.engines.factory import create_adapter
from src.analysis.metrics import normalize_token


def compute_metrics(seq1: List[str], seq2: List[str]) -> Dict[str, Any]:
    """Compare two token-string sequences (normalised for cross-engine alignment)."""
    norm1 = [normalize_token(t) for t in seq1]
    norm2 = [normalize_token(t) for t in seq2]

    exact = norm1 == norm2

    divergence = None
    for i, (a, b) in enumerate(zip(norm1, norm2)):
        if a != b:
            divergence = i
            break

    lcp = 0
    for a, b in zip(norm1, norm2):
        if a == b:
            lcp += 1
        else:
            break

    return {
        "exact_match": exact,
        "first_divergence": divergence,
        "lcp_length": lcp,
        "len_seq1": len(seq1),
        "len_seq2": len(seq2),
        "original_seq1": seq1,
        "original_seq2": seq2,
    }


async def run_greedy_decode(
    model_key: str,
    prompts: List[str],
    engines: List[str],
    max_tokens: int = 50,
    temperature: float = 0.0,
    runs: int = 3,
) -> Dict[str, Any]:
    """Run the greedy-decode experiment for one model.

    Args:
        model_key:   Key from config.MODELS (e.g. "TinyLlama-1.1B-Chat")
        prompts:     List of prompt strings
        engines:     List of engine names to evaluate
        max_tokens:  Maximum tokens to generate per prompt
        temperature: Sampling temperature (0.0 = greedy)
        runs:        Number of repetitions per prompt (to detect non-determinism)

    Returns:
        Dict with keys: model, engines, prompts, results, comparisons
    """
    model_info = MODELS.get(model_key)
    if model_info is None:
        raise ValueError(f"Unknown model key: {model_key!r}")

    adapters = {}
    for engine in engines:
        adapter = create_adapter(engine, model_key, model_info)
        if adapter is not None:
            adapters[engine] = adapter

    # Generate token sequences (runs × engines × prompts)
    results: Dict[str, Dict] = {}
    for prompt in prompts:
        prompt_results = {}
        for engine_name, adapter in adapters.items():
            seqs = []
            for _ in range(runs):
                result = await adapter.generate_with_logprobs(
                    prompt, max_tokens=max_tokens, temperature=temperature, top_logprobs=1
                )
                token_strings = [pos.generated_token.token_str for pos in result.per_position_logprobs]
                seqs.append(token_strings)
            prompt_results[engine_name] = seqs
        results[prompt] = prompt_results

    # Pairwise comparisons (first run only)
    comparisons: Dict[str, Dict] = {}
    engines_present = list(adapters.keys())
    for prompt, engine_seqs in results.items():
        comparisons[prompt] = {}
        for i in range(len(engines_present)):
            for j in range(i + 1, len(engines_present)):
                e1, e2 = engines_present[i], engines_present[j]
                metrics = compute_metrics(engine_seqs[e1][0], engine_seqs[e2][0])
                comparisons[prompt][f"{e1}_vs_{e2}"] = metrics

    for adapter in adapters.values():
        await adapter.close()

    return {
        "model": model_key,
        "engines": engines,
        "prompts": list(prompts),
        "results": results,
        "comparisons": comparisons,
    }


# ---------------------------------------------------------------------------
# Quick standalone test
# ---------------------------------------------------------------------------

async def _main():
    datasets = load_all_datasets()
    prompts = datasets["wikitext"][:5]
    engines = ["huggingface", "llamacpp", "lemonade"]

    for model_key in MODELS:
        print(f"Testing {model_key}")
        results = await run_greedy_decode(model_key, prompts, engines)
        print(f"Completed {model_key}")
        import json
        with open(f"results/greedy_decode_{model_key}.json", "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    asyncio.run(_main())
