"""Experiment 2: Next-Token Distribution Comparison.

For each prompt prefix, generates max_tokens=1 with top_logprobs=50 from
every engine, then computes ALL pairwise distribution metrics (not just vs.
HuggingFace), so that the full N×N matrix can be plotted.
"""
import asyncio
import numpy as np
from typing import List, Dict, Any
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import MODELS
from src.data.loader import load_all_datasets
from src.data.prompts import prepare_prefixes
from src.engines.factory import create_adapter
from src.analysis.metrics import normalize_token


# ---------------------------------------------------------------------------
# Pairwise metric computation
# ---------------------------------------------------------------------------

def _kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))


def _js(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    m = 0.5 * (p + q)
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def compute_pair_metrics(
    dist1: Dict[str, float],
    dist2: Dict[str, float],
    k: int = 10,
) -> Dict[str, Any]:
    """Compute distribution metrics between two top-k logprob dicts.

    Each dict maps normalized token string → logprob.
    Returns KL (symmetrized), JS, top-k overlap, Spearman ρ, logprob RMSE.
    """
    if not dist1 or not dist2:
        return {"kl_divergence": 1000.0, "js_divergence": 1000.0,
                "top10_overlap": 0.0, "spearman_rank": 0.0, "logprob_rmse": 1000.0}

    all_tokens = list(set(dist1) | set(dist2))
    p = np.array([np.exp(dist1.get(t, -100.0)) for t in all_tokens])
    q = np.array([np.exp(dist2.get(t, -100.0)) for t in all_tokens])
    p /= p.sum()
    q /= q.sum()

    # Symmetrized KL = (KL(P‖Q) + KL(Q‖P)) / 2
    kl = 0.5 * (_kl(p, q) + _kl(q, p))
    js = _js(p, q)

    top1 = [t for t, _ in sorted(dist1.items(), key=lambda x: x[1], reverse=True)]
    top2 = [t for t, _ in sorted(dist2.items(), key=lambda x: x[1], reverse=True)]
    overlap = len(set(top1[:k]) & set(top2[:k])) / k

    common = set(dist1) & set(dist2)
    if len(common) >= 2:
        from scipy.stats import spearmanr
        vals1 = np.array([dist1[t] for t in common])
        vals2 = np.array([dist2[t] for t in common])
        rho, _ = spearmanr(vals1, vals2)
        rmse = float(np.sqrt(np.mean((vals1 - vals2) ** 2)))
    else:
        rho, rmse = 0.0, 0.0

    return {
        "kl_divergence": kl,
        "js_divergence": js,
        "top10_overlap": overlap,
        "spearman_rank": float(rho),
        "logprob_rmse": rmse,
    }


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

async def run_distribution_experiment(
    model_key: str,
    prompts: List[str],
    engines: List[str],
    top_logprobs: int = 50,
) -> Dict[str, Any]:
    """Run the next-token distribution experiment for one model.

    Collects top-k distributions from every engine for each prompt, then
    computes ALL pairwise metrics so the full N×N matrix is available.

    Results structure:
        results[prompt]["{e1}_vs_{e2}"] = {kl_divergence, js_divergence, ...}
    """
    model_info = MODELS.get(model_key)
    if model_info is None:
        raise ValueError(f"Unknown model key: {model_key!r}")

    if model_info.get("hf") is None:
        print(f"Skipping {model_key}: HF ground-truth model not available")
        return {"model": model_key, "engines": engines, "prompts": list(prompts), "results": {}}

    adapters = {}
    for engine in engines:
        adapter = create_adapter(engine, model_key, model_info)
        if adapter is not None:
            adapters[engine] = adapter

    results: Dict[str, Dict] = {}

    for prompt in prompts:
        # Step 1 — collect the top-k distribution from every engine
        dists: Dict[str, Dict[str, float]] = {}
        for engine_name, adapter in adapters.items():
            try:
                result = await adapter.generate_with_logprobs(
                    prompt, max_tokens=1, top_logprobs=top_logprobs, temperature=0.0
                )
                if result.per_position_logprobs:
                    pos = result.per_position_logprobs[0]
                    dists[engine_name] = {
                        normalize_token(t.token_str): t.logprob
                        for t in pos.top_logprobs
                    }
                else:
                    print(f"WARNING: {engine_name} returned empty logprobs for: {prompt[:50]}…")
                    dists[engine_name] = {}
            except Exception as e:
                print(f"ERROR {engine_name} on prompt '{prompt[:40]}…': {e}")
                import traceback; traceback.print_exc()
                dists[engine_name] = {}

        # Step 2 — compute all pairwise metrics (upper triangle, symmetric)
        prompt_results: Dict[str, Dict] = {}
        engine_names = list(adapters.keys())
        for i, e1 in enumerate(engine_names):
            for j, e2 in enumerate(engine_names):
                if j <= i:
                    continue
                key = f"{e1}_vs_{e2}"
                prompt_results[key] = compute_pair_metrics(dists.get(e1, {}), dists.get(e2, {}))
        results[prompt] = prompt_results

    for adapter in adapters.values():
        await adapter.close()

    return {
        "model": model_key,
        "engines": list(adapters.keys()),
        "prompts": list(prompts),
        "results": results,
    }


# ---------------------------------------------------------------------------
# Quick standalone test
# ---------------------------------------------------------------------------

async def _main():
    datasets = load_all_datasets()
    wikitext = datasets["wikitext"][:1]
    prefixes = []
    for text in wikitext:
        prefixes.extend(prepare_prefixes(text, [0.25]))

    engines = ["huggingface", "llamacpp", "lemonade"]
    model_key = "TinyLlama-1.1B-Chat"
    print(f"Testing {model_key}")
    results = await run_distribution_experiment(model_key, prefixes, engines)
    print(f"Completed {model_key}")
    import json
    with open(f"results/distribution_{model_key}.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    asyncio.run(_main())
