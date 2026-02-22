#!/usr/bin/env python3
"""Run all benchmark experiments and save results to a single JSON file.

Quick run:
    python scripts/run_all.py --num-prompts 5

Full run:
    python scripts/run_all.py
"""
import asyncio
import argparse
import json
import sys
from pathlib import Path
from typing import Optional, List

sys.path.append(str(Path(__file__).parent.parent))

from src.experiments.greedy_decode import run_greedy_decode
from src.experiments.distribution import run_distribution_experiment
from src.experiments.perplexity import run_perplexity_experiment
from src.config import MODELS, ENGINE_ENDPOINTS
from src.data.loader import load_all_datasets
from src.data.prompts import prepare_prefixes


async def run_full_benchmark(
    num_prompts: Optional[int] = None,
    model_keys: Optional[List[str]] = None,
    engines: Optional[List[str]] = None,
) -> dict:
    datasets = load_all_datasets()
    # Multilingual Wikipedia dataset scripts are deprecated upstream; skip.
    datasets.pop("multilingual", None)
    if num_prompts:
        datasets = {k: v[:num_prompts] for k, v in datasets.items()}

    if model_keys is None:
        model_keys = list(MODELS.keys())
    if engines is None:
        engines = list(ENGINE_ENDPOINTS.keys())

    # SGLang requires gfx942+ hardware; all other configured engines are active.
    working_engines = [e for e in engines if e != "sglang"]

    results: dict = {}

    # ------------------------------------------------------------------
    # Experiment 1: Greedy Decode
    # ------------------------------------------------------------------
    print("=== Experiment 1: Greedy Decode ===")
    greedy_results: dict = {}
    for model_key in model_keys:
        model_datasets: dict = {}
        for dataset_name, prompts in datasets.items():
            subset = prompts[:num_prompts or 5]
            result = await run_greedy_decode(
                model_key, subset, working_engines,
                max_tokens=50, temperature=0.0, runs=3,
            )
            model_datasets[dataset_name] = result
            print(f"  {model_key}/{dataset_name} done")
        greedy_results[model_key] = model_datasets
    results["greedy_decode"] = greedy_results

    # ------------------------------------------------------------------
    # Experiment 2: Next-Token Distribution
    # ------------------------------------------------------------------
    print("=== Experiment 2: Next-Token Distribution ===")
    distribution_results: dict = {}
    for model_key in model_keys:
        wikitext = datasets.get("wikitext", [])[:3]
        prefixes: list = []
        for text in wikitext:
            prefixes.extend(prepare_prefixes(text, [0.25, 0.5, 0.75]))
        prefixes = prefixes[:num_prompts or 9]

        result = await run_distribution_experiment(
            model_key, prefixes, working_engines, top_logprobs=50
        )
        distribution_results[model_key] = result
        print(f"  {model_key} done")
    results["distribution"] = distribution_results

    # ------------------------------------------------------------------
    # Experiment 3: Perplexity
    # ------------------------------------------------------------------
    print("=== Experiment 3: Per-Token Prompt Logprobs ===")
    perplexity_results: dict = {}
    for model_key in model_keys:
        wikitext = datasets.get("wikitext", [])[:5]
        result = await run_perplexity_experiment(
            model_key, wikitext, working_engines, top_logprobs=50
        )
        perplexity_results[model_key] = result
        print(f"  {model_key} done")
    results["perplexity"] = perplexity_results

    return results


def main():
    parser = argparse.ArgumentParser(description="Run the Lemonade correctness benchmark")
    parser.add_argument(
        "--num-prompts", type=int, default=5,
        help="Number of prompts per dataset (default: 5 for a quick run; "
             "use dataset max sizes from config for a full run)",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Model keys to test (default: all models in config)",
    )
    parser.add_argument(
        "--engines", nargs="+", default=None,
        help="Engines to test (default: all active engines in config)",
    )
    parser.add_argument(
        "--output", default="results/raw/full_run.json",
        help="Path to write the combined results JSON",
    )
    args = parser.parse_args()

    results = asyncio.run(
        run_full_benchmark(args.num_prompts, args.models, args.engines)
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
