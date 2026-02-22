#!/usr/bin/env python3
"""Generate article-quality figures from benchmark results.

Reads results/raw/full_run.json (produced by scripts/run_all.py) and writes
PNG figures to results/plots/.
"""
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

ENGINE_COLORS = {
    "huggingface": "#1f77b4",
    "vllm":        "#ff7f0e",
    "llamacpp":    "#d62728",
    "lemonade":    "#9467bd",
    "sglang":      "#2ca02c",
}
ENGINE_LABELS = {
    "huggingface": "HuggingFace\n(FP16 ref)",
    "vllm":        "vLLM\n(Q4_K_M)",
    "llamacpp":    "llama.cpp\n(Q4_K_M)",
    "lemonade":    "Lemonade\n(Q4_K_M)",
    "sglang":      "SGLang",
}
ENGINE_SHORT = {
    "huggingface": "HuggingFace",
    "vllm":        "vLLM",
    "llamacpp":    "llama.cpp",
    "lemonade":    "Lemonade",
}
DATASET_LABELS = {
    "wikitext":  "WikiText-2",
    "humaneval": "HumanEval",
    "gsm8k":     "GSM8K",
}


def set_style():
    sns.set_theme(style="whitegrid", font_scale=1.15)
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
    })


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------

def _canonical_pair(e1: str, e2: str) -> tuple:
    """Return (e1, e2) in a consistent order for lookups."""
    return (e1, e2) if (e1, e2) <= (e2, e1) else (e2, e1)


def extract_distribution_metrics(dist_data: dict) -> dict:
    """Return {engine: {metric: mean_value}} for bars (vs. HuggingFace only)."""
    acc: dict = {}
    for model_data in dist_data.values():
        for prompt_pairs in model_data.get("results", {}).values():
            for pair_key, m in prompt_pairs.items():
                parts = pair_key.split("_vs_")
                if len(parts) != 2:
                    continue
                e1, e2 = parts
                # Only collect pairs that involve huggingface
                if "huggingface" not in (e1, e2):
                    continue
                eng = e2 if e1 == "huggingface" else e1
                for k in ("kl_divergence", "js_divergence", "top10_overlap", "spearman_rank"):
                    v = m.get(k)
                    if v is not None and v < 999:
                        acc.setdefault(eng, {}).setdefault(k, []).append(v)
    return {eng: {k: float(np.mean(v)) for k, v in metrics.items()}
            for eng, metrics in acc.items()}


def extract_pairwise_matrix(dist_data: dict, metric: str = "kl_divergence") -> tuple:
    """Return (ordered_engines, NxN mean-metric matrix) for all engine pairs.

    The matrix is symmetric (symmetrised KL or JS); diagonal = 0.
    """
    # Accumulate per pair
    pair_acc: dict = {}
    all_engines: set = set()

    for model_data in dist_data.values():
        engine_list = model_data.get("engines", [])
        all_engines.update(engine_list)
        for prompt_pairs in model_data.get("results", {}).values():
            for pair_key, m in prompt_pairs.items():
                parts = pair_key.split("_vs_")
                if len(parts) != 2:
                    continue
                e1, e2 = parts
                all_engines.update([e1, e2])
                v = m.get(metric)
                if v is not None and v < 999:
                    canon = _canonical_pair(e1, e2)
                    pair_acc.setdefault(canon, []).append(v)

    # Stable engine ordering: HF first, then alphabetical
    engines = sorted(all_engines, key=lambda e: (e != "huggingface", e))
    n = len(engines)
    idx = {e: i for i, e in enumerate(engines)}

    mat = np.full((n, n), np.nan)
    np.fill_diagonal(mat, 0.0)

    for (e1, e2), vals in pair_acc.items():
        i, j = idx[e1], idx[e2]
        v = float(np.mean(vals))
        mat[i, j] = v
        mat[j, i] = v  # symmetric

    return engines, mat


def extract_greedy_matrix(greedy_data: dict) -> tuple:
    """Return (ordered_engines, NxN mean exact-match-rate matrix).

    Averaged across all datasets and prompts.  Diagonal = 1.0.
    """
    pair_acc: dict = {}
    all_engines: set = set()

    for model_data in greedy_data.values():
        for dataset, ddata in model_data.items():
            comps = ddata.get("comparisons", {})
            for pairs in comps.values():
                for pair_key, m in pairs.items():
                    parts = pair_key.split("_vs_")
                    if len(parts) != 2:
                        continue
                    e1, e2 = parts
                    all_engines.update([e1, e2])
                    em = float(m.get("exact_match", False))
                    canon = _canonical_pair(e1, e2)
                    pair_acc.setdefault(canon, []).append(em)

    engines = sorted(all_engines, key=lambda e: (e != "huggingface", e))
    n = len(engines)
    idx = {e: i for i, e in enumerate(engines)}

    mat = np.ones((n, n))  # diagonal = 1.0 (self-match)
    for (e1, e2), vals in pair_acc.items():
        i, j = idx[e1], idx[e2]
        v = float(np.mean(vals))
        mat[i, j] = v
        mat[j, i] = v

    return engines, mat


# ---------------------------------------------------------------------------
# Figure 1 — Distribution metrics bar chart (vs. HF)
# ---------------------------------------------------------------------------

def fig_distribution_metrics(metrics: dict, out: Path):
    if not metrics:
        return

    engines = sorted(metrics.keys())
    metric_keys   = ["kl_divergence",          "top10_overlap",        "spearman_rank"]
    metric_labels = ["KL Divergence\n(↓ better)", "Top-10 Overlap\n(↑ better)", "Spearman ρ\n(↑ better)"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle("Next-Token Distribution vs. HuggingFace FP16 Ground Truth", fontsize=14, y=1.02)

    for ax, key, label in zip(axes, metric_keys, metric_labels):
        vals   = [metrics.get(e, {}).get(key, 0) for e in engines]
        colors = [ENGINE_COLORS.get(e, "#888") for e in engines]
        bars = ax.bar(
            [ENGINE_LABELS.get(e, e) for e in engines],
            vals, color=colors, edgecolor="white", linewidth=0.8,
        )
        ax.set_ylabel(label)
        ax.set_title(label.split("\n")[0])
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005 * (ax.get_ylim()[1] or 1),
                f"{v:.3f}", ha="center", va="bottom", fontsize=9,
            )
        ax.tick_params(axis="x", labelsize=9)
        ax.grid(axis="y", alpha=0.4)
        ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Shared heatmap helper
# ---------------------------------------------------------------------------

def _draw_heatmap(ax, engines: list, mat: np.ndarray, cmap, vmin, vmax,
                  fmt: str = ".2f", colorbar_label: str = ""):
    labels = [ENGINE_SHORT.get(e, e) for e in engines]
    n = len(engines)

    # Mask NaN cells
    display = np.where(np.isnan(mat), -1, mat)
    im = ax.imshow(display, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(colorbar_label, fontsize=10)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)

    for i in range(n):
        for j in range(n):
            v = mat[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=10, color="#aaa")
            else:
                mid = vmin + (vmax - vmin) * 0.55
                color = "white" if v > mid else "black"
                ax.text(j, i, format(v, fmt), ha="center", va="center",
                        fontsize=10, color=color, fontweight="bold")

    return im


# ---------------------------------------------------------------------------
# Figure 2 — KL-divergence heatmap (all pairs)
# ---------------------------------------------------------------------------

def fig_kl_heatmap(engines: list, mat: np.ndarray, out: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    vmax = float(np.nanmax(mat[mat > 0])) * 1.05 if np.any(mat > 0) else 1.0
    _draw_heatmap(ax, engines, mat, cmap="YlOrRd", vmin=0, vmax=vmax,
                  fmt=".2f", colorbar_label="Mean Symmetrised KL Divergence")
    ax.set_title("Pairwise KL Divergence Between Engines", fontsize=13)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — Greedy decode exact-match heatmap (all pairs)
# ---------------------------------------------------------------------------

def fig_greedy_heatmap(engines: list, mat: np.ndarray, out: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    _draw_heatmap(ax, engines, mat * 100, cmap="RdYlGn", vmin=0, vmax=100,
                  fmt=".0f", colorbar_label="Exact Match Rate (%)")
    ax.set_title("Greedy Decode Exact Match Rate (%)\n(temp = 0, 50 tokens, averaged across datasets)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 — LCP heatmap (all pairs)
# ---------------------------------------------------------------------------

def extract_lcp_matrix(greedy_data: dict) -> tuple:
    pair_acc: dict = {}
    all_engines: set = set()

    for model_data in greedy_data.values():
        for dataset, ddata in model_data.items():
            comps = ddata.get("comparisons", {})
            for pairs in comps.values():
                for pair_key, m in pairs.items():
                    parts = pair_key.split("_vs_")
                    if len(parts) != 2:
                        continue
                    e1, e2 = parts
                    all_engines.update([e1, e2])
                    lcp = float(m.get("lcp_length", 0))
                    canon = _canonical_pair(e1, e2)
                    pair_acc.setdefault(canon, []).append(lcp)

    engines = sorted(all_engines, key=lambda e: (e != "huggingface", e))
    n = len(engines)
    idx = {e: i for i, e in enumerate(engines)}
    max_tokens = 50

    mat = np.full((n, n), float(max_tokens))  # diagonal = max (perfect agreement)
    for (e1, e2), vals in pair_acc.items():
        i, j = idx[e1], idx[e2]
        v = float(np.mean(vals))
        mat[i, j] = v
        mat[j, i] = v

    return engines, mat


def fig_lcp_heatmap(engines: list, mat: np.ndarray, out: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    vmax = float(np.nanmax(mat))
    _draw_heatmap(ax, engines, mat, cmap="RdYlGn", vmin=0, vmax=vmax,
                  fmt=".1f", colorbar_label="Mean LCP Length (tokens)")
    ax.set_title("Greedy Decode: Mean Longest Common Prefix\n(max 50 tokens, averaged across datasets)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate article figures from benchmark results")
    parser.add_argument("--full-run", default="results/raw/full_run.json")
    parser.add_argument("--output-dir", default="results/plots")
    args = parser.parse_args()

    data = load(Path(args.full_run))
    if not data:
        print(f"No results at {args.full_run}. Run scripts/run_all.py first.")
        sys.exit(1)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    set_style()

    saved = []

    if "distribution" in data:
        # Fig 1 — bar chart (vs. HF only)
        metrics = extract_distribution_metrics(data["distribution"])
        if metrics:
            p = out / "fig1_distribution_metrics.png"
            fig_distribution_metrics(metrics, p)
            saved.append(str(p))

        # Fig 2 — KL heatmap (all pairs)
        engines, mat = extract_pairwise_matrix(data["distribution"], "kl_divergence")
        if engines:
            p = out / "fig2_kl_heatmap.png"
            fig_kl_heatmap(engines, mat, p)
            saved.append(str(p))

    if "greedy_decode" in data:
        # Fig 3 — exact-match heatmap (all pairs)
        engines, mat = extract_greedy_matrix(data["greedy_decode"])
        if engines:
            p = out / "fig3_greedy_heatmap.png"
            fig_greedy_heatmap(engines, mat, p)
            saved.append(str(p))

        # Fig 4 — LCP heatmap (all pairs)
        engines, mat = extract_lcp_matrix(data["greedy_decode"])
        if engines:
            p = out / "fig4_lcp_heatmap.png"
            fig_lcp_heatmap(engines, mat, p)
            saved.append(str(p))

    if saved:
        print("Figures saved:")
        for p in saved:
            print(f"  {p}")
    else:
        print("No plottable data found.")


if __name__ == "__main__":
    main()
