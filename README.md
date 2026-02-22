# Lemonade Engine Benchmark

Numerical correctness benchmark for AMD LLM serving engines. Measures whether
[Lemonade](https://github.com/amd/lemon-engine), llama.cpp, and vLLM produce
output probability distributions that agree with HuggingFace transformers (FP16,
used as ground truth).

**Hardware**: AMD GPU, ROCm 7.0

---

## Install

### 1. Python environment

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```

### 2. Engines

```bash
cd setup
bash install_all.sh   # auto-detects GPU arch; skips SGLang on RDNA3 (gfx1100)
```

Or individually:

```bash
bash install_huggingface.sh   # torch + transformers (ROCm wheel)
bash install_llamacpp.sh      # builds from source with -DGGML_HIP=ON
bash install_lemonade.sh      # lemonade-sdk pip package or binary
bash install_vllm.sh          # optional; requires ROCm vLLM wheel
bash install_sglang.sh        # optional; requires MI300X / gfx942+
```

### 3. Models

```bash
bash setup/download_models.sh
```

Downloads HuggingFace checkpoints into `models/`, then converts them to F16
GGUF and quantizes to Q4_K_M using the llama.cpp tools built in step 2.
All GGUF files are generated from the same HF weights so that every engine
in the benchmark uses identical underlying weights.

> **Llama-3.2-1B-Instruct is a gated model.** Run `huggingface-cli login`
> and accept the license at huggingface.co/meta-llama/Llama-3.2-1B-Instruct
> before running the download script.

| Model key | HF path | GGUF (Q4_K_M) |
|---|---|---|
| `TinyLlama-1.1B-Chat` | `models/TinyLlama-1.1B-Chat-v1.0-HF` | `models/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf` |
| `Llama-3.2-1B-Instruct` | `models/Llama-3.2-1B-Instruct-HF` | `models/Llama-3.2-1B-Instruct-Q4_K_M.gguf` |

### 4. Register models with Lemonade

```bash
python setup/register_models.py
```

Lemonade maintains its own model registry at `~/.cache/lemonade/user_models.json`.
This script writes entries pointing to the GGUF files in `models/` so that
Lemonade and llama.cpp use the **exact same GGUF files** — not independently
cached or re-quantized copies.

Re-run this script any time you regenerate or move the GGUF files.

### 5. Launch servers

HuggingFace runs in-process. The other engines are HTTP servers that must be
running before you start the benchmark:

```bash
bash setup/launch_lemonade.sh   # port 8000
bash setup/launch_llamacpp.sh   # port 8003
# bash setup/launch_vllm.sh     # port 8001 (optional)
# bash setup/launch_sglang.sh   # port 8002 (optional, gfx942+ only)
```

### 6. Verify

```bash
python setup/health_check.py
```

Pings each configured server and confirms logprobs are present in responses.

---

## Run

### Quick run (5 prompts per dataset, ~minutes)

```bash
python scripts/run_all.py --num-prompts 5
```

### Full run (100/50/50 prompts per dataset, ~hours)

```bash
python scripts/run_all.py
```

### Subset of models or engines

```bash
python scripts/run_all.py --models TinyLlama-1.1B-Chat --engines huggingface llamacpp lemonade
```

All results are written to `results/raw/full_run.json`.

### Generate plots

```bash
python scripts/generate_plots.py
```

Reads `results/raw/full_run.json` and writes four PNG figures to `results/plots/`:

| File | Contents |
|---|---|
| `fig1_distribution_metrics.png` | KL divergence, top-10 overlap, Spearman ρ vs. HuggingFace ground truth |
| `fig2_kl_heatmap.png` | Pairwise KL divergence heatmap (all engine pairs) |
| `fig3_greedy_heatmap.png` | Pairwise greedy-decode exact-match rate (%) |
| `fig4_lcp_heatmap.png` | Pairwise mean longest-common-prefix length |

---

## Engines

| Engine | Status | Port | Logprobs API |
|---|---|---|---|
| HuggingFace transformers | reference | in-process | full-vocab logits via `model.generate()` |
| llama.cpp (HIP) | active | 8003 | `/completion` with `n_probs=N` |
| Lemonade | active | 8000 | `/api/v1/completions` with `logprobs=N, stream=false` |
| vLLM | optional | 8001 | `/v1/completions` with `logprobs=N` |
| SGLang | optional (gfx942+) | 8002 | `/generate` with `return_logprob=true` |

> **Lemonade constraint**: logprobs are only available on the `/api/v1/completions`
> endpoint with `stream=False`. The chat endpoint does not return logprobs.

---

## Experiments

### Exp 1 — Greedy decode agreement (`src/experiments/greedy_decode.py`)

Sends identical prompts to all engines at temperature=0 and compares output
token sequences. Each prompt is run 3× to detect non-determinism.

Metrics: exact-match rate, first-divergence position, longest-common-prefix length.

### Exp 2 — Next-token distribution (`src/experiments/distribution.py`)

Generates `max_tokens=1` with `top_logprobs=50` from each engine and compares
the top-50 logprobs against HuggingFace's full-vocab distribution. WikiText
paragraphs are truncated at 25%/50%/75% to create prefixes of varying lengths.

Metrics: KL divergence, JS divergence, top-10 overlap, Spearman ρ, logprob RMSE.

### Exp 3 — Per-token prompt logprobs (`src/experiments/perplexity.py`)

Feeds complete texts, collects per-token prompt logprobs from each engine, and
computes perplexity.

---

## Datasets

| Split | Source | Prompts | Purpose |
|---|---|---|---|
| In-distribution | WikiText-2 (≥50 tokens) | 100 | Standard training-domain text |
| OOD — code | HumanEval | 50 | Code completion |
| OOD — math | GSM8K | 50 | Mathematical reasoning |

---

## Project structure

```
.
├── setup/
│   ├── install_all.sh           # installs all engines (GPU-arch aware)
│   ├── install_{engine}.sh      # per-engine install scripts
│   ├── launch_{engine}.sh       # per-engine server launch scripts
│   ├── download_models.sh       # downloads HF checkpoints + GGUF files
│   └── health_check.py          # pings servers and probes logprob responses
├── src/
│   ├── config.py                # models, endpoints, experiment parameters
│   ├── engines/
│   │   ├── base.py              # EngineAdapter ABC + GenerationResult dataclasses
│   │   ├── factory.py           # create_adapter() — single construction point
│   │   ├── token_utils.py       # TokenMapper: normalises token strings across engines
│   │   ├── huggingface.py
│   │   ├── llamacpp_engine.py
│   │   ├── lemonade_engine.py
│   │   ├── vllm_engine.py
│   │   └── sglang_engine.py
│   ├── data/
│   │   ├── loader.py            # WikiText-2, HumanEval, GSM8K loaders
│   │   └── prompts.py           # chat-template helpers, prefix generation
│   ├── experiments/
│   │   ├── greedy_decode.py
│   │   ├── distribution.py
│   │   └── perplexity.py
│   └── analysis/
│       └── metrics.py           # KL/JS divergence, top-k overlap, Spearman ρ, RMSE
├── scripts/
│   ├── run_all.py               # main entry point
│   └── generate_plots.py        # produces PNG figures from full_run.json
├── models/                      # local model files (gitignored)
├── results/                     # experiment outputs (gitignored)
├── pyproject.toml
└── requirements.txt
```

---

## Notes

**Token normalisation**: llama.cpp returns token strings with a leading space
(`' Paris'`); HuggingFace uses a `▁` prefix (`'▁Paris'`). `normalize_token()`
in `src/analysis/metrics.py` strips both to a bare form (`'Paris'`) for
cross-engine comparison. `TokenMapper` in `src/engines/token_utils.py` maps
normalised strings back to HuggingFace token IDs.

**Lemonade model IDs**: Lemonade's internal model registry uses names like
`user.TinyLlama` rather than HuggingFace repo names. The mapping lives in
`LEMONADE_MODEL_IDS` in `src/engines/factory.py`.

**SGLang**: skipped automatically on RDNA3 (gfx1100); requires MI300X (gfx942+).
