import numpy as np
from typing import List, Dict, Any, Tuple
from scipy.stats import spearmanr


def normalize_token(token_str: str) -> str:
    """Normalize token strings for cross-engine comparison.

    llama.cpp returns tokens with a leading space (' Paris'), HuggingFace
    uses a '▁' prefix ('▁Paris'), while the HF *adapter* strips that prefix
    and returns 'Paris'.  This function converts all representations to the
    bare form used by the HF adapter so tokens can be aligned by string.
    """
    if token_str == " ":
        return ""  # lone-space token → empty string (avoids collisions)
    if token_str.startswith(" "):
        stripped = token_str.lstrip()
        if stripped == "":
            return ""
        token_str = "▁" + stripped
    if token_str.startswith("▁"):
        token_str = token_str[1:] if len(token_str) > 1 else ""
    return token_str.rstrip()

def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return np.sum(p * np.log(p / q))

def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

def top_k_overlap(p_ids: List[int], q_ids: List[int], k: int = 10) -> float:
    p_set = set(p_ids[:k])
    q_set = set(q_ids[:k])
    return len(p_set.intersection(q_set)) / k

def spearman_rank_correlation(p_probs: np.ndarray, q_probs: np.ndarray) -> float:
    if len(p_probs) < 2 or len(q_probs) < 2:
        return 0.0
    rho, _ = spearmanr(p_probs, q_probs)
    return rho

def logprob_rmse(p_logprobs: np.ndarray, q_logprobs: np.ndarray) -> float:
    min_len = min(len(p_logprobs), len(q_logprobs))
    if min_len == 0:
        return 0.0
    p = p_logprobs[:min_len]
    q = q_logprobs[:min_len]
    return np.sqrt(np.mean((p - q) ** 2))

def compute_perplexity(logprobs: List[float]) -> float:
    if not logprobs:
        return float('inf')
    nll = -np.mean(logprobs)
    return np.exp(nll)

def greedy_decode_metrics(seq1: List[int], seq2: List[int]) -> Dict[str, Any]:
    exact = seq1 == seq2
    divergence = None
    for i, (a, b) in enumerate(zip(seq1, seq2)):
        if a != b:
            divergence = i
            break
    lcp = 0
    for a, b in zip(seq1, seq2):
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
    }

def aggregate_metrics_across_prompts(metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    aggregated = {}
    for key in metrics_list[0].keys():
        values = [m[key] for m in metrics_list if isinstance(m[key], (int, float))]
        if values:
            aggregated[f"{key}_mean"] = np.mean(values)
            aggregated[f"{key}_std"] = np.std(values)
            aggregated[f"{key}_min"] = np.min(values)
            aggregated[f"{key}_max"] = np.max(values)
    return aggregated