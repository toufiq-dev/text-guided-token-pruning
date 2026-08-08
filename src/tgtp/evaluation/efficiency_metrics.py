"""Efficiency metrics for pruning evaluation.

The headline numbers the thesis reports:

* ``token_reduction_ratio``: (N - K) / N — how many visual tokens were removed.
* ``estimate_flops_reduction``: decoder-attention-FLOPs saved by removing tokens.
  Modeled per decoder layer as 4·L·H² per token per layer for the two QKV+output
  projections plus the attention score/logit products (standard back-of-envelope;
  see Section 4 of the proposal). Conservative and transparent — this is an
  *estimate for relative comparison*, not a profiler.
* ``measure_latency``: wall-clock ms per call (works on CPU and GPU).
* ``peak_vram_mb``: torch.cuda.max_memory_allocated when on CUDA.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np


def token_reduction_ratio(kept: int, total: int) -> float:
    """Fraction of tokens removed: (total - kept) / total."""
    assert 0 <= kept <= total, "kept must be within [0, total]"
    return (total - kept) / total


def estimate_decoder_flops(
    num_visual_tokens: int,
    llm_layers: int,
    hidden_dim: int,
    num_text_tokens: int = 0,
) -> float:
    """Back-of-envelope attention FLOPs for one image through the LLM decoder.

    Per layer, per token: QKV projection (3·H²), output projection (H²), and the
    attention logits + softmax-weighted sum (~2·L·H where L = sequence length).
    We drop the small L·H term and report 4·layers·H²·tokens — the dominant
    projection cost. Multiply by 2 if you want MACs-based FLOPs (we count
    multiply-adds as 2 FLOPs consistently for all configs).
    """
    tokens = num_visual_tokens + num_text_tokens
    per_token_per_layer = 4 * hidden_dim * hidden_dim
    return float(llm_layers * tokens * per_token_per_layer)


def estimate_flops_reduction(
    original_tokens: int,
    kept_tokens: int,
    llm_layers: int,
    hidden_dim: int,
    num_text_tokens: int = 0,
) -> dict[str, float]:
    """FLOPs before/after pruning plus the reduction ratio."""
    full = estimate_decoder_flops(original_tokens, llm_layers, hidden_dim, num_text_tokens)
    pruned = estimate_decoder_flops(kept_tokens, llm_layers, hidden_dim, num_text_tokens)
    return {
        "flops_full": full,
        "flops_pruned": pruned,
        "flops_saved": full - pruned,
        "flops_reduction_ratio": (full - pruned) / full if full > 0 else 0.0,
    }


def measure_latency(fn: Callable[[], Any], repeats: int = 5, warmup: int = 1) -> dict[str, float]:
    """Mean/median/std wall-clock latency of ``fn`` in ms (CPU- and GPU-safe)."""
    try:
        import torch

        sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    except Exception:  # noqa: BLE001 - torch unavailable
        sync = lambda: None  # noqa: E731

    for _ in range(warmup):
        fn()
        sync()
    times: list[float] = []
    for _ in range(repeats):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()  # ensure GPU work finished before reading the clock
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(times, dtype=float)
    return {
        "latency_mean_ms": float(arr.mean()),
        "latency_median_ms": float(np.median(arr)),
        "latency_std_ms": float(arr.std()),
        "latency_repeats": len(times),
    }


def peak_vram_mb(device: str = "cuda") -> float | None:
    """Peak allocated VRAM in MB (None if CUDA unavailable)."""
    try:
        import torch

        if device == "cuda" and torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated() / 1024**2)
    except Exception:  # noqa: BLE001 - pragma: no cover - defensive
        return None
    return None


def format_metrics_table(rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> str:
    """Render a list of metric dicts as a GitHub-flavored markdown table."""
    header = "| " + " | ".join(columns) + " |"
    sep = "|" + "|".join(["---"] * len(columns)) + "|"
    lines = [header, sep]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(c)) for c in columns) + " |")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
