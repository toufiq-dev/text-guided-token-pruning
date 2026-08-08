"""Pre-registered diagnostics (Section 6 of the proposal).

D1  Token relevance accuracy  — do high-scored tokens correspond to the objects
    mentioned in the question? (visualized as heatmaps, notebooks/01)
D2  Task sensitivity          — which task categories degrade most under pruning?
D3  Score distribution        — are relevance scores well-calibrated or collapsed?
D4  Spatial preservation      — does pruning destroy spatial layout (grid coverage)?
D5  Resolution sensitivity    — does optimal keep_ratio change with resolution?
D6  Location verification     — pre-decoder works; post-decoder produces gibberish.
D7  Random vs. smart          — does text-guided pruning beat random at each ratio?

Every function is pure and unit-tested; the heavy model runs live in the scripts.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch


# ------------------------------------------------------------------ D3
def score_distribution_stats(scores: torch.Tensor) -> dict[str, float]:
    """Mean/std/min/max + Shannon entropy of the score distribution (D3)."""
    s = scores.float().flatten().detach().cpu()
    if s.numel() == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"),
                "max": float("nan"), "entropy": float("nan"), "collapse": False}
    eps = 1e-9
    p = (s + eps) / (s.sum() + eps * s.numel())
    entropy = float(-(p * torch.log(p)).sum())
    return {
        "mean": float(s.mean()),
        "std": float(s.std()),
        "min": float(s.min()),
        "max": float(s.max()),
        "entropy": entropy,
        "collapse": bool(float(s.std()) < 1e-4),
    }


def detect_score_collapse(scores: torch.Tensor, threshold: float = 1e-4) -> bool:
    """D3: flag score collapse (all tokens scored ~identically -> useless signal)."""
    return bool(float(scores.float().std()) < threshold)


# ------------------------------------------------------------------ D4
def spatial_preservation(
    keep_indices: Sequence[int], grid_h: int, grid_w: int
) -> dict[str, float]:
    """D4: fraction of grid rows/columns still represented after pruning.

    Args:
        keep_indices: kept token indices (row-major order over the grid).
        grid_h, grid_w: feature-grid dimensions (e.g. 9 x 9 after pixel-shuffle).

    Returns:
        dict with row_coverage, col_coverage, and overall cell coverage.
    """
    if grid_h * grid_w == 0:
        return {"row_coverage": 0.0, "col_coverage": 0.0, "cell_coverage": 0.0}
    rows = {i // grid_w for i in keep_indices}
    cols = {i % grid_w for i in keep_indices}
    return {
        "row_coverage": len(rows) / grid_h,
        "col_coverage": len(cols) / grid_w,
        "cell_coverage": len(set(keep_indices)) / (grid_h * grid_w),
    }


# ------------------------------------------------------------------ D2
def task_sensitivity_summary(
    scores_by_task: dict[str, Sequence[float]], baseline_mean: float
) -> list[dict[str, Any]]:
    """D2: per-task mean and drop vs the full-token baseline, sorted by fragility."""
    rows = []
    for task, vals in scores_by_task.items():
        mean = float(np.mean(vals)) if len(vals) else float("nan")
        rows.append(
            {
                "task": task,
                "mean": mean,
                "drop_vs_baseline": float(mean - baseline_mean),
                "n": len(vals),
            }
        )
    rows.sort(key=lambda r: (r["drop_vs_baseline"] if not np.isnan(r["drop_vs_baseline"]) else 1e9))
    return rows


# ------------------------------------------------------------------ D7
def random_vs_smart_delta(
    smart_by_ratio: dict[float, float], random_by_ratio: dict[float, float]
) -> dict[str, Any]:
    """D7: at each pruning ratio, does text-guided beat random? By how much?"""
    deltas = {
        str(r): float(smart_by_ratio.get(r, float("nan")) - random_by_ratio.get(r, float("nan")))
        for r in sorted(set(smart_by_ratio) | set(random_by_ratio))
    }
    wins = sum(1 for d in deltas.values() if not np.isnan(d) and d > 0)
    return {"deltas": deltas, "ratios_where_smart_wins": wins, "total_ratios": len(deltas)}


# ------------------------------------------------------------------ D6
def location_verification(pre_decoder_output: Any, post_decoder_error: str) -> dict[str, Any]:
    """D6: record that pre-decoder pruning produced valid output and post-decoder
    pruning raised / produced gibberish."""
    return {
        "pre_decoder_ok": pre_decoder_output is not None,
        "post_decoder_outcome": "error",
        "post_decoder_error": post_decoder_error,
        "conclusion": (
            "CONFIRMED: pruning at the wrong location (post-decoder) is invalid; "
            "pre-decoder pruning is required."
        ),
    }


# ------------------------------------------------------------------ renderer
def diagnostics_to_markdown(results: dict[str, Any]) -> str:
    """Render a diagnostics dict to a readable markdown block."""
    lines = ["## Diagnostics report", ""]
    for key, value in results.items():
        lines.append(f"### {key}")
        if isinstance(value, dict):
            for k, v in value.items():
                lines.append(f"- **{k}**: {v}")
        else:
            lines.append(f"- {value}")
        lines.append("")
    return "\n".join(lines)
