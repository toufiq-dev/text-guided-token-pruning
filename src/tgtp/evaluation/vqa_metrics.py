"""VQA answer metrics.

* ``normalize_answer``: canonical VQA normalization (lowercase, strip articles
  and punctuation, collapse whitespace).
* ``vqa_accuracy``: VQAv2-style accuracy over up to 10 ground-truth answers —
  ``acc = min(1, matches / 3)``.
* ``exact_match``: 1.0 if the normalized prediction matches any reference.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

import numpy as np

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT = re.compile(r"[^\w\s]")


def normalize_answer(answer: str) -> str:
    """Canonical VQA answer normalization (mirrors the standard VQA eval)."""
    a = str(answer).lower().strip()
    a = _ARTICLES.sub(" ", a)
    a = _PUNCT.sub(" ", a)
    return " ".join(a.split())


def vqa_accuracy(prediction: str, references: Sequence[str]) -> float:
    """VQAv2-style accuracy: min(matches/3, 1) over up to 10 references."""
    pred = normalize_answer(prediction)
    matches = sum(1 for ref in references if pred == normalize_answer(ref))
    return min(matches / 3.0, 1.0)


def exact_match(prediction: str, references: Sequence[str]) -> float:
    """1.0 if the normalized prediction matches any reference."""
    pred = normalize_answer(prediction)
    return 1.0 if any(pred == normalize_answer(ref) for ref in references) else 0.0


def compute_vqa_scores(
    predictions: Iterable[str],
    references: Iterable[Sequence[str]],
    metric: str = "vqa_accuracy",
) -> np.ndarray:
    """Per-sample scores for a list of predictions/reference lists."""
    fn = vqa_accuracy if metric == "vqa_accuracy" else exact_match
    return np.array([fn(p, r) for p, r in zip(predictions, references, strict=False)], dtype=float)


def summarize_scores(
    per_sample: np.ndarray | list[float], by_task: dict = None  # type: ignore[assignment]
) -> dict:
    """Summarize per-sample scores; optionally group by task category (D2)."""
    arr = np.asarray(per_sample, dtype=float)
    summary: dict = {"mean": float(arr.mean()), "std": float(arr.std()), "n": int(arr.size)}
    if by_task:
        grouped: dict = {}
        for key, mask in by_task.items():
            vals = arr[np.asarray(mask, dtype=bool)]
            grouped[key] = {"mean": float(vals.mean()) if vals.size else float("nan"), "n": int(vals.size)}
        summary["by_task"] = grouped
    return summary
