"""Evaluation: VQA metrics, efficiency metrics, and D1–D7 diagnostics."""

from tgtp.evaluation.efficiency_metrics import (
    estimate_flops_reduction,
    measure_latency,
    token_reduction_ratio,
)
from tgtp.evaluation.vqa_metrics import exact_match, normalize_answer, vqa_accuracy

__all__ = [
    "estimate_flops_reduction",
    "exact_match",
    "measure_latency",
    "normalize_answer",
    "token_reduction_ratio",
    "vqa_accuracy",
]
