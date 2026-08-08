"""Unit tests for evaluation metrics, efficiency metrics, and diagnostics."""


import numpy as np
import pytest
import torch

from tgtp.evaluation.diagnostics import (
    detect_score_collapse,
    random_vs_smart_delta,
    score_distribution_stats,
    spatial_preservation,
    task_sensitivity_summary,
)
from tgtp.evaluation.efficiency_metrics import (
    estimate_decoder_flops,
    estimate_flops_reduction,
    measure_latency,
    peak_vram_mb,
    token_reduction_ratio,
)
from tgtp.evaluation.vqa_metrics import (
    compute_vqa_scores,
    exact_match,
    normalize_answer,
    summarize_scores,
    vqa_accuracy,
)


# ---------------------------------------------------------------- VQA metrics
def test_normalize_answer():
    assert normalize_answer("The car is Red.") == normalize_answer("car is red")
    assert normalize_answer("  A CAT  ") == "cat"
    assert normalize_answer("") == ""


def test_vqa_accuracy_multi_answer():
    # 'cat' matches both 'cat' and 'a cat' (article stripping) -> 2/3.
    assert vqa_accuracy("cat", ["cat", "a cat", "dog"]) == pytest.approx(2.0 / 3.0)
    # 2 of 3 -> 2/3.
    assert vqa_accuracy("dog", ["dog", "a dog", "cat"]) == pytest.approx(2.0 / 3.0)
    # all 3 (capped at 3) -> 1.0.
    assert vqa_accuracy("bird", ["bird", "the bird", "bird"]) == 1.0


def test_exact_match():
    assert exact_match("turquoise", ["turquoise", "blue"]) == 1.0
    assert exact_match("red", ["blue"]) == 0.0
    assert exact_match("The sky is blue", ["sky blue"]) == 0.0  # normalization, not fuzzy


def test_compute_and_summarize_scores():
    preds = ["cat", "dog", "bird"]
    refs = [["cat"], ["cat"], ["bird"]]
    scores = compute_vqa_scores(preds, refs, metric="exact_match")
    assert scores.tolist() == [1.0, 0.0, 1.0]
    summary = summarize_scores(scores)
    assert summary["mean"] == pytest.approx(2.0 / 3.0)
    assert summary["n"] == 3


def test_summarize_by_task():
    scores = np.array([1.0, 0.0, 0.5])
    summary = summarize_scores(scores, by_task={"spatial": [True, False, True], "ocr": [False, True, False]})
    assert summary["by_task"]["spatial"]["mean"] == pytest.approx(0.75)
    assert summary["by_task"]["ocr"]["mean"] == 0.0


# -------------------------------------------------------- efficiency metrics
def test_token_reduction_ratio():
    assert token_reduction_ratio(kept=40, total=100) == pytest.approx(0.6)
    assert token_reduction_ratio(kept=100, total=100) == 0.0
    assert token_reduction_ratio(kept=0, total=100) == 1.0


def test_flops_estimates_are_sane():
    full = estimate_decoder_flops(729, llm_layers=24, hidden_dim=2048)
    pruned = estimate_decoder_flops(364, llm_layers=24, hidden_dim=2048)
    assert full > pruned > 0
    stats = estimate_flops_reduction(729, 364, 24, 2048)
    assert 0.4 < stats["flops_reduction_ratio"] < 0.6  # ~50% token cut -> ~50% FLOPs cut


def test_measure_latency_cpu():
    stats = measure_latency(lambda: np.dot(np.ones((64, 64)), np.ones((64, 64))), repeats=3, warmup=1)
    assert stats["latency_mean_ms"] > 0
    assert stats["latency_repeats"] == 3


def test_peak_vram_mb_cpu_is_none():
    assert peak_vram_mb("cuda") is None  # no CUDA in CI


# ------------------------------------------------------------- diagnostics
def test_score_distribution_stats_and_collapse():
    scores = torch.rand(4, 100)
    stats = score_distribution_stats(scores)
    assert set(stats) >= {"mean", "std", "entropy", "collapse"}
    assert stats["collapse"] is False

    flat = torch.full((4, 100), 0.5)
    assert detect_score_collapse(flat) is True
    assert detect_score_collapse(scores) is False


def test_spatial_preservation():
    # 3x3 grid, keep the middle column-ish tokens (indices 1, 4, 7).
    stats = spatial_preservation([1, 4, 7], grid_h=3, grid_w=3)
    assert stats["row_coverage"] == 1.0
    assert stats["col_coverage"] == pytest.approx(1 / 3)
    assert stats["cell_coverage"] == pytest.approx(3 / 9)


def test_task_sensitivity_ordering():
    summary = task_sensitivity_summary(
        {"ocr": [0.5, 0.6], "spatial": [0.9, 0.95], "count": [0.8]}, baseline_mean=1.0
    )
    assert summary[0]["task"] == "ocr"  # most fragile first
    assert summary[-1]["task"] == "spatial"


def test_random_vs_smart_delta():
    d7 = random_vs_smart_delta({0.25: 0.80, 0.5: 0.75}, {0.25: 0.78, 0.5: 0.60})
    assert d7["ratios_where_smart_wins"] == 2
    assert d7["deltas"]["0.5"] == pytest.approx(0.15)


# ---------------------------------------------------------------- report text
def test_report_string_generation(tmp_path):
    from tgtp.evaluation.efficiency_metrics import format_metrics_table

    rows = [{"method": "full", "acc": 0.8, "flops": 0.0}, {"method": "random", "acc": 0.7, "flops": 0.5}]
    table = format_metrics_table(rows, ["method", "acc", "flops"])
    assert table.startswith("| method | acc | flops |")
    assert "random" in table
