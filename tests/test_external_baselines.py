"""Unit tests for external head-to-head baseline orchestration (FastV, SparseVLM).

These test the pure logic — budget mapping, command construction, log parsing,
and results-JSON schema — without requiring the external repos or their
environments (FastV needs a patched transformers; SparseVLM needs its own conda
env). The actual subprocess integration is exercised by scripts/08 and /09 on
weeks 3–4 with the real checkouts.
"""

from __future__ import annotations

import json

import pytest

from tgtp.evaluation.external_baselines import (
    FASTV_AGG_LAYER,
    FASTV_ENTRY_SCRIPTS,
    FASTV_IMAGE_TOKEN_LENGTH,
    LLAVA15_NUM_VISUAL,
    SPARSEVLM_SUPPORTED_RETAIN,
    build_fastv_command,
    build_sparsevlm_command,
    build_sparsevlm_env,
    fastv_rank_for_keep_ratio,
    keep_ratio_from_retain,
    parse_accuracy_from_log,
    sparsevlm_retain_for_keep_ratio,
    write_external_results,
)


# ---------------------------------------------------------------- budget map
def test_fastv_rank_exact_quarter_steps():
    # LLaVA-1.5 has 576 visual tokens; keep 0.75/0.50/0.25 -> 432/288/144.
    assert fastv_rank_for_keep_ratio(0.75) == 432
    assert fastv_rank_for_keep_ratio(0.50) == 288
    assert fastv_rank_for_keep_ratio(0.25) == 144
    assert fastv_rank_for_keep_ratio(1.0) == 576


def test_fastv_rank_bounds():
    with pytest.raises(ValueError):
        fastv_rank_for_keep_ratio(0.0)
    with pytest.raises(ValueError):
        fastv_rank_for_keep_ratio(1.5)
    assert 1 <= fastv_rank_for_keep_ratio(0.001) <= LLAVA15_NUM_VISUAL


def test_sparsevlm_supported_set_is_native():
    # SparseVLM natively supports only these retained counts (README).
    assert SPARSEVLM_SUPPORTED_RETAIN == (192, 128, 96, 64)


def test_sparsevlm_exact_and_snapped_budgets():
    # 192/576 = 1/3 -> exact.
    retain, exact = sparsevlm_retain_for_keep_ratio(1 / 3)
    assert (retain, exact) == (192, True)
    # 0.5 of 576 = 288 -> not natively supported; snaps to nearest (192).
    retain, exact = sparsevlm_retain_for_keep_ratio(0.5)
    assert (retain, exact) == (192, False)


def test_keep_ratio_from_retain():
    assert keep_ratio_from_retain(432, 576) == pytest.approx(0.25)
    assert keep_ratio_from_retain(288, 576) == pytest.approx(0.5)
    assert keep_ratio_from_retain(192, 576) == pytest.approx(1 - 1 / 3)


# ------------------------------------------------------------ command builders
def test_build_fastv_command_defaults():
    # Interpreter-free: argv[0] is the entry script; the runner prepends its own
    # FastV-env python (never a literal "python" — that caused a double-interpreter bug).
    cmd = build_fastv_command("/repo", "src/FastV/inference/eval/inference_ocrvqa.py", rank=288)
    assert cmd[0] == "/repo/src/FastV/inference/eval/inference_ocrvqa.py"
    assert "python" not in cmd
    assert "--use-fast-v" in cmd
    assert "--fast-v-agg-layer" in cmd and str(FASTV_AGG_LAYER) in cmd
    assert "--fast-v-attention-rank" in cmd and "288" in cmd
    assert "--fast-v-image-token-length" in cmd and str(FASTV_IMAGE_TOKEN_LENGTH) in cmd
    assert "--fast-v-inplace" not in cmd
    # Runner-side composition: [interpreter, *cmd].
    argv = ["/opt/fastv_env/bin/python", *cmd]
    assert argv[0] == "/opt/fastv_env/bin/python"
    assert argv[1] == "/repo/src/FastV/inference/eval/inference_ocrvqa.py"


def test_build_fastv_command_inplace_and_extra():
    cmd = build_fastv_command("/repo", "x.py", rank=144, inplace=True, extra_args=["--model-path", "llava-hf/llava-1.5-7b-hf"])
    assert "--fast-v-inplace" in cmd
    assert cmd[-2:] == ["--model-path", "llava-hf/llava-1.5-7b-hf"]


def test_build_sparsevlm_env():
    env = build_sparsevlm_env(192)
    assert env["RETAIN_TOKN"] == "192"
    assert "USE_VERSION" not in env
    env2 = build_sparsevlm_env(128, use_version="2_0")
    assert env2["USE_VERSION"] == "2_0"
    # base env vars preserved
    base = {"PATH": "/x", "RETAIN_TOKN": "999"}
    env3 = build_sparsevlm_env(96, base_env=base)
    assert env3["PATH"] == "/x"
    assert env3["RETAIN_TOKN"] == "96"


def test_build_sparsevlm_command():
    argv = build_sparsevlm_command("/repo", "scripts/v1_5/eval/gqa.sh")
    assert argv == ["bash", "/repo/scripts/v1_5/eval/gqa.sh"]


def test_fastv_entry_scripts_cover_textvqa():
    # The head-to-head protocol mandates TextVQA; the repo ships this entry.
    assert "textvqa" in FASTV_ENTRY_SCRIPTS
    assert FASTV_ENTRY_SCRIPTS["textvqa"].endswith("inference_ocrvqa.py")


# ------------------------------------------------------------------- parsing
def test_parse_accuracy_percentage():
    log = "2026-08-09 12:00:00 INFO acc: 78.5% on 500 questions"
    assert parse_accuracy_from_log(log) == pytest.approx(0.785)


def test_parse_accuracy_fraction_and_overall():
    assert parse_accuracy_from_log("overall accuracy: 0.623") == pytest.approx(0.623)
    assert parse_accuracy_from_log("final accuracy = 0.5") == pytest.approx(0.5)
    assert parse_accuracy_from_log("Acc: 0.501") == pytest.approx(0.501)


def test_parse_accuracy_bare_percentage_without_pct_sign():
    # LLaVA-Evaluation (used by SparseVLM) prints e.g. "Accuracy: 71.2" — no %.
    assert parse_accuracy_from_log("Accuracy: 71.2") == pytest.approx(0.712)
    assert parse_accuracy_from_log("Overall accuracy: 0.71") == pytest.approx(0.71)
    assert parse_accuracy_from_log("acc = 48.52") == pytest.approx(0.4852)
    # The % form still works and is not double-divided.
    assert parse_accuracy_from_log("acc: 78.5%") == pytest.approx(0.785)
    # Bare "Overall: 0.71" without an accuracy keyword is ambiguous -> None.
    assert parse_accuracy_from_log("Overall: 0.71") is None


def test_parse_accuracy_none_when_absent():
    assert parse_accuracy_from_log("no accuracy here, just loss 1.23") is None
    assert parse_accuracy_from_log("") is None


# ---------------------------------------------------------- results JSON schema
def test_write_external_results_schema(tmp_path):
    path = write_external_results(
        output_dir=str(tmp_path),
        benchmark="textvqa",
        method="fastv",
        accuracy=0.71,
        retain=288,
        num_visual=576,
        llm_layers=32,
        llm_hidden=4096,
        num_samples=500,
        metric="vqa_accuracy",
        seed=0,
        repo_path="/repo",
        log_path="/tmp/x.log",
        parse_status="ok",
    )
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    assert data["benchmark"] == "textvqa"
    assert data["scoring"] == "fastv"
    assert data["prune_ratio"] == pytest.approx(0.5)
    assert data["avg_kept_tokens"] == 288
    assert data["num_visual_tokens"] == 576
    assert data["accuracy"] == pytest.approx(0.71)
    assert 0.4 < data["flops_reduction_ratio"] < 0.6  # ~50% token cut at 576->288
    assert data["external"]["parse_status"] == "ok"
    assert data["external"]["retain_tokens"] == 288
    # schema compatibility with scripts/03 + 07 fields
    for key in ("benchmark", "model", "scoring", "prune_ratio", "num_samples", "metric",
                "accuracy", "num_visual_tokens", "avg_kept_tokens", "flops_reduction_ratio",
                "latency_mean_ms", "seed"):
        assert key in data


def test_write_external_results_snapped_budget_records_actual(tmp_path):
    # SparseVLM snapped 0.5 -> retain 192: JSON must record the ACTUAL budget.
    path = write_external_results(
        output_dir=str(tmp_path),
        benchmark="gqa",
        method="sparsevlm",
        accuracy=None,
        retain=192,
        num_visual=576,
        llm_layers=32,
        llm_hidden=4096,
        num_samples=None,
        metric="exact_match",
        seed=0,
        repo_path="/repo",
        log_path=None,
        parse_status="manual",
        extra={"retain_requested_exact": False},
    )
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    # JSON stores prune_ratio rounded to 4 decimals.
    assert data["prune_ratio"] == pytest.approx(1 - 192 / 576, abs=1e-4)
    assert data["accuracy"] is None
    assert data["external"]["retain_requested_exact"] is False
