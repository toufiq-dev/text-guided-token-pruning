"""External head-to-head baseline runners (FastV, SparseVLM).

NOT reimplementations. These are thin, auditable orchestration helpers that drive
the *official public code* of the two mandatory head-to-head baselines from the
proposal (§5, pre-registration P2–P3) and convert their outputs into the same
results-JSON schema as scripts/03 and scripts/07, so scripts/04_evaluate.py can
aggregate everything on a matched-budget basis.

Verified integration facts (Aug 2026, from the official repos):

* **FastV** — github.com/pkunlp-icler/FastV (ECCV 2024, arXiv:2403.06764).
  LLaVA-only (llava-hf/llava-1.5-7b-hf / 13b). Prunes inside the decoder after
  layer ``K`` ("fastv_k"), keeping ``R`` of the 576 image tokens
  ("fastv_r" / ``--fast-v-attention-rank``).
    - Runs from source and needs a *patched* transformers (editable install under
      ``./src/FastV/llava-hf/transformers``) — do NOT reuse the tgtp venv.
    - Entry::

          python src/FastV/inference/eval/inference_<bench>.py \\
              --use-fast-v --fast-v-agg-layer K --fast-v-attention-rank R \\
              --fast-v-image-token-length 576 [--fast-v-inplace]

    - Verified eval scripts: ``inference_ocrvqa.py`` (TextVQA), ``inference_aokvqa.py``
      (A-OKVQA). Other benchmarks: pass ``--entry-script`` in the runner.

* **SparseVLM** — github.com/Gumpest/SparseVLMs (ICML 2025, arXiv:2410.04417).
  LLaVA-1.5. Sparsifies progressively across decoder layers (in-decoder).
    - Install: ``pip install -e .`` + ``transformers==4.37.0`` +
      ``flash_attn==2.3.3`` (own conda env, python 3.10).
    - Driven by the ``RETAIN_TOKN`` env var; natively supports only
      ``{192, 128, 96, 64}`` retained tokens. Entry::

          RETAIN_TOKN=<n> [USE_VERSION=2_0] bash scripts/v1_5/eval/<bench>.sh

    - Verified: ``textvqa.sh``, ``gqa.sh``, ``sqa.sh``, ``mmbench.sh``, ``mme.sh``.
      VQAv2: pass ``--eval-script`` if present in your checkout.

**Matched-budget honesty.** Both baselines operate on LLaVA-1.5-7B (576 visual
tokens). FastV supports any retained count (144/288/432 for keep 0.25/0.50/0.75),
so budgets match our 25/50/75% protocol exactly. SparseVLM natively supports only
{192, 128, 96, 64} retained tokens — the requested budget may be unattainable.
``sparsevlm_retain_for_keep_ratio`` therefore returns the *nearest supported*
count plus an ``exact`` flag; the runner records the *actual* kept-token count and
computed prune_ratio in the results JSON so comparisons stay at matched budgets.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from typing import Any

from tgtp.evaluation.efficiency_metrics import estimate_flops_reduction
from tgtp.utils.logging import get_logger

logger = get_logger("tgtp.external_baselines")

# --------------------------------------------------------------------------- #
# Verified repo / model facts
# --------------------------------------------------------------------------- #
FASTV_REPO_URL = "https://github.com/pkunlp-icler/FastV"
SPARSEVLM_REPO_URL = "https://github.com/Gumpest/SparseVLMs"

# Head-to-head runs on LLaVA-1.5-7B (both baselines are LLaVA-only).
LLAVA15_NUM_VISUAL = 576
LLAVA15_LLM_LAYERS = 32
LLAVA15_LLM_HIDDEN = 4096

FASTV_IMAGE_TOKEN_LENGTH = LLAVA15_NUM_VISUAL
FASTV_AGG_LAYER = 2  # FastV default: prune after layer 2 (paper + CLI default)

# SparseVLM's only natively supported retained-token counts (from its README).
SPARSEVLM_SUPPORTED_RETAIN = (192, 128, 96, 64)

# FastV benchmark -> entry script (relative to the FastV repo root).
FASTV_ENTRY_SCRIPTS: dict[str, str] = {
    "textvqa": "src/FastV/inference/eval/inference_ocrvqa.py",
    "aokvqa": "src/FastV/inference/eval/inference_aokvqa.py",
}

# SparseVLM benchmark -> shell script (relative to the SparseVLMs repo root).
SPARSEVLM_EVAL_SCRIPTS: dict[str, str] = {
    "vqav2": "scripts/v1_5/eval/vqav2.sh",  # ⚠️ verify presence in your checkout
    "gqa": "scripts/v1_5/eval/gqa.sh",
    "textvqa": "scripts/v1_5/eval/textvqa.sh",
    "mmbench": "scripts/v1_5/eval/mmbench.sh",
    "sqa": "scripts/v1_5/eval/sqa.sh",
    "mme": "scripts/v1_5/eval/mme.sh",
}

# --------------------------------------------------------------------------- #
# Token-budget mapping
# --------------------------------------------------------------------------- #
def fastv_rank_for_keep_ratio(
    keep_ratio: float, num_visual: int = LLAVA15_NUM_VISUAL
) -> int:
    """Retained-token rank for FastV at a given keep ratio (exact)."""
    if not 0.0 < keep_ratio <= 1.0:
        raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")
    rank = int(round(num_visual * keep_ratio))
    return max(1, min(rank, num_visual))


def sparsevlm_retain_for_keep_ratio(
    keep_ratio: float,
    num_visual: int = LLAVA15_NUM_VISUAL,
    supported: tuple[int, ...] = SPARSEVLM_SUPPORTED_RETAIN,
) -> tuple[int, bool]:
    """Nearest SparseVLM-supported retained count for a keep ratio.

    Returns:
        (retain, exact): the nearest natively supported count and whether the
        requested budget is exactly attainable. When not exact, comparisons
        must use the *actual* kept count (recorded in the results JSON).
    """
    target = max(1, int(round(num_visual * keep_ratio)))
    if target in supported:
        return target, True
    nearest = min(supported, key=lambda r: abs(r - target))
    return nearest, False


def keep_ratio_from_retain(retain: int, num_visual: int = LLAVA15_NUM_VISUAL) -> float:
    """Actual prune-ratio implied by a retained count: 1 - retain/num_visual."""
    return 1.0 - retain / num_visual


# --------------------------------------------------------------------------- #
# Command / environment construction (builders are pure and unit-testable)
# --------------------------------------------------------------------------- #
def build_fastv_command(
    repo_path: str,
    entry_script: str,
    rank: int,
    agg_layer: int = FASTV_AGG_LAYER,
    image_token_length: int = FASTV_IMAGE_TOKEN_LENGTH,
    inplace: bool = False,
    extra_args: list[str] | None = None,
) -> list[str]:
    """FastV arguments as argv (INTERPRETER-FREE; the runner prepends its own python).

    The first element is the entry script path. The runner combines these with the
    FastV-env interpreter: ``[args.python, *build_fastv_command(...)]``.
    """
    cmd = [
        os.path.join(repo_path, entry_script),
        "--use-fast-v",
        "--fast-v-agg-layer",
        str(agg_layer),
        "--fast-v-attention-rank",
        str(rank),
        "--fast-v-image-token-length",
        str(image_token_length),
    ]
    if inplace:
        cmd.append("--fast-v-inplace")
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def build_sparsevlm_env(
    retain_tokn: int,
    use_version: str | None = None,
    base_env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Environment for the SparseVLM LLaVA-1.5 eval shell scripts."""
    env = dict(os.environ if base_env is None else base_env)
    env["RETAIN_TOKN"] = str(retain_tokn)
    if use_version:
        env["USE_VERSION"] = use_version
    return env


def build_sparsevlm_command(repo_path: str, eval_script: str) -> list[str]:
    """SparseVLM command: bash scripts/v1_5/eval/<bench>.sh (env carries RETAIN_TOKN)."""
    return ["bash", os.path.join(repo_path, eval_script)]


# --------------------------------------------------------------------------- #
# Execution + parsing
# --------------------------------------------------------------------------- #
def run_capture(
    argv: list[str],
    env: dict[str, str] | None = None,
    timeout: int = 3600,
) -> tuple[int, str]:
    """Run a command, capture combined stdout+stderr as text.

    Returns:
        (returncode, output). Raises TimeoutExpired on timeout.
    """
    logger.info("Running: %s", " ".join(argv))
    proc = subprocess.run(
        argv,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    output = proc.stdout or ""
    if proc.stderr:
        output += "\n[stderr]\n" + proc.stderr
    return proc.returncode, output


_ACC_PATTERNS = [
    # Explicit percentage (with % sign).
    re.compile(r"(?:accuracy|acc)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*%", re.IGNORECASE),
    # Labelled forms: 'accuracy: 71.2', 'overall accuracy = 0.71', 'Acc: 0.501'.
    re.compile(r"(?:overall|final|total)?\s*(?:accuracy|acc)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE),
]


def parse_accuracy_from_log(text: str) -> float | None:
    """Best-effort accuracy extraction from an external harness log.

    Accepts percentages (with or without ``%``) and 0–1 fractions: any bare value
    > 1.0 is treated as a percentage and divided by 100 (LLaVA-Evaluation prints
    e.g. ``Accuracy: 71.2`` with no % sign). Returns None when nothing looks like
    an accuracy line (the runner then flags ``parse_status: manual``).
    """
    for pat in _ACC_PATTERNS:
        for m in pat.finditer(text):
            value = float(m.group(1))
            if value > 1.0:
                value /= 100.0
            if 0.0 <= value <= 1.0:
                return value
    return None


# --------------------------------------------------------------------------- #
# Results-JSON (schema-compatible with scripts/03 + 07)
# --------------------------------------------------------------------------- #
def write_external_results(
    output_dir: str,
    benchmark: str,
    method: str,  # "fastv" | "sparsevlm"
    accuracy: float | None,
    retain: int,
    num_visual: int,
    llm_layers: int,
    llm_hidden: int,
    num_samples: int | None,
    metric: str,
    seed: int,
    repo_path: str,
    log_path: str | None,
    parse_status: str = "ok",
    latency_mean_ms: float = 0.0,
    extra: dict[str, Any] | None = None,
) -> str:
    """Write an external-baseline result JSON and return its path."""
    prune_ratio = keep_ratio_from_retain(retain, num_visual)
    flops = estimate_flops_reduction(num_visual, retain, llm_layers, llm_hidden)

    results: dict[str, Any] = {
        "benchmark": benchmark,
        "model": "llava-hf/llava-1.5-7b-hf",
        "scoring": method,
        "prune_ratio": round(prune_ratio, 4),
        "merge_strategy": "external",
        "num_samples": num_samples if num_samples is not None else 0,
        "metric": metric,
        "accuracy": accuracy,
        "num_visual_tokens": num_visual,
        "avg_kept_tokens": retain,
        "flops_reduction_ratio": round(flops["flops_reduction_ratio"], 4),
        "latency_mean_ms": latency_mean_ms,
        "seed": seed,
        "external": {
            "method": method,
            "repo_path": repo_path,
            "log_path": log_path,
            "parse_status": parse_status,
            "retain_tokens": retain,
        },
    }
    if extra:
        results["external"].update(extra)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{benchmark}_{method}_{prune_ratio:g}_{seed}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    logger.info("Wrote %s (acc=%s, prune_ratio=%.2f)", path, accuracy, prune_ratio)
    return path
