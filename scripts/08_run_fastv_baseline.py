#!/usr/bin/env python3
"""08_run_fastv_baseline.py — FastV (ECCV 2024) head-to-head baseline.

Wraps the OFFICIAL FastV public code (github.com/pkunlp-icler/FastV) so the
weeks 3–4 mandatory head-to-head comparison (proposal §5, pre-registration P2)
has a concrete entry point. This is an *orchestrator*, not a reimplementation:
it clones/configures the repo, runs its own inference script at a matched token
budget, parses the accuracy from its log, and writes a results JSON in the same
schema as scripts/03 + 07 so scripts/04_evaluate.py aggregates everything.

IMPORTANT — environment:
    FastV is LLaVA-only and requires a *patched* transformers (editable install
    under ./src/FastV/llava-hf/transformers). It must NOT run in the tgtp venv.
    Use --clone to fetch the repo, then set up FastV's own environment following
    its README/setup.sh, and point --python at that environment's interpreter.

    The baseline runs on LLaVA-1.5-7B (576 visual tokens) — the secondary model
    of the evaluation protocol — because FastV does not support SmolVLM.

Usage:
    python scripts/08_run_fastv_baseline.py \\
        --repo-path third_party/FastV \\
        --python /path/to/fastv_env/bin/python \\
        --benchmark textvqa --keep-ratio 0.5 --output-dir outputs/results

    # first time only (fetch + inspect):
    python scripts/08_run_fastv_baseline.py --clone --repo-path third_party/FastV \\
        --benchmark textvqa --keep-ratio 0.5 --dry-run
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tgtp.data.vqa_loader import BENCHMARK_CONFIGS
from tgtp.evaluation.external_baselines import (
    FASTV_ENTRY_SCRIPTS,
    FASTV_REPO_URL,
    LLAVA15_LLM_HIDDEN,
    LLAVA15_LLM_LAYERS,
    LLAVA15_NUM_VISUAL,
    build_fastv_command,
    fastv_rank_for_keep_ratio,
    parse_accuracy_from_log,
    run_capture,
    write_external_results,
)
from tgtp.utils.logging import get_logger

logger = get_logger("tgtp.fastv_baseline")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FastV (ECCV 2024) head-to-head baseline runner.")
    p.add_argument("--repo-path", default=None, help="Path to the FastV checkout.")
    p.add_argument("--clone", action="store_true", help="git clone FastV into --repo-path if absent.")
    p.add_argument("--python", default="python", help="Interpreter of the FastV env (patched transformers).")
    p.add_argument("--benchmark", choices=sorted(FASTV_ENTRY_SCRIPTS), default="textvqa")
    p.add_argument("--entry-script", default=None, help="Override the eval entry script (repo-relative).")
    p.add_argument("--keep-ratio", type=float, default=0.5, help="Token keep ratio (0.25/0.50/0.75 → 144/288/432 of 576).")
    p.add_argument("--agg-layer", type=int, default=2, help="FastV pruning layer (paper default: after layer 2).")
    p.add_argument("--inplace", action="store_true", help="Pass --fast-v-inplace (real in-place acceleration).")
    p.add_argument("--extra-args", nargs="*", default=None, help="Extra CLI args forwarded to FastV's script.")
    p.add_argument("--num-samples", type=int, default=None,
                   help="Sample count to RECORD in the JSON (metadata only — the external "
                        "FastV script runs its own full question file and is not limited by this).")
    p.add_argument("--metric", default=None, help="Metric override (default from benchmark config).")
    p.add_argument("--accuracy", type=float, default=None, help="Manually recorded accuracy (parse_status='manual').")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timeout", type=int, default=3600)
    p.add_argument("--output-dir", default="outputs/results")
    p.add_argument("--dry-run", action="store_true", help="Print the command without executing.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.repo_path is None:
        raise SystemExit("--repo-path is required (or use --clone --repo-path <dir>).")

    if args.clone and not os.path.isdir(args.repo_path):
        logger.info("Cloning FastV into %s", args.repo_path)
        os.makedirs(args.repo_path, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", FASTV_REPO_URL, args.repo_path], check=True)

    if not os.path.isdir(os.path.join(args.repo_path, "src", "FastV")):
        raise SystemExit(
            f"{args.repo_path} does not look like the FastV repo (missing src/FastV). "
            "Clone it with --clone or point --repo-path at the checkout."
        )

    rank = fastv_rank_for_keep_ratio(args.keep_ratio, LLAVA15_NUM_VISUAL)
    if args.num_samples is not None:
        logger.info("Note: --num-samples=%d is recorded metadata only; FastV's own script "
                    "evaluates its full question file.", args.num_samples)
    entry = args.entry_script or FASTV_ENTRY_SCRIPTS[args.benchmark]
    entry_path = os.path.join(args.repo_path, entry)
    if not os.path.exists(entry_path):
        raise SystemExit(
            f"Entry script not found: {entry_path}\n"
            f"FastV ships eval scripts for textvqa/aokvqa out of the box; for other "
            "benchmarks pass --entry-script with a repo-relative path to your script."
        )

    argv = [args.python, *build_fastv_command(
        args.repo_path, entry, rank, args.agg_layer, inplace=args.inplace, extra_args=args.extra_args
    )]
    logger.info("keep_ratio=%.2f -> retain %d of %d visual tokens (prune %.1f%%)",
                args.keep_ratio, rank, LLAVA15_NUM_VISUAL, (1 - args.keep_ratio) * 100)

    if args.dry_run:
        print(" ".join(argv))
        return

    returncode, output = run_capture(argv, timeout=args.timeout)
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, f"{args.benchmark}_fastv_{args.keep_ratio:g}_{args.seed}.log")
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write(output)

    accuracy = args.accuracy if args.accuracy is not None else parse_accuracy_from_log(output)
    parse_status = "ok" if accuracy is not None else "manual"
    if accuracy is None:
        logger.warning(
            "Could not auto-parse accuracy for %s (rc=%d). Inspect the log and, if the "
            "harness printed accuracy elsewhere, record it manually via the --accuracy "
            "override (or edit the JSON). Log: %s", args.benchmark, returncode, log_path
        )
    else:
        logger.info("%s FastV accuracy (keep %.2f): %.4f", args.benchmark, args.keep_ratio, accuracy)

    bcfg = BENCHMARK_CONFIGS.get(args.benchmark)
    metric = args.metric or (bcfg["metric"] if bcfg else "exact_match")
    write_external_results(
        output_dir=args.output_dir,
        benchmark=args.benchmark,
        method="fastv",
        accuracy=accuracy,
        retain=rank,
        num_visual=LLAVA15_NUM_VISUAL,
        llm_layers=LLAVA15_LLM_LAYERS,
        llm_hidden=LLAVA15_LLM_HIDDEN,
        num_samples=args.num_samples,
        metric=metric,
        seed=args.seed,
        repo_path=args.repo_path,
        log_path=log_path,
        parse_status=parse_status,
        extra={"agg_layer": args.agg_layer, "inplace": args.inplace, "returncode": returncode},
    )

    if returncode != 0:
        logger.warning("FastV exited with code %d (see log).", returncode)


if __name__ == "__main__":
    main()
