#!/usr/bin/env python3
"""09_run_sparsevlm_baseline.py — SparseVLM (ICML 2025) head-to-head baseline.

Wraps the OFFICIAL SparseVLM public code (github.com/Gumpest/SparseVLMs) so the
weeks 3–4 mandatory head-to-head comparison (proposal §5, pre-registration P3)
has a concrete entry point. Orchestrator only: it configures the repo's own
LLaVA-1.5 eval shell scripts via the RETAIN_TOKN env var, runs them, parses the
accuracy from the harness output, and writes a results JSON in the same schema
as scripts/03 + 07.

IMPORTANT — environment:
    SparseVLM needs its own conda env (python 3.10): pip install -e .,
    transformers==4.37.0, flash_attn==2.3.3 (see the repo README). The eval
    shell scripts must be run from that env; activate it before running this
    script (the subprocess inherits the current environment).

    SparseVLM is LLaVA-1.5-only and natively supports only RETAIN_TOKN in
    {192, 128, 96, 64}. For a requested keep ratio that is not exactly
    attainable, the runner uses the NEAREST supported retained count and records
    the ACTUAL prune_ratio + kept tokens in the JSON so comparisons stay at
    matched budgets (the pre-registration requirement).

Usage:
    # inside the SparseVLMs conda env:
    python scripts/09_run_sparsevlm_baseline.py \\
        --repo-path third_party/SparseVLMs \\
        --benchmark textvqa --keep-ratio 0.33 --output-dir outputs/results

    python scripts/09_run_sparsevlm_baseline.py --clone \\
        --repo-path third_party/SparseVLMs --benchmark gqa --keep-ratio 0.33 --dry-run
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tgtp.data.vqa_loader import BENCHMARK_CONFIGS
from tgtp.evaluation.external_baselines import (
    LLAVA15_LLM_HIDDEN,
    LLAVA15_LLM_LAYERS,
    LLAVA15_NUM_VISUAL,
    SPARSEVLM_EVAL_SCRIPTS,
    SPARSEVLM_REPO_URL,
    SPARSEVLM_SUPPORTED_RETAIN,
    build_sparsevlm_command,
    build_sparsevlm_env,
    parse_accuracy_from_log,
    run_capture,
    sparsevlm_retain_for_keep_ratio,
    write_external_results,
)
from tgtp.utils.logging import get_logger

logger = get_logger("tgtp.sparsevlm_baseline")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SparseVLM (ICML 2025) head-to-head baseline runner.")
    p.add_argument("--repo-path", default=None, help="Path to the SparseVLMs checkout.")
    p.add_argument("--clone", action="store_true", help="git clone SparseVLMs into --repo-path if absent.")
    p.add_argument("--benchmark", choices=sorted(SPARSEVLM_EVAL_SCRIPTS), default="textvqa")
    p.add_argument("--eval-script", default=None, help="Override the eval shell script (repo-relative).")
    p.add_argument("--keep-ratio", type=float, default=1 / 3, help="Requested keep ratio; snapped to nearest RETAIN_TOKN.")
    p.add_argument("--retain-tokens", type=int, default=None, help="Exact RETAIN_TOKN override (192/128/96/64).")
    p.add_argument("--use-version", default=None, help="SparseVLM+ mode: pass '2_0' to enable.")
    p.add_argument("--num-samples", type=int, default=None, help="Override the sample count recorded in the JSON.")
    p.add_argument("--metric", default=None, help="Metric override (default from benchmark config).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timeout", type=int, default=7200)
    p.add_argument("--output-dir", default="outputs/results")
    p.add_argument("--accuracy", type=float, default=None, help="Manually recorded accuracy (parse_status='manual').")
    p.add_argument("--dry-run", action="store_true", help="Print the command/env without executing.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.repo_path is None:
        raise SystemExit("--repo-path is required (or use --clone --repo-path <dir>).")

    if args.clone and not os.path.isdir(args.repo_path):
        logger.info("Cloning SparseVLMs into %s", args.repo_path)
        os.makedirs(args.repo_path, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", SPARSEVLM_REPO_URL, args.repo_path], check=True)

    if not os.path.isdir(os.path.join(args.repo_path, "scripts", "v1_5", "eval")):
        raise SystemExit(
            f"{args.repo_path} does not look like the SparseVLMs repo (missing "
            "scripts/v1_5/eval). Clone it with --clone or point --repo-path at the checkout."
        )

    if args.retain_tokens is not None:
        retain, exact = args.retain_tokens, True
        if retain not in SPARSEVLM_SUPPORTED_RETAIN:
            logger.warning(
                "RETAIN_TOKN=%d is not in SparseVLM's natively supported set %s; "
                "the harness may reject it.", retain, SPARSEVLM_SUPPORTED_RETAIN
            )
            exact = False
    else:
        retain, exact = sparsevlm_retain_for_keep_ratio(args.keep_ratio, LLAVA15_NUM_VISUAL)
        if args.keep_ratio > 1 / 3:
            logger.warning(
                "keep_ratio=%.2f exceeds SparseVLM's maximum attainable budget "
                "(192/576 = 1/3); it will be snapped down to retain=%d "
                "(prune_ratio=%.3f).", args.keep_ratio, retain, 1 - retain / LLAVA15_NUM_VISUAL
            )

    actual_prune = 1 - retain / LLAVA15_NUM_VISUAL
    if not exact:
        logger.warning(
            "Requested keep_ratio=%.2f is not exactly attainable by SparseVLM "
            "(supported RETAIN_TOKN: %s). Using nearest retain=%d -> prune_ratio=%.3f. "
            "Results JSON records the ACTUAL budget so comparisons stay matched.",
            args.keep_ratio, SPARSEVLM_SUPPORTED_RETAIN, retain, actual_prune,
        )

    eval_script = args.eval_script or SPARSEVLM_EVAL_SCRIPTS[args.benchmark]
    script_path = os.path.join(args.repo_path, eval_script)
    if not os.path.exists(script_path):
        raise SystemExit(
            f"Eval script not found: {script_path}\n"
            "SparseVLM ships scripts/v1_5/eval/{textvqa,gqa,sqa,mmbench,mme}.sh; for "
            "other benchmarks pass --eval-script with a repo-relative path."
        )

    env = build_sparsevlm_env(retain, use_version=args.use_version)
    argv = build_sparsevlm_command(args.repo_path, eval_script)
    logger.info("RETAIN_TOKN=%d (%s) -> %s", retain, "exact" if exact else "snapped", " ".join(argv))

    if args.dry_run:
        print(f"RETAIN_TOKN={env['RETAIN_TOKN']}" + (f" USE_VERSION={env['USE_VERSION']}" if args.use_version else "") + " " + " ".join(argv))
        return

    returncode, output = run_capture(argv, env=env, timeout=args.timeout)
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, f"{args.benchmark}_sparsevlm_{actual_prune:g}_{args.seed}.log")
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write(output)

    accuracy = args.accuracy if args.accuracy is not None else parse_accuracy_from_log(output)
    parse_status = "ok" if accuracy is not None else "manual"
    if accuracy is None:
        logger.warning(
            "Could not auto-parse accuracy for %s (rc=%d). The LLaVA-Evaluation harness "
            "prints per-answer files; read the log and re-run with --accuracy <float> to "
            "record it. Log: %s", args.benchmark, returncode, log_path
        )
    else:
        logger.info("%s SparseVLM accuracy (retain %d): %.4f", args.benchmark, retain, accuracy)

    bcfg = BENCHMARK_CONFIGS.get(args.benchmark)
    metric = args.metric or (bcfg["metric"] if bcfg else "exact_match")
    write_external_results(
        output_dir=args.output_dir,
        benchmark=args.benchmark,
        method="sparsevlm",
        accuracy=accuracy,
        retain=retain,
        num_visual=LLAVA15_NUM_VISUAL,
        llm_layers=LLAVA15_LLM_LAYERS,
        llm_hidden=LLAVA15_LLM_HIDDEN,
        num_samples=args.num_samples,
        metric=metric,
        seed=args.seed,
        repo_path=args.repo_path,
        log_path=log_path,
        parse_status=parse_status,
        latency_mean_ms=0.0,
        extra={
            "retain_requested_exact": exact,
            "use_version": args.use_version,
            "returncode": returncode,
            "eval_script": eval_script,
        },
    )

    if returncode != 0:
        logger.warning("SparseVLM eval exited with code %d (see log).", returncode)


if __name__ == "__main__":
    main()
