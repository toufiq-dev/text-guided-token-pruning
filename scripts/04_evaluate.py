#!/usr/bin/env python3
"""04_evaluate.py — aggregate per-config result JSONs into a comparison report.

Produces a markdown table: full tokens vs. random vs. text-guided at each ratio,
plus the per-ratio delta vs the full-token baseline (the thesis headline table).

Usage:
    python scripts/04_evaluate.py --results-dir outputs/results --report outputs/report.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tgtp.utils.logging import get_logger

logger = get_logger("tgtp.eval")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate inference results into a report.")
    p.add_argument("--results-dir", default="outputs/results")
    p.add_argument("--report", default="outputs/report.md")
    p.add_argument("--accuracy-budget", type=float, default=2.0,
                   help="pp accuracy budget vs full tokens (thesis success criterion)")
    return p.parse_args()


def load_results(results_dir: str) -> list:
    rows = []
    if not os.path.isdir(results_dir):
        return rows
    for fname in sorted(os.listdir(results_dir)):
        if fname.endswith(".json"):
            with open(os.path.join(results_dir, fname), "r", encoding="utf-8") as fh:
                rows.append(json.load(fh))
    return rows


def main() -> None:
    args = parse_args()
    rows = load_results(args.results_dir)
    if not rows:
        logger.error("No results found in %s", args.results_dir)
        sys.exit(1)

    # Group by benchmark; within each benchmark group by scoring method + ratio.
    by_bench: dict = {}
    for r in rows:
        by_bench.setdefault(r["benchmark"], []).append(r)

    lines = ["# Evaluation report", ""]
    for benchmark, results in sorted(by_bench.items()):
        full = [
            r for r in results
            if r.get("prune_ratio", 1.0) == 0.0 and r.get("scoring", "") in {"none", "full", "cross_attention"}
        ]
        full_acc = np_mean([r["accuracy"] for r in full]) if full else None
        lines.append(f"## {benchmark}")
        lines.append("")
        lines.append("| Method | Ratio | Accuracy | Δ vs full (pp) | FLOPs ↓ | Latency (ms) | Within budget? |")
        lines.append("|---|---|---|---|---|---|---|")

        for r in sorted(results, key=lambda x: (x["scoring"], x["prune_ratio"])):
            acc = r["accuracy"]
            delta = (acc - full_acc) if full_acc is not None else float("nan")
            within = "✅" if (full_acc is not None and delta >= -args.accuracy_budget) else "❌"
            lines.append(
                f"| {r['scoring']} | {r['prune_ratio']:.2f} | {acc:.4f} | {delta:+.2f} | "
                f"{r['flops_reduction_ratio'] * 100:.1f}% | {r['latency_mean_ms']:.1f} | {within} |"
            )
        lines.append("")

    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    logger.info("Wrote report -> %s", args.report)
    print("\n".join(lines))


def np_mean(vals):
    if not vals:
        return None
    return sum(vals) / len(vals)


if __name__ == "__main__":
    main()
