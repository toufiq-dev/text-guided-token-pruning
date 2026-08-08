#!/usr/bin/env python3
"""07_random_pruning_baseline.py — the empirically validated random baseline.

EMPIRICAL FINDING (this thesis, replicated on SmolVLM-Instruct):

    Random pruning of 50% of visual tokens produces correct answers.
    This is a surprisingly strong baseline that ALL pruning methods MUST
    be compared against (FastV et al. compare against random too, and show
    it degrades at high ratios — we verify this ourselves).

    Sanity check (3 questions, 1 image):
      - Baseline (100% tokens)    : all correct
      - Random (50% tokens)       : all correct
      - "Smart" at WRONG location : all gibberish  (post-decoder = bug)

This script runs the random baseline at --prune-ratios on a benchmark and
writes per-ratio accuracy JSONs (input to scripts/04_evaluate.py).

Usage:
    python scripts/07_random_pruning_baseline.py --model HuggingFaceTB/SmolVLM-Instruct \
        --prune-ratios 0.25 0.5 0.75 --benchmark vqav2 --max-samples 50 \
        --output-dir outputs/results
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tgtp.data.vqa_loader import BENCHMARK_CONFIGS, load_vqa_dataset
from tgtp.evaluation.efficiency_metrics import estimate_flops_reduction
from tgtp.evaluation.vqa_metrics import compute_vqa_scores
from tgtp.models.smolvlm_wrapper import SmolVLMPruningWrapper
from tgtp.utils.logging import get_logger

logger = get_logger("tgtp.random_baseline")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Random visual-token pruning baseline.")
    p.add_argument("--model", default="HuggingFaceTB/SmolVLM-Instruct")
    p.add_argument("--benchmark", choices=sorted(BENCHMARK_CONFIGS), default="vqav2")
    p.add_argument("--split", default=None)
    p.add_argument("--prune-ratios", nargs="+", type=float, default=[0.25, 0.5, 0.75])
    p.add_argument("--max-samples", type=int, default=50)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="outputs/results")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(args.model, torch_dtype=dtype).to(device)
    model.eval()
    wrapper = SmolVLMPruningWrapper(model)
    text_cfg = getattr(model.config, "text_config", model.config)
    llm_layers = getattr(text_cfg, "num_hidden_layers", 24)
    llm_hidden = getattr(text_cfg, "hidden_size", 2048)

    samples = load_vqa_dataset(args.benchmark, split=args.split, max_samples=args.max_samples)
    metric = BENCHMARK_CONFIGS[args.benchmark]["metric"]

    for ratio in args.prune_ratios:
        keep_ratio = 1.0 - ratio
        accs = []
        keep_counts = []
        for sample in samples:
            pv = processor(images=sample["image"], return_tensors="pt")["pixel_values"].to(device)
            enc = processor(text=sample["question"], return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            attn_mask = enc["attention_mask"].to(device)

            with torch.no_grad():
                generated_ids, keep_idx, _ = wrapper.generate_with_pruning(
                    pv, input_ids, attn_mask,
                    keep_ratio=keep_ratio,
                    scoring_method="random",
                    max_new_tokens=args.max_new_tokens,
                )
            pred = processor.decode(generated_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
            refs = sample["answers"] if isinstance(sample["answers"], list) else [sample["answers"]]
            accs.append(float(compute_vqa_scores([pred], [[str(r) for r in refs]], metric)[0]))
            keep_counts.append(int(keep_idx.numel()))

        acc = float(np.mean(accs))
        avg_kept = float(np.mean(keep_counts))
        # Total visual tokens = kept / keep_ratio (per sample); independent of ratio.
        num_visual = int(round(keep_counts[0] / keep_ratio)) if keep_counts else 0
        flops = estimate_flops_reduction(num_visual, int(avg_kept), llm_layers, llm_hidden)
        logger.info("random ratio=%.2f keep=%.2f%% acc=%.4f flops_saved=%.1f%%",
                    ratio, (1 - ratio) * 100, acc, flops["flops_reduction_ratio"] * 100)

        os.makedirs(args.output_dir, exist_ok=True)
        result = {
            "benchmark": args.benchmark,
            "model": args.model,
            "scoring": "random",
            "prune_ratio": ratio,
            "merge_strategy": "drop",
            "num_samples": len(accs),
            "metric": metric,
            "accuracy": acc,
            "num_visual_tokens": num_visual,
            "avg_kept_tokens": avg_kept,
            "flops_reduction_ratio": flops["flops_reduction_ratio"],
            "latency_mean_ms": 0.0,
            "seed": args.seed,
        }
        out = os.path.join(args.output_dir, f"{args.benchmark}_random_{ratio:g}_{args.seed}.json")
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)


if __name__ == "__main__":
    main()
