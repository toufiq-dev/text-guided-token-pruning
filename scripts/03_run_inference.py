#!/usr/bin/env python3
"""03_run_inference.py — run VQA inference with a chosen pruning configuration.

Measures accuracy + efficiency (FLOPs estimate, latency) and writes a JSON
results file under ``--output-dir`` for scripts/04_evaluate.py to aggregate.

Scoring methods:
    random          — random baseline (mandatory in all experiments)
    cross_attention — text-conditioned scoring, no training (fast sanity)
    cosine          — text-conditioned cosine scoring (ablation)
    learned         — loads a trained TextGuidedPruner from --pruner-checkpoint

Usage:
    python scripts/03_run_inference.py --prune-ratio 0.5 --scoring learned \
        --pruner-checkpoint outputs/pruner_50pct/pruner.pt \
        --benchmark vqav2 --max-samples 50 --output-dir outputs/results
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tgtp.data.vqa_loader import BENCHMARK_CONFIGS, load_vqa_dataset
from tgtp.evaluation.efficiency_metrics import (
    estimate_flops_reduction,
    measure_latency,
)
from tgtp.evaluation.vqa_metrics import compute_vqa_scores
from tgtp.models.pruning_module import TextGuidedPruner
from tgtp.models.smolvlm_wrapper import SmolVLMPruningWrapper
from tgtp.utils.logging import get_logger

logger = get_logger("tgtp.infer")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run pruned VQA inference.")
    p.add_argument("--model", default="HuggingFaceTB/SmolVLM-Instruct")
    p.add_argument("--benchmark", choices=sorted(BENCHMARK_CONFIGS), default="vqav2")
    p.add_argument("--split", default=None)
    p.add_argument("--max-samples", type=int, default=50)
    p.add_argument("--prune-ratio", type=float, default=0.5)
    p.add_argument("--scoring", choices=["random", "cross_attention", "cosine", "learned"], default="cross_attention")
    p.add_argument("--pruner-checkpoint", default=None)
    p.add_argument("--merge-strategy", choices=["drop", "merge"], default="drop")
    p.add_argument("--llm-layers", type=int, default=None, help="override for FLOPs estimate")
    p.add_argument("--llm-hidden", type=int, default=None, help="override for FLOPs estimate")
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="outputs/results")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def greedy_decode_from_embeddings(wrapper: SmolVLMPruningWrapper, seq: torch.Tensor,
                                  max_new_tokens: int) -> torch.Tensor:
    """Greedy-argmax decode on an embedding sequence; returns generated token ids."""
    gen_ids = []
    for _ in range(max_new_tokens):
        out = wrapper.text_model(inputs_embeds=seq)
        nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        gen_ids.append(nxt)
        seq = torch.cat([seq, wrapper.text_model.get_input_embeddings()(nxt)], dim=1)
    return torch.cat(gen_ids, dim=1)  # (B, gen)


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
    llm_layers = args.llm_layers or getattr(text_cfg, "num_hidden_layers", 24)
    llm_hidden = args.llm_hidden or getattr(text_cfg, "hidden_size", 2048)
    probe = Image.new("RGB", (384, 384))
    num_visual = wrapper.extract_visual_tokens(
        processor(images=probe, return_tensors="pt")["pixel_values"].to(device)
    ).shape[1]

    pruner = None
    if args.scoring == "learned":
        assert args.pruner_checkpoint, "--pruner-checkpoint required for --scoring learned"
        cfg = {}
        cfg_path = os.path.join(os.path.dirname(args.pruner_checkpoint), "config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as fh:
                cfg = json.load(fh)
        pruner = TextGuidedPruner(
            visual_dim=cfg.get("visual_dim", llm_hidden),
            text_dim=cfg.get("text_dim", llm_hidden),
            hidden_dim=cfg.get("hidden_dim", 256),
            num_heads=cfg.get("num_heads", 8),
            keep_ratio=1.0 - args.prune_ratio,
            merge_strategy=cfg.get("merge_strategy", args.merge_strategy),
        ).to(device)
        pruner.load_state_dict(torch.load(args.pruner_checkpoint, map_location=device))
        pruner.eval()

    samples = load_vqa_dataset(args.benchmark, split=args.split, max_samples=args.max_samples)
    logger.info("Benchmark %s, %d samples, prune_ratio=%.2f scoring=%s",
                args.benchmark, len(samples), args.prune_ratio, args.scoring)

    predictions: list = []
    references: list = []
    keep_counts: list = []
    latencies: list = []
    for sample in samples:
        latency = None
        pv = processor(images=sample["image"], return_tensors="pt")["pixel_values"].to(device)
        enc = processor(text=sample["question"], return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc["attention_mask"].to(device)

        if args.scoring == "learned":
            with torch.no_grad():
                visual = wrapper.extract_visual_tokens(pv)
                text_embeds = wrapper.extract_text_tokens(input_ids)
                pruned, keep_idx, _ = pruner(visual, text_embeds, attn_mask, training=False)
                seq = torch.cat([pruned, text_embeds], dim=1)
                gen_ids = greedy_decode_from_embeddings(wrapper, seq, args.max_new_tokens)
            keep_counts.append(int(keep_idx.numel()))
            pred = processor.decode(gen_ids[0], skip_special_tokens=True).strip()
        else:
            def _gen(pv=pv, input_ids=input_ids, attn_mask=attn_mask) -> tuple:
                return wrapper.generate_with_pruning(
                    pv, input_ids, attn_mask,
                    keep_ratio=1.0 - args.prune_ratio,
                    scoring_method=args.scoring,
                    max_new_tokens=args.max_new_tokens,
                )

            latency = measure_latency(_gen, repeats=3, warmup=0)
            with torch.no_grad():
                generated_ids, keep_idx, _ = _gen()
            keep_counts.append(int(keep_idx.numel()))
            pred = processor.decode(generated_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()

        predictions.append(pred)
        refs = sample["answers"] if isinstance(sample["answers"], list) else [sample["answers"]]
        references.append([str(r) for r in refs])
        latencies.append(latency["latency_mean_ms"] if latency else 0.0)
        if len(predictions) % 10 == 0:
            logger.info("  %d/%d samples", len(predictions), len(samples))

    metric = BENCHMARK_CONFIGS[args.benchmark]["metric"]
    per_sample = compute_vqa_scores(predictions, references, metric)
    avg_kept = float(np.mean(keep_counts)) if keep_counts else 0.0
    flops = estimate_flops_reduction(num_visual, int(avg_kept), llm_layers, llm_hidden)

    results = {
        "benchmark": args.benchmark,
        "model": args.model,
        "scoring": args.scoring,
        "prune_ratio": args.prune_ratio,
        "merge_strategy": args.merge_strategy,
        "num_samples": len(predictions),
        "metric": metric,
        "accuracy": float(per_sample.mean()),
        "num_visual_tokens": int(num_visual),
        "avg_kept_tokens": avg_kept,
        "flops_reduction_ratio": flops["flops_reduction_ratio"],
        "latency_mean_ms": float(np.mean(latencies)) if latencies else 0.0,
        "seed": args.seed,
        "sample_predictions": predictions[:20],
        "sample_references": references[:20],
    }
    os.makedirs(args.output_dir, exist_ok=True)
    tag = f"{args.benchmark}_{args.scoring}_{args.prune_ratio:g}_{args.seed}"
    out_path = os.path.join(args.output_dir, f"{tag}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    logger.info("Accuracy %.4f | FLOPs reduction %.1f%% | latency %.1f ms -> %s",
                results["accuracy"], results["flops_reduction_ratio"] * 100,
                results["latency_mean_ms"], out_path)


if __name__ == "__main__":
    main()
