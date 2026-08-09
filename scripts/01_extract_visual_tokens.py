#!/usr/bin/env python3
"""01_extract_visual_tokens.py — cache pre-decoder visual tokens.

Runs the vision encoder + connector ONLY (never the LLM decoder) and saves the
resulting (1, N, D) tensors to ``cached_features/<dataset>/`` as .npy files.

WHY THIS IS CORRECT: the cached tensors are the *pre-decoder* visual tokens
(connector output) — exactly the representation the TextGuidedPruner operates
on. Caching once makes every later experiment (training, inference, ablations)
independent of Colab disconnects and repeated ViT forward passes.

Usage:
    python scripts/01_extract_visual_tokens.py --model HuggingFaceTB/SmolVLM-Instruct \
        --dataset vqav2 --split validation --max-samples 100 --output-dir cached_features
"""

from __future__ import annotations

import utils.transport
import utils.session
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tgtp.data.vqa_loader import (
    BENCHMARK_CONFIGS,
    cache_visual_tokens,
    load_vqa_dataset,
)
from tgtp.models.smolvlm_wrapper import SmolVLMPruningWrapper
from tgtp.utils.logging import get_logger

logger = get_logger("tgtp.extract")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cache pre-decoder visual tokens (ViT + connector only).")
    p.add_argument("--model", default="HuggingFaceTB/SmolVLM-Instruct")
    p.add_argument("--dataset", choices=sorted(BENCHMARK_CONFIGS), default="vqav2")
    p.add_argument("--split", default=None, help="override the default split for the benchmark")
    p.add_argument("--max-samples", type=int, default=100)
    p.add_argument("--output-dir", default="cached_features")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="auto", choices=["auto", "float16", "float32"])
    p.add_argument("--local-jsonl", default=None, help="optional local JSONL fallback dataset")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    from transformers import AutoModelForImageTextToText

    logger.info("Loading %s (vision encoder + connector only are used) ...", args.model)
    dtype = torch.float16 if args.dtype in {"auto", "float16"} and args.device == "cuda" else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(args.model, torch_dtype=dtype).to(device)
    model.eval()
    wrapper = SmolVLMPruningWrapper(model)

    samples = load_vqa_dataset(args.dataset, split=args.split, max_samples=args.max_samples,
                               local_jsonl=args.local_jsonl)
    logger.info("Loaded %d samples from %s", len(samples), args.dataset)

    out_dir = os.path.join(args.output_dir, args.dataset)
    count, num_patches = cache_visual_tokens(
        wrapper, samples, out_dir, device, batch_size=args.batch_size, model_name=args.model
    )
    logger.info("Cached %d samples (%d patches each) -> %s", count, num_patches, out_dir)
    logger.info("Next: python scripts/02_train_pruning_module.py --cache-dir %s", out_dir)


if __name__ == "__main__":
    main()
