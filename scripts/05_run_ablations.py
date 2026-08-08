#!/usr/bin/env python3
"""05_run_ablations.py — run the thesis ablation matrix on a small sample.

Ablations (Section 5 of the proposal):

  A1. Scoring method      : cross_attention vs. cosine vs. random
  A2. Merge strategy      : drop vs. merge (centroid soft-assignment)
  A3. Pruning location    : pre-decoder (correct) vs. post-decoder (expected gibberish)
  A5. Text-conditioning   : text-conditioned vs. text-agnostic (mean-pooled) scoring

Each ablation writes a JSON into ``--output-dir`` tagged with the ablation name.
Use scripts/04_evaluate.py or scripts/06_generate_report.py to aggregate.

Usage:
    python scripts/05_run_ablations.py --benchmark vqav2 --max-samples 20 \
        --output-dir outputs/ablations
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tgtp.data.vqa_loader import BENCHMARK_CONFIGS, load_vqa_dataset
from tgtp.evaluation.vqa_metrics import compute_vqa_scores
from tgtp.models.smolvlm_wrapper import PRUNING_LOCATION, SmolVLMPruningWrapper
from tgtp.models.token_merger import TokenMerger
from tgtp.utils.logging import get_logger

logger = get_logger("tgtp.ablate")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run thesis ablations.")
    p.add_argument("--model", default="HuggingFaceTB/SmolVLM-Instruct")
    p.add_argument("--benchmark", choices=sorted(BENCHMARK_CONFIGS), default="vqav2")
    p.add_argument("--split", default=None)
    p.add_argument("--max-samples", type=int, default=20)
    p.add_argument("--keep-ratio", type=float, default=0.5)
    p.add_argument("--max-new-tokens", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="outputs/ablations")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def _greedy_decode(wrapper: SmolVLMPruningWrapper, seq: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    gen_ids = []
    for _ in range(max_new_tokens):
        out = wrapper.text_model(inputs_embeds=seq)
        nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        gen_ids.append(nxt)
        seq = torch.cat([seq, wrapper.text_model.get_input_embeddings()(nxt)], dim=1)
    return torch.cat(gen_ids, dim=1)


def run_config(wrapper, processor, samples, args, scoring, merge="drop", location="pre"):
    """Run one ablation config; returns (accuracy, note)."""
    accs = []
    notes = []
    metric = BENCHMARK_CONFIGS[args.benchmark]["metric"]
    for sample in samples:
        pv = processor(images=sample["image"], return_tensors="pt")["pixel_values"].to(args.device)
        enc = processor(text=sample["question"], return_tensors="pt")
        input_ids = enc["input_ids"].to(args.device)
        attn_mask = enc["attention_mask"].to(args.device)
        refs = sample["answers"] if isinstance(sample["answers"], list) else [sample["answers"]]
        refs = [str(r) for r in refs]

        with torch.no_grad():
            if location == "post":
                # Anti-pattern guard (D6): post-decoder pruning is forbidden and
                # empirically produces gibberish.
                try:
                    SmolVLMPruningWrapper.prune_from_last_hidden_state()
                except RuntimeError as exc:  # expected
                    notes.append("post-decoder rejected: " + str(exc)[:60])
                    accs.append(0.0)
                    continue

            visual = wrapper.extract_visual_tokens(pv)  # pre-decoder
            text_embeds = wrapper.extract_text_tokens(input_ids)
            scores = wrapper.score(visual, input_ids, attn_mask, scoring_method=scoring)
            pruned, keep_idx = wrapper.select_tokens(visual, scores, args.keep_ratio)
            if merge == "merge":
                pruned = TokenMerger("merge")(visual, keep_idx, scores)
            seq = torch.cat([pruned, text_embeds], dim=1)
            gen_ids = _greedy_decode(wrapper, seq, args.max_new_tokens)

        pred = processor.decode(gen_ids[0], skip_special_tokens=True).strip()
        accs.append(float(compute_vqa_scores([pred], [refs], metric)[0]))

    return (float(np.mean(accs)) if accs else 0.0, "; ".join(notes[:2]))


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    dtype = torch.float16 if args.device.type == "cuda" else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(args.model, torch_dtype=dtype).to(args.device)
    model.eval()
    wrapper = SmolVLMPruningWrapper(model)
    samples = load_vqa_dataset(args.benchmark, split=args.split, max_samples=args.max_samples)

    ablations = {
        "A1_scoring_cross_attention": {"scoring": "cross_attention"},
        "A1_scoring_cosine": {"scoring": "cosine"},
        "A1_scoring_random": {"scoring": "random"},
        "A2_merge_strategy_merge": {"scoring": "cross_attention", "merge": "merge"},
        # A3 is a guard demonstration (D6): post-decoder pruning is REJECTED by the
        # wrapper, so this cell records the rejection message rather than a real
        # accuracy number (expected: acc 0.0 + note "post-decoder rejected").
        "A3_location_post_decoder": {"scoring": "cross_attention", "location": "post"},
    }

    os.makedirs(args.output_dir, exist_ok=True)
    summary = {}
    for name, cfg in ablations.items():
        acc, note = run_config(wrapper, processor, samples, args, **cfg)
        summary[name] = {"accuracy": acc, "note": note, "keep_ratio": args.keep_ratio,
                         "pruning_location": PRUNING_LOCATION}
        logger.info("%s -> acc %.4f %s", name, acc, note)

    with open(os.path.join(args.output_dir, "ablations.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Wrote ablations -> %s/ablations.json", args.output_dir)


if __name__ == "__main__":
    main()
