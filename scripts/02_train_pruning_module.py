#!/usr/bin/env python3
"""02_train_pruning_module.py — train the TextGuidedPruner (distillation).

Loss (Section 4.4 of the proposal):

    L = L_task + lambda_distill * L_distill + lambda_entropy * L_entropy

* L_task      — VQA cross-entropy on the *pruned* sequence (causal LM loss).
* L_distill   — KL(softmax(teacher_logits / tau) || softmax(student_logits / tau)) * tau^2
                where the teacher runs on the FULL visual token sequence.
* L_entropy   — entropy penalty that prevents score collapse (D3).

The vision encoder, connector, and LLM decoder are all FROZEN; only the pruner's
~O(256^2) parameters are trained. Visual tokens can be read from the disk cache
created by 01_extract_visual_tokens.py (Colab-disconnect-safe).

Usage:
    python scripts/02_train_pruning_module.py --model HuggingFaceTB/SmolVLM-Instruct \
        --dataset vqav2 --keep-ratio 0.5 --num-epochs 2 --lr 3e-4 --batch-size 4 \
        --cache-dir cached_features/vqav2 --output-dir outputs/pruner_50pct
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tgtp.data.vqa_loader import (
    BENCHMARK_CONFIGS,
    load_cached_visual_tokens,
    load_vqa_dataset,
)
from tgtp.models.pruning_module import TextGuidedPruner
from tgtp.models.smolvlm_wrapper import SmolVLMPruningWrapper
from tgtp.utils.logging import ExperimentLogger, get_logger

logger = get_logger("tgtp.train")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the TextGuidedPruner via distillation.")
    p.add_argument("--model", default="HuggingFaceTB/SmolVLM-Instruct")
    p.add_argument("--dataset", choices=sorted(BENCHMARK_CONFIGS), default="vqav2")
    p.add_argument("--split", default=None)
    p.add_argument("--cache-dir", default=None, help="visual-token cache from 01_extract_visual_tokens.py")
    p.add_argument("--max-samples", type=int, default=256)
    p.add_argument("--keep-ratio", type=float, default=0.5)
    p.add_argument("--num-epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--accum-steps", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=0, help="0 = run full epochs")
    p.add_argument("--temperature-init", type=float, default=5.0)
    p.add_argument("--temperature-min", type=float, default=0.1)
    p.add_argument("--anneal-steps", type=int, default=1000)
    p.add_argument("--distill-weight", type=float, default=1.0)
    p.add_argument("--entropy-weight", type=float, default=0.1)
    p.add_argument("--distill-temperature", type=float, default=2.0)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--merge-strategy", choices=["drop", "merge"], default="drop")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--output-dir", default="outputs/pruner")
    p.add_argument("--max-seq-len", type=int, default=4096)
    return p.parse_args()


def tokenize(processor, question: str, max_len: int) -> dict:
    enc = processor(text=question, return_tensors="pt")
    ids = enc["input_ids"][0][:max_len]
    mask = enc["attention_mask"][0][:max_len]
    return {"input_ids": ids, "attention_mask": mask}


def build_embeds(wrapper: SmolVLMPruningWrapper, visual: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Concatenate (pre-decoder) visual tokens with text embeddings."""
    text_embeds = wrapper.extract_text_tokens(input_ids)
    return torch.cat([visual, text_embeds], dim=1)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(args.model, torch_dtype=dtype).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    wrapper = SmolVLMPruningWrapper(model)

    samples = load_vqa_dataset(args.dataset, split=args.split, max_samples=args.max_samples)
    logger.info("Loaded %d samples", len(samples))

    # Read a single cached token tensor to infer dims (or extract on the fly).
    if args.cache_dir and os.path.isdir(args.cache_dir) and os.listdir(args.cache_dir):
        first = sorted(f for f in os.listdir(args.cache_dir) if f.endswith(".npy"))[0]
        sample_tokens = torch.from_numpy(np.load(os.path.join(args.cache_dir, first)))
        visual_dim = sample_tokens.shape[-1]
    else:
        logger.info("No cache found; extracting visual tokens on the fly (slower).")
        visual_dim = None  # resolved after first forward

    pruner = TextGuidedPruner(
        visual_dim=visual_dim or model.config.text_config.hidden_size,
        text_dim=model.config.text_config.hidden_size,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        keep_ratio=args.keep_ratio,
        temperature_init=args.temperature_init,
        temperature_min=args.temperature_min,
        temperature_anneal_steps=args.anneal_steps,
        merge_strategy=args.merge_strategy,
    ).to(device)
    trainable = sum(p.numel() for p in pruner.parameters() if p.requires_grad)
    logger.info("Pruner trainable params: %d", trainable)

    optimizer = torch.optim.AdamW(pruner.parameters(), lr=args.lr)
    tracker = ExperimentLogger(os.path.join(args.output_dir, "train.jsonl"))

    cache_dir = args.cache_dir or os.path.join("cached_features", args.dataset)
    os.makedirs(args.output_dir, exist_ok=True)

    global_step = 0
    for epoch in range(args.num_epochs):
        rng = np.random.default_rng(args.seed + epoch)
        order = rng.permutation(len(samples))
        for start in range(0, len(order), args.batch_size):
            idxs = order[start : start + args.batch_size]
            losses = []
            for i in idxs:
                sample = samples[int(i)]
                enc = tokenize(processor, sample["question"], args.max_seq_len)
                input_ids = enc["input_ids"].unsqueeze(0).to(device)
                attn_mask = enc["attention_mask"].unsqueeze(0).to(device)

                if cache_dir and os.path.exists(os.path.join(cache_dir, f"{int(i):08d}.npy")):
                    visual = load_cached_visual_tokens(cache_dir, int(i)).to(device)
                    visual = visual.to(dtype)
                else:
                    pv = processor(images=sample["image"], return_tensors="pt")["pixel_values"].to(device)
                    with torch.no_grad():
                        visual = wrapper.extract_visual_tokens(pv)

                with torch.no_grad():
                    teacher_embeds = build_embeds(wrapper, visual, input_ids)
                    # Full-sequence mask of ones: the sequence is [visual; text].
                    teacher_mask = torch.ones(
                        teacher_embeds.shape[:2], dtype=attn_mask.dtype, device=device
                    )
                    teacher_out = wrapper.text_model(inputs_embeds=teacher_embeds, attention_mask=teacher_mask)
                    teacher_logits = teacher_out.logits  # (1, S, V)

                pruned, _, scores = pruner(
                    visual, wrapper.extract_text_tokens(input_ids), attn_mask, training=True
                )
                student_embeds = torch.cat([pruned, wrapper.extract_text_tokens(input_ids)], dim=1)
                student_mask = torch.ones(
                    student_embeds.shape[:2], dtype=attn_mask.dtype, device=device
                )
                student_out = wrapper.text_model(inputs_embeds=student_embeds, attention_mask=student_mask)
                student_logits = student_out.logits

                # Task loss (hard-label distillation at the answer position): the
                # student's prediction at the final prompt position is supervised by
                # the FULL-TOKEN teacher's argmax. Using the student's own argmax
                # would be a degenerate self-teaching loop; using the teacher's is a
                # standard distillation target and keeps shapes aligned (the two
                # sequences differ in length, so only the final position matches).
                teacher_target = teacher_logits[:, -1, :].argmax(dim=-1)
                loss_task = F.cross_entropy(student_logits[:, -1, :], teacher_target)

                # Distillation: KL(teacher || student) at the final position.
                tau = args.distill_temperature
                t_dist = F.log_softmax(student_logits[:, -1, :] / tau, dim=-1)
                s_dist = F.softmax(teacher_logits[:, -1, :] / tau, dim=-1)
                loss_distill = F.kl_div(t_dist, s_dist, reduction="batchmean") * (tau * tau)

                # Soft-masking surrogate distillation: KL against the soft-gated
                # (differentiable) student sequence — the ONLY path that carries
                # gradients into the scorer (the hard top-K gather does not).
                loss_surrogate = torch.tensor(0.0, device=device)
                if pruner.last_soft_masked is not None:
                    soft_seq = torch.cat(
                        [pruner.last_soft_masked, wrapper.extract_text_tokens(input_ids)], dim=1
                    )
                    soft_mask = torch.ones(
                        soft_seq.shape[:2], dtype=attn_mask.dtype, device=device
                    )
                    soft_out = wrapper.text_model(inputs_embeds=soft_seq, attention_mask=soft_mask)
                    soft_dist = F.log_softmax(soft_out.logits[:, -1, :] / tau, dim=-1)
                    loss_surrogate = F.kl_div(soft_dist, s_dist, reduction="batchmean") * (tau * tau)

                # Entropy penalty on the score distribution (prevents collapse, D3).
                # loss = -H(mask_soft); minimizing it raises entropy.
                mask_soft = F.softmax(scores / max(pruner.get_temperature(), 1e-4), dim=-1)
                loss_entropy = (mask_soft * torch.log(mask_soft + 1e-9)).sum(dim=-1).mean()

                loss = (
                    loss_task
                    + args.distill_weight * (loss_distill + loss_surrogate)
                    + args.entropy_weight * loss_entropy
                )
                (loss / args.accum_steps).backward()
                losses.append(float(loss.detach()))

                pruner.anneal_temperature(global_step)
                global_step += 1
                if args.max_steps and global_step >= args.max_steps:
                    break

            if (start // args.batch_size) % args.accum_steps == args.accum_steps - 1 or (
                start + args.batch_size >= len(order)
            ):
                torch.nn.utils.clip_grad_norm_(pruner.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            if global_step % 20 == 0:
                tracker.log_metrics(
                    global_step,
                    {"loss": float(np.mean(losses)), "temp": pruner.get_temperature()},
                    tag="train",
                )
                logger.info("epoch %d step %d loss %.4f temp %.3f", epoch, global_step,
                            float(np.mean(losses)), pruner.get_temperature())

    # Save pruner + config.
    torch.save(pruner.state_dict(), os.path.join(args.output_dir, "pruner.pt"))
    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {
                "visual_dim": pruner.scorer.visual_dim,
                "text_dim": pruner.scorer.text_dim,
                "hidden_dim": args.hidden_dim,
                "num_heads": args.num_heads,
                "keep_ratio": args.keep_ratio,
                "merge_strategy": args.merge_strategy,
                "temperature_min": args.temperature_min,
            },
            fh,
            indent=2,
        )
    logger.info("Saved pruner -> %s", os.path.join(args.output_dir, "pruner.pt"))


if __name__ == "__main__":
    main()
