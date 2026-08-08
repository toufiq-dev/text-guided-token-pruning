"""VQA dataset loading + visual-token caching.

Caching strategy (defends against Colab disconnects — see docs/colab_guide.md):

    1. Run scripts/01_extract_visual_tokens.py once per dataset split.
       The (potentially expensive) ViT+connector forward passes are cached to
       disk as .npy arrays keyed by sample index.
    2. Every subsequent experiment (training, inference, ablations) reads the
       cached tensors, so a 2-hour training run only needs the cache + the
       frozen decoder, and a Colab disconnect costs nothing.

``datasets`` and ``torch`` are imported lazily where possible.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

# Hub paths per benchmark (verify at runtime; ⚠️ may require HF auth or alternate names).
BENCHMARK_CONFIGS: dict[str, dict[str, Any]] = {
    "vqav2": {
        "hub_path": "HuggingFaceM4/VQAv2",
        "split": "validation",
        "image_key": "image",
        "question_key": "question",
        "answers_key": "answers",
        "metric": "vqa_accuracy",
    },
    "gqa": {
        "hub_path": "lmms-lab/GQA",
        "split": "testdev",
        "image_key": "image",
        "question_key": "question",
        "answers_key": "answer",
        "metric": "exact_match",
    },
    "textvqa": {
        "hub_path": "lmms-lab/TextVQA",
        "split": "validation",
        "image_key": "image",
        "question_key": "question",
        "answers_key": "answers",
        "metric": "vqa_accuracy",
    },
}


def load_vqa_dataset(
    benchmark: str,
    split: str | None = None,
    max_samples: int | None = None,
    seed: int = 0,
    local_jsonl: str | None = None,
) -> list[dict[str, Any]]:
    """Load a VQA benchmark as a list of sample dicts.

    Falls back to a local JSONL file (one JSON object per line) when the
    HuggingFace dataset is unavailable (offline / no auth). JSONL schema:
    {"image": <path|url>, "question": str, "answers": [str, ...]}.
    """
    cfg = BENCHMARK_CONFIGS.get(benchmark)
    if cfg is None:
        raise ValueError(f"Unknown benchmark '{benchmark}'. Choose from {sorted(BENCHMARK_CONFIGS)}")
    if local_jsonl is not None and os.path.exists(local_jsonl):
        return _load_jsonl(local_jsonl, max_samples, seed)

    from datasets import load_dataset

    ds = load_dataset(cfg["hub_path"], split=split or cfg["split"])
    samples = []
    img_key = cfg["image_key"]
    q_key = cfg["question_key"]
    a_key = cfg["answers_key"]
    for row in ds:
        answers = row.get(a_key)
        if isinstance(answers, str):
            answers = [answers]
        samples.append(
            {
                "image": row.get(img_key),
                "question": row.get(q_key, ""),
                "answers": answers or [],
                "metric": cfg["metric"],
                "benchmark": benchmark,
            }
        )
        if max_samples is not None and len(samples) >= max_samples:
            break
    return samples


def _load_jsonl(path: str, max_samples: int | None, seed: int) -> list[dict[str, Any]]:
    samples = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    if max_samples is not None:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(samples), size=min(max_samples, len(samples)), replace=False)
        samples = [samples[i] for i in sorted(idx)]
    return samples


class VQADataset(Dataset):
    """Torch dataset yielding (image, question, answers) tuples."""

    def __init__(self, samples: list[dict[str, Any]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = dict(self.samples[idx])
        sample["index"] = idx
        return sample


def cache_visual_tokens(
    wrapper: Any,
    samples: list[dict[str, Any]],
    output_dir: str,
    device: torch.device,
    batch_size: int = 8,
    model_name: str = "HuggingFaceTB/SmolVLM-Instruct",
) -> tuple[int, int]:
    """Extract pre-decoder visual tokens for each sample and cache them as .npy.

    This is the *correct extraction point* (pre-decoder): vision_model + connector
    only — the LLM decoder is never invoked here, and pruning happens later on the
    cached tensors. Saving this cache is what makes Colab workflows resilient.

    Returns:
        (num_cached, num_patches_per_image).
    """
    from tgtp.data.image_processor import get_image_processor, process_image

    os.makedirs(output_dir, exist_ok=True)
    processor = get_image_processor(model_name)
    num_patches = -1
    count = 0

    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        pvs = []
        valid = []
        for sample in batch:
            image = sample["image"]
            if image is None:
                continue
            pv = process_image(image, processor, model_name).to(device)
            pvs.append(pv)
            valid.append(sample)
        if not pvs:
            continue
        pixel_values = torch.cat(pvs, dim=0)
        with torch.no_grad():
            tokens = wrapper.extract_visual_tokens(pixel_values).float().cpu().numpy()
        for offset, sample in enumerate(valid):
            np.save(os.path.join(output_dir, f"{sample['index']:08d}.npy"), tokens[offset])
            count += 1
            if num_patches < 0:
                num_patches = tokens[offset].shape[0]
        if device.type == "cuda":
            torch.cuda.empty_cache()

    manifest = {"model": model_name, "num_cached": count, "num_patches": num_patches}
    with open(os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return count, num_patches


def load_cached_visual_tokens(cache_dir: str, index: int) -> torch.Tensor:
    """Load a cached (B=1, N, D) pre-decoder visual-token tensor for a sample."""
    return torch.from_numpy(np.load(os.path.join(cache_dir, f"{index:08d}.npy"))).unsqueeze(0)
