"""Visualization helpers (D1 heatmaps, D3 score histograms, tradeoff curves).

``matplotlib`` is imported lazily; if it is unavailable the functions degrade to
saving a numpy-array PNG via PIL or writing a textual summary, so notebooks and
scripts never hard-crash on a bare Colab image.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
from PIL import Image


def overlay_token_heatmap(
    image,
    scores: np.ndarray,
    grid_h: int,
    grid_w: int,
    save_path: str,
    alpha: float = 0.45,
) -> str:
    """D1: overlay per-token relevance scores on the image as a heatmap.

    Args:
        image: PIL image (or path).
        scores: (grid_h * grid_w,) relevance scores in [0, 1] (row-major).
        grid_h, grid_w: feature grid dimensions (e.g. 9x9 = 81 SmolVLM tokens).
        save_path: where to write the PNG.
        alpha: overlay opacity.

    Returns:
        the path the image was written to.
    """
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    else:
        image = image.convert("RGB")

    if scores.size != grid_h * grid_w:
        raise ValueError(f"scores size {scores.size} != grid {grid_h}x{grid_w}")

    # Normalize scores to [0, 1] for the colormap.
    lo, hi = float(scores.min()), float(scores.max())
    norm = (scores - lo) / max(hi - lo, 1e-8)

    img = np.asarray(image, dtype=np.float32) / 255.0
    h, w = img.shape[:2]

    # Nearest-neighbor upscale the score grid to image size.
    grid = norm.reshape(grid_h, grid_w)
    heat = np.repeat(np.repeat(grid, np.ceil(h / grid_h).astype(int), axis=0),
                     np.ceil(w / grid_w).astype(int), axis=1)[:h, :w]
    heat = heat[..., None]

    # Red-hot colormap approximation (jet-like: blue->cyan->yellow->red).
    r = np.clip(4 * heat - 1, 0, 1) * np.clip(4 - 4 * heat, 0, 1) + np.clip(4 * heat - 3, 0, 1)
    g = np.clip(4 * heat - 1, 0, 1) * np.clip(4 - 4 * heat, 0, 1) * 0.6
    b = np.clip(1 - 4 * heat, 0, 1) + np.clip(3 - 4 * heat, 0, 1) * 0.2
    color = np.stack([r, g, b], axis=-1)

    overlay = img * (1 - alpha) + color * alpha
    out = Image.fromarray((np.clip(overlay, 0, 1) * 255).astype(np.uint8))
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    out.save(save_path)
    return save_path


def score_histogram(scores: np.ndarray, save_path: str, bins: int = 20) -> str:
    """D3: histogram of score distributions. PNG via matplotlib if present,
    otherwise a simple PIL bar chart."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(np.asarray(scores, dtype=float).ravel(), bins=bins, color="#4C72B0")
        ax.set_xlabel("relevance score")
        ax.set_ylabel("tokens")
        ax.set_title("Visual token relevance scores")
        fig.tight_layout()
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        return save_path
    except Exception:  # noqa: BLE001 - pragma: no cover - degrade gracefully
        s = np.asarray(scores, dtype=float).ravel()
        hist, edges = np.histogram(s, bins=bins)
        w = max(int(edges[1] - edges[0]), 1)
        img = Image.new("RGB", (w * bins, 200), "white")
        px = img.load()
        max_count = max(hist.max(), 1)
        for i, count in enumerate(hist):
            hgt = int(190 * count / max_count)
            for x in range(i * w, (i + 1) * w):
                for y in range(200 - hgt, 200):
                    px[x, y] = (76, 114, 176)
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        img.save(save_path)
        return save_path


def plot_pruning_tradeoff(
    ratios: Sequence[float],
    acc_drops: Sequence[float],
    save_path: str,
    labels: Sequence[str] | None = None,
) -> str:
    """Accuracy drop vs pruning ratio curve (matplotlib; textual fallback)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(ratios, acc_drops, marker="o", color="#C44E52")
        ax.axhline(2.0, ls="--", color="gray", label="2% thesis budget")
        ax.set_xlabel("pruning ratio (fraction removed)")
        ax.set_ylabel("accuracy drop (pp)")
        ax.set_title("Accuracy–efficiency tradeoff")
        ax.legend()
        fig.tight_layout()
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        return save_path
    except Exception:  # noqa: BLE001 - pragma: no cover - degrade gracefully
        with open(save_path, "w", encoding="utf-8") as fh:
            fh.write(f"ratios: {list(ratios)}\ndrops: {list(acc_drops)}\n")
        return save_path
