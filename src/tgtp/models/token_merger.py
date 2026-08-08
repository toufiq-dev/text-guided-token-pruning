"""Token merging after top-K selection.

Two strategies are supported (both operate on pre-decoder visual tokens):

* ``drop``:  the bottom (N - K) tokens are discarded. Zero extra compute.
* ``merge``: dropped tokens are *soft-assigned* to their nearest kept token
  (cosine similarity) and each kept token is replaced by the centroid of its
  cluster (kept token + assigned dropped tokens). No information is destroyed;
  spatial position of the kept token is preserved, so spatial reasoning (GQA)
  is affected less than with hard dropping.

The merge strategy is one of the thesis ablations (Section 5 of the proposal):
at a fixed token budget K, does merging beat dropping?
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class TokenMerger:
    """Merge/drop visual tokens down to the top-K kept by the pruner."""

    def __init__(self, strategy: str = "drop", assignment_temperature: float = 0.1) -> None:
        assert strategy in {"drop", "merge"}, f"Unknown merge strategy: {strategy}"
        self.strategy = strategy
        self.assignment_temperature = assignment_temperature

    def forward(
        self,
        visual_tokens: torch.Tensor,
        keep_indices: torch.Tensor,
        scores: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reduce ``visual_tokens`` to the tokens indexed by ``keep_indices``.

        Args:
            visual_tokens: (B, N, D) pre-decoder visual tokens.
            keep_indices: (B, K) indices of kept tokens (already sorted).
            scores: optional (B, N) relevance scores (used for soft assignment).

        Returns:
            merged: (B, K, D) — kept tokens, optionally enriched by dropped ones.
        """
        bsz, num_tokens, dim = visual_tokens.shape
        keep_indices = keep_indices.to(visual_tokens.device)
        batch_idx = torch.arange(bsz, device=visual_tokens.device).unsqueeze(1)
        kept = visual_tokens[batch_idx, keep_indices]  # (B, K, D)

        if self.strategy == "drop":
            return kept

        # ---- merge: soft-assign dropped tokens to nearest kept token ----
        dropped_mask = torch.ones(
            bsz, num_tokens, dtype=torch.bool, device=visual_tokens.device
        )
        dropped_mask.scatter_(1, keep_indices, False)
        dropped = visual_tokens[dropped_mask].view(bsz, num_tokens - keep_indices.shape[1], dim)

        if dropped.shape[1] == 0:
            return kept

        kept_n = F.normalize(kept, dim=-1)
        dropped_n = F.normalize(dropped, dim=-1)
        sim = torch.bmm(dropped_n, kept_n.transpose(1, 2))  # (B, N-K, K)

        # Soft assignment weights: temperature-scaled softmax over kept tokens.
        if scores is not None:
            dropped_scores = scores[dropped_mask].view(bsz, num_tokens - keep_indices.shape[1], 1)
            sim = sim * (0.5 + dropped_scores)  # relevance-weighted similarity

        weights = F.softmax(sim / self.assignment_temperature, dim=-1)  # (B, N-K, K) soft assignment
        contrib = torch.bmm(weights.transpose(1, 2), dropped)  # (B, K, D)
        counts = weights.sum(dim=1).clamp(min=1.0).unsqueeze(-1)  # (B, K, 1)
        merged = kept + contrib / counts
        return merged

    def __call__(self, *args, **kwargs) -> torch.Tensor:  # type: ignore[no-untyped-def]
        return self.forward(*args, **kwargs)
