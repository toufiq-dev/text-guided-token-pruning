"""TextGuidedPruner: the core novel module of the thesis.

The pruner sits **between the vision connector and the LLM decoder**
(pre-decoder). It receives:

    * visual tokens ``(B, N, D)`` — output of the connector (NOT ``last_hidden_state``
      of a full model forward pass; those are post-decoder and unusable here), and
    * text token embeddings ``(B, M, D)`` — the LLM's input embeddings for the prompt.

It scores each visual token's relevance to the prompt with a small learnable
cross-attention module, selects the top-K tokens (K = N * keep_ratio), and
returns the pruned sequence ready to feed the frozen LLM decoder.

Differentiability: during training we use a Gumbel-perturbed, straight-through
estimate of the top-K selection (soft gradients flow through the scorer while the
forward pass uses the hard selection). At inference the noise is disabled and the
selection is a deterministic hard top-K. A temperature schedule anneals
tau: 5.0 -> 0.1 for stable early training.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from tgtp.models.cross_attention_scorer import (
    CrossAttentionScorer,
    cosine_score,
    cross_attention_score,
)
from tgtp.models.token_merger import TokenMerger


def _gumbel_noise(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    """Standard Gumbel(0,1) noise via inverse-CDF transform."""
    u = torch.rand(shape, device=device).clamp_min(1e-8)
    return -torch.log(-torch.log(u))


def topk_indices(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Return sorted top-K indices per batch element (preserves spatial order)."""
    _, idx = torch.topk(scores, k, dim=-1)
    idx, _ = torch.sort(idx, dim=-1)
    return idx


class TextGuidedPruner(nn.Module):
    """Lightweight, text-conditioned, pre-decoder visual token pruner."""

    def __init__(
        self,
        visual_dim: int,
        text_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        keep_ratio: float = 0.5,
        temperature_init: float = 5.0,
        temperature_min: float = 0.1,
        temperature_anneal_steps: int = 1000,
        aggregation: str = "max",
        merge_strategy: str = "drop",
    ) -> None:
        super().__init__()
        assert 0.0 < keep_ratio <= 1.0, "keep_ratio must be in (0, 1]"
        self.scorer = CrossAttentionScorer(
            visual_dim=visual_dim,
            text_dim=text_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            aggregation=aggregation,
        )
        self.merger = TokenMerger(merge_strategy)
        self.keep_ratio = keep_ratio
        self.temperature = temperature_init
        self.temperature_min = temperature_min
        self.temperature_anneal_steps = max(1, temperature_anneal_steps)
        self.aggregation = aggregation
        self.merge_strategy = merge_strategy

    # ------------------------------------------------------------------ utils
    def set_keep_ratio(self, keep_ratio: float) -> None:
        """Adjust the target keep ratio (e.g. at inference time)."""
        assert 0.0 < keep_ratio <= 1.0
        self.keep_ratio = float(keep_ratio)

    def get_temperature(self) -> float:
        return float(self.temperature)

    def anneal_temperature(self, step: int) -> float:
        """Cosine-anneal temperature from init down to min over the schedule."""
        progress = min(1.0, max(0.0, step / self.temperature_anneal_steps))
        self.temperature = self.temperature_min + 0.5 * (1.0 + math.cos(math.pi * progress)) * (
            self.temperature - self.temperature_min
        )
        return float(self.temperature)

    # ------------------------------------------------------------- selection
    def selection_mask(
        self,
        scores: torch.Tensor,
        training: bool,
        temperature: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute a (B, N) one-hot keep mask and the (B, K) kept indices.

        Args:
            scores: (B, N) relevance scores in [0, 1].
            training: if True, use Gumbel-perturbed straight-through selection.
            temperature: gumbel temperature; defaults to the annealed value.

        Returns:
            (mask, keep_indices): mask is (B, N) float one-hot; keep_indices (B, K).
        """
        bsz, num_tokens = scores.shape
        k = max(1, int(round(num_tokens * self.keep_ratio)))
        tau = self.temperature if temperature is None else temperature

        if training:
            # Perturbed top-K: add Gumbel noise to the logits so the soft weights
            # carry gradient signal, while the forward selection stays hard (STE).
            logits = scores / max(tau, 1e-4)
            noisy = logits + _gumbel_noise(scores.shape, scores.device)
            soft_weights = F.softmax(noisy, dim=-1)  # (B, N) smooth selection probs
            hard = torch.zeros_like(scores)
            idx = topk_indices(scores, k)
            hard.scatter_(1, idx, 1.0)
            mask = hard + soft_weights - soft_weights.detach()  # straight-through
        else:
            idx = topk_indices(scores, k)
            mask = torch.zeros_like(scores)
            mask.scatter_(1, idx, 1.0)

        return mask, idx

    # ------------------------------------------------------------- forward
    def forward(
        self,
        visual_tokens: torch.Tensor,
        text_embeds: torch.Tensor,
        text_attention_mask: torch.Tensor | None = None,
        training: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prune visual tokens given the text prompt.

        CRITICAL: ``visual_tokens`` MUST be pre-decoder tokens (connector output).
        Post-decoder hidden states produce garbage selections and garbage answers.

        Args:
            visual_tokens: (B, N, D) pre-decoder visual tokens.
            text_embeds: (B, M, D) text token embeddings.
            text_attention_mask: optional (B, M) padding mask.
            training: enable Gumbel straight-through top-K (True) or hard top-K.

        Returns:
            (pruned_tokens, keep_indices, scores):
                pruned_tokens (B, K, D), keep_indices (B, K) sorted,
                scores (B, N) in [0, 1].
        """
        scores = self.scorer(visual_tokens, text_embeds, text_attention_mask)  # (B, N)
        _, keep_indices = self.selection_mask(scores, training=training)
        pruned = self.merger(visual_tokens, keep_indices, scores)  # (B, K, D)

        if training:
            # Differentiable soft-masking surrogate: gates every visual token by its
            # Gumbel-softmax keep-probability. The hard top-K gather (above) does not
            # propagate gradients to the scorer, so the training loop adds a loss term
            # on ``last_soft_masked`` (soft pruning surrogate) — gradients flow through
            # the soft weights and into the scorer. At inference this surrogate is
            # discarded and only the hard top-K remains.
            tau = max(self.temperature, 1e-4)
            soft_weights = F.softmax(scores / tau, dim=-1)  # (B, N)
            self.last_soft_weights = soft_weights
            self.last_soft_masked = visual_tokens * soft_weights.unsqueeze(-1)  # (B, N, D)
        else:
            self.last_soft_weights = None
            self.last_soft_masked = None

        return pruned, keep_indices, scores

    # ------------------------------------------------- functional scorers
    @staticmethod
    def score_with_method(
        method: str,
        visual_tokens: torch.Tensor,
        text_embeds: torch.Tensor,
        text_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score with any supported method (used by scripts and baselines).

        Supported methods: 'cross_attention' (max), 'cross_attention_mean',
        'cosine', 'random'. All return (B, N) scores in [0, 1].
        """
        if method == "cross_attention":
            return cross_attention_score(visual_tokens, text_embeds, text_attention_mask, aggregation="max")
        if method == "cross_attention_mean":
            return cross_attention_score(visual_tokens, text_embeds, text_attention_mask, aggregation="mean")
        if method == "cosine":
            return cosine_score(visual_tokens, text_embeds, text_attention_mask)
        if method == "random":
            return torch.rand(visual_tokens.shape[:2], device=visual_tokens.device)
        raise ValueError(f"Unknown scoring method: {method}")
