"""Cross-attention scoring of visual tokens against text embeddings.

These functions and the ``CrossAttentionScorer`` module answer one question:
*which visual tokens does the text prompt actually care about?*

The scoring operates on **pre-decoder** representations:

    visual tokens (connector output)  x  text token embeddings (LLM input embeddings)

It must NEVER be applied to ``last_hidden_state`` of a full forward pass, because
those hidden states are post-decoder (fully contextualized by the LLM).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

EPS = 1e-8


def _masked_softmax(attn_logits: torch.Tensor, text_attention_mask: torch.Tensor | None) -> torch.Tensor:
    """Apply optional padding mask to attention logits before softmax.

    Args:
        attn_logits: (B, N, M) logits between visual and text tokens.
        text_attention_mask: (B, M) float/bool; 1 for real tokens, 0 for padding.
    """
    if text_attention_mask is not None:
        pad = (1.0 - text_attention_mask.float()) * -1e9
        attn_logits = attn_logits + pad.unsqueeze(1)
    return F.softmax(attn_logits, dim=-1)


def cross_attention_score(
    visual_tokens: torch.Tensor,
    text_embeds: torch.Tensor,
    text_attention_mask: torch.Tensor | None = None,
    aggregation: str = "max",
    temperature: float = 1.0,
) -> torch.Tensor:
    """Score each visual token by its attention to the text tokens.

    Args:
        visual_tokens: (B, N, D) pre-decoder visual tokens (connector output).
        text_embeds: (B, M, D) text token embeddings (LLM input embeddings).
        text_attention_mask: optional (B, M) mask marking real text positions.
        aggregation: 'max' (attention to most-relevant text token) or 'mean'.
        temperature: softmax temperature (higher = flatter attention).

    Returns:
        scores: (B, N) in [0, 1] — per-visual-token relevance to the prompt.
    """
    q = F.normalize(visual_tokens, dim=-1)
    k = F.normalize(text_embeds, dim=-1)
    attn_logits = torch.bmm(q, k.transpose(1, 2)) / max(temperature, EPS)  # (B, N, M)
    attn = _masked_softmax(attn_logits, text_attention_mask)
    if aggregation == "max":
        return attn.max(dim=-1).values
    if aggregation == "mean":
        return attn.mean(dim=-1)
    raise ValueError(f"Unknown aggregation '{aggregation}'; use 'max' or 'mean'.")


def cosine_score(
    visual_tokens: torch.Tensor,
    text_embeds: torch.Tensor,
    text_attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score each visual token by cosine similarity to the mean text embedding.

    The raw cosine similarity lies in [-1, 1]; we map it to [0, 1] via (x + 1) / 2
    so scores from every scoring method share a common, interpretable range.

    Returns:
        scores: (B, N) in [0, 1].
    """
    q = F.normalize(visual_tokens, dim=-1)
    k = F.normalize(text_embeds, dim=-1)
    if text_attention_mask is not None:
        mask = text_attention_mask.unsqueeze(-1).float()  # (B, M, 1)
        text_mean = (k * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)  # (B, D)
    else:
        text_mean = k.mean(dim=1)
    sim = (q * text_mean.unsqueeze(1)).sum(dim=-1)  # (B, N)
    return (sim + 1.0) / 2.0


class CrossAttentionScorer(nn.Module):
    """Learnable cross-attention scorer (the trainable heart of the pruner).        Visual tokens are projected to queries; text tokens to keys.
        Per-head attention scores are aggregated (max over text, then mean over
        heads) into a single (B, N) relevance vector in [0, 1].

    This module is deliberately *tiny* (~O(hidden^2) parameters) so it can be
    trained on a single GPU / Colab session while the MLLM stays frozen.
    """

    def __init__(
        self,
        visual_dim: int,
        text_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        aggregation: str = "max",
    ) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.visual_dim = visual_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.aggregation = aggregation
        self.head_dim = hidden_dim // num_heads

        self.visual_proj = nn.Linear(visual_dim, hidden_dim, bias=False)
        self.text_key_proj = nn.Linear(text_dim, hidden_dim, bias=False)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        text_embeds: torch.Tensor,
        text_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute per-visual-token relevance scores.

        Args:
            visual_tokens: (B, N, D_v) pre-decoder visual tokens.
            text_embeds: (B, M, D_t) text token embeddings.
            text_attention_mask: optional (B, M) padding mask.

        Returns:
            scores: (B, N) in [0, 1].
        """
        bsz, num_vis, _ = visual_tokens.shape
        num_text = text_embeds.shape[1]
        H, hd = self.num_heads, self.head_dim

        q = self.visual_proj(visual_tokens).view(bsz, num_vis, H, hd).transpose(1, 2)  # (B,H,N,hd)
        k = self.text_key_proj(text_embeds).view(bsz, num_text, H, hd).transpose(1, 2)  # (B,H,M,hd)

        attn_logits = torch.matmul(q, k.transpose(-1, -2)) / (hd**0.5)  # (B,H,N,M)

        if text_attention_mask is not None:
            pad = (1.0 - text_attention_mask.float()).unsqueeze(1).unsqueeze(2) * -1e9
            attn_logits = attn_logits + pad

        attn = F.softmax(attn_logits, dim=-1)  # (B,H,N,M)
        if self.aggregation == "max":
            scores = attn.max(dim=-1).values  # (B,H,N) best text token per head
        elif self.aggregation == "mean":
            scores = attn.mean(dim=-1)
        else:
            raise ValueError(f"Unknown aggregation '{self.aggregation}'")
        scores = scores.mean(dim=1)  # (B,N) mean over heads
        return scores
