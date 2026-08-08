"""Unit tests for tgtp.models.cross_attention_scorer."""

import pytest
import torch

from tgtp.models.cross_attention_scorer import (
    CrossAttentionScorer,
    cosine_score,
    cross_attention_score,
)

B, N, M, D = 2, 32, 16, 24


@pytest.fixture()
def tensors():
    torch.manual_seed(0)
    return {
        "visual": torch.randn(B, N, D),
        "text": torch.randn(B, M, D),
        "mask": torch.ones(B, M, dtype=torch.bool),
    }


def test_functional_cross_attention_shape_and_range(tensors):
    scores = cross_attention_score(tensors["visual"], tensors["text"])
    assert scores.shape == (B, N)
    assert torch.all(scores >= 0.0) and torch.all(scores <= 1.0)


def test_functional_cosine_shape_and_range(tensors):
    scores = cosine_score(tensors["visual"], tensors["text"])
    assert scores.shape == (B, N)
    assert torch.all(scores >= 0.0) and torch.all(scores <= 1.0)


def test_padding_mask_produces_uniform_attention():
    """If ALL text tokens are masked, softmax must degrade to uniform (no NaN)."""
    visual = torch.randn(2, 8, 8)
    text = torch.randn(2, 8, 8)
    zero_mask = torch.zeros(2, 8, dtype=torch.bool)
    scores = cross_attention_score(visual, text, zero_mask)
    assert torch.isfinite(scores).all()
    # Uniform attention over M=8 masked positions -> max = 1/8.
    assert torch.allclose(scores, torch.full_like(scores, 1.0 / 8), atol=1e-4)


def test_mask_changes_scores(tensors):
    """Masking out the first half of text tokens must change the scores."""
    unmasked = cross_attention_score(tensors["visual"], tensors["text"])
    mask = torch.ones(B, M, dtype=torch.bool)
    mask[:, : M // 2] = False
    masked = cross_attention_score(tensors["visual"], tensors["text"], mask)
    assert not torch.allclose(unmasked, masked)


def test_max_vs_mean_aggregation():
    torch.manual_seed(0)
    visual = torch.randn(B, N, D)
    text = torch.randn(B, M, D)
    mx = cross_attention_score(visual, text, aggregation="max")
    mn = cross_attention_score(visual, text, aggregation="mean")
    assert torch.all(mx >= mn - 1e-6)


def test_learnable_scorer_trains():
    torch.manual_seed(0)
    scorer = CrossAttentionScorer(visual_dim=D, text_dim=D, hidden_dim=16, num_heads=4)
    visual = torch.randn(B, N, D)
    text = torch.randn(B, M, D)
    scores = scorer(visual, text)
    assert scores.shape == (B, N)
    loss = scores.sum()
    loss.backward()
    assert scorer.visual_proj.weight.grad is not None
    assert scorer.visual_proj.weight.grad.abs().sum() > 0


def test_learnable_scorer_mask(tensors):
    scorer = CrossAttentionScorer(visual_dim=D, text_dim=D, hidden_dim=16, num_heads=4)
    scores_masked = scorer(tensors["visual"], tensors["text"], tensors["mask"])
    assert torch.isfinite(scores_masked).all()
    assert torch.all(scores_masked >= 0) and torch.all(scores_masked <= 1)
