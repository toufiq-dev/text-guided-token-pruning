"""Unit tests for tgtp.models.pruning_module.TextGuidedPruner.

Covers: output shapes, score normalization, keep-ratio exactness, gradient flow
through the Gumbel/STE top-K, hard top-K at inference, and temperature annealing.
"""

import pytest
import torch

from tgtp.models.pruning_module import TextGuidedPruner

B, N, D = 2, 64, 32
M = 16


@pytest.fixture()
def pruner() -> TextGuidedPruner:
    torch.manual_seed(0)
    return TextGuidedPruner(visual_dim=D, text_dim=D, hidden_dim=32, num_heads=4, keep_ratio=0.5)


@pytest.fixture()
def inputs():
    torch.manual_seed(1)
    visual = torch.randn(B, N, D)
    text = torch.randn(B, M, D)
    mask = torch.ones(B, M, dtype=torch.bool)
    return visual, text, mask


def test_forward_shapes(pruner, inputs):
    visual, text, mask = inputs
    pruned, keep_indices, scores = pruner(visual, text, mask, training=False)
    k = max(1, int(round(N * pruner.keep_ratio)))
    assert pruned.shape == (B, k, D)
    assert keep_indices.shape == (B, k)
    assert scores.shape == (B, N)
    # Indices must be sorted (spatial order preserved) and within range.
    assert torch.all(keep_indices[0, 1:] >= keep_indices[0, :-1])
    assert keep_indices.min() >= 0 and keep_indices.max() < N


def test_scores_in_unit_range(pruner, inputs):
    visual, text, mask = inputs
    _, _, scores = pruner(visual, text, mask, training=False)
    assert torch.all(scores >= 0.0) and torch.all(scores <= 1.0)


def test_keep_ratio_exactness():
    torch.manual_seed(0)
    for keep_ratio, expected_k in [(0.5, 32), (0.25, 16), (0.75, 48), (1.0, 64)]:
        pruner = TextGuidedPruner(D, D, hidden_dim=32, num_heads=4, keep_ratio=keep_ratio)
        visual = torch.randn(1, N, D)
        text = torch.randn(1, M, D)
        pruned, keep_indices, _ = pruner(visual, text, training=False)
        assert pruned.shape[1] == expected_k, f"keep_ratio={keep_ratio}"
        assert keep_indices.shape[1] == expected_k


def test_gradient_flow_through_gumbel(pruner, inputs):
    """The soft-masked surrogate must carry gradients to the scorer."""
    visual, text, mask = inputs
    pruned, keep_indices, scores = pruner(visual, text, mask, training=True)
    assert pruner.last_soft_masked is not None
    assert pruner.last_soft_masked.shape == (B, N, D)
    loss = pruner.last_soft_masked.pow(2).mean()
    loss.backward()
    scorer = pruner.scorer
    assert scorer.visual_proj.weight.grad is not None
    assert scorer.text_key_proj.weight.grad is not None
    assert scorer.visual_proj.weight.grad.abs().sum() > 0
    assert scores.requires_grad


def test_training_selection_uses_ste(pruner, inputs):
    """During training the mask is one-hot (hard selection) despite soft gradients."""
    visual, text, mask = inputs
    _, _, scores = pruner(visual, text, mask, training=True)
    mask_hard, idx = pruner.selection_mask(scores, training=False)
    assert torch.all(mask_hard.sum(dim=-1) == mask_hard.shape[1] // 2)  # K = 32
    assert idx.shape == (B, 32)


def test_hard_topk_at_inference_is_deterministic(pruner, inputs):
    visual, text, mask = inputs
    _, idx1, _ = pruner(visual, text, mask, training=False)
    _, idx2, _ = pruner(visual, text, mask, training=False)
    assert torch.equal(idx1, idx2)


def test_temperature_annealing():
    pruner = TextGuidedPruner(D, D, hidden_dim=32, num_heads=4,
                              temperature_init=5.0, temperature_min=0.1,
                              temperature_anneal_steps=100)
    t0 = pruner.get_temperature()
    pruner.anneal_temperature(50)
    t1 = pruner.get_temperature()
    pruner.anneal_temperature(200)
    t2 = pruner.get_temperature()
    assert t0 == 5.0
    assert t1 < t0
    assert abs(t2 - 0.1) < 1e-3  # annealed to min


def test_merge_strategy_returns_valid_tokens():
    torch.manual_seed(0)
    pruner = TextGuidedPruner(D, D, hidden_dim=32, num_heads=4, keep_ratio=0.5,
                              merge_strategy="merge")
    visual = torch.randn(1, N, D)
    text = torch.randn(1, M, D)
    pruned, keep_indices, _ = pruner(visual, text, training=False)
    assert pruned.shape == (1, 32, D)
    assert torch.isfinite(pruned).all()


def test_score_with_method_random_range():
    scores = TextGuidedPruner.score_with_method("random", torch.randn(2, 20, D), torch.randn(2, 5, D))
    assert scores.shape == (2, 20)
    assert torch.all(scores >= 0) and torch.all(scores < 1)
