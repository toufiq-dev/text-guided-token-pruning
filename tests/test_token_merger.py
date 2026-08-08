"""Unit tests for tgtp.models.token_merger.TokenMerger."""

import pytest
import torch

from tgtp.models.token_merger import TokenMerger

B, N, D = 2, 32, 16


@pytest.fixture()
def tokens():
    torch.manual_seed(0)
    return torch.randn(B, N, D)


def _keep_indices(k: int) -> torch.Tensor:
    return torch.arange(k, dtype=torch.long).unsqueeze(0).repeat(B, 1)


def test_drop_keeps_only_kept_tokens(tokens):
    k = 12
    merged = TokenMerger("drop")(tokens, _keep_indices(k))
    assert merged.shape == (B, k, D)
    assert torch.equal(merged, tokens[:, :k])


def test_merge_shapes_and_finite(tokens):
    k = 12
    merged = TokenMerger("merge")(tokens, _keep_indices(k))
    assert merged.shape == (B, k, D)
    assert torch.isfinite(merged).all()


def test_merge_with_no_dropped_tokens_is_identity(tokens):
    """If K == N nothing is dropped; centroids reduce to the original tokens."""
    k = N
    merged = TokenMerger("merge")(tokens, _keep_indices(k))
    assert torch.allclose(merged, tokens, atol=1e-6)


def test_merge_differs_from_drop(tokens):
    k = 8
    dropped = TokenMerger("drop")(tokens, _keep_indices(k))
    merged = TokenMerger("merge")(tokens, _keep_indices(k))
    assert not torch.allclose(dropped, merged)


def test_merge_with_scores_is_stable(tokens):
    k = 8
    scores = torch.rand(B, N)
    merged = TokenMerger("merge")(tokens, _keep_indices(k), scores)
    assert merged.shape == (B, k, D)
    assert torch.isfinite(merged).all()


def test_unknown_strategy_raises():
    with pytest.raises(AssertionError):
        TokenMerger("bogus")
