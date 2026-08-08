"""Unit tests for tgtp.models.smolvlm_wrapper.

The critical contract under test:

    extract_visual_tokens() returns PRE-DECODER tokens (vision_model + connector)
    and never touches the text model; post-decoder pruning is rejected loudly.
"""

import pytest
import torch

from tgtp.models.smolvlm_wrapper import PRUNING_LOCATION, SmolVLMPruningWrapper, make_mock_backend

B = 2
VISION_TOKENS = 16
VISION_DIM = 8
LLM_DIM = 16
M = 8
VOCAB = 64


@pytest.fixture()
def wrapper() -> SmolVLMPruningWrapper:
    return SmolVLMPruningWrapper(make_mock_backend(vision_tokens=VISION_TOKENS, vocab=VOCAB))


@pytest.fixture()
def inputs():
    torch.manual_seed(0)
    pv = torch.randn(B, 3, 32, 32)
    input_ids = torch.randint(0, VOCAB, (B, M))
    mask = torch.ones(B, M, dtype=torch.bool)
    return pv, input_ids, mask


def test_extract_visual_tokens_returns_pre_decoder_tokens(wrapper, inputs):
    """Extraction must equal vision_model -> connector output (pre-decoder)."""
    pv, _, _ = inputs
    tokens = wrapper.extract_visual_tokens(pv)
    assert tokens.shape == (B, VISION_TOKENS, LLM_DIM)

    # Recompute manually to prove it is the connector output.
    with torch.no_grad():
        feats = wrapper.vision_model(pixel_values=pv).last_hidden_state
        expected = wrapper.connector(feats)
    assert torch.allclose(tokens, expected)


def test_extraction_never_touches_text_model(wrapper, inputs):
    """Prove extract_visual_tokens is NOT post-decoder: the text model must never run."""
    pv, _, _ = inputs
    calls = {"n": 0}

    def _boom(*a, **k):
        calls["n"] += 1
        raise AssertionError("text_model.forward must not be called during extraction")

    wrapper.text_model.forward = _boom  # type: ignore[method-assign]
    tokens = wrapper.extract_visual_tokens(pv)
    assert calls["n"] == 0
    assert tokens.shape == (B, VISION_TOKENS, LLM_DIM)


def test_post_decoder_pruning_is_forbidden():
    with pytest.raises(RuntimeError, match="post-decoder pruning is forbidden"):
        SmolVLMPruningWrapper.prune_from_last_hidden_state(torch.randn(1, 5, 16))


def test_init_requires_backend_contract():
    class NoBackend:
        pass

    with pytest.raises(ValueError, match="Expected an Idefics3ForConditionalGeneration"):
        SmolVLMPruningWrapper(NoBackend())

    class PartialBackend:
        class model:
            vision_model = object()

    with pytest.raises(ValueError, match="connector"):
        SmolVLMPruningWrapper(PartialBackend())


def test_prune_and_generate_shapes(wrapper, inputs):
    pv, input_ids, mask = inputs
    hidden, keep_indices, scores = wrapper.prune_and_generate(
        pv, input_ids, mask, keep_ratio=0.5, scoring_method="cross_attention"
    )
    k = VISION_TOKENS // 2
    assert hidden.shape == (B, k + M, LLM_DIM)
    assert keep_indices.shape == (B, k)
    assert scores.shape == (B, VISION_TOKENS)
    assert torch.all(scores >= 0) and torch.all(scores <= 1)


def test_select_tokens_keep_ratio(wrapper):
    torch.manual_seed(0)
    visual = torch.randn(B, VISION_TOKENS, LLM_DIM)
    scores = torch.rand(B, VISION_TOKENS)
    for ratio, expected in [(0.25, 4), (0.5, 8), (0.75, 12)]:
        pruned, idx = wrapper.select_tokens(visual, scores, ratio)
        assert pruned.shape == (B, expected, LLM_DIM)
        assert idx.shape == (B, expected)
        assert torch.all(idx[0, 1:] >= idx[0, :-1])


def test_random_scoring_is_valid(wrapper, inputs):
    pv, input_ids, mask = inputs
    visual = wrapper.extract_visual_tokens(pv)
    scores = wrapper.score(visual, input_ids, mask, scoring_method="random")
    assert scores.shape == (B, VISION_TOKENS)
    _, idx = wrapper.select_tokens(visual, scores, 0.5)
    assert torch.all(idx >= 0) and torch.all(idx < VISION_TOKENS)


def test_generate_with_pruning_returns_ids(wrapper, inputs):
    pv, input_ids, mask = inputs
    gen_ids, keep_indices, scores = wrapper.generate_with_pruning(
        pv, input_ids, mask, keep_ratio=0.5, scoring_method="cosine", max_new_tokens=4
    )
    assert gen_ids.shape == (B, M + 4)
    assert gen_ids.dtype == torch.long
    assert keep_indices.shape == (B, VISION_TOKENS // 2)


def test_generate_full_requires_native_generate(wrapper, inputs):
    pv, input_ids, mask = inputs
    with pytest.raises(RuntimeError, match="model.generate not available"):
        wrapper.generate_full(pv, input_ids, mask)


def test_sanity_check_attributes(wrapper):
    attrs = wrapper.sanity_check_attributes()
    assert set(attrs) == {"vision_model", "connector", "text_model", "pruning_location"}
    assert attrs["pruning_location"] == PRUNING_LOCATION == "pre_decoder"


def test_cosine_vs_cross_attention_scores_differ(wrapper, inputs):
    pv, input_ids, mask = inputs
    visual = wrapper.extract_visual_tokens(pv)
    ca = wrapper.score(visual, input_ids, mask, scoring_method="cross_attention")
    co = wrapper.score(visual, input_ids, mask, scoring_method="cosine")
    assert ca.shape == co.shape == (B, VISION_TOKENS)
    assert not torch.allclose(ca, co)
