"""SmolVLM/Idefics3 wrapper for correct visual token extraction and pruning.

CRITICAL LESSONS FROM EXPERIMENTAL TESTING (encoded as code):

1. DO NOT use ``model.model(**inputs).last_hidden_state`` for pruning decisions.
   That tensor is the OUTPUT of the entire LLM decoder. Every token has been
   transformed by causal self-attention with every other token through all
   decoder layers, so at that stage there is no "pure visual token" left to
   score. Cosine similarity computed there is meaningless, and pruning on that
   signal destroys the coherent sequence structure -> GIBBERISH ANSWERS.

2. DO extract vision tokens from ``model.model.vision_model(pixel_values)``,
   project them through ``model.model.connector``, and compute text-vision
   relevance BEFORE the LLM decoder sees them.

3. Random pruning works surprisingly well. Always include it as a baseline.

Verified architecture (HuggingFaceTB/SmolVLM-Instruct, Aug 2026):

    pixel_values -> vision_model (SigLIP ViT-SO400M, 27x27=729 patches at 384px)
                 -> connector (pixel-shuffle, scale 3) -> 9x9 = 81 tokens
                 -> [PRUNE HERE]  (pre-decoder)
                 -> text_model (SmolLM2, Llama-style decoder, hidden 2048, 24 layers)

The wrapper is deliberately Idefics3-specific and mock-friendly: it only touches
the three submodule paths listed above, so unit tests can drive it with a small
fake model (see tests/test_smolvlm_wrapper.py).
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from tgtp.models.cross_attention_scorer import cosine_score, cross_attention_score

PRUNING_LOCATION = "pre_decoder"  # the only location at which pruning is valid


def _check_pre_decoder_inputs() -> None:
    """Fail loudly if anyone attempts post-decoder pruning."""
    assert PRUNING_LOCATION == "pre_decoder", "PRUNING_LOCATION must never be changed"


class SmolVLMPruningWrapper:
    """Wraps SmolVLM-Instruct (Idefics3ForConditionalGeneration) and exposes the
    correct pre-decoder extraction points for visual token pruning.

    Architecture contract::

        pixel_values -> vision_model -> connector -> [PRUNE HERE] -> text_model -> output

    WRONG approach (produces gibberish)::

        model.model(**inputs).last_hidden_state -> prune -> ???
        (post-decoder: all representations are already mixed by the LLM)
    """

    def __init__(self, model: object) -> None:
        _check_pre_decoder_inputs()
        if not hasattr(model, "model"):
            raise ValueError(
                "Expected an Idefics3ForConditionalGeneration (or a mock with a "
                "`model` attribute exposing vision_model / connector / text_model)."
            )
        backend = model.model
        for attr in ("vision_model", "connector", "text_model"):
            if not hasattr(backend, attr):
                raise ValueError(
                    f"model.model.{attr} not found. The wrapper requires the Idefics3 "
                    "pre-decoder contract; if you are using a different MLLM, write a "
                    "thin adapter exposing the same three attributes (pre-decoder!)."
                )
        self.model = model
        self.vision_model = backend.vision_model
        self.connector = backend.connector
        self.text_model = backend.text_model

    # ------------------------------------------------------------ extraction
    def extract_visual_tokens(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract visual tokens at the CORRECT location (pre-decoder).

        Args:
            pixel_values: (B, C, H, W) input images.

        Returns:
            visual_tokens: (B, N, D_llm) connector output — ready for pruning.
        """
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        image_features = vision_outputs.last_hidden_state
        image_features = self.connector(image_features)
        return image_features

    def extract_text_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Text token embeddings used as keys for cross-attention scoring.

        Args:
            input_ids: (B, M) tokenized prompt.

        Returns:
            text_embeds: (B, M, D_llm).
        """
        return self.text_model.get_input_embeddings()(input_ids)

    # ------------------------------------------------------------- scoring
    def _cross_attention_score(
        self,
        visual_tokens: torch.Tensor,
        text_embeds: torch.Tensor,
        text_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-visual-token relevance to the prompt (softmax cross-attention)."""
        return cross_attention_score(visual_tokens, text_embeds, text_attention_mask, aggregation="max")

    def _cosine_score(
        self,
        visual_tokens: torch.Tensor,
        text_embeds: torch.Tensor,
        text_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-visual-token cosine similarity to the mean text embedding ([0,1])."""
        return cosine_score(visual_tokens, text_embeds, text_attention_mask)

    def _random_score(self, bsz: int, num_tokens: int, device: torch.device) -> torch.Tensor:
        return torch.rand(bsz, num_tokens, device=device)

    def score(
        self,
        visual_tokens: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        scoring_method: str = "cross_attention",
    ) -> torch.Tensor:
        """Score visual tokens with the requested method.

        Args:
            visual_tokens: (B, N, D) pre-decoder visual tokens.
            input_ids: (B, M) tokenized prompt.
            attention_mask: optional (B, M) padding mask.
            scoring_method: 'cross_attention' | 'cosine' | 'random'.

        Returns:
            scores: (B, N) in [0, 1].
        """
        if scoring_method == "random":
            return self._random_score(visual_tokens.shape[0], visual_tokens.shape[1], visual_tokens.device)
        text_embeds = self.extract_text_tokens(input_ids)
        if scoring_method == "cross_attention":
            return self._cross_attention_score(visual_tokens, text_embeds, attention_mask)
        if scoring_method == "cosine":
            return self._cosine_score(visual_tokens, text_embeds, attention_mask)
        raise ValueError(f"Unknown scoring method: {scoring_method}")

    # ------------------------------------------------------------ selection
    def select_tokens(
        self, visual_tokens: torch.Tensor, scores: torch.Tensor, keep_ratio: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Keep the top-K visual tokens (K = round(N * keep_ratio)), in spatial order.

        Returns:
            (pruned_visual (B, K, D), keep_indices (B, K) sorted).
        """
        num_tokens = visual_tokens.shape[1]
        k = max(1, int(round(num_tokens * keep_ratio)))
        _, keep_indices = torch.topk(scores, k, dim=-1)
        keep_indices, _ = torch.sort(keep_indices, dim=-1)
        batch_indices = torch.arange(visual_tokens.shape[0], device=visual_tokens.device)
        pruned_visual = visual_tokens[batch_indices.unsqueeze(1), keep_indices]  # (B, K, D)
        return pruned_visual, keep_indices

    # ---------------------------------------------------------- main pipeline
    def prune_and_generate(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        keep_ratio: float = 0.5,
        scoring_method: str = "cross_attention",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prune visual tokens at the correct pre-decoder location and run the decoder.

        This is the CORRECT pipeline:

        1. Extract visual tokens from vision_model + connector   (pre-decoder)
        2. Score them against text embeddings via cross-attention
        3. Keep top-K tokens (spatial order preserved)
        4. Concatenate pruned visual tokens with text embeddings
        5. Feed the sequence to the frozen LLM decoder

        Returns:
            (output_hidden (B, K+M, D), keep_indices (B, K), scores (B, N)).
        """
        visual_tokens = self.extract_visual_tokens(pixel_values)  # (B, N, D)
        scores = self.score(visual_tokens, input_ids, attention_mask, scoring_method)
        pruned_visual, keep_indices = self.select_tokens(visual_tokens, scores, keep_ratio)
        text_embeds = self.extract_text_tokens(input_ids)  # (B, M, D)

        seq = torch.cat([pruned_visual, text_embeds], dim=1)
        seq_mask = torch.ones(seq.shape[:2], dtype=attention_mask.dtype, device=seq.device)
        outputs = self.text_model(inputs_embeds=seq, attention_mask=seq_mask)
        return outputs.last_hidden_state, keep_indices, scores

    def generate_with_pruning(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        keep_ratio: float = 0.5,
        scoring_method: str = "cross_attention",
        max_new_tokens: int = 16,
        temperature: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Greedy/argmax generation on the pruned sequence.

        Returns:
            (generated_ids (B, M + gen) including the prompt, keep_indices, scores).
        """
        visual_tokens = self.extract_visual_tokens(pixel_values)
        scores = self.score(visual_tokens, input_ids, attention_mask, scoring_method)
        pruned_visual, keep_indices = self.select_tokens(visual_tokens, scores, keep_ratio)
        text_embeds = self.extract_text_tokens(input_ids)

        gen_ids = input_ids.clone()
        seq = torch.cat([pruned_visual, text_embeds], dim=1)
        for _ in range(max_new_tokens):
            seq_mask = torch.ones(seq.shape[:2], dtype=attention_mask.dtype, device=seq.device)
            outputs = self.text_model(inputs_embeds=seq, attention_mask=seq_mask)
            logits = getattr(outputs, "logits", None)
            if logits is None:
                raise RuntimeError(
                    "text_model.forward did not return logits; cannot generate. "
                    "Use prune_and_generate() and decode externally."
                )
            next_logits = logits[:, -1, :]  # (B, V)
            if temperature > 0.0:
                probs = F.softmax(next_logits / temperature, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1)
            else:
                next_ids = next_logits.argmax(dim=-1, keepdim=True)
            gen_ids = torch.cat([gen_ids, next_ids], dim=1)
            seq = torch.cat([seq, self.text_model.get_input_embeddings()(next_ids)], dim=1)
        return gen_ids, keep_indices, scores

    def generate_full(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 16,
    ) -> torch.Tensor:
        """Full-token baseline generation through the native model.generate.

        Used only for the no-pruning baseline (keep_ratio == 1.0). Raises if the
        backend does not expose generate (e.g. mocks in unit tests).
        """
        if not hasattr(self.model, "generate"):
            raise RuntimeError("model.generate not available (mock backend).")
        return self.model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
        )

    # ------------------------------------------------- anti-pattern guard
    @staticmethod
    def prune_from_last_hidden_state(*args: object, **kwargs: object) -> None:
        """Anti-pattern guard: post-decoder pruning is forbidden.

        ``model.model(**inputs).last_hidden_state`` is the output of the ENTIRE
        LLM decoder. All tokens are fully contextualized there; there is no clean
        visual-token signal left to score, and pruning on that representation
        empirically produces gibberish. This method exists only to raise a clear
        error if it is ever called.
        """
        raise RuntimeError(
            "post-decoder pruning is forbidden: last_hidden_state from a full model "
            "forward is fully contextualized by the LLM and useless for token "
            "selection. Use extract_visual_tokens(pixel_values) (pre-decoder) instead."
        )

    def sanity_check_attributes(self) -> dict:
        """Return the attribute contract as a dict (used by tests + D6 diagnostics)."""
        return {
            "vision_model": self.vision_model.__class__.__name__,
            "connector": self.connector.__class__.__name__,
            "text_model": self.text_model.__class__.__name__,
            "pruning_location": PRUNING_LOCATION,
        }


def make_mock_backend(
    vision_tokens: int = 16,
    vision_dim: int = 8,
    llm_dim: int = 16,
    vocab: int = 64,
    seed: int = 0,
) -> object:
    """Build a small mock Idefics3-style backend for tests and quick demos.

    Mirrors the pre-decoder contract: model.model.vision_model / connector /
    text_model, where text_model has get_input_embeddings() and a forward that
    accepts inputs_embeds and returns last_hidden_state + logits.
    """
    torch.manual_seed(seed)

    class _VisionModel(torch.nn.Module):
        """Deterministic mock: a linear map from flattened pixels to patch tokens."""

        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(3 * 32 * 32, vision_tokens * vision_dim)

        def forward(self, pixel_values: torch.Tensor) -> SimpleNamespace:
            bsz = pixel_values.shape[0]
            out = self.proj(pixel_values.flatten(1)).view(bsz, vision_tokens, vision_dim)
            return SimpleNamespace(last_hidden_state=out)

    class _Connector(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(vision_dim, llm_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.proj(x)

    class _TextModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = torch.nn.Embedding(vocab, llm_dim)
            self.norm = torch.nn.LayerNorm(llm_dim)
            self.lm_head = torch.nn.Linear(llm_dim, vocab)

        def get_input_embeddings(self) -> torch.nn.Module:
            return self.embed

        def forward(
            self,
            inputs_embeds: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
        ) -> SimpleNamespace:
            out = self.norm(inputs_embeds)
            return SimpleNamespace(last_hidden_state=out, logits=self.lm_head(out))

    class _Backend:
        pass

    backend = _Backend()
    backend.vision_model = _VisionModel()
    backend.connector = _Connector()
    backend.text_model = _TextModel()

    class _MockModel:
        pass

    mock = _MockModel()
    mock.model = backend
    return mock
