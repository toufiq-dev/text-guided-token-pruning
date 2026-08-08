"""tgtp: Text-Guided Visual Token Pruning for Efficient Multimodal Large Language Models.

The central claim of this package (and thesis):

    Visual tokens can be pruned BEFORE they enter the LLM decoder (pre-decoder),
    guided by cross-attention between visual tokens and the text prompt, cutting
    inference FLOPs by ~40-50% while keeping VQA accuracy within ~2%.

The single most important design rule, encoded throughout this package:

    NEVER prune from ``model.model(**inputs).last_hidden_state`` (post-decoder).
    Those representations are fully contextualized by the LLM and are useless for
    deciding which visual tokens the question needs. Prune at the connector output.
"""

__version__ = "0.1.0"

from tgtp.models.cross_attention_scorer import (
    CrossAttentionScorer,
    cosine_score,
    cross_attention_score,
)
from tgtp.models.pruning_module import TextGuidedPruner
from tgtp.models.smolvlm_wrapper import PRUNING_LOCATION, SmolVLMPruningWrapper
from tgtp.models.token_merger import TokenMerger

__all__ = [
    "PRUNING_LOCATION",
    "CrossAttentionScorer",
    "SmolVLMPruningWrapper",
    "TextGuidedPruner",
    "TokenMerger",
    "__version__",
    "cosine_score",
    "cross_attention_score",
]
