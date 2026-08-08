# Licenses and Attribution

## This repository

- **License:** Apache-2.0 (see `LICENSE`). You may use, modify, and distribute this
  code freely, provided you retain the license notice and attribute the source.
- **Code origin:** written for the MSc thesis *Text-Guided Visual Token Pruning for
  Efficient Multimodal Large Language Models*.

## Third-party software used

| Component | License | Notes |
|---|---|---|
| PyTorch (`torch`) | BSD-3-Clause | ML framework |
| HuggingFace `transformers`, `accelerate`, `datasets`, `evaluate` | Apache-2.0 | Model/processor/dataset loading |
| `numpy`, `scipy`, `Pillow`, `tqdm`, `pyyaml`, `pytest`, `ruff` | BSD/MIT/etc. | Standard tooling |

## Model checkpoints (downloaded at runtime, not redistributed)

- **HuggingFaceTB/SmolVLM-Instruct** — open weights (SmolVLM license; see the HF
  model card). Used for experiments only; results and derived artifacts in this
  repo do not include model weights.
- **llava-hf/llava-1.5-7b-hf** — open weights (LLaVA license; see the HF model
  card).

## Dataset licenses (used via HuggingFace hub at runtime)

- VQAv2, GQA, TextVQA, MMBench, SEED-Bench — see each dataset's hub page for
  licensing terms. We use them for academic evaluation only.

## Citation of prior work

All cited papers are listed with arXiv IDs / venues in
`docs/thesis/proposal.md` (§2.7 literature matrix). Anything marked ⚠️ has not been
independently re-verified at the time of writing.
