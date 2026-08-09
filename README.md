# Text-Guided Visual Token Pruning for Efficient Multimodal Large Language Models

**tgtp** is a research repository for a Master's thesis that asks a single question: *can a lightweight, text-conditioned pruning module decide which visual tokens a Multimodal Large Language Model (MLLM) actually needs, before those tokens ever reach the LLM decoder?*

MLLMs waste enormous compute on redundant background tokens. A 384×384 image produces 729 SigLIP patch tokens (27×27 grid); an LLaVA-NeXT image produces up to 2,880 tokens, and every one of them is processed by every layer of the LLM decoder. We inject a **TextGuidedPruner** between the vision connector and the LLM decoder that scores each visual token against the *text prompt* via cross-attention, keeps the top-K most relevant tokens, and drops the rest — cutting inference FLOPs by up to ~50% while holding VQA accuracy within ~2 pp of the full-token baseline.

```
┌─────────┐   ┌──────────────┐   ┌────────────┐   ┌────────────────────────┐   ┌──────────────┐   ┌────────┐
│  Image  │──▶│ ViT (SigLIP) │──▶│  Connector │──▶│ [TextGuidedPruner]  ═══▶│ LLM Decoder  │──▶│ Answer │
│ (384²)  │   │ 729 patches  │   │ (project)  │   │ (cross-attn scoring)│   │ (SmolLM2 /   │   │        │
└─────────┘   └──────────────┘   └────────────┘   │  keep top-K of N     │   │  Vicuna-7B)  │   └────────┘
                                                    └─────────▲──────────┘   └──────────────┘
                                                              │
                                         Text Prompt ─────────┘
                                       (tokens condition the
                                        pruning decision)
```

> **CRITICAL NOTE — Pruning location.**
> Pruning **MUST** happen between the connector and the LLM decoder (pre-decoder).
> It must **NEVER** be done on `model.model(**inputs).last_hidden_state` (post-decoder).
> That tensor is the output of the *entire* LLM — every token has been fully contextualized by self-attention through all layers, so cosine similarity computed there is meaningless for identifying "which visual tokens the question needs." Empirically, pruning from post-decoder representations produces **gibberish**; pruning at the correct pre-decoder location produces correct answers (see below).

## Empirical evidence (the foundation of this thesis)

Sanity check on **SmolVLM-Instruct**, 1 image, 3 questions:

| Condition | Q1: "What color is the car?" | Q2: "What is the weather?" | Q3: "Any people on the street?" |
|---|---|---|---|
| 🟢 Baseline (100% tokens) | ✅ "The car is turquoise." | ✅ "It is sunny in the background." | ✅ "No, there are no people..." |
| 🔴 Random pruning (50%) | ✅ correct | ✅ correct | ✅ correct |
| 🔵 "Smart" pruning at **wrong location** (post-decoder, 50%) | ❌ gibberish | ❌ gibberish | ❌ gibberish |

Two conclusions:
1. **Random 50% pruning works.** The LLM decoder is highly robust to visual token removal — massive redundancy exists.
2. **The "smart" failure was an implementation bug**, not a concept failure. The scores were computed on fully-contextualized post-decoder hidden states. The correct implementation scores tokens at the **input** to the LLM decoder (connector output), exactly as FastV, PyramidDrop, LLaVA-PruMerge, and VisionZip do in their respective locations.

This is the empirical foundation for the entire thesis: *redundancy exists, random is a strong baseline, and text-conditioned scoring at the correct location is the proposed differentiator.*

## Installation

```bash
cd text-guided-token-pruning
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # core deps + pytest/ruff
# GPU users: install the CUDA build of torch first (see https://pytorch.org)
```

## Quick start

```bash
# 1. Cache visual tokens for a small benchmark slice (no GPU needed after this for scoring)
python scripts/01_extract_visual_tokens.py --model HuggingFaceTB/SmolVLM-Instruct \
    --dataset vqav2 --split validation --max-samples 100 --output-dir cached_features

# 2. Train the pruning module (distillation from full-token teacher)
python scripts/02_train_pruning_module.py --keep-ratio 0.5 --num-epochs 2 \
    --output-dir outputs/pruner_50pct

# 3. Run inference with the trained pruner
python scripts/03_run_inference.py --prune-ratio 0.5 --scoring learned \
    --output-dir outputs/results

# 4. Evaluate and generate the comparison report
python scripts/04_evaluate.py --results-dir outputs/results --report outputs/report.md
```

Random baseline (mandatory in every experiment):

```bash
python scripts/07_random_pruning_baseline.py --model HuggingFaceTB/SmolVLM-Instruct \
    --prune-ratios 0.25 0.5 0.75 --benchmark vqav2 --max-samples 100
```

Head-to-head baselines (mandatory since the 2026-08 literature audit — see
`docs/thesis/proposal.md` §2.7): **FastV** (in-decoder) and **SparseVLM**
(text-guided, in-decoder) are compared at matched token budgets using their
public code (weeks 3–4 of the plan). Runners wrap the official repos:

```bash
# FastV (LLaVA-1.5-7B; needs FastV's own patched-transformers env)
python scripts/08_run_fastv_baseline.py --clone --repo-path third_party/FastV \
    --python /path/to/fastv_env/bin/python --benchmark textvqa --keep-ratio 0.5

# SparseVLM (LLaVA-1.5-7B; run inside the SparseVLMs conda env)
python scripts/09_run_sparsevlm_baseline.py --clone --repo-path third_party/SparseVLMs \
    --benchmark gqa --keep-ratio 0.33

# inspect commands first with --dry-run
```

Both baselines are LLaVA-1.5-only, so the head-to-head comparison runs on the
LLaVA-1.5-7B (576-token) model — see the pre-registration P2/P3 and the
`baseline_fastv.yaml` / `baseline_sparsevlm.yaml` configs.

Colab: the `notebooks/` folder is Colab-ready; `notebooks/00_sanity_check_colab.ipynb` is the recommended entry point. (Colab guide: `docs/colab_guide.md`.)

## Repository structure

```
text-guided-token-pruning/
├── configs/            # YAML configs: models, datasets, experiments
├── src/tgtp/
│   ├── models/         # TextGuidedPruner, CrossAttentionScorer, TokenMerger, SmolVLM wrapper
│   ├── data/           # VQA loaders + image processing (+ token caching)
│   ├── evaluation/     # VQA metrics, efficiency metrics, D1–D7 diagnostics
│   └── utils/          # logging + visualization
├── scripts/            # 01..09: extract → train → infer → evaluate → ablate → report → random baseline → FastV → SparseVLM
├── notebooks/          # Colab-ready sanity checks and analysis
├── tests/              # pytest suite (run: pytest tests/ -v)
├── docs/               # thesis proposal, pre-registration, guides
└── data/               # download scripts + .gitkeep
```

## Thesis documents

- `docs/thesis/proposal.md` — defense-ready thesis proposal (problem, related work + literature matrix, RQs, method, evaluation protocol, diagnostics, timeline, risks, pre-registration).
- `docs/pre_registration.md` — frozen hypotheses, statistical plan, success/failure criteria, anti-pattern registration.
- `docs/colab_guide.md` — step-by-step Colab reproduction guide.

## Citation

If you use this repository in your work, please cite the thesis (placeholder until defense):

```bibtex
@misc{tgtp2026,
  title  = {Text-Guided Visual Token Pruning for Efficient Multimodal Large Language Models},
  author = {Toufiq, M.},
  year   = {2026},
  note   = {MSc thesis, under review. Repository: text-guided-token-pruning}
}
```

## License

Apache-2.0. See [LICENSE](LICENSE).
