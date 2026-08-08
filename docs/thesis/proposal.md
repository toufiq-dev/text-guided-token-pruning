# Text-Guided Visual Token Pruning for Efficient Multimodal Large Language Models

**MSc Thesis Proposal — Version 1.2 (defense-ready draft, audit-revised 2026-08-09)**

*Author: M. Toufiq · Advisor: TBD · Department: Computer Science / AI*

> We chose Multimodal Large Language Models (MLLMs) as the testbed because they
> represent the current frontier of compute waste in vision-language systems, and
> their standard ViT + Connector + LLM architecture provides a clean, reproducible
> testbed for pre-decoder token manipulation. The proposal is motivated by a
> reproducible empirical finding: **randomly dropping 50% of the visual tokens that
> enter the LLM decoder of SmolVLM-Instruct still produces correct VQA answers** —
> proof that massive visual-token redundancy exists. The question this thesis tests
> is whether a *text-conditioned*, *pre-decoder* pruning module trained by
> distillation can beat that random baseline — and the field's published methods
> (FastV, SparseVLM) — at matched token budgets while cutting inference FLOPs by
> 30–50%.

---

## Section 1 — Problem Statement

### 1.1 The compute waste

A Multimodal Large Language Model (MLLM) converts an image into hundreds or thousands
of visual tokens, then runs **every one of them through every layer of the LLM decoder**.
Most of those tokens encode background (sky, walls, grass, empty space) that is
irrelevant to any specific question. The waste grows quadratically with the number
of visual tokens because per-layer self-attention cost is quadratic in sequence
length.

Verified token counts (Aug 2026, from model configs and the literature):

| Model | Vision encoder | Image resolution | Patch tokens (encoder out) | Tokens entering the LLM decoder | Decoder layers | Token-layer ops per image |
|---|---|---|---|---|---|---|
| **SmolVLM-Instruct** (Idefics3) | SigLIP SO400M | 384² | **729** (27×27) | **81** (pixel-shuffle, ×3) | 24 (SmolLM2) | 81 × 24 ≈ 1,944 |
| **LLaVA-1.5-7B** | CLIP ViT-L/14 | 336² | 576 (24×24) | 576 | 32 (Vicuna-7B) | 576 × 32 = 18,432 |
| **LLaVA-NeXT-7B** | CLIP ViT-L/14 (AnyRes) | up to 672² | up to 2,880 | 2,880 | 32 | 2,880 × 32 = 92,160 |

> ⚠️ **Honest correction to our earlier internal notes:** SmolVLM's Idefics3 connector
> uses a *pixel-shuffle* with scale factor 3, compressing the 729 SigLIP patches to
> **81 tokens** (`image_seq_len: 81`) that actually enter the LLM decoder at 384².
> This was verified against `HuggingFaceTB/SmolVLM-Instruct/config.json`. The thesis
> numbers are therefore reported at the *LLM-input* level (81 tokens for SmolVLM).
> This *strengthens* the redundancy claim: even after aggressive compression, further
> 50% random pruning of the tokens the LLM actually processes still yields correct
> answers. For LLaVA-1.5/LLaVA-NeXT — where no such compression exists — the
> bottleneck is 576 / 2,880 tokens respectively, and the savings are far larger.
> LLaVA-NeXT-7B is scoped into the evaluation protocol as a **reduced-slice tertiary
> model** (1,000 samples on GQA + TextVQA; see §5), so the strongest motivating case
> in the table above is actually measured, not merely cited.

Back-of-envelope decoder FLOPs (4·H² per token per layer; see `efficiency_metrics.py`):
SmolVLM ≈ **32.6 GFLOPs/image**, LLaVA-1.5-7B ≈ **1.24 TFLOPs/image**,
LLaVA-NeXT-7B ≈ **6.2 TFLOPs/image**. A 50% reduction in visual tokens cuts these
roughly in half.

### 1.2 Why existing methods fall short

1. **Decoder-side pruning (FastV, PyramidDrop, SparseVLM)** must run *inside* the
   autoregressive generation loop and requires access to LLM attention weights (or
   in-decoder attention matrices); it cannot reduce the KV-cache size before
   generation starts.
2. **Training-free heuristics (LLaVA-PruMerge, VisionZip, ToMe)** score tokens with
   *text-agnostic* signals (encoder attention, pairwise similarity). They cannot
   condition on the question — a background token that is useless for "what color is
   the car?" may be decisive for "is it raining?".
3. **Projector compression (TokenPacker)** replaces the connector and requires
   retraining the projector; it is not a plug-in and is not text-conditioned.

### 1.3 The empirical foundation (our finding)

On SmolVLM-Instruct (1 image, 3 questions):

| Condition | Q1 color | Q2 weather | Q3 people |
|---|---|---|---|
| 🟢 Baseline (100% tokens) | ✅ | ✅ | ✅ |
| 🔴 Random pruning (50%) | ✅ | ✅ | ✅ |
| 🔵 "Smart" pruning at **wrong location** (post-decoder) | ❌ gibberish | ❌ gibberish | ❌ gibberish |

Two conclusions:
- **Redundancy is real.** The decoder is robust to 50% visual-token removal, so a
  "smart" method must justify itself against a random baseline at matched budgets.
- **Location matters.** Scoring tokens on *post-decoder* `last_hidden_state` (fully
  contextualized by the LLM) produces meaningless selections and gibberish. The
  correct extraction point is the **connector output, before the LLM decoder**.

### 1.4 Anti-pattern (registered)

**Never** prune from `model.model(**inputs).last_hidden_state`. This is the single
most common implementation failure in this area (and the root cause of the
gibberish failure documented in §1.3). The codebase encodes this as a hard guard
(`SmolVLMPruningWrapper.prune_from_last_hidden_state` raises; diagnostic D6).

---

## Section 2 — Related Work

### 2.1 Training-free decoder-side pruning

**FastV** (Chen et al., ECCV 2024, arXiv:2403.06764): after layer 2 of the LLM,
prune visual tokens whose attention to text tokens is low. Training-free,
text-conditioned, ~45% FLOPs reduction on LLaVA-1.5-13B with <1pp drop at 50%
pruning; they show random pruning degrades far more. *Limitation:* lives inside the
generation loop; needs attention access; KV-cache is already growing by the time
pruning starts.

**PyramidDrop** (Xing et al., CVPR 2025, arXiv:2410.17247): stage-wise progressive
token dropping driven by token-token similarity. >55% FLOPs ↓ and ~40% training-time
↓ on LLaVA-NeXT with ~0.6pp drop; optional training. *Limitation:* not
text-conditioned; hierarchical in-decoder design; cannot be frozen before decoding.

**IVTP** (Huang, Zou, Xi, Wang, Xie & Yu, ECCV 2024; Springer DOI
10.1007/978-3-031-72643-9_13, no arXiv preprint): instruction-guided visual token
pruning via a Group-wise Token Pruning (GTP) module based on attention rollout,
integrated into the grouped transformer layer, with a semantically related pseudo
CLS token as the pruning reference in the LLM stage. Works **without retraining**
(plug-in; verified from the ECCV abstract); 88.9% token reduction → >46% compute ↓
with ~1.0 pp average drop across 12 benchmarks on LLaVA-1.5. *Limitation:* in-decoder
surgery (attention rollout inside the model); random-baseline comparison ⚠️
unverified; no pre-registered diagnostics.

**LVPruning** (Cheng et al., NAACL 2025 Findings, arXiv:2501.13652): auxiliary
cross-attention modules score visual-token importance from their interaction with
language tokens, then prune in the model's middle layers; reported up to 90% token
removal with ~0.45 pp average loss across nine benchmarks; the auxiliary decision
modules are trained while the original MLLM parameters stay frozen; random-pruning
comparison included. *Limitation:* in-decoder (middle-layer attention required);
auxiliary modules sit inside the generation path.

### 2.2 Learned token selection / merging

**LLaVA-PruMerge** (Shang et al., arXiv:2403.15388, **ICCV 2025 ✓**): selects
"unpruned" tokens by visual-encoder attention sparsity and clusters redundant
ones. Up to 14× token compression, near-lossless. Training-free. *Limitation:*
text-agnostic heuristic; no learned scorer.

**FitPruner** (Ye, Wu, Lin & Zhou, **AAAI 2025**, **arXiv:2409.10197**, DOI
10.1609/aaai.v39i21.34366): frames pruning as a *distribution-fitting* problem —
searches an optimal layer-wise pruning recipe that minimizes the divergence of
self- and cross-attention distributions before/after pruning, then executes token
removal inside the attention blocks at inference. Training-free; includes a
`FitPrune(Random)` variant as an explicit baseline. *Limitation:* heuristic; no
trained scorer; text-conditioning is indirect (via cross-attention statistics),
not an explicit question-guidance module.

**SaMer** (Park, Jung, Park & Kang, arXiv:2607.04605, July 2026 preprint, ID
verified — not yet peer-reviewed): *"Do All Visual Tokens Matter Equally?
Object-Evidence Preserving Token Merging for Vision-Language Retrieval"* —
compresses post-projector tokens into K centroids via feature-spatial soft
assignment under an object-aware training prior (Flickr30k-Entities boxes, used
only during training; no detector needed at inference), preserving the MaxSim
late-interaction interface. Tested on ColPali/ColQwen2 with >93% token removal.
*Limitation:* retrieval domain, not generative VQA; random-merging baseline not
reported in main tables ⚠️.

**VisionSelector** (Zhu et al., arXiv:2510.16598, Oct 2025 ⚠️ preprint): a
lightweight learnable scorer (~12.85M params) decoupled from the backbone,
differentiable top-K with curriculum annealing, operating on the pre-decoder
boundary. *Limitation:* text-conditioning is via end-to-end prompt alignment rather
than an explicit cross-attention scoring module ⚠️ (verify the full paper at
defense); no distillation from a full-token teacher; no diagnostics protocol.

**EvoComp** (CVPR 2026, arXiv:2604.17087): a lightweight trained transformer
compressor that selects informative, non-redundant visual tokens using both visual
and textual context, supervised by semantic-guided evolutionary labeling. This is
the closest *trained + text-aware* prior at the pre-decoder boundary. *Limitation:*
supervision is offline evolutionary labeling, not distillation from the full-token
teacher output distribution; the compressor replaces rather than wraps the token
stream; no random-baseline comparison (verified: compares only against
attention/similarity heuristics); no pre-registered diagnostics protocol.

### 2.3 Token compression / packing

**TokenPacker** (Li et al., ECCV 2024, arXiv:2407.02392): replaces the MLP connector
with a coarse-to-fine injection module; 75–89% visual-token compression. *Limitation:*
requires (re)training the projector; text-agnostic; not a plug-in module.

### 2.4 Attention-based efficiency (the efficiency-toolbox family)

**SparQ Attention** (Ribar et al., ICLR 2024, arXiv:2312.04985): query-aware
KV-cache sparsification to cut memory bandwidth for LLM decoding. *Complementary:*
reduces per-token bandwidth; does not reduce token count. Combining SparQ-style KV
sparsity with our token pruning is future work.

**ToMe** (Bolya et al., ICLR 2023, arXiv:2210.09461): bipartite soft token merging
inside ViT blocks (~2× throughput, ~0.3% drop). *Limitation:* visual-only
similarity; no text signal; operates in the encoder, not at the LLM input.

### 2.5 Text-conditioned visual token reduction (the direct competition)

A 2024–2026 lineage does exactly text-guided scoring for pruning — this is the
center of our comparison, not an empty niche:

**SparseVLM** (Zhang et al., ICML 2025, arXiv:2410.04417): training-free. Selects
text tokens relevant to visual tokens as "raters" and reuses decoder self-attention
to rate visual-token significance, then **progressively sparsifies visual tokens
across LLM decoder layers** with a rank-adaptive ratio and token recycling (4.5×
compression at 97% of original accuracy; 37% latency ↓ at only 0.9 pp drop; beats
FastV by 11.2–17.3% at similar latency on LLaVA; verified from the arXiv full
text). *Limitation:* operates **inside the decoder** (requires attention-matrix
access, FlashAttention-compatible extraction); heuristic (training-free) scoring;
no learned scorer, no distillation, no diagnostics.

**CDPruner** (NeurIPS 2025, arXiv:2506.10967): training-free, model-agnostic.
Defines *conditional similarity* of visual tokens given the user instruction and
selects a diverse subset via a determinantal point process (DPP). *Limitation:*
diversity-based rather than relevance-based; no training.

**ResPrune** (arXiv:2603.21105, Mar 2026 ⚠️ preprint): training-free. Frames
pruning as subspace reconstruction — greedy residual-energy expansion of a token
subspace with selection conditioned on textual relevance. *Limitation:* preprint;
heuristic; no trained scorer.

**Nüwa** (arXiv:2602.02951, Feb 2026 ⚠️ preprint): two-stage — a first stage after
the vision encoder preserves spatial integrity (swarm-inspired separation,
alignment, aggregation), and a second stage inside the LLM performs text-guided
pruning. *Limitation:* in-LLM second stage; preprint.

**Our position:** none of this lineage trains an explicit text-conditioned
cross-attention scorer by **distillation from the full-token teacher** at the
strict pre-decoder boundary, and none evaluates on the compressed-token SmolVLM
regime with pre-registered random-baseline-first diagnostics (D1–D7). The closest
*trained* competitors (EvoComp, VisionSelector) are covered in §2.2; see §2.8 for
the honest gap statement.

### 2.6 Random pruning as a baseline

Verified: FastV, PyramidDrop, LLaVA-PruMerge, TokenPacker, ToMe, SparseVLM,
CDPruner, and ALVTS all include random/uniform pruning comparisons, and (where
reported) random degrades more than their method at aggressive ratios — while our
SmolVLM sanity check shows random 50% still answers correctly (i.e., random is
*stronger* than assumed for this architecture). *"All You Need Are Random Visual Tokens? Demystifying Token Pruning
in VLLMs"* (arXiv:2512.07580, CVPR 2026) independently shows that beyond roughly
layer 20, training-free pruning methods are **no better than random** ("vanishing
token information"), and that fusing random pruning with methods like DivPrune
reaches SOTA. This independently supports both our random-first baseline design
(D7) and RQ5 (location sensitivity). **Our pre-registration makes the random
baseline a first-class citizen** (D7).

### 2.7 Literature matrix

| Paper | Year/Venue | Mechanism | Pipeline Location | Training? | Text-Conditioned? | Models | FLOPs ↓ | VQA Acc ↓ | Random Baseline? | My Differentiation |
|---|---|---|---|---|---|---|---|---|---|---|
| FastV (arXiv:2403.06764) | ECCV 2024 | prune low-attention tokens after layer 2 | inside decoder | free | ✅ (attention) | LLaVA-1.5 7B/13B, QwenVL, Video-LLaVA | ~45% (13B) | <1pp @50% | ✅ (random much worse) | pre-decoder, trained scorer, no attention access |
| PyramidDrop (arXiv:2410.17247) | CVPR 2025 | stage-wise similarity dropping | inside decoder | free/opt. | ❌ | LLaVA-1.5, LLaVA-NeXT | >55% + ~40% train ↓ | ~0.6pp | ✅ | text-conditioned pre-decoder |
| LLaVA-PruMerge (arXiv:2403.15388) | ICCV 2025 ✓ | attention-sparsity select + cluster merge | pre-decoder | free | ❌ | LLaVA-1.5 | up to 14× | near-0 | ✅ | learned scorer, text signal |
| FitPruner (arXiv:2409.10197) | AAAI 2025 ✓ | distribution-fitting attention divergence (self+cross) | inside decoder (attn blocks) | free | partial | LLaVA series | significant | minimal | ✅ FitPrune(Random) | distillation-trained scorer |
| TokenPacker (arXiv:2407.02392) | ECCV 2024 | coarse-to-fine projector compression | replaces connector | ✅ retrain | ❌ | LLaVA | 75–89% tokens | ≈0 / + | ✅ | plug-in, post-hoc, text-conditioned |
| SaMer (arXiv:2607.04605 ✓) | 2026 preprint (not peer-reviewed) | object-evidence centroid merging | post-projector (retrieval) | proj-only (train) | ❌ | ColPali, ColQwen2 | >93% tokens | retrieval gains | ⚠️ not in main tables | generative VQA; no object priors at inference |
| ToMe (arXiv:2210.09461) | ICLR 2023 | bipartite soft merging | inside ViT | free | ❌ | ViT/DeiT, VLMs | ~2× throughput | ~0.3% | ✅ | text-conditioned, LLM-input level |
| VisionZip (arXiv:2412.04467) | CVPR 2025 | attention+cluster subset selection | pre-decoder | free | ❌ | LLaVA-1.5/NeXT, Qwen2-VL | up to 8× prefill | SOTA gains | ✅ | learned, distillation, diagnostics |
| SparQ Attention (arXiv:2312.04985) | ICLR 2024 | KV sparsification | in-decoder attention | post-train | n/a | LLMs | bandwidth ↓ | small | n/a | orthogonal (token count) |
| VisionSelector (arXiv:2510.16598) | 2025 ⚠️ preprint (OpenReview ✓) | learnable scorer + diff. top-K | pre-decoder | ✅ | ⚠️ end-to-end alignment | MLLMs | moderate | small | ⚠️ | explicit text cross-attention + full-token distillation + D2/D4/D6 |
| **SparseVLM** (arXiv:2410.04417) | **ICML 2025** | text-token-guided attention rating + progressive sparsification | inside decoder (progressive layers) | free | ✅ | LLaVA-1.5/2, Qwen-VL | 4.5× @ 97% acc; 37% latency ↓ @ 0.9pp | 0.9pp @ 37% latency ↓ | ✅ | pre-decoder, distillation-trained scorer, no attention access |
| **IVTP** (Springer DOI 10.1007/978-3-031-72643-9_13) | **ECCV 2024** | instruction-guided attention-rollout pruning (GTP) | inside decoder | free/opt. (plug-in, no retraining) | ✅ | LLaVA-1.5 | >46% ↓ @ 88.9% tokens | ~1.0pp avg (12 benchmarks) | ⚠️ | strict pre-decoder, no in-layer surgery |
| **LVPruning** (arXiv:2501.13652) | **NAACL 2025 Findings** | aux cross-attention importance, prune middle layers | inside decoder | ✅ aux modules (MLLM frozen) | ✅ | LLaVA-1.5 | up to 90% tokens | ~0.45pp avg | ✅ | pre-decoder + distillation + D2/D4/D6 |
| **EvoComp** (arXiv:2604.17087) | **CVPR 2026** | trained compressor, semantic-guided evolutionary labeling | pre-decoder | ✅ | ✅ | LLaVA, Qwen | high compression | small | ❌ (heuristics only) | distillation from full-token teacher + pre-registered diagnostics |
| **CDPruner** (arXiv:2506.10967) | **NeurIPS 2025** | instruction-conditioned DPP diversity selection | pre-LLM (inference) | free | ✅ | model-agnostic MLLMs | moderate | small | ✅ | relevance-based cross-attention scoring, trained |
| **ResPrune** (arXiv:2603.21105) | 2026 ⚠️ preprint | subspace-reconstruction selection, text-relevance-conditioned | pre-decoder | free | ✅ | LLaVA-1.5/NeXT, Qwen2.5-VL | significant | small | ⚠️ | trained scorer + distillation + diagnostics |
| **Nüwa** (arXiv:2602.02951) | 2026 ⚠️ preprint | two-stage: spatial integrity + text-guided in-LLM | pre-encoder + in-decoder | ⚠️ | ✅ | LVLMs | significant | small | ⚠️ | single-stage strict pre-decoder, no in-LLM stage |
| **One Layer's Trash** (arXiv:2606.14277) | **CVPR 2026** | adaptive layer-wise selection with token reintegration | inside decoder (layer-wise) | free | n/a | LLaVA, Qwen2.5-VL | 89% tokens | retains 96.7% acc | ✅ | static pre-decoder reduction, KV savings from token zero |

### 2.8 Gap analysis (honest statement, verified 2026-08)

1. **Pre-decoder, text-conditioned pruning module?** The individual ingredients
   exist (verified Aug 2026): SparseVLM (ICML 2025) is text-guided but sparsifies
   progressively *inside decoder layers*; EvoComp (CVPR 2026) and VisionSelector
   (2025 ⚠️) are trained and pre-decoder; CDPruner (NeurIPS 2025), ResPrune, and
   Nüwa (2026 ⚠️ preprints) are text-conditioned. **What none of them does** is combine *all* of: (i) strict
   pre-decoder placement — no in-LLM-layer surgery, no generation-loop or attention
   access, full KV-cache savings from token zero; (ii) an explicit
   text-conditioned **cross-attention scoring module trained by distillation from
   the full-token teacher output**; (iii) a pre-registered, random-baseline-first
   protocol with D1–D7 failure diagnostics. The "niche confirmed" phrasing in
   earlier drafts was too strong; the defensible claim is a controlled,
   pre-registered ablation of text-conditioning at the pre-decoder boundary (§5).
2. **Why it is still worth doing:** the field has not run the head-to-head
   comparison that matters — the same pruning ratios, matched token budgets, and
   diagnostics applied to SparseVLM / FastV / IVTP / LVPruning / EvoComp versus a
   distillation-trained pre-decoder scorer — and none of the above has been
   evaluated on the compressed-token SmolVLM regime (81 LLM-input tokens), where
   redundancy claims are *harder* (already compressed) and where our finding that
   random 50% pruning still answers correctly is surprising and under-explored.
3. **Distillation-trained importance scorer?** Still rare. EvoComp uses
   evolutionary labeling; VisionSelector uses end-to-end alignment ⚠️; ours
   additionally distills the full-token teacher output distribution (KL) through a
   soft-masked surrogate gradient path.
4. **Spatial vs. object-recognition sensitivity?** Not systematically separated in
   prior work. We pre-register this (D2/D4) using GQA (spatial) vs. VQAv2
   (recognition) vs. TextVQA (OCR).
5. **Best accuracy–efficiency bar to beat (matched budgets):** FastV <1pp @ ~45%
   FLOPs ↓; PyramidDrop ~0.6pp @ >55% ↓; LVPruning ~0.45pp @ up-to-90% tokens;
   IVTP ~1.0pp @ >46% ↓ (88.9% tokens); SparseVLM 0.9pp @ 37% latency ↓ (4.5×
   compression) — all verified from primary sources (Aug 2026). Our success bar
   (pre-registration §5) is calibrated against these, not against random alone.
6. **Standard benchmarks:** VQAv2, GQA, TextVQA, MMBench, SEED-Bench (LLaVA-1.5-7B
   reference accuracies: VQAv2 ~78.5, GQA ~62.0, TextVQA ~58.2, MMBench ~64.3,
   SEED-Bench ~58.6; ⚠️ source variance across reports).
7. **Random as a strong baseline:** confirmed by our tests, by FastV's own
   comparisons, and independently by arXiv:2512.07580 (CVPR 2026), which shows
   deep-layer training-free pruning ≈ random. Verified.
8. **Wrong-location / late-stage scoring failures:** not systematically documented
   in peer-reviewed work; our D6 + notebook 03 document it. Related 2026 work:
   *"One Layer's Trash is Another Layer's Treasure: Adaptive Layer-wise Visual
   Token Selection in LVLMs"* (arXiv:2606.14277, CVPR 2026) — layer-wise selection
   with token reintegration rather than static hard pruning — and *SwiftVLM*
   (arXiv:2602.03134, cross-layer token bypass). Both bear directly on RQ5.
   (Correction: an earlier draft cited the wrong arXiv ID, 2602.03134, for the
   first of these; the correct ID is 2606.14277.)

---

## Section 3 — Research Questions and Hypotheses

- **RQ1 (Primary).** Can text-guided visual token pruning reduce MLLM decoder FLOPs
  by 30–50% with <2pp VQA accuracy degradation?
  *H1: yes at 25–50% pruning with a distillation-trained scorer; 75% exceeds the
  budget.*
- **RQ2 (Task sensitivity).** How does pruning affect spatial reasoning (GQA) vs.
  object recognition (VQAv2) vs. OCR (TextVQA)?
  *H2: OCR is most fragile (small text patches carry unique information); spatial
  layout survives if grid coverage is preserved (D4).*
- **RQ3 (Text conditioning).** Does text-conditioned pruning beat text-agnostic
  (cosine-to-mean, random) pruning at matched token budgets?
  *H3: yes on VQAv2 (object-centric questions); possibly not on all tasks — D7
  decides.*
- **RQ4 (Scaling).** How does the optimal keep-ratio change with resolution and
  question complexity?
  *H4: higher resolution → larger redundancy → deeper pruning tolerable (D5).*
- **RQ5 (Location sensitivity).** Does pruning location (pre-decoder vs. in-decoder
  vs. post-decoder) affect accuracy?
  *H5: pre-decoder works; post-decoder is invalid (gibberish — our empirical
  finding); in-decoder (FastV) is competitive but requires generation-loop access.*

- **H-fail (pre-registered failure hypothesis).** If text-conditioned pruning does
**not** beat random pruning at matched budgets — or if a training-free text-guided
method (SparseVLM) matches our trained scorer — then visual-token importance is
largely task-independent and the text signal adds no pruning information. **Fallback:**
(a) learned text-agnostic scorer (still distinct from training-free heuristics),
(b) adaptive keep-ratio per image based on a cheap complexity proxy (e.g., entropy
of scores), or (c) reframe the contribution as the rigorous head-to-head + D1–D7
diagnostics comparison the field lacks.

---

## Section 4 — Method

### 4.1 Architecture overview

```
Image ──▶ Frozen ViT ──▶ Connector ──▶ Visual Tokens (B, N, D_v)
                                        │
Text Prompt ──▶ Tokenizer/Embeddings ──▶ Text Embeddings (B, M, D_t)
                                        │
                              [TextGuidedPruner]      ← THE CONTRIBUTION
                              (cross-attention scoring,
                               Gumbel-Softmax training,
                               hard top-K inference)
                                        │
                            Pruned Visual Tokens (B, K, D_v), K = N·keep_ratio
                                        │
                            Frozen LLM Decoder ──▶ Answer
```

**CRITICAL:** the pruner operates on the **connector output (pre-decoder)**. It does
**not** operate on `model.model(**inputs).last_hidden_state` (post-decoder). This is
enforced in code (`PRUNING_LOCATION = "pre_decoder"`; wrapper guards).

### 4.2 Cross-attention scoring

Visual tokens are projected to queries; text tokens to keys/values
(`CrossAttentionScorer`, hidden 256, 8 heads):

```
Q = W_q · V_visual            (B, N, D_h)
K = W_k · T_text              (B, M, D_h)
V = W_v · T_text              (B, M, D_h)
A = softmax( Q·Kᵀ / √d )      (B, N, M)      [padding-masked]
r_i = max_j A_ij              (token relevance; 'mean' ablated)
keep = top-K of r_i           (K = round(N · keep_ratio))
```

Two scoring variants are ablated: **cross-attention max** (proposed) and **cosine to
mean text embedding** (baseline). Scores are normalized to [0, 1].

### 4.3 Differentiable training (Gumbel-Softmax)

The hard top-K gather is non-differentiable, so training uses a **Gumbel-perturbed
straight-through estimator**:

- `z̃_i = r_i/τ + Gumbel(0,1)`; soft weights `w = softmax(z̃)`;
- forward selection is hard top-K; a **soft-masked surrogate**
  `Ṽ = V ⊙ w` provides the gradient path into the scorer;
- temperature anneals **τ: 5.0 → 0.1** (cosine schedule, ~1k steps);
- at inference: hard top-K, no noise, no gradients.

### 4.4 Training objective (distillation)

```
L = L_task + λ_distill·(L_distill_hard + L_distill_soft) + λ_entropy·L_entropy
L_task      = CE(decoder(pruned seq))                     # causal LM loss
L_distill   = KL( P_full ∥ P_pruned )·τ²                  # teacher = full tokens
L_distill_soft = KL( P_full ∥ P_soft-gated )·τ²           # gradient path to scorer
L_entropy   = −Σ p·log p over w                            # prevents score collapse (D3)
```

Teacher: the same frozen model run on the full token sequence. Vision encoder,
connector, and decoder are **frozen**; only the pruner's ≈256²-scale parameters
train. λ_distill = 1.0, λ_entropy = 0.1.

### 4.5 Inference mode

Deterministic hard top-K; no gradient computation; compatible with KV-cache
optimization (fewer visual tokens ⇒ smaller cache ⇒ cheaper long generations).

### 4.6 Random pruning baseline (mandatory)

Every experiment reports the random baseline at the same keep-ratio
(`scoring_method="random"`). Empirical anchor: random 50% works on SmolVLM. These
are necessary conditions for success, mirroring pre-registration §5: text-guided
pruning must (i) beat random at matched budgets (D7), (ii) stay within 2 pp of the
full-token baseline at keep-ratio 0.50, and (iii) meet the field bar against
FastV/SparseVLM at matched budgets.

---

## Section 5 — Evaluation Protocol

- **Models:** SmolVLM-Instruct (primary; cheap on Colab), LLaVA-1.5-7B (secondary;
  cross-architecture validation), LLaVA-NeXT-7B (tertiary, **reduced slice**: 1,000
  samples on GQA + TextVQA, compute-permitting — this measures the strongest
  motivating case from §1.1).
- **Benchmarks:** VQAv2 (recognition), GQA (spatial), TextVQA (OCR), MMBench,
  SEED-Bench (if compute allows).
- **Pruning ratios:** 25% / 50% / 75% removed (keep 0.75 / 0.50 / 0.25).
- **Metrics:** VQA accuracy (VQAv2-style), exact match, FLOPs reduction estimate,
  peak VRAM, tokens/sec / latency (ms).
- **Statistical protocol:** 5 seeds per config (3 minimum for the full matrix);
  report mean ± std; decision statistics = paired t-test, Cohen's d (with CI), and
  a percentile bootstrap CI (10,000 resamples) on the mean difference; Wilcoxon
  signed-rank is reported as a *non-binding* robustness check — at n=3 the minimum
  attainable p is 1/8 = 0.125 one-sided, so it cannot reach 0.05; at n=5 it can
  (min 1/32 = 0.03125).
- **Mandatory baselines:** full tokens; random pruning at each ratio; **FastV**
  (in-decoder) and **SparseVLM** (text-guided, in-decoder) as mandatory
  head-to-head comparisons at matched budgets — both have public code and are
  implemented in weeks 3–4, not hedged as "if implementable". LVPruning / IVTP /
  CDPruner are added if their public code is usable.
- **Ablations:** text-conditioned vs. text-agnostic scoring; cross-attention vs.
  cosine; drop vs. merge; pre-decoder vs. in-decoder; Gumbel/STE vs. hard top-K
  training; pre- vs. post-decoder location (gibberish verification).

---

## Section 6 — Diagnostics (D1–D7)

| # | Name | Question | Method |
|---|---|---|---|
| D1 | Token relevance accuracy | Do high-scored tokens match the objects in the question? | Score heatmaps overlaid on images (notebook 01) |
| D2 | Task sensitivity | Which tasks degrade most? | Per-benchmark accuracy deltas + per-question-type breakdown |
| D3 | Score distribution | Collapsed or calibrated? | std/entropy of scores; detect_score_collapse |
| D4 | Spatial preservation | Is spatial layout kept after pruning? | Row/column coverage of kept grid indices |
| D5 | Resolution sensitivity | Does optimal keep-ratio shift with resolution? | Run 25/50/75% at 336/384/512 |
| D6 | Location verification | Does post-decoder pruning produce gibberish? | Replicate the bug; guard raises (notebook 03) |
| D7 | Random vs. smart | Does text-guided beat random at each ratio? | Δ(acc) per ratio across seeds |

---

## Section 7 — Timeline (12 weeks)

| Week | Phase | Deliverables |
|---|---|---|
| 1–2 | Setup + literature | Repo, environment, verified literature matrix, pre-registration |
| 3–4 | Baselines + extraction | Cache ViT tokens; run full-token + random + **FastV + SparseVLM** head-to-head baselines on all benchmarks |
| 5–6 | Pruning module | Implement TextGuidedPruner; distillation training; temperature schedule |
| 7–8 | Evaluation | All ratios × all benchmarks × head-to-head vs. FastV/SparseVLM × all ablations; 5 seeds |
| 9–10 | Diagnostics + analysis | D1–D7, failure analysis, visualizations, report |
| 11–12 | Writing + revision | Full draft, figures, tables, defense preparation |

---

## Section 8 — Risks and Fallback

| Risk | Mitigation |
|---|---|
| Pruning hurts accuracy >2pp | Reduce to 25% pruning; switch drop → merge |
| Text conditioning doesn't beat random | Fall back to learned text-agnostic scorer (still distinct from training-free heuristics); or reframe as the head-to-head + diagnostics contribution (H-fail fallback c) |
| Training unstable (Gumbel) | STE; narrower temperature range; longer warmup |
| VRAM too high for training | Gradient checkpointing; small subsets; cached visual tokens |
| Colab disconnects | All ViT tokens cached first (script 01); training <2h per config |
| Post-decoder pruning attempted by mistake | Hard guard in wrapper + assertion checks + D6 |
| Random baseline too strong | Report honestly; switch to adaptive ratio or merge strategy |

---

## Section 9 — Pre-Registration

Frozen in `docs/pre_registration.md` (summary):

- **Primary comparison:** text-guided pruning vs. random pruning vs. no pruning, at
  matched token budgets.
- **Mandatory head-to-head:** vs. FastV (in-decoder) and SparseVLM (text-guided,
  in-decoder) at matched budgets; vs. text-agnostic learned pruning.
- **Statistical tests:** paired t-test, Cohen's d (with CI), percentile bootstrap
  CI on the mean difference; Wilcoxon W reported as non-binding (at n=3 its
  minimum p = 0.125, so it cannot decide significance).
- **Success:** text-guided beats random at matched budget with <2pp drop vs.
  full-token baseline, **and** does not underperform FastV/SparseVLM beyond 1pp at
  matched budgets (field bar: FastV <1pp @ ~45% FLOPs ↓).
- **Failure → fallback:** text-guided does not beat random ⇒ text-agnostic learned
  scorer, adaptive keep-ratio, or reframe as the head-to-head + D1–D7 diagnostics
  contribution (H-fail fallback c).
- **Anti-pattern:** never prune from `model.model(**inputs).last_hidden_state`.

---

## Section 10 — Expected Outcomes

1. A quantitative accuracy–efficiency characterization of text-guided visual token
   pruning on three models across two MLLM families (Idefics3/SmolLM2, LLaVA) at
   three pruning ratios.
2. Evidence on whether text conditioning improves pruning quality vs. random and
   text-agnostic methods (D7).
3. Task-sensitivity analysis: which VQA categories are robust vs. fragile to
   pruning (D2).
4. Verification that pruning location matters: pre-decoder works, post-decoder
   produces gibberish (D6).
5. A fully reproducible, Colab-friendly implementation with cached features,
   training scripts, and a documented diagnostic framework.
6. A clear position statement for the community on when and why token pruning
   fails (D3/D4/D7 failure modes).

---

*Flags: ⚠️ = claim reported by web research but not independently re-verified at
defense time; verify arXiv IDs / venues before final submission. Verified facts are
marked inline. All experimental claims in this proposal are reproducible via the
accompanying repository (`text-guided-token-pruning`).*
