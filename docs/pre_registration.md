# Pre-Registration

*Version 1.1 — revised 2026-08-09 per external review (mandatory FastV/SparseVLM
head-to-head baselines, field-bar success criteria, 5 seeds, bootstrap CI,
Wilcoxon demoted to non-binding). Still frozen before running the main experiments
(weeks 5–8); any further deviation must be logged in `docs/change_log.md`.*

This document freezes the experimental commitments of the thesis so that results
cannot be selectively reported. Any deviation from this plan must be logged in
`docs/change_log.md` with a justification.

---

## 1. Primary comparisons

> Note (2026-08-09, external review): FastV and SparseVLM were previously
> secondary / "if implementable". Given verified prior work (SparseVLM, ICML
> 2025; IVTP, ECCV 2024; LVPruning, NAACL 2025 Findings; EvoComp, CVPR 2026;
> CDPruner, NeurIPS 2025), head-to-head comparison against the field is now
> **mandatory**, and the success criteria include a field-bar clause (§5).

| Comparison | Description |
|---|---|
| **P1 (headline)** | Text-guided pruning vs. **random pruning** vs. **no pruning**, at matched token budgets (keep-ratio ∈ {0.75, 0.50, 0.25}) |
| **P2 (mandatory head-to-head)** | Text-guided pruning vs. **FastV** (in-decoder; public code) at matched budgets — implemented in weeks 3–4 |
| **P3 (mandatory head-to-head)** | Text-guided pruning vs. **SparseVLM** (text-guided, in-decoder; training-free; public code) at matched budgets — implemented in weeks 3–4 |
| **P4 (secondary)** | Text-guided pruning vs. text-agnostic learned pruning (same scorer, mean-pooled text input) |
| **P5 (mandatory if code available / cited-context if not)** | vs. **GRIP-VLM** (arXiv:2605.13375, no public code — cited-context), **AsymVLM** (arXiv:2605.29535, no public code — cited-context), **Script** (TMLR 2025, no public code — cited-context), **IDPruner** (arXiv:2602.13315, public code — implement as baseline). For methods without public code: equations documented; reimplementation justified out of scope for this thesis timeline. LVPruning / IVTP / CDPruner remain conditional on usable code |

## 2. Models and benchmarks

- Models: **SmolVLM-Instruct** (primary), **LLaVA-1.5-7B** (secondary),
  **LLaVA-NeXT-7B** (tertiary — reduced slice: 1,000 samples on GQA + TextVQA,
  compute-permitting; this measures the strongest motivating case of the proposal
  §1.1).
- Benchmarks: **VQAv2** (recognition), **GQA** (spatial), **TextVQA** (OCR);
  MMBench + SEED-Bench if compute allows.
- Pruning ratios: **25%, 50%, 75%** (keep-ratio 0.75, 0.50, 0.25). All three are
  mandatory; no ratio may be dropped post hoc.

## 3. Metrics

- **Accuracy:** VQA-accuracy (VQAv2-style `min(matches/3, 1)`) and exact match.
- **Efficiency:** FLOPs-reduction estimate (transparent back-of-envelope, see
  `efficiency_metrics.py`), peak VRAM, latency (ms), tokens/sec.
- Reported per (model × benchmark × ratio × method), **mean ± std over seeds, all fixed via `--seed`**, with a fixed, non-discretionary allocation: **5 seeds for every P1 cell on the primary model (SmolVLM-Instruct × each benchmark × each ratio) — the cells that feed the binding decision statistics (§4) and success criteria (§5) — and 3 seeds for all other cells (secondary/tertiary models, P2/P3 external baselines, P4, ablations)**.
- **Per-benchmark sample counts are pinned in advance:** 500 samples for VQAv2,
  GQA, and TextVQA on the primary model (SmolVLM-Instruct) and secondary model
  (LLaVA-1.5-7B); **1,000 samples** for the LLaVA-NeXT-7B reduced slice (GQA +
  TextVQA only). These values match the experiment configs
  (`evaluation.max_samples_per_benchmark: 500` in `configs/experiments/*.yaml`;
  reduced slice per `configs/models/llava_next_7b.yaml`). If compute constraints
  prevent reaching the pinned sample count for a given cell, the deviation must
  be logged in `change_log.md` **before** the reduced run is executed — never
  after inspecting results.
- **Full-token (no-pruning) cells are exempt from multi-seed repetition.** They
  are run once per benchmark per model. A single **determinism verification** is
  performed: the full-token cell is run twice on the primary model's headline
  benchmark (VQAv2), and the two runs must agree to 4 decimal places on the
  reported metric. If they do not agree, the source of non-determinism is
  investigated and resolved before any pruned cells are run. This exemption is
  justified because the full-token inference path contains no stochastic
  component (eval mode, no dropout, greedy decoding, fixed inputs).

## 4. Statistical tests (per comparison cell)

Decision statistics (binding):

1. **Paired t-test** of per-sample scores across seeds (text-guided vs. each
   baseline: random, FastV, SparseVLM, text-agnostic learned).
2. **Cohen's d** effect size, reported with a confidence interval.
3. **Percentile bootstrap CI** (10,000 resamples) on the mean difference — no
   normality assumption, no discreteness floor.

Robustness check (non-binding):

4. **Wilcoxon signed-rank** W is *reported but never decisive*. At n = 3 paired
   seeds the minimum attainable p is 1/8 = 0.125 (one-sided) / 0.25 (two-sided),
   so it **cannot** reach p < 0.05 regardless of effect size. At n = 5 seeds the
   minimum is 1/32 = 0.03125 (one-sided), so Wilcoxon is only meaningful on the 5-seed
   P1 cells of the primary model (§3). (Drawing on best practices for rigorous
   empirical ML: never lean on a test that cannot produce significance at your
   seed count.)

## 5. Success criteria

- Text-guided pruning **beats random pruning at matched token budgets** (p < 0.05
  on the paired t-test, or the bootstrap CI on the mean difference excludes 0;
  positive effect size) on the 5-seed P1 cells of the primary model at
  **keep-ratios 0.50 (primary binding) and 0.25 (secondary binding)**. Keep-ratio
  0.75 is reported for completeness but is **non-binding**: at 25% token removal
  the expected gap between random and guided pruning is near zero by construction,
  and a non-significant result at 0.75 does not falsify H1. Failure at 0.25
  triggers the H-fail pivot ladder even if 0.50 succeeds, **and**
- VQA accuracy stays **within 2 pp of the full-token baseline** at keep-ratio 0.50,
  **and**
- **Field bar (matched budgets):** the accuracy gap to **FastV** and **SparseVLM**
  is ≤ 1 pp and the FLOPs reduction is within 5 pp of theirs. Published bar:
  FastV < 1 pp @ ~45% FLOPs ↓, PyramidDrop ~0.6 pp @ >55% ↓, LVPruning ~0.45 pp @
  up-to-90% tokens. Beating random alone is **necessary but not sufficient**.

Rationale: the pre-registration's primary comparison against random catches
redundancy-exploitation failures; the field-bar criterion catches the failure to
add anything beyond existing published methods.

## 6. Failure criteria and fallback ladder

| Condition | Verdict | Fallback |
|---|---|---|
| Text-guided ≤ random at 0.50 or 0.25 (the binding ratios, §5) | H-fail confirmed | (a) learned text-agnostic scorer; (b) adaptive keep-ratio from score entropy |
| Underperforms FastV/SparseVLM by > 1 pp at matched budgets | Claim weakened | Reframe as the rigorous head-to-head + D1–D7 diagnostics contribution; investigate why training-free text-guided scoring is hard to beat (D3/D4) |
| Accuracy drop > 2 pp at 50% | Budget exceeded | Reduce to 25% pruning; try merge instead of drop |
| Score collapse (D3) | Training signal dead | Entropy loss upweighting; straight-through only; smaller τ range |
| Post-decoder pruning encountered | Implementation bug | Guard already raises; fix code, not results |

## 7. Anti-pattern registration

> **NEVER prune from `model.model(**inputs).last_hidden_state` (post-decoder).**
> Empirically this produces gibberish (documented in proposal §1.3 and notebook
> 03). The wrapper raises a
> `RuntimeError` on any attempt (`prune_from_last_hidden_state`). Any experiment
> JSON containing such a path is automatically invalid.

## 8. Diagnostics locked

D1 token-relevance heatmaps · D2 task sensitivity · D3 score-collapse · D4 spatial
preservation · D5 resolution sensitivity · D6 location verification · D7
random-vs-smart deltas. All seven are mandatory outputs of the final report.

## 9. Reproducibility commitments

- All seeds fixed and recorded (`--seed`).
- All configs in `configs/experiments/*.yaml`; all results JSON in `outputs/`.
- Visual-token cache (script 01) makes reruns cheap and Colab-disconnect safe.
- Every table in the thesis maps to a script and a results file.

**Literature watch protocol:** Given the rapid pace of this subfield (6 new
competitor papers identified between May–August 2026), we commit to monthly
arXiv/ACL anthology re-checks for new token-pruning papers. Any new competitors
discovered after pre-registration freezing will be logged in
`docs/change_log.md` with a one-paragraph assessment of threat level and whether
they invalidate the pre-registered comparisons.
