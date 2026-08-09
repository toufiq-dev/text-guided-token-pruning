# Change Log

All significant changes to the repository and the thesis documents, newest first.

## 2026-08-09 — Novelty verification + docs-consistency (proposal v1.2)

**Novelty verification (6 parallel web researchers):**
- Added 5 new 2025/2026 competitors: Script (TMLR 2025), AsymVLM (May 2026), GRIP-VLM (May 2026), IDPruner (Feb 2026), SVD-Prune (2026).
- Cited 2 surveys: ACL 2026 systematic review + arXiv:2507.20198.
- Re-scoped 'different objective' claim: GRIP-VLM uses KL as one term in multi-objective + RL; we use KL as primary sole objective with soft-masked STE.
- Added mandatory ablation A6: MSE-on-hidden-states vs KL-on-logits objective comparison.
- Added P5 baseline category for GRIP-VLM/AsymVLM/Script/IDPruner with code-availability justification.
- Added literature watch protocol to §9.
- Verified: SmolVLM 81-token regime genuinely unaddressed; no paper compares MSE vs KL objectives.

**Docs-consistency (3 pinned decisions):**
- Success criterion scoped to keep-ratios 0.50 (binding) and 0.25 (binding); 0.75 non-binding.
- Sample counts pinned: 500 (VQAv2/GQA/TextVQA primary+secondary), 1,000 (LLaVA-NeXT reduced slice, per `configs/models/llava_next_7b.yaml`).
- Full-token cells exempt from multi-seed; single determinism verification required.

Mirrored across proposal.md, pre_registration.md, and change_log.md.

## 2026-08-10 — Docs consistency: failure-ladder scope + wording alignment

Applied the remaining consistency fixes flagged in the 2026-08-09 audit:
- Pre-registration §6: the H-fail ladder now triggers when text-guided pruning is
  ≤ random at keep-ratio 0.50 **or** 0.25 (the binding ratios per §5); 0.75 stays
  non-binding.
- Proposal §7 timeline: evaluation row now references the fixed seed allocation
  (pre-registration §3) instead of the ambiguous "5 seeds".
- README: accuracy budget "~2%" → "~2 pp" (matches proposal/pre-registration);
  scripts tree "01..07" → "01..09" (adds FastV, SparseVLM runners).

No commitments changed; wording aligned to the frozen documents.

## 2026-08-09 — Pre-registration: fixed seed allocation

Clarified §3 of the pre-registration: the 5-vs-3 seed split is fixed in advance —
5 seeds for every P1 cell on the primary model (SmolVLM-Instruct), 3 seeds for all
other cells (secondary/tertiary models, P2/P3 external baselines, P4, ablations).
No commitment changed; the wording removes the ambiguity so the split cannot later
be read (or re-read) as post-hoc flexibility. Mirrored in the proposal §5
statistical protocol and the Colab guide.

## 2026-08-09 — Initial repository scaffold

Initial repository scaffold and proposal draft for Text-Guided Visual Token Pruning thesis.
