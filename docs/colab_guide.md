# Colab Reproduction Guide

Everything in this thesis is designed to be reproducible on a free Colab GPU
(T4), with one script per stage and disk caching so disconnects cost nothing.

## 0. Setup (one-time)

```bash
git clone <repo-url> text-guided-token-pruning && cd text-guided-token-pruning
python -m venv .venv && source .venv/bin/activate
pip install -e ".[colab]"
```

Or upload the repo to Colab / open the notebooks directly from GitHub.

## 1. Sanity check (5 min)

Open `notebooks/00_sanity_check_colab.ipynb` → Runtime → Run all.
This verifies the pre-decoder extraction contract and runs one baseline question.

## 2. Replicate the key finding (10 min)

`notebooks/03_random_vs_smart_pruning.ipynb` — runs baseline vs. random 50% vs.
smart 50% (correct location) vs. smart 50% (wrong location) on 3 questions.
**This is the empirical foundation of the thesis; reproduce it first.**

## 3. Cache visual tokens (30–60 min, one-time per dataset)

```bash
python scripts/01_extract_visual_tokens.py --model HuggingFaceTB/SmolVLM-Instruct \
    --dataset vqav2 --split validation --max-samples 500 --output-dir cached_features
```

Only the vision encoder + connector run here (never the LLM decoder). The `.npy`
cache is what makes later steps Colab-disconnect-proof.

## 4. Full + random baselines (1–2 h per benchmark slice)

```bash
python scripts/03_run_inference.py --scoring cross_attention --prune-ratio 0.0 ...
python scripts/07_random_pruning_baseline.py --prune-ratios 0.25 0.5 0.75 ...
```

## 5. Train the pruner (<2 h per config on T4)

```bash
python scripts/02_train_pruning_module.py --keep-ratio 0.5 --num-epochs 2 \
    --batch-size 2 --cache-dir cached_features/vqav2 --output-dir outputs/pruner_50pct
```

Tip: use `--gradient-checkpointing` if VRAM is tight; reduce `--max-samples`.

## 6. Evaluate and report

```bash
python scripts/03_run_inference.py --scoring learned --pruner-checkpoint outputs/pruner_50pct/pruner.pt ...
python scripts/04_evaluate.py --results-dir outputs/results --report outputs/report.md
python scripts/05_run_ablations.py --max-samples 20
python scripts/06_generate_report.py
```

## Conventions that keep you safe

- **Always** include `--scoring random` results (D7) — they are the bar to beat.
- The wrapper **raises** if you try to prune post-decoder hidden states; never
  silence that error (D6).
- Log every run's seed; use 5 seeds for the final numbers (3 minimum — see
  `docs/pre_registration.md` §3; Wilcoxon is non-binding below 5 seeds).
- If a notebook/script disconnects: rerun — the cache means nothing is lost.
