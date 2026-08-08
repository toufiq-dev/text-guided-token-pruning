# Data Download Scripts

This directory holds scripts for downloading/preparing evaluation datasets.
The current pipeline loads datasets **directly from the HuggingFace hub at
runtime** (`tgtp.data.vqa_loader.BENCHMARK_CONFIGS`), so no download step is
required for VQAv2 / GQA / TextVQA when the hub names resolve and you have a
HuggingFace token where needed.

## Hub paths used (⚠️ verify at runtime)

| Benchmark | Hub path | Split | Metric |
|---|---|---|---|
| VQAv2 | `HuggingFaceM4/VQAv2` | validation | vqa_accuracy |
| GQA | `lmms-lab/GQA` | testdev | exact_match |
| TextVQA | `lmms-lab/TextVQA` | validation | vqa_accuracy |

If a hub path fails (auth / renamed dataset), either:

1. Use an alternative hub path, or
2. Provide a local JSONL dataset:

```bash
# JSONL schema: {"image": <path|url>, "question": str, "answers": [str, ...]}
python scripts/03_run_inference.py ... --local-jsonl data/raw/my_benchmark.jsonl
```

## Planned download scripts (future work)

- `download_vqav2.py` — download images + annotations from the official VQA
  server (requires registration) and emit the JSONL schema above.
- `download_gqa.py` — GQA scene graphs + images → JSONL.
- `download_textvqa.py` — TextVQA images/annotations → JSONL.

## Policy

- Raw images and annotations are **git-ignored** (`data/raw/`, `data/annotations/`).
- Only scripts and tiny sample manifests live in the repo.
