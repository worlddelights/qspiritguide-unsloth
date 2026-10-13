# QSpiritGuide Unsloth — Agent Guide

This repository's supported workflow starts from a curated wiki corpus, distills it into reviewed ChatML training data, and fine-tunes a model locally with Unsloth on an NVIDIA GPU. Use the command-line scripts described below; Jupyter notebooks are not part of the supported workflow.

## Active workflow

### 1. Curate the focused corpus

- Treat `wiki_complete/` as read-only.
- Put proposed rewritten articles in `wiki_focused/review_queue/`.
- After human review, move or copy approved Markdown articles into `wiki_focused/articles/`. Only articles in this directory are eligible for distillation.
- Keep articles accessible and culturally contextualized. Distinguish historical beliefs, personal experiences, and symbolism from established evidence.

### 2. Audit the corpus (optional, recommended)

Run the corpus audit before distillation:

```bash
uv run python audit_corpus.py
```

The semantic audit uses the model configured in `config.json` under `lm_studio` and sends focused article text to that configured endpoint. It does not edit articles or training data. Its findings are review signals, not factual judgments. For a deterministic-only audit that makes no model calls, run:

```bash
uv run python audit_corpus.py --offline
```

Reports are written to `reports/corpus_audit/`. By default, the semantic pass checks at most 200 lexically related article pairs; `--max-pairs` changes that limit.

### 3. Distill approved articles

```bash
uv run python distill_wiki.py
```

The script processes Markdown files in `wiki_focused/articles/`, writes generated pairs to `data/staged_data.json`, and uses `.distill_state.json` to skip unchanged articles. Use `--force` to process all articles again or `--clear` to clear the state cache.

### 4. Review and compile training data

Review `data/staged_data.json`. Correct generated pairs as needed and clear `requires_review` only after review. Then compile the approved pairs:

```bash
uv run python compile_dataset.py
```

This overwrites `data/train.jsonl` and excludes records whose `requires_review` value is true.

### 5. Fine-tune locally

```bash
uv run python finetune.py
```

Fine-tuning runs locally with Unsloth and requires an NVIDIA GPU with CUDA-enabled PyTorch. Select exactly one configured checkpoint per invocation with `--model`; the LM Studio model configured under `lm_studio` is used for distillation and semantic auditing, not as a fine-tuning base model. Each run saves its own LoRA adapter and merged 16-bit model under `training_runs/`.

#### Planned fine-tuning base models

The planned base checkpoints for the browser models are the following three
instruction-tuned models. Fine-tune each independently from its original
Hugging Face checkpoint, using the same approved dataset:

| App tier | Base checkpoint | Browser export target |
| --- | --- | --- |
| `1B-WebGPU` | `meta-llama/Llama-3.2-1B-Instruct` | WebGPU, `q4f16` |
| `360M-WebGPU` | `HuggingFaceTB/SmolLM2-360M-Instruct` | WebGPU, `q4f16` |
| `135M-WASM` | `HuggingFaceTB/SmolLM2-135M-Instruct` | WASM, `int8` |

See [MODEL_FINETUNING_HANDOFF.md](./MODEL_FINETUNING_HANDOFF.md) for the
conversion, packaging, and acceptance requirements. The planned bases above
supersede Qwen as the intended fine-tuning targets; the LM Studio model remains
the separate distillation/audit model.

`finetune.py --list-models` lists these configured choices. Run one model at a
time, for example `uv run python finetune.py --model smollm2-135m --max-steps 3`
for a short smoke run, then omit `--max-steps` for a full configured epoch.
Each invocation writes to a unique timestamped directory under
`training_runs/<model-id>/`, including adapter, merged checkpoint, trainer
checkpoints, and run metadata. The trainer uses each checkpoint's original
tokenizer and chat template. Smoke-test each exact checkpoint with the
installed Unsloth version; do not assume that sharing a `LlamaForCausalLM`
architecture label guarantees Unsloth support for either SmolLM2 size.

## Configuration and state

- `config.json`
  - `lm_studio`: endpoint and model for distillation and semantic auditing.
  - `finetune.models`: explicit Hugging Face checkpoints selectable for individual local training runs.
  - `finetune.training`: shared initial Unsloth training settings.
- `wiki_complete/`: read-only source corpus.
- `wiki_focused/review_queue/`: proposed articles awaiting human review.
- `wiki_focused/articles/`: approved articles used by the active workflow.
- `data/staged_data.json`: generated pairs awaiting review.
- `data/train.jsonl`: compiled dataset consumed by fine-tuning.
- `.distill_state.json`: hashes used to skip unchanged articles during distillation.
- `reports/corpus_audit/`: audit reports.
- `training_runs/`: ignored local checkpoints, adapters, merged models, and run metadata.

## Agent guidance

- Before running any model-backed command, check the relevant `config.json` settings and confirm the configured service is intended and available.
- Inspect `data/staged_data.json` before proposing changes to the training data; preserve its existing record schema and review flags.
- Keep `wiki_complete/` unchanged. Make proposed edits in the review queue and only distill approved articles.
- Do not use the legacy ingestion/refinement scripts or notebooks as part of the active workflow unless the user specifically requests work on those components.
