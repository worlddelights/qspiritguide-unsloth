# QSpiritGuide Unsloth Project (Wiki-to-Training Pipeline)

This project starts with a completed knowledge base in `wiki/`, distills it into reviewed ChatML training data, and fine-tunes a model with Unsloth on an NVIDIA GPU.

The earlier ingestion, wiki compilation, and refinement stages are intentionally out of scope for this environment.

## Installation

1. Install an NVIDIA driver compatible with the CUDA-enabled PyTorch build required by your GPU.

2. Sync the project environment with [uv](https://docs.astral.sh/uv/):
   ```bash
   uv sync
   ```

3. Start an LM Studio server with the model configured in `config.json`:
   - Install and open [LM-Studio](https://lmstudio.ai/).
   - Load an instruction model capable of generating training pairs.
   - Navigate to the **Local Server** tab and click **Start Server**.

4. Confirm PyTorch can see the GPU before training:
   ```bash
   uv run python -c "import torch; print(torch.cuda.get_device_name(0))"
   ```

## Configuration

- `lm_studio`: endpoint and model used to distill and optionally audit focused wiki articles.
- `finetune.models`: named base checkpoints that can be selected for individual Unsloth runs.
- `finetune.training`: shared initial training settings; tune them only after evaluating each model.

## Workflow

### 1. Curate the focused corpus

Treat `wiki_complete/` as read-only. Place proposed rewritten articles in
`wiki_focused/review_queue/`. After human review, move or copy approved,
training-ready Markdown files into `wiki_focused/articles/`. Only files in
`wiki_focused/articles/` are eligible for distillation.

Spiritual, religious, contemplative, and occult material may be retained when
it is accessible, culturally contextualized, and clear about the difference
between historical belief, personal experience, symbolism, and established
evidence.

### 2. Audit the focused corpus (optional, recommended)

Run a read-only audit before distillation to find exact and likely near
duplicates, flag concepts without corpus definitions, identify possible
consistency issues, and review topic emphasis:

```bash
uv run python audit_corpus.py
```

The semantic pass uses the `lm_studio` model configured in `config.json` and
sends focused article text to that private endpoint. It treats the articles as
the corpus authority and does not consult external sources. Results are review
signals, not factual judgments; an automated report cannot establish what the
fine-tuning base model already knows. A content-hash cache in the report
directory avoids re-analyzing unchanged articles, and each successful model
result is saved immediately so a later failure can resume from completed work.
The CLI prints progress for each article and consistency pair.
By default, at most the 200 most lexically related article pairs are checked
for possible conflicts; adjust this with `--max-pairs`.

Use `--offline` for a deterministic-only report containing exact and near
duplicates plus recurring-vocabulary signals, without calling LM Studio:

```bash
uv run python audit_corpus.py --offline
```

Both Markdown and JSON reports are written to
`reports/corpus_audit/`. The script does not edit articles or training data.
Similarity and pair-selection thresholds can be adjusted with
`--near-duplicate-threshold` and `--similarity-threshold`.

### 3. Distill focused wiki articles

Generate instruction/response pairs from the approved focused articles. The
script writes pairs to `data/staged_data.json` and skips unchanged articles
unless `--force` is supplied.

```bash
uv run python distill_wiki.py
```

### 4. Review and compile the training set

Review pairs marked with `requires_review` in `data/staged_data.json`, then compile the approved records. This overwrites `data/train.jsonl`, keeping the dataset idempotent.

```bash
uv run python compile_dataset.py
```

### 5. Fine-tune with Unsloth

List the available base models, then explicitly select exactly one model per
invocation. To smoke-test a selection, cap it at a few steps first:

```bash
uv run python finetune.py --list-models
uv run python finetune.py --model smollm2-135m --max-steps 3
```

For a full run, omit `--max-steps`; training uses the configured epoch count.
Run models sequentially on the same GPU. Each invocation creates a timestamped
directory under `training_runs/<model-id>/` containing trainer checkpoints,
the LoRA adapter, a merged 16-bit checkpoint, and `run.json` metadata. Outputs
are ignored by Git and a later run will not overwrite an earlier one.

The Llama 3.2 base may require accepting Meta's Hugging Face model terms and
authenticating with the Hub before downloading it. The trainer uses each
checkpoint's own tokenizer chat template; it does not rewrite templates to
ChatML.

See [MODEL_FINETUNING_HANDOFF.md](./MODEL_FINETUNING_HANDOFF.md) for the ONNX
export, quantization, packaging, and browser acceptance steps. Merged 16-bit
weights are the source for that separate conversion workflow; the trainer
does not produce browser-ready quantized ONNX files.
