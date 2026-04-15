---
name: qspiritguide-unsloth-instructions
description: "QSpiritGuide Unsloth: Wiki-First LLM Fine-Tuning Pipeline. Use when: working on data pipeline stages, modifying wiki/raw_data synthesis, improving ChatML training data generation, understanding information flow from sources to training."
---

# QSpiritGuide Unsloth — Copilot Instructions

## Project Overview

QSpiritGuide Unsloth implements a **"Wiki-First" fine-tuning pipeline** for training specialized LLMs on high-quality spiritual and esoteric knowledge. The core philosophy is to avoid training on messy raw data. Instead, a high-capacity LLM (the "Librarian" running in LM-Studio) synthesizes raw scrapes and metadata into a clean, distraction-free Knowledge Base (Wiki). This Wiki then feeds into training data generation using ChatML format.

## Architecture & Data Flow

### 1. **Raw Data Layer** (`raw_data/`)
- **Input**: Web scrapes (`.md`, `.txt`, `.pdf` files from `fetch_webpages.py`) or manually added files
- **Location**: `raw_data/` directory
- **Purpose**: Messy, context-heavy source material
- **Status**: NOT used directly for training; only consumed by the Wiki synthesis stage

### 2. **Wiki Layer** (`wiki/`)
The heart of the pipeline. Contains two types of synthesized articles:

#### Current Understanding (`.md` files)
- Cleaned, distraction-free synthesis of raw data
- Represents the consolidated, latest knowledge on a topic
- **Only these are used** for generating training pairs (see Stage 4)
- Example: `alchemy.md`, `angels.md`, `magic.md`

#### Historical Evolution (`history_*.md` files)
- Extracted chronological/evolutionary context from raw sources
- Separated to keep training data focused on "current" knowledge
- Useful for reference but excluded from training data generation
- Example: `history_alchemy.md`, `history_christianity.md`

### 3. **Training Data Generation** (`data/`)
- **Staged Data**: `data/staged_data.json` — Chat-style Q&A pairs in JSON format
- **Final Dataset**: `data/train.jsonl` — ChatML-formatted Instruction/Response pairs ready for fine-tuning

---

## Pipeline Stages

### Stage 1: Ingest Raw Data
**Script**: `fetch_webpages.py`
- Fetches URLs from `urls.txt` and saves as markdown in `raw_data/`
- Or: Manually place `.md`, `.txt`, `.pdf` files into `raw_data/`

### Stage 2: Compile the Knowledge Base (Wiki Synthesis)
**Script**: `compile_wiki.py`
- The Librarian LLM reads raw data chunks and synthesizes them into coherent `wiki/*.md` articles
- Separation logic: Historical context → `wiki/history_*.md`; Current understanding → `wiki/*.md`
- **Key concept**: This stage transforms messy source material into high-density, factual articles

### Stage 3: Refine & Merge Concepts
**Script**: `refine_wiki.py`
- Auditor scans the entire wiki for overlapping or redundant concepts
- Merges conflicting information into unified, high-quality articles
- Improves consistency across the knowledge base

### Stage 4: Distill into Training Pairs
**Script**: `distill_wiki.py`
- **Critical**: Only processes "Current Understanding" articles (excludes `history_*.md`)
- Generates 5-10 high-quality Q&A pairs per article using the Librarian LLM
- Output: `data/staged_data.json` (list of Instruction/Response objects in JSON)
- Philosophy: Ensures the fine-tuned model learns only latest facts, not contradictory historical context

### Stage 5: Compile Final Dataset
**Script**: `compile_dataset.py`
- Converts `data/staged_data.json` into `data/train.jsonl` (line-delimited ChatML format)
- Each line: `{"text": "<s>user\n{instruction}</s><s>assistant\n{response}</s>"}`

### Stage 6: Fine-Tune the Model
**Script**: `finetune.py` (or interactive `finetune.ipynb`)
- Trains a 500M parameter model locally on M-series Macs using MLX framework
- Input: `data/train.jsonl`
- Output: `model_16bit/` directory with fine-tuned model weights

---

## Configuration

**File**: `config.json`

Key settings:
- **`lm_studio.base_url`**: LM-Studio endpoint (default: `http://localhost:1234/v1`)
- **`lm_studio.model`**: Model name running in LM-Studio (e.g., `Qwen2`, `Llama 3`)
- **`wiki.max_context_token`**: Context window size for Wiki synthesis (default: 4096)
- **`wiki.temperature`**: LLM temperature for synthesis (lower → more deterministic, default: 0.2)
- **`wiki.enable_thinking`**: Enable extended thinking for complex synthesis tasks

---

## Key Design Principles

1. **Separation of Concerns**
   - Raw data stays raw (unprocessed source material)
   - Wiki is the single source of truth for training (synthesized, cleaned, current-focused)
   - History is tracked separately (prevents contradictions in training)

2. **LLM-Mediated Quality**
   - The Librarian LLM (Stage 2) filters noise from raw sources
   - The Auditor (Stage 3) ensures consistency across the kb
   - Q&A generation (Stage 4) prioritizes high-density, educational pairs

3. **Training Data Purity**
   - Only "current understanding" articles → training pairs
   - History articles excluded → prevents model confusion from contradictory evolution timelines
   - Higher temperature (0.7) in Stage 4 → diverse, pedagogically rich Q&A

---

## Working with Notebooks

### `distill.ipynb` — Interactive Experimentation
- Use for Stage 4 testing and prompting refinement
- Manually review how Q&A pairs are extracted from a specific wiki article
- Tweak LLM prompts before running full `distill_wiki.py`

### `finetune.ipynb` — Interactive Training (Recommended)
- Use as replacement for Stage 6 (interactive alternative to `finetune.py`)
- Provides real-time loss curves and immediate inference testing
- Recommended for first few training runs to monitor performance

---

## Common Tasks & Guidance

### Adding New Knowledge
1. Add URLs to `urls.txt` and run `python fetch_webpages.py`
2. Or manually place `.md`/`.txt`/`.pdf` files in `raw_data/`
3. Run `python compile_wiki.py` to synthesize into wiki
4. Review `wiki/*.md` articles and `wiki/history_*.md` for evolution tracking
5. Run `python refine_wiki.py` to merge overlapping concepts
6. Proceed to Stage 4 (distill) when satisfied

### Refining Training Quality
- Edit `wiki/*.md` articles directly (fixes persist in training data)
- DO NOT edit `history_*.md` files (they're references, not training sources)
- Re-run `python distill_wiki.py` to regenerate training pairs
- Delete `data/staged_data.json` before re-running to avoid stale entries

### Debugging the Pipeline
- Check `.wiki_state.json` to see which raw files were processed
- Look at first few entries in `data/staged_data.json` to inspect Q&A quality
- Check LM-Studio logs for synthesis errors (temperature/context issues)
- Use `distill.ipynb` to test prompts on individual articles

---

## Environment Setup

**Required**:
- Python 3.9+ (via `.venv` virtual environment)
- LM-Studio running locally with a high-capacity model (Qwen2, Llama 3, etc.)
- `config.json` configured with correct LM-Studio endpoint and model name

**Installation**:
```bash
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

**For M-series Macs**: MLX framework included automatically for local fine-tuning.

---

## Use Case Focus

This Copilot context is optimized for:
- **Understanding the wiki-first philosophy** and why each pipeline stage exists
- **Modifying synthesis workflows** (how raw data becomes wiki articles)
- **Improving training data quality** (Q&A pair generation in ChatML format)
- **Debugging information flow** (tracking data from raw sources to training inputs)
- **Working with notebooks** for interactive experimentation and training
- **Configuring LLM behavior** across synthesis, refinement, and distillation stages

When editing code, keep in mind the pipeline sequential order and the critical role of the Wiki as the single source of truth for the fine-tuning process.
