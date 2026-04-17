# QSpiritGuide Unsloth Project (Wiki-First Pipeline)

This repository implements a high-quality "Wiki-First" fine-tuning pipeline. 

Instead of training on messy raw data, we use a high-capacity LLM (the **Librarian**) to synthesize raw scrapes into a clean, distraction-free Knowledge Base (Wiki). This Wiki is then used to generate factual, high-density training data.

## 🛠️ Installation & Set Up

1. **Create and activate a virtual environment:**
   ```bash
   uv venv
   source .venv/bin/activate
   ```

2. **Select Python Interpreter in VSCode:**
   - Press `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Windows/Linux).
   - Type **"Python: Select Interpreter"**.
   - Select the one located in the project's `.venv` folder. This ensures all tools (linting, notebook kernels) use the correct environment.

3. **Install dependencies:**
   ```bash
   uv pip install -r requirements.txt
   ```
   *(Note: For Apple Silicon Macs, `mlx-tune` is automatically included to support local fine-tuning via the MLX framework.)*

4. **Set Up LM-Studio (The "Librarian"):**
   - Install and open [LM-Studio](https://lmstudio.ai/).
   - Load your preferred high-capacity model (e.g., Qwen2, Llama 3).
   - Navigate to the **Local Server** tab and click **Start Server**.

## ⚙️ Configuration

Settings for LM-Studio and processing are managed in [config.json](file:///Users/adir1/git/qspiritguide-unsloth/config.json).
- `base_url`: Your LM-Studio endpoint (default `http://localhost:1234/v1`).
- `model`: The model name running in LM-Studio (e.g., `Gemini4`).
- `chunk_size_words`: The size of text chunks for processing (Optimized for 32k context).

## � Ignoring Concepts

If you want to exclude certain concepts from the pipeline (preventing hours of unnecessary LLM processing), you can move them to `wiki/ignore/`:

```bash
# See all active and ignored concepts
python manage_ignore.py list

# Ignore a concept (prevents compile_wiki.py, refine_wiki.py, and distill_wiki.py from processing it)
python manage_ignore.py ignore concept_name

# Re-enable a concept
python manage_ignore.py unignore concept_name
```

**How it works:**
- When you run `python manage_ignore.py ignore alchemy`, it moves `wiki/alchemy.md` to `wiki/ignore/alchemy.md`
- All pipeline scripts automatically skip anything in `wiki/ignore/`, preventing wasted LLM calls during synthesis and refinement
- The concept is completely excluded from training data generation
- You can re-enable it anytime by running `unignore`

## �🚀 The workflow

### 1. Ingest Raw Data
Add URLs to `urls.txt` and run:
```bash
python fetch_webpages.py
```
Or place your own raw files into `raw_data/`.

Supported raw data formats:
- `.md`
- `.txt`
- `.pdf`
- `.epub` (supported by `compile_wiki.py` via `ebooklib` + `beautifulsoup4`)

### 2. Compile the Knowledge Base (The Wiki)
Synthesize raw data into "Current Understanding" articles. This step also separates historical evolution into separate files to keep the training source pure.
```bash
python compile_wiki.py
```
*Outputs: `wiki/*.md` and `wiki/history_*.md`*

### 3. Refine and Merge Concepts
Run the auditor to scan the entire wiki for overlapping concepts and merge them into unified, high-quality articles.
```bash
python refine_wiki.py
```

### 4. Distill into Training Pairs
Generate high-density Q&A pairs (ChatML format) from the synthesized wiki. **Note:** Only "Current Understanding" articles are used; history is excluded to ensure the model learns the latest facts.
```bash
python distill_wiki.py
```
*Outputs: `data/staged_data.json`*

### 5. Compile Final Dataset
Convert the staged JSON pairs into the final `train.jsonl` format.
```bash
python compile_dataset.py
```

### 6. Fine-Tune the Model
Train your 500M parameter model locally on your M5 Mac.
```bash
python finetune.py
```

---

## 📓 Using the Notebooks

There are two primary Jupyter notebooks included. Here is when to use them:

### `distill.ipynb`
**When to use:** Use this for **interactive experimentation** during Stage 4. 
- If you want to manually review how the LLM extracts Q&A pairs from a specific wiki article.
- If you want to tweak prompts for specific types of data before running the full `distill_wiki.py` script.

### `finetune.ipynb`
**When to use:** Use this as a replacement for Stage 6 if you prefer an **interactive training environment**.
- It provides visual loss curves and allows you to test the model (Inference) immediately after training within the same environment.
- Highly recommended for your first few runs on the M5 Mac to monitor performance.

---

## 🛠 Model Export (WebGPU / Transformers.js)
To export your fine-tuned model for use in the browser:
1. Ensure the model is saved to `model_16bit` in the training script.
2. Run the conversion:
   ```bash
   uv pip install 'optimum[exporters]'
   optimum-cli export onnx --model ./model_16bit ./onnx_output
   ```
