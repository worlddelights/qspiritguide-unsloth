# QSpiritGuide Unsloth Project (Wiki-First Pipeline)

This repository implements a high-quality "Wiki-First" fine-tuning pipeline. 

Instead of training on messy raw data, we use a high-capacity LLM (the **Librarian**) to synthesize raw scrapes into a clean, distraction-free Knowledge Base (Wiki). This Wiki is then used to generate factual, high-density training data.

## ⚙️ Configuration

Settings for LM-Studio and processing are managed in [config.json](file:///Users/adir1/git/qspiritguide-unsloth/config.json).
- `base_url`: Your LM-Studio endpoint (default `http://localhost:1234/v1`).
- `model`: The model name running in LM-Studio (e.g., `Gemini4`).
- `chunk_size_words`: The size of text chunks for processing (Optimized for 32k context).

## 🚀 The workflow

### 1. Ingest Raw Data
Add URLs to `urls.txt` and run:
```bash
python fetch_webpages.py
```
Or place your own `.md`, `.txt`, or `.pdf` files into `raw_data/`.

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
   optimum-cli export onnx --model ./model_16bit ./onnx_output
   ```
