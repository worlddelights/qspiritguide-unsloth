# QSpiritGuide Unsloth Project

This repository contains scripts and notebooks to fine-tune the `unsloth/Qwen2.5-0.5B` model on custom English texts.

## Set Up

1. Install the requirements:
   ```bash
   pip install torch torchvision torchaudio
   pip install -r requirements.txt
   ```
   *(Note: For Apple Silicon (M-series) Macs, `mlx-tune` is included in the requirements to run efficiently locally via Apple's MLX framework. For Windows, you may need to compile xformers from source or use a pre-built wheel depending on your PyTorch version. Official Unsloth requires Linux or WSL, but our script automatically detects Mac and swaps to `mlx-tune`.)*

2. Distill Your Training Data:
   We use an automated pipeline to extract ChatML formatted instruction/response pairs from large PDFs.
   - Install and open [LM-Studio](https://lmstudio.ai/).
   - Load your preferred "medium" model (e.g. Llama 3 8B, Qwen, etc.).
   - Start the **Local Server** in LM-Studio (running on `http://localhost:1234/v1`).
   - Place your raw PDF files in the `raw_data/` folder.
   - Run the distillation notebook to generate your `data/train.jsonl` dataset:
     ```bash
     jupyter notebook distill.ipynb
     ```

3. Run the notebook/script:
   Run either `finetune.py` or start Jupyter and open `finetune.ipynb`:
   ```bash
   jupyter notebook
   ```

## Model Export (Transformers.js / WebGPU)

To export to Transformers.js:
1. Ensure you have run the merging step at the end of the script to save your model in 16-bit to `model_16bit`.
2. Convert the HuggingFace model to ONNX using `optimum-cli`:
   ```bash
   pip install optimum[exporters]
   optimum-cli export onnx --model ./model_16bit ./onnx_output
   ```
