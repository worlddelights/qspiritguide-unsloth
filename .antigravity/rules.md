# Antigravity IDE Agent Instructions

## Project Context
- **Primary Goal**: Initialize and fine-tune a small LLM (~500M parameters) on a collection of custom English texts using Unsloth.
- **Primary Model**: `unsloth/Qwen2.5-0.5B`.
- **Final Output Goal**: LoRA adapters must be trained and then explicitly merged into a 16-bit model (`merged_16bit`). This output will subsequently be exported to ONNX format for use with Transformers.js and WebGPU.

## Agent Development Guidelines
1. **Unsloth Tooling & API**:
   - Always load the base model using `unsloth.FastLanguageModel.from_pretrained`.
   - Utilize `unsloth.FastLanguageModel.get_peft_model` for LoRA setup.
   - When updating the dataset configuration during the upcoming deep dive, ensure you map the data carefully to memory-efficient representations.

2. **File Structure Parity**:
   - Ensure that any modifications made to `finetune.py` are mirrored or capable of being generated into `finetune.ipynb`. Assume the main source of truth is the `.py` script.

3. **Data Preparation Strategy (Upcoming Phase)**:
   - Carefully identify if the training requires casual Language Modeling (LM) or Instruction-tuning (QA mapping).
   - Datasets should be loaded out of the `data/` branch.

4. **Task and Artifact Tracking**:
   - Continuously update `task.md` whenever core steps (like Dataset formatting or Training execution) are completed or adjusted.
