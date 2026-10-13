# Antigravity IDE Agent Instructions

## Project Context
- **Primary Goal**: Fine-tune the selected browser-sized instruction models on the reviewed project corpus using Unsloth.
- **Planned Base Models**: `meta-llama/Llama-3.2-1B-Instruct`, `HuggingFaceTB/SmolLM2-360M-Instruct`, and `HuggingFaceTB/SmolLM2-135M-Instruct`.
- **Run Selection**: Train exactly one model per `finetune.py --model <model-id>` invocation. Keep adapters, checkpoints, and merged outputs in that run's timestamped directory under `training_runs/`.
- **Final Output Goal**: Save each LoRA adapter and merge it into a 16-bit checkpoint. The merged checkpoint is an input to a separate ONNX export and browser quantization workflow; it is not itself a browser-ready model.

## Agent Development Guidelines
1. **Unsloth Tooling & API**:
   - Always load the base model using `unsloth.FastLanguageModel.from_pretrained`.
   - Utilize `unsloth.FastLanguageModel.get_peft_model` for LoRA setup.
   - When updating the dataset configuration during the upcoming deep dive, ensure you map the data carefully to memory-efficient representations.

2. **Training Source of Truth**:
   - Use `finetune.py` and its CLI for supported training runs. Do not update or run the legacy notebook unless the user specifically requests notebook work.

3. **Data Preparation Strategy (Upcoming Phase)**:
   - Carefully identify if the training requires casual Language Modeling (LM) or Instruction-tuning (QA mapping).
   - Datasets should be loaded out of the `data/` branch.

4. **Task and Artifact Tracking**:
   - Continuously update `task.md` whenever core steps (like Dataset formatting or Training execution) are completed or adjusted.
