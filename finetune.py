"""
Unsloth Fine-Tuning Script
--------------------------
This script is configured for the 500M parameter Qwen2.5 model.
The dataset and prompt formats are currently placeholders, pending a deeper
dive into the custom English texts formatting.
"""

import torch
import sys
from datasets import load_dataset
from transformers import TrainingArguments

# Check for Apple Silicon (Mac) to use MLX for efficient local execution
if sys.platform == "darwin":
    print("Apple Silicon Mac detected. Swapping Unsloth for mlx-tune...")
    from mlx_tune import FastLanguageModel
    from mlx_tune import SFTTrainer
    def is_bfloat16_supported(): return False
else:
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from unsloth import is_bfloat16_supported

# 1. Configuration
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# 2. Load Model
# Using the requested Qwen2.5-0.5B model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-0.5B",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# 3. Add LoRA Adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

# 4. Data Preparation
# We use the ChatML format, which produces higher quality models for instruction following.
# The `distill.ipynb` notebook generates data in this `"messages"` format.
from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "chatml",
)

def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }

# NOTE: Uncomment to load your newly distilled dataset
'''
dataset = load_dataset("json", data_files="data/train.jsonl", split="train")
dataset = dataset.map(formatting_prompts_func, batched = True,)
'''

print("Model & PEFT setup complete. Dataset loading and Trainer are currently commented out.")

# 5. Training Configuration
'''
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60, # Set num_train_epochs = 1 for full training runs
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)
trainer.train()
'''

# 6. Saving / Exporting Model
'''
# Local LoRA saving
model.save_pretrained("lora_model") 
tokenizer.save_pretrained("lora_model")

# Merge to 16bit (Required for Transformers.js WebGPU export)
print("Merging to 16-bit...")
model.save_pretrained_merged("model_16bit", tokenizer, save_method = "merged_16bit",)

# Export notes for Transformers.js:
# After saving the 16-bit merged model locally in 'model_16bit', 
# you can use the HuggingFace Optimum CLI to convert it to ONNX.
# Command: optimum-cli export onnx --model ./model_16bit ./onnx_output
'''
