"""Fine-tune one configured browser model per invocation with Unsloth."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

CONFIG_PATH = Path("config.json")
TRAIN_DATASET_PATH = Path("data/train.jsonl")
RUNS_DIR = Path("training_runs")


def load_config():
    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        config = json.load(config_file)

    models = config.get("finetune", {}).get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError("config.json must define at least one finetune.models entry.")
    return config, models


def parse_args(model_ids):
    parser = argparse.ArgumentParser(
        description="Fine-tune exactly one configured base model with Unsloth."
    )
    parser.add_argument("--model", choices=model_ids, help="Configured model ID to train.")
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List configured model IDs and checkpoints, then exit.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Limit training steps for a smoke run; -1 trains for configured epochs.",
    )
    args = parser.parse_args()

    if args.list_models and args.model:
        parser.error("--list-models and --model cannot be used together.")
    if not args.list_models and not args.model:
        parser.error("Choose one model with --model, or use --list-models.")
    if args.max_steps == 0 or args.max_steps < -1:
        parser.error("--max-steps must be -1 or a positive integer.")
    return args


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_run_metadata(path, metadata):
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main():
    config, models = load_config()
    args = parse_args(models)

    if args.list_models:
        for model_id, model_info in models.items():
            print(f"{model_id}: {model_info['base_model']}")
        return

    if not TRAIN_DATASET_PATH.is_file():
        raise FileNotFoundError(
            f"Training data not found at {TRAIN_DATASET_PATH}. "
            "Run distill_wiki.py, review data/staged_data.json, then run "
            "compile_dataset.py first."
        )

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "An NVIDIA CUDA GPU is required for Unsloth training. "
            "Install a CUDA-enabled PyTorch build and verify it with "
            "`python -c \"import torch; print(torch.cuda.is_available())\"`."
        )

    from datasets import load_dataset
    from transformers import TrainingArguments
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from trl import SFTTrainer

    selected_model = args.model
    model_info = models[selected_model]
    training_config = config["finetune"].get("training", {})
    max_seq_length = training_config.get("max_seq_length", 2048)
    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RUNS_DIR / selected_model / run_timestamp
    checkpoint_dir = run_dir / "checkpoints"
    adapter_dir = run_dir / "adapter"
    merged_dir = run_dir / "merged"
    run_dir.mkdir(parents=True, exist_ok=False)

    dataset = load_dataset("json", data_files=str(TRAIN_DATASET_PATH), split="train")
    if not dataset:
        raise ValueError(f"Training dataset is empty: {TRAIN_DATASET_PATH}")
    if "messages" not in dataset.column_names:
        raise ValueError(
            f"Training dataset must contain a 'messages' field: {TRAIN_DATASET_PATH}"
        )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_info["base_model"],
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=training_config.get("load_in_4bit", True),
    )

    if not tokenizer.chat_template:
        raise ValueError(
            f"{model_info['base_model']} has no tokenizer chat template. "
            "Configure and validate a model-specific template before training."
        )

    model = FastLanguageModel.get_peft_model(
        model,
        r=training_config.get("lora_rank", 16),
        target_modules=training_config.get(
            "target_modules",
            [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
        lora_alpha=training_config.get("lora_alpha", 16),
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=training_config.get("seed", 3407),
        use_rslora=False,
        loftq_config=None,
    )

    def format_conversations(examples):
        texts = [
            tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=False,
            )
            for conversation in examples["messages"]
        ]
        return {"text": texts}

    dataset = dataset.map(format_conversations, batched=True)

    run_metadata = {
        "status": "training",
        "model_id": selected_model,
        "base_model": model_info["base_model"],
        "source_revision": getattr(model.config, "_commit_hash", None),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(TRAIN_DATASET_PATH),
        "dataset_sha256": sha256_file(TRAIN_DATASET_PATH),
        "max_seq_length": max_seq_length,
        "max_steps": args.max_steps,
        "training_config": training_config,
        "software_versions": {
            "torch": torch.__version__,
            "transformers": version("transformers"),
            "trl": version("trl"),
            "unsloth": version("unsloth"),
        },
        "output_directories": {
            "checkpoints": str(checkpoint_dir),
            "adapter": str(adapter_dir),
            "merged": str(merged_dir),
        },
    }
    metadata_path = run_dir / "run.json"
    write_run_metadata(metadata_path, run_metadata)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=training_config.get("dataset_num_proc", 2),
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=training_config.get(
                "per_device_train_batch_size", 2
            ),
            gradient_accumulation_steps=training_config.get(
                "gradient_accumulation_steps", 4
            ),
            warmup_steps=training_config.get("warmup_steps", 5),
            num_train_epochs=training_config.get("num_train_epochs", 1),
            max_steps=args.max_steps,
            learning_rate=training_config.get("learning_rate", 2e-4),
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=training_config.get("logging_steps", 10),
            optim="adamw_8bit",
            weight_decay=training_config.get("weight_decay", 0.01),
            lr_scheduler_type="linear",
            seed=training_config.get("seed", 3407),
            output_dir=str(checkpoint_dir),
        ),
    )

    print(f"Training {selected_model} ({model_info['base_model']})...")
    trainer.train()

    print(f"Saving adapter to {adapter_dir}...")
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    if not hasattr(model, "save_pretrained_merged"):
        raise RuntimeError(
            "This Unsloth model does not support merged 16-bit saving, which is "
            "required before ONNX conversion."
        )

    print(f"Merging and saving the 16-bit model to {merged_dir}...")
    model.save_pretrained_merged(
        str(merged_dir),
        tokenizer,
        save_method="merged_16bit",
    )

    run_metadata["status"] = "completed"
    run_metadata["completed_at"] = datetime.now(timezone.utc).isoformat()
    write_run_metadata(metadata_path, run_metadata)
    print(f"Fine-tuning run complete: {run_dir}")


if __name__ == "__main__":
    main()
