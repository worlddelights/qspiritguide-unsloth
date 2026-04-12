"""
ChatML Compilation Script
-------------------------
This scripts reads your reviewed `staged_data.json` and compiles it into the final `data/train.jsonl` dataset in ChatML format.
"""

import json
import os

STAGED_FILE = "data/staged_data.json"
FINAL_FILE = "data/train.jsonl"

def compile_dataset():
    if not os.path.exists(STAGED_FILE):
        print(f"Error: {STAGED_FILE} not found. Run the distillation notebook first.")
        return

    with open(STAGED_FILE, "r") as f:
        staged_data = json.load(f)

    compiled_count = 0
    skipped_count = 0

    with open(FINAL_FILE, "a") as f:
        for item in staged_data:
            # Check if it was flagged for review and never cleared.
            # You can clear it by setting "requires_review": false in the JSON.
            if item.get("requires_review"):
                skipped_count += 1
                continue
            
            # Format to ChatML Unsloth structure
            conversation = {
                "messages": [
                    {"role": "user", "content": item.get("instruction", "")},
                    {"role": "assistant", "content": item.get("output", "")}
                ]
            }
            f.write(json.dumps(conversation) + "\n")
            compiled_count += 1
            
    print(f"Compilation Complete!")
    print(f" -> Added {compiled_count} parsed Q&A pairs to {FINAL_FILE}")
    print(f" -> Skipped {skipped_count} pairs that still require review.")

if __name__ == "__main__":
    compile_dataset()
