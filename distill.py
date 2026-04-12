"""
Data Distillation Pipeline using LM-Studio
------------------------------------------
This script reads PDFs, chunks them, and uses a local LM-Studio server to extract high-quality ChatML Q&A pairs for fine-tuning.
It includes a conflict check against previously generated content, heavily focused on spirituality, new-age, and quantum physics.
"""

import fitz  # PyMuPDF
import json
import os
from openai import OpenAI

# 1. Configuration
with open("config.json", "r") as f:
    config = json.load(f)

lm_cfg = config["lm_studio"]
client = OpenAI(
    base_url=lm_cfg["base_url"],
    api_key=lm_cfg["api_key"]
)
MODEL_NAME = lm_cfg["model"]

RAW_DATA_DIR = "raw_data"
STAGED_FILE = "data/staged_data.json"
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs("data", exist_ok=True)

# 2. Chunking Logic
def extract_text_chunks(filepath, words_per_chunk=500):
    text = ""
    if filepath.lower().endswith('.pdf'):
        doc = fitz.open(filepath)
        for page in doc:
            text += page.get_text() + "\n"
    elif filepath.lower().endswith(('.md', '.txt')):
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
            
    words = text.split()
    for i in range(0, len(words), words_per_chunk):
        yield " ".join(words[i:i + words_per_chunk])

# 3. Conflict-Aware Prompting
def process_chunk(chunk, previous_context):
    """Sends chunk and previous context to LM-Studio for extraction and conflict checking."""
    
    # We pass the last 20 generated Q&A pairs as context to ensure consistency.
    context_str = json.dumps(previous_context[-20:], indent=2) if previous_context else "No prior context generated yet."
    
    system_prompt = f"""You are an expert data annotator specializing in spirituality, new-age material, and quantum physics frameworks. Your job is to extract educational Instruction/Response pairs from the provided text.

CRITICAL TASK: CONFLICT DETECTION
You must compare the new text and generated pairs against the `PREVIOUS_GENERATED_CONTEXT`. 
If a new fact or claim directly contradicts the previous context, OR if the text contains ambiguous, highly controversial, or internally inconsistent statements, you MUST flag it by setting "requires_review": true and explaining the conflict in "review_reason".

Return a STRICT JSON array of objects. Each object must have these exact keys:
[
  {{
    "instruction": "The question derived from text",
    "output": "The answer derived from text",
    "requires_review": false,
    "review_reason": "" 
  }}
]

PREVIOUS_GENERATED_CONTEXT:
{context_str}
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Extract Q&A from this new text:\n\n{chunk}"}
            ],
            temperature=0.3
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```json"):
            content = content[7:-3]
        elif content.startswith("```"):
            content = content[3:-3]
            
        return json.loads(content)
    except Exception as e:
        print(f"Error processing chunk: {e}")
        return []

# 4. Processing Loop
def main():
    print(f"Reading files from {RAW_DATA_DIR}...")
    files = [f for f in os.listdir(RAW_DATA_DIR) if f.lower().endswith((".pdf", ".md", ".txt"))]
    
    if not files:
        print(f"No source files found. Please drop them in {RAW_DATA_DIR}.")
        return

    # Load existing staged data to use as context
    staged_data = []
    if os.path.exists(STAGED_FILE):
        with open(STAGED_FILE, "r") as f:
            try:
                staged_data = json.load(f)
            except json.JSONDecodeError:
                pass

    for file in files:
        print(f"Processing {file}...")
        filepath = os.path.join(RAW_DATA_DIR, file)
        
        for chunk_idx, chunk in enumerate(extract_text_chunks(filepath)):
            print(f"  -> Distilling chunk {chunk_idx + 1}...")
            qa_pairs = process_chunk(chunk, staged_data)
            
            if qa_pairs:
                staged_data.extend(qa_pairs)
                # Overwrite staged file cumulatively
                with open(STAGED_FILE, "w") as f:
                    json.dump(staged_data, f, indent=2)
            
            print(f"     Found {len(qa_pairs)} pairs. Total Staged: {len(staged_data)}")
            
    print(f"Done! Staged data saved to {STAGED_FILE} for your review.")

# main()
