import os
import json
import hashlib
import re
from tqdm import tqdm
from openai import OpenAI

# Configuration
with open("config.json", "r") as f:
    CONFIG = json.load(f)

CLIENT = OpenAI(
    base_url=CONFIG["lm_studio"]["base_url"], 
    api_key=CONFIG["lm_studio"]["api_key"]
)
MODEL = CONFIG["lm_studio"]["model"]
RAW_DATA_DIR = "raw_data"
WIKI_DIR = "wiki"
STATE_FILE = ".wiki_state.json"
CHUNK_SIZE_WORDS = CONFIG["wiki"]["chunk_size_words"]

def get_file_hash(filepath):
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def chunk_text(text, size=CHUNK_SIZE_WORDS):
    words = text.split()
    for i in range(0, len(words), size):
        yield " ".join(words[i:i + size])

def call_llm(system_prompt, user_prompt):
    try:
        response = CLIENT.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return None

def extract_concepts(chunk):
    system_prompt = "You are a Master Librarian. Extract a clean list of core conceptual topics, entities, or spiritual/scientific principles from the provided text. Return ONLY a comma-separated list of titles."
    result = call_llm(system_prompt, f"Extract concepts from this text:\n\n{chunk}")
    if result:
        return [c.strip().replace("#", "").replace(".", "") for c in result.split(",") if c.strip()]
    return []

def process_concept(concept_name, chunk_text):
    safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', concept_name.lower().replace(" ", "_"))
    main_file = os.path.join(WIKI_DIR, f"{safe_name}.md")
    hist_file = os.path.join(WIKI_DIR, f"history_{safe_name}.md")
    
    existing_main = ""
    if os.path.exists(main_file):
        with open(main_file, 'r') as f:
            existing_main = f.read()
            
    existing_hist = ""
    if os.path.exists(hist_file):
        with open(hist_file, 'r') as f:
            existing_hist = f.read()

    system_prompt = f"""You are an expert synthesizer of the Seth material and metaphysical knowledge. 
Your goal is to maintain a 'Current Understanding' article for the concept: '{concept_name}'.

RULES:
1. FOCUS: Only include the most current, accurate, and stable understanding of the concept.
2. DISTRACTION-FREE: Stripped of any source URLs, mentions of 'The Seth Material', page numbers, or external attributions. Just the knowledge.
3. EVOLUTION: If the new text describes an evolution or change in how the concept was understood over time (e.g. 'Jane once thought X, but now knows Y'), provide that separately.
4. FORMAT: Return your response in two clearly marked blocks:
===CURRENT===
[The refined article text]
===HISTORY===
[Any historical context or changes in understanding]
"""

    user_prompt = f"Existing Article:\n{existing_main}\n\nExisting History:\n{existing_hist}\n\nNew Raw Information:\n{chunk_text}\n\nPlease synthesize the new information into the article."
    
    result = call_llm(system_prompt, user_prompt)
    if result:
        # Simple extraction logic for the blocks
        main_match = re.search(r'===CURRENT===\n(.*?)(?===HISTORY===|$)', result, re.DOTALL)
        hist_match = re.search(r'===HISTORY===\n(.*)', result, re.DOTALL)
        
        if main_match:
            new_main = main_match.group(1).strip()
            if new_main:
                with open(main_file, 'w') as f:
                    f.write(new_main)
                    
        if hist_match:
            new_hist = hist_match.group(1).strip()
            if new_hist and new_hist.lower() != "none" and len(new_hist) > 10:
                with open(hist_file, 'w') as f:
                    f.write(new_hist)

def main():
    if not os.path.exists(STATE_FILE):
        state = {}
    else:
        with open(STATE_FILE, 'r') as f:
            state = json.load(f)

    raw_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith(('.txt', '.md'))]
    print(f"Found {len(raw_files)} raw files: {raw_files}")
    
    for filename in tqdm(raw_files, desc="Processing raw files"):
        filepath = os.path.join(RAW_DATA_DIR, filename)
        current_hash = get_file_hash(filepath)
        
        if state.get(filename) == current_hash:
            print(f"Skipping {filename} (already processed)")
            continue
            
        print(f"\nProcessing {filename}...")
        with open(filepath, 'r') as f:
            data = f.read()
            
        chunks = list(chunk_text(data))
        print(f"Split {filename} into {len(chunks)} chunks")
        for chunk in chunks:
            concepts = extract_concepts(chunk)
            print(f"  Chunk concepts: {concepts}")
            for concept in concepts:
                print(f"    Processing concept: {concept}")
                process_concept(concept, chunk)
                
        state[filename] = current_hash
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)

if __name__ == "__main__":
    main()
