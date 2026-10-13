import os
import json
import re
from tqdm import tqdm
from openai import OpenAI

# 1. Configuration
with open("config.json", "r") as f:
    CONFIG = json.load(f)

CLIENT = OpenAI(
    base_url=CONFIG["lm_studio"]["base_url"], 
    api_key=CONFIG["lm_studio"]["api_key"]
)
MODEL = CONFIG["lm_studio"]["model"]
WIKI_DIR = os.path.join("wiki_focused", "articles")
STAGED_FILE = "data/staged_data.json"

# 2. Content Validation & LLM Call
def is_meaningful_content(text):
    """Validate that text is meaningful (not junk or placeholder)."""
    if not isinstance(text, str):
        return False
    text = text.strip()
    return (
        len(text) > 100 and  # At least 100 characters
        bool(re.search(r'[a-zA-Z]{3,}', text))  # Contains at least one 3+ letter word
    )

def call_llm(system_prompt, user_prompt):
    try:
        response = CLIENT.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7 # Higher temperature for diverse Q&A
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return None

# 3. Distillation Loop
def get_file_hash(filepath):
    import hashlib
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Distill approved, focused wiki articles into training Q&A pairs."
    )
    parser.add_argument("--force", "-f", action="store_true", help="Force distillation of all articles regardless of cache.")
    parser.add_argument("--clear", action="store_true", help="Clear the distillation state cache and exit.")
    args = parser.parse_args()

    DISTILL_STATE_FILE = ".distill_state.json"

    if args.clear:
        if os.path.exists(DISTILL_STATE_FILE):
            os.remove(DISTILL_STATE_FILE)
            print("Cleared distillation state cache.")
        else:
            print("No state cache found to clear.")
        return

    if not os.path.exists("data"):
        os.makedirs("data")

    # Only articles explicitly approved by moving them into this directory are eligible.
    wiki_files = [f for f in os.listdir(WIKI_DIR) if f.endswith('.md') and not f.startswith('history_')]
    
    if not wiki_files:
        print("No wiki files found to distill.")
        return

    # Load existing state cache
    state = {}
    if not args.force and os.path.exists(DISTILL_STATE_FILE):
        try:
            with open(DISTILL_STATE_FILE, 'r') as f:
                state = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load state file: {e}")

    # Determine files to process
    pending_files = []
    for filename in wiki_files:
        filepath = os.path.join(WIKI_DIR, filename)
        current_hash = get_file_hash(filepath)
        if state.get(filename) != current_hash:
            pending_files.append((filename, filepath, current_hash))

    print(f"Found {len(wiki_files)} wiki files. {len(pending_files)} pending distillation.")
    if not pending_files:
        print("Everything is up to date.")
        return

    staged_data = []
    if os.path.exists(STAGED_FILE):
        with open(STAGED_FILE, 'r') as f:
            try:
                staged_data = json.load(f)
            except:
                pass

    processed_count = 0
    for filename, filepath, current_hash in tqdm(pending_files, desc="Distilling wiki articles"):
        with open(filepath, "rb") as f:
            raw_content = f.read()
        try:
            content = raw_content.decode("utf-8-sig")
        except UnicodeDecodeError:
            content = raw_content.decode("cp1252", errors="replace")

        system_prompt = """You are an expert educational data generator. 
Your task is to generate 5-10 high-quality, complex Instruction/Response pairs based ONLY on the provided knowledge article.
Guidelines:
1. Complexity: Questions should be nuanced and require deep understanding of the concepts.
2. Formats: Use diverse styles (e.g., 'Explain the relationship between...', 'How does X affect Y?', 'Contrast A and B').
3. Voice: The Assistant should be knowledgeable, clear, and profound.
4. Output: Return a STRICT JSON array of objects with keys: "instruction", "output".
"""

        user_prompt = f"Article: {filename}\nContent:\n{content}\n\nGenerate 5-10 ChatML Q&A pairs."
        
        result = call_llm(system_prompt, user_prompt)
        if result:
            try:
                # Clean up JSON if LLM added markdown blocks
                json_str = re.search(r'\[.*\]', result, re.DOTALL).group(0)
                pairs = json.loads(json_str)
                
                # Validate and add only meaningful pairs
                valid_pairs = []
                for p in pairs:
                    if (isinstance(p, dict) and 
                        is_meaningful_content(p.get("instruction", "")) and
                        is_meaningful_content(p.get("output", ""))):
                        # Add review flag as in original pipeline
                        p["requires_review"] = False
                        p["review_reason"] = ""
                        # Track the source wiki file for easy caching / editing
                        p["source_file"] = filename
                        valid_pairs.append(p)
                
                # Filter out any existing staged pairs for this specific source file to ensure idempotency
                staged_data = [item for item in staged_data if item.get("source_file") != filename]
                staged_data.extend(valid_pairs)
                
                # Cumulative save
                with open(STAGED_FILE, 'w') as f:
                    json.dump(staged_data, f, indent=2)

                # Update the state file
                state[filename] = current_hash
                with open(DISTILL_STATE_FILE, 'w') as f:
                    json.dump(state, f, indent=2)

                processed_count += 1
            except Exception as e:
                print(f"Failed to parse Q&A for {filename}: {e}")

    print(f"Done! Processed {processed_count} files. Total staged pairs: {len(staged_data)}")

if __name__ == "__main__":
    main()
