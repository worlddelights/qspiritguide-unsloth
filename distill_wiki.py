import os
import json
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
WIKI_DIR = "wiki"
STAGED_FILE = "data/staged_data.json"

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

def main():
    if not os.path.exists("data"):
        os.makedirs("data")

    # Get all "Current Understanding" articles (exclude history_)
    wiki_files = [f for f in os.listdir(WIKI_DIR) if f.endswith('.md') and not f.startswith('history_')]
    
    if not wiki_files:
        print("No wiki files found to distill.")
        return

    staged_data = []
    if os.path.exists(STAGED_FILE):
        with open(STAGED_FILE, 'r') as f:
            try:
                staged_data = json.load(f)
            except:
                pass

    for filename in tqdm(wiki_files, desc="Distilling wiki articles"):
        filepath = os.path.join(WIKI_DIR, filename)
        with open(filepath, 'r') as f:
            content = f.read()

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
                for p in pairs:
                    # Add review flag as in original pipeline
                    p["requires_review"] = False
                    p["review_reason"] = ""
                
                staged_data.extend(pairs)
                
                # Cumulative save
                with open(STAGED_FILE, 'w') as f:
                    json.dump(staged_data, f, indent=2)
            except Exception as e:
                print(f"Failed to parse Q&A for {filename}: {e}")

    print(f"Done! Total staged pairs: {len(staged_data)}")

if __name__ == "__main__":
    main()
