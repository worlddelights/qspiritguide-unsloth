import os
import json
import re
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

def get_wiki_files():
    files = [f for f in os.listdir(WIKI_DIR) if f.endswith('.md') and not f.startswith('history_')]
    return files

def merge_concepts(primary, secondary):
    print(f"Merging {secondary} into {primary}...")
    
    primary_path = os.path.join(WIKI_DIR, primary)
    secondary_path = os.path.join(WIKI_DIR, secondary)
    
    with open(primary_path, 'r') as f:
        primary_content = f.read()
    with open(secondary_path, 'r') as f:
        secondary_content = f.read()
        
    system_prompt = "You are a Master Librarian. You have two articles that represent different perspectives on the same phenomenon. Merge them into a single, unified, high-quality 'Current Understanding' article. Maintain the distraction-free, source-free style."
    
    user_prompt = f"Article 1 ({primary}):\n{primary_content}\n\nArticle 2 ({secondary}):\n{secondary_content}"
    
    result = call_llm(system_prompt, user_prompt)
    if result:
        with open(primary_path, 'w') as f:
            f.write(result)
        os.remove(secondary_path)
        
        # Also check for history files
        primary_hist = os.path.join(WIKI_DIR, f"history_{primary}")
        secondary_hist = os.path.join(WIKI_DIR, f"history_{secondary}")
        
        if os.path.exists(secondary_hist):
            with open(secondary_hist, 'r') as f:
                s_hist = f.read()
            p_hist = ""
            if os.path.exists(primary_hist):
                with open(primary_hist, 'r') as f:
                    p_hist = f.read()
            
            hist_merge_prompt = "Merge these two historical evolution summaries into one."
            hist_result = call_llm(hist_merge_prompt, f"Hist 1:\n{p_hist}\n\nHist 2:\n{s_hist}")
            if hist_result:
                with open(primary_hist, 'w') as f:
                    f.write(hist_result)
            os.remove(secondary_hist)

def main():
    files = get_wiki_files()
    if not files:
        print("No wiki files found.")
        return

    titles_str = ", ".join(files)
    system_prompt = "You are an expert analyst. Look at this list of knowledge base article titles. Identify any titles that likely refer to the same concept or phenomena from different perspectives and should be merged. Return your suggestions as a JSON array of pairs: [['primary_file.md', 'duplicate_file.md'], ...]. If none, return []."
    
    suggestions_raw = call_llm(system_prompt, f"Article Titles: {titles_str}")
    
    try:
        # Clean up JSON if LLM added markdown blocks
        json_str = re.search(r'\[.*\]', suggestions_raw, re.DOTALL).group(0)
        suggestions = json.loads(json_str)
        
        for primary, secondary in suggestions:
            if os.path.exists(os.path.join(WIKI_DIR, primary)) and os.path.exists(os.path.join(WIKI_DIR, secondary)):
                merge_concepts(primary, secondary)
            else:
                print(f"Skipping merge: {primary} or {secondary} not found.")
                
    except Exception as e:
        print(f"Failed to parse suggestions: {e}")
        print(f"Raw output: {suggestions_raw}")

if __name__ == "__main__":
    main()
