import argparse
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
    files = [f for f in os.listdir(WIKI_DIR) if f.endswith('.md') and not f.startswith('history_') and f != 'ignore']
    return files

def canonical_name(name):
    name = os.path.splitext(name)[0]
    name = name.strip().lower()
    name = re.sub(r'[_\-\s]+', ' ', name)
    name = re.sub(r'[^a-z0-9]+', '', name)
    return name

def build_name_map(files):
    name_map = {}
    collisions = {}
    for f in files:
        key = canonical_name(f)
        if key in name_map:
            group = collisions.setdefault(key, [])
            if name_map[key] not in group:
                group.append(name_map[key])
            if f not in group:
                group.append(f)
        else:
            name_map[key] = f
    return name_map, collisions


def auto_merge_collisions(collisions):
    for key, group in collisions.items():
        unique_group = []
        for f in group:
            if f not in unique_group:
                unique_group.append(f)

        if len(unique_group) < 2:
            continue

        primary = unique_group[0]
        print(f"Auto-merging canonical collision group for key '{key}': {unique_group}")
        for secondary in unique_group[1:]:
            if secondary == primary:
                continue
            merged = merge_concepts(primary, secondary)
            if not merged:
                print(f"Failed to auto-merge {secondary} into {primary}.")


def resolve_wiki_name(candidate, files, name_map):
    if candidate in files:
        return candidate

    key = canonical_name(candidate)
    if key in name_map:
        return name_map[key]
    return None

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
        return True

    return False

def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge wiki article files. If article names are provided, the first is the primary target and the rest are merged into it."
    )
    parser.add_argument(
        "articles",
        nargs="*",
        help="Wiki article filenames to merge. First is primary; remaining are merged into it."
    )
    return parser.parse_args()


def process_explicit_merge(articles, files, name_map):
    primary_candidate = articles[0]
    primary = resolve_wiki_name(primary_candidate, files, name_map)
    if not primary:
        print(f"Primary article not found: {primary_candidate}")
        return

    for secondary_candidate in articles[1:]:
        secondary = resolve_wiki_name(secondary_candidate, files, name_map)
        if not secondary:
            print(f"Skipping merge: article not found: {secondary_candidate}")
            continue
        if secondary == primary:
            print(f"Skipping merge: secondary article is the same as primary: {secondary_candidate}")
            continue
        merge_concepts(primary, secondary)


def main():
    args = parse_args()
    files = get_wiki_files()
    if not files:
        print("No wiki files found.")
        return

    name_map, collisions = build_name_map(files)
    if collisions:
        print("Warning: canonical name collisions detected for these file groups:")
        for key, group in collisions.items():
            print(f"  {key}: {sorted(set(group))}")

        auto_merge_collisions(collisions)
        files = get_wiki_files()
        name_map, collisions = build_name_map(files)
        if collisions:
            print("Warning: canonical name collisions remain after auto-merge:")
            for key, group in collisions.items():
                print(f"  {key}: {sorted(set(group))}")

    if args.articles:
        if len(args.articles) < 2:
            print("Please provide at least two article names when using explicit merge mode.")
            print("Usage: python refine_wiki.py primary.md secondary1.md [secondary2.md ...]")
            return

        process_explicit_merge(args.articles, files, name_map)
        return

    titles_str = ", ".join(files)
    system_prompt = (
        "You are an expert analyst. Look at this list of knowledge base article titles. "
        "Identify any titles that likely refer to the same concept or phenomena from different perspectives and should be merged. "
        "Return your suggestions as a JSON array of pairs: [['primary_file.md', 'duplicate_file.md'], ...]. "
        "Use the exact file names from the list and do not invent new names. If none, return []."
    )
    
    suggestions_raw = call_llm(system_prompt, f"Article Titles: {titles_str}")
    try:
        # Clean up JSON if LLM added markdown blocks
        json_str = re.search(r'\[.*\]', suggestions_raw, re.DOTALL).group(0)
        suggestions = json.loads(json_str)
        
        for primary, secondary in suggestions:
            resolved_primary = resolve_wiki_name(primary, files, name_map)
            resolved_secondary = resolve_wiki_name(secondary, files, name_map)

            if resolved_primary and resolved_secondary:
                if resolved_primary != primary:
                    print(f"Mapped primary '{primary}' -> '{resolved_primary}'")
                if resolved_secondary != secondary:
                    print(f"Mapped secondary '{secondary}' -> '{resolved_secondary}'")
                merge_concepts(resolved_primary, resolved_secondary)
            else:
                missing = []
                if not resolved_primary:
                    missing.append(primary)
                if not resolved_secondary:
                    missing.append(secondary)
                print(f"Skipping merge: {', '.join(missing)} not found (or could not be mapped to an existing file).")
                print(f"  Candidate pair: {primary}, {secondary}")
                print(f"  Existing wiki files: {titles_str}")
                
    except Exception as e:
        print(f"Failed to parse suggestions: {e}")
        print(f"Raw output: {suggestions_raw}")

if __name__ == "__main__":
    main()
