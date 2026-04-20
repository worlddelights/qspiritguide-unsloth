import os
import json
import hashlib
import re
from tqdm import tqdm
import fitz  # PyMuPDF
try:
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup
    EPUB_SUPPORT = True
except ImportError:
    EPUB_SUPPORT = False
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
HISTORY_DIR = os.path.join(WIKI_DIR, "history")
IGNORE_DIR = os.path.join(WIKI_DIR, "ignore")
STATE_FILE = ".wiki_state.json"
TEMPERATURE = CONFIG["wiki"].get("temperature", 0.2)
ENABLE_THINKING = CONFIG["wiki"].get("enable_thinking", False)

# Calculate chunk_size and max_tokens from max_context_token
MAX_CONTEXT_TOKEN = CONFIG["wiki"].get("max_context_token", 4096)
# Reserve ~33% for output tokens, use ~67% for input context
MAX_TOKENS = max(512, MAX_CONTEXT_TOKEN // 3)
# Estimate ~1.3 tokens per word on average
INPUT_CONTEXT_TOKENS = MAX_CONTEXT_TOKEN * 2 // 3
CHUNK_SIZE_WORDS = max(500, INPUT_CONTEXT_TOKENS // 1.3)

# Convert to int for clarity
CHUNK_SIZE_WORDS = int(CHUNK_SIZE_WORDS)
MAX_TOKENS = int(MAX_TOKENS)

# Display calculated parameters at startup
print("[Config] Parameters:")
print(f"  max_context_token (from config): {MAX_CONTEXT_TOKEN}")
print(f"  temperature (from config): {TEMPERATURE}")
print(f"  enable_thinking (from config): {ENABLE_THINKING}")
print("[Calculated] Derived parameters:")
print(f"  max_tokens: {MAX_TOKENS} (33% of context for output)")
print(f"  chunk_size_words: {CHUNK_SIZE_WORDS} (67% of context for input, ~1.3 tokens/word)")
print()

def get_file_hash(filepath):
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def extract_text_from_pdf(filepath):
    """Extract plain text from a PDF using PyMuPDF."""
    try:
        doc = fitz.open(filepath)
    except Exception as e:
        raise RuntimeError(f"Unable to open PDF '{filepath}': {e}") from e

    parts = []
    try:
        for page_number, page in enumerate(doc, start=1):
            text = page.get_text("text") or page.get_text()
            if not text:
                blocks = page.get_text("blocks")
                if isinstance(blocks, list):
                    text = "\n".join(block[4] for block in blocks if len(block) > 4 and block[4])
            parts.append(text or "")
    finally:
        doc.close()

    return "\n".join(parts)

def extract_text_from_epub(filepath):
    """Extract plain text from an EPUB using ebooklib + BeautifulSoup."""
    if not EPUB_SUPPORT:
        raise ImportError(
            "ebooklib and beautifulsoup4 are required for EPUB support. "
            "Install them with: uv add ebooklib beautifulsoup4"
        )
    book = epub.read_epub(filepath)
    parts = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        if text:
            parts.append(text)
    return "\n".join(parts)

def read_raw_file(filepath):
    """Read a raw file and return its text content, dispatching by extension."""
    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext == ".pdf":
            return extract_text_from_pdf(filepath)
        elif ext == ".epub":
            return extract_text_from_epub(filepath)
        else:  # .txt, .md
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
    except Exception as e:
        raise RuntimeError(f"Failed to read raw file '{filepath}': {e}") from e


def chunk_text(text, size=CHUNK_SIZE_WORDS):
    words = text.split()
    for i in range(0, len(words), size):
        yield " ".join(words[i:i + size])

def call_llm(system_prompt, user_prompt):
    """Call the LLM. Raises on error — callers must handle failures explicitly."""
    request_kwargs = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }
    
    # Add thinking parameter if enabled (OpenAI-compatible API)
    if ENABLE_THINKING:
        request_kwargs["thinking"] = {"type": "enabled", "budget_tokens": MAX_CONTEXT_TOKEN // 2}
    
    response = CLIENT.chat.completions.create(**request_kwargs)
    
    # Extract text content, skipping any thinking blocks
    if hasattr(response.choices[0].message, 'content') and isinstance(response.choices[0].message.content, list):
        # Multiple content blocks (thinking + text)
        text_content = ""
        for block in response.choices[0].message.content:
            if block.type == "text":
                text_content += block.text + " "
        return text_content.strip()
    else:
        # Single text response
        return response.choices[0].message.content.strip()

def extract_concepts(chunk):
    """Extract concept names from a chunk. Raises on LLM failure."""
    system_prompt = "You are a Master Librarian. Extract a clean list of core conceptual topics, entities, or spiritual/scientific principles from the provided text. Return ONLY a comma-separated list of titles."
    result = call_llm(system_prompt, f"Extract concepts from this text:\n\n{chunk}")
    return [c.strip().replace("#", "").replace(".", "") for c in result.split(",") if c.strip()]

def process_concept(concept_name, chunk_text):
    """Write/update wiki files for a concept. Raises on LLM failure."""
    safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', concept_name.lower().replace(" ", "_"))
    main_file = os.path.join(WIKI_DIR, f"{safe_name}.md")
    hist_file = os.path.join(HISTORY_DIR, f"{safe_name}.md")
    
    # Skip if concept is in ignore directory
    ignore_file = os.path.join(IGNORE_DIR, f"{safe_name}.md")
    if os.path.exists(ignore_file):
        return  # Concept is explicitly ignored

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
4. CRITICAL: If no meaningful new content exists, just repeat the existing article verbatim in the ===CURRENT=== section.
5. FORMAT: Return your response in two clearly marked blocks:
===CURRENT===
[The refined article text]
===HISTORY===
[Any historical context or changes in understanding]
"""

    user_prompt = f"Existing Article:\n{existing_main}\n\nExisting History:\n{existing_hist}\n\nNew Raw Information:\n{chunk_text}\n\nPlease synthesize the new information into the article."

    result = call_llm(system_prompt, user_prompt)  # raises on error

    # Extract the two blocks
    main_match = re.search(r'===CURRENT===\n(.*?)(?===HISTORY===|$)', result, re.DOTALL)
    hist_match = re.search(r'===HISTORY===\n(.*)', result, re.DOTALL)

    if main_match:
        new_main = main_match.group(1).strip()
        # Check for placeholder messages that indicate no real change
        placeholder_phrases = [
            "no specific information",
            "no new information",
            "remains unchanged",
            "no additional information",
            "no further information",
            "no conceptual updates",
            "no updates were provided",
            "no content provided",
            "no material",
            "nothing to add",
            "no relevant",
        ]
        is_placeholder = any(phrase.lower() in new_main.lower() for phrase in placeholder_phrases)
        
        # Also check for generic parenthesized "no" messages
        if not is_placeholder and re.match(r'^\([^)]*[Nn]o\s+\w+', new_main):
            is_placeholder = True
        
        if new_main and not is_placeholder:
            with open(main_file, 'w') as f:
                f.write(new_main)
        elif is_placeholder and not existing_main:
            # If it's a placeholder and there's no existing content, still write it
            # (better to have something than nothing)
            pass

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

    os.makedirs(WIKI_DIR, exist_ok=True)
    os.makedirs(HISTORY_DIR, exist_ok=True)
    os.makedirs(IGNORE_DIR, exist_ok=True)
    raw_files = sorted([f for f in os.listdir(RAW_DATA_DIR) if f.endswith(('.txt', '.md', '.pdf', '.epub'))])
    pending = [f for f in raw_files if state.get(f) != get_file_hash(os.path.join(RAW_DATA_DIR, f))]
    print(f"Found {len(raw_files)} raw files total, {len(pending)} pending.")

    for filename in raw_files:
        filepath = os.path.join(RAW_DATA_DIR, filename)
        current_hash = get_file_hash(filepath)

        if state.get(filename) == current_hash:
            print(f"Skipping {filename} (already processed)")
            continue

        print(f"\nProcessing {filename}...")
        data = read_raw_file(filepath)

        chunks = list(chunk_text(data))
        print(f"Split {filename} into {len(chunks)} chunks")
        if not chunks:
            raise ValueError(
                f"No text chunks were created for '{filename}'. "
                f"Extracted text length: {len(data)} characters. "
                "The PDF may contain only scanned images or otherwise unreadable text."
            )
        file_ok = True
        for i, chunk in enumerate(chunks, 1):
            print(f"  Chunk {i}/{len(chunks)}")
            try:
                concepts = extract_concepts(chunk)
            except Exception as e:
                print(f"  ❌ LLM error during concept extraction (chunk {i}): {e}")
                file_ok = False
                break
            print(f"  Concepts: {concepts}")
            for concept in concepts:
                print(f"    Processing concept: {concept}")
                try:
                    process_concept(concept, chunk)
                except Exception as e:
                    print(f"    ❌ LLM error processing '{concept}' (chunk {i}): {e}")
                    file_ok = False
                    break
            if not file_ok:
                break

        if not file_ok:
            print(f"\n⚠️  Halting — LLM error encountered while processing '{filename}'. State NOT updated.")
            return

        state[filename] = current_hash
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)

        remaining = [f for f in raw_files if state.get(f) != get_file_hash(os.path.join(RAW_DATA_DIR, f))]
        print(f"\n✅  Done with '{filename}'.")
        if remaining:
            print(f"   {len(remaining)} file(s) still pending: {remaining}")
            print("   Review the wiki/ output, then re-run to continue.")
        else:
            print("   All files have been processed.")
        return  # exit after one new file so the human can review

if __name__ == "__main__":
    main()
