"""
Web Page to Markdown Fetcher
----------------------------
This script reads URLs from `urls.txt`, downloads them, extracts their main content
into Markdown using `trafilatura`, and saves the `.md` files in the `raw_data/` directory.

No proxies or cloud APIs are used; all extraction runs locally.
"""

import os
import trafilatura
from urllib.parse import urlparse
import re

URLS_FILE = "urls.txt"
OUTPUT_DIR = "raw_data"

def sanitize_filename(url):
    """Generate a clean filename from a URL."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.replace("www.", "")
        path = parsed.path.strip("/").replace("/", "_")
        
        if not path:
            name = domain
        else:
            name = f"{domain}_{path}"
            
        # keep alphanumeric, dash, underscore
        name = re.sub(r'[^a-zA-Z0-9_\-]', '_', name)
        
        # basic truncation if it's wildly long
        return name[:100] + ".md"
    except Exception:
        # fallback
        return "downloaded_page.md"

def fetch_url(url):
    print(f"Fetching {url}...")
    downloaded = trafilatura.fetch_url(url)
    
    if downloaded is None:
        print(f"  -> Failed to download {url}")
        return False
        
    # Extract clean text and format as markdown
    result = trafilatura.extract(downloaded, output_format='markdown')
    
    if result is None:
        print(f"  -> Failed to extract readable content from {url}")
        return False
        
    filename = sanitize_filename(url)
    out_path = os.path.join(OUTPUT_DIR, filename)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# Source: {url}\n\n")
        f.write(result)
        
    print(f"  -> Saved to {out_path}")
    return True

def comment_out_line(line_index):
    """Prepend '# ' to the line at line_index (0-based) in URLS_FILE."""
    with open(URLS_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
    lines[line_index] = "# " + lines[line_index]
    with open(URLS_FILE, "w", encoding="utf-8") as f:
        f.writelines(lines)


def main():
    if not os.path.exists(URLS_FILE):
        print(f"File {URLS_FILE} not found. Creating it.")
        with open(URLS_FILE, "w", encoding="utf-8") as f:
            f.write("# Add URLs you want to download below, one per line.\n")
        print(f"Please add some URLs to {URLS_FILE} and run again.")
        return

    with open(URLS_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Collect (line_index, url) for active (non-comment, non-empty) lines
    pending = [
        (i, line.strip())
        for i, line in enumerate(lines)
        if line.strip() and not line.strip().startswith("#")
    ]

    if not pending:
        print(f"No valid URLs found in {URLS_FILE}.")
        return

    success_count = 0
    for line_index, url in pending:
        if fetch_url(url):
            success_count += 1
            comment_out_line(line_index)
            print(f"  -> Commented out line {line_index + 1} in {URLS_FILE}")

    print(f"\nFinished! Successfully downloaded and extracted {success_count} out of {len(pending)} URLs.")


if __name__ == "__main__":
    main()
