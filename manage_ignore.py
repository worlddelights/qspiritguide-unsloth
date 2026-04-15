#!/usr/bin/env python3
"""
Manage ignored concepts in wiki/ignore/ directory.

Usage:
    python manage_ignore.py list                  # Show all ignored concepts
    python manage_ignore.py ignore <concept>     # Move concept to ignore/
    python manage_ignore.py unignore <concept>   # Move concept back from ignore/
"""

import os
import sys
import shutil

WIKI_DIR = "wiki"
IGNORE_DIR = os.path.join(WIKI_DIR, "ignore")
HISTORY_DIR = os.path.join(WIKI_DIR, "history")
HISTORY_IGNORE_DIR = os.path.join(HISTORY_DIR, "ignore")

def ensure_ignore_dirs():
    """Create ignore directories if they don't exist."""
    os.makedirs(IGNORE_DIR, exist_ok=True)
    os.makedirs(HISTORY_IGNORE_DIR, exist_ok=True)

def list_ignored():
    """List all ignored concepts."""
    ensure_ignore_dirs()
    files = [f for f in os.listdir(IGNORE_DIR) if f.endswith('.md')]
    if not files:
        print("No ignored concepts.")
        return
    print("Ignored concepts:")
    for f in sorted(files):
        size = os.path.getsize(os.path.join(IGNORE_DIR, f))
        print(f"  - {f} ({size:,} bytes)")

def list_active():
    """List all active (non-ignored) concepts."""
    files = [f for f in os.listdir(WIKI_DIR) 
             if f.endswith('.md') and not f.startswith('history_')]
    if not files:
        print("No active concepts.")
        return
    print("Active concepts:")
    for f in sorted(files):
        size = os.path.getsize(os.path.join(WIKI_DIR, f))
        print(f"  - {f} ({size:,} bytes)")

def ignore_concept(concept_name):
    """Move a concept to ignore/ directory."""
    ensure_ignore_dirs()
    
    # Handle with or without .md extension
    if not concept_name.endswith('.md'):
        concept_name = concept_name + '.md'
    
    source = os.path.join(WIKI_DIR, concept_name)
    dest = os.path.join(IGNORE_DIR, concept_name)
    
    if not os.path.exists(source):
        print(f"Error: {concept_name} not found in wiki/")
        return False
    
    if os.path.exists(dest):
        print(f"Error: {concept_name} already in ignore/")
        return False
    
    shutil.move(source, dest)
    print(f"✓ Moved {concept_name} to wiki/ignore/")
    
    # Also move history file if it exists
    hist_source = os.path.join(HISTORY_DIR, concept_name)
    hist_dest = os.path.join(HISTORY_IGNORE_DIR, concept_name)
    
    if os.path.exists(hist_source):
        shutil.move(hist_source, hist_dest)
        print(f"✓ Moved history_{concept_name} to wiki/history/ignore/")
    
    return True

def unignore_concept(concept_name):
    """Move a concept back from ignore/ to wiki/."""
    ensure_ignore_dirs()
    
    # Handle with or without .md extension
    if not concept_name.endswith('.md'):
        concept_name = concept_name + '.md'
    
    source = os.path.join(IGNORE_DIR, concept_name)
    dest = os.path.join(WIKI_DIR, concept_name)
    
    if not os.path.exists(source):
        print(f"Error: {concept_name} not found in wiki/ignore/")
        return False
    
    if os.path.exists(dest):
        print(f"Error: {concept_name} already in wiki/")
        return False
    
    shutil.move(source, dest)
    print(f"✓ Moved {concept_name} back from wiki/ignore/")
    
    # Also move history file if it exists
    hist_source = os.path.join(HISTORY_IGNORE_DIR, concept_name)
    hist_dest = os.path.join(HISTORY_DIR, concept_name)
    
    if os.path.exists(hist_source):
        shutil.move(hist_source, hist_dest)
        print(f"✓ Moved history_{concept_name} back from wiki/history/ignore/")
    
    return True

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == 'list':
        print()
        list_active()
        print()
        list_ignored()
    elif command == 'active':
        list_active()
    elif command == 'ignore':
        if len(sys.argv) < 3:
            print("Usage: python manage_ignore.py ignore <concept>")
            sys.exit(1)
        ignore_concept(sys.argv[2])
    elif command == 'unignore':
        if len(sys.argv) < 3:
            print("Usage: python manage_ignore.py unignore <concept>")
            sys.exit(1)
        unignore_concept(sys.argv[2])
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)

if __name__ == "__main__":
    main()
