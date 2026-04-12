import json

with open("finetune.py", "r") as f:
    code = f.read()

# Separate the code into cells (roughly by top-level comments)
blocks = code.split("\n# ")
cells = []

# First block (imports and header)
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [blocks[0]]
})

# Remaining blocks
for block in blocks[1:]:
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": ["# " + block]
    })

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("finetune.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)
