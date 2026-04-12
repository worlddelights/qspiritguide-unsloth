import json

with open("distill.py", "r") as f:
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

# Add Best Practices Markdown Cell
best_practices = """## ChatML Data & Distillation Best Practices

As you review the `data/staged_data.json` buffer, keep the following ChatML best practices in mind to maximize the intelligence of your fine-tuned model:

1. **Maintain Consistent Persona:** Ensure the extracted "assistant" outputs always use the same tone. If your domain is spirituality, the voice should be consistently wise and structured.
2. **Remove Unnecessary Formatting:** Do not include metadata like "Here is the answer:" in the output. The output should purely be the informational response.
3. **Handle Ambiguity:** If the LM-Studio model flagged a chunk for review due to conflicting statements regarding quantum physics or new-age theory, explicitly rewrite the output to address *both* perspectives or synthesize a cohesive framework.
4. **Length Balancing:** Ensure the `instruction` length varies. Mix short questions (e.g. "Define X") with long instructions (e.g. "Synthesize a framework connecting Y and Z").
"""

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": best_practices.splitlines(True)
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

with open("distill.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)
