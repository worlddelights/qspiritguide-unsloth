"""Create an initial focused-corpus review from wiki_complete.

The source corpus is read-only. High-confidence, low-risk material is copied
to wiki_focused/articles; worldview-sensitive or ambiguous material is copied
to wiki_focused/review_queue. Files not copied are excluded from this pass.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


SOURCE_DIR = Path("wiki_complete")
FOCUSED_DIR = Path("wiki_focused")
ARTICLES_DIR = FOCUSED_DIR / "articles"
REVIEW_DIR = FOCUSED_DIR / "review_queue"

PRACTICAL_TERMS = {
    "acceptance",
    "active_listening",
    "agency",
    "autonomy",
    "boundaries",
    "career",
    "cognitive_bias",
    "commitment",
    "communication",
    "communication_skills",
    "conflict",
    "cooperation",
    "courage",
    "creative_process",
    "creativity",
    "decision",
    "deliberate_living",
    "direction",
    "emotions",
    "empathy",
    "equanimity",
    "expectancy_theory",
    "experience",
    "forgiveness",
    "freedom",
    "grief",
    "habits",
    "identity",
    "motivation",
    "optimism",
    "patience",
    "purpose",
    "reflection",
    "resilience",
    "values",
}

WORLDVIEW_TERMS = {
    "alchemy",
    "angel",
    "astrology",
    "aura",
    "buddh",
    "chakra",
    "christ",
    "contempl",
    "dharma",
    "divination",
    "dream",
    "esoteric",
    "faith",
    "gnostic",
    "god",
    "guru",
    "hermetic",
    "hindu",
    "kabb",
    "karma",
    "magic",
    "mystic",
    "occult",
    "prayer",
    "relig",
    "ritual",
    "spiritual",
    "symbol",
    "tao",
    "tantra",
    "theology",
    "tradition",
    "vedic",
    "zen",
    "abyss",
    "ahura",
    "ananke",
    "astral",
    "cosmic",
    "erebus",
    "esoteric",
    "heaven",
    "idolatry",
    "mazda",
    "panpsych",
    "sin",
    "tartarus",
}

SCIENCE_CONTEXT_TERMS = {
    "cognitive",
    "consciousness",
    "neuroscience",
    "psycholog",
    "quantum",
    "science",
    "scientific",
}

EXCLUDE_TERMS = {
    "abortion",
    "acupuncture",
    "addiction",
    "antidepress",
    "anti-seizure",
    "asbestos",
    "autoimmune",
    "clinical_trial",
    "central_sensitivity",
    "disease",
    "dysautonomia",
    "fecal",
    "fibromyalgia",
    "fatigue",
    "gut_microbiome",
    "medication",
    "medical",
    "medicine",
    "neuromodulation",
    "resistance_training",
    "seizure",
    "stigma_in_chronic_pain",
    "substance_use",
    "therapeutic",
    "vagus_nerve",
}

HIGH_RISK_TERMS = {
    "channeling",
    "coercive",
    "demon",
    "exorc",
    "fortune",
    "miracle",
    "prophecy",
    "psychic",
    "ufo",
    "extraterrestrial",
    "miraculous",
}

AMBIGUOUS_TERMS = {
    "anthropic",
    "consciousness",
    "cosmic",
    "digital",
    "energy",
    "gravity",
    "heisenberg",
    "physics",
    "quantum",
    "renormalization",
    "simulation",
    "state_vector",
    "symmetry",
    "theory",
}


def normalize_name(path: Path) -> str:
    return re.sub(r"[^a-z0-9]+", "_", path.stem.lower()).strip("_")


def clean_article(text: str) -> str:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped in {"=", "===", "---"}:
            continue
        if stripped.lower() in {"table of contents", "contents"}:
            continue
        cleaned.append(line.rstrip())

    result = "\n".join(cleaned).strip()
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result + "\n" if result else ""


def contains_term(value: str, terms: set[str]) -> bool:
    return any(term in value for term in terms)


def classify(path: Path, text: str) -> str:
    name = normalize_name(path)
    content = text.lower()
    combined = f"{name} {content[:4000]}"

    if contains_term(name, EXCLUDE_TERMS):
        return "exclude"

    has_worldview = contains_term(combined, WORLDVIEW_TERMS)
    has_risk = contains_term(combined, HIGH_RISK_TERMS)
    has_science = contains_term(combined, SCIENCE_CONTEXT_TERMS)
    has_practical = contains_term(combined, PRACTICAL_TERMS)
    has_ambiguity = contains_term(name, AMBIGUOUS_TERMS)

    if has_risk or has_ambiguity or (has_worldview and has_science):
        return "review_queue"
    if has_worldview:
        return "review_queue"
    if has_practical:
        return "articles"
    return "exclude"


def copy_candidates(reset: bool) -> dict[str, int]:
    ARTICLES_DIR.mkdir(parents=True, exist_ok=True)
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)

    if reset:
        for directory in (ARTICLES_DIR, REVIEW_DIR):
            for path in directory.glob("*.md"):
                path.unlink()

    counts = {"articles": 0, "review_queue": 0, "exclude": 0}
    for source in sorted(SOURCE_DIR.glob("*.md")):
        text = source.read_text(encoding="utf-8", errors="replace")
        decision = classify(source, text)
        counts[decision] += 1
        if decision == "exclude":
            continue

        destination_dir = ARTICLES_DIR if decision == "articles" else REVIEW_DIR
        destination = destination_dir / source.name
        destination.write_text(clean_article(text), encoding="utf-8")

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Remove generated Markdown candidates before creating the pass.",
    )
    args = parser.parse_args()

    if not SOURCE_DIR.is_dir():
        raise SystemExit(f"Source directory not found: {SOURCE_DIR}")

    counts = copy_candidates(reset=args.reset)
    print(f"Copied {counts['articles']} high-confidence articles.")
    print(f"Queued {counts['review_queue']} worldview-sensitive or ambiguous articles.")
    print(f"Excluded {counts['exclude']} articles from this initial pass.")


if __name__ == "__main__":
    main()
