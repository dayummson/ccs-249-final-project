from transformers import pipeline
import re
import sys

sys.path.append("../../")

from utils.extract_blocks import extract_blocks_from_pdf, extract_blocks_from_docx

MODEL_PATH = "../final_requirement_model"

pipe = pipeline("text-classification", model=MODEL_PATH)

CONFIDENCE_THRESHOLD = 0.75

id_2_label = {
    "LABEL_0": "ADMIN_TRAP",
    "LABEL_1": "IGNORE",
    "LABEL_2": "OPTIONAL_BONUS",
    "LABEL_3": "PRE_REQUISITE",
    "LABEL_4": "TECHNICAL_TASK",
    "LABEL_5": "WEIGHTED_PRIORITY",
}


def heuristic_classify(text, block_meta=None):
    """Rule-based fallback — deterministic and expandable."""
    low = text.lower()
    meta = block_meta or {}

    # Scoring / rubric
    if re.search(r"\b\d+\s*(pts?|points?)\b", low):
        return (
            "OPTIONAL_BONUS"
            if "bonus" in low or "extra" in low
            else "WEIGHTED_PRIORITY"
        )

    # Admin noise
    if re.search(r"\b(name|section|date|submit|lms|email|attendance)\b", low):
        return "ADMIN_TRAP"
    if re.search(r"page \d+ of \d+|copyright|©|\bconfidential\b", low):
        return "IGNORE"

    # Technical verbs
    if re.search(
        r"\b(implement|extract|modify|write|code|create|develop|train|build|upload|push|commit)\b",
        low,
    ):
        return "TECHNICAL_TASK"

    # Prerequisite
    if re.search(
        r"\b(before|prerequisite|assume|required prior|install|setup|configure)\b", low
    ):
        return "PRE_REQUISITE"

    # Bonus
    if re.search(r"\b(optional|bonus|extra credit|challenge)\b", low):
        return "OPTIONAL_BONUS"

    # Structural hints: headings are usually TECHNICAL_TASK or ADMIN_TRAP
    if meta.get("is_heading") and meta.get("heading_level", 0) <= 2:
        return "TECHNICAL_TASK"

    return None  # No confident heuristic match


def classify_blocks(blocks):
    """blocks: list of dicts with 'text' + structural metadata."""
    texts = [b["text"] for b in blocks]
    model_results = pipe(texts)

    print("[MODEL RESULTS]: ", model_results)

    classified = []
    for block, result in zip(blocks, model_results):
        model_label = id_2_label.get(result["label"], "IGNORE")
        confidence = result["score"]

        # Try heuristic first for structural cues (always wins for these)
        heuristic = heuristic_classify(block["text"], block)

        if heuristic:
            final_label = heuristic
            source = "heuristic"
        elif confidence >= CONFIDENCE_THRESHOLD:
            final_label = model_label
            source = "model"
        else:
            # Low-confidence model: use heuristic or fall to IGNORE
            final_label = "IGNORE"
            source = "fallback"

        classified.append(
            {
                **block,
                "label": final_label,
                "confidence": confidence,
                "source": source,
            }
        )

    return classified


def group_to_briefing(classified_blocks):
    categories = {
        "TECHNICAL_TASK": [],
        "WEIGHTED_PRIORITY": [],
        "OPTIONAL_BONUS": [],
        "ADMIN_TRAP": [],
        "PRE_REQUISITE": [],
    }
    for block in classified_blocks:
        label = block["label"]
        if label in categories:
            categories[label].append(block["text"])
    return categories


def write_to_txt(briefing, filename="briefing.txt"):
    with open(filename, "w") as f:
        for category, items in briefing.items():
            f.write(f"{category}:\n")
            for item in items:
                f.write(f" - {item}\n")
            f.write("\n")


blocks = extract_blocks_from_pdf("../../samples/unit-2-activity.docx")

classified = classify_blocks(blocks)

briefing = group_to_briefing(classified)

# print(briefing)

write_to_txt(briefing)

print("Briefing generated and saved to briefing.txt")
