import sys

sys.path.append("../../")

import os
import re
from transformers import pipeline
from constants.label import LABELS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "../../models/alpha"))

print(f"Loading model from: {MODEL_PATH}")


# LOAD MODEL AT STARTUP
print("Loading model...")
pipe = pipeline("text-classification", model=MODEL_PATH)
print("Model ready.")

CONFIDENCE_THRESHOLD = 0.75


def heuristic_classify(text, meta=None):
    low = text.lower().strip()
    meta = meta or {}

    # HARD IGNORE
    # TODO: TRAIN ANOTHER MODEL CLASSIFYING THESE "NOISE" BLOCKS
    # THEN USE THAT AS A HEURISTIC SIGNAL INSTEAD OF HARD PATTERNS
    ignore_patterns = [
        r"wvsu\.edu\.ph",
        r"west visayas state university",
        r"college of information",
        r"luna st\.|la paz|iloilo",
        r"trunkline|telefax",
        r"^ccs \d{3}\b",
        r"^page \d+ of \d+$",
        r"^©|copyright",
        r"^\d+$",
        r"^https?://",
        r"www\.[a-z]+\.[a-z]",
    ]
    for pattern in ignore_patterns:
        if re.search(pattern, low):
            return "IGNORE"

    # Short bold headings → section titles → IGNORE
    if meta.get("is_heading") and len(text.split()) <= 6:
        return "IGNORE"

    # HARD WEIGHTED_PRIORITY
    if re.search(r"\b\d+\s*(pts?|points?)\b", low):
        if re.search(r"\b(bonus|extra|optional|challenge)\b", low):
            return "OPTIONAL_BONUS"
        return "WEIGHTED_PRIORITY"

    if re.search(r"project score|score raw|eval score", low):
        return "WEIGHTED_PRIORITY"

    # HARD OPTIONAL_BONUS
    if re.search(r"\b(optional|bonus|extra credit|challenge)\b", low):
        return "OPTIONAL_BONUS"

    # HARD ADMIN_TRAP
    admin_patterns = [
        r"\b(name|section|date)\s*[:\-_]{1,3}\s*_{2,}",
        r"\bsubmit\b.*\b(lms|canvas|google|email)\b",
        r"\blms\b",
        r"\b(late submission|deadline|due date|due by)\b",
        r"\bfork the repository\b",
        r"\bpull request\b|\bcreate a pr\b",
        r"\bgithub (repository|organization)\b",
        r"\bself and peer (assessment|evaluation)\b",
        r"\bteam leader\b",
        r"\bpresentation time\b",
        r"\bq&a session\b",
        r"\bscheduled date\b",
        r"\bai.?generated|not ai\b",
        r"\bmake sure.*(yours|original|not copied)\b",
        r"\bacademic integrity\b",
        r"\bdistributed to everyone\b",
    ]
    for pattern in admin_patterns:
        if re.search(pattern, low):
            return "ADMIN_TRAP"

    # HARD TECHNICAL_TASK
    if re.search(
        r"\b(implement|extract|modify|write|code|create|develop|train|"
        r"build|upload|push|commit|generate|calculate|classify|tokenize|"
        r"preprocess|vectorize|fine.?tune|design|construct)\b",
        low,
    ):
        return "TECHNICAL_TASK"

    # HARD PRE_REQUISITE
    if re.search(
        r"\b(before starting|prior to|prerequisite|install|make sure|"
        r"ensure that|you (must|should) have|is required before)\b",
        low,
    ):
        return "PRE_REQUISITE"

    return None


def classify_blocks(blocks):
    texts = [b["text"] for b in blocks]
    results = pipe(texts)

    classified = []
    for block, result in zip(blocks, results):
        model_label = LABELS.get(result["label"], "IGNORE")
        confidence = result["score"]

        heuristic = heuristic_classify(block["text"], block)

        if heuristic:
            final_label = heuristic
            source = "heuristic"
        elif confidence >= CONFIDENCE_THRESHOLD:
            final_label = model_label
            source = "model"
        else:
            final_label = "IGNORE"
            source = "low_confidence"

        classified.append(
            {
                **block,
                "label": final_label,
                "confidence": round(confidence, 3),
                "source": source,
            }
        )

    return classified


def group_to_briefing(classified):
    briefing = {
        "TECHNICAL_TASK": [],
        "WEIGHTED_PRIORITY": [],
        "OPTIONAL_BONUS": [],
        "PRE_REQUISITE": [],
        "ADMIN_TRAP": [],
    }
    for block in classified:
        label = block["label"]
        if label in briefing:
            briefing[label].append(block["text"])
    return briefing
