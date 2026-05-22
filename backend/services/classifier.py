import sys

sys.path.append("../../")

import os
import re
from transformers import pipeline
from constants.label import LABELS, RAW_ID2LABEL

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../models"))
DEFAULT_MODEL_NAME = os.getenv("CLASSIFIER_MODEL", "firewolf")
OLLAMA_MODELS = {"mistral:latest", "llama3.2:latest"}

MODEL_CACHE = {}
ACTIVE_MODEL_NAME = None
ACTIVE_MODEL_KIND = None
pipe = None

CONFIDENCE_THRESHOLD = 0.75

CLASSIFY_PROMPT = """Classify the following sentence from a university activity sheet into exactly one of these six categories:

TECHNICAL_TASK     - Instructions to build, implement, code, write, create, train, extract, or calculate something
WEIGHTED_PRIORITY  - Mentions points, grades, scores, or rubrics
ADMIN_TRAP         - Submission rules, deadlines, GitHub/LMS instructions, grouping, formatting requirements
OPTIONAL_BONUS     - Optional, bonus, or extra credit items
PRE_REQUISITE      - Setup steps or conditions before starting
IGNORE             - Headers, footers, URLs, university name, page numbers, institutional text

Reply with only the category name, nothing else. No explanation. No punctuation. Just one of the six category names above.

Sentence: "{sentence}"""


def _list_local_models():
    try:
        return {
            name
            for name in os.listdir(MODELS_DIR)
            if os.path.isdir(os.path.join(MODELS_DIR, name))
        }
    except FileNotFoundError:
        return set()


def _resolve_model(model_name):
    local_models = _list_local_models()
    requested = model_name or DEFAULT_MODEL_NAME

    if requested in OLLAMA_MODELS:
        return "ollama", requested

    if requested in local_models:
        return "hf", os.path.join(MODELS_DIR, requested)

    if os.path.exists(requested):
        return "hf", os.path.abspath(requested)

    if DEFAULT_MODEL_NAME in OLLAMA_MODELS:
        return "ollama", DEFAULT_MODEL_NAME

    if DEFAULT_MODEL_NAME in local_models:
        return "hf", os.path.join(MODELS_DIR, DEFAULT_MODEL_NAME)

    return "hf", requested


def _load_hf_pipeline(model_path):
    if model_path not in MODEL_CACHE:
        MODEL_CACHE[model_path] = pipeline("text-classification", model=model_path)
    return MODEL_CACHE[model_path]


def _extract_label(raw_response):
    cleaned = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL)
    cleaned = cleaned.strip().upper()

    if cleaned in LABELS:
        return cleaned

    for label in LABELS:
        if label in cleaned:
            return label

    return "IGNORE"


def _ollama_predict(texts, model_name):
    try:
        import ollama
    except Exception as exc:
        raise RuntimeError("Ollama is not installed or not available.") from exc

    results = []
    for text in texts:
        prompt = CLASSIFY_PROMPT.format(sentence=text)
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0},
        )
        raw = response["message"]["content"]
        label = _extract_label(raw)
        results.append({"label": label, "score": 1.0})
    return results


def _set_active_model(model_name=None):
    global ACTIVE_MODEL_NAME, ACTIVE_MODEL_KIND, pipe

    kind, model_id = _resolve_model(model_name)
    if model_id == ACTIVE_MODEL_NAME and kind == ACTIVE_MODEL_KIND:
        return

    ACTIVE_MODEL_NAME = model_id
    ACTIVE_MODEL_KIND = kind

    if kind == "hf":
        print(f"Loading model from: {model_id}")
        print("Loading model...")
        pipe = _load_hf_pipeline(model_id)
        print("Model ready.")
    else:
        pipe = None
        print(f"Using Ollama model: {model_id}")


_set_active_model(DEFAULT_MODEL_NAME)


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


def classify_blocks(blocks, model_name=None):
    _set_active_model(model_name)

    texts = [b["text"] for b in blocks]
    if ACTIVE_MODEL_KIND == "ollama":
        results = _ollama_predict(texts, ACTIVE_MODEL_NAME)
    else:
        results = pipe(texts)

    classified = []
    for block, result in zip(blocks, results):
        if ACTIVE_MODEL_KIND == "ollama":
            model_label = result["label"]
            confidence = result["score"]
        else:
            model_label = RAW_ID2LABEL.get(result["label"], "IGNORE")
            confidence = result["score"]

        heuristic = heuristic_classify(block["text"], block)

        if heuristic:
            final_label = heuristic
            source = "heuristic"
        elif ACTIVE_MODEL_KIND == "ollama":
            final_label = model_label
            source = "ollama"
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
