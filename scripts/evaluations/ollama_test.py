import re
import ollama
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)

import sys

sys.path.append("../../")

from constants.label import LABELS

GROUND_TRUTH = "../../datasets/evaluations/eval.csv"

CLASSIFY_PROMPT = """Classify the following sentence from a university \
activity sheet into exactly one of these six categories:

TECHNICAL_TASK     - Instructions to build, implement, code, write, \
create, train, extract, or calculate something
WEIGHTED_PRIORITY  - Mentions points, grades, scores, or rubrics
ADMIN_TRAP         - Submission rules, deadlines, GitHub/LMS \
instructions, grouping, formatting requirements
OPTIONAL_BONUS     - Optional, bonus, or extra credit items
PRE_REQUISITE      - Setup steps or conditions before starting
IGNORE             - Headers, footers, URLs, university name, \
page numbers, institutional text

Reply with only the category name, nothing else. No explanation. \
No punctuation. Just one of the six category names above.

Sentence: \"{sentence}\""""


def extract_label(raw_response):
    """
    Handles three response formats:
    1. Clean:    "TECHNICAL_TASK"
    2. DeepSeek: "<think>...</think>\nTECHNICAL_TASK"
    3. Verbose:  "The sentence is about TECHNICAL_TASK because..."
    """
    cleaned = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL)
    cleaned = cleaned.strip().upper()

    if cleaned in LABELS:
        return cleaned

    for label in LABELS:
        if label in cleaned:
            return label

    return "IGNORE"


def evaluate_ollama(model_name, texts, true_labels):
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name} (Ollama)")
    print(f"{'='*60}")

    predicted = []
    failures = 0

    for i, text in enumerate(texts):
        prompt = CLASSIFY_PROMPT.format(sentence=text)

        try:
            response = ollama.chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0},  # deterministic output
            )
            raw = response["message"]["content"]
            label = extract_label(raw)

            if label == "IGNORE" and raw.strip().upper() not in LABELS:
                failures += 1
                print(f"  [!]  Unexpected response [{i}]: {raw[:60]!r}")

        except Exception as e:
            print(f"  [X] Error on sample {i}: {e}")
            label = "IGNORE"

        predicted.append(label)

        if i % 10 == 0:
            print(f"  Progress: {i}/{len(texts)}")

    accuracy = accuracy_score(true_labels, predicted)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predicted, average="weighted", zero_division=0
    )

    print(f"\nOverall Metrics:")
    print(f"  Accuracy  : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  Failures  : {failures} responses didn't match any label")
    print(f"\nPer-Class Report:")
    print(classification_report(true_labels, predicted, labels=LABELS, zero_division=0))

    return {
        "model": model_name,
        "predicted": predicted,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


df = pd.read_csv(GROUND_TRUTH)
texts = df["text"].tolist()
true_labels = df["true_label"].tolist()

print(f"Ground truth loaded: {len(df)} samples")
print(df["true_label"].value_counts())

metrics = evaluate_ollama("llama3.2:latest", texts, true_labels)
