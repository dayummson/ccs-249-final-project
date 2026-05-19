import sys

sys.path.append("../../")


import csv
import os
import ollama
from constants.label import LABELS, RAW_ID2LABEL
import re
from utils.extractors.docx_extractor import extract_blocks_from_docx

FILE_PATH = "../../samples/unit-4-activity.docx"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "deepseek-coder:6.7b"


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


def evaluate(model_name, texts):

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

        predicted.append({"text": text, "predicted_label": label})

    return predicted


blocks = extract_blocks_from_docx(FILE_PATH)

texts = [b["text"] for b in blocks]


result = evaluate(model_name=MODEL_NAME, texts=texts)

print(result)

# Save Results to CSV
# This saves the file in the exact directory where your script runs
csv_file_path = os.path.join(
    BASE_DIR, f"{MODEL_NAME}-unit-4-activity-classified-raw.csv"
)

# Extract the header keys from your dictionary items
csv_headers = ["text", "label"]

try:
    with open(csv_file_path, mode="w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_headers)

        # Write the column headers
        writer.writeheader()

        # Write all rows
        writer.writerows(result)

    print(f"Success! Data written to: {csv_file_path}")
except Exception as e:
    print(f"An error occurred while writing the CSV file: {e}")
