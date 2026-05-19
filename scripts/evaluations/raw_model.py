import sys

sys.path.append("../../")


import csv
import os
from transformers import pipeline, AutoTokenizer
from utils.extractors.docx_extractor import extract_blocks_from_docx
from constants.label import LABELS, RAW_ID2LABEL

FILE_PATH = "../../samples/unit-4-activity.docx"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = "distilbert-base-uncased"


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

pipe = pipeline("text-classification", model=MODEL_PATH)


def resolve_label(raw_label):
    if raw_label in LABELS:
        return raw_label
    return RAW_ID2LABEL.get(raw_label, "IGNORE")


def evaluate():
    blocks = extract_blocks_from_docx(FILE_PATH)
    texts = [b["text"] for b in blocks]

    # Warn about blocks that will be truncated
    for block in blocks:
        token_len = len(tokenizer.encode(block["text"]))
        if token_len > 512:
            print(
                f"[WARN] Block truncated ({token_len} tokens): {block['text'][:60]}..."
            )

    results = pipe(texts, truncation=True, max_length=512)

    classified = []
    for block, result in zip(blocks, results):
        model_label = resolve_label(result["label"])
        print(f"[BLOCK]: {block['text']} | [MODEL LABEL]: {model_label}")
        classified.append({"text": block["text"], "predicted_label": model_label})

    return classified


result = evaluate()

# Save Results to CSV
# This saves the file in the exact directory where your script runs
csv_file_path = os.path.join(
    BASE_DIR, "distillbert-base-uncased-unit-4-activity-classified-raw-v2.csv"
)

# Extract the header keys from your dictionary items
csv_headers = ["text", "predicted_label"]

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
