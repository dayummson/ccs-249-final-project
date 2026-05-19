import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)

import sys

sys.path.append("../../")

from constants.label import LABELS

EVAL_CONFIG = {
    "chatgpt": {
        "path": "../../zero-shot-results/chatgpt-unit-4-activity-classified.csv"
    },
    "grok": {
        "path": "../../zero-shot-results/grok-fast-unit-4-activity-classified.csv"
    },
    "claude": {
        "path": "../../zero-shot-results/claude-sonnet-4.6-unit-4-activity-classified.csv"
    },
}

GROUND_TRUTH = "../../datasets/evaluations/eval.csv"

df = pd.read_csv(GROUND_TRUTH)
texts = df["text"].tolist()
true_labels = df["true_label"].tolist()


# For each manually collected LLM result
def evaluate_from_csv(predictions_csv, true_labels, model_name):
    df = pd.read_csv(predictions_csv)
    predicted = df["predicted_label"].tolist()

    accuracy = accuracy_score(true_labels, predicted)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predicted, average="weighted", zero_division=0
    )

    print(f"\n{model_name}:")
    print(f"  F1: {f1:.4f}  Accuracy: {accuracy:.4f}")
    print(classification_report(true_labels, predicted, labels=LABELS, zero_division=0))

    return {
        "model": model_name,
        "f1": f1,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
    }


if __name__ == "__main__":

    result = evaluate_from_csv(
        EVAL_CONFIG["chatgpt"]["path"], true_labels=true_labels, model_name="chatgpt"
    )
