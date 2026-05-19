import pandas as pd
from transformers import pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)
import sys

sys.path.append("../../")
from constants.label import LABELS, RAW_ID2LABEL

GROUND_TRUTH = "../../datasets/evaluations/eval.csv"
FINETUNED_MODEL = "../../models/alpha"
RAW_MODEL = "distilbert-base-uncased"

df = pd.read_csv(GROUND_TRUTH)
texts = df["text"].tolist()
true_labels = df["true_label"].tolist()


def resolve_label(raw_label):
    if raw_label in LABELS:
        return raw_label
    return RAW_ID2LABEL.get(raw_label, "IGNORE")


def evaluate(model_path, model_name):
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")

    pipe = pipeline(
        "text-classification",
        model=model_path,
        truncation=True,
        max_length=128,
        batch_size=16,
    )

    results = pipe(texts)
    predicted = [resolve_label(r["label"]) for r in results]

    # Metrics
    accuracy = accuracy_score(true_labels, predicted)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predicted, average="weighted", zero_division=0
    )

    print(f"\nOverall Metrics:")
    print(f"  Accuracy  : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"\nPer-Class Report:")
    print(classification_report(true_labels, predicted, labels=LABELS, zero_division=0))

    return predicted


raw_preds = evaluate(RAW_MODEL, "Raw DistilBERT")
tuned_preds = evaluate(FINETUNED_MODEL, "Fine-tuned (ours)")
