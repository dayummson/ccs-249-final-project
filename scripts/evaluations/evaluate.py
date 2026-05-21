import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

sys.path.append("../../")

from constants.label import LABELS, RAW_ID2LABEL

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TUNED_MODEL = os.path.abspath(os.path.join(BASE_DIR, "../../models/eagle"))
GROUND_TRUTH = os.path.join(BASE_DIR, "../../datasets/evaluations/eval.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "eval_results/eagle")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def resolve_label(raw_label):
    if raw_label in LABELS:
        return raw_label

    # Raw DistilBERT has no id2label config so it returns LABEL_0 style
    # Fine-tuned model has id2label saved so it returns human-readable labels
    # This handles both cases
    return RAW_ID2LABEL.get(raw_label, "IGNORE")


# ================================================================
#   LOAD DATA
# ================================================================

df = pd.read_csv(GROUND_TRUTH)
print(f"Ground truth loaded: {len(df)} samples")
print("\nClass distribution:")
print(df["true_label"].value_counts())

texts = df["text"].tolist()
true_labels = df["true_label"].tolist()


# ================================================================
#   EVALUATE
# ================================================================


def evaluate_model(model_path, model_name, texts, true_labels):
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")

    pipe = pipeline("text-classification", model=model_path)
    results = pipe(texts)

    # Show sample output so you can confirm label format
    print(f"  Sample output: {results[0]}")

    predicted = [resolve_label(r["label"]) for r in results]
    confidence = [r["score"] for r in results]

    accuracy = accuracy_score(true_labels, predicted)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predicted, average="weighted", zero_division=0
    )

    print(f"\nOverall Metrics:")
    print(f"  Accuracy  : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  Avg Conf  : {np.mean(confidence):.4f}")

    print(f"\nPer-Class Report:")
    report = classification_report(
        true_labels, predicted, labels=LABELS, zero_division=0
    )
    print(report)

    report_path = os.path.join(OUTPUT_DIR, f"{model_name}_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Samples: {len(texts)}\n\n")
        f.write(f"Accuracy : {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall   : {recall:.4f}\n")
        f.write(f"F1 Score : {f1:.4f}\n\n")
        f.write(report)
    print(f"Report saved: {report_path}")

    return {
        "model": model_name,
        "predicted": predicted,
        "confidence": confidence,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ================================================================
#   PLOTS
# ================================================================


def plot_confusion_matrix(true_labels, predicted, model_name):
    cm = confusion_matrix(true_labels, predicted, labels=LABELS)
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABELS,
        yticklabels=LABELS,
    )
    plt.title(f"Confusion Matrix — {model_name}", fontsize=13)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"{model_name}_confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved: {path}")


def plot_comparison(raw_metrics, tuned_metrics):
    metrics = ["accuracy", "precision", "recall", "f1"]
    labels = ["Accuracy", "Precision", "Recall", "F1"]
    x = np.arange(len(metrics))
    width = 0.35

    raw_vals = [raw_metrics[m] for m in metrics]
    tuned_vals = [tuned_metrics[m] for m in metrics]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(
        x - width / 2, raw_vals, width, label="Raw DistilBERT", color="#ef4444"
    )
    bars2 = ax.bar(
        x + width / 2, tuned_vals, width, label="Fine-tuned (Ours)", color="#22c55e"
    )

    ax.set_ylim(0, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_title("Raw DistilBERT vs Fine-tuned Model")
    ax.legend()

    for bar in bars1 + bars2:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.02,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "comparison_chart.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Comparison chart saved: {path}")


def plot_confidence_distribution(tuned_metrics, true_labels):
    correct = []
    incorrect = []

    for pred, conf, true in zip(
        tuned_metrics["predicted"], tuned_metrics["confidence"], true_labels
    ):
        if pred == true:
            correct.append(conf)
        else:
            incorrect.append(conf)

    plt.figure(figsize=(8, 4))
    plt.hist(
        correct, bins=20, alpha=0.7, color="#22c55e", label=f"Correct ({len(correct)})"
    )
    plt.hist(
        incorrect,
        bins=20,
        alpha=0.7,
        color="#ef4444",
        label=f"Incorrect ({len(incorrect)})",
    )
    plt.axvline(x=0.75, color="gray", linestyle="--", label="Threshold (0.75)")
    plt.xlabel("Confidence Score")
    plt.ylabel("Count")
    plt.title("Confidence Distribution: Correct vs Incorrect Predictions")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "confidence_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Confidence distribution saved: {path}")


# ================================================================
#   MAIN
# ================================================================

if __name__ == "__main__":
    raw_metrics = evaluate_model(
        "distilbert-base-uncased", "Raw_DistilBERT", texts, true_labels
    )
    plot_confusion_matrix(true_labels, raw_metrics["predicted"], "Raw_DistilBERT")

    tuned_metrics = evaluate_model(TUNED_MODEL, "Fine_Tuned", texts, true_labels)
    plot_confusion_matrix(true_labels, tuned_metrics["predicted"], "Fine_Tuned")

    plot_comparison(raw_metrics, tuned_metrics)
    plot_confidence_distribution(tuned_metrics, true_labels)

    print("\n" + "=" * 60)
    print("SUMMARY TABLE (copy this into your paper)")
    print("=" * 60)
    print(
        f"{'Metric':<15} {'Raw DistilBERT':>15} {'Fine-tuned':>15} {'Improvement':>15}"
    )
    print("-" * 60)
    for metric in ["accuracy", "precision", "recall", "f1"]:
        raw = raw_metrics[metric]
        tuned = tuned_metrics[metric]
        diff = tuned - raw
        print(f"{metric.capitalize():<15} {raw:>15.4f} {tuned:>15.4f} {diff:>+15.4f}")
    print("=" * 60)
