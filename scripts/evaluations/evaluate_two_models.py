import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from tqdm import tqdm
import ollama
from transformers import pipeline

# ================================================================
#   CONFIGURATION
# ================================================================
CSV_PATH = "../../datasets/evaluations/eval.csv"
OUTPUT_DIR = "./eval_results/versus_mistral"

# 1. HuggingFace Model Config
TUNED_MODEL_PATH = "../../models/firewolf"  # Path to your local saved model
TUNED_MODEL_NAME = "DistilBERT_FineTuned"

# 2. Ollama Model Config
OLLAMA_MODEL_NAME = "mistral:latest"  # Change to mistral, phi3, etc.

# Define your exact 6 classes here
LABELS = [
    "ADMIN_TRAP",
    "IGNORE",
    "OPTIONAL_BONUS",
    "PRE_REQUISITE",
    "TECHNICAL_TASK",
    "WEIGHTED_PRIORITY",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================================================================
#   HELPER FUNCTIONS
# ================================================================
def resolve_label(raw_label):
    """
    If your HF model outputs 'LABEL_0', 'LABEL_1', map them here.
    If your model already outputs the string names, this just returns it.
    """
    # Example mapping if your config.json didn't have id2label set:
    # mapping = {"LABEL_0": "ADMIN_TRAP", "LABEL_1": "IGNORE", ...}
    # return mapping.get(raw_label, raw_label)
    return raw_label


def get_ollama_prediction(model_name, text, labels_list):
    """
    Prompts Ollama to classify the text into one of the exact labels.
    """
    valid_labels_str = ", ".join(labels_list)
    prompt = (
        f"You are a strict text classification system parsing academic documents. "
        f"Classify the following text into exactly ONE of these categories: [{valid_labels_str}].\n\n"
        f'Text: "{text}"\n\n'
        f"Rules: Respond with ONLY the exact category name. Do not add explanations or punctuation."
    )

    try:
        response = ollama.generate(model=model_name, prompt=prompt)
        output = response["response"].strip()

        for label in labels_list:
            if label.lower() in output.lower():
                return label, 1.0  # Pseudo-confidence for successful match

        return "UNKNOWN", 0.0
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        return "UNKNOWN", 0.0


# ================================================================
#   EVALUATION ENGINE
# ================================================================
def evaluate_model(
    model_name, inference_type, texts, true_labels, labels_list, model_path=None
):
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name} ({inference_type.upper()})")
    print(f"{'='*60}")

    predicted = []
    confidence = []

    if inference_type == "hf":
        # Load HuggingFace Pipeline (device=0 for GPU, -1 for CPU)
        pipe = pipeline("text-classification", model=model_path, device=-1)

        for text in tqdm(texts, desc=f"Inference ({model_name})"):
            # Truncation ensures long paragraphs don't crash DistilBERT
            res = pipe(text, truncation=True, max_length=512)[0]
            predicted.append(resolve_label(res["label"]))
            confidence.append(res["score"])

    elif inference_type == "ollama":
        for text in tqdm(texts, desc=f"Inference ({model_name})"):
            pred, conf = get_ollama_prediction(model_name, text, labels_list)
            predicted.append(pred)
            confidence.append(conf)

    print(f"  Sample text: {texts[0][:50]}...")
    print(f"  Sample pred: {predicted[0]}")

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
        true_labels, predicted, labels=labels_list, zero_division=0
    )
    print(report)

    # Save Report
    safe_model_name = model_name.replace(":", "_")
    report_path = os.path.join(OUTPUT_DIR, f"{safe_model_name}_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Model: {model_name} ({inference_type})\n")
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
def plot_confusion_matrix(true_labels, predicted, model_name, labels_list):
    safe_model_name = model_name.replace(":", "_")
    display_labels = (
        labels_list + ["UNKNOWN"] if "UNKNOWN" in predicted else labels_list
    )

    cm = confusion_matrix(true_labels, predicted, labels=display_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=display_labels,
        yticklabels=display_labels,
    )
    plt.title(f"Confusion Matrix — {model_name}", fontsize=13)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, f"{safe_model_name}_confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved: {path}")


def plot_comparison(tuned_metrics, ollama_metrics):
    metrics = ["accuracy", "precision", "recall", "f1"]
    labels = ["Accuracy", "Precision", "Recall", "F1"]
    x = np.arange(len(metrics))
    width = 0.35

    m1_vals = [tuned_metrics[m] for m in metrics]
    m2_vals = [ollama_metrics[m] for m in metrics]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(
        x - width / 2, m1_vals, width, label=tuned_metrics["model"], color="#22c55e"
    )
    bars2 = ax.bar(
        x + width / 2, m2_vals, width, label=ollama_metrics["model"], color="#3b82f6"
    )

    ax.set_ylim(0, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_title(f"Comparison: DistilBERT vs Ollama")
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


def plot_confidence_distribution(metrics, true_labels):
    correct, incorrect = [], []

    for pred, conf, true in zip(
        metrics["predicted"], metrics["confidence"], true_labels
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

    plt.xlabel("Confidence Score")
    plt.ylabel("Count")
    plt.title(f"Confidence Distribution: {metrics['model']}")
    plt.legend()
    plt.tight_layout()

    safe_model = metrics["model"].replace(":", "_")
    path = os.path.join(OUTPUT_DIR, f"{safe_model}_confidence.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Confidence distribution saved: {path}")


# ================================================================
#   MAIN
# ================================================================
if __name__ == "__main__":
    # 1. Load Data
    print(f"Loading data from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    texts = df["text"].tolist()
    true_labels = df["true_label"].tolist()

    # 2. Evaluate Fine-Tuned DistilBERT
    tuned_metrics = evaluate_model(
        model_name=TUNED_MODEL_NAME,
        inference_type="hf",
        texts=texts,
        true_labels=true_labels,
        labels_list=LABELS,
        model_path=TUNED_MODEL_PATH,
    )
    plot_confusion_matrix(
        true_labels, tuned_metrics["predicted"], TUNED_MODEL_NAME, LABELS
    )
    plot_confidence_distribution(
        tuned_metrics, true_labels
    )  # DistilBERT will show actual probability spread

    # 3. Evaluate Ollama Baseline
    ollama_metrics = evaluate_model(
        model_name=OLLAMA_MODEL_NAME,
        inference_type="ollama",
        texts=texts,
        true_labels=true_labels,
        labels_list=LABELS,
    )
    plot_confusion_matrix(
        true_labels, ollama_metrics["predicted"], OLLAMA_MODEL_NAME, LABELS
    )
    # Note: Ollama confidence will mostly just show 1.0 vs 0.0 due to generative nature
    plot_confidence_distribution(ollama_metrics, true_labels)

    # 4. Generate Comparison Charts
    plot_comparison(tuned_metrics, ollama_metrics)

    # 5. Print Summary Table
    print("\n" + "=" * 65)
    print("SUMMARY TABLE (DistilBERT vs Ollama)")
    print("=" * 65)
    print(f"{'Metric':<15} {TUNED_MODEL_NAME:>20} {OLLAMA_MODEL_NAME:>20} {'Diff':>8}")
    print("-" * 65)
    for metric in ["accuracy", "precision", "recall", "f1"]:
        tuned = tuned_metrics[metric]
        baseline = ollama_metrics[metric]
        diff = tuned - baseline
        print(
            f"{metric.capitalize():<15} {tuned:>20.4f} {baseline:>20.4f} {diff:>+8.4f}"
        )
    print("=" * 65)
