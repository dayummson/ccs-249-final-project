import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
MODEL_NAME = "distilbert-base-uncased"
DATA_FILE = "../../datasets/synthetic_data/synthetic_data_v3.csv"
OUTPUT_MODEL_DIR = "../../models/firewolf"
BATCH_SIZE = 16
NUM_EPOCHS = 8

categories = sorted(
    [
        "TECHNICAL_TASK",
        "ADMIN_TRAP",
        "IGNORE",
        "WEIGHTED_PRIORITY",
        "PRE_REQUISITE",
        "OPTIONAL_BONUS",
    ]
)
label2id = {label: idx for idx, label in enumerate(categories)}
id2label = {idx: label for idx, label in enumerate(categories)}

print(f"Target Labels Map: {label2id}")

# ==========================================
# 2. DATA PROCESSING & STRATIFIED SPLIT
# ==========================================
df = pd.read_csv(DATA_FILE)
df["label_id"] = df["label"].map(label2id)

# CRITICAL FIX: Replace random split with a Stratified Split
# This ensures identical class distributions in both training and test sets
train_df, test_df = train_test_split(
    df, test_size=0.2, stratify=df["label_id"], random_state=42
)

print(f"Train Dataset Size: {len(train_df)} rows")
print(f"Test Dataset Size: {len(test_df)} rows")


# ==========================================
# 3. HUGGINGFACE DATASET ADAPTER
# ==========================================
class AcademicDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_encodings = tokenizer(
    train_df["text"].tolist(), truncation=True, padding=True, max_length=128
)
test_encodings = tokenizer(
    test_df["text"].tolist(), truncation=True, padding=True, max_length=128
)

train_dataset = AcademicDataset(train_encodings, train_df["label_id"].tolist())
test_dataset = AcademicDataset(test_encodings, test_df["label_id"].tolist())


# ==========================================
# 4. METRICS DEFINITION
# ==========================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted"
    )
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


# ==========================================
# 5. TRAINING ARGUMENTS & LOGISTICS
# ==========================================
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(categories),
    id2label=id2label,
    label2id=label2id,
)

args = TrainingArguments(
    output_dir="./training_results",
    num_train_epochs=NUM_EPOCHS,
    learning_rate=3e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    weight_decay=0.01,
    logging_dir="./training_logs",
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    save_total_limit=2,
    metric_for_best_model="f1",
    fp16=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# ==========================================
# 6. RUN ENGINE
# ==========================================
print("\nCommencing optimized model fine-tuning...")
trainer.train()

print("\nEvaluating stabilized model metrics...")
eval_results = trainer.evaluate()
print(f"\nFinal Evaluation Results: {eval_results}")

# Save the final pristine model
model.save_pretrained(OUTPUT_MODEL_DIR)
tokenizer.save_pretrained(OUTPUT_MODEL_DIR)
print(f"\nOptimized baseline saved successfully to '{OUTPUT_MODEL_DIR}'")
