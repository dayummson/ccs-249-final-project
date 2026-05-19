import sys

# since this is located at the root level and this
# script is in a subfolder, we need to add
# the parent folder to sys.path
sys.path.append("../../")

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,  # Added for dynamic VRAM padding
    Trainer,
    TrainingArguments,
)
from utils.metrics.metric import compute_metrics

DATASET = "../../datasets/synthetic_data/synthetic_data_v2.csv"
OUTPUT_MODEL_PATH = "../../models/beta"
MODEL_NAME = "distilbert-base-uncased"
BATCH_SIZE = 8
NUM_EPOCHS = 5
MAX_SEQUENCE_LENGTH = 64  # Task sentences are short; 64 saves VRAM safely

df = pd.read_csv(DATASET)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Explicitly defined categories to prevent random ID shifting across sessions
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
label_map = {cat: i for i, cat in enumerate(categories)}
id_2_label = {v: k for k, v in label_map.items()}

df["label_id"] = df["label"].map(label_map)

print("Deterministic Label map:", label_map)
print("Class distribution:\n", df["label"].value_counts())

train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index)

raw_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array(sorted(label_map.values())),
    y=train_df["label_id"].values,
)
class_weights = torch.tensor(raw_weights, dtype=torch.float)
print("Class weights:", dict(zip(label_map.keys(), raw_weights.round(2))))


def prepare_dataset(data_frame):
    data_frame = data_frame[["text", "label_id"]].rename(columns={"label_id": "labels"})
    return Dataset.from_pandas(data_frame, preserve_index=False)


train_dataset = prepare_dataset(train_df)
test_dataset = prepare_dataset(test_df)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_func(examples):
    return tokenizer(examples["text"], truncation=True, max_length=MAX_SEQUENCE_LENGTH)


tokenized_train = train_dataset.map(tokenize_func, batched=True)
tokenized_test = test_dataset.map(tokenize_func, batched=True)

tokenized_train = tokenized_train.remove_columns(["text"])
tokenized_test = tokenized_test.remove_columns(["text"])

tokenized_train.set_format("torch")
tokenized_test.set_format("torch")

# Initialize the dynamic padder
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_map),
    id2label=id_2_label,
    label2id=label_map,
)


class WeightedTrainer(Trainer):

    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss = nn.functional.cross_entropy(
            logits,
            labels,
            weight=(
                self.class_weights.to(logits.device)
                if self.class_weights is not None
                else None
            ),
        )
        return (loss, outputs) if return_outputs else loss


total_steps = (len(train_df) // BATCH_SIZE) * NUM_EPOCHS
warmup = max(50, total_steps // 10)
print(f"Total steps: {total_steps}, Warmup steps: {warmup}")

args = TrainingArguments(
    output_dir="./training_results",
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    warmup_steps=warmup,
    weight_decay=0.01,
    logging_dir="./training_logs",
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    save_total_limit=2,  # limits saved checkpoints so your drive doesn't fill up
    metric_for_best_model="f1",
    fp16=True,  # Fully optimal acceleration step for Turing cards like 1650Ti
    report_to="none",
)

trainer = WeightedTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    data_collator=data_collator,  # Passed data collator for dynamic optimization
    compute_metrics=compute_metrics,
    class_weights=class_weights,  # Injected directly into initialization
)

trainer.train()

trainer.save_model(OUTPUT_MODEL_PATH)
tokenizer.save_pretrained(OUTPUT_MODEL_PATH)
print(f"Model saved to {OUTPUT_MODEL_PATH}")
