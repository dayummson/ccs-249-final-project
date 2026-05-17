import sys

# since this is located at the root level and this
# script is in a subfolder, we need to add
# the parent folder to sys.path
sys.path.append("../../")

import pandas as pd
import torch
from torch import nn
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from utils.metrics.metric import compute_metrics

DATASET = "../../datasets/synthetic_data/synthetic_data.csv"
OUTPUT_MODEL_PATH = "../../models/alpha"

df = pd.read_csv(DATASET)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df["label"] = df["label"].astype("category")

label_map = {cat: i for i, cat in enumerate(df["label"].cat.categories)}

id_2_label = {v: k for k, v in label_map.items()}  # reverse map — useful later

df["label_id"] = df["label"].map(label_map)

print("Label map:", label_map)

print("Class distribution:\n", df["label"].value_counts())

train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index)

# Class Weights
#
# The synthetic data has 300 samples per class (balanced).
# But real activity sheets are NOT balanced:
#   - TECHNICAL_TASK appears 20+ times per sheet
#   - OPTIONAL_BONUS appears maybe once or twice
#
# Without class weights, the model learns to favor common classes
# because getting them right gives more gradient signal.
# Class weights fix this by penalizing mistakes on rare classes more.
#
# compute_class_weight("balanced") automatically calculates:
#   weight[class] = total_samples / (n_classes * samples_in_class)
# So rare classes get higher weights → model pays more attention to them.

raw_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array(sorted(label_map.values())),
    y=train_df["label_id"].values,
)
class_weights = torch.tensor(raw_weights, dtype=torch.float)
print("Class weights:", dict(zip(label_map.keys(), raw_weights.round(2))))

# Tokenization


def prepare_dataset(df):
    df = df[["text", "label_id"]].rename(columns={"label_id": "labels"})
    return Dataset.from_pandas(df, preserve_index=False)


train_dataset = prepare_dataset(train_df)
test_dataset = prepare_dataset(test_df)

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize_func(examples):
    return tokenizer(
        examples["text"], padding="max_length", truncation=True, max_length=128
    )


tokenized_train = train_dataset.map(tokenize_func, batched=True)
tokenized_test = test_dataset.map(tokenize_func, batched=True)

tokenized_train = tokenized_train.remove_columns(["text"])
tokenized_test = tokenized_test.remove_columns(["text"])

tokenized_train.set_format("torch")
tokenized_test.set_format("torch")

# Model

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label_map),
    id2label=id_2_label,
    label2id=label_map,
)

# Weighted Trainer
#
# The default Trainer uses plain cross-entropy with no weights.
# We subclass it and override compute_loss to inject our class weights.
# That's the "one-liner" concept — one override, everything else stays the same.


class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss = nn.functional.cross_entropy(
            logits,
            labels,
            weight=class_weights.to(logits.device),  # must be on same device as model
        )
        return (loss, outputs) if return_outputs else loss


# Training Args
#
# warmup_steps=500 is too high for 1800 samples × 0.8 = 1440 train samples
# At batch_size=8 → 180 steps per epoch → 540 steps total (3 epochs)
# 500 warmup steps means, warming up for almost the entire training! ( woah )
# Rule of thumb: warmup = 10% of total steps

total_steps = (len(train_df) // 8) * 3  # (samples / batch_size) * epochs
warmup = max(50, total_steps // 10)
print(f"Total steps: {total_steps}, Warmup steps: {warmup}")

args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,  # bumped from 3 — small dataset benefits from more
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=warmup,  # fixed — was 500 which is too high
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",  # save based on F1, not just loss
    fp16=True,
    report_to="none",  # disables wandb if not installed
)

trainer = WeightedTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.save_model(OUTPUT_MODEL_PATH)
tokenizer.save_pretrained(OUTPUT_MODEL_PATH)
print(f"Model saved to {OUTPUT_MODEL_PATH}")
