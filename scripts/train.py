import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

DATASET = "./datasets/synthetic_data.csv"


df = pd.read_csv(DATASET)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df["label"] = df["label"].astype("category")

label_map = {cat: i for i, cat in enumerate(df["label"].cat.categories)}
df["label_id"] = df["label"].map(label_map)


# 80/20 Split
train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index)


print(train_df)


def prepare_dataset(df):
    df = df.rename(columns={"label_id": "labels"})
    return Dataset.from_pandas(df)


train_dataset = prepare_dataset(train_df)
test_dataset = prepare_dataset(test_df)


# Setup Tokenizer & Model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize_func(examples):
    return tokenizer(
        examples["text"], padding="max_length", truncation=True, max_length=128
    )


tokenized_train = train_dataset.map(tokenize_func, batched=True)
tokenized_test = test_dataset.map(tokenize_func, batched=True)

tokenized_train = tokenized_train.remove_columns(["text", "label"])
tokenized_test = tokenized_test.remove_columns(["text", "label"])

tokenized_train.set_format("torch")
tokenized_test.set_format("torch")

model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=len(label_map)
)


# Metrics Function
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


args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("./final_requirement_model")
tokenizer.save_pretrained("./final_requirement_model")
