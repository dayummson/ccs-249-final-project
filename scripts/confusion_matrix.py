import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset

MODEL_PATH = "./final_requirement_model"
DATASET_PATH = "./datasets/synthetic_data.csv"

df = pd.read_csv(DATASET_PATH)
df["label"] = df["label"].astype("category")
category_names = df["label"].cat.categories
label_map = {cat: i for i, cat in enumerate(category_names)}
df["label_id"] = df["label"].map(label_map)

test_df = df.drop(df.sample(frac=0.8, random_state=42).index).reset_index(drop=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)


test_dataset = Dataset.from_pandas(test_df.rename(columns={"label_id": "labels"}))


def tokenize_func(examples):
    return tokenizer(
        examples["text"], padding="max_length", truncation=True, max_length=128
    )


test_dataset = Dataset.from_pandas(test_df.rename(columns={"label_id": "labels"}))

tokenized_test = test_dataset.map(tokenize_func, batched=True)

tokenized_test = tokenized_test.remove_columns(["text", "label"])

tokenized_test.set_format("torch")

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

trainer = Trainer(model=model)

print("Running evaluation on test set...")
predictions = trainer.predict(tokenized_test)
preds = np.argmax(predictions.predictions, axis=-1)
actuals = predictions.label_ids

cm = confusion_matrix(actuals, preds)

plt.figure(figsize=(12, 10))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Purples",
    xticklabels=category_names,
    yticklabels=category_names,
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("WVSU Student Assistant Model: Confusion Matrix")
plt.savefig("confusion_matrix_results.png")
plt.show()

print("\nDetailed Classification Report:")
print(classification_report(actuals, preds, target_names=category_names))
