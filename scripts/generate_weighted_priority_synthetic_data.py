import pandas as pd
import random


def generate_weighted_priority(target_count=1000):
    data = []

    headers = [
        "(REQUIRED)",
        "MANDATORY:",
        "MAJOR TASK:",
        "GRADING CRITERIA:",
        "Evaluation:",
        "CRITICAL:",
    ]

    ml_actions = [
        "Fine-tuning",
        "Hyperparameter optimization of",
        "Architecting",
        "Evaluating",
        "Preprocessing",
        "Cross-validating",
    ]
    ml_models = [
        "DistilBERT",
        "ResNet-50",
        "Random Forest",
        "LSTM",
        "YOLOv8",
        "Vision Transformer",
        "Logistic Regression",
    ]
    ml_datasets = [
        "the MNIST corpus",
        "the Sarcasm Dataset",
        "the CIFAR-10 image set",
        "the IMDb sentiment bank",
        "custom WVSU student data",
    ]
    ml_metrics = [
        "F1-score",
        "Mean Squared Error",
        "Accuracy",
        "AUC-ROC curve",
        "Precision-Recall tradeoff",
    ]

    points = [10, 15, 20, 25, 30, 40, 50, 75, 100, 150]
    percentages = [5, 10, 15, 20, 25, 30, 40, 50]

    print(f"Generating {target_count} unique ML weighted priority rows...")

    while len(data) < target_count:
        current_task = f"{random.choice(ml_actions)} {random.choice(ml_models)} using {random.choice(ml_datasets)}"

        choice = random.randint(1, 3)
        if choice == 1:
            text = f"{random.choice(headers)} ({random.choice(points)} pts) {current_task}."
        elif choice == 2:
            text = f"The {random.choice(ml_metrics)} analysis accounts for {random.choice(percentages)}% of your semester grade."
        else:
            text = f"Assessment {random.randint(1, 50)}: {current_task} based on {random.choice(ml_metrics)}. Worth {random.choice(points)} points."

        data.append({"text": text, "label": "WEIGHTED_PRIORITY"})

        if len(data) % 200 == 0:
            df_temp = pd.DataFrame(data).drop_duplicates(subset=["text"])
            data = df_temp.to_dict("records")
            print(f"Current unique count: {len(data)}")

    return pd.DataFrame(data).iloc[:target_count]


df_weighted = generate_weighted_priority(1000)
df_weighted.to_csv("weighted_priority.csv", index=False)
print(f"Success! Final Count: {len(df_weighted)} Unique ML Priority rows.")
