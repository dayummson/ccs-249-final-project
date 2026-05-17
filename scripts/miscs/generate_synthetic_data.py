import pandas as pd
import random

wvsu_real_data = [
    {
        "text": "Implement association rule mining, classification, and clustering.",
        "label": "TECHNICAL_TASK",
    },
    {
        "text": "Perform thorough evaluation using confusion matrices and F1-scores.",
        "label": "WEIGHTED_PRIORITY",
    },
    {
        "text": "Integrate a domain-specific dataset from Hugging Face.",
        "label": "TECH_STACK",
    },
    {
        "text": "Visualise vectors using PCA after fine-tuning.",
        "label": "TECHNICAL_TASK",
    },
    {
        "text": "Final project must include a full pipeline from preprocessing to evaluation.",
        "label": "ADMIN_TRAP",
    },
    {"text": "Download the NLTK package and import webtext.", "label": "PRE_REQUISITE"},
    {
        "text": "Submit a PDF report containing the methodology and results.",
        "label": "ADMIN_TRAP",
    },
]

noise_phrases = [
    "Welcome to the second semester of the computer science program.",
    "Please read the instructions carefully before starting.",
    "This exercise is designed to test your understanding of Unit 2.",
    "Discuss your findings with your group members during the lab.",
    "Good luck with your implementation.",
]


def generate_robust_dataset(iterations=1200):
    rows = []
    for _ in range(iterations):
        # 70% of data is labeled requirements
        if random.random() > 0.3:
            seed = random.choice(wvsu_real_data)
            # Add random academic fluff to the beginning or end to simulate real rubrics
            text = seed["text"]
            if random.random() > 0.5:
                text = f"{random.choice(noise_phrases)} {text}"
            rows.append({"text": text, "label": seed["label"]})
        # 30% of data is "Noise" (Label: IGNORE)
        else:
            rows.append({"text": random.choice(noise_phrases), "label": "IGNORE"})

    return pd.DataFrame(rows)


# Create the expanded dataset
df_robust = generate_robust_dataset()
df_robust.to_csv("./datasets/wvsu_robust_requirements.csv", index=False)
print(
    f"Dataset ready: {len(df_robust)} rows across {len(df_robust['label'].unique())} classes."
)
