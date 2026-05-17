import pandas as pd
import random


def generate_technical_tasks(target_count=1000):
    data = []

    verbs = [
        "Implement",
        "Calculate",
        "Compute",
        "Train",
        "Fine-tune",
        "Visualize",
        "Optimize",
        "Pre-process",
        "Extract",
        "Analyze",
    ]
    subjects = [
        "TF-IDF matrix",
        "cosine similarity",
        "Euclidean distance",
        "DistilBERT model",
        "confusion matrix",
        "gradient descent",
        "backpropagation",
        "tokenized sequences",
        "CNN architecture",
        "MNIST digits",
        "stop-word removal",
        "feature vectors",
    ]
    tools = [
        "PyTorch",
        "TensorFlow",
        "Scikit-learn",
        "NLTK",
        "Pandas",
        "Drizzle ORM",
        "SvelteKit",
    ]

    templates = [
        "{verb} the {subject} using the {tool} library.",
        "Task {num}: {verb} {subject} for the given dataset.",
        "Modify the existing code to {verb} the {subject}.",
        "Exercise {num}.{sub}: {verb} a script that performs {subject}.",
        "Use {tool} to {verb} the {subject} and output the results.",
        "Apply {subject} techniques to {verb} the model's accuracy.",
        "Write a function to {verb} {subject} from the raw input.",
    ]

    while len(data) < target_count:
        tpl = random.choice(templates)
        sentence = tpl.format(
            verb=random.choice(verbs),
            subject=random.choice(subjects),
            tool=random.choice(tools),
            num=random.randint(1, 10),
            sub=random.randint(1, 5),
        )

        data.append({"text": sentence, "label": "TECHNICAL_TASK"})

        if len(data) % 100 == 0:
            temp_df = pd.DataFrame(data).drop_duplicates(subset=["text"])
            data = temp_df.to_dict("records")

    return pd.DataFrame(data).iloc[:target_count]


df_tech = generate_technical_tasks(1000)
df_tech.to_csv("technical_task.csv", index=False)
print(f"Final Count: {len(df_tech)} Unique Technical Tasks.")
