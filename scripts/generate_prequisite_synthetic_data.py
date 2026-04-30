import pandas as pd
import random


def generate_pre_requisites(target_count=1000):
    data = []

    starters = [
        "Before you begin",
        "Prior to starting",
        "As a first step",
        "To prepare",
        "Ensure you first",
    ]

    actions = [
        "install",
        "download",
        "clone",
        "import",
        "setup",
        "initialize",
        "configure",
        "update",
        "verify",
    ]

    items = [
        "the NLTK library",
        "the provided CSV",
        "the GitHub repository",
        "a virtual environment",
        "the DistilBERT tokenizer",
        "the Neon DB credentials",
        "Drizzle ORM",
        "the SvelteKit boilerplate",
        "the Tailwind CSS config",
        "the requirements.txt file",
        "the Python 3.10 interpreter",
        "the latest Fedora updates",
        "supergfxctl for your GPU",
        "the 24GB RAM allocation",
        "the UGREEN power bank driver",
        "the MNIST dataset",
        "the Kaggle API key",
    ]

    print(f"Generating {target_count} unique pre-requisites...")

    while len(data) < target_count:
        version = f"{random.randint(1, 5)}.{random.randint(0, 9)}"
        text = f"{random.choice(starters)}, {random.choice(actions)} {random.choice(items)} v{version}."

        data.append({"text": text, "label": "PRE_REQUISITE"})

        if len(data) % 200 == 0:
            df_temp = pd.DataFrame(data).drop_duplicates(subset=["text"])
            data = df_temp.to_dict("records")
            print(f"Current unique count: {len(data)}")

    return pd.DataFrame(data).iloc[:target_count]


df_pre_req = generate_pre_requisites(1000)
df_pre_req.to_csv("pre_requisites.csv", index=False)
print(f"Success! Final Count: {len(df_pre_req)} Unique rows.")
