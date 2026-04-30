import pandas as pd
import random


def generate_infinite_ignore(target_count=1000):
    data = []

    topics = [
        "Data Science",
        "Web Development",
        "Operating Systems",
        "Networking",
        "Artificial Intelligence",
        "Cybersecurity",
        "Discrete Math",
        "Software Engineering",
    ]
    concepts = [
        "loops",
        "arrays",
        "pointers",
        "recursion",
        "normalization",
        "encryption",
        "inheritance",
        "polymorphism",
    ]
    filler_phrases = [
        "is a fundamental concept in",
        "is widely used within",
        "was originally developed for",
        "can be difficult to master in",
        "is essential for understanding",
        "is discussed in detail during",
    ]
    greetings = [
        "Hello class,",
        "Good morning everyone.",
        "Hi guys.",
        "Welcome back to another session.",
    ]
    fluff = [
        "I hope you had a good lunch.",
        "The traffic in Iloilo was bad today.",
        "Is the aircon too cold?",
        "Please take your seats.",
        "Wait for a while.",
    ]

    while len(data) < target_count:
        choice = random.randint(1, 5)

        if choice == 1:
            text = f"{random.choice(concepts).capitalize()} {random.choice(filler_phrases)} {random.choice(topics)}."
        elif choice == 2:
            text = f"{random.choice(greetings)} {random.choice(fluff)}"
        elif choice == 3:
            text = f"Unit {random.randint(1, 20)}: {random.choice(topics)} - Part {random.randint(1, 5)}"
        elif choice == 4:
            text = f"The study of {random.choice(topics)} involves {random.choice(concepts)} and related ideas."
        else:
            text = f"We will discuss {random.choice(topics)} at {random.randint(1, 12)}:{random.choice(['00', '30'])} PM."

        data.append({"text": text, "label": "IGNORE"})

        if len(data) % 100 == 0:
            temp_df = pd.DataFrame(data).drop_duplicates(subset=["text"])
            data = temp_df.to_dict("records")

    return pd.DataFrame(data).iloc[:target_count]


df_ignore = generate_infinite_ignore(1000)
df_ignore.to_csv("ignore.csv", index=False)
print(f"Final Count: {len(df_ignore)} Unique IGNORE rows.")
