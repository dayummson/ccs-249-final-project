import pandas as pd
import random


def generate_optional_bonus(target_count=1000):
    data = []
    bonus_triggers = [
        "Bonus",
        "Optional",
        "Extra Credit",
        "Additional Task",
        "For extra points",
        "Volunteer Task",
    ]

    tasks = [
        "implement a dark mode",
        "add a unit test",
        "optimize the runtime",
        "create a README.md",
        "deploy to Vercel",
        "refactor the middleware",
    ]

    contexts = [
        "for the UI",
        "in the backend",
        "within the user dashboard",
        "for the login flow",
        "on the landing page",
    ]

    print(f"Starting generation for {target_count} rows...")

    while len(data) < target_count:
        text = (
            f"{random.choice(bonus_triggers)}: {random.choice(tasks)} "
            f"{random.choice(contexts)} for an additional {random.randint(1, 100)} points."
        )

        data.append({"text": text, "label": "OPTIONAL_BONUS"})

        if len(data) % 200 == 0:
            df_temp = pd.DataFrame(data).drop_duplicates(subset=["text"])
            data = df_temp.to_dict("records")
            print(f"Current unique count: {len(data)}")

            if len(data) < (len(data) - 50):
                print("Warning: Low entropy detected. Add more vocabulary!")

    return pd.DataFrame(data).iloc[:target_count]


optional_bonus = generate_optional_bonus(1000)
optional_bonus.to_csv("optional_bonus.csv", index=False)
print(f"Success! Final Count: {len(optional_bonus)} Unique rows.")
