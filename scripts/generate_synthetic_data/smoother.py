import ollama
import pandas as pd

SMOOTH_PROMPT = """Rephrase the following sentence to sound more natural, \
like something a university professor would actually write on an activity sheet. \
Keep the exact same meaning and intent. Do not change the action being described. \
Do not add or remove information. Reply with only the rephrased sentence, nothing else.

Sentence: \"{sentence}\""""

df = pd.read_csv("synthetic_data_v2.csv")
rows = []

for i, row in df.iterrows():
    try:
        response = ollama.chat(
            model="llama3.2:latest",
            messages=[
                # todo: include label and update the prompt
                # so it will have context before smoothing
                {"role": "user", "content": SMOOTH_PROMPT.format(sentence=row["text"])}
            ],
            # TODO: improve by adding penalty ( frequency_penalty around 0.2 to 0.5 )
            options={"temperature": 0.7},  # some variation is good here
        )
        smoothed = response["message"]["content"].strip()

        # Sanity check — if LLM went off the rails, keep original
        if len(smoothed) < 10 or len(smoothed) > 400:
            smoothed = row["text"]

    except Exception as e:
        print(f"Error on row {i}: {e}")
        smoothed = row["text"]

    rows.append({"text": smoothed, "label": row["label"]})

    if i % 50 == 0:
        print(f"  Smoothed {i}/{len(df)}")

    # Save checkpoint every 100 rows in case it crashes
    if i % 100 == 0 and i > 0:
        pd.DataFrame(rows).to_csv("synthetic_data_smoothed_checkpoint.csv", index=False)

pd.DataFrame(rows).to_csv("synthetic_data_smoothed.csv", index=False)
print(f"Done. Saved {len(rows)} smoothed rows.")
