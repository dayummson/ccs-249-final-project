import pandas as pd
import re

df = pd.read_csv("synthetic_data_v2_smoothed.csv")


def clean_text(text):
    # Strip surrounding quotes that Ollama added
    text = text.strip()

    # Remove wrapping double quotes — only if they wrap the ENTIRE string
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]

    # Remove triple quotes just in case
    text = text.replace('"""', "").replace("'''", "")

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


df["text"] = df["text"].apply(clean_text)

# Verify the fix
print("After cleaning — first 5:")
for i, row in df.head(5).iterrows():
    print(f"  [{i}] {repr(row['text'])}")

# Sanity check — find any remaining quotes wrapping full strings
still_quoted = df[df["text"].str.startswith('"') & df["text"].str.endswith('"')]
print(f"\nStill wrapped in quotes: {len(still_quoted)} rows")
if len(still_quoted) > 0:
    print(still_quoted["text"].head(3).tolist())

# Check distribution is intact
print(f"\nTotal rows: {len(df)}")
print(df["label"].value_counts())

df.to_csv("synthetic_data_v2_clean.csv", index=False)
print("\nSaved to synthetic_data_v2_clean.csv")
