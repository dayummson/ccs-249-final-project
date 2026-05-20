from collections import Counter
import pandas as pd

df = pd.read_csv("../../datasets/synthetic_data/synthetic_data_v2.csv")


def get_top_words(df, label, n=20):
    texts = " ".join(df[df["label"] == label]["text"].tolist()).lower()
    words = [w for w in texts.split() if len(w) > 3]
    return set(w for w, _ in Counter(words).most_common(n))


tech_words = get_top_words(df, "TECHNICAL_TASK")
admin_words = get_top_words(df, "ADMIN_TRAP")
weighted_words = get_top_words(df, "WEIGHTED_PRIORITY")
ignore_words = get_top_words(df, "IGNORE")

# Overlap between classes — lower is better
print("TECHNICAL_TASK ∩ ADMIN_TRAP:", tech_words & admin_words)
print("TECHNICAL_TASK ∩ WEIGHTED_PRIORITY:", tech_words & weighted_words)
print("ADMIN_TRAP ∩ IGNORE:", admin_words & ignore_words)


total = len(df)
unique = df["text"].nunique()
dupes = total - unique
dupe_pct = dupes / total * 100

print(f"\nDuplicate rate: {dupes}/{total} ({dupe_pct:.1f}%)")
print("Target: < 2%")

# Show the actual duplicates
print("\nDuplicate rows:")
print(df[df.duplicated(subset=["text"])][["text", "label"]].head(10))
