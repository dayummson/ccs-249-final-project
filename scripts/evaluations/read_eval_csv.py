import pandas as pd

GROUND_TRUTH = "../../datasets/evaluations/eval.csv"
df = pd.read_csv(GROUND_TRUTH)
print(f"Ground truth loaded: {len(df)} samples")
print("\nClass distribution:")
print(df["true_label"].value_counts())
