# debug_check.py — run this first
import pandas as pd

df = pd.read_csv("synthetic_data_v2_smoothed.csv")

print(f"Total rows: {len(df)}")
print(f"\nFirst 5 raw text values:")
for i, row in df.head(5).iterrows():
    print(f"  [{i}] repr: {repr(row['text'])}")
    print(f"       type: {type(row['text'])}")
