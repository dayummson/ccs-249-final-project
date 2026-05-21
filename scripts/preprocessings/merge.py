import os
import glob
import pandas as pd

# ==========================================
# CONFIGURATION CONSTANTS
# ==========================================
INPUT_FOLDER = "cleaned"
OUTPUT_FILE = "synthetic_data_v3.csv"


def merge_cleaned_datasets():
    search_pattern = os.path.join(INPUT_FOLDER, "*.csv")
    csv_files = glob.glob(search_pattern)

    if not csv_files:
        print(f"No CSV files found in directory: '{INPUT_FOLDER}'")
        return

    print(f"Found {len(csv_files)} files to merge.")

    dataframes = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)

            # 1. Drop completely empty/blank columns that cause double commas
            df = df.dropna(how="all", axis=1)

            # 2. Clean and lowercase column headers to catch trailing spaces
            df.columns = df.columns.str.strip().str.lower()

            # 3. Dynamically find the text and label columns
            text_col = next((c for c in df.columns if "text" in c), None)
            label_col = next(
                (c for c in df.columns if "label" in c or "class" in c), None
            )

            # Fallback: If headers are completely broken, use position
            if text_col is None or label_col is None:
                if len(df.columns) >= 2:
                    text_col = df.columns[0]
                    label_col = df.columns[1]

            if text_col and label_col:
                # 4. Strictly isolate only these two columns and normalize names
                df = df[[text_col, label_col]].rename(
                    columns={text_col: "text", label_col: "label"}
                )
                dataframes.append(df)
                print(
                    f" - Loaded & Standardized: {os.path.basename(file)} ({len(df)} rows)"
                )
            else:
                print(
                    f" - Warning: Skipping {os.path.basename(file)} due to unreadable structure."
                )

        except Exception as e:
            print(f" - Error loading {os.path.basename(file)}: {e}")

    if not dataframes:
        print("No valid dataframes found to merge.")
        return

    # Combine cleanly aligned dataframes
    merged_df = pd.concat(dataframes, ignore_index=True)

    # Clean out any null values or exact duplicates across files
    merged_df = merged_df.dropna(subset=["text", "label"])
    merged_df = merged_df.drop_duplicates()

    # Shuffle dataset
    merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Export cleanly
    merged_df.to_csv(OUTPUT_FILE, index=False)

    print("=" * 40)
    print(f"Merge Complete! Saved to: {OUTPUT_FILE}")
    print(f"Total rows in final dataset: {len(merged_df)}")
    print("=" * 40)


if __name__ == "__main__":
    merge_cleaned_datasets()
