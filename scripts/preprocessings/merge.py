import os
import glob
import pandas as pd

# ==========================================
# CONFIGURATION CONSTANTS
# ==========================================
INPUT_FOLDER = "cleaned"
OUTPUT_FILE = "synthetic_data_v3.csv"


def merge_cleaned_datasets():
    # Find all CSV files in the target directory
    search_pattern = os.path.join(INPUT_FOLDER, "*.csv")
    csv_files = glob.glob(search_pattern)

    if not csv_files:
        print(f"No CSV files found in directory: '{INPUT_FOLDER}'")
        return

    print(f"Found {len(csv_files)} files to merge.")

    # Read and combine all CSV files into a single list
    dataframes = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            dataframes.append(df)
            print(f" - Loaded: {os.path.basename(file)} ({len(df)} rows)")
        except Exception as e:
            print(f" - Error loading {os.path.basename(file)}: {e}")

    # Concatenate all DataFrames
    merged_df = pd.concat(dataframes, ignore_index=True)

    # Drop any exact duplicates that might have leaked across files
    merged_df = merged_df.drop_duplicates()

    # Shuffle the dataset to ensure proper class distribution during training
    merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save to the final output file
    merged_df.to_csv(OUTPUT_FILE, index=False)

    print("=" * 40)
    print(f"Merge Complete! Saved to: {OUTPUT_FILE}")
    print(f"Total rows in final dataset: {len(merged_df)}")
    print("=" * 40)


if __name__ == "__main__":
    merge_cleaned_datasets()
