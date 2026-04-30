import pandas as pd


def merge_csv_files(file_list, output_file):
    merged_df = pd.DataFrame()

    for file in file_list:
        df = pd.read_csv(file)
        merged_df = pd.concat([merged_df, df], ignore_index=True)

    merged_df.to_csv(output_file, index=False)
    print(
        f"Successfully merged {len(file_list)} files into {output_file} with {len(merged_df)} rows."
    )


if __name__ == "__main__":
    csv_files = [
        "admin_trap.csv",
        "technical_task.csv",
        "pre_requisites.csv",
        "ignore.csv",
        "optional_bonus.csv",
        "weighted_priority.csv",
    ]
    merge_csv_files(csv_files, "./datasets/synthetic_data.csv")
