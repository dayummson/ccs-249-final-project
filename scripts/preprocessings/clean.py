import pandas as pd
import re

# ==========================================
# CONFIGURATION CONSTANTS
# ==========================================
INPUT_FILE_PATH = "./ignore.csv"
OUTPUT_FILE_PATH = "./cleaned/ignore_cleaned.csv"
TEXT_COLUMN_NAME = "text"


def clean_production_text(text):
    if not isinstance(text, str):
        return text

    # 1. Remove literal escaped newline characters
    text = text.replace("\\n", " ")

    # 2. Remove all bracketed tracking headers like [CORE TASK] or [TO DO:]
    text = re.sub(r"\[.*?\]", "", text)

    # 3. Strip all asterisks anywhere in the text (fixes bold/italic markdown)
    text = text.replace("*", "")

    # 4. Strip leading numbers if they act as list items (e.g., "1. ", "12) ")
    text = re.sub(r"^\s*\d+[\.\)]\s+", "", text)

    # 5. AGGRESSIVE PREFIX STRIP: Removes ANY combination of bullets (all unicode variants),
    # dashes, colons, dots, and spaces at the start of the string until the first word character.
    text = re.sub(r"^[\s\.\-–—:;•●○▪·■]+", "", text)

    # 6. Collapse multiple spaces into a single space and strip trailing spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def main():
    df = pd.read_csv(INPUT_FILE_PATH)

    # Apply the cleaning function
    df[TEXT_COLUMN_NAME] = df[TEXT_COLUMN_NAME].apply(clean_production_text)

    # Clean out any empty rows caused by cleaning blank lines
    df = df[df[TEXT_COLUMN_NAME].str.strip() != ""]

    df.to_csv(OUTPUT_FILE_PATH, index=False)
    print(f"Data wash complete. Cleaned file saved to: {OUTPUT_FILE_PATH}")


if __name__ == "__main__":
    main()
