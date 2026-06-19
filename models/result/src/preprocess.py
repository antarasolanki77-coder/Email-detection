"""
Email Text Preprocessing Module
Cleans and prepares raw email text for ML feature extraction.
"""

import re
import pandas as pd


def clean_text(text):
    """
    Clean a single email text string.

    Steps:
        1. Convert to lowercase
        2. Remove URLs
        3. Remove email addresses
        4. Remove numbers
        5. Remove punctuation and special characters
        6. Remove extra whitespace

    Args:
        text (str): Raw email text.

    Returns:
        str: Cleaned text.
    """
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", "", text)

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove punctuation and special characters (keep only letters and spaces)
    text = re.sub(r"[^a-z\s]", "", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_dataset(df):
    """
    Preprocess the full email dataset.

    Steps:
        1. Drop rows with missing text
        2. Clean all email text
        3. Encode labels: spam=1, ham=0
        4. Drop duplicates

    Args:
        df (pd.DataFrame): DataFrame with 'label' and 'text' columns.

    Returns:
        pd.DataFrame: Preprocessed DataFrame with 'label' (int) and 'text' (str).
    """
    df = df.copy()

    # Handle missing values
    df.dropna(subset=["text"], inplace=True)
    df.dropna(subset=["label"], inplace=True)

    # Clean text
    df["text"] = df["text"].apply(clean_text)

    # Remove empty strings after cleaning
    df = df[df["text"].str.len() > 0]

    # Encode labels
    df["label"] = df["label"].map({"spam": 1, "ham": 0})

    # Drop any rows where label mapping failed (unexpected values)
    df.dropna(subset=["label"], inplace=True)
    df["label"] = df["label"].astype(int)

    # Drop duplicates
    df.drop_duplicates(subset=["text"], inplace=True)

    # Reset index
    df.reset_index(drop=True, inplace=True)

    return df


if __name__ == "__main__":
    # Quick test
    sample = pd.DataFrame({
        "label": ["spam", "ham", "spam"],
        "text": [
            "FREE!!! Win $1000 NOW at http://scam.com!!!",
            "Hi John, the meeting is at 10 AM tomorrow.",
            "   Buy cheap meds!!! Visit www.pills.com   "
        ]
    })

    print("=== Before Preprocessing ===")
    print(sample)
    print()

    result = preprocess_dataset(sample)
    print("=== After Preprocessing ===")
    print(result)
