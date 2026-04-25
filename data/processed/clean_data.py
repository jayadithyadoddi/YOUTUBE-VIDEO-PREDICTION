"""
Data Cleaning & Preprocessing
YouTube Video Performance Prediction
Author: Mandaka Nadini
"""

import pandas as pd
import numpy as np


def load_data(filepath: str) -> pd.DataFrame:
    """Load the raw INvideos CSV dataset."""
    df = pd.read_csv(filepath, encoding="utf-8", on_bad_lines="skip")
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows based on video_id and trending_date."""
    before = len(df)
    df = df.drop_duplicates(subset=["video_id", "trending_date"])
    print(f"Removed {before - len(df)} duplicate rows")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values across the dataset."""
    # Fill missing description with empty string
    df["description"] = df["description"].fillna("")
    df["tags"] = df["tags"].fillna("[none]")

    # Drop rows where core engagement metrics are null
    df = df.dropna(subset=["views", "likes", "dislikes", "comment_count"])
    print(f"After handling missing values: {df.shape[0]} rows")
    return df


def convert_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Convert publish_time to datetime and extract useful components."""
    df["publish_time"] = pd.to_datetime(df["publish_time"], errors="coerce")
    df["publish_hour"] = df["publish_time"].dt.hour
    df["publish_dayofweek"] = df["publish_time"].dt.dayofweek
    df["publish_month"] = df["publish_time"].dt.month
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Encode boolean columns and convert category_id to int."""
    bool_cols = ["comments_disabled", "ratings_disabled", "video_error_or_removed"]
    for col in bool_cols:
        df[col] = df[col].map({True: 1, False: 0, "True": 1, "False": 0}).fillna(0).astype(int)

    df["category_id"] = pd.to_numeric(df["category_id"], errors="coerce").fillna(0).astype(int)
    return df


def preprocess_pipeline(filepath: str) -> pd.DataFrame:
    """Run the full preprocessing pipeline."""
    df = load_data(filepath)
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = convert_datetime(df)
    df = encode_categoricals(df)
    print("Preprocessing complete.")
    return df


if __name__ == "__main__":
    df = preprocess_pipeline("../../data/raw/INvideos.csv")
    df.to_csv("../../data/processed/INvideos_cleaned.csv", index=False)
    print("Saved cleaned data.")
