"""
Feature Engineering
YouTube Video Performance Prediction
Author: Mandaka Nadini
"""

import pandas as pd
import numpy as np


def add_title_length(df: pd.DataFrame) -> pd.DataFrame:
    """Create title_length feature from the video title."""
    df["title_length"] = df["title"].apply(lambda x: len(str(x)))
    return df


def add_tag_count(df: pd.DataFrame) -> pd.DataFrame:
    """Count the number of tags for each video."""
    df["tag_count"] = df["tags"].apply(
        lambda x: len(str(x).split("|")) if x != "[none]" else 0
    )
    return df


def add_engagement_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate engagement rate: (likes + comment_count) / views.
    Clipped to avoid division by zero and extreme outliers.
    """
    df["engagement_rate"] = (
        (df["likes"] + df["comment_count"]) / df["views"].replace(0, np.nan)
    ).fillna(0)
    df["engagement_rate"] = df["engagement_rate"].clip(upper=1.0)
    return df


def add_like_dislike_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate like-to-dislike ratio as a sentiment signal."""
    df["like_dislike_ratio"] = (
        df["likes"] / (df["dislikes"] + 1)  # +1 to avoid division by zero
    )
    return df


def define_viral_label(df: pd.DataFrame, threshold_percentile: int = 75) -> pd.DataFrame:
    """
    Define the target label 'is_viral'.
    Videos above the Nth percentile of views are considered viral (1), others are not (0).
    """
    threshold = df["views"].quantile(threshold_percentile / 100)
    df["is_viral"] = (df["views"] >= threshold).astype(int)
    viral_count = df["is_viral"].sum()
    print(f"Viral threshold (views >= {threshold:,.0f}): {viral_count} viral videos ({viral_count/len(df)*100:.1f}%)")
    return df


def select_features(df: pd.DataFrame) -> tuple:
    """
    Select final features for modeling and return X, y split.
    Returns:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target labels
    """
    feature_cols = [
        "likes",
        "dislikes",
        "comment_count",
        "title_length",
        "tag_count",
        "engagement_rate",
        "like_dislike_ratio",
        "category_id",
        "comments_disabled",
        "ratings_disabled",
        "publish_hour",
        "publish_dayofweek",
        "publish_month",
    ]
    X = df[feature_cols].fillna(0)
    y = df["is_viral"]
    return X, y


def build_features_pipeline(df: pd.DataFrame) -> tuple:
    """Run the full feature engineering pipeline."""
    df = add_title_length(df)
    df = add_tag_count(df)
    df = add_engagement_rate(df)
    df = add_like_dislike_ratio(df)
    df = define_viral_label(df)
    X, y = select_features(df)
    print(f"Feature matrix shape: {X.shape}")
    return X, y


if __name__ == "__main__":
    df = pd.read_csv("../../data/processed/INvideos_cleaned.csv")
    X, y = build_features_pipeline(df)
    print(X.head())
    print(y.value_counts())
