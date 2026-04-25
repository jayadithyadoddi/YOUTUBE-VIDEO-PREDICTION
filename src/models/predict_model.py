"""
Model Prediction
YouTube Video Performance Prediction
Author: Mandaka Nadini
"""

import joblib
import pandas as pd
import numpy as np


def load_model(model_path: str):
    """Load a saved model from disk."""
    return joblib.load(model_path)


def predict_viral(input_features: dict, model_path: str = "models/random_forest.pkl") -> dict:
    """
    Predict whether a video will go viral given its features.

    Args:
        input_features (dict): Dictionary of feature values.
            Expected keys: likes, dislikes, comment_count, title_length,
                           tag_count, engagement_rate, like_dislike_ratio,
                           category_id, comments_disabled, ratings_disabled,
                           publish_hour, publish_dayofweek, publish_month
        model_path (str): Path to the saved model file.

    Returns:
        dict: {
            "prediction": "Viral" | "Not Viral",
            "probability": float (0.0 - 1.0)
        }
    """
    model = load_model(model_path)
    X = pd.DataFrame([input_features])

    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]  # Probability of class 1 (Viral)

    return {
        "prediction": "Viral 🔥" if prediction == 1 else "Not Viral",
        "probability": round(float(probability), 4),
    }


if __name__ == "__main__":
    # Example prediction
    sample = {
        "likes": 50000,
        "dislikes": 1000,
        "comment_count": 3000,
        "title_length": 65,
        "tag_count": 20,
        "engagement_rate": 0.15,
        "like_dislike_ratio": 50.0,
        "category_id": 10,
        "comments_disabled": 0,
        "ratings_disabled": 0,
        "publish_hour": 14,
        "publish_dayofweek": 4,
        "publish_month": 11,
    }

    result = predict_viral(sample)
    print(f"Prediction: {result['prediction']}")
    print(f"Viral Probability: {result['probability'] * 100:.1f}%")
