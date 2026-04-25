"""
Model Training
YouTube Video Performance Prediction
Author: Mandaka Nadini
"""

import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from src.preprocessing.clean_data import preprocess_pipeline
from src.features.build_features import build_features_pipeline


def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into train/test sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def train_logistic_regression(X_train, y_train):
    """Train a Logistic Regression baseline model."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    print("Logistic Regression trained.")
    return model, scaler


def train_random_forest(X_train, y_train, tune_hyperparams=True):
    """
    Train a Random Forest classifier.
    If tune_hyperparams=True, runs GridSearchCV for best parameters.
    """
    if tune_hyperparams:
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
        }
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring="f1", n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        print(f"Best RF params: {grid_search.best_params_}")
        model = grid_search.best_estimator_
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1")
    print(f"Random Forest CV F1 scores: {scores.round(3)} | Mean: {scores.mean():.3f}")
    return model


def save_model(model, path: str, filename: str):
    """Save a trained model to disk using joblib."""
    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, filename)
    joblib.dump(model, full_path)
    print(f"Model saved to: {full_path}")


if __name__ == "__main__":
    # 1. Load and preprocess
    df = preprocess_pipeline("data/raw/INvideos.csv")

    # 2. Feature engineering
    X, y = build_features_pipeline(df)

    # 3. Train/test split
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # 4. Train models
    lr_model, scaler = train_logistic_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train, tune_hyperparams=False)

    # 5. Save models
    save_model(lr_model, "models", "logistic_regression.pkl")
    save_model(scaler, "models", "scaler.pkl")
    save_model(rf_model, "models", "random_forest.pkl")

    # Save test data for evaluation
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)
    print("Training complete.")
