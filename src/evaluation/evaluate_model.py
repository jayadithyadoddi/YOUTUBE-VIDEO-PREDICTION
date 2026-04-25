"""
Model Evaluation
YouTube Video Performance Prediction
Author: Mandaka Nadini
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)


def evaluate(model, X_test, y_test, model_name: str = "Model") -> dict:
    """Compute and print all evaluation metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

    print(f"\n{'='*40}")
    print(f"  {model_name} Evaluation")
    print(f"{'='*40}")
    for k, v in metrics.items():
        print(f"  {k.upper():<15}: {v:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Not Viral', 'Viral'])}")

    return metrics


def plot_confusion_matrix(model, X_test, y_test, model_name: str = "Model", save_path: str = None):
    """Plot and optionally save a confusion matrix heatmap."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not Viral", "Viral"],
                yticklabels=["Not Viral", "Viral"])
    plt.title(f"Confusion Matrix — {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Confusion matrix saved to {save_path}")
    plt.show()


def plot_roc_curve(models: dict, X_test, y_test, save_path: str = None):
    """
    Plot ROC curves for multiple models.
    Args:
        models (dict): {"Model Name": model_object}
    """
    plt.figure(figsize=(8, 6))
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(loc="lower right")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"ROC curve saved to {save_path}")
    plt.show()


def plot_feature_importance(model, feature_names: list, top_n: int = 15, save_path: str = None):
    """Plot feature importances for tree-based models."""
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.nlargest(top_n).sort_values()

    plt.figure(figsize=(8, 6))
    importances.plot(kind="barh", color="steelblue")
    plt.title(f"Top {top_n} Feature Importances")
    plt.xlabel("Importance Score")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Feature importance plot saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

    rf_model = joblib.load("models/random_forest.pkl")
    lr_model = joblib.load("models/logistic_regression.pkl")

    evaluate(rf_model, X_test, y_test, "Random Forest")
    evaluate(lr_model, X_test, y_test, "Logistic Regression")

    plot_confusion_matrix(rf_model, X_test, y_test, "Random Forest",
                          save_path="reports/figures/rf_confusion_matrix.png")

    plot_roc_curve({"Random Forest": rf_model, "Logistic Regression": lr_model},
                   X_test, y_test,
                   save_path="reports/figures/roc_curves.png")

    plot_feature_importance(rf_model, X_test.columns.tolist(),
                            save_path="reports/figures/feature_importance.png")
