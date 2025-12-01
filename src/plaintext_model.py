# src/plaintext_model.py

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def train_logreg(X_train, y_train) -> LogisticRegression:
    """
    Trains a logistic regression classifier as the plaintext baseline.
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    """
    Trains a Random Forest classifier to provide a stronger plaintext-only baseline.

    This model is NOT used in the FHE pipeline but helps compare accuracy vs
    FHE-friendliness in the report.
    """
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    return rf


def evaluate_plain(model, X_test, y_test, model_name: str = "Model"):
    """
    Evaluates any sklearn classifier on the test set and prints accuracy + confusion matrix.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n=== Plaintext Evaluation: {model_name} ===")
    print(f"Accuracy: {acc:.3f}")
    print("Confusion matrix:\n", cm)
    print("\nClassification report:\n", classification_report(y_test, y_pred))
    return acc, cm
