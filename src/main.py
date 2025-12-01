import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

from fhe_utils import FHEContext
from experiments import (
    single_sample_demo,
    batch_experiment,
    multi_config_fhe_experiments,
)


def main():
    # 1. Load and preprocess data
    # Assumes diabetes.csv is in the same folder as main.py (src/)
    df = pd.read_csv("diabetes.csv")

    if "Outcome" not in df.columns:
        raise ValueError("Expected 'Outcome' column in diabetes.csv")

# Drop Id if present (not a real medical feature)
    drop_cols = ["Outcome"]
    if "Id" in df.columns:
        drop_cols.append("Id")

    y = df["Outcome"].values
    X = df.drop(columns=drop_cols).astype(float)
    feature_names = X.columns
    X = X.values

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # Standardize features (very important for LR and for FHE stability)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Loaded data.")
    print(f"Train shape: {X_train_scaled.shape} Test shape: {X_test_scaled.shape}")
    print(f"Features: {list(feature_names)}")

    # 2. Train plaintext models
    logreg_model = LogisticRegression(max_iter=1000, solver="lbfgs")
    logreg_model.fit(X_train_scaled, y_train)

    rf_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
    )
    rf_model.fit(X_train_scaled, y_train)

    # 3. Evaluate plaintext Logistic Regression
    print("\n=== Plaintext Evaluation: Logistic Regression ===")
    y_pred_lr = logreg_model.predict(X_test_scaled)
    print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.3f}")
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_lr))
    print("\nClassification report:\n", classification_report(y_test, y_pred_lr))

    # 4. Evaluate plaintext Random Forest
    print("\n\n=== Plaintext Evaluation: Random Forest ===")
    y_pred_rf = rf_model.predict(X_test_scaled)
    print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.3f}")
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_rf))
    print("\nClassification report:\n", classification_report(y_test, y_pred_rf))

    # 5. Comparison of plaintext models
    acc_lr = accuracy_score(y_test, y_pred_lr)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    print("\n\nComparison of plaintext models:")
    print(f"  Logistic Regression accuracy: {acc_lr:.3f}")
    print(f"  Random Forest accuracy:       {acc_rf:.3f}")
    print("Note: Only Logistic Regression is used for FHE, due to its linear structure.")

    # 6. Baseline FHE context (medium config by default)
    fhe_baseline = FHEContext()

    # 7. Single-sample encrypted demo (dot product encrypted, sigmoid plaintext)
    single_sample_demo(logreg_model, X_test_scaled, y_test, fhe_baseline)

    # 8. Batch encrypted experiment
    batch_experiment(logreg_model, X_test_scaled, y_test, fhe_baseline, n_samples=50)

    # 9. Multi-config FHE experiments (small / medium / large)
    multi_config_fhe_experiments(logreg_model, X_test_scaled, y_test)


if __name__ == "__main__":
    main()
