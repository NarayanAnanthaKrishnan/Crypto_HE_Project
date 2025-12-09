"""
Evaluation Module for FHE-based Machine Learning
Replace your entire src/evaluation.py with this file
"""

import numpy as np
import time
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)
from .client import FHEClient
from .server import FHEServer


def evaluate_plain_model(model, X_test, y_test):
    """
    Evaluate a plaintext model.
    
    Args:
        model: Trained sklearn model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Tuple of (accuracy, confusion_matrix, classification_report)
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=3)
    return acc, cm, report


def evaluate_fhe_batch(model, X_test, y_test, fhe_context, n_samples=50, random_state=42):
    """
    Run FHE inference on a batch of test samples and compare to plaintext.
    
    Args:
        model: Trained sklearn LogisticRegression model
        X_test: Test features (standardized)
        y_test: Test labels
        fhe_context: Initialized FHEContext
        n_samples: Number of samples to evaluate
        random_state: Random seed for sample selection
        
    Returns:
        Dictionary with metrics including:
        - acc_fhe: FHE accuracy
        - mean_prob_error: Mean absolute probability error vs plaintext
        - mean_time: Mean inference time per sample
        - cm_plain: Confusion matrix for plaintext predictions
        - cm_fhe: Confusion matrix for FHE predictions
        - report_fhe: Classification report for FHE
    """
    # Select random samples
    rng = np.random.default_rng(random_state)
    n_total = len(X_test)
    idx = rng.choice(n_total, size=min(n_samples, n_total), replace=False)
    
    # Initialize server and client
    server = FHEServer(model, fhe_context)
    client = FHEClient(fhe_context, bias=server.b)
    
    # Storage for results
    y_true = []
    y_pred_plain = []
    y_pred_fhe = []
    prob_errors = []
    times = []
    
    print(f"Evaluating {len(idx)} samples...")
    
    for i, sample_idx in enumerate(idx):
        x = X_test[sample_idx]
        y = int(y_test[sample_idx])
        
        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(idx)} samples...")
        
        # Plaintext prediction
        z_plain = float(np.dot(server.w, x) + server.b)
        prob_plain = client.sigmoid(z_plain)
        pred_plain = 1 if prob_plain >= 0.5 else 0
        
        # FHE prediction
        start = time.time()
        ctxt_x = client.encrypt_features(x)
        ctxt_dot = server.encrypted_dot(ctxt_x)
        prob_fhe, pred_fhe = client.decrypt_score_and_predict(ctxt_dot)
        elapsed = time.time() - start
        
        # Store results
        y_true.append(y)
        y_pred_plain.append(pred_plain)
        y_pred_fhe.append(pred_fhe)
        prob_errors.append(abs(prob_plain - prob_fhe))
        times.append(elapsed)
    
    # Compute metrics
    y_true = np.array(y_true)
    y_pred_plain = np.array(y_pred_plain)
    y_pred_fhe = np.array(y_pred_fhe)
    
    metrics = {
        "acc_fhe": accuracy_score(y_true, y_pred_fhe),
        "acc_plain": accuracy_score(y_true, y_pred_plain),
        "mean_prob_error": float(np.mean(prob_errors)),
        "max_prob_error": float(np.max(prob_errors)),
        "mean_time": float(np.mean(times)),
        "total_time": float(np.sum(times)),
        "cm_plain": confusion_matrix(y_true, y_pred_plain),
        "cm_fhe": confusion_matrix(y_true, y_pred_fhe),
        "report_fhe": classification_report(y_true, y_pred_fhe, digits=3),
        "report_plain": classification_report(y_true, y_pred_plain, digits=3),
        "prediction_agreement": float(np.mean(y_pred_plain == y_pred_fhe)),
    }
    
    return metrics