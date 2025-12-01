# src/experiments.py

import time
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

from fhe_utils import FHEContext


def single_sample_demo(model, X_test, y_test, fhe: FHEContext):
    """
    Single-sample demo:
      - Compare plaintext logistic regression probability/prediction
        with FHE-based dot-product + plaintext sigmoid.
    """
    x = X_test[0]
    y_true = int(y_test[0])

    # Extract LR parameters
    w = model.coef_[0]
    b = float(model.intercept_[0])

    # ----- Plaintext reference -----
    score_plain = float(np.dot(w, x) + b)
    prob_plain = 1.0 / (1.0 + np.exp(-score_plain))
    y_pred_plain = int(prob_plain >= 0.5)

    # ----- Encrypted dot-product pipeline -----
    # Encrypt input
    t_enc_start = time.time()
    ctxt_x = fhe.encrypt_vector(x)
    t_enc = time.time() - t_enc_start

    # Encode weights & compute encrypted dot product
    ptxt_w = fhe.encode_plain_vector(w)
    t_dot_start = time.time()
    ctxt_dot = fhe.dot_encrypted_plain(ctxt_x, ptxt_w)
    t_dot = time.time() - t_dot_start

    # Decrypt and apply sigmoid in plaintext
    t_dec_start = time.time()
    score_fhe = fhe.decrypt_scalar(ctxt_dot) + b
    t_dec = time.time() - t_dec_start

    prob_fhe = 1.0 / (1.0 + np.exp(-score_fhe))
    y_pred_fhe = int(prob_fhe >= 0.5)

    print("\n=== Single-Sample Encrypted Inference Demo ===")
    print("True label:             ", y_true)
    print("Plain prob / pred:      ", f"{prob_plain:.4f}", y_pred_plain)
    print("FHE prob / pred:        ", f"{prob_fhe:.4f}", y_pred_fhe)
    print("Score difference:       ", abs(prob_plain - prob_fhe))
    print("Encryption time (s):    ", f"{t_enc:.4f}")
    print("FHE dot-product (s):    ", f"{t_dot:.4f}")
    print("Decryption time (s):    ", f"{t_dec:.4f}")


def batch_experiment(model, X_test, y_test, fhe: FHEContext, n_samples: int = 50):
    """
    Batch FHE experiment:
      - Runs encrypted inference on n_samples
      - Compares FHE probabilities & predictions to plaintext LR
      - Returns: accuracy, mean probability error, mean FHE runtime
    """
    n_samples = min(n_samples, len(X_test))

    w = model.coef_[0]
    b = float(model.intercept_[0])

    ptxt_w = fhe.encode_plain_vector(w)

    y_true_list = []
    y_fhe_list = []
    prob_errs = []
    fhe_times = []

    for i in range(n_samples):
        x = X_test[i]
        y_true = int(y_test[i])
        y_true_list.append(y_true)

        # Plaintext reference probability
        score_plain = float(np.dot(w, x) + b)
        prob_plain = 1.0 / (1.0 + np.exp(-score_plain))

        # Encrypted dot-product
        ctxt_x = fhe.encrypt_vector(x)

        t_fhe_start = time.time()
        ctxt_dot = fhe.dot_encrypted_plain(ctxt_x, ptxt_w)
        score_fhe = fhe.decrypt_scalar(ctxt_dot) + b
        prob_fhe = 1.0 / (1.0 + np.exp(-score_fhe))
        fhe_time = time.time() - t_fhe_start

        fhe_times.append(fhe_time)
        prob_errs.append(abs(prob_plain - prob_fhe))

        y_pred_fhe = int(prob_fhe >= 0.5)
        y_fhe_list.append(y_pred_fhe)

    y_true_arr = np.array(y_true_list)
    y_fhe_arr = np.array(y_fhe_list)

    acc_fhe = float(accuracy_score(y_true_arr, y_fhe_arr))
    mean_prob_err = float(np.mean(prob_errs))
    mean_time = float(np.mean(fhe_times))

    print("\n=== Batch Encrypted Inference Experiment ===")
    print(f"Samples evaluated:      {n_samples}")
    print(f"FHE accuracy:           {acc_fhe:.3f}")
    print(f"Mean score error:       {mean_prob_err:.6f}")
    print(f"Mean FHE runtime (s):   {mean_time:.4f}")

    return acc_fhe, mean_prob_err, mean_time


def multi_config_fhe_experiments(model, X_test, y_test):
    """
    Runs batch_experiment for multiple CKKS parameter configurations:
      - small, medium, large polynomial modulus degrees
    Prints a summary + tabular form suitable for plotting.
    """
    configs = {
        "small": {
            "n": 8192,
            "qi_sizes": [60, 40, 40, 60],
        },
        "medium": {
            "n": 16384,
            "qi_sizes": [60, 40, 40, 60],
        },
        "large": {
            "n": 32768,
            "qi_sizes": [60, 40, 40, 40, 60],
        },
    }

    results = []

    for name, cfg in configs.items():
        print(
            f"\n>>> Running FHE config: {name} "
            f"(n={cfg['n']}, qi_sizes={cfg['qi_sizes']})"
        )

        # Measure key generation time by creating a new FHEContext
        t_key_start = time.time()
        fhe_cfg = FHEContext(n=cfg["n"], qi_sizes=cfg["qi_sizes"])
        keygen_time = time.time() - t_key_start

        acc_fhe, mean_err, mean_time = batch_experiment(
            model, X_test, y_test, fhe_cfg, n_samples=50
        )

        results.append(
            (
                name,
                cfg["n"],
                keygen_time,
                acc_fhe,
                mean_err,
                mean_time,
            )
        )

    print("\n=== Summary of FHE Config Experiments ===")
    for name, n, keygen_time, acc, err, tmean in results:
        print(
            f"{name}: n={n}, keygen={keygen_time:.3f}s, "
            f"acc={acc:.3f}, err={err:.2e}, time={tmean:.4f}s"
        )

    # Keep this block as-is for your plotting workflow
    print("\nTabular results (for plotting):")
    for name, n, keygen_time, acc, err, tmean in results:
        print(
            f"{name}, n={n}, keygen={keygen_time:.3f}, "
            f"acc={acc:.3f}, err={err:.2e}, time={tmean:.4f}"
        )

    return results
