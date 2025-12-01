# src/plot_fhe_results.py

import matplotlib.pyplot as plt

# These values are taken from your latest run of main.py
results = [
    {"name": "small",  "n": 8192,   "keygen": 0.358, "acc": 0.800, "err": 7.83e-07, "time": 0.0190},
    {"name": "medium", "n": 16384,  "keygen": 0.744, "acc": 0.800, "err": 2.13e-06, "time": 0.0448},
    {"name": "large",  "n": 32768,  "keygen": 2.432, "acc": 0.800, "err": 5.38e-06, "time": 0.1689},
]

names  = [r["name"] for r in results]
times  = [r["time"] for r in results]
errors = [r["err"] for r in results]
accs   = [r["acc"] for r in results]
keygen = [r["keygen"] for r in results]

# 1) Runtime vs config
plt.figure()
plt.bar(names, times)
plt.ylabel("Mean FHE runtime per sample (s)")
plt.xlabel("CKKS configuration")
plt.title("Encrypted inference runtime vs CKKS parameters")
plt.tight_layout()
plt.savefig("fhe_runtime_vs_config.png")

# 2) Approximation error vs config (log-scale y)
plt.figure()
plt.bar(names, errors)
plt.ylabel("Mean score error |plain - FHE|")
plt.xlabel("CKKS configuration")
plt.yscale("log")
plt.title("CKKS approximation error vs parameters")
plt.tight_layout()
plt.savefig("fhe_error_vs_config.png")

# 3) Key generation time vs config
plt.figure()
plt.bar(names, keygen)
plt.ylabel("Key generation time (s)")
plt.xlabel("CKKS configuration")
plt.title("CKKS context + key generation cost")
plt.tight_layout()
plt.savefig("fhe_keygen_vs_config.png")

# 4) (Optional) Accuracy vs config
plt.figure()
plt.bar(names, accs)
plt.ylabel("FHE accuracy")
plt.xlabel("CKKS configuration")
plt.ylim(0.7, 1.0)
plt.title("Encrypted logistic regression accuracy vs CKKS configuration")
plt.tight_layout()
plt.savefig("fhe_accuracy_vs_config.png")

print("Plots saved as PNGs in src/ directory.")
