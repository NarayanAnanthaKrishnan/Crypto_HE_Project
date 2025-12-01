# Encrypted Health Prediction Using Fully Homomorphic Encryption (FHE)

This project implements a **privacy-preserving diabetes risk prediction system** using **Fully Homomorphic Encryption (FHE)** with the **CKKS** scheme via **Pyfhel**.

A Logistic Regression model is trained on a diabetes dataset in plaintext, and then inference is performed on **encrypted patient feature vectors**, so that the server can compute predictions **without ever seeing the raw data**.

The project also benchmarks different CKKS parameter configurations (small / medium / large) to study the trade-off between **security, runtime, and numerical error**.

---

## 1. Project Overview

### Goal
- Enable a server to compute diabetes risk scores on encrypted patient data.
- Ensure that **sensitive medical features never appear in plaintext** on the server side.
- Compare:
  - Plaintext Logistic Regression vs. Random Forest.
  - Plaintext vs. FHE-based inference.
  - Multiple CKKS configurations for performance and accuracy.

### Core Ideas
- **Plaintext side**:
  - Train Logistic Regression (for FHE) and Random Forest (upper-bound benchmark).
  - Use standardized medical features like Glucose, BMI, BloodPressure, etc.

- **FHE side (CKKS)**:
  - Client encrypts a standardized feature vector $x$.
  - Server holds plaintext LR weights $w$ and bias $b$.
  - Server computes encrypted dot product $\langle w, x \rangle$ using CKKS.
  - Server sends encrypted score back.
  - Client decrypts and applies the sigmoid function in plaintext to obtain the probability.

- **Parameter exploration**:
  - Run the same encrypted inference pipeline with different CKKS polynomial degrees:
    - **small**: $n = 8192$
    - **medium**: $n = 16384$
    - **large**: $n = 32768$

---

## 2. Repository Structure

Assuming this layout:

```text
project_root/
│
├── README.md
├── requirements.txt
└── src/
    ├── main.py              # Entry point: training, evaluation, FHE experiments
    ├── experiments.py       # Encrypted single-sample, batch, and multi-config runs
    ├── fhe_utils.py         # CKKS context, keygen, encrypt/decrypt, dot product
    ├── diabetes.csv         # Dataset (PIMA-like diabetes data, with Outcome label)
    └── (other files, if any)
````
## 3. How to run
python -m venv venv

# On Linux / macOS
source venv/bin/activate

# On Windows
venv\Scripts\activate

pip install -r requirements.txt

cd src
python main.py
