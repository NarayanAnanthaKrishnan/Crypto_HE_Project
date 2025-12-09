# 🔐 EncryptedHealthFHE

## Privacy-Preserving Diabetes Prediction using Fully Homomorphic Encryption

A secure machine learning system that predicts diabetes risk on **encrypted patient data** using the CKKS homomorphic encryption scheme. The server computes predictions without ever seeing the actual medical information.


---

## 🎯 Overview

Healthcare analytics increasingly relies on machine learning to predict patient outcomes. However, this creates a privacy dilemma: patients must share sensitive medical data with servers to receive predictions.

**Fully Homomorphic Encryption (FHE)** solves this by allowing computations directly on encrypted data. This project demonstrates a practical implementation where:

1. **Patient encrypts** their medical features locally
2. **Server computes** diabetes risk on encrypted data
3. **Patient decrypts** the result - server learns nothing

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🔒 **End-to-End Encryption** | Patient data never exposed to server |
| 🎯 **High Accuracy** | FHE predictions match plaintext (~100% agreement) |
| ⚡ **Practical Performance** | Sub-second inference times |
| 📊 **Comprehensive Evaluation** | Parameter sweep, error analysis, visualizations |
| 🏥 **Healthcare Ready** | Designed for medical risk prediction |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         OFFLINE PHASE                           │
├─────────────────────────────────────────────────────────────────┤
│  Training Data  ──►  Train Logistic Regression  ──►  Weights w  │
│                                                       Bias b    │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                         ONLINE PHASE                            │
├────────────────────────┬────────────────────────────────────────┤
│       CLIENT           │              SERVER                    │
│    (Data Owner)        │          (Model Host)                  │
├────────────────────────┼────────────────────────────────────────┤
│                        │                                        │
│  Patient Features x    │     Weights w, Bias b (plaintext)     │
│         │              │                                        │
│         ▼              │                                        │
│  ┌─────────────┐       │                                        │
│  │ Standardize │       │                                        │
│  └─────────────┘       │                                        │
│         │              │                                        │
│         ▼              │                                        │
│  ┌─────────────┐       │                                        │
│  │  Encrypt    │       │                                        │
│  │  (CKKS)     │       │                                        │
│  └─────────────┘       │                                        │
│         │              │                                        │
│         │  Enc(x)      │                                        │
│         └──────────────┼──────────►  ┌──────────────────┐      │
│                        │             │ Homomorphic      │      │
│                        │             │ Dot Product      │      │
│                        │             │ Enc(z) = Enc(w·x)│      │
│                        │             └──────────────────┘      │
│         ┌──────────────┼─────────────────────┘                 │
│         │   Enc(z)     │                                        │
│         ▼              │                                        │
│  ┌─────────────┐       │                                        │
│  │  Decrypt    │       │                                        │
│  └─────────────┘       │                                        │
│         │              │                                        │
│         ▼              │                                        │
│  ┌─────────────┐       │                                        │
│  │ Add Bias +  │       │                                        │
│  │ Sigmoid     │       │                                        │
│  └─────────────┘       │                                        │
│         │              │                                        │
│         ▼              │                                        │
│   Prediction           │                                        │
│   (0 or 1)             │                                        │
│                        │                                        │
└────────────────────────┴────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
EncryptedHealthFHE/
│
├── 📂 data/
│   └── diabetes.csv              # Pima Indians Diabetes Dataset
│
├── 📂 src/
│   ├── __init__.py               # Package initializer
│   ├── config.py                 # Configuration management
│   ├── data_utils.py             # Data loading & preprocessing
│   ├── models.py                 # Model training (LogReg, RF)
│   ├── fhe_context.py            # CKKS encryption wrapper
│   ├── server.py                 # Server-side FHE computation
│   ├── client.py                 # Client-side encrypt/decrypt
│   └── evaluation.py             # Metrics & evaluation functions
│
├── 📂 notebooks/
│   └── main.ipynb      # Main experimental notebook
│
├── 📄 requirements.txt           # Python dependencies
├── 📄 README.md                  # This file

```

---

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- C++ compiler (required for Pyfhel)
  - **Windows**: Visual Studio Build Tools
  - **Linux**: `sudo apt-get install build-essential cmake`
  - **macOS**: `xcode-select --install`

### Step-by-Step Setup

```bash
# 1. Clone the repository
git clone https://github.com/NarayanAnanthaKrishnan/Crypto_HE_Project.git

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Verify installation
python -c "from Pyfhel import Pyfhel; print('Pyfhel installed successfully!')"
```

### Requirements

```txt
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
Pyfhel>=3.4.0
matplotlib>=3.5.0
jupyter>=1.0.0
```

---


### Running the Notebook

```bash
cd notebooks
jupyter notebook 01_experiments.ipynb
```

---

## 📊 Experimental Results

### Plaintext Model Comparison

| Model | Accuracy | FHE Compatible |
|-------|----------|----------------|
| Logistic Regression | ~77% | ✅ Yes |
| Random Forest | ~99% | ❌ No |

### FHE vs Plaintext Performance

| Metric | Value |
|--------|-------|
| Prediction Agreement | ~98-100% |
| Mean Probability Error | ~10⁻² to 10⁻³ |
| Mean Inference Time | ~0.1-0.4s |

### Parameter Configuration Trade-offs

| Config | Poly Degree | Key Gen Time | Inference Time |
|--------|-------------|--------------|----------------|
| Small | 8,192 | ~0.3s | ~0.05s |
| Medium | 16,384 | ~1.2s | ~0.15s |
| Large | 32,768 | ~5.0s | ~0.40s |

---

## 🔧 Technical Details

### Why CKKS?

CKKS (Cheon-Kim-Kim-Song) is chosen because:
- Native support for **real numbers** (perfect for ML)
- Efficient **SIMD operations** (parallel slot computation)
- **Approximate arithmetic** acceptable for ML (small errors don't change predictions)

### Why Logistic Regression?

| Aspect | Logistic Regression | Random Forest |
|--------|---------------------|---------------|
| Core Operation | Dot product (w·x) | Comparisons (if x < t) |
| FHE Complexity | O(n) multiplications | O(depth × trees) comparisons |
| FHE Feasibility | ✅ Efficient | ❌ Very expensive |

Comparisons in FHE require high-degree polynomial approximations, making decision trees impractical.

### Security Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| n (poly degree) | 16,384 | Security level (~128-bit) |
| qi_sizes | [60,40,40,60] | Coefficient modulus chain |
| scale | 2³⁰ | Encoding precision |

---
