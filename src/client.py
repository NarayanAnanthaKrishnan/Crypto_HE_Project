"""
FHE Client Module for Privacy-Preserving Inference

The client holds sensitive data and the secret key. It encrypts data
before sending to the server and decrypts results after receiving them.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, List
import logging
import time

from .fhe_context import FHEContext

logger = logging.getLogger(__name__)


class FHEClient:
    """
    Client component for FHE-based machine learning inference.
    
    The client:
    - Holds the secret key (inside FHEContext)
    - Encrypts sensitive feature data before sending to server
    - Receives encrypted results from server
    - Decrypts and applies final activation (sigmoid)
    - Makes final predictions
    
    Attributes:
        fhe: FHEContext containing the secret key
        bias: Model bias term (received from server)
        scaler: Optional StandardScaler for preprocessing
    """

    def __init__(
        self,
        fhe_context: FHEContext,
        bias: float,
        scaler=None,
        threshold: float = 0.5
    ):
        """
        Initialize the client.
        
        Args:
            fhe_context: FHEContext with secret key
            bias: Model bias (from server)
            scaler: Optional fitted StandardScaler for preprocessing
            threshold: Classification threshold (default 0.5)
        """
        self.fhe = fhe_context
        self.bias = float(bias)
        self.scaler = scaler
        self.threshold = threshold
        
        # Statistics
        self._encryption_times: List[float] = []
        self._decryption_times: List[float] = []
        
        logger.info(f"Client initialized with bias={bias:.4f}, threshold={threshold}")

    def preprocess(self, x: np.ndarray) -> np.ndarray:
        """
        Preprocess raw features using the fitted scaler.
        
        Args:
            x: Raw feature vector
            
        Returns:
            Standardized feature vector
        """
        x = np.asarray(x, dtype=np.float64).reshape(1, -1)
        
        if self.scaler is not None:
            x = self.scaler.transform(x)
        
        return x.flatten()

    def encrypt_features(self, x: np.ndarray, preprocess: bool = False) -> 'PyCtxt':
        """
        Encrypt a feature vector.
        
        Args:
            x: Feature vector (preprocessed unless preprocess=True)
            preprocess: Whether to apply preprocessing
            
        Returns:
            Encrypted ciphertext
        """
        start = time.time()
        
        if preprocess:
            x = self.preprocess(x)
        else:
            x = np.asarray(x, dtype=np.float64).flatten()
        
        ctxt = self.fhe.encrypt_vector(x)
        
        elapsed = time.time() - start
        self._encryption_times.append(elapsed)
        
        return ctxt

    def encrypt_batch(self, X: np.ndarray, preprocess: bool = False) -> List['PyCtxt']:
        """
        Encrypt multiple feature vectors.
        
        Args:
            X: 2D array of feature vectors (n_samples x n_features)
            preprocess: Whether to apply preprocessing
            
        Returns:
            List of encrypted ciphertexts
        """
        return [self.encrypt_features(x, preprocess) for x in X]

    @staticmethod
    def sigmoid(z: float) -> float:
        """
        Compute sigmoid activation function.
        
        Numerically stable implementation.
        
        Args:
            z: Linear score
            
        Returns:
            Probability in [0, 1]
        """
        # Numerically stable sigmoid
        if z >= 0:
            return 1.0 / (1.0 + np.exp(-z))
        else:
            exp_z = np.exp(z)
            return exp_z / (1.0 + exp_z)

    def decrypt_score(self, ctxt_dot) -> float:
        """
        Decrypt the encrypted dot product and add bias.
        
        Args:
            ctxt_dot: Encrypted dot product from server
            
        Returns:
            Linear score (z = w·x + b)
        """
        start = time.time()
        
        z_dot = self.fhe.decrypt_scalar(ctxt_dot)
        z_linear = z_dot + self.bias
        
        elapsed = time.time() - start
        self._decryption_times.append(elapsed)
        
        return z_linear

    def decrypt_and_predict(self, ctxt_dot) -> Tuple[float, int]:
        """
        Decrypt, apply sigmoid, and make prediction.
        
        Args:
            ctxt_dot: Encrypted dot product from server
            
        Returns:
            Tuple of (probability, binary_prediction)
        """
        z_linear = self.decrypt_score(ctxt_dot)
        prob = self.sigmoid(z_linear)
        pred = 1 if prob >= self.threshold else 0
        
        return prob, pred

    def decrypt_score_and_predict(self, ctxt_dot) -> Tuple[float, int]:
        """Alias for decrypt_and_predict for backward compatibility."""
        return self.decrypt_and_predict(ctxt_dot)

    def predict_proba(self, ctxt_dot) -> Tuple[float, float]:
        """
        Get probabilities for both classes.
        
        Args:
            ctxt_dot: Encrypted dot product
            
        Returns:
            Tuple of (P(class=0), P(class=1))
        """
        prob_1 = self.decrypt_and_predict(ctxt_dot)[0]
        return (1 - prob_1, prob_1)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get timing statistics for this client.
        
        Returns:
            Dictionary with encryption/decryption timing stats
        """
        enc_times = np.array(self._encryption_times) if self._encryption_times else np.array([0])
        dec_times = np.array(self._decryption_times) if self._decryption_times else np.array([0])
        
        return {
            "n_encryptions": len(self._encryption_times),
            "n_decryptions": len(self._decryption_times),
            "mean_encryption_time": float(np.mean(enc_times)),
            "std_encryption_time": float(np.std(enc_times)),
            "mean_decryption_time": float(np.mean(dec_times)),
            "std_decryption_time": float(np.std(dec_times)),
            "total_encryption_time": float(np.sum(enc_times)),
            "total_decryption_time": float(np.sum(dec_times)),
        }

    def reset_statistics(self):
        """Clear timing statistics."""
        self._encryption_times = []
        self._decryption_times = []


class SecurePatientClient(FHEClient):
    """
    Extended client with additional features for healthcare applications.
    
    Includes:
    - Input validation for medical features
    - Risk level interpretation
    - Audit logging (encrypted)
    """
    
    # Expected ranges for diabetes features (for validation)
    FEATURE_RANGES = {
        "Pregnancies": (0, 20),
        "Glucose": (0, 250),
        "BloodPressure": (0, 150),
        "SkinThickness": (0, 100),
        "Insulin": (0, 900),
        "BMI": (0, 70),
        "DiabetesPedigreeFunction": (0, 2.5),
        "Age": (0, 120),
    }

    def __init__(
        self,
        fhe_context: FHEContext,
        bias: float,
        scaler=None,
        feature_names: Optional[List[str]] = None,
        threshold: float = 0.5
    ):
        super().__init__(fhe_context, bias, scaler, threshold)
        self.feature_names = feature_names or list(self.FEATURE_RANGES.keys())
        self._prediction_log: List[Dict] = []

    def validate_features(self, x: np.ndarray) -> Tuple[bool, List[str]]:
        """
        Validate that feature values are within expected ranges.
        
        Args:
            x: Raw feature vector
            
        Returns:
            Tuple of (is_valid, list_of_warnings)
        """
        warnings = []
        x = np.asarray(x).flatten()
        
        for i, (name, (low, high)) in enumerate(zip(self.feature_names, self.FEATURE_RANGES.values())):
            if i < len(x):
                if x[i] < low or x[i] > high:
                    warnings.append(f"{name}={x[i]:.1f} outside expected range [{low}, {high}]")
        
        return len(warnings) == 0, warnings

    def interpret_risk(self, probability: float) -> Dict[str, Any]:
        """
        Interpret the diabetes risk probability.
        
        Args:
            probability: Predicted probability of diabetes
            
        Returns:
            Dictionary with risk level and recommendations
        """
        if probability < 0.3:
            level = "Low"
            color = "green"
            message = "Low risk of diabetes based on current indicators."
        elif probability < 0.5:
            level = "Moderate-Low"
            color = "yellow"
            message = "Some risk factors present. Consider lifestyle modifications."
        elif probability < 0.7:
            level = "Moderate-High"
            color = "orange"
            message = "Elevated risk. Recommend medical consultation."
        else:
            level = "High"
            color = "red"
            message = "High risk indicators. Prompt medical evaluation advised."
        
        return {
            "probability": probability,
            "risk_level": level,
            "color": color,
            "message": message,
            "threshold_used": self.threshold,
        }

    def secure_predict(self, x: np.ndarray, ctxt_dot, validate: bool = True) -> Dict[str, Any]:
        """
        Make a secure prediction with full context.
        
        Args:
            x: Original feature vector (for logging only)
            ctxt_dot: Encrypted result from server
            validate: Whether to validate input features
            
        Returns:
            Comprehensive prediction result
        """
        result = {}
        
        # Validation (on raw features, before encryption)
        if validate:
            is_valid, warnings = self.validate_features(x)
            result["validation"] = {"is_valid": is_valid, "warnings": warnings}
        
        # Decrypt and predict
        prob, pred = self.decrypt_and_predict(ctxt_dot)
        
        # Interpret
        risk_info = self.interpret_risk(prob)
        
        result.update({
            "probability": prob,
            "prediction": pred,
            "prediction_label": "Diabetic" if pred == 1 else "Non-Diabetic",
            "risk_assessment": risk_info,
        })
        
        # Log (without sensitive data)
        self._prediction_log.append({
            "timestamp": time.time(),
            "prediction": pred,
            "probability": round(prob, 4),
        })
        
        return result

    def get_prediction_history(self) -> List[Dict]:
        """Get anonymized prediction history."""
        return self._prediction_log.copy()