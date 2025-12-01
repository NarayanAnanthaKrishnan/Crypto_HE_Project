# src/fhe_utils.py

import numpy as np
from Pyfhel import Pyfhel


class FHEContext:
    """
    Wrapper around Pyfhel CKKS context for our project.

    Responsibilities:
      - Generate CKKS context & keys
      - Encrypt / decrypt vectors & scalars
      - Encode plaintext weight vectors
      - Compute encrypted dot products
      - Provide helpers for encrypted linear "sigmoid" approximation
    """

    def __init__(self, n: int = 2 ** 14, qi_sizes=None, scale: float = 2 ** 30):
        if qi_sizes is None:
            qi_sizes = [60, 40, 40, 60]

        self.n = n
        self.qi_sizes = qi_sizes
        self.scale = scale

        self.HE = Pyfhel()
        # CKKS context
        self.HE.contextGen(
            scheme="CKKS",
            n=self.n,
            scale=self.scale,
            qi_sizes=self.qi_sizes,
        )
        # Keys
        self.HE.keyGen()
        self.HE.rotateKeyGen()
        self.HE.relinKeyGen()

    # -----------------------
    # Basic encode/encrypt APIs
    # -----------------------

    def encrypt_vector(self, x: np.ndarray):
        """
        Encrypt a real-valued feature vector using CKKS.
        """
        arr = np.array(x, dtype=float)
        # encryptFrac does encode + encrypt for CKKS
        ctxt = self.HE.encryptFrac(arr)
        return ctxt

    def encode_plain_vector(self, w: np.ndarray):
        """
        Encode a real-valued weight vector as a CKKS plaintext.
        This will be used for encrypted-plaintext dot products.
        """
        arr = np.array(w, dtype=float)
        ptxt = self.HE.encodeFrac(arr)
        return ptxt

    def dot_encrypted_plain(self, ctxt_x, ptxt_w):
        """
        Compute an encrypted dot product between an encrypted vector (ctxt_x)
        and a plaintext encoded vector (ptxt_w).
        """
        # Pyfhel supports scalar (dot) product with scalar_prod_plain
        ctxt_dot = self.HE.scalar_prod_plain(ctxt_x, ptxt_w)
        return ctxt_dot

    def decrypt_scalar(self, ctxt):
        """
        Decrypt a CKKS ciphertext assumed to contain a single scalar
        (or a vector where we care about the first slot).
        Returns a Python float.
        """
        vals = self.HE.decryptFrac(ctxt)
        # decryptFrac usually returns a numpy array or list of floats
        if isinstance(vals, (list, tuple, np.ndarray)):
            return float(vals[0])
        return float(vals)

    # -----------------------
    # Helpers for ciphertext-plaintext operations
    # -----------------------

    def _encode_scalar_full_slots(self, scalar: float):
        """
        Encodes a scalar as a CKKS plaintext repeated across all slots.

        Some Pyfhel versions are picky about the plaintext size matching
        the ciphertext slots, so we encode a full-length vector of size n//2.
        """
        slot_count = self.n // 2
        vec = np.full(slot_count, scalar, dtype=float)
        ptxt = self.HE.encodeFrac(vec)
        return ptxt

    def _add_plain_scalar(self, ctxt, scalar: float):
        """
        Adds a real scalar to all slots of a ciphertext:
            ctxt + scalar
        Implemented using CKKS encode + add_plain.
        """
        ptxt = self._encode_scalar_full_slots(scalar)
        res = self.HE.add_plain(ctxt, ptxt)
        return res

    def _mul_plain_scalar(self, ctxt, scalar: float):
        """
        Multiplies all slots of a ciphertext by a real scalar:
            ctxt * scalar
        Implemented using CKKS encode + multiply_plain.
        """
        ptxt = self._encode_scalar_full_slots(scalar)
        res = self.HE.multiply_plain(ctxt, ptxt)
        return res

    # -----------------------
    # Encrypted "sigmoid" approximation
    # -----------------------

    def encrypted_sigmoid_linear(self, ctxt_dot, bias: float, alpha: float = 0.125):
        """
        Fully-encrypted linear approximation of the sigmoid function:

            sigmoid(x) ≈ 0.5 + alpha * (x + b)

        where:
          - ctxt_dot = ciphertext encoding w·x
          - b        = bias (plaintext)
          - alpha    = slope of the linear approximation

        Steps under FHE:
          1) ctxt_score = ctxt_dot + b      (add bias homomorphically)
          2) ctxt_scaled = alpha * ctxt_score
          3) ctxt_sigmoid = 0.5 + ctxt_scaled

        Returns:
          - ctxt_sigmoid: ciphertext encoding approx sigmoid(w·x + b)
        """
        # (1) Add bias under encryption: (w·x + b)
        ctxt_score = self._add_plain_scalar(ctxt_dot, bias)

        # (2) Multiply by alpha: alpha * (w·x + b)
        ctxt_scaled = self._mul_plain_scalar(ctxt_score, alpha)

        # (3) Add 0.5: 0.5 + alpha * (w·x + b)
        ctxt_sigmoid = self._add_plain_scalar(ctxt_scaled, 0.5)

        return ctxt_sigmoid
