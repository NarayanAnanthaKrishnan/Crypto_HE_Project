"""
FHE Context Module - Final Version
"""

from Pyfhel import Pyfhel
import numpy as np


class FHEContext:
    def __init__(self, n=16384, qi_sizes=None, scale=2**30):
        if qi_sizes is None:
            qi_sizes = [60, 40, 40, 60]

        self.HE = Pyfhel()
        self.HE.contextGen(
            scheme="CKKS",
            n=n,
            qi_sizes=qi_sizes,
            scale=scale,
        )
        self.HE.keyGen()
        self.HE.rotateKeyGen()
        self.HE.relinKeyGen()
        
        self.n = n
        self.qi_sizes = qi_sizes
        self.scale = scale
        self.n_slots = n // 2

    def encrypt_vector(self, x):
        x = np.asarray(x, dtype=np.float64).flatten()
        return self.HE.encryptFrac(x)

    def encode_plain_vector(self, w):
        w = np.asarray(w, dtype=np.float64).flatten()
        return self.HE.encodeFrac(w)

    def dot_encrypted_plain(self, ctxt_x, ptxt_w, vector_length):
        """
        Compute dot product of encrypted vector and plaintext vector.
        Result is stored in slot 0 of the returned ciphertext.
        """
        # Step 1: Element-wise multiplication
        # Enc([x0,x1,...]) * [w0,w1,...] = Enc([x0*w0, x1*w1, ...])
        ctxt_prod = ctxt_x * ptxt_w
        
        # Step 2: Sum all slots into slot 0 using rotations
        # We rotate the product vector and add to accumulate sum
        ctxt_sum = self.HE.cumul_add(ctxt_prod)
        
        return ctxt_sum

    def decrypt_scalar(self, ctxt):
        decoded = self.HE.decryptFrac(ctxt)
        return float(decoded[0])

    def decrypt_vector(self, ctxt, length):
        decoded = self.HE.decryptFrac(ctxt)
        return np.array(decoded[:length])