"""
FHE Server Module - FINAL VERSION
"""

import numpy as np
from .fhe_context import FHEContext


class FHEServer:
    def __init__(self, model, fhe_context: FHEContext):
        self.model = model
        self.fhe = fhe_context
        self.w = np.array(model.coef_[0], dtype=np.float64)
        self.b = float(model.intercept_[0])
        self.n_features = len(self.w)
        self.ptxt_w = self.fhe.encode_plain_vector(self.w)

    def encrypted_dot(self, ctxt_x):
        ctxt_dot = self.fhe.dot_encrypted_plain(ctxt_x, self.ptxt_w, self.n_features)
        return ctxt_dot