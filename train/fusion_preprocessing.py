"""Invertible tail compression after training-fold descriptor standardization."""
import numpy as np


def transform_descriptors(z, kind="none"):
    if kind == "none":
        return z
    if kind == "signed_log":
        return np.sign(z) * np.log1p(np.abs(z))
    raise ValueError(kind)
