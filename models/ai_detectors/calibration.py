"""Temperature-scaling calibration (Guo et al., 2017, "On Calibration of
Modern Neural Networks").

Fits a single scalar T > 0 that divides pre-softmax logits before the
softmax, minimizing negative log-likelihood on a held-out labeled set. This
smooths an overconfident model's probability estimates without changing
which class it predicts -- dividing every logit by the same positive
constant never changes which one is largest (the argmax), only how sharply
peaked the resulting distribution is.

This module has no dependency on the transformer itself -- it operates on
whatever logits you hand it -- so it's fully unit-testable with synthetic
arrays, independent of whether the model can actually be downloaded.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def negative_log_likelihood(temperature: float, logits: np.ndarray, true_indices: np.ndarray) -> float:
    scaled = logits / temperature
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return float(-log_probs[np.arange(len(true_indices)), true_indices].mean())


def fit_temperature(logits: np.ndarray, true_indices: np.ndarray) -> float:
    """logits: shape (n, n_classes). true_indices: shape (n,), the index of
    the correct class for each row. Returns the temperature minimizing NLL
    on this data.
    """
    result = minimize_scalar(
        lambda temperature: negative_log_likelihood(temperature, logits, true_indices),
        bounds=(0.05, 10.0),
        method="bounded",
    )
    return float(result.x)
