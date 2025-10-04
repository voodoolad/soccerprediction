
"""
Probability calibration utilities:
- Isotonic regression (for any market)
- Logistic (Platt scaling) for binary markets

Also exposes reliability diagnostics (expected calibration error) and
a simple PIT histogram utility for totals/scorelines (returned as arrays;
plotting is up to the caller).
"""

from __future__ import annotations
from typing import Tuple, Dict
import numpy as np

try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
except Exception as e:
    IsotonicRegression = None
    LogisticRegression = None

def calibrate_isotonic(y_true: np.ndarray, p_pred: np.ndarray):
    if IsotonicRegression is None:
        raise ImportError("sklearn is required for isotonic calibration")
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p_pred, y_true)
    return iso

def calibrate_platt(y_true: np.ndarray, p_pred: np.ndarray):
    if LogisticRegression is None:
        raise ImportError("sklearn is required for Platt scaling")
    # Fit LR on logits to avoid saturation
    eps = 1e-12
    logits = np.log(np.clip(p_pred, eps, 1-eps) / np.clip(1-p_pred, eps, 1-eps))
    lr = LogisticRegression(C=1.0, solver="lbfgs")
    lr.fit(logits.reshape(-1,1), y_true.astype(int))
    return lr

def apply_calibrator(calibrator, p_pred: np.ndarray) -> np.ndarray:
    # both sklearn objects implement predict or transform differently
    if hasattr(calibrator, "predict_proba"):
        # LogisticRegression returns proba for class 1 in column 1
        proba = calibrator.predict_proba(np.log(p_pred/(1-p_pred)).reshape(-1,1))[:,1]
    else:
        proba = calibrator.transform(p_pred)
    return np.clip(proba, 0.0, 1.0)

def brier_score(y_true: np.ndarray, p_pred: np.ndarray) -> float:
    y = y_true.astype(float)
    return float(np.mean((p_pred - y)**2))

def expected_calibration_error(y_true: np.ndarray, p_pred: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0,1,n_bins+1)
    idx = np.digitize(p_pred, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        mask = (idx == b)
        if not np.any(mask): continue
        conf = p_pred[mask].mean()
        acc  = y_true[mask].mean()
        ece += (mask.mean()) * abs(acc - conf)
    return float(ece)
