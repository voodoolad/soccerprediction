from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import pandas as pd
import numpy as np
import statsmodels.api as sm

@dataclass
class PoissonSpec:
    target_col: str
    feature_cols: Tuple[str, ...]

def _to_numeric_frame(df: pd.DataFrame, cols):
    work = df[list(cols)].copy()
    for c in work.columns:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan).dropna()
    return work

def fit_poisson_glm(df: pd.DataFrame, spec: PoissonSpec):
    cols = list(spec.feature_cols) + [spec.target_col]
    work = _to_numeric_frame(df, cols)
    if work.empty or work[spec.target_col].sum() <= 0:
        raise ValueError(f"Insufficient numeric data for Poisson GLM '{spec.target_col}'")
    X = work[list(spec.feature_cols)].astype(float)
    X = sm.add_constant(X, has_constant="add")
    y = work[spec.target_col].astype(float)
    model = sm.GLM(y, X, family=sm.families.Poisson())
    return model.fit()

def predict_rate(res, features: pd.Series) -> float:
    cols = res.model.exog_names
    row = {"const": 1.0}
    for name in cols:
        if name == "const": continue
        val = pd.to_numeric(features.get(name, 0.0), errors="coerce")
        row[name] = float(0.0 if pd.isna(val) else val)
    X = pd.DataFrame([row], columns=cols)
    lam = float(res.predict(X)[0])
    return max(lam, 1e-6)
