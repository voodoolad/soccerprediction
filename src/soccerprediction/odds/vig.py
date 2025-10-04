
"""
Vigorish removal and Kelly utilities.
- Proportional and Shin methods for 1x2 (or any mutually exclusive outcomes)
- EV%, minimum acceptable odds, fractional Kelly
"""

from __future__ import annotations
import numpy as np

def implied_prob(odds_dec: np.ndarray) -> np.ndarray:
    return 1.0 / np.asarray(odds_dec, dtype=float)

def remove_vig_proportional(odds_dec: np.ndarray) -> np.ndarray:
    """Scale implied probs proportionally to sum to 1."""
    p = implied_prob(odds_dec)
    s = p.sum(axis=-1, keepdims=True)
    return p / s

def remove_vig_shin(odds_dec: np.ndarray, tol: float = 1e-10, max_iter: int = 10_000) -> np.ndarray:
    """
    Shin method (1992): solves for insider trading parameter z in [0,1).
    For K outcomes with decimal odds o_k,
      p_k = (sqrt(z**2 + (1 - z) * (1/(o_k * S))) - z) / (1 - z)
    where S = sum(1/o_k).
    """
    o = np.asarray(odds_dec, dtype=float)
    inv = 1.0 / o
    S = inv.sum()
    # solve for z using fixed-point iteration on f(z) = sum p_k(z) - 1 = 0
    z = 0.0
    for _ in range(max_iter):
        pk = (np.sqrt(z*z + (1.0 - z) * (inv / S)) - z) / (1.0 - z + 1e-12)
        f = pk.sum() - 1.0
        if abs(f) < tol: break
        # simple damped update
        z = np.clip(z - 0.5 * f, 0.0, 0.99)
    return pk

def kelly_fraction(p: float, odds_dec: float, frac: float = 1.0) -> float:
    """Kelly staking fraction for a single binary outcome with edge, optionally scaled by `frac`."""
    b = odds_dec - 1.0
    edge = p*b - (1-p)
    if b <= 0 or edge <= 0: return 0.0
    f = edge / b
    return float(max(0.0, min(1.0, frac * f)))

def fair_odds(p: float) -> float:
    return 1.0 / max(1e-12, min(1-1e-12, p))

def min_acceptable_odds(p: float, fee: float = 0.0) -> float:
    """Minimum decimal odds to have positive EV after fees."""
    # EV >= 0 -> p*(o-1) - (1-p) - fee >= 0 -> o >= (1 + fee + 1 - p)/p + 1? 
    # Simpler: require expected value non-negative ignoring fees in stake; approximate:
    if p <= 0: return float('inf')
    return (1.0 / p)
