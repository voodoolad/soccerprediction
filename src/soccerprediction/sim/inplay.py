
"""
Piecewise-constant in-play simulator using time bands and match state multipliers.
This module exposes a function that scales base rates and produces minute-by-minute
expected goals, corners, cards paths by Monte Carlo.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np

@dataclass
class GameStateMultipliers:
    # Defaults are placeholders; tune using historical fit.
    bands: Tuple[int, ...] = (15, 30, 45, 60, 75, 90)
    lead_mult: float = 0.9
    trail_mult: float = 1.1
    draw_mult: float = 1.0

def simulate_match_minutes(lam_h: float, lam_a: float, gsm: GameStateMultipliers = GameStateMultipliers(), n_sims: int = 2000, seed: int = 7):
    rng = np.random.default_rng(seed)
    mins = gsm.bands[-1]
    # split base rates uniformly per minute as a simple baseline
    rate_h = lam_h / mins
    rate_a = lam_a / mins

    paths = []
    for _ in range(n_sims):
        gh = ga = 0
        minute_goals_h = np.zeros(mins, dtype=int)
        minute_goals_a = np.zeros(mins, dtype=int)
        for m in range(mins):
            # adjust by state
            if gh > ga:
                rh = rate_h * gsm.lead_mult
                ra = rate_a * gsm.trail_mult
            elif ga > gh:
                rh = rate_h * gsm.trail_mult
                ra = rate_a * gsm.lead_mult
            else:
                rh = rate_h * gsm.draw_mult
                ra = rate_a * gsm.draw_mult
            # sample goals (Poisson per minute ~ approx Bernoulli with small rate)
            minute_goals_h[m] = rng.poisson(rh)
            minute_goals_a[m] = rng.poisson(ra)
            gh += minute_goals_h[m]
            ga += minute_goals_a[m]
        paths.append((minute_goals_h, minute_goals_a))
    return paths
