
"""
Probability helpers for corners/cards using simple Poisson totals.

We estimate team-specific rates from recent matches and combine them with
opponent-allowed rates to produce a total-match expected rate lambda_total.
Then we compute P(TOTAL > line) with a Poisson survival function.
"""

from __future__ import annotations
from typing import Tuple, Optional, Iterable
import numpy as np
import pandas as pd
from math import exp, factorial

def poisson_cdf(k: int, lam: float) -> float:
    # CDF(X <= k) = sum_{i=0}^k e^-lam * lam^i / i!
    s = 0.0
    for i in range(max(0, k)+1):
        s += (lam**i) / float(factorial(i))
    return float(exp(-lam) * s)

def poisson_sf(threshold: float, lam: float) -> float:
    """
    Survival P(X > threshold). If threshold is x.5 (e.g., 7.5),
    we use floor(threshold) to compute P(X >= floor(threshold)+1).
    """
    k = int(np.floor(threshold))
    return 1.0 - poisson_cdf(k, lam)

def recency_weights(dates: pd.Series, half_life_days: float = 180.0) -> np.ndarray:
    d = pd.to_datetime(dates).dt.tz_localize(None)
    age = (pd.Timestamp.utcnow().normalize() - d).dt.days.clip(lower=0)
    w = 0.5 ** (age / float(half_life_days))
    return (w / w.sum()).to_numpy()

def team_rate(df: pd.DataFrame, team: str, col_for: str, col_against: str,
              home_col="home", away_col="away", date_col="date",
              half_life_days: float = 180.0, last_n: Optional[int] = 20) -> Tuple[float,float]:
    """
    Return (for_rate, against_rate) for a team using recency weights.
    Expects per-match counts in columns col_for/col_against at the match level.
    """
    d = df[(df[home_col] == team) | (df[away_col] == team)].copy()
    d = d.sort_values(date_col)
    if last_n is not None and len(d) > last_n:
        d = d.tail(last_n)
    w = recency_weights(d[date_col], half_life_days=half_life_days)
    # take the team's perspective
    f = []; a = []
    for r in d.itertuples(index=False):
        if getattr(r, home_col) == team:
            f.append(getattr(r, col_for + "_home"))
            a.append(getattr(r, col_against + "_away"))
        else:
            f.append(getattr(r, col_for + "_away"))
            a.append(getattr(r, col_against + "_home"))
    f = np.asarray(f, float); a = np.asarray(a, float)
    return float((f*w).sum()), float((a*w).sum())

def combined_total_lambda(df: pd.DataFrame, home: str, away: str,
                          col_for: str, col_against: str,
                          home_col="home", away_col="away", date_col="date",
                          half_life_days: float = 180.0, last_n: Optional[int] = 20,
                          league_avg: Optional[float] = None) -> float:
    """
    Simple multiplicative blending:
      team_for_adj = (team_for / league_avg)
      opp_allow_adj = (opp_against / league_avg)
      expected total ≈ league_avg * (home_for_adj * away_allow_adj + away_for_adj * home_allow_adj) / 2
    """
    if league_avg is None:
        # compute league average per match (home+away) for the same window
        tmp = df.sort_values(date_col)
        if last_n is not None and len(tmp) > last_n*2:
            tmp = tmp.tail(last_n*2)
        total = tmp[f"{col_for}_home"] + tmp[f"{col_for}_away"]
        league_avg = float(total.mean())

    h_for, h_allow = team_rate(df, home, col_for, col_against, home_col, away_col, date_col, half_life_days, last_n)
    a_for, a_allow = team_rate(df, away, col_for, col_against, home_col, away_col, date_col, half_life_days, last_n)

    h_for_adj = h_for / (league_avg + 1e-9)
    a_allow_adj = a_allow / (league_avg + 1e-9)
    a_for_adj = a_for / (league_avg + 1e-9)
    h_allow_adj = h_allow / (league_avg + 1e-9)

    lam_total = league_avg * 0.5 * (h_for_adj * a_allow_adj + a_for_adj * h_allow_adj)
    return float(max(0.01, lam_total))
