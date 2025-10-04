"""
Dixon-Coles Model for Soccer Match Outcomes
✅ FIXED: Now supports time-weighted fitting
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd
from math import factorial
from scipy.optimize import minimize

@dataclass
class DCConfig:
    max_goals: int = 8
    home_field_log: float = 0.07
    rho: float = 0.06
    reg_lambda: float = 1e-3
    time_decay_xi: float = 0.00385  # ln(2)/180 for 180-day half-life

def _poisson_pmf(k: int, lam: float) -> float:
    """Poisson probability mass function"""
    if lam <= 0: 
        return 0.0
    if k > 20:  # Prevent overflow
        return 0.0
    return float((lam**k) * np.exp(-lam) / factorial(k))

def fit_dixon_coles(matches: pd.DataFrame, teams: List[str], cfg: DCConfig, 
                    weights: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Fit Dixon-Coles model with optional time weighting
    
    ✅ NEW: Supports time-weighted fitting via weights parameter
    
    Args:
        matches: DataFrame with columns [home_team, away_team, home_goals, away_goals]
        teams: List of all teams
        cfg: DCConfig object
        weights: Optional array of weights for each match (e.g., time decay)
    
    Returns:
        Dict of parameters: {a_TeamName, d_TeamName, h, rho}
    """
    n = len(teams)
    idx = {t: i for i, t in enumerate(teams)}
    
    # Initialize parameters
    x0 = np.zeros(2*n + 2)
    x0[-2] = cfg.home_field_log  # Home advantage
    x0[-1] = cfg.rho  # Correlation parameter
    
    # Default to uniform weights if not provided
    if weights is None:
        weights = np.ones(len(matches))
    else:
        weights = np.array(weights)
        # Normalize weights to sum to number of matches (keeps scale similar)
        weights = weights * len(matches) / weights.sum()
    
    def objective(x):
        """Negative log-likelihood with optional weights"""
        # Extract parameters
        a = x[:n] - np.mean(x[:n])  # Attack strengths (centered)
        d = x[n:2*n]  # Defense strengths
        h = x[-2]  # Home advantage
        rho = x[-1]  # Low-score correlation
        
        ll = 0.0
        
        for idx_match, (_, r) in enumerate(matches.iterrows()):
            i, j = idx[r['home_team']], idx[r['away_team']]
            
            # Expected goals
            lam_h = np.exp(a[i] - d[j] + h)
            lam_a = np.exp(a[j] - d[i])
            
            hg = int(r['home_goals'])
            ag = int(r['away_goals'])
            
            # Poisson log-likelihood
            match_ll = (hg * np.log(lam_h) - lam_h - np.log(factorial(hg)) +
                       ag * np.log(lam_a) - lam_a - np.log(factorial(ag)))
            
            # Dixon-Coles low-score adjustment
            if hg == 0 and ag == 0:
                match_ll += np.log(1 - lam_h * lam_a * rho)
            elif hg == 0 and ag == 1:
                match_ll += np.log(1 + lam_h * rho)
            elif hg == 1 and ag == 0:
                match_ll += np.log(1 + lam_a * rho)
            elif hg == 1 and ag == 1:
                match_ll += np.log(1 - rho)
            
            # Apply weight (time decay or other)
            ll += weights[idx_match] * match_ll
        
        # Regularization (L2 penalty on parameters)
        ll -= cfg.reg_lambda * (np.sum(a * a) + np.sum(d * d))
        
        return -ll  # Minimize negative log-likelihood
    
    # Optimization
    res = minimize(objective, x0, method="L-BFGS-B")
    
    if not res.success:
        raise RuntimeError(f"Dixon-Coles fit failed: {res.message}")
    
    # Extract fitted parameters
    x = res.x
    a = x[:n] - np.mean(x[:n])
    d = x[n:2*n]
    
    params = {}
    for i in range(n):
        params[f"a_{teams[i]}"] = float(a[i])
        params[f"d_{teams[i]}"] = float(d[i])
    
    params["h"] = float(x[-2])
    params["rho"] = float(x[-1])
    
    return params

def scoreline_pmf(home: str, away: str, params: Dict[str, float], 
                  cfg: DCConfig) -> Dict[Tuple[int, int], float]:
    """
    Calculate scoreline probability mass function
    
    Args:
        home: Home team name
        away: Away team name
        params: Fitted parameters from fit_dixon_coles
        cfg: DCConfig object
    
    Returns:
        Dict of (home_goals, away_goals) -> probability
    """
    # Extract parameters
    a_h = params[f"a_{home}"]
    d_h = params[f"d_{home}"]
    a_a = params[f"a_{away}"]
    d_a = params[f"d_{away}"]
    h = params["h"]
    rho = params["rho"]
    
    # Expected goals
    lam_h = np.exp(a_h - d_a + h)
    lam_a = np.exp(a_a - d_h)
    
    # Calculate probabilities for all scorelines
    grid = {}
    total = 0.0
    
    for hg in range(cfg.max_goals + 1):
        for ag in range(cfg.max_goals + 1):
            # Base Poisson probability
            base = _poisson_pmf(hg, lam_h) * _poisson_pmf(ag, lam_a)
            
            # Dixon-Coles adjustment for low scores
            adjustment = 1.0
            if hg == 0 and ag == 0:
                adjustment = 1 - lam_h * lam_a * rho
            elif hg == 0 and ag == 1:
                adjustment = 1 + lam_h * rho
            elif hg == 1 and ag == 0:
                adjustment = 1 + lam_a * rho
            elif hg == 1 and ag == 1:
                adjustment = 1 - rho
            
            # Final probability (ensure non-negative)
            p = max(0.0, base * adjustment)
            grid[(hg, ag)] = p
            total += p
    
    # Normalize to ensure probabilities sum to 1
    if total > 0:
        for k in grid:
            grid[k] /= total
    
    return grid

def calculate_market_probabilities(pmf: Dict[Tuple[int, int], float]) -> Dict[str, float]:
    """
    Calculate common market probabilities from scoreline PMF
    
    Args:
        pmf: Scoreline probability mass function
    
    Returns:
        Dict of market -> probability
    """
    probs = {}
    
    # 1X2 (Match Result)
    probs['home_win'] = sum(p for (h, a), p in pmf.items() if h > a)
    probs['draw'] = sum(p for (h, a), p in pmf.items() if h == a)
    probs['away_win'] = sum(p for (h, a), p in pmf.items() if h < a)
    
    # Totals
    probs['over_0.5'] = sum(p for (h, a), p in pmf.items() if h + a >= 1)
    probs['over_1.5'] = sum(p for (h, a), p in pmf.items() if h + a >= 2)
    probs['over_2.5'] = sum(p for (h, a), p in pmf.items() if h + a >= 3)
    probs['over_3.5'] = sum(p for (h, a), p in pmf.items() if h + a >= 4)
    probs['over_4.5'] = sum(p for (h, a), p in pmf.items() if h + a >= 5)
    
    probs['under_0.5'] = 1 - probs['over_0.5']
    probs['under_1.5'] = 1 - probs['over_1.5']
    probs['under_2.5'] = 1 - probs['over_2.5']
    probs['under_3.5'] = 1 - probs['over_3.5']
    probs['under_4.5'] = 1 - probs['over_4.5']
    
    # Both Teams To Score
    probs['btts_yes'] = sum(p for (h, a), p in pmf.items() if h > 0 and a > 0)
    probs['btts_no'] = 1 - probs['btts_yes']
    
    # Asian Handicaps (common lines)
    probs['home_-0.5'] = probs['home_win']
    probs['away_+0.5'] = 1 - probs['home_win']
    probs['home_-1.0'] = sum(p for (h, a), p in pmf.items() if h - a >= 2)
    probs['home_-1.5'] = sum(p for (h, a), p in pmf.items() if h - a >= 2)
    
    # Double Chance
    probs['home_or_draw'] = probs['home_win'] + probs['draw']
    probs['away_or_draw'] = probs['away_win'] + probs['draw']
    probs['home_or_away'] = probs['home_win'] + probs['away_win']
    
    # Team Totals
    home_goals_dist = {}
    away_goals_dist = {}
    for (h, a), p in pmf.items():
        home_goals_dist[h] = home_goals_dist.get(h, 0) + p
        away_goals_dist[a] = away_goals_dist.get(a, 0) + p
    
    probs['home_over_0.5'] = sum(p for g, p in home_goals_dist.items() if g >= 1)
    probs['home_over_1.5'] = sum(p for g, p in home_goals_dist.items() if g >= 2)
    probs['home_over_2.5'] = sum(p for g, p in home_goals_dist.items() if g >= 3)
    
    probs['away_over_0.5'] = sum(p for g, p in away_goals_dist.items() if g >= 1)
    probs['away_over_1.5'] = sum(p for g, p in away_goals_dist.items() if g >= 2)
    probs['away_over_2.5'] = sum(p for g, p in away_goals_dist.items() if g >= 3)
    
    return probs

def get_expected_goals(pmf: Dict[Tuple[int, int], float]) -> Tuple[float, float]:
    """
    Calculate expected goals for home and away teams
    
    Returns:
        (home_xg, away_xg)
    """
    home_xg = sum(h * p for (h, a), p in pmf.items())
    away_xg = sum(a * p for (h, a), p in pmf.items())
    
    return home_xg, away_xg
