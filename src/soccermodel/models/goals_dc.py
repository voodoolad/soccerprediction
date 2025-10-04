from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List
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

def _poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0: return 0.0
    return float((lam**k) * np.exp(-lam) / factorial(k))

def fit_dixon_coles(matches: pd.DataFrame, teams: List[str], cfg: DCConfig) -> Dict[str, float]:
    n = len(teams)
    idx = {t:i for i,t in enumerate(teams)}
    x0 = np.zeros(2*n + 2); x0[-2] = cfg.home_field_log; x0[-1] = cfg.rho
    def obj(x):
        a = x[:n] - np.mean(x[:n]); d = x[n:2*n]; h = x[-2]; rho = x[-1]
        ll = 0.0
        for _,r in matches.iterrows():
            i, j = idx[r['home_team']], idx[r['away_team']]
            lam_h = np.exp(a[i]-d[j]+h); lam_a = np.exp(a[j]-d[i])
            hg = int(r['home_goals']); ag = int(r['away_goals'])
            ll += hg*np.log(lam_h) - lam_h - np.log(factorial(hg))
            ll += ag*np.log(lam_a) - lam_a - np.log(factorial(ag))
            if hg==0 and ag==0: ll += np.log(1 - lam_h*lam_a*rho)
            elif hg==0 and ag==1: ll += np.log(1 + lam_h*rho)
            elif hg==1 and ag==0: ll += np.log(1 + lam_a*rho)
            elif hg==1 and ag==1: ll += np.log(1 - rho)
        ll -= cfg.reg_lambda*(np.sum(a*a)+np.sum(d*d))
        return -ll
    res = minimize(obj, x0, method="L-BFGS-B")
    if not res.success: raise RuntimeError(f"DC fit failed: {res.message}")
    x = res.x; a = x[:n]-np.mean(x[:n]); d = x[n:2*n]
    params = {f"a_{teams[i]}":float(a[i]) for i in range(n)}
    params.update({f"d_{teams[i]}":float(d[i]) for i in range(n)})
    params["h"]=float(x[-2]); params["rho"]=float(x[-1])
    return params

def scoreline_pmf(home: str, away: str, params: Dict[str,float], cfg: DCConfig) -> Dict[Tuple[int,int],float]:
    a_h, d_h = params[f"a_{home}"], params[f"d_{home}"]
    a_a, d_a = params[f"a_{away}"], params[f"d_{away}"]
    h, rho = params["h"], params["rho"]
    lam_h = np.exp(a_h - d_a + h); lam_a = np.exp(a_a - d_h)
    grid = {}; tot = 0.0
    for hg in range(cfg.max_goals+1):
        for ag in range(cfg.max_goals+1):
            base = _poisson_pmf(hg, lam_h) * _poisson_pmf(ag, lam_a)
            adj = 1.0
            if hg==0 and ag==0: adj = 1 - lam_h*lam_a*rho
            elif hg==0 and ag==1: adj = 1 + lam_h*rho
            elif hg==1 and ag==0: adj = 1 + lam_a*rho
            elif hg==1 and ag==1: adj = 1 - rho
            p = max(0.0, base*adj); grid[(hg,ag)] = p; tot += p
    for k in grid: grid[k] /= tot if tot>0 else 1.0
    return grid
