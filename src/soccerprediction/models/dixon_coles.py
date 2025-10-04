
"""
Vectorized Dixon–Coles (DC) models with optional covariates, time-decay weights,
L2 regularization, and utilities to convert rates to scoreline PMFs and markets.

Improvements implemented:
- Fixed optimizer call (objective vs obj)
- Vectorized likelihood (no Python loops over matches)
- Uses scipy.special.gammaln for log-factorials (stable/fast)
- Optional covariate terms for both home and away lambdas
- Time-decay weighting helper
- PMF grid with DC tau adjustment and normalization
- Convenience aggregations to 1X2 and totals
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize
from scipy.special import gammaln

# ---- helpers ----

def time_decay_weights(match_dates: ArrayLike, as_of_date: Optional[np.datetime64] = None, half_life_days: float = 180.0) -> np.ndarray:
    """
    Compute exponential time-decay weights.
    w = 0.5 ** (age_days / half_life_days)
    """
    dates = np.asarray(match_dates, dtype='datetime64[D]')
    if as_of_date is None:
        as_of_date = np.datetime64('today', 'D')
    age = (as_of_date - dates).astype('timedelta64[D]').astype(float)
    w = np.power(0.5, np.clip(age, 0, None) / float(half_life_days))
    # normalize to keep likelihood scale stable
    s = w.sum()
    return w / s if s > 0 else w

def dc_tau(hg: np.ndarray, ag: np.ndarray, lam_h: np.ndarray, lam_a: np.ndarray, rho: float) -> np.ndarray:
    """
    Dixon–Coles low-score adjustment factor tau(hg, ag).
    Applies only to (0,0), (1,0), (0,1), (1,1).
    """
    tau = np.ones_like(lam_h, dtype=float)
    # build masks
    m00 = (hg == 0) & (ag == 0)
    m10 = (hg == 1) & (ag == 0)
    m01 = (hg == 0) & (ag == 1)
    m11 = (hg == 1) & (ag == 1)
    # apply
    if np.any(m00): tau[m00] = 1.0 - (lam_h[m00] * lam_a[m00] * rho)
    if np.any(m10): tau[m10] = 1.0 + (lam_a[m10] * rho)
    if np.any(m01): tau[m01] = 1.0 + (lam_h[m01] * rho)
    if np.any(m11): tau[m11] = 1.0 - rho
    return tau

def _build_team_params(x: np.ndarray, n_teams: int) -> Tuple[np.ndarray, np.ndarray, float, float, np.ndarray, np.ndarray]:
    """
    Parameter packing (unconstrained) -> (attack, defense, home_adv, rho, beta_h, beta_a)
    Identifiability: enforce sum(a)=0, sum(d)=0 via last param as negative sum of others.
    Layout:
      [a_0, ..., a_{N-2}, d_0, ..., d_{N-2}, home_adv, rho, beta_h..., beta_a...]
    """
    ofs = 0
    a_free = x[ofs:ofs + (n_teams - 1)]; ofs += (n_teams - 1)
    d_free = x[ofs:ofs + (n_teams - 1)]; ofs += (n_teams - 1)
    home = x[ofs]; ofs += 1
    rho  = x[ofs]; ofs += 1
    beta_h = x[ofs:]
    # beta_a will be split outside when needed
    return a_free, d_free, home, rho, beta_h, None

def _expand_team_params(a_free: np.ndarray, d_free: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    a_last = -np.sum(a_free)
    d_last = -np.sum(d_free)
    a = np.concatenate([a_free, np.array([a_last])])
    d = np.concatenate([d_free, np.array([d_last])])
    return a, d

@dataclass
class FitResult:
    x: np.ndarray
    success: bool
    fun: float
    message: str
    niter: int
    nit: int

class DixonColes:
    """
    Classic Dixon–Coles with optional covariates (shared between home/away with side-specific weights).
    If X is provided (shape [M, K]), the log-rate is:
      log lam_h = a_home - d_away + home_adv + (X @ beta) + (X @ gamma_side_home)
      log lam_a = a_away - d_home           + (X @ beta) + (X @ gamma_side_away)
    For simplicity we use disjoint beta vectors per side: beta_h, beta_a.
    Pack them as a single vector [beta_h, beta_a] at the end of x.
    """

    def __init__(
        self,
        n_teams: int,
        home_idx: ArrayLike,
        away_idx: ArrayLike,
        hg: ArrayLike,
        ag: ArrayLike,
        match_dates: Optional[ArrayLike] = None,
        X: Optional[np.ndarray] = None,  # features shared for both sides
        reg_lambda: float = 1e-3,
        half_life_days: float = 180.0,
    ) -> None:
        self.n_teams = int(n_teams)
        self.home_idx = np.asarray(home_idx, dtype=int)
        self.away_idx = np.asarray(away_idx, dtype=int)
        self.hg = np.asarray(hg, dtype=int)
        self.ag = np.asarray(ag, dtype=int)
        assert self.home_idx.shape == self.away_idx.shape == self.hg.shape == self.ag.shape
        self.M = self.hg.shape[0]
        self.X = None if X is None else np.asarray(X, dtype=float)
        self.K = 0 if self.X is None else self.X.shape[1]
        if match_dates is not None:
            self.w = time_decay_weights(match_dates, half_life_days=half_life_days)
        else:
            self.w = np.ones(self.M) / float(self.M)
        self.reg_lambda = float(reg_lambda)

    def _unpack(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float, np.ndarray, np.ndarray]:
        a_free, d_free, home, rho, betas, _ = _build_team_params(x, self.n_teams)
        a, d = _expand_team_params(a_free, d_free)
        if self.K > 0:
            # split betas into beta_h and beta_a
            assert betas.size == 2*self.K, "Parameter length mismatch for covariates"
            beta_h = betas[:self.K]
            beta_a = betas[self.K:]
        else:
            beta_h = np.zeros(0); beta_a = np.zeros(0)
        return a, d, home, rho, beta_h, beta_a

    def _rates(self, a: np.ndarray, d: np.ndarray, home: float, beta_h: np.ndarray, beta_a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        xb_h = 0.0 if self.K == 0 else self.X @ beta_h
        xb_a = 0.0 if self.K == 0 else self.X @ beta_a
        eta_h = a[self.home_idx] - d[self.away_idx] + home + xb_h
        eta_a = a[self.away_idx] - d[self.home_idx] + xb_a
        lam_h = np.exp(eta_h)
        lam_a = np.exp(eta_a)
        return lam_h, lam_a

    def _objective(self, x: np.ndarray) -> float:
        a, d, home, rho, beta_h, beta_a = self._unpack(x)
        lam_h, lam_a = self._rates(a, d, home, beta_h, beta_a)
        # Poisson log-likelihood terms (vectorized)
        ll = (
            self.hg * np.log(lam_h + 1e-12)
            - lam_h
            - gammaln(self.hg + 1.0)
            + self.ag * np.log(lam_a + 1e-12)
            - lam_a
            - gammaln(self.ag + 1.0)
        )
        # DC tau adjustments
        tau = dc_tau(self.hg, self.ag, lam_h, lam_a, rho)
        ll += np.log(np.clip(tau, 1e-12, None))

        # weighted sum
        nll = -np.sum(self.w * ll)

        # L2 regularization on free attack/defense (excluding the last constrained param),
        # and on betas to avoid overfit if provided
        ofs = 0
        a_free = x[ofs:ofs + (self.n_teams - 1)]; ofs += (self.n_teams - 1)
        d_free = x[ofs:ofs + (self.n_teams - 1)]; ofs += (self.n_teams - 1)
        _home = x[ofs]; ofs += 1
        _rho  = x[ofs]; ofs += 1
        betas = x[ofs:]
        reg = self.reg_lambda * (np.dot(a_free, a_free) + np.dot(d_free, d_free) + np.dot(betas, betas))
        return nll + reg

    def fit(self, x0: Optional[np.ndarray] = None) -> FitResult:
        T = self.n_teams
        if x0 is None:
            # init around small values
            x0 = np.zeros((2*(T-1) + 1 + 1 + 2*self.K,), dtype=float)
            x0[-(2*self.K + 2)] = 0.2  # small home advantage
            x0[-(2*self.K + 1)] = -0.05  # small negative rho
            # betas start at 0
        res = minimize(self._objective, x0, method="L-BFGS-B")  # <- fixed call
        return FitResult(x=res.x, success=res.success, fun=res.fun, message=res.message, niter=getattr(res, "niter", -1), nit=res.nit)

    # ---- prediction utilities ----

    @staticmethod
    def scoreline_pmf(lam_h: float, lam_a: float, rho: float, max_goals: int = 10) -> np.ndarray:
        """
        Compute joint PMF over [0..G]x[0..G] with DC adjustments applied to the four low-score cells.
        Returns a (G+1, G+1) array normalized to 1.
        """
        G = int(max_goals)
        hg = np.arange(G+1)
        ag = np.arange(G+1)
        # independent Poisson grid
        ph = np.exp(-lam_h) * np.power(lam_h, hg) / np.exp(gammaln(hg + 1.0))
        pa = np.exp(-lam_a) * np.power(lam_a, ag) / np.exp(gammaln(ag + 1.0))
        grid = np.outer(ph, pa)

        # DC tau adjust
        H, A = np.meshgrid(hg, ag, indexing='ij')
        tau = dc_tau(H, A, lam_h*np.ones_like(H, float), lam_a*np.ones_like(A, float), rho)
        grid *= tau

        # normalize to guard against truncation + numerical drift
        s = grid.sum()
        return grid / s if s > 0 else grid

    @staticmethod
    def markets_from_pmf(pmf: np.ndarray, goal_line: float = 2.5) -> Dict[str, float]:
        """
        Aggregate scoreline PMF into common markets.
        Returns dict with keys: 'home', 'draw', 'away', f'over_{goal_line}', f'under_{goal_line}', 'btts'.
        """
        G = pmf.shape[0] - 1
        home = np.tril(pmf, k=-1).sum()  # hg > ag
        away = np.triu(pmf, k=1).sum()   # ag > hg
        draw = np.trace(pmf)
        # totals
        over = 0.0
        under = 0.0
        btts = 0.0
        for i in range(G+1):
            for j in range(G+1):
                s = i + j
                if s > goal_line: over += pmf[i, j]
                else:             under += pmf[i, j]
                if i > 0 and j > 0: btts += pmf[i, j]
        return {
            "home": float(home),
            "draw": float(draw),
            "away": float(away),
            f"over_{goal_line}": float(over),
            f"under_{goal_line}": float(under),
            "btts": float(btts),
        }
