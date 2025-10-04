
"""
Bivariate Poisson model for (GoalsH, GoalsA):
X = Z1 + Z3, Y = Z2 + Z3, Zk ~ Poisson(lambda_k), capturing positive dependence via lambda_3.

Improvements implemented:
- Optional covariates for lambda_1 (home) and lambda_2 (away); lambda_3 kept global.
- Stable log-likelihood with gammaln and summation over k in [0..min(h,a)]
- Time-decay weights and L2 regularization
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike
from typing import Optional, Tuple
from scipy.optimize import minimize
from scipy.special import gammaln

def time_decay_weights(match_dates: ArrayLike, as_of_date: Optional[np.datetime64] = None, half_life_days: float = 180.0) -> np.ndarray:
    dates = np.asarray(match_dates, dtype='datetime64[D]')
    if as_of_date is None:
        as_of_date = np.datetime64('today', 'D')
    age = (as_of_date - dates).astype('timedelta64[D]').astype(float)
    w = np.power(0.5, np.clip(age, 0, None) / float(half_life_days))
    s = w.sum()
    return w / s if s > 0 else w

@dataclass
class FitResult:
    x: np.ndarray
    success: bool
    fun: float
    message: str
    niter: int
    nit: int

class BivariatePoisson:
    """
    Bivariate Poisson with team attack/defense + home advantage for lambda1,2; global gamma for lambda3.
    log lambda1 = a_home - d_away + home + X @ beta_h
    log lambda2 = a_away - d_home       + X @ beta_a
    log lambda3 = gamma0   (constant)   [optionally could add X @ beta_c later]
    """

    def __init__(
        self,
        n_teams: int,
        home_idx: ArrayLike,
        away_idx: ArrayLike,
        hg: ArrayLike,
        ag: ArrayLike,
        match_dates: Optional[ArrayLike] = None,
        X: Optional[np.ndarray] = None,
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

    def _unpack(self, x: np.ndarray):
        T = self.n_teams
        ofs = 0
        a_free = x[ofs:ofs + (T-1)]; ofs += (T-1)
        d_free = x[ofs:ofs + (T-1)]; ofs += (T-1)
        home  = x[ofs]; ofs += 1
        gamma0 = x[ofs]; ofs += 1  # log lambda3 baseline
        betas = x[ofs:]
        if self.K > 0:
            assert betas.size == 2*self.K, "Parameter length mismatch for covariates"
            beta_h = betas[:self.K]
            beta_a = betas[self.K:]
        else:
            beta_h = np.zeros(0); beta_a = np.zeros(0)
        a_last = -np.sum(a_free)
        d_last = -np.sum(d_free)
        a = np.concatenate([a_free, [a_last]])
        d = np.concatenate([d_free, [d_last]])
        return a, d, home, gamma0, beta_h, beta_a

    def _rates(self, a, d, home, gamma0, beta_h, beta_a):
        xb_h = 0.0 if self.K == 0 else self.X @ beta_h
        xb_a = 0.0 if self.K == 0 else self.X @ beta_a
        lam1 = np.exp(a[self.home_idx] - d[self.away_idx] + home + xb_h)
        lam2 = np.exp(a[self.away_idx] - d[self.home_idx] + xb_a)
        lam3 = np.exp(gamma0) * np.ones_like(lam1)
        return lam1, lam2, lam3

    def _logpmf_pair(self, h: np.ndarray, a: np.ndarray, l1: np.ndarray, l2: np.ndarray, l3: np.ndarray) -> np.ndarray:
        """
        log P(H=h, A=a) for bivariate Poisson, vectorized over matches with match-specific (h,a,l1,l2,l3).
        Uses the summation over k = 0..min(h,a).
        """
        # we implement a per-match loop over k, but vectorized internally using broadcasting
        M = h.shape[0]
        out = np.empty(M, dtype=float)
        const = -(l1 + l2 + l3) + (h * np.log(l1 + 1e-12)) + (a * np.log(l2 + 1e-12))
        const -= (gammaln(h + 1.0) + gammaln(a + 1.0))
        for i in range(M):
            hi, ai = int(h[i]), int(a[i])
            kmax = min(hi, ai)
            if kmax == 0:
                # only k=0 term
                out[i] = const[i]
            else:
                k = np.arange(kmax + 1)
                # term for sum_k (l3^k / k!) * (l1^-k) * (l2^-k) * (h choose k) * (a choose k) ???
                # Safer: direct formula on original PMF in log-space
                # log sum exp: log( sum_k exp( log_term_k ) )
                # log_term_k = k*log(l3) - gammaln(k+1) - k*log(l1) - k*log(l2) + gammaln(hi+1) - gammaln(hi-k+1) + gammaln(ai+1) - gammaln(ai-k+1) - [gammaln(hi+1)+gammaln(ai+1)] cancels since already in const.
                log_terms = (
                    k * np.log(l3[i] + 1e-12)
                    - gammaln(k + 1.0)
                    - k * np.log(l1[i] + 1e-12)
                    - k * np.log(l2[i] + 1e-12)
                    + (gammaln(hi + 1.0) - gammaln(hi - k + 1.0))
                    + (gammaln(ai + 1.0) - gammaln(ai - k + 1.0))
                )
                # log-sum-exp
                m = np.max(log_terms)
                out[i] = const[i] + (m + np.log(np.sum(np.exp(log_terms - m))))
        return out

    def _objective(self, x: np.ndarray) -> float:
        a, d, home, gamma0, beta_h, beta_a = self._unpack(x)
        l1, l2, l3 = self._rates(a, d, home, gamma0, beta_h, beta_a)
        ll = self._logpmf_pair(self.hg, self.ag, l1, l2, l3)
        nll = -np.sum(self.w * ll)

        # L2 reg on free params + betas
        T = self.n_teams
        ofs = 0
        a_free = x[ofs:ofs + (T-1)]; ofs += (T-1)
        d_free = x[ofs:ofs + (T-1)]; ofs += (T-1)
        _home = x[ofs]; ofs += 1
        _gamma0 = x[ofs]; ofs += 1
        betas = x[ofs:]
        reg = self.reg_lambda * (np.dot(a_free, a_free) + np.dot(d_free, d_free) + np.dot(betas, betas) + _gamma0*_gamma0)
        return nll + reg

    def fit(self, x0: Optional[np.ndarray] = None) -> FitResult:
        T = self.n_teams
        if x0 is None:
            x0 = np.zeros((2*(T-1) + 1 + 1 + 2*self.K,), dtype=float)
            x0[-(2*self.K + 2)] = 0.2  # home
            x0[-(2*self.K + 1)] = -1.0 # gamma0 small (lambda3 ~ 0.37)
        res = minimize(self._objective, x0, method="L-BFGS-B")
        return FitResult(x=res.x, success=res.success, fun=res.fun, message=res.message, niter=getattr(res, "niter", -1), nit=res.nit)

    @staticmethod
    def markets_from_rates(l1: float, l2: float, l3: float, max_goals: int = 10, goal_line: float = 2.5):
        # Build joint PMF from bivariate Poisson by summation
        G = max_goals
        pmf = np.zeros((G+1, G+1), dtype=float)
        from math import exp, log
        for h in range(G+1):
            for a in range(G+1):
                # compute pmf(h,a)
                s = 0.0
                kmax = min(h, a)
                for k in range(kmax+1):
                    term = (
                        (l1 ** (h - k)) * (l2 ** (a - k)) * (l3 ** k)
                        / (math_factorial(h - k) * math_factorial(a - k) * math_factorial(k))
                    )
                    s += term
                pmf[h, a] = np.exp(-(l1 + l2 + l3)) * s
        # normalize safeguard
        s = pmf.sum()
        if s > 0: pmf /= s

        # aggregate to markets (reuse same as DC utility)
        home = float(np.tril(pmf, k=-1).sum())
        away = float(np.triu(pmf, k=1).sum())
        draw = float(np.trace(pmf))
        over = under = btts = 0.0
        for i in range(G+1):
            for j in range(G+1):
                if i + j > goal_line: over += pmf[i, j]
                else: under += pmf[i, j]
                if i > 0 and j > 0: btts += pmf[i, j]
        return {
            "home": home, "draw": draw, "away": away,
            f"over_{goal_line}": over, f"under_{goal_line}": under, "btts": btts
        }

# helper to avoid importing math.factorial repeatedly
_fact_cache = {0:1, 1:1}
def math_factorial(n:int)->int:
    if n in _fact_cache: return _fact_cache[n]
    v = 1
    for i in range(2, n+1): v *= i
    _fact_cache[n]=v
    return v
