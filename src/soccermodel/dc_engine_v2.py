
"""
soccermodel.dc_engine_v2
------------------------
Drop-in "better engine" adapter that uses the new vectorized Dixon–Coles model
from `soccerprediction` under the hood, but keeps the import path in the
`soccermodel` namespace so your existing scripts don't have to change.

Usage (old code can switch to this without changing package names):
    from soccermodel.dc_engine_v2 import DixonColesV2Adapter, DCConfig
    model = DixonColesV2Adapter(DCConfig())
    model.fit(df_matches)  # expects columns: date, home, away, home_goals, away_goals
    out = model.predict_fixture("Team A", "Team B")
    out["markets"] -> dict of 1X2, Over/Under 2.5, BTTS
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

from soccerprediction.models.dixon_coles import DixonColes
from soccerprediction.evaluation.calibration import apply_calibrator

@dataclass
class DCConfig:
    reg_lambda: float = 1e-3
    half_life_days: float = 180.0
    max_goals: int = 10
    goal_line: float = 2.5
    calibrator: Optional[object] = None  # set to isotonic/platt object if available

class DixonColesV2Adapter:
    def __init__(self, cfg: DCConfig = DCConfig()):
        self.cfg = cfg
        self._model: Optional[DixonColes] = None
        self._fitx = None
        self._team_to_idx = {}
        self._idx_to_team = {}

    def fit(self, df_matches: pd.DataFrame,
            home_col="home", away_col="away",
            hg_col="home_goals", ag_col="away_goals",
            date_col="date",
            X: Optional[np.ndarray] = None):
        teams = pd.Index(pd.unique(pd.concat([df_matches[home_col], df_matches[away_col]])))
        self._team_to_idx = {t:i for i,t in enumerate(teams)}
        self._idx_to_team = dict(enumerate(teams))

        home_idx = df_matches[home_col].map(self._team_to_idx).to_numpy()
        away_idx = df_matches[away_col].map(self._team_to_idx).to_numpy()
        hg = df_matches[hg_col].to_numpy(int)
        ag = df_matches[ag_col].to_numpy(int)
        dates = pd.to_datetime(df_matches[date_col]).to_numpy("datetime64[D]")

        self._model = DixonColes(
            n_teams=len(teams),
            home_idx=home_idx, away_idx=away_idx,
            hg=hg, ag=ag,
            match_dates=dates,
            X=X,
            reg_lambda=self.cfg.reg_lambda,
            half_life_days=self.cfg.half_life_days,
        )
        self._fitx = self._model.fit()
        return self

    def _fixture_rates(self, home_team: str, away_team: str, xrow: Optional[np.ndarray]):
        a, d, home_adv, rho, beta_h, beta_a = self._model._unpack(self._fitx.x)  # intentionally use model internals
        hi = self._team_to_idx[home_team]
        ai = self._team_to_idx[away_team]
        xb_h = 0.0 if xrow is None or beta_h.size == 0 else float(xrow @ beta_h)
        xb_a = 0.0 if xrow is None or beta_a.size == 0 else float(xrow @ beta_a)
        lam_h = float(np.exp(a[hi] - d[ai] + home_adv + xb_h))
        lam_a = float(np.exp(a[ai] - d[hi] + xb_a))
        return lam_h, lam_a, float(rho)

    def predict_fixture(self, home_team: str, away_team: str, xrow: Optional[np.ndarray] = None):
        lam_h, lam_a, rho = self._fixture_rates(home_team, away_team, xrow)
        pmf = DixonColes.scoreline_pmf(lam_h, lam_a, rho, max_goals=self.cfg.max_goals)
        markets = DixonColes.markets_from_pmf(pmf, goal_line=self.cfg.goal_line)

        # optional probability calibration
        if self.cfg.calibrator is not None:
            keys = [k for k in markets.keys() if k.startswith("over_") or k in ("home","draw","away","btts")]
            arr = np.array([markets[k] for k in keys])
            arr_cal = apply_calibrator(self.cfg.calibrator, arr)
            for k, v in zip(keys, arr_cal):
                markets[k] = float(v)

        return {
            "lambdas": {"home": lam_h, "away": lam_a, "rho": rho},
            "markets": markets,
        }
