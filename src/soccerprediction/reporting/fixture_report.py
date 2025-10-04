
"""
Fixture report: one function that returns a dict with
- calibrated 1X2 / totals / BTTS from Dixon–Coles
- Poisson-based probabilities for total corners and total cards
- Trend percentages over last N matches for goals, corners, cards

This expects you have a "match-level" dataframe with one row per match, including:
  date, home, away, home_goals, away_goals, home_corners, away_corners, home_cards, away_cards
You can build it from soccerdata FBref/Football-Data and your own joins.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd

from soccerprediction.models.dixon_coles import DixonColes
from soccerprediction.markets.props import combined_total_lambda, poisson_sf

@dataclass
class ReportConfig:
    goal_lines: Tuple[float, ...] = (1.5, 2.5, 3.5)
    corner_lines: Tuple[float, ...] = (7.5, 8.5, 9.5)
    card_lines: Tuple[float, ...] = (3.5, 4.5)
    half_life_days: float = 180.0
    last_n_trend: int = 10
    max_goals_pmf: int = 10

def _select_last(df: pd.DataFrame, team: str, n: int, date_col="date") -> pd.DataFrame:
    d = df[(df["home"] == team) | (df["away"] == team)].copy()
    d = d.sort_values(date_col).tail(n)
    return d

def _pct_over(series: pd.Series, line: float) -> float:
    if series.empty: return float("nan")
    if float(line).is_integer():
        # >= line+1 (e.g., over 2 means >=3)
        k = int(line) + 1
        return float((series >= k).mean())
    else:
        # e.g., over 2.5
        return float((series > line).mean())

def _trend_block(df: pd.DataFrame, home: str, away: str, cfg: ReportConfig) -> Dict[str, Any]:
    # union of last_n matches for both teams
    dh = _select_last(df, home, cfg.last_n_trend)
    da = _select_last(df, away, cfg.last_n_trend)
    d = pd.concat([dh, da]).drop_duplicates(subset=["date","home","away"]).sort_values("date").tail(cfg.last_n_trend)

    out = {"samples": int(len(d))}
    goals = d["home_goals"] + d["away_goals"]
    if "home_corners" in d and "away_corners" in d:
        corners = d["home_corners"] + d["away_corners"]
    else:
        corners = None
    if "home_cards" in d and "away_cards" in d:
        cards = d["home_cards"] + d["away_cards"]
    else:
        cards = None

    out["goals"] = {f"over_{gl}": _pct_over(goals, gl) for gl in cfg.goal_lines}
    if corners is not None:
        out["corners"] = {f"over_{cl}": _pct_over(corners, cl) for cl in cfg.corner_lines}
    if cards is not None:
        out["cards"] = {f"over_{cl}": _pct_over(cards, cl) for cl in cfg.card_lines}

    # team-specific corner trends
    if corners is not None:
        d_home = _select_last(df, home, cfg.last_n_trend)
        d_away = _select_last(df, away, cfg.last_n_trend)
        tA = pd.Series([r.home_corners if r.home == home else r.away_corners for r in d_home.itertuples(index=False)])
        tB = pd.Series([r.home_corners if r.home == away else r.away_corners for r in d_away.itertuples(index=False)])
        out["team_corners"] = {
            home: {f"over_{x}": _pct_over(tA, x) for x in (3.5, 4.5, 5.5)},
            away: {f"over_{x}": _pct_over(tB, x) for x in (3.5, 4.5, 5.5)},
        }
    return out

def build_fixture_report(df_matches: pd.DataFrame, home: str, away: str, cfg: ReportConfig = ReportConfig()) -> Dict[str, Any]:
    # --- 1) Fit DC to goals (uses whole df_matches; you can limit by league/season)
    teams = pd.Index(pd.unique(pd.concat([df_matches["home"], df_matches["away"]])))
    team_to_idx = {t:i for i,t in enumerate(teams)}
    home_idx = df_matches["home"].map(team_to_idx).to_numpy()
    away_idx = df_matches["away"].map(team_to_idx).to_numpy()
    hg = df_matches["home_goals"].to_numpy(int)
    ag = df_matches["away_goals"].to_numpy(int)
    dates = pd.to_datetime(df_matches["date"]).to_numpy("datetime64[D]")

    dc = DixonColes(n_teams=len(teams), home_idx=home_idx, away_idx=away_idx, hg=hg, ag=ag,
                    match_dates=dates, X=None, half_life_days=cfg.half_life_days)
    fres = dc.fit()

    # Extract lambda for the requested fixture
    a, d, home_adv, rho, beta_h, beta_a = dc._unpack(fres.x)
    hi = team_to_idx[home]; ai = team_to_idx[away]
    lam_h = float(np.exp(a[hi] - d[ai] + home_adv))
    lam_a = float(np.exp(a[ai] - d[hi]))

    pmf = DixonColes.scoreline_pmf(lam_h, lam_a, rho, max_goals=cfg.max_goals_pmf)
    # 1X2 + BTTS
    markets = DixonColes.markets_from_pmf(pmf, goal_line=2.5)

    # add extra goal lines
    for gl in cfg.goal_lines:
        over_key = f"over_{gl}"
        if over_key not in markets:
            # compute from PMF
            G = pmf.shape[0]-1
            over = 0.0
            for i in range(G+1):
                for j in range(G+1):
                    if i + j > gl: over += pmf[i, j]
            markets[over_key] = float(over)
            markets[f"under_{gl}"] = 1.0 - float(over)

    # --- 2) Corners & Cards via Poisson totals (if columns exist)
    extra = {}
    if {"home_corners","away_corners"}.issubset(df_matches.columns):
        lam_corners = combined_total_lambda(
            df_matches.rename(columns={
                "home_corners": "corners_home", "away_corners": "corners_away"
            }),
            home, away, col_for="corners", col_against="corners",
            half_life_days=cfg.half_life_days, last_n=20
        )
        extra["corners_lambda"] = lam_corners
        extra["corners"] = {f"over_{cl}": float(poisson_sf(cl, lam_corners)) for cl in cfg.corner_lines}

    if {"home_cards","away_cards"}.issubset(df_matches.columns):
        lam_cards = combined_total_lambda(
            df_matches.rename(columns={
                "home_cards": "cards_home", "away_cards": "cards_away"
            }),
            home, away, col_for="cards", col_against="cards",
            half_life_days=cfg.half_life_days, last_n=20
        )
        extra["cards_lambda"] = lam_cards
        extra["cards"] = {f"over_{cl}": float(poisson_sf(cl, lam_cards)) for cl in cfg.card_lines}

    # --- 3) Trends
    trends = _trend_block(df_matches, home, away, cfg)

    return {
        "fixture": {"home": home, "away": away},
        "goals": {"lambda_home": lam_h, "lambda_away": lam_a, "rho": float(rho)},
        "markets": markets,
        "props": extra,
        "trends": trends,
    }
