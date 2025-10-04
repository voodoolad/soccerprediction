
"""
Example end-to-end pipeline:
1) Ingest data via DataHub (FBref, Understat, WhoScored)
2) Build features (Understat NP-xG, PPDA, set-piece share placeholder)
3) Fit Dixon–Coles with covariates; also fit Bivariate Poisson baseline
4) Build scoreline PMF, aggregate to markets
5) Remove vig (Shin) from odds and compute Kelly stake

This is a template script. Adjust team/league IDs and feature joins for your environment.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from soccerprediction.datahub import DataHub
from soccerprediction.models.dixon_coles import DixonColes
from soccerprediction.models.bivariate_poisson import BivariatePoisson
from soccerprediction.odds.vig import remove_vig_shin, fair_odds, kelly_fraction

LEAGUES = ['ENG-Premier League']
SEASONS = ['2023-2024']

def main():
    hub = DataHub(LEAGUES, SEASONS)
    fb_logs = hub.fbref_team_logs()
    under_team = hub.understat_team_match()

    # Minimal match dataset (you should derive this from your fixture join)
    # Expect columns: date, home_team, away_team, home_goals, away_goals
    if 'schedule' in fb_logs and not fb_logs['schedule'].empty:
        sched = fb_logs['schedule'].rename(columns={'home_team':'home', 'away_team':'away', 'score_home':'home_goals', 'score_away':'away_goals'})  # adjust based on real column names
    else:
        raise RuntimeError("FBref schedule not available - adjust this example to your schema")

    # Build team index map
    teams = pd.Index(pd.unique(pd.concat([sched['home'], sched['away']])), name='team')
    team_to_idx = {t:i for i,t in enumerate(teams)}

    # Prepare arrays for DC fit
    home_idx = sched['home'].map(team_to_idx).to_numpy()
    away_idx = sched['away'].map(team_to_idx).to_numpy()
    hg = sched['home_goals'].to_numpy(dtype=int)
    ag = sched['away_goals'].to_numpy(dtype=int)
    dates = pd.to_datetime(sched['date']).to_numpy(dtype='datetime64[D]')

    # Example covariate: Understat non-penalty xG differential last N matches (placeholder join)
    X = np.zeros((len(sched), 1), dtype=float)  # fill with engineered feature

    dc = DixonColes(n_teams=len(teams), home_idx=home_idx, away_idx=away_idx, hg=hg, ag=ag, match_dates=dates, X=X)
    res = dc.fit()
    print("DC fit:", res.success, res.fun)

    # Example: compute PMF for last match and aggregate
    lam_h, lam_a = 1.3, 1.1  # in practice, compute from fitted params for a specific fixture
    pmf = DixonColes.scoreline_pmf(lam_h, lam_a, rho=-0.05, max_goals=10)
    markets = DixonColes.markets_from_pmf(pmf, goal_line=2.5)
    print("Markets example:", markets)

    # Odds example
    odds_1x2 = np.array([1.95, 3.5, 3.9])
    fair_p = remove_vig_shin(odds_1x2)
    min_odds = fair_odds(fair_p[0])
    stake_frac = kelly_fraction(fair_p[0], odds_1x2[0], frac=0.33)
    print("Fair probs:", fair_p, "min_odds_home", min_odds, "kelly_0.33", stake_frac)

if __name__ == "__main__":
    main()
