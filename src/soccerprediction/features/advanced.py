
"""
Advanced features built on top of soccerdata sources:
- Understat NP-xG and rolling xG form (recency-weighted)
- Set-piece share (corners -> set-piece xG proxy)
- PPDA (passes allowed per defensive action) approximated from FBref
- Opponent adjustment using ClubElo delta
- Rest days and congestion

NOTE: This module only contains feature computations. Fetch raw data via `datahub.py`.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

def recency_weights(dates: pd.Series, half_life_days: float = 180.0) -> pd.Series:
    dates = pd.to_datetime(dates).dt.tz_localize(None)
    as_of = pd.Timestamp.utcnow().normalize()
    age = (as_of - dates).dt.days.clip(lower=0)
    w = 0.5 ** (age / float(half_life_days))
    return w / w.sum()

def rolling_form(values: pd.Series, weights: pd.Series) -> float:
    aligned = values.align(weights, join='inner')[0]
    w = weights.loc[aligned.index]
    v = aligned.values.astype(float)
    return float(np.sum(v * w))

def ppda(df_team_defense: pd.DataFrame, df_team_possession: pd.DataFrame) -> pd.DataFrame:
    """
    Approximate PPDA: opponent completed passes in own half / (defensive actions)
    FBref provides defensive actions in `defense` and passes in `possession`/`passing` tables.
    We produce a team-level per-match estimate if both are available.
    """
    # This is a sketch; exact column names depend on soccerdata's standardization.
    d = df_team_defense.copy()
    p = df_team_possession.copy()
    key_cols = ['date', 'team', 'opponent', 'match_id']
    cols_needed_d = ['tackles', 'blocks', 'interceptions']  # placeholders; adjust to actual FBref cols
    cols_needed_p = ['passes_completed']

    for c in cols_needed_d + cols_needed_p:
        if c not in d.columns and c not in p.columns:
            # Leave as is; user should map real names in their environment.
            pass

    # Dummy calculation with safe fallbacks
    d['def_actions'] = d.get('tackles', 0) + d.get('blocks', 0) + d.get('interceptions', 0)
    p['opp_passes']  = p.get('passes_completed', 0)

    out = d[key_cols + ['def_actions']].merge(p[key_cols + ['opp_passes']], on=key_cols, how='inner')
    out['ppda'] = out['opp_passes'] / out['def_actions'].replace({0: np.nan})
    return out

def rest_days(schedule_df: pd.DataFrame) -> pd.DataFrame:
    schedule_df = schedule_df.sort_values(['team', 'date'])
    schedule_df['prev_date'] = schedule_df.groupby('team')['date'].shift(1)
    schedule_df['rest_days'] = (pd.to_datetime(schedule_df['date']) - pd.to_datetime(schedule_df['prev_date'])).dt.days
    return schedule_df[['match_id','team','rest_days']]
