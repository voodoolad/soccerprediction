
"""
Quick ingestion check script:
- Pull a tiny slice from FBref / Understat / WhoScored
- Verify schema presence and print row counts
"""

from __future__ import annotations
import pandas as pd
from soccerprediction.datahub import DataHub
from soccerprediction.utils.health_checks import assert_required_columns, warn_if_stale

LEAGUES = ['ENG-Premier League']
SEASONS = ['2023-2024']

REQ_FBREF_COLS = {
    'schedule': ['date', 'home_team', 'away_team', 'score_home', 'score_away'],
    'misc': ['date', 'team', 'opponent', 'CrdY', 'CrdR', 'Fouls'],
}

if __name__ == "__main__":
    hub = DataHub(LEAGUES, SEASONS)
    logs = hub.fbref_team_logs()
    for k, df in logs.items():
        print(f"[FBref/{k}] rows={len(df)}")
        if k in REQ_FBREF_COLS:
            try:
                assert_required_columns(df, REQ_FBREF_COLS[k], context=f"FBref[{k}]")
            except AssertionError as e:
                print("  ->", e)
        warn_if_stale(df, 'date', context=f"FBref[{k}]")

    ust = hub.understat_team_match()
    print("[Understat team_match] rows=", len(ust))
    warn_if_stale(ust, 'date', context="Understat team_match")

    try:
        inj = hub.injuries()
        print("[WhoScored injuries] rows=", len(inj))
    except Exception as e:
        print("[WhoScored injuries] skipped or requires Selenium/Chrome:", e)
