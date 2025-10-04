
"""
DataHub: thin orchestrator over soccerdata sources with caching and canonical IDs.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional, Dict
from pathlib import Path
import pandas as pd

from .ingest.fbref import FBrefClient
from .ingest.understat import UnderstatClient
from .ingest.whoscored import WhoScoredClient
from .utils.health_checks import assert_required_columns, warn_if_stale
from .utils.names import normalize_name

@dataclass
class DataHub:
    leagues: Iterable[str]
    seasons: Iterable[str]
    cache_dir: Path = Path("./out/cache")

    def __post_init__(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.fbref = FBrefClient(self.leagues, self.seasons)
        self.understat = UnderstatClient(self.leagues, self.seasons)
        self.whoscored = WhoScoredClient(self.leagues, self.seasons)

    # ---- FBref ----
    def fbref_team_logs(self) -> Dict[str, pd.DataFrame]:
        logs = self.fbref.read_team_match_logs()
        for k, df in logs.items():
            if 'date' in df.columns:
                warn_if_stale(df, 'date', context=f"FBref[{k}]")
        return logs

    # ---- Understat ----
    def understat_team_match(self) -> pd.DataFrame:
        df = self.understat.read_team_match_stats()
        warn_if_stale(df, 'date', context="Understat team match")
        return df

    # ---- WhoScored ----
    def injuries(self) -> pd.DataFrame:
        return self.whoscored.read_missing_players()

    # ---- Canonicalization helpers ----
    @staticmethod
    def add_canonical_team(df: pd.DataFrame, team_col: str = "team") -> pd.DataFrame:
        df = df.copy()
        df['team_key'] = df[team_col].map(normalize_name)
        return df

    # ---- Simple cache ----
    def save_parquet(self, df: pd.DataFrame, name: str) -> Path:
        path = self.cache_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        return path
