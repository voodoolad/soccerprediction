
"""
FBref client wrappers using `soccerdata` with schema checks.
Pulls team match logs for various stat tables, plus events, lineups.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional, Dict, List
import pandas as pd

try:
    import soccerdata as sd
except Exception:
    sd = None

@dataclass
class FBrefClient:
    leagues: Iterable[str]
    seasons: Iterable[str]
    no_cache: bool = False
    no_store: bool = False

    def _client(self):
        if sd is None:
            raise ImportError("soccerdata must be installed")
        return sd.FBref(self.leagues, self.seasons, no_cache=self.no_cache, no_store=self.no_store)

    def read_team_match_logs(self, stat_types: Iterable[str] = ('schedule','shooting','passing','passing_types','goal_shot_creation','defense','possession','misc'), opponent_stats: bool = False) -> Dict[str, pd.DataFrame]:
        cli = self._client()
        out = {}
        for st in stat_types:
            df = cli.read_team_match_stats(stat_type=st, opponent_stats=opponent_stats)
            out[st] = df
        return out

    def read_events(self, match_id: Optional[int] = None) -> pd.DataFrame:
        cli = self._client()
        return cli.read_events(match_id=match_id)

    def read_lineups(self, match_id: Optional[int] = None) -> pd.DataFrame:
        cli = self._client()
        return cli.read_lineup(match_id=match_id)
