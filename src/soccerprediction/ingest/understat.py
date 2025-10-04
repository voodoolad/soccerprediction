
"""
Understat client wrapper.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional
import pandas as pd

try:
    import soccerdata as sd
except Exception:
    sd = None

@dataclass
class UnderstatClient:
    leagues: Iterable[str]
    seasons: Iterable[str]
    no_cache: bool = False
    no_store: bool = False

    def _client(self):
        if sd is None:
            raise ImportError("soccerdata must be installed")
        return sd.Understat(self.leagues, self.seasons, no_cache=self.no_cache, no_store=self.no_store)

    def read_team_match_stats(self) -> pd.DataFrame:
        return self._client().read_team_match_stats()

    def read_schedule(self, include_matches_without_data: bool = True) -> pd.DataFrame:
        return self._client().read_schedule(include_matches_without_data=include_matches_without_data)

    def read_shots(self, match_id: Optional[int] = None) -> pd.DataFrame:
        return self._client().read_shot_events(match_id=match_id)
