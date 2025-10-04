
"""
WhoScored client wrapper for injuries/suspensions and event streams.
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
class WhoScoredClient:
    leagues: Iterable[str]
    seasons: Iterable[str]
    no_cache: bool = False
    no_store: bool = False
    headless: bool = True

    def _client(self):
        if sd is None:
            raise ImportError("soccerdata must be installed")
        return sd.WhoScored(self.leagues, self.seasons, no_cache=self.no_cache, no_store=self.no_store, headless=self.headless)

    def read_missing_players(self, match_id: Optional[int] = None) -> pd.DataFrame:
        return self._client().read_missing_players(match_id=match_id)

    def read_events(self, match_id: Optional[int] = None) -> pd.DataFrame:
        return self._client().read_events(match_id=match_id)
