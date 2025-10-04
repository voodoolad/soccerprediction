from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional

try:
    import soccerdata as sd
except Exception:
    sd = None

@dataclass
class SourceConfig:
    leagues: Iterable[str]
    seasons: Iterable[str] | Iterable[int]
    cache_dir: Optional[str] = None
    no_cache: bool = False
    no_store: bool = False

class DataHub:
    def __init__(self, cfg: SourceConfig, enable_whoscored: bool = False, enable_sofascore: bool = True):
        if sd is None:
            raise RuntimeError("soccerdata is not installed.")
        self.cfg = cfg
        common = dict(no_cache=cfg.no_cache, no_store=cfg.no_store)
        if cfg.cache_dir:
            common["data_dir"] = cfg.cache_dir

        self.fbref      = sd.FBref      (leagues=list(cfg.leagues), seasons=list(cfg.seasons), **common)
        self.understat  = sd.Understat  (leagues=list(cfg.leagues), seasons=list(cfg.seasons), **common)
        self.espn       = sd.ESPN       (leagues=list(cfg.leagues), seasons=list(cfg.seasons), **common)
        self.fotmob     = sd.FotMob     (leagues=list(cfg.leagues), seasons=list(cfg.seasons), **common)
        self.matchhist  = sd.MatchHistory(leagues=list(cfg.leagues), seasons=list(cfg.seasons), **common)
        self.whoscored  = sd.WhoScored  (leagues=list(cfg.leagues), seasons=list(cfg.seasons), **common) if enable_whoscored else None
        self.sofascore  = sd.Sofascore  (leagues=list(cfg.leagues), seasons=list(cfg.seasons), **common) if enable_sofascore else None

        if cfg.cache_dir:
            self.clubelo = sd.ClubElo(data_dir=cfg.cache_dir, no_cache=cfg.no_cache, no_store=cfg.no_store)
        else:
            self.clubelo = sd.ClubElo(no_cache=cfg.no_cache, no_store=cfg.no_store)
