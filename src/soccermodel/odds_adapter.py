from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import json

@dataclass
class OddsSnapshot:
    market: str
    prices: Dict[str, float]
    timestamp: str

def load_odds_from_json(path: str) -> Dict[str, OddsSnapshot]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    out = {}
    markets = payload.get("markets", {})
    for mkt, data in markets.items():
        ts = data.get("ts") or data.get("timestamp") or ""
        prices = {k.lower(): float(v) for k,v in data.items() if k not in ("ts","timestamp")}
        out[mkt] = OddsSnapshot(market=mkt, prices=prices, timestamp=ts)
    return out

def kelly_fraction(p: float, o: float) -> float:
    f = (p*o - 1.0) / (o - 1.0)
    return max(0.0, f)

def american_from_decimal(d: float) -> int:
    return int(round((d-1.0)*100)) if d>=2.0 else int(round(-100.0/(d-1.0)))
