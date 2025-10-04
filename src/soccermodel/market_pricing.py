from __future__ import annotations
def fair_odds(p: float) -> float: return 1.0/max(min(p,0.999999), 1e-6)
def american_from_decimal(d: float) -> int: return int(round((d-1)*100)) if d>=2 else int(round(-100/(d-1)))
def ev_percent(p: float, o: float) -> float: return 100.0*(p*(o-1)-(1-p))
def min_acceptable_odds(p: float, tau: float=0.03, safest_floor: bool=False) -> float:
    o=(1+tau)/max(p,1e-6); return max(o,1.08) if safest_floor else o
