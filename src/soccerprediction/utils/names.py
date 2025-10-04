
"""
Team/league name canonicalization and fuzzy matching to reconcile sources.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import re
import unicodedata
import difflib

def normalize_name(name: str) -> str:
    if name is None: return ""
    s = unicodedata.normalize("NFKD", name)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s\-\&\.]", "", s)
    s = re.sub(r"\s+", " ", s)
    replacements = {
        "fc ": "",
        " cf ": " ",
        "afc ": "",
        " athletic": "",
        " futbol club": "",
    }
    for k, v in replacements.items():
        s = s.replace(k, v)
    return s.strip()

def fuzzy_map(name: str, universe: Dict[str, str], cutoff: float = 0.85) -> Tuple[str, float]:
    """
    Map `name` to canonical key using difflib against normalized universe keys.
    `universe` maps canonical_key -> display_name
    """
    norm = normalize_name(name)
    keys = list(universe.keys())
    match = difflib.get_close_matches(norm, keys, n=1, cutoff=cutoff)
    if match:
        key = match[0]
        return key, difflib.SequenceMatcher(None, norm, key).ratio()
    return norm, 0.0
