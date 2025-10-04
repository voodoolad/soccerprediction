
"""
Ingestion health checks:
- row parity between fixtures and match logs
- required columns presence
- last updated recency
"""

from __future__ import annotations
from typing import Iterable, Dict
import pandas as pd
import numpy as np

def assert_required_columns(df: pd.DataFrame, cols: Iterable[str], context: str = "") -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise AssertionError(f"Missing columns {missing} in {context}")

def warn_if_stale(df: pd.DataFrame, date_col: str = "date", max_age_days: int = 14, context: str = "") -> None:
    if date_col not in df.columns: return
    last = pd.to_datetime(df[date_col]).max()
    age = (pd.Timestamp.utcnow().normalize() - last).days
    if age > max_age_days:
        print(f"[WARN] {context} appears stale: last date {last.date()} ({age} days old)")

def assert_join_one_to_one(left: pd.DataFrame, right: pd.DataFrame, on: Iterable[str], name_left: str, name_right: str) -> None:
    merged = left.merge(right, on=list(on), how="outer", indicator=True)
    if (merged['_merge'] != 'both').any():
        counts = merged['_merge'].value_counts().to_dict()
        raise AssertionError(f"Join between {name_left} and {name_right} not 1:1. _merge counts={counts}")
