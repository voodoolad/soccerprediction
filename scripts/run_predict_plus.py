#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run a one-off fixture report (scrape -> model -> extended markdown).

What you get per fixture:
- Goals (Dixon–Coles v2): 1X2, BTTS, Over/Under 1.5/2.5/3.5
- Corners totals: Over 7.5/8.5/9.5 (if Football-Data has HC/AC)
- Cards totals: Over 3.5/4.5 (if Football-Data has HY/HR/AY/AR)
- Trend percentages (last N): goals/corners/cards + per-team corners

No CSV needed. You keep your old flags:
  --league "ENG-Premier League" --season 2025
  --fixtures "Chelsea vs Liverpool|2025-10-05 03:30:00"
  --timezone "Australia/Sydney" --out out\\chelsea_liverpool.md

Notes:
- PPDA is intentionally hidden until a reliable source is wired to avoid 'nan'.
- If Football-Data corners/cards are missing for that league/season, those sections are omitted.

This file replaces the previous run_predict_plus.py that printed the "Picks by Tier"
layout and mixed old engine imports. (See your current output markdown for reference.)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import difflib
import numpy as np
import pandas as pd

# Ensure local 'src' is importable
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

# Your existing hub (kept)
from soccermodel.data_sources import DataHub, SourceConfig

# New report/engine (kept minimal; the report builds/fits DC v2 internally)
from soccerprediction.reporting.fixture_report import build_fixture_report, ReportConfig
from soccerprediction.utils.names import normalize_name

# -------------------------
# CLI parsing and helpers
# -------------------------

def parse_fixture(s: str):
    """
    Expected format per your old CLI: "Home Team vs Away Team|YYYY-MM-DD HH:MM:SS"
    The date/time is optional for probabilities; it's used only to disambiguate fixtures.
    """
    if "|" in s:
        teams, ts = s.split("|", 1)
        kickoff = ts.strip()
    else:
        teams, kickoff = s, ""
    if " vs " not in teams:
        raise ValueError(f"Fixture must look like 'A vs B|...': got {s}")
    home, away = teams.split(" vs ", 1)
    return {"home": home.strip(), "away": away.strip(), "kickoff": kickoff}

def _flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            " ".join([str(x) for x in tup if x and x != "Unnamed: 0"]).strip()
            for tup in df.columns.to_list()
        ]
    else:
        df.columns = [str(c) for c in df.columns]
    return df

def _norm(df: pd.DataFrame) -> pd.DataFrame:
    df = _flatten_cols(df)
    # Normalize a 'date' column if present
    if "date" in (c.lower() for c in df.columns):
        # leave as-is
        pass
    else:
        # Try to infer a date-like col
        for c in df.columns:
            lc = c.lower()
            if any(tok in lc for tok in ("date", "time", "kickoff", "utc", "start", "datetime")):
                try:
                    dt = pd.to_datetime(df[c], errors="coerce", utc=True)
                    df["date"] = dt.dt.tz_convert(None)
                except Exception:
                    pass
                break
    return df

def _pick(cols_map: dict[str, str], *alts: str) -> str | None:
    for a in alts:
        if a in cols_map:
            return cols_map[a]
    return None

def _canonicalize(name: str, candidates: list[str]) -> str:
    """
    Return the best-matching team name from candidates using normalized names and difflib.
    """
    key = normalize_name(name)
    table = {normalize_name(c): c for c in candidates}
    if key in table:
        return table[key]
    # try fuzzy
    match = difflib.get_close_matches(key, list(table.keys()), n=1, cutoff=0.75)
    return table[match[0]] if match else name  # fall back to original

# -------------------------
# Build a match-level DataFrame from FBref schedule
# -------------------------

def build_matches_df(hub: DataHub) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      date, home, away, home_goals, away_goals
    using FBref schedule for the configured league/season.
    """
    # Try a dedicated schedule reader if available, else fall back to "team_match_stats(schedule)"
    try:
        fb = hub.fbref.read_schedule()
    except Exception:
        fb = hub.fbref.read_team_match_stats(stat_type="schedule", opponent_stats=False)

    fb = _norm(pd.DataFrame(fb))
    # Lower-case lookups
    cmap = {c.lower(): c for c in fb.columns}

    # Two possible formats from soccerdata:
    # 1) One row per match with home_team, away_team, score_home, score_away, date
    # 2) Team logs with 'team', 'opponent', 'home/away' or 'venue', goals for/against, date
    home_c = _pick(cmap, "home_team", "home")
    away_c = _pick(cmap, "away_team", "away")
    sh_c   = _pick(cmap, "score_home", "home_goals", "gf", "goals_for", "goals for", "scored")
    sa_c   = _pick(cmap, "score_away", "away_goals", "ga", "goals_against", "goals against", "conceded")
    date_c = _pick(cmap, "date", "datetime", "kickoff")

    if home_c and away_c and sh_c and sa_c:
        df = fb[[home_c, away_c, sh_c, sa_c] + ([date_c] if date_c else [])].copy()
        df = df.rename(columns={
            home_c: "home",
            away_c: "away",
            sh_c: "home_goals",
            sa_c: "away_goals",
        })
    else:
        # Team logs -> keep only home rows to reconstruct match list
        team_c  = _pick(cmap, "team", "squad")
        opp_c   = _pick(cmap, "opponent", "opp")
        venue_c = _pick(cmap, "home/away", "homeaway", "home_away", "venue")
        gf_c    = _pick(cmap, "gf", "goals_for", "goals for", "scored")
        ga_c    = _pick(cmap, "ga", "goals_against", "goals against", "conceded")
        if not all([team_c, opp_c, venue_c, gf_c, ga_c]):
            raise RuntimeError("Could not map FBref schedule columns.")
        temp = fb[[team_c, opp_c, venue_c, gf_c, ga_c] + ([date_c] if date_c else [])].copy()
        temp["_is_home"] = temp[venue_c].astype(str).str.lower().str.startswith("home")
        temp = temp[temp["_is_home"]]
        df = temp.rename(columns={
            team_c: "home",
            opp_c: "away",
            gf_c: "home_goals",
            ga_c: "away_goals",
        })

    # Parse and clean types
    for g in ("home_goals", "away_goals"):
        df[g] = pd.to_numeric(df[g], errors="coerce").astype("Int64")
    if "date" not in df.columns and date_c:
        df["date"] = pd.to_datetime(fb[date_c], errors="coerce")
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["date"] = df["date"].dt.tz_localize(None)
    df = df.dropna(subset=["home", "away", "home_goals", "away_goals"]).copy()
    return df

# -------------------------
# Optional: augment with corners/cards from Football-Data
# -------------------------

def augment_with_footballdata(df_matches: pd.DataFrame, league: str, season: int) -> pd.DataFrame:
    """
    Adds columns when available:
      home_corners, away_corners, home_cards, away_cards
    using Football-Data (HC, AC, HY, AY, HR, AR).
    Best-effort join on (date, normalized home/away).
    """
    try:
        import soccerdata as sd
        fd = sd.FootballData(leagues=[league], seasons=[season])
        df_fd = fd.read_results()
    except Exception:
        return df_matches  # silently skip if missing

    df_fd = _norm(pd.DataFrame(df_fd))
    cmap = {c.lower(): c for c in df_fd.columns}

    home_c = _pick(cmap, "home_team", "hometeam", "home")
    away_c = _pick(cmap, "away_team", "awayteam", "away")
    date_c = _pick(cmap, "date", "datetime")
    hc_c   = _pick(cmap, "hc")
    ac_c   = _pick(cmap, "ac")
    hy_c   = _pick(cmap, "hy")
    ay_c   = _pick(cmap, "ay")
    hr_c   = _pick(cmap, "hr")
    ar_c   = _pick(cmap, "ar")

    if not (home_c and away_c and date_c):
        return df_matches

    # Normalize join keys
    def _keyframe(df, home_col, away_col, date_col):
        out = df.copy()
        out["_home_key"] = out[home_col].astype(str).map(normalize_name)
        out["_away_key"] = out[away_col].astype(str).map(normalize_name)
        out["_date_key"] = pd.to_datetime(out[date_col], errors="coerce").dt.tz_localize(None).dt.floor("D")
        return out

    left  = _keyframe(df_matches, "home", "away", "date")
    right = _keyframe(df_fd, home_c, away_c, date_c)

    merged = left.merge(
        right[["_home_key", "_away_key", "_date_key"] + [x for x in [hc_c, ac_c, hy_c, ay_c, hr_c, ar_c] if x]],
        on=["_home_key", "_away_key", "_date_key"],
        how="left",
        suffixes=("", "_fd"),
    )

    # Map corners
    if hc_c and ac_c:
        merged["home_corners"] = pd.to_numeric(merged[hc_c], errors="coerce")
        merged["away_corners"] = pd.to_numeric(merged[ac_c], errors="coerce")
    # Map cards (simple sum of yellows + reds)
    if hy_c and ay_c:
        merged["home_cards"] = pd.to_numeric(merged[hy_c], errors="coerce").fillna(0)
        merged["away_cards"] = pd.to_numeric(merged[ay_c], errors="coerce").fillna(0)
        if hr_c:
            merged["home_cards"] = merged["home_cards"] + pd.to_numeric(merged[hr_c], errors="coerce").fillna(0)
        if ar_c:
            merged["away_cards"] = merged["away_cards"] + pd.to_numeric(merged[ar_c], errors="coerce").fillna(0)

    # Clean and drop join keys
    keep = ["date", "home", "away", "home_goals", "away_goals", "home_corners", "away_corners", "home_cards", "away_cards"]
    for k in keep:
        if k not in merged.columns:
            merged[k] = merged.get(k)
    result = merged[keep].copy()
    return result

# -------------------------
# Rendering (extended)
# -------------------------

def render_extended(report: dict, last_n: int) -> str:
    f = report["fixture"]
    g = report["goals"]
    mk = report["markets"]

    lines: list[str] = []
    lines.append(f"# {f['home']} vs {f['away']}")
    lines.append("")
    lines.append(f"**Goals λ** — home: {g['lambda_home']:.2f}, away: {g['lambda_away']:.2f}, rho: {g['rho']:.2f}")
    lines.append("")
    lines.append("## Model probabilities")
    for k in ("home", "draw", "away", "btts", "over_1.5", "under_1.5", "over_2.5", "under_2.5", "over_3.5", "under_3.5"):
        if k in mk:
            label = k.replace("_", " ").upper()
            lines.append(f"- **{label}**: {mk[k]:.3f}")

    # Props (corners/cards) if available
    props = report.get("props", {})
    if "corners" in props:
        lines.append("")
        lines.append("## Corners (Poisson totals)")
        lines.append(f"- λ_total ≈ {props.get('corners_lambda', float('nan')):.2f}")
        for k, v in props["corners"].items():
            lines.append(f"  - **{k.replace('_',' ').title()}**: {v:.3f}")
    if "cards" in props:
        lines.append("")
        lines.append("## Cards (Poisson totals)")
        lines.append(f"- λ_total ≈ {props.get('cards_lambda', float('nan')):.2f}")
        for k, v in props["cards"].items():
            lines.append(f"  - **{k.replace('_',' ').title()}**: {v:.3f}")

    # Trends
    tr = report.get("trends", {"samples": last_n})
    lines.append("")
    lines.append(f"## Trends (last {tr.get('samples', last_n)} matches)")
    if "goals" in tr:
        lines.append("- **Goals**:")
        for k, v in tr["goals"].items():
            lines.append(f"  - {k.replace('_',' ').title()}: {v:.0%}")
    if "corners" in tr:
        lines.append("- **Corners**:")
        for k, v in tr["corners"].items():
            lines.append(f"  - {k.replace('_',' ').title()}: {v:.0%}")
    if "cards" in tr:
        lines.append("- **Cards**:")
        for k, v in tr["cards"].items():
            lines.append(f"  - {k.replace('_',' ').title()}: {v:.0%}")
    if "team_corners" in tr:
        lines.append("- **Team corners**:")
        for team, block in tr["team_corners"].items():
            s = ", ".join([f"{kk}={vv:.0%}" for kk, vv in block.items()])
            lines.append(f"  - {team}: {s}")

    return "\n".join(lines)

# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", required=True)
    ap.add_argument("--season", required=True, type=int)
    ap.add_argument("--fixtures", required=True, help='Semicolon-separated: "Home vs Away|YYYY-MM-DD HH:MM:SS; ..."')
    ap.add_argument("--timezone", default="Australia/Sydney")  # kept for compatibility (not used by model)
    ap.add_argument("--out", default="out/preview_plus.md")
    ap.add_argument("--last-n", type=int, default=10, help="Trend window size")
    args = ap.parse_args()

    # Build hub
    hub = DataHub(SourceConfig(leagues=[args.league], seasons=[args.season]),
                  enable_whoscored=False, enable_sofascore=False)

    # Build matches from FBref schedule, then augment with corners/cards if we can
    matches_df = build_matches_df(hub)
    matches_df = augment_with_footballdata(matches_df, args.league, args.season)

    # Prepare output
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    blocks: list[str] = []

    # Candidate names (to canonicalize 'Liverpool' vs 'Liverpool FC', etc.)
    candidates = sorted(set(matches_df["home"]).union(matches_df["away"]))

    for part in args.fixtures.split(";"):
        fx = parse_fixture(part.strip())
        home = _canonicalize(fx["home"], candidates)
        away = _canonicalize(fx["away"], candidates)

        # Build the full report (fits DC v2 internally; uses recency weighting)
        rep_cfg = ReportConfig(
            last_n_trend=args.last_n,
            goal_lines=(1.5, 2.5, 3.5),
            corner_lines=(7.5, 8.5, 9.5),
            card_lines=(3.5, 4.5),
        )
        report = build_fixture_report(matches_df, home, away, rep_cfg)

        # Render extended markdown (no PPDA / no legacy "Picks by Tier")
        md = render_extended(report, args.last_n)
        blocks.append(md)

    final_md = "\n\n---\n\n".join(blocks)
    out_path.write_text(final_md, encoding="utf-8")
    print(f"✅ Wrote extended preview to: {out_path}")

    # Also print to stdout for quick view
    try:
        print("\n" + final_md + "\n")
    except Exception:
        pass

if __name__ == "__main__":
    main()
