
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run a one-off fixture report (scrape -> model -> extended markdown).

Keeps your original CLI flags:
  --league "ENG-Premier League" --season 2025
  --fixtures "Chelsea vs Liverpool|2025-10-05 03:30:00"
  --timezone "Australia/Sydney" --out out\\chelsea_liverpool.md

What this prints for EACH fixture:
- Goals (Dixon–Coles v2): 1X2, BTTS, Over/Under 1.5/2.5/3.5
- Corners totals: Over 7.5/8.5/9.5 (if Football-Data has HC/AC for that league/season)
- Cards totals: Over 3.5/4.5 (if Football-Data has HY/HR/AY/AR)
- Trend percentages (last N): goals/corners/cards + per-team corner trends

If FBref schedule mapping varies by league/season, this script now tries multiple
shapes and falls back to Football-Data schedule (and even FotMob where available).

No PPDA is rendered (to avoid 'nan' from partial sources). We can wire PPDA later.
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

# New report/engine (the report fits DC v2 internally)
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
    df = _flatten_cols(pd.DataFrame(df).copy())
    # Add a lowercase map
    df.columns = [str(c) for c in df.columns]
    return df

def _first(cols_map: dict[str,str], *cands: str) -> str | None:
    for c in cands:
        lc = c.lower()
        if lc in cols_map:
            return cols_map[lc]
    return None

def _canonicalize(name: str, candidates: list[str]) -> str:
    """Map a free-typed name to the closest candidate via normalized names + difflib."""
    key = normalize_name(name)
    table = {normalize_name(c): c for c in candidates}
    if key in table:
        return table[key]
    match = difflib.get_close_matches(key, list(table.keys()), n=1, cutoff=0.75)
    return table[match[0]] if match else name

# -------------------------
# Extract match table from various shapes
# -------------------------

def _extract_matches(raw: pd.DataFrame) -> pd.DataFrame | None:
    """
    Try to normalize a schedule/dataframe into:
      [date?, home, away, home_goals, away_goals]
    Returns None if the shape can't be recognized.
    """
    if raw is None:
        return None
    df = _norm(raw)
    cols = {str(c).lower(): c for c in df.columns}

    # Helper to parse date
    def _parse_date(col: str | None) -> pd.Series | None:
        if not col: return None
        try:
            dt = pd.to_datetime(df[col], errors="coerce")
            return dt.dt.tz_localize(None)
        except Exception:
            return None

    # Pattern A: explicit home/away + goal columns
    home = _first(cols, "home_team", "hometeam", "home team", "home")
    away = _first(cols, "away_team", "awayteam", "away team", "away")
    date = _first(cols, "date", "datetime", "kickoff", "start time", "start")
    gh   = _first(cols, "score_home", "home_goals", "home score", "goals_for", "goals for", "gf", "fthg")
    ga   = _first(cols, "score_away", "away_goals", "away score", "goals_against", "goals against", "ga", "ftag")
    if home and away:
        out = df[[home, away]].copy()
        out = out.rename(columns={home:"home", away:"away"})
        if gh and ga:
            out["home_goals"] = pd.to_numeric(df[gh], errors="coerce")
            out["away_goals"] = pd.to_numeric(df[ga], errors="coerce")
        else:
            # Pattern A2: a single "score" field like "2-1"
            score = _first(cols, "score", "result", "ft")
            if score:
                s = df[score].astype(str).str.replace(r"[^0-9\-:–]", "", regex=True)
                parts = s.str.extract(r"(?P<h>\d+)[-:–](?P<a>\d+)")
                out["home_goals"] = pd.to_numeric(parts["h"], errors="coerce")
                out["away_goals"] = pd.to_numeric(parts["a"], errors="coerce")
            else:
                # Pattern A3: FBref style gf/ga columns under different names
                gh2 = _first(cols, "gf", "goals_for", "goals for", "scored")
                ga2 = _first(cols, "ga", "goals_against", "goals against", "conceded")
                if gh2 and ga2:
                    out["home_goals"] = pd.to_numeric(df[gh2], errors="coerce")
                    out["away_goals"] = pd.to_numeric(df[ga2], errors="coerce")
        if date:
            out["date"] = _parse_date(date)
        out = out.dropna(subset=["home","away","home_goals","away_goals"])
        if len(out):
            return out

    # Pattern B: team logs (team/opponent + venue + gf/ga); keep only home rows
    team  = _first(cols, "team", "squad")
    opp   = _first(cols, "opponent", "opp")
    venue = _first(cols, "home/away", "home_away", "homeaway", "venue")
    gf    = _first(cols, "gf", "goals_for", "goals for", "scored")
    ga    = _first(cols, "ga", "goals_against", "goals against", "conceded")
    date2 = _first(cols, "date", "datetime", "kickoff")
    if team and opp and venue and gf and ga:
        tmp = df[[team, opp, venue, gf, ga] + ([date2] if date2 else [])].copy()
        mask = (
            tmp[venue].astype(str).str[0].str.upper().isin(["H","1"])
            | tmp[venue].astype(str).str.lower().str.startswith("home")
        )
        tmp = tmp[mask]
        tmp = tmp.rename(columns={team:"home", opp:"away", gf:"home_goals", ga:"away_goals"})
        tmp["home_goals"] = pd.to_numeric(tmp["home_goals"], errors="coerce")
        tmp["away_goals"] = pd.to_numeric(tmp["away_goals"], errors="coerce")
        if date2:
            tmp["date"] = _parse_date(date2)
        tmp = tmp.dropna(subset=["home","away","home_goals","away_goals"])
        if len(tmp):
            return tmp[["date","home","away","home_goals","away_goals"]]

    return None

# -------------------------
# Build a match-level DataFrame from multiple sources
# -------------------------

def build_matches_df(hub: DataHub, league: str, season) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      date, home, away, home_goals, away_goals
    Strategy:
      1) FBref: read_schedule()
      2) FBref: read_team_match_stats('schedule')
      3) FotMob: read_schedule()        (if available in your hub)
      4) Football-Data: read_results()  (fallback)
    Raises with visible column samples if nothing matches.
    """
    tried = []

    # 1) FBref schedule (if the adapter exposes it)
    try:
        if hasattr(hub.fbref, "read_schedule"):
            raw = hub.fbref.read_schedule()
            tried.append(("fbref.read_schedule", list(pd.DataFrame(raw).columns)))
            out = _extract_matches(raw)
            if out is not None and len(out):
                return out
    except Exception as e:
        tried.append(("fbref.read_schedule(ERR)", [str(e)]))

    # 2) FBref team match stats ('schedule' table)
    try:
        raw = hub.fbref.read_team_match_stats(stat_type="schedule", opponent_stats=False)
        tried.append(("fbref.read_team_match_stats(schedule)", list(pd.DataFrame(raw).columns)))
        out = _extract_matches(raw)
        if out is not None and len(out):
            return out
    except Exception as e:
        tried.append(("fbref.team_match_stats(ERR)", [str(e)]))

    # 3) FotMob schedule (if present in your hub)
    try:
        if getattr(hub, "fotmob", None) and hasattr(hub.fotmob, "read_schedule"):
            raw = hub.fotmob.read_schedule()
            tried.append(("fotmob.read_schedule", list(pd.DataFrame(raw).columns)))
            out = _extract_matches(raw)
            if out is not None and len(out):
                return out
    except Exception as e:
        tried.append(("fotmob.read_schedule(ERR)", [str(e)]))

    # 4) Football-Data results
    try:
        import soccerdata as sd
        fd = sd.FootballData(leagues=[league], seasons=[season])
        raw = fd.read_results()
        tried.append(("footballdata.read_results", list(pd.DataFrame(raw).columns)))
        out = _extract_matches(raw)
        if out is not None and len(out):
            return out
    except Exception as e:
        tried.append(("footballdata.read_results(ERR)", [str(e)]))

    # If we got here, dump column samples to help debugging
    sample_info = "; ".join([f"{name}: {cols[:8]}" for name, cols in tried])
    raise RuntimeError(f"Could not map schedule columns from any source. Column samples -> {sample_info}")

# -------------------------
# Optional: augment with corners/cards from Football-Data
# -------------------------

def augment_with_footballdata(df_matches: pd.DataFrame, league: str, season) -> pd.DataFrame:
    """
    Adds columns when available:
      home_corners, away_corners, home_cards, away_cards
    using Football-Data (HC, AC, HY, AY, HR, AR).
    Best-effort join on (date, normalized home/away).  If no data, returns original df.
    """
    try:
        import soccerdata as sd
        fd = sd.FootballData(leagues=[league], seasons=[season])
        df_fd = fd.read_results()
    except Exception:
        return df_matches  # silently skip if missing

    df_fd = _norm(pd.DataFrame(df_fd))
    cols = {c.lower(): c for c in df_fd.columns}

    home_c = _first(cols, "home_team", "hometeam", "home")
    away_c = _first(cols, "away_team", "awayteam", "away")
    date_c = _first(cols, "date", "datetime")
    hc_c   = _first(cols, "hc", "home corners", "homecorners")
    ac_c   = _first(cols, "ac", "away corners", "awaycorners")
    hy_c   = _first(cols, "hy", "home y", "home yellow")
    ay_c   = _first(cols, "ay", "away y", "away yellow")
    hr_c   = _first(cols, "hr", "home r", "home red")
    ar_c   = _first(cols, "ar", "away r", "away red")

    if not (home_c and away_c and date_c):
        return df_matches

    # Build join keys
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

    # Corners
    if hc_c and ac_c:
        merged["home_corners"] = pd.to_numeric(merged[hc_c], errors="coerce")
        merged["away_corners"] = pd.to_numeric(merged[ac_c], errors="coerce")
    # Cards (yellows + reds)
    if hy_c and ay_c:
        merged["home_cards"] = pd.to_numeric(merged[hy_c], errors="coerce").fillna(0)
        merged["away_cards"] = pd.to_numeric(merged[ay_c], errors="coerce").fillna(0)
        if hr_c:
            merged["home_cards"] = merged["home_cards"] + pd.to_numeric(merged[hr_c], errors="coerce").fillna(0)
        if ar_c:
            merged["away_cards"] = merged["away_cards"] + pd.to_numeric(merged[ar_c], errors="coerce").fillna(0)

    keep = ["date", "home", "away", "home_goals", "away_goals", "home_corners", "away_corners", "home_cards", "away_cards"]
    for k in keep:
        if k not in merged.columns:
            merged[k] = merged.get(k)
    return merged[keep].copy()

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
    for k in ("home","draw","away","btts","over_1.5","under_1.5","over_2.5","under_2.5","over_3.5","under_3.5"):
        if k in mk:
            lines.append(f"- **{k.replace('_',' ').upper()}**: {mk[k]:.3f}")

    props = report.get("props", {})
    if "corners" in props:
        lines.append("")
        lines.append("## Corners (Poisson totals)")
        if "corners_lambda" in report["props"]:
            lines.append(f"- λ_total ≈ {report['props']['corners_lambda']:.2f}")
        for k, v in props["corners"].items():
            lines.append(f"  - **{k.replace('_',' ').title()}**: {v:.3f}")
    if "cards" in props:
        lines.append("")
        lines.append("## Cards (Poisson totals)")
        if "cards_lambda" in report["props"]:
            lines.append(f"- λ_total ≈ {report['props']['cards_lambda']:.2f}")
        for k, v in props["cards"].items():
            lines.append(f"  - **{k.replace('_',' ').title()}**: {v:.3f}")

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
    ap.add_argument("--season", required=True)  # accept int or 'YYYY-YYYY'
    ap.add_argument("--fixtures", required=True, help='Semicolon-separated: "Home vs Away|YYYY-MM-DD HH:MM:SS; ..."')
    ap.add_argument("--timezone", default="Australia/Sydney")
    ap.add_argument("--out", default="out/preview_plus.md")
    ap.add_argument("--last-n", type=int, default=10, help="Trend window size")
    args = ap.parse_args()

    # Build hub (keep same adapter you already use)
    # season can be int or 'YYYY-YYYY'; SourceConfig accepts either
    season = args.season
    try:
        season = int(season)
    except Exception:
        pass

    hub = DataHub(SourceConfig(leagues=[args.league], seasons=[season]),
                  enable_whoscored=False, enable_sofascore=False)

    # Build matches from FBref schedule, with robust mapping + fallbacks
    matches_df = build_matches_df(hub, args.league, season)
    matches_df["date"] = pd.to_datetime(matches_df.get("date"), errors="coerce").dt.tz_localize(None)

    # Augment with Football-Data corners/cards when available
    matches_df = augment_with_footballdata(matches_df, args.league, season)

    # Prepare output
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    blocks: list[str] = []

    # Canonicalize names
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

    # Also print to stdout
    try:
        print("\n" + final_md + "\n")
    except Exception:
        pass

if __name__ == "__main__":
    main()
