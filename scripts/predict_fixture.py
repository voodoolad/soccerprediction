
#!/usr/bin/env python
"""
CLI: predict_fixture.py
-----------------------
Produce an all-in-one, readable report for a single fixture using:
- Dixon–Coles goals model (1X2, Over/Under lines, BTTS)
- Poisson totals for Corners and Cards (if counts exist)
- Trend percentages over last N matches

USAGE
-----
python scripts/predict_fixture.py --csv data/matches.csv --home "Team A" --away "Team B"

The CSV should have at least these columns:
  date, home, away, home_goals, away_goals
Optional (for corners/cards trends and props):
  home_corners, away_corners, home_cards, away_cards
"""

from __future__ import annotations
import argparse, json
import pandas as pd
from soccerprediction.reporting.fixture_report import build_fixture_report, ReportConfig

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to a match-level CSV")
    ap.add_argument("--home", required=True, help="Home team name as appears in CSV")
    ap.add_argument("--away", required=True, help="Away team name as appears in CSV")
    ap.add_argument("--out", default="", help="Optional JSON output path")
    ap.add_argument("--last-n", type=int, default=10, help="N matches for trend stats")
    ap.add_argument("--half-life", type=float, default=180.0, help="Recency half-life (days)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    cfg = ReportConfig(last_n_trend=args.last_n, half_life_days=args.half_life)
    report = build_fixture_report(df, args.home, args.away, cfg)

    # pretty print text summary
    print(f"\nFixture: {report['fixture']['home']} vs {report['fixture']['away']}")
    g = report["goals"]
    print(f"Goals λ: home={g['lambda_home']:.2f}, away={g['lambda_away']:.2f}, rho={g['rho']:.2f}")
    print("\nModel probabilities:")
    for k in ("home","draw","away","btts","over_1.5","under_1.5","over_2.5","under_2.5","over_3.5","under_3.5"):
        if k in report["markets"]:
            print(f"  {k:>10}: {report['markets'][k]:.3f}")

    if report["props"]:
        print("\nProps (Poisson totals):")
        if "corners" in report["props"]:
            print(f"  corners λ_total≈{report['props']['corners_lambda']:.2f}")
            for k, v in report["props"]["corners"].items():
                print(f"    {k:>10}: {v:.3f}")
        if "cards" in report["props"]:
            print(f"  cards   λ_total≈{report['props']['cards_lambda']:.2f}")
            for k, v in report["props"]["cards"].items():
                print(f"    {k:>10}: {v:.3f}")

    print("\nTrends (last N matches):")
    tr = report["trends"]
    print(f"  sample size: {tr['samples']} matches")
    if "goals" in tr:
        print("  Goals:")
        for k, v in tr["goals"].items():
            print(f"    {k:>10}: {v:.0%}")
    if "corners" in tr:
        print("  Corners:")
        for k, v in tr["corners"].items():
            print(f"    {k:>10}: {v:.0%}")
    if "cards" in tr:
        print("  Cards:")
        for k, v in tr["cards"].items():
            print(f"    {k:>10}: {v:.0%}")
    if "team_corners" in tr:
        print("  Team corners (last N):")
        for team, block in tr["team_corners"].items():
            s = ", ".join([f"{k}={v:.0%}" for k, v in block.items()])
            print(f"    {team}: {s}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved JSON: {args.out}")

if __name__ == "__main__":
    main()
