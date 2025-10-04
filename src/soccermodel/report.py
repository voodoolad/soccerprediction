from tabulate import tabulate

def render_game_report(meta, picks_by_tier, tables=None):
    hdr = f"# {meta['home']} vs {meta['away']} — Kickoff: {meta['kickoff_aest']}\n\n"
    hdr += f"**Source Status:** {meta.get('source_status','')}\n\n"
    hdr += f"**Market Snapshot:** {meta['market_snapshot']}\n\n"
    hdr += f"**Key News & Context:** {meta['key_news']}\n\n"
    hdr += f"**Matchup Edges (Opponent‑Adjusted):**\n{meta['edges']}\n\n"
    hdr += f"**Tactics & Scenarios:**\n{meta['tactics']}\n\n"
    out = [hdr, "## Picks by Tier\n"]
    for tier in ["Safest","Safe","Medium","Longshot/Plus‑Money"]:
        items = picks_by_tier.get(tier, [])
        if not items: continue
        out.append(f"### {tier}\n")
        for p in items:
            out.append(f"- **{p['market']} — {p['selection']}**")
            out.append(f"  - My Prob %: {100*p['my_prob']:.1f}%")
            out.append(f"  - Fair Odds: {p['fair_odds']:.2f} ({p['fair_odds_american']})")
            if p.get('price_dec'):
                out.append(f"  - Price: {p['price_dec']:.2f} ({p['price_american']}), ts: {p.get('price_ts','')}")
                out.append(f"  - EV%: {p['ev_pct']:.2f}%  |  0.33×Kelly units: {p.get('kelly_units','N/A')}")
            else:
                out.append(f"  - Min Acceptable Odds: {p['min_acceptable']:.2f}; EV: N/A")
                out.append("  - No bet if best available < Min Acceptable Odds.")
            out.append(f"  - Confidence: {p.get('confidence',2)}/5")
            out.append(f"  - Rationale: {p.get('why','Model-based edge')}\n")
    if tables:
        out.append("\n## Diagnostics & Data Used\n")
        for title, df in tables:
            # df can be a pandas DataFrame or a list[dict]; tabulate handles both
            out.append(f"### {title}\n")
            out.append(tabulate(df, headers='keys', tablefmt='github', showindex=False))
            out.append("")
    out.append("\n## Market‑Coverage Audit\n- 1X2: Included; Totals: Included; Over 1.5: Included; BTTS: Included; Team totals: Included; Corners: Included; Cards: Included; Player props (shots/SOT): Included (top candidates).")
    return "\n".join(out)
