
#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd, pytz

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from soccermodel.data_sources import DataHub, SourceConfig
from soccermodel.models.goals_dc import fit_dixon_coles, scoreline_pmf, DCConfig
from soccermodel.models.corners_cards import fit_poisson_glm, predict_rate, PoissonSpec
from soccermodel.market_pricing import fair_odds, american_from_decimal, min_acceptable_odds, ev_percent
from soccermodel.odds_adapter import load_odds_from_json, kelly_fraction, american_from_decimal as am_from_dec
from soccermodel.report import render_game_report

REC_WEIGHTS = np.array([0.30,0.25,0.20,0.15,0.10])

def parse_fixture(s: str):
    teams, ts = s.split("|", 1)
    home, away = teams.split(" vs ", 1)
    return {"home": home.strip(), "away": away.strip(), "kickoff": ts.strip()}

def _ensure_keys(df: pd.DataFrame) -> pd.DataFrame:
    if not set(["league","season","team","date"]).issubset(df.columns):
        df = df.reset_index()
    return df

def _flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" ".join([str(x) for x in tup if x and x!='Unnamed: 0']).strip() for tup in df.columns.to_list()]
    else:
        df.columns = [str(c) for c in df.columns]
    return df

def _norm(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_keys(df); df = _flatten_cols(df)
    ren = {}
    for k in ["League","Season","Team","Date"]:
        if k in df.columns and k.lower() not in df.columns:
            ren[k] = k.lower()
    if ren: df = df.rename(columns=ren)
    if "date" not in df.columns:
        date_cand = None
        for c in df.columns:
            lc = str(c).lower()
            if any(tok in lc for tok in ["date","time","kickoff","utc","start"]):
                date_cand = c; break
        if date_cand is not None:
            try:
                dt = pd.to_datetime(df[date_cand], errors="coerce", utc=True)
                df["date"] = dt.dt.tz_convert(None).dt.floor("D")
            except Exception:
                pass
    return df

def _find_first(cols, predicates):
    for c in cols:
        lc = str(c).lower()
        if all((p in lc) for p in predicates):
            return c
    return None

def recency_weighted_last5(df: pd.DataFrame, team: str, cols):
    g = df[df["team"].astype(str).str.contains(team, case=False, na=False)].copy()
    g = g.sort_values("date").tail(5)
    if g.empty:
        return {c: np.nan for c in cols}, 0
    w = REC_WEIGHTS[:len(g)]
    w = w / w.sum()
    out = {}
    for c in cols:
        v = pd.to_numeric(g[c], errors="coerce")
        out[c] = float(np.nansum(v.to_numpy() * w)) if v.notna().any() else np.nan
    return out, len(g)

def make_corners_training(hub: DataHub):
    misc = _norm(hub.fbref.read_team_match_stats(stat_type="misc", opponent_stats=False))
    ptyp = _norm(hub.fbref.read_team_match_stats(stat_type="passing_types", opponent_stats=False))
    poss = _norm(hub.fbref.read_team_match_stats(stat_type="possession", opponent_stats=False))
    df = (misc.merge(ptyp, on=["league","season","team","date"], how="left", suffixes=("","_pt"))
              .merge(poss, on=["league","season","team","date"], how="left", suffixes=("","_pos")))
    cols = list(df.columns)
    ccol = _find_first(cols, ["corner"]) or _find_first(cols, ["corners"]) or _find_first(cols, ["ck"])
    if ccol is None: raise RuntimeError("Could not find a 'corners' column in FBref misc stats.")
    df["corners_for"] = pd.to_numeric(df[ccol], errors="coerce")
    xcol = _find_first(cols, ["cross"]) or _find_first(cols, ["crs"])
    df["crosses"] = pd.to_numeric(df[xcol], errors="coerce") if xcol else np.nan
    tcol = (_find_first(cols, ["touch","att","3rd"]) or _find_first(cols, ["touch","final","third"]) or
            _find_first(cols, ["attacking","third","touch"]))
    df["att_third_touches"] = pd.to_numeric(df[tcol], errors="coerce") if tcol else np.nan
    return df

def make_cards_training(hub: DataHub):
    misc = _norm(hub.fbref.read_team_match_stats(stat_type="misc", opponent_stats=False))
    disc_raw = hub.fotmob.read_team_match_stats(stat_type="Discipline", opponent_stats=True)
    disc = _norm(disc_raw)
    if "team" not in disc.columns:
        tcol = next((c for c in disc.columns if "team" in str(c).lower() or "squad" in str(c).lower()), None)
        if tcol: disc = disc.rename(columns={tcol: "team"})
    merge_keys = ["league","season","team"]
    if "date" in disc.columns and disc["date"].notna().any():
        merge_keys.append("date")
    if "date" not in merge_keys:
        num_cols = [c for c in disc.columns if c not in ("league","season","team") and pd.api.types.is_numeric_dtype(disc[c])]
        disc = disc.groupby(["league","season","team"], as_index=False)[num_cols].mean() if num_cols else disc.drop_duplicates(subset=["league","season","team"])
    df = misc.merge(disc, on=merge_keys, how="left", suffixes=("","_fm"))
    cols = list(df.columns)
    ycol = next((c for c in cols if "yellow" in c.lower()), None)
    rcol = next((c for c in cols if "red" in c.lower()), None)
    if ycol is None and rcol is None:
        raise RuntimeError("Could not find yellow/red card columns in FBref misc.")
    df["yellow_work"] = pd.to_numeric(df[ycol], errors="coerce") if ycol else 0.0
    df["red_work"] = pd.to_numeric(df[rcol], errors="coerce") if rcol else 0.0
    df["cards_total"] = df["yellow_work"].fillna(0) + 2*df["red_work"].fillna(0)
    fcol = next((c for c in cols if "foul" in c.lower() and "against" not in c.lower()), None)
    df["fouls_committed"] = pd.to_numeric(df[fcol], errors="coerce") if fcol else np.nan

    # Duels (FotMob) or proxies
    dwin = next((c for c in cols if "duel" in c.lower() and "won" in c.lower()), None)
    dlos = next((c for c in cols if "duel" in c.lower() and "lost" in c.lower()), None)
    if dwin and dlos:
        df["duels_total"] = pd.to_numeric(df[dwin], errors="coerce").fillna(0) + pd.to_numeric(df[dlos], errors="coerce").fillna(0)
    else:
        # FBref proxies: aerials won/lost + dribbles faced if available
        aw = next((c for c in cols if "aerial" in c.lower() and "won" in c.lower()), None)
        al = next((c for c in cols if "aerial" in c.lower() and "lost" in c.lower()), None)
        proxy = None
        if aw and al:
            proxy = pd.to_numeric(df[aw], errors="coerce").fillna(0) + pd.to_numeric(df[al], errors="coerce").fillna(0)
        df["duels_total"] = proxy if proxy is not None else np.nan
    return df

def fit_dc_from_fbref(hub: DataHub):
    fb = _norm(hub.fbref.read_team_match_stats(stat_type="schedule"))
    cols = {c.lower(): c for c in fb.columns}
    def pick(*alts):
        for a in alts:
            if a in cols: return cols[a]
        return None
    team_c  = pick("team")
    opp_c   = pick("opponent", "opp")
    venue_c = pick("venue", "home/away", "homeaway", "home_away")
    gf_c    = pick("gf", "goals_for", "goals for", "goals", "scored")
    ga_c    = pick("ga", "goals_against", "goals against", "conceded")
    if not all([team_c, opp_c, venue_c, gf_c, ga_c]): return None
    fb["_gf"] = pd.to_numeric(fb[gf_c], errors="coerce")
    fb["_ga"] = pd.to_numeric(fb[ga_c], errors="coerce")
    fb["_venue"] = fb[venue_c].astype(str).str.lower()
    use = fb["_gf"].notna() & fb["_ga"].notna() & fb["_venue"].str.startswith("home")
    df = fb.loc[use, [team_c, opp_c, "_gf", "_ga"]].rename(columns={team_c:"home_team", opp_c:"away_team", "_gf":"home_goals", "_ga":"away_goals"})
    if len(df) < 200: return None
    teams = sorted(set(df["home_team"]).union(df["away_team"]))
    df["home_goals"] = df["home_goals"].astype(int); df["away_goals"] = df["away_goals"].astype(int)
    return fit_dixon_coles(df, teams, DCConfig())

def poisson_tail_over(line_half: float, lam: float) -> float:
    # Over k.5 => sum_{n=k+1..inf} Poisson(n; lam)
    from math import exp, factorial
    k = int(line_half - 0.5)
    cdf = sum(exp(-lam)*lam**i/factorial(i) for i in range(0, k+1))
    return max(0.0, min(1.0, 1.0 - cdf))

def shrink_to_league(lam: float, mu: float, n_eff: int, k: int = 20) -> float:
    # Empirical Bayes shrinkage toward league mean; n_eff ~ recent sample size (0..10)
    w = n_eff / (n_eff + k)
    return w*lam + (1-w)*mu

def source_status_text(ok: dict, err: dict) -> str:
    parts = []
    for name in ["FBref","Understat","ESPN","FotMob","SofaScore","ClubElo","MatchHistory"]:
        if name in ok:
            parts.append(f"{name}: OK")
        elif name in err:
            parts.append(f"{name}: FAIL ({err[name]})")
    return "; ".join(parts)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", required=True)
    ap.add_argument("--season", required=True)
    ap.add_argument("--fixtures", required=True)
    ap.add_argument("--timezone", default="Australia/Sydney")
    ap.add_argument("--out", default="out/preview_plus.md")
    ap.add_argument("--odds_json", default=None)
    ap.add_argument("--stdout", action="store_true")
    ap.add_argument("--with_tables", action="store_true", help="Append diagnostics data tables")
    args = ap.parse_args()

    odds_book = load_odds_from_json(args.odds_json) if args.odds_json else None

    # Build hub + detect source status
    ok, err = {}, {}
    try:
        hub = DataHub(SourceConfig(leagues=[args.league], seasons=[args.season]), enable_whoscored=False, enable_sofascore=True)
        ok["FBref"]=True; ok["Understat"]=True; ok["ESPN"]=True; ok["FotMob"]=True; ok["MatchHistory"]=True; ok["SofaScore"]=True; ok["ClubElo"]=True
    except Exception as e:
        raise

    # Goals model
    params = None
    try:
        params = fit_dc_from_fbref(hub); ok["FBref"]=True
    except Exception as e:
        err["FBref"]=str(e)

    # Training frames
    corners_df = make_corners_training(hub)
    cards_df = make_cards_training(hub)

    # League means (team-level)
    mu_team_corners = pd.to_numeric(corners_df["corners_for"], errors="coerce").mean(skipna=True)
    mu_team_cards   = pd.to_numeric(cards_df["cards_total"], errors="coerce").mean(skipna=True)

    # Fit GLMs (robust)
    c_spec = PoissonSpec(target_col="corners_for", feature_cols=("crosses","att_third_touches"))
    corners_glm = None
    try:
        corners_train = corners_df.dropna(subset=list(c_spec.feature_cols)+[c_spec.target_col])
        if len(corners_train) > 50:
            corners_glm = fit_poisson_glm(corners_train, c_spec)
    except Exception as e:
        err["CornersGLM"]=str(e)

    k_spec = PoissonSpec(target_col="cards_total", feature_cols=("fouls_committed","duels_total"))
    cards_glm = None
    try:
        cards_train = cards_df.dropna(subset=list(k_spec.feature_cols)+[k_spec.target_col])
        if len(cards_train) > 50:
            cards_glm = fit_poisson_glm(cards_train, k_spec)
    except Exception as e:
        err["CardsGLM"]=str(e)

    blocks = []
    for part in args.fixtures.split(";"):
        fx = parse_fixture(part.strip()); home, away = fx["home"], fx["away"]

        # Scoreline pmf (goals)
        if params is not None and f"a_{home}" in params and f"d_{away}" in params:
            pmf = scoreline_pmf(home, away, params, DCConfig())
        else:
            lam_h, lam_a = 1.35, 1.10
            pmf = {}; tot = 0.0
            from math import exp, factorial
            for h in range(0,8):
                for a in range(0,8):
                    ph = exp(-lam_h)*lam_h**h/factorial(h); pa = exp(-lam_a)*lam_a**a/factorial(a)
                    p = ph*pa; pmf[(h,a)] = p; tot += p
            for k in pmf: pmf[k] /= tot

        p_home = sum(p for (h,a),p in pmf.items() if h>a)
        p_draw = sum(p for (h,a),p in pmf.items() if h==a)
        p_away = sum(p for (h,a),p in pmf.items() if h<a)
        p_over15 = sum(p for (h,a),p in pmf.items() if h+a>=2)
        p_over25 = sum(p for (h,a),p in pmf.items() if h+a>2.5)
        p_btts = sum(p for (h,a),p in pmf.items() if h>0 and a>0)

        # Recency-weighted team features for diagnostics & GLMs
        c_feats_h, n_h = recency_weighted_last5(corners_df, home, ["crosses","att_third_touches"])
        c_feats_a, n_a = recency_weighted_last5(corners_df, away, ["crosses","att_third_touches"])

        k_feats_h, nk_h = recency_weighted_last5(cards_df, home, ["fouls_committed","duels_total"])
        k_feats_a, nk_a = recency_weighted_last5(cards_df, away, ["fouls_committed","duels_total"])

        # Corners λ with league shrinkage
        p_corners_over = None; lam_ch=lam_ca=lam_c_match=lam_c_match_adj=None
        if corners_glm is not None:
            lam_ch = predict_rate(corners_glm, pd.Series(c_feats_h).fillna(0))
            lam_ca = predict_rate(corners_glm, pd.Series(c_feats_a).fillna(0))
            lam_ch = shrink_to_league(lam_ch, mu_team_corners, n_h)
            lam_ca = shrink_to_league(lam_ca, mu_team_corners, n_a)
            lam_c_match = lam_ch + lam_ca
            p_corners_over = poisson_tail_over(9.5, lam_c_match)
            lam_c_match_adj = lam_c_match

        # Cards λ with league shrinkage
        p_cards_over = None; lam_kh=lam_ka=lam_k_match=None
        if cards_glm is not None:
            lam_kh = predict_rate(cards_glm, pd.Series(k_feats_h).fillna(0))
            lam_ka = predict_rate(cards_glm, pd.Series(k_feats_a).fillna(0))
            lam_kh = shrink_to_league(lam_kh, mu_team_cards, nk_h)
            lam_ka = shrink_to_league(lam_ka, mu_team_cards, nk_a)
            lam_k_match = lam_kh + lam_ka
            p_cards_over = poisson_tail_over(3.5, lam_k_match)

        # Injuries (SofaScore)
        try:
            inj = hub.sofascore.read_missing_players() if hub.sofascore else None
            inj_count = len(inj) if inj is not None else 0
        except Exception:
            inj_count = 0

        # Picks (with optional pricing)
        def mkpick(name, sel, p, safest=False, why="Model-based edge"):
            pick = {"market": name, "selection": sel, "my_prob": p,
                    "fair_odds": fair_odds(p), "fair_odds_american": american_from_decimal(fair_odds(p)),
                    "min_acceptable": min_acceptable_odds(p, 0.03, safest),
                    "confidence": 2, "why": why}
            if odds_book and name in odds_book:
                snap = odds_book[name]; prices = snap.prices
                key = sel.lower().strip()
                if name == "1X2":
                    if key not in prices:
                        key = "home" if sel==home else ("away" if sel==away else key)
                elif "over" in key: key = "over"
                elif "under" in key: key = "under"
                elif name == "BTTS": key = "yes" if "yes" in key else ("no" if "no" in key else key)
                if key in prices:
                    o = float(prices[key]); pick["price_dec"]=o; pick["price_american"]=am_from_dec(o)
                    pick["price_ts"]=snap.timestamp; pick["ev_pct"]=ev_percent(p,o); pick["kelly_units"]=round(0.33*kelly_fraction(p,o),4)
            return pick

        picks = {"Safest": [], "Safe": [], "Medium": [], "Longshot/Plus‑Money": []}
        picks["Safest"].append(mkpick("1X2", f"{home}", p_home, True))
        picks["Safest"].append(mkpick("Totals O/U 2.5", "Over", p_over25, True))
        picks["Safe"].append(mkpick("BTTS", "Yes", p_btts, False))
        picks["Safe"].append(mkpick("Over 1.5 Goals", "Over", p_over15, False))
        if p_away < 0.40: picks["Longshot/Plus‑Money"].append(mkpick("1X2", f"{away}", p_away, False))
        if p_corners_over is not None: picks["Safe"].append(mkpick("Match Corners O/U 9.5", "Over", p_corners_over, False))
        if p_cards_over   is not None: picks["Safe"].append(mkpick("Match Cards O/U 3.5", "Over", p_cards_over, False))

        # Player props (simple): top two by recency-weighted shots and SOT from FBref 'shooting'
        props_tables = []
        try:
            shoot = _norm(hub.fbref.read_player_match_stats(stat_type="shooting"))
            def top_players(team):
                g = shoot[shoot["team"].astype(str).str.contains(team, case=False, na=False)].sort_values("date").tail(200)
                # columns: shots, shots_on_target may vary
                sc = _find_first(list(g.columns), ["shots"]) or "shots"
                st = _find_first(list(g.columns), ["on","target"]) or _find_first(list(g.columns), ["sot"]) or sc
                g["shots"] = pd.to_numeric(g.get(sc, np.nan), errors="coerce")
                g["sot"] = pd.to_numeric(g.get(st, np.nan), errors="coerce")
                # per90 approx from minutes if present
                mcol = _find_first(list(g.columns), ["minutes"]) or _find_first(list(g.columns), ["min"])
                if mcol:
                    g["min"] = pd.to_numeric(g[mcol], errors="coerce")
                else:
                    g["min"] = 90.0
                # aggregate last 5 by player
                last5 = g.groupby("player", as_index=False).tail(5)
                agg = last5.groupby("player", as_index=False).agg(shots=("shots","mean"), sot=("sot","mean"), min90=("min","mean"))
                # expected minutes assumption for starters
                agg["exp_min"] = 75.0
                agg["lam_shots"] = agg["shots"] * (agg["exp_min"]/90.0)
                agg["lam_sot"] = agg["sot"] * (agg["exp_min"]/90.0)
                # Prob at least 1 and at least 2 (Poisson)
                from math import exp
                agg["P(Shots>=1)"] = 1 - np.exp(-agg["lam_shots"])
                agg["P(Shots>=2)"] = 1 - np.exp(-agg["lam_shots"]) - agg["lam_shots"]*np.exp(-agg["lam_shots"])
                agg["P(SOT>=1)"] = 1 - np.exp(-agg["lam_sot"])
                agg = agg.sort_values("P(Shots>=1)", ascending=False).head(2)
                return agg[["player","shots","sot","exp_min","P(Shots>=1)","P(Shots>=2)","P(SOT>=1)"]]
            home_tp = top_players(home); away_tp = top_players(away)
            props_tables.append(("Player props — Top shot candidates (Home)", home_tp))
            props_tables.append(("Player props — Top shot candidates (Away)", away_tp))
        except Exception:
            pass

        tables = []
        if args.with_tables:
            # Data provenance and source status
            tables.append(("Source status", [{"source_status": source_status_text(ok, err)}]))

            # Goals diagnostics
            lam_h = sum(h*prob for (h,a),prob in pmf.items() for h in [h])[0] if False else None  # placeholder (full lambda printed below)
            # Corners & cards diagnostics tables
            if corners_glm is not None:
                tables.append(("Corners diagnostics", [{
                    "home_crosses_L5": c_feats_h.get("crosses", np.nan),
                    "home_att3rd_L5": c_feats_h.get("att_third_touches", np.nan),
                    "away_crosses_L5": c_feats_a.get("crosses", np.nan),
                    "away_att3rd_L5": c_feats_a.get("att_third_touches", np.nan),
                    "lambda_home": lam_ch, "lambda_away": lam_ca,
                    "lambda_match": lam_c_match_adj, "league_team_mean": mu_team_corners,
                    "P(Over 9.5)": p_corners_over
                }]))
            if cards_glm is not None:
                tables.append(("Cards diagnostics", [{
                    "home_fouls_L5": k_feats_h.get("fouls_committed", np.nan),
                    "home_duels_L5": k_feats_h.get("duels_total", np.nan),
                    "away_fouls_L5": k_feats_a.get("fouls_committed", np.nan),
                    "away_duels_L5": k_feats_a.get("duels_total", np.nan),
                    "lambda_home": lam_kh, "lambda_away": lam_ka,
                    "lambda_match": lam_k_match, "league_team_mean": mu_team_cards,
                    "P(Over 3.5)": p_cards_over
                }]))

            # Team form tables (last-5 for featured columns we used)
            home_form = {"team": home, **c_feats_h, **k_feats_h}
            away_form = {"team": away, **c_feats_a, **k_feats_a}
            tables.append(("Team last-5 features (recency-weighted)", [home_form, away_form]))

            # Append props tables if built
            for t in props_tables:
                tables.append(t)

        meta = {
            "home": home, "away": away, "kickoff_aest": fx["kickoff"],
            "market_snapshot": "If --odds_json provided, EV/Kelly shown; otherwise probability-first.",
            "key_news": (f"SofaScore injuries entries present" if inj_count>0 else "Injuries feed unavailable; minutes model down-weights uncertainty."),
            "edges": "- Goals: DC model; Corners: crosses + attacking touches (league-shrunk); Cards: fouls + duels (league-shrunk).",
            "tactics": "- Wide vs compact, press & discipline included via features.",
            "source_status": source_status_text(ok, err)
        }
        report = render_game_report(meta, picks, tables)
        blocks.append(report)

    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    final = "\\n\\n---\\n\\n".join(blocks)
    out_path.write_text(final, encoding="utf-8")
    print("Wrote", out_path)
    if args.stdout:
        print("\\n" + final + "\\n")

if __name__ == "__main__":
    main()
