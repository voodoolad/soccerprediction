# Soccer Market Model — End‑to‑End Starter

A reproducible pipeline that turns your betting prompt into code:
- **Scrapes** FBref, Understat, SofaScore, ESPN, FotMob, ClubElo, MatchHistory using the official [`soccerdata`](https://github.com/probberechts/soccerdata) package.
- **Models** match outcomes and derivatives:
  - **Goals**: Dixon–Coles (scoreline PMF → 1X2, Asian, Totals, BTTS, Over 1.5, Team Totals, Double Chance)
  - **Corners**: Poisson GLM (features: crosses, attacking‑third touches)
  - **Cards**: Poisson GLM (features: fouls, duels)
  - **Player props (scaffolded)**: ATGS, shots, SOT, assists, fouls (minutes model + Understat usage)
- **Prices** markets with **Fair Odds** and **Min Acceptable**; optional **live odds** → **EV%** and **0.33×Kelly**.
- **Outputs** a Markdown preview with picks per tier. Optionally prints to console.

---

## 1) Requirements

- **Python 3.9–3.12** (tested on 3.12). `soccerdata` is not yet built for 3.13.
- OS: Windows (PowerShell or CMD), macOS, or Linux.
- Internet access for scraping.

---

## 2) Install

```bat
py -3.12 -m venv .venv312
.\.venv312\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
