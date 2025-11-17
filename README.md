NHL Over/Under Prediction Model
================================

Quick start
-----------

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the model (predict-only, no posting/logging):

```bash
python nhl_model3.py
```

Command-line options
--------------------

- --odds-path: Path to odds JSON (default `odds.json`)
- --closing-odds-path: Path to closing odds JSON for CLV logging (default `closing_odds.json`)
- --log-bets: Enable logging recommended bets to CSV
- --log-path: Bets log CSV path (default `bets_log.csv`)
- --post-social: Enable social posting (Twitter/Discord)
- --team-rates-path: CSV/URL with team rates (xGF60 5v5, HDCF60 5v5, PP xGF60, PK xGA60)
- --goalie-gsax-path: CSV/URL with goalie rolling GSAx and optional `prob_start`
- --penalty-rates-path: CSV/URL with team penalties drawn/taken per 60 (default `penalties.csv`)
- --referee-rates-path: CSV/URL with referee penalties per 60 (optional)
- --environment-path: Path to `environment.json` (outdoor/start time/weather per game)
- --lineup-path: Path to `lineup_strength.csv` (team lineup strength)
- --auto-populate: Auto-fetch MoneyPuck URLs and write normalized CSVs
- --team-rates-url / --goalie-gsax-url / --penalties-url / --referees-url: Source URLs for auto-populate
- --realtime-odds: Fetch live totals from The Odds API (US region only; requires `ODDS_API_KEY`)
- --log-odds-history / --log-odds: Append realtime odds snapshots to `odds_history.csv`

Examples
--------

1) Use custom odds, log bets, compute CLV from closing odds:

```bash
python nhl_model3.py --odds-path odds.json --closing-odds-path closing_odds.json --log-bets
```

2) Predict and post to social media:

```bash
python nhl_model3.py --post-social
```

3) Auto-populate team/goalie metrics from MoneyPuck and run:

```bash
python nhl_model3.py \
  --auto-populate \
  --team-rates-url https://moneypuck.com/teams.htm \
  --team-rates-path team_rates.csv \
  --goalie-gsax-url https://moneypuck.com/goalies.htm \
  --goalie-gsax-path goalie_gsax.csv
```

4) Use environment and lineup inputs:

```bash
python nhl_model3.py --environment-path environment.json --lineup-path lineup_strength.csv
```

`environment.json` schema (keyed by `game_id` or `AWAY@HOME`):

```json
{
  "2025010101": {"outdoor": true, "start_hour_local": 13, "temp_f": 28, "wind_mph": 5},
  "TOR@BOS": {"outdoor": false, "start_hour_local": 19}
}
```

`lineup_strength.csv` schema:

```csv
team,lineup_strength
BOS,2.5
TOR,2.3
UTA,2.0
```

Odds JSON formats
-----------------

Single book per game id or matchup string:

```json
{
  "VGK@FLA": {"total": 6.0, "over": -110, "under": -105},
  "demo_0": 6.5
}
```

Multi-book array (consensus total is the median; best price per side is selected):

```json
{
  "VGK@FLA": [
    {"book": "PINN", "total": 6.0, "over": -112, "under": 100},
    {"book": "DK",   "total": 6.0, "over": -110, "under": -105}
  ]
}
```

Closing odds JSON (for CLV):

```json
{
  "VGK@FLA": {"closing_total": 6.0},
  "demo_0": {"closing_total": 6.5}
}
```

Social posting
--------------

Provide credentials via environment variables or `social_config.json` (created automatically if missing). The dashboard is saved to `nhl_real_data_dashboard.html`.


Player SOG Props
----------------

Evaluate player shots-on-goal props using contextual team and referee data.

1) Prepare props JSON (a sample is provided at `player_props.json`):

```json
[
  {"player": "David Pastrnak", "team": "BOS", "opponent": "TOR", "line": 4.5, "market": "sog", "ref": "Ref A"}
]
```

2) Run the SOG props evaluator:

```bash
python run_player_props.py --props player_props.json --min-edge 0.55
```

Outputs expected SOG and a Poisson-based recommendation with probability edge.

Using NHL Stats & The Odds API
------------------------------

You can evaluate SOG props for today's games with live data:

```bash
python run_today_sog.py --use-api --use-stats --use-odds --min-edge 0.55 --odds-key YOUR_THE_ODDS_API_KEY
```

- `--use-api`: fetches today's NHL schedule
- `--use-stats`: augments players with top shooters from NHL Stats REST (`https://api.nhle.com/stats/rest/en`)
- `--use-odds`: pulls SOG lines from The Odds API (set `--odds-key` or `THE_ODDS_API_KEY` env var)
- `--all-players`: include all skaters from each team (requires `--use-stats`)
- `--only-odds`: restrict to players that have an odds line

Dashboard
---------

The model writes an HTML dashboard `nhl_real_data_dashboard.html` with Env/Lineup columns, confidence, EV, and odds details.

```bash
python run_today_sog.py --use-api --use-stats --use-odds --min-edge 0.55
open bets_dashboard.html  # or your OS equivalent
```

