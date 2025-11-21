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

Risk/edge feedback loop
-----------------------

`nhl_model3.py` now treats the bet log as a self-correcting safety valve:

- Every run, the model ingests `bets_log.csv`, compares implied vs. realized edge, unit ROI, CLV, and live loss streaks, then updates the thresholds it will accept.
- If performance cools, the minimum required edge/probability rises, Kelly stakes are downscaled, and the slate-level exposure cap tightens; when results improve, the loop slowly relaxes those constraints.
- A short status line (e.g., `🛡️ Risk feedback loop: edge≥0.26; prob≥0.58; Kelly×0.78; exposure≤4.8%; loss streak 3`) prints after the calibration summary so you know why recommendations throttled up or down.
- No special flag is required—just keep `--log-bets` enabled so the loop has fresh data. If you disable logging, the system automatically falls back to the static 0.22 edge / 0.56 probability guardrails.

Automated MoneyPuck ETL
-----------------------

Use `moneypuck_etl.py` to keep the MoneyPuck-derived inputs fresh without manual downloads.

- Discovers the newest MoneyPuck CSVs for teams and goalies (current and prior season fallbacks)
- Normalizes to the schema expected by `nhl_model3.py`
- Runs anomaly detection for data drift, stale seasons, and entity gaps
- Versions every refresh beneath `data/history/` (`versions.json`, dated CSVs, and `anomalies/*.json`)
- NEW: Generates `player_metrics.csv` (skater RAPM/xGAR proxies, lineup buckets, probabilities) directly from MoneyPuck skater summaries

Examples:

```bash
# Basic refresh (writes team_rates.csv & goalie_gsax.csv in the repo root)
python moneypuck_etl.py

# Custom output paths, history directory, and explicit seasons/stages
python moneypuck_etl.py \
  --team-output data/latest/team_rates.csv \
  --goalie-output data/latest/goalie_gsax.csv \
  --player-output data/latest/player_metrics.csv \
  --history-dir data/history \
  --seasons 2025 2024 \
  --stages regular playoffs
```

To run the ETL automatically before generating predictions, add `--refresh-moneypuck` when
calling `nhl_model3.py`. The inline run now refreshes `team_rates.csv`, `goalie_gsax.csv`, **and**
`player_metrics.csv`. You can further tune the inline run with `--moneypuck-history-dir`,
`--moneypuck-stages`, `--moneypuck-seasons`, `--moneypuck-dry-run`, `--fail-on-moneypuck-anomaly`,
and `--moneypuck-request-timeout`.

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
  --goalie-gsax-path goalie_gsax.csv \
  --player-metrics-path player_metrics.csv
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

Skater & Goalie Context
-----------------------

Daily predictions can now ingest player- and goalie-level context:

- Point `--player-metrics-path` to a CSV/JSON with `team`, `player`, `position`, `status`, `prob_play`, `rapm_off`, `rapm_def`, `xgar`.  
  The model aggregates forward/defense xGAR, elite availability, RAPM splits, and applies those covariates both as dashboard notes and as adjustments to the projected total (`skater_goalie_edge`).
- Provide rolling goalie GSAx via `--goalie-gsax-path` (or URL). Expected starter quality is converted into `home_goalie_gsax` / `away_goalie_gsax` features and automatically debits/credits the total based on projected netminders.
- Both inputs surface in the dashboard (`lineup_info`) so you can see which skaters are missing and which starters are driving goalie adjustments.

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

Schedule-aware validation recommendation
----------------------------------------

Schedule-aware validation used to rely on a single 75/25 chronological split plus RandomizedSearch, which meant full seasons were never rolled forward and out-of-time leakage from feature engineering could persist. To approach bookmaker-grade robustness, the codebase now uses rolling-origin cross-validation with explicit season/era-style horizons, walk-forward retraining, and locked feature-generation pipelines so every evaluation reflects a true out-of-sample workflow.

Implementation details
----------------------

- Rolling-origin cross-validation now drives the hyperparameter search inside `RealDataNHLModel.train_model`, ensuring each trial retrains on an expanding window and scores on a forward horizon.
- Every ensemble member sits inside a locked `Pipeline(StandardScaler, model)` so feature transforms are refit only on the relevant training fold and the same pipeline object is reused at inference time.
- A walk-forward retraining harness re-clones the tuned ensemble across sequential chunks, surfacing RMSE/MAE summaries that mirror real deployment where the model is continually refreshed.
- Goal distribution modeling now stacks three layers: (1) classical Poisson regressors for home/away goals, (2) a neural "Poisson flow" MLP that learns non-linear intensities, and (3) a Gaussian-copula mixture simulator that blends the two regimes with dynamic weights/correlation derived from skater/goalie covariates. The mixture powers the over/under probabilities and keeps pace with bookmaker-style conditional totals.

