NHL Over/Under Prediction Model
================================

Quick start
-----------

Install dependencies:

```bash
pip install -r requirements.txt
```

Optional social posting dependencies (Twitter/Discord):

```bash
pip install -r requirements-social.txt
```

Run the model (predict-only, no posting/logging):

```bash
python nhl_model3.py
```

Web apps (Streamlit)
--------------------

This repo now provides two Streamlit entry points:

- **Public app** (`streamlit_app.py` or `streamlit_public.py`)
  - Read-only public-facing view for `predictions.png` + dashboard output.
- **Admin app** (`streamlit_admin.py`)
  - Full controls to run `nhl_model3.py`, upload input JSON files, and download artifacts.
  - Requires passphrase login.

Run locally:

```bash
# Public
streamlit run streamlit_app.py

# Admin
streamlit run streamlit_admin.py
```

For Streamlit Cloud deployment:

1. Push this repo to GitHub.
2. Create one app with `streamlit_app.py` (public).
3. Create a second app with `streamlit_admin.py` (admin).
4. Set secrets:
   - `ODDS_API_KEY = "your_key"` (optional, for realtime odds)
   - `ADMIN_PASSPHRASE = "your_admin_passphrase"` (required for admin app)
5. The repo includes `runtime.txt` (`python-3.12`) to keep Cloud dependency installs stable.

GitHub CI automation (no local runs required)
--------------------------------------------

This repo includes `.github/workflows/run-nhl-model.yml` so the model can run on GitHub Actions instead of your local machine.

What it does:

- Runs daily on a schedule (and supports manual runs from the Actions tab).
- Installs dependencies and executes `nhl_model3.py`.
- Uses live odds when `ODDS_API_KEY` is configured; otherwise it falls back to `odds.json`.
- Updates and commits generated artifacts (`predictions.png`, `nhl_real_data_dashboard.html`, `bets_log.csv`, etc.) back to the repo.

One-time GitHub setup:

1. Go to **Settings → Secrets and variables → Actions** and add:
   - `ODDS_API_KEY` (optional but recommended for realtime odds)
2. Go to **Settings → Actions → General → Workflow permissions** and set:
   - **Read and write permissions** (required so CI can commit updated outputs)
3. In GitHub, open **Actions → Run NHL Model** and click **Run workflow** for an initial manual run.

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
- --status-history-path: Optional CSV with historical goalie/injury adjustments (time-aligned) to enrich training features
- --auto-populate: Auto-fetch MoneyPuck URLs and write normalized CSVs
- --team-rates-url / --goalie-gsax-url / --penalties-url / --referees-url: Source URLs for auto-populate
- --realtime-odds: Fetch live totals from The Odds API (US region only; requires `ODDS_API_KEY`)
- --log-odds-history / --log-odds: Append realtime odds snapshots to `odds_history.csv`
- --build-closing-lines: Build `data/history/closing_lines.csv` from `--odds-history-path` and exit
- --closing-lines-path: Output path for canonical closing lines CSV (default `data/history/closing_lines.csv`)
- --closing-lines-version: Optional version prefix to use from `CLOSING_LINES_VERSION_DIR` (e.g., `20260107` or `20260107T120000Z`) for reproducible backtests
- --train-speed: Choose `turbo`, `fast`, `balanced` (default), or `full` to control CV depth, walk-forward checks, and tuning iterations
- --fast-train: Shortcut for `--train-speed fast`
- --train-target: Choose `auto` (default), `edge` (train on total_goals - closing_total), or `total` (train on total goals)
- --max-train-samples: Cap how many historical games are used for training (useful for fast experimentation)
- --model-path: Path to persist/load a serialized ensemble artifact (default `data/cache/trained_model.joblib`)
- --save-trained-model: Persist the ensemble from the current run to `--model-path`
- --use-saved-model: Skip retraining and hydrate the ensemble from `--model-path`
- --historical-days: Limit how many days of completed games are downloaded before training (default `HISTORICAL_DAYS` env or 365)
- --historical-cache-path: CSV cache for historical games (default `data/history/historical_games.csv`; set to empty to disable)
- --historical-cache-refresh: Force-refresh the historical cache even if the file already exists

Faster training presets
-----------------------

- `turbo` reuses preset hyperparameters (no RandomizedSearch), limits training data to the most recent 400 games, skips goal-flow modeling, and turns off walk-forward/stacked-weight recalibration—use when you just need a quick sanity check (<5 minutes on most machines).
- `balanced` trims the RandomizedSearch grid, scales down rolling-origin splits, and still runs walk-forward checks—this replaces the previous "full" default to cut runtimes ~40–60%.
- `fast` further caps the training sample (600 games by default), skips stacking weight recalibration and walk-forward validation, and reduces each model search to two iterations for sub-10-minute runs.
- `full` preserves the original exhaustive workflow (long rolling CV, eight-iteration searches, and walk-forward auditing).
- You can also set `TRAIN_SPEED=turbo|fast|balanced|full` and optionally `MAX_TRAIN_SAMPLES=<N>` via environment variables for automation.

Persisting trained ensembles
----------------------------

- Run `python nhl_model3.py --save-trained-model` after a full training cycle to write the ensemble (pipelines, weights, goal models, conformal stats) to `--model-path` (default `data/cache/trained_model.joblib`).
- Subsequent prediction-only runs can skip the expensive training step via `python nhl_model3.py --use-saved-model`, which hydrates the cached ensemble but still refreshes historical data, context tables, and odds.
- Override the default location globally with `TRAINED_MODEL_PATH=/path/to/model.joblib`, or pass a one-off path with `--model-path`.

Risk/edge feedback loop
-----------------------

`nhl_model3.py` now treats the bet log as a self-correcting safety valve:

- Every run, the model ingests `bets_log.csv`, compares implied vs. realized edge, unit ROI, CLV, and live loss streaks, then updates the thresholds it will accept.
- If performance cools, the minimum required edge/probability rises, Kelly stakes are downscaled, and the slate-level exposure cap tightens; when results improve, the loop slowly relaxes those constraints.
- A short status line (e.g., `🛡️ Risk feedback loop: edge≥0.26; prob≥0.58; Kelly×0.78; exposure≤4.8%; loss streak 3`) prints after the calibration summary so you know why recommendations throttled up or down.
- No special flag is required—just keep `--log-bets` enabled so the loop has fresh data. If you disable logging, the system automatically falls back to the static 0.22 edge / 0.56 probability guardrails.

High-precision accuracy guardrails
----------------------------------

To push recommendation accuracy as high as possible, the training routine now back-tests two defensive filters on the holdout split and automatically activates them at inference time:

- **Precision edge floor:** we scan a grid of absolute-edge cutoffs, measure the accuracy achieved when only acting on games whose model edge exceeds that cutoff, and retain the tightest floor that materially improves accuracy. This dynamic threshold (often 0.40+ goals) overrides the historical 0.22 edge minimum whenever it proves superior on the validation slate.
- **Consensus variance cap:** we also inspect the standard deviation between ensemble members for every holdout game. If near-100% accuracy only occurs when the ensemble disagrees by ≤0.12 goals (example), we store that cap and decline future bets whenever the live disagreement exceeds it.

Every prediction now carries `model_consensus_std`, `model_consensus_range`, and `edge_threshold_used` diagnostics, and `OverUnderPrediction.no_bet_reason` will surface `precision_guard` or `consensus_guard` whenever these high-precision filters suppress a wager. Together with the risk feedback loop, this provides a stacked safety system that only green-lights the most unanimous, high-edge positions.

Moneyline probabilities
-----------------------

The same Poisson/flow goal simulations that power the totals market now emit implied win probabilities, EV, and Kelly sizing for both sides of the Moneyline. Provide `home_moneyline` / `away_moneyline` prices (or per-book arrays) in `odds.json` and the model will select the best available price, track consensus values, and surface the metrics everywhere predictions are written (CLI output, Excel/HTML exports, dashboard). Realtime odds fetches automatically request the `h2h` market so no config changes are required when using `--realtime-odds`.

Automated MoneyPuck ETL
-----------------------

Use `moneypuck_etl.py` to keep the MoneyPuck-derived inputs fresh without manual downloads.

- Discovers the newest MoneyPuck CSVs for teams and goalies (current and prior season fallbacks)
- Normalizes to the schema expected by `nhl_model3.py`
- Runs anomaly detection for data drift, stale seasons, and entity gaps
- Versions every refresh beneath `data/history/` (`versions.json`, dated CSVs, and `anomalies/*.json`)

Examples:

```bash
# Basic refresh (writes team_rates.csv & goalie_gsax.csv in the repo root)
python moneypuck_etl.py

# Custom output paths, history directory, and explicit seasons/stages
python moneypuck_etl.py \
  --team-output data/latest/team_rates.csv \
  --goalie-output data/latest/goalie_gsax.csv \
  --history-dir data/history \
  --seasons 2025 2024 \
  --stages regular playoffs
```

To run the ETL automatically before generating predictions, add `--refresh-moneypuck` when
calling `nhl_model3.py`. The inline run now refreshes `team_rates.csv` and `goalie_gsax.csv`.
You can further tune the inline run with `--moneypuck-history-dir`, `--moneypuck-stages`,
`--moneypuck-seasons`, `--moneypuck-dry-run`, `--fail-on-moneypuck-anomaly`,
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
  --goalie-gsax-path goalie_gsax.csv
```

4) Use environment and lineup inputs:

```bash
python nhl_model3.py --environment-path environment.json --lineup-path lineup_strength.csv
```

5) Reuse a cached ensemble without retraining:

```bash
python nhl_model3.py --use-saved-model --model-path data/cache/trained_model.joblib
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

Canonical closing lines (from odds history)
------------------------------------------

To make market-line training reproducible, build a canonical open/close totals+prices table from `odds_history.csv`:

```bash
python nhl_model3.py --build-closing-lines --odds-history-path odds_history.csv --closing-lines-path data/history/closing_lines.csv
```

When present, the model will prefer `data/history/closing_lines.csv` to populate `closing_total`, `closing_over_price`, and `closing_under_price` for training/backtests.

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

Goalie Context
--------------

Daily predictions ingest goalie-level context:

- Provide rolling goalie GSAx via `--goalie-gsax-path` (or URL). Expected starter quality is converted into `home_goalie_gsax` / `away_goalie_gsax` features and automatically debits/credits the total based on projected netminders.
- Goalie availability also surfaces in the dashboard (`lineup_info`) so you can see which starters are driving adjustments.

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
- Goal distribution modeling now stacks three layers: (1) classical Poisson regressors for home/away goals, (2) a neural "Poisson flow" MLP that learns non-linear intensities, and (3) a Gaussian-copula mixture simulator that blends the two regimes with dynamic weights/correlation derived from team and goalie covariates. The mixture powers the over/under probabilities and keeps pace with bookmaker-style conditional totals.

