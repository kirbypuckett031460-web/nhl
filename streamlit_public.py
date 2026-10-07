import csv
import io
import json
import os
import subprocess
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import streamlit as st

try:
    import pandas as pd
except Exception:  # pragma: no cover - fallback only
    pd = None

try:
    from nhl_model.common import format_team_display, get_team_abbreviation, get_team_primary_color
except Exception:  # pragma: no cover - fallback only
    format_team_display = None
    get_team_abbreviation = None
    get_team_primary_color = None


APP_ROOT = Path(__file__).resolve().parent

DARK_MODE_CSS = """
<style>
[data-testid="stAppViewContainer"] {
  background-color: #0b1220;
  --primary-color: #ff4b4b !important;
  --accent-color: #ff4b4b !important;
}
[data-testid="stHeader"] { background: transparent; }
/* Hide Streamlit Cloud header links/actions (Fork + GitHub). */
#MainMenu,
[data-testid="stToolbar"],
[data-testid="stHeaderActionElements"],
[data-testid="stAppDeployButton"],
[data-testid="stStatusWidget"],
[data-testid="stHeader"] a {
  display: none !important;
}
[data-testid="stTabs"] [data-baseweb="tab-list"] {
  gap: 0.25rem;
  border-bottom: 1px solid #27324c;
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
  background-color: #ff4b4b !important;
  height: 2px !important;
}
[data-testid="stTabs"] [data-baseweb="tab"],
[data-testid="stTabs"] button[role="tab"] {
  font-size: 0.78rem !important;
  color: #f8fafc !important;
  background: transparent !important;
  border: none !important;
  border-bottom: 2px solid transparent !important;
  padding: 0.15rem 0.4rem 0.45rem 0.4rem !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] p {
  font-size: 0.78rem !important;
  color: inherit !important;
  font-weight: 600 !important;
}
[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"],
[data-testid="stTabs"] button[aria-selected="true"] {
  color: #ff4b4b !important;
  border-bottom-color: #ff4b4b !important;
  box-shadow: inset 0 -2px 0 #ff4b4b !important;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] p,
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] span,
[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] p {
  color: #ff4b4b !important;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"]::after,
[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"]::after {
  background-color: #ff4b4b !important;
  border-bottom-color: #ff4b4b !important;
}
[data-testid="stMetric"] {
  background-color: #111827;
  border: 1px solid #1f2937;
  border-radius: 10px;
  padding: 0.45rem 0.75rem;
}
</style>
"""


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = str(os.getenv(name, "")).strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on", "y"}


def _raw_data_url(path_name: str, branch_override: Optional[str] = None) -> Optional[str]:
    base_url = str(os.getenv("PUBLIC_DATA_BASE_URL", "")).strip().rstrip("/")
    if base_url:
        return f"{base_url}/{path_name.lstrip('/')}"
    repo = str(os.getenv("PUBLIC_DATA_REPO", "kirbypuckett031460-web/nhl")).strip().strip("/")
    branch = str(branch_override or os.getenv("PUBLIC_DATA_BRANCH", "main")).strip() or "main"
    branch_ref = quote(branch, safe="")
    if not repo:
        return None
    return f"https://raw.githubusercontent.com/{repo}/{branch_ref}/{path_name.lstrip('/')}"


def _current_git_branch() -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(APP_ROOT),
            text=True,
            capture_output=True,
            timeout=5,
        )
        if proc.returncode != 0:
            return None
        branch = str(proc.stdout or "").strip()
        if not branch or branch == "HEAD":
            return None
        return branch
    except Exception:
        return None


def _candidate_data_branches() -> List[str]:
    candidates: List[str] = []
    configured = str(os.getenv("PUBLIC_DATA_BRANCH", "main")).strip() or "main"
    candidates.append(configured)
    git_branch = _current_git_branch()
    if git_branch:
        candidates.append(git_branch)
    candidates.append("main")
    unique: List[str] = []
    for branch in candidates:
        val = str(branch or "").strip()
        if val and val not in unique:
            unique.append(val)
    return unique


def _fetch_remote_text(path_name: str, branch_override: Optional[str] = None) -> Optional[str]:
    if not _env_truthy("PUBLIC_APP_PREFER_REMOTE_DATA", default=True):
        return None
    url = _raw_data_url(path_name, branch_override=branch_override)
    if not url:
        return None
    nonce = int(time.time())
    connector = "&" if "?" in url else "?"
    final_url = f"{url}{connector}cb={nonce}"
    request = Request(
        final_url,
        headers={
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "User-Agent": "nhl-streamlit-public/1.0",
        },
    )
    try:
        with urlopen(request, timeout=10) as response:
            payload = response.read()
        return payload.decode("utf-8", errors="ignore")
    except Exception:
        return None


def _read_log_rows(log_path: Path, prefer_remote: bool = False, remote_branch: Optional[str] = None) -> List[Dict[str, str]]:
    if prefer_remote:
        remote_text = _fetch_remote_text(log_path.name, branch_override=remote_branch)
        if remote_text:
            try:
                reader = csv.DictReader(io.StringIO(remote_text))
                rows = [dict(r) for r in reader]
                if rows:
                    return rows
            except Exception:
                pass
    if not log_path.exists():
        return []
    with log_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(r) for r in reader]


def _parse_logged_datetime(raw_value: str) -> datetime:
    raw = str(raw_value or "").strip()
    if not raw:
        return datetime.min
    for fmt in ("%Y-%m-%d %H:%M:%S", "%m/%d/%Y %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(raw, fmt)
        except Exception:
            continue
    return datetime.min


def _safe_float(raw_value: object) -> Optional[float]:
    try:
        return float(raw_value)
    except Exception:
        return None


def _safe_int(raw_value: object) -> Optional[int]:
    try:
        return int(float(raw_value))
    except Exception:
        return None


def _team_display(value: object) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "—"
    if format_team_display is not None:
        try:
            display = str(format_team_display(raw) or "").strip()
            if display:
                return display
        except Exception:
            pass
    return raw


def _team_code(value: object) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    if get_team_abbreviation is not None:
        try:
            code = str(get_team_abbreviation(raw) or "").strip().upper()
            if code:
                return code
        except Exception:
            pass
    return raw.upper()


def _split_matchup(matchup: str) -> Tuple[str, str]:
    raw = str(matchup or "").strip()
    if "@" in raw:
        away, home = raw.split("@", 1)
        return away.strip(), home.strip()
    return raw, "—"


def _fmt_signed(value: Optional[float], places: int = 1, pct: bool = False) -> str:
    if value is None:
        return "—"
    suffix = "%" if pct else ""
    return f"{value:+.{places}f}{suffix}"


def _fmt_decimal(value: Optional[float], places: int = 1) -> str:
    if value is None:
        return "—"
    return f"{value:.{places}f}"


def _fmt_american(value: Optional[int]) -> str:
    if value is None:
        return "—"
    return f"{value:+d}" if value > 0 else str(value)


def _latest_run_rows(log_path: Path, prefer_remote: bool = False, remote_branch: Optional[str] = None) -> Tuple[List[Dict[str, str]], Optional[datetime]]:
    source_rows = _read_log_rows(log_path, prefer_remote=prefer_remote, remote_branch=remote_branch)
    if not source_rows:
        return [], None
    rows: List[Tuple[datetime, Dict[str, str]]] = []
    for row in source_rows:
        rows.append((_parse_logged_datetime(row.get("date", "")), row))
    if not rows:
        return [], None
    latest_dt = max(dt for dt, _ in rows)
    if latest_dt == datetime.min:
        return [r for _, r in rows], None
    return [r for dt, r in rows if dt == latest_dt], latest_dt


def _latest_record(log_path: Path) -> Optional[Dict[str, float]]:
    """Return latest per-game graded record summary for admin metric display."""
    source_rows = _read_log_rows(log_path, prefer_remote=False)
    if not source_rows:
        return None

    latest_by_game: Dict[str, Dict[str, str]] = {}
    latest_dt_by_game: Dict[str, datetime] = {}
    for row in source_rows:
        game_id = str(row.get("game_id") or "").strip()
        if not game_id:
            continue
        dt = _parse_logged_datetime(row.get("date", ""))
        prev_dt = latest_dt_by_game.get(game_id)
        if prev_dt is None or dt >= prev_dt:
            latest_dt_by_game[game_id] = dt
            latest_by_game[game_id] = row

    wins = losses = pushes = 0
    for row in latest_by_game.values():
        result = str(row.get("result") or "").strip().upper()
        if result == "WIN":
            wins += 1
        elif result == "LOSS":
            losses += 1
        elif result == "PUSH":
            pushes += 1

    decided = wins + losses
    if decided <= 0:
        return None
    return {
        "games": len(latest_by_game),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_rate": wins / decided,
    }


def _read_public_predictions(path: Path, prefer_remote: bool = False) -> Tuple[List[Dict[str, object]], Optional[datetime], Optional[str], Optional[date]]:
    payload_text: Optional[str] = None
    selected_branch: Optional[str] = None
    selected_slate_date: Optional[date] = None

    def _parse_slate_date(value: object) -> Optional[date]:
        raw = str(value or "").strip()
        if not raw:
            return None
        for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
            try:
                return datetime.strptime(raw, fmt).date()
            except Exception:
                continue
        return None

    if prefer_remote:
        best_games: Optional[List[Dict[str, object]]] = None
        best_dt: Optional[datetime] = None
        best_branch: Optional[str] = None
        best_slate_date: Optional[date] = None
        best_key: Optional[Tuple[object, ...]] = None
        for branch in _candidate_data_branches():
            remote_text = _fetch_remote_text(path.name, branch_override=branch)
            if not remote_text:
                continue
            try:
                payload = json.loads(remote_text)
            except Exception:
                continue
            games_raw = payload.get("games")
            if not isinstance(games_raw, list):
                continue
            games = [g for g in games_raw if isinstance(g, dict)]
            generated_raw = str(payload.get("generated_at") or "").strip()
            generated_dt = _parse_logged_datetime(generated_raw) if generated_raw else None
            if generated_dt == datetime.min:
                generated_dt = None
            slate_dt = _parse_slate_date(payload.get("slate_date"))
            def _is_synthetic_game(game: Dict[str, object]) -> bool:
                gid = str(game.get("game_id") or "").strip().upper()
                if gid.startswith(("DEMO_", "OFFLINE_", "SAMPLE_")):
                    return True
                away = str(game.get("away_abbrev") or game.get("away_team") or "").strip().upper()
                home = str(game.get("home_abbrev") or game.get("home_team") or "").strip().upper()
                generic = {"AWAY", "HOME", "TBD", "—", ""}
                return away in generic or home in generic

            synthetic_count = sum(1 for g in games if _is_synthetic_game(g))
            real_count = max(0, len(games) - synthetic_count)
            totals_market_count = 0
            moneyline_market_count = 0
            for g in games:
                if _safe_float(g.get("totals_line")) is not None:
                    totals_market_count += 1
                if _safe_int(g.get("moneyline_market_odds")) is not None:
                    moneyline_market_count += 1
                elif any(
                    _safe_int(g.get(k)) is not None
                    for k in ("home_moneyline_odds", "away_moneyline_odds", "consensus_home_moneyline", "consensus_away_moneyline")
                ):
                    moneyline_market_count += 1
            # Prefer payloads with real scheduled games over synthetic/demo rows,
            # then pick the freshest generated_at among those.
            key = (
                1 if real_count > 0 else 0,
                real_count,
                totals_market_count,
                moneyline_market_count,
                1 if generated_dt is not None else 0,
                generated_dt or datetime.min,
                -synthetic_count,
                len(games),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_games = games
                best_dt = generated_dt
                best_branch = branch
                best_slate_date = slate_dt
        if best_games is not None:
            return best_games, best_dt, best_branch, best_slate_date
    if not payload_text:
        if not path.exists():
            return [], None, None, None
        try:
            payload_text = path.read_text(encoding="utf-8")
        except Exception:
            return [], None, None, None
    try:
        payload = json.loads(payload_text)
    except Exception:
        return [], None, None, None
    games = payload.get("games")
    if not isinstance(games, list):
        return [], None, None, None
    generated_raw = str(payload.get("generated_at") or "").strip()
    generated_dt = _parse_logged_datetime(generated_raw) if generated_raw else None
    if generated_dt == datetime.min:
        generated_dt = None
    selected_slate_date = _parse_slate_date(payload.get("slate_date"))
    return [g for g in games if isinstance(g, dict)], generated_dt, selected_branch, selected_slate_date


def _compute_record_blocks(
    log_path: Path,
    prefer_remote: bool = False,
    remote_branch: Optional[str] = None,
    reference_date: Optional[date] = None,
) -> Dict[str, Tuple[str, str]]:
    ref_date = reference_date or datetime.now().date()
    season_start_raw = str(os.getenv("NHL_SEASON_START", "")).strip()
    if not season_start_raw:
        current_year_start = date(ref_date.year, 9, 29)
        # During late off-season (Jul-Sep before opening day), track against the upcoming season start.
        if ref_date.month >= 7 and ref_date < current_year_start:
            season_start_raw = current_year_start.strftime("%Y-%m-%d")
        elif ref_date >= current_year_start:
            season_start_raw = current_year_start.strftime("%Y-%m-%d")
        else:
            season_start_raw = date(ref_date.year - 1, 9, 29).strftime("%Y-%m-%d")
    try:
        season_start = datetime.strptime(season_start_raw, "%Y-%m-%d").date()
    except Exception:
        season_start = datetime.now().date().replace(month=1, day=1)
    yesterday = ref_date - timedelta(days=1)
    blocks = {
        "ml_prev_day": [0, 0],
        "ml_ytd": [0, 0],
        "tot_prev_day": [0, 0],
        "tot_ytd": [0, 0],
    }
    source_rows = _read_log_rows(log_path, prefer_remote=prefer_remote, remote_branch=remote_branch)
    # On opening day (and pre-season), show all records as 0-0.
    if not source_rows or ref_date <= season_start:
        return {
            "ml_prev_day": ("0-0", "0.0%"),
            "ml_ytd": ("0-0", "0.0%"),
            "tot_prev_day": ("0-0", "0.0%"),
            "tot_ytd": ("0-0", "0.0%"),
        }
    # Deduplicate repeated runs: keep latest graded row per (market bucket, game).
    latest_by_market_game: Dict[str, Tuple[int, datetime, date, str, str]] = {}
    for idx_row, row in enumerate(source_rows):
        result = str(row.get("result") or "").strip().upper()
        if result not in {"WIN", "LOSS"}:
            continue
        dt = _parse_logged_datetime(row.get("date", ""))
        if dt == datetime.min:
            continue
        d = dt.date()
        action = str(row.get("action") or "").strip().upper()
        side = str(row.get("side") or "").strip().upper()
        is_ml = ("ML" in action) or ("ML" in side) or (side in {"HOME", "AWAY", "HML", "AML"})
        bucket = "ml" if is_ml else ("tot" if side in {"OVER", "UNDER"} else "")
        if not bucket:
            continue
        gid = str(row.get("game_id") or "").strip()
        if gid:
            key = f"{bucket}|gid:{gid}"
        else:
            matchup = str(row.get("matchup") or "").strip().upper()
            key = f"{bucket}|matchup:{matchup}|date:{d.isoformat()}"
        prev = latest_by_market_game.get(key)
        if prev is None or idx_row >= prev[0]:
            latest_by_market_game[key] = (idx_row, dt, d, bucket, result)

    for _, (_, _, d, bucket, result) in latest_by_market_game.items():
        idx = 0 if result == "WIN" else 1
        if d >= season_start:
            blocks[f"{bucket}_ytd"][idx] += 1
        if d == yesterday:
            blocks[f"{bucket}_prev_day"][idx] += 1

    def _fmt(block: List[int]) -> Tuple[str, str]:
        wins, losses = int(block[0]), int(block[1])
        decided = wins + losses
        pct = (wins / decided * 100.0) if decided > 0 else 0.0
        return f"{wins}-{losses}", f"{pct:.1f}%"

    return {
        "ml_prev_day": _fmt(blocks["ml_prev_day"]),
        "ml_ytd": _fmt(blocks["ml_ytd"]),
        "tot_prev_day": _fmt(blocks["tot_prev_day"]),
        "tot_ytd": _fmt(blocks["tot_ytd"]),
    }


def _build_tables_from_public(games: List[Dict[str, object]]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    moneyline_rows: List[Dict[str, object]] = []
    totals_rows: List[Dict[str, object]] = []
    def _resolve_moneyline_market(game: Dict[str, object], side_hint: str) -> Optional[int]:
        direct = _safe_int(game.get("moneyline_market_odds"))
        if direct is not None:
            return direct
        side = str(side_hint or "").strip().lower()
        if side == "home":
            for key in ("home_moneyline_odds", "consensus_home_moneyline"):
                val = _safe_int(game.get(key))
                if val is not None:
                    return val
        if side == "away":
            for key in ("away_moneyline_odds", "consensus_away_moneyline"):
                val = _safe_int(game.get(key))
                if val is not None:
                    return val
        for key in ("home_moneyline_odds", "away_moneyline_odds", "consensus_home_moneyline", "consensus_away_moneyline"):
            val = _safe_int(game.get(key))
            if val is not None:
                return val
        return None

    for game in games:
        away_raw = game.get("away_team") or game.get("away_abbrev") or ""
        home_raw = game.get("home_team") or game.get("home_abbrev") or ""
        away = _team_display(away_raw)
        home = _team_display(home_raw)
        away_code = _team_code(away_raw)
        home_code = _team_code(home_raw)
        game_time = str(game.get("game_time_et") or "").strip() or "—"

        totals_edge = _safe_float(game.get("totals_edge"))
        totals_conf = _safe_float(game.get("totals_confidence_pct"))
        totals_rows.append({
            "Game Time (ET)": game_time,
            "Away": away,
            "Home": home,
            "Mkt": _fmt_decimal(_safe_float(game.get("totals_line")), places=1),
            "Fair": _fmt_decimal(_safe_float(game.get("totals_fair")), places=1),
            "Pick": str(game.get("totals_pick") or "—").upper(),
            "Edge": _fmt_signed(totals_edge, places=2, pct=False),
            "Confidence": _fmt_signed(totals_conf, places=1, pct=True).replace("+", ""),
            "_edge_abs": abs(totals_edge) if totals_edge is not None else -1.0,
        })

        ml_pick_raw = str(game.get("moneyline_pick_team") or "").strip()
        ml_pick = _team_display(ml_pick_raw) if ml_pick_raw else ""
        if not ml_pick:
            side_hint = str(game.get("moneyline_pick_side") or "").strip().lower()
            ml_pick = home if side_hint == "home" else away if side_hint == "away" else home
        else:
            side_hint = str(game.get("moneyline_pick_side") or "").strip().lower()
            if not side_hint:
                pick_code = _team_code(ml_pick_raw or ml_pick)
                if pick_code and pick_code == away_code:
                    side_hint = "away"
                elif pick_code and pick_code == home_code:
                    side_hint = "home"
        market_ml = _resolve_moneyline_market(game, side_hint)
        ml_edge = _safe_float(game.get("moneyline_edge"))
        ml_conf = _safe_float(game.get("moneyline_confidence_pct"))
        moneyline_rows.append({
            "Game Time (ET)": game_time,
            "Away": away,
            "Home": home,
            "Mkt": _fmt_american(market_ml),
            "Fair": _fmt_american(_safe_int(game.get("moneyline_fair_odds"))),
            "Pick": ml_pick,
            "Edge": _fmt_signed(ml_edge * 100.0 if ml_edge is not None else None, places=1, pct=True),
            "Confidence": _fmt_signed(ml_conf, places=1, pct=True).replace("+", ""),
            "_edge_abs": abs(ml_edge) if ml_edge is not None else -1.0,
        })

    totals_rows.sort(key=lambda r: str(r.get("Game Time (ET)") or ""))
    moneyline_rows.sort(key=lambda r: str(r.get("Game Time (ET)") or ""))
    return moneyline_rows, totals_rows


def _build_totals_from_log_rows(run_rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for row in run_rows:
        away_raw, home_raw = _split_matchup(row.get("matchup", ""))
        away = _team_display(away_raw) if format_team_display is not None else away_raw
        home = _team_display(home_raw) if format_team_display is not None else home_raw
        line = _safe_float(row.get("line"))
        fair = _safe_float(row.get("pred_total"))
        edge = _safe_float(row.get("edge"))
        confidence = _safe_float(row.get("confidence"))
        if confidence is not None and confidence <= 1.0:
            confidence *= 100.0
        rows.append({
            "Game Time (ET)": "—",
            "Away": away,
            "Home": home,
            "Mkt": _fmt_decimal(line, places=1),
            "Fair": _fmt_decimal(fair, places=1),
            "Pick": str(row.get("side") or "—").strip().upper() or "—",
            "Edge": _fmt_signed(edge, places=2, pct=False),
            "Confidence": _fmt_signed(confidence, places=1, pct=True).replace("+", ""),
            "_edge_abs": abs(edge) if edge is not None else -1.0,
        })
    rows.sort(key=lambda r: str(r.get("Away") or ""))
    return rows


def _contrast_text_color(hex_color: str) -> str:
    color = str(hex_color or "").strip().lstrip("#")
    if len(color) != 6:
        return "#f8fafc"
    try:
        r = int(color[0:2], 16)
        g = int(color[2:4], 16)
        b = int(color[4:6], 16)
    except Exception:
        return "#f8fafc"
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
    return "#0b1220" if luminance > 0.55 else "#f8fafc"


def _style_pick_cell(val: object) -> str:
    txt = str(val or "").strip().upper()
    if txt == "OVER":
        return "background-color: #166534; color: #dcfce7; font-weight: 700; text-align: center;"
    if txt == "UNDER":
        return "background-color: #991b1b; color: #fee2e2; font-weight: 700; text-align: center;"
    if get_team_primary_color is not None and txt not in {"", "—", "NO BET"}:
        team_color = str(get_team_primary_color(txt) or "#1d4ed8").strip()
        return f"background-color: {team_color}; color: {_contrast_text_color(team_color)}; font-weight: 700; text-align: center;"
    return "text-align: center;"


def _style_edge_cell(val: object) -> str:
    text = str(val or "").strip().replace("%", "")
    try:
        num = float(text)
    except Exception:
        return ""
    intensity = min(0.8, 0.22 + min(abs(num), 12.0) * 0.05)
    if num >= 0:
        return f"background-color: rgba(16, 185, 129, {intensity:.3f}); color: #ecfeff;"
    return f"background-color: rgba(244, 63, 94, {intensity:.3f}); color: #ffe4e6;"


def _style_conf_cell(val: object) -> str:
    text = str(val or "").strip().replace("%", "")
    try:
        num = float(text)
    except Exception:
        return ""
    centered = max(-1.0, min(1.0, (num - 50.0) / 50.0))
    intensity = 0.2 + abs(centered) * 0.6
    if centered >= 0:
        return f"background-color: rgba(20, 184, 166, {intensity:.3f}); color: #ecfeff;"
    return f"background-color: rgba(236, 72, 153, {intensity:.3f}); color: #fdf2f8;"


def _format_last_updated_et(dt: datetime) -> str:
    if not isinstance(dt, datetime):
        return "—"
    eastern = ZoneInfo("America/New_York")
    try:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        dt_et = dt.astimezone(eastern)
    except Exception:
        dt_et = dt
    return dt_et.strftime("%Y-%m-%d %I:%M:%S %p ET")


def _render_table(rows: List[Dict[str, object]], title: str = "", subtitle: str = "") -> None:
    if title:
        st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)
    if not rows:
        st.info("No rows available.")
        return
    clean_rows = [{k: v for k, v in row.items() if not str(k).startswith("_")} for row in rows]
    # Expand table height to fit all rows so users can view the full slate
    # without scrolling inside the dataframe widget.
    table_height = max(180, min(1800, 42 + len(clean_rows) * 34))
    if pd is None:
        st.dataframe(clean_rows, use_container_width=True, hide_index=True, height=table_height)
        return
    frame = pd.DataFrame(clean_rows)
    width_map = {
        "Game Time (ET)": "110px",
        "Away": "230px",
        "Home": "230px",
        "Mkt": "64px",
        "Fair": "64px",
        "Pick": "220px",
        "Edge": "74px",
        "Confidence": "90px",
    }
    table_styles = [
        {
            "selector": "thead th",
            "props": [
                ("background-color", "#2a334a"),
                ("color", "#dbe7ff"),
                ("font-size", "11px"),
                ("font-weight", "700"),
                ("padding", "3px 6px"),
                ("text-align", "center"),
                ("border", "1px solid #32415f"),
                ("white-space", "nowrap"),
            ],
        },
        {
            "selector": "tbody td",
            "props": [
                ("background-color", "#0f1a2e"),
                ("color", "#f1f5f9"),
                ("font-size", "10.5px"),
                ("font-weight", "600"),
                ("padding", "2px 6px"),
                ("line-height", "1.12"),
                ("text-align", "center"),
                ("border", "1px solid #23314a"),
                ("white-space", "nowrap"),
            ],
        },
    ]
    for col_idx, col_name in enumerate(frame.columns):
        width = width_map.get(str(col_name), "100px")
        table_styles.append({
            "selector": f".col{col_idx}",
            "props": [("min-width", width), ("max-width", width), ("width", width)],
        })

    try:
        styled = frame.style.map(_style_pick_cell, subset=["Pick"])
        styled = styled.map(_style_edge_cell, subset=["Edge"])
        styled = styled.map(_style_conf_cell, subset=["Confidence"])
    except Exception:
        styled = frame.style.applymap(_style_pick_cell, subset=["Pick"])
        styled = styled.applymap(_style_edge_cell, subset=["Edge"])
        styled = styled.applymap(_style_conf_cell, subset=["Confidence"])
    styled = styled.set_table_styles(table_styles)
    styled = styled.set_properties(**{"text-align": "center"})
    st.dataframe(styled, use_container_width=True, hide_index=True, height=table_height)


def render_public_app() -> None:
    st.set_page_config(page_title="NHL Picks", layout="wide")
    st.markdown(DARK_MODE_CSS, unsafe_allow_html=True)
    st.title("NHL Picks")

    if st.button("Refresh", type="secondary"):
        try:
            st.cache_data.clear()
        except Exception:
            pass
        st.rerun()

    log_path = APP_ROOT / "bets_log.csv"
    board_path = APP_ROOT / "public_predictions.json"
    prefer_remote = _env_truthy("PUBLIC_APP_PREFER_REMOTE_DATA", default=True)
    board_games, board_dt, board_branch, board_slate_date = _read_public_predictions(board_path, prefer_remote=prefer_remote)
    run_rows, run_dt = _latest_run_rows(log_path, prefer_remote=prefer_remote, remote_branch=board_branch)
    # "Yesterday" metrics should reflect the real calendar day in schedule timezone,
    # not the selected slate date (which can be intentionally backdated).
    schedule_tz = str(os.getenv("SCHEDULE_TZ", "US/Eastern") or "US/Eastern").strip() or "US/Eastern"
    try:
        metrics_ref_date = datetime.now(ZoneInfo(schedule_tz)).date()
    except Exception:
        metrics_ref_date = datetime.now().date()
    metrics = _compute_record_blocks(
        log_path,
        prefer_remote=prefer_remote,
        remote_branch=board_branch,
        reference_date=metrics_ref_date,
    )

    ml_rows, ou_rows = _build_tables_from_public(board_games)
    if not ou_rows and run_rows:
        ou_rows = _build_totals_from_log_rows(run_rows)

    shown_dt = board_dt or run_dt or datetime.now()
    last_updated_et = _format_last_updated_et(shown_dt)
    if board_slate_date is not None:
        slate_dt_display = datetime.combine(board_slate_date, datetime.min.time())
    else:
        slate_dt_display = shown_dt
    st.write(f"Slate Date: {slate_dt_display.strftime('%A, %b %d, %Y')}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Moneyline Yesterday", metrics["ml_prev_day"][0])
    c1.caption(metrics["ml_prev_day"][1])
    c2.metric("Moneyline YTD", metrics["ml_ytd"][0])
    c2.caption(metrics["ml_ytd"][1])
    c3.metric("Totals Yesterday", metrics["tot_prev_day"][0])
    c3.caption(metrics["tot_prev_day"][1])
    c4.metric("Totals YTD", metrics["tot_ytd"][0])
    c4.caption(metrics["tot_ytd"][1])

    tab_ml, tab_ou = st.tabs(["Moneyline Picks", "Over/Under Picks"])
    with tab_ml:
        _render_table(ml_rows)
        top_ml = sorted(ml_rows, key=lambda r: float(r.get("_edge_abs", -1.0)), reverse=True)[:5]
        _render_table(top_ml, "Top Plays")
    with tab_ou:
        _render_table(ou_rows)
        top_ou = sorted(ou_rows, key=lambda r: float(r.get("_edge_abs", -1.0)), reverse=True)[:5]
        _render_table(top_ou, "Top Plays")

    st.caption(f"Last updated: {last_updated_et}")


if __name__ == "__main__":
    render_public_app()

