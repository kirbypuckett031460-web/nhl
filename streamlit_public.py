import csv
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st

try:
    import pandas as pd
except Exception:  # pragma: no cover - fallback only
    pd = None

try:
    from nhl_model.common import get_team_primary_color
except Exception:  # pragma: no cover - fallback only
    get_team_primary_color = None


APP_ROOT = Path(__file__).resolve().parent

DARK_MODE_CSS = """
<style>
[data-testid="stAppViewContainer"] { background-color: #0b1220; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stMetric"] {
  background-color: #111827;
  border: 1px solid #1f2937;
  border-radius: 10px;
  padding: 0.45rem 0.75rem;
}
</style>
"""


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


def _latest_run_rows(log_path: Path) -> Tuple[List[Dict[str, str]], Optional[datetime]]:
    if not log_path.exists():
        return [], None
    rows: List[Tuple[datetime, Dict[str, str]]] = []
    with log_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append((_parse_logged_datetime(row.get("date", "")), row))
    if not rows:
        return [], None
    latest_dt = max(dt for dt, _ in rows)
    if latest_dt == datetime.min:
        return [r for _, r in rows], None
    return [r for dt, r in rows if dt == latest_dt], latest_dt


def _read_public_predictions(path: Path) -> Tuple[List[Dict[str, object]], Optional[datetime]]:
    if not path.exists():
        return [], None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return [], None
    games = payload.get("games")
    if not isinstance(games, list):
        return [], None
    generated_raw = str(payload.get("generated_at") or "").strip()
    generated_dt = _parse_logged_datetime(generated_raw) if generated_raw else None
    if generated_dt == datetime.min:
        generated_dt = None
    return [g for g in games if isinstance(g, dict)], generated_dt


def _compute_record_blocks(log_path: Path) -> Dict[str, Tuple[str, str]]:
    season_start_raw = str(os.getenv("NHL_SEASON_START", "2025-10-07")).strip() or "2025-10-07"
    try:
        season_start = datetime.strptime(season_start_raw, "%Y-%m-%d").date()
    except Exception:
        season_start = datetime.now().date().replace(month=1, day=1)
    today = datetime.now().date()
    prev_week_start = today - timedelta(days=7)
    blocks = {
        "ml_prev": [0, 0],
        "ml_ytd": [0, 0],
        "tot_prev": [0, 0],
        "tot_ytd": [0, 0],
    }
    if not log_path.exists():
        return {
            "ml_prev": ("0-0", "+0.0%"),
            "ml_ytd": ("0-0", "+0.0%"),
            "tot_prev": ("0-0", "+0.0%"),
            "tot_ytd": ("0-0", "+0.0%"),
        }
    with log_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
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
            idx = 0 if result == "WIN" else 1
            if d >= season_start:
                blocks[f"{bucket}_ytd"][idx] += 1
            if prev_week_start <= d < today:
                blocks[f"{bucket}_prev"][idx] += 1

    def _fmt(block: List[int]) -> Tuple[str, str]:
        wins, losses = int(block[0]), int(block[1])
        decided = wins + losses
        pct = (wins / decided * 100.0) if decided > 0 else 0.0
        return f"{wins}-{losses}", f"{pct:+.1f}%"

    return {
        "ml_prev": _fmt(blocks["ml_prev"]),
        "ml_ytd": _fmt(blocks["ml_ytd"]),
        "tot_prev": _fmt(blocks["tot_prev"]),
        "tot_ytd": _fmt(blocks["tot_ytd"]),
    }


def _build_tables_from_public(games: List[Dict[str, object]]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    moneyline_rows: List[Dict[str, object]] = []
    totals_rows: List[Dict[str, object]] = []
    for game in games:
        away = str(game.get("away_abbrev") or game.get("away_team") or "").strip() or "—"
        home = str(game.get("home_abbrev") or game.get("home_team") or "").strip() or "—"
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

        ml_pick = str(game.get("moneyline_pick_team") or "").strip().upper()
        if not ml_pick:
            side_hint = str(game.get("moneyline_pick_side") or "").strip().lower()
            ml_pick = home if side_hint == "home" else away if side_hint == "away" else home
        ml_edge = _safe_float(game.get("moneyline_edge"))
        ml_conf = _safe_float(game.get("moneyline_confidence_pct"))
        moneyline_rows.append({
            "Game Time (ET)": game_time,
            "Away": away,
            "Home": home,
            "Mkt": _fmt_american(_safe_int(game.get("moneyline_market_odds"))),
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
        away, home = _split_matchup(row.get("matchup", ""))
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


def _render_table(rows: List[Dict[str, object]], title: str) -> None:
    st.markdown(f"### {title}")
    if not rows:
        st.info("No rows available.")
        return
    clean_rows = [{k: v for k, v in row.items() if not str(k).startswith("_")} for row in rows]
    if pd is None:
        st.dataframe(clean_rows, use_container_width=True, hide_index=True)
        return
    frame = pd.DataFrame(clean_rows)
    try:
        styled = frame.style.map(_style_pick_cell, subset=["Pick"])
        styled = styled.map(_style_edge_cell, subset=["Edge"])
        styled = styled.map(_style_conf_cell, subset=["Confidence"])
    except Exception:
        styled = frame.style.applymap(_style_pick_cell, subset=["Pick"])
        styled = styled.applymap(_style_edge_cell, subset=["Edge"])
        styled = styled.applymap(_style_conf_cell, subset=["Confidence"])
    st.dataframe(styled, use_container_width=True, hide_index=True)


def render_public_app() -> None:
    st.set_page_config(page_title="NHL Picks", layout="wide")
    st.markdown(DARK_MODE_CSS, unsafe_allow_html=True)
    st.title("NHL Picks")

    if st.button("Refresh", type="secondary"):
        st.rerun()

    log_path = APP_ROOT / "bets_log.csv"
    board_path = APP_ROOT / "public_predictions.json"
    board_games, board_dt = _read_public_predictions(board_path)
    run_rows, run_dt = _latest_run_rows(log_path)
    metrics = _compute_record_blocks(log_path)

    ml_rows, ou_rows = _build_tables_from_public(board_games)
    if not ou_rows and run_rows:
        ou_rows = _build_totals_from_log_rows(run_rows)

    shown_dt = board_dt or run_dt or datetime.now()
    st.write(f"Slate Date: {shown_dt.strftime('%A, %b %d, %Y')}")
    st.caption(f"Last updated: {shown_dt.isoformat()}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Moneyline Prev Week", metrics["ml_prev"][0], metrics["ml_prev"][1])
    c2.metric("Moneyline YTD", metrics["ml_ytd"][0], metrics["ml_ytd"][1])
    c3.metric("Totals Prev Week", metrics["tot_prev"][0], metrics["tot_prev"][1])
    c4.metric("Totals YTD", metrics["tot_ytd"][0], metrics["tot_ytd"][1])

    week_label = f"Week {shown_dt.isocalendar()[1]}"
    st.selectbox("Week", options=[week_label], index=0, disabled=True)
    st.caption(f"Slate: {week_label}")

    board_view = st.radio(
        "Board View",
        options=["Moneyline Picks", "Over/Under Picks"],
        index=0,
        horizontal=True,
        label_visibility="collapsed",
    )
    if board_view == "Moneyline Picks":
        _render_table(ml_rows, "Full Slate")
        top_ml = sorted(ml_rows, key=lambda r: float(r.get("_edge_abs", -1.0)), reverse=True)[:5]
        _render_table(top_ml, "Top Plays")
    else:
        _render_table(ou_rows, "Full Slate")
        top_ou = sorted(ou_rows, key=lambda r: float(r.get("_edge_abs", -1.0)), reverse=True)[:5]
        _render_table(top_ou, "Top Plays")


if __name__ == "__main__":
    render_public_app()

