import csv
import html
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st


APP_ROOT = Path(__file__).resolve().parent


APP_STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
  font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

[data-testid="stAppViewContainer"] {
  background: #f5f7fb;
}

[data-testid="stHeader"] {
  background: transparent;
}

[data-testid="stSidebar"] {
  background: #ffffff;
}

[data-testid="stAppViewContainer"] .main .block-container {
  max-width: 1100px;
  padding-top: 2rem;
  padding-bottom: 2.5rem;
}

h1 {
  font-weight: 800 !important;
  color: #0f172a;
  letter-spacing: -0.02em;
  margin-bottom: 0.25rem !important;
}

h3 {
  font-weight: 700 !important;
  color: #0f172a;
  margin-top: 1.5rem !important;
}

.meta-line {
  color: #475569;
  font-size: 0.98rem;
  font-weight: 500;
  margin-bottom: 0.35rem;
}

.meta-caption {
  color: #64748b;
  font-size: 0.86rem;
  margin-bottom: 1.15rem;
}

[data-testid="stButton"] > button {
  border: 1px solid #d0d8e5;
  border-radius: 10px;
  background: #ffffff;
  color: #0f172a;
  font-weight: 600;
  padding: 0.42rem 1rem;
}

[data-testid="stButton"] > button:hover {
  border-color: #9aa9c2;
  background: #f8fafc;
}

.summary-card {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  padding: 0.9rem 1rem;
  min-height: 92px;
  box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);
  margin-bottom: 0.65rem;
}

.summary-title {
  color: #64748b;
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  font-weight: 700;
  margin-bottom: 0.25rem;
}

.summary-record {
  color: #0f172a;
  font-size: 1.5rem;
  font-weight: 800;
  line-height: 1.25;
}

.summary-rate {
  color: #334155;
  font-size: 0.9rem;
  font-weight: 500;
  margin-top: 0.25rem;
}

.top-plays-table-wrap {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  overflow-x: auto;
  box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);
}

.top-plays-table {
  width: 100%;
  border-collapse: collapse;
  border-spacing: 0;
  min-width: 760px;
}

.top-plays-table th {
  text-align: left;
  font-size: 0.82rem;
  letter-spacing: 0.03em;
  text-transform: uppercase;
  color: #64748b;
  padding: 0.78rem 0.85rem;
  border-bottom: 1px solid #e2e8f0;
  background: #f8fafc;
  font-weight: 700;
}

.top-plays-table td {
  color: #0f172a;
  font-size: 0.94rem;
  padding: 0.78rem 0.85rem;
  border-bottom: 1px solid #eef2f7;
  vertical-align: middle;
}

.top-plays-table tr:last-child td {
  border-bottom: none;
}

.top-plays-table tr:nth-child(even) td {
  background: #fcfdff;
}

.pick-over {
  color: #15803d;
  font-weight: 800;
}

.pick-under {
  color: #b91c1c;
  font-weight: 800;
}
</style>
"""


def _parse_date_only(raw_value: str) -> Optional[datetime.date]:
    dt = _parse_logged_datetime(raw_value)
    if dt == datetime.min:
        return None
    return dt.date()


def _parse_logged_datetime(raw_value: str) -> datetime:
    raw = str(raw_value or "").strip()
    if not raw:
        return datetime.min
    for fmt in ("%Y-%m-%d %H:%M:%S", "%m/%d/%Y %H:%M"):
        try:
            return datetime.strptime(raw, fmt)
        except Exception:
            continue
    return datetime.min


def _safe_float(raw_value: str) -> Optional[float]:
    try:
        return float(raw_value)
    except Exception:
        return None


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
        # Fallback: show latest row per game_id when dates are malformed.
        by_game: Dict[str, Dict[str, str]] = {}
        for _, row in rows:
            gid = str(row.get("game_id") or "").strip()
            if gid:
                by_game[gid] = row
        return list(by_game.values()), None
    latest_rows = [row for dt, row in rows if dt == latest_dt]
    return latest_rows, latest_dt


def _graded_summary(log_path: Path) -> Dict[str, Optional[float]]:
    summary: Dict[str, Optional[float]] = {
        "yesterday_wins": 0,
        "yesterday_losses": 0,
        "yesterday_pushes": 0,
        "yesterday_decided": 0,
        "yesterday_win_rate": None,
        "ytd_wins": 0,
        "ytd_losses": 0,
        "ytd_pushes": 0,
        "ytd_decided": 0,
        "ytd_win_rate": None,
    }
    if not log_path.exists():
        return summary

    season_start_raw = str(os.getenv("NHL_SEASON_START", "2025-10-07")).strip() or "2025-10-07"
    try:
        season_start_date = datetime.strptime(season_start_raw, "%Y-%m-%d").date()
    except Exception:
        season_start_date = datetime(datetime.now().year, 1, 1).date()

    today_date = datetime.now().date()
    yesterday_date = today_date.fromordinal(today_date.toordinal() - 1)

    with log_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            side = str(row.get("side") or "").strip().upper()
            if side not in {"OVER", "UNDER"}:
                continue
            result = str(row.get("result") or "").strip().upper()
            if result not in {"WIN", "LOSS", "PUSH"}:
                continue
            row_date = _parse_date_only(row.get("date", ""))
            if row_date is None:
                continue

            if row_date >= season_start_date:
                if result == "WIN":
                    summary["ytd_wins"] += 1
                elif result == "LOSS":
                    summary["ytd_losses"] += 1
                else:
                    summary["ytd_pushes"] += 1

            if row_date == yesterday_date:
                if result == "WIN":
                    summary["yesterday_wins"] += 1
                elif result == "LOSS":
                    summary["yesterday_losses"] += 1
                else:
                    summary["yesterday_pushes"] += 1

    summary["ytd_decided"] = int(summary["ytd_wins"] + summary["ytd_losses"])
    summary["yesterday_decided"] = int(summary["yesterday_wins"] + summary["yesterday_losses"])
    if summary["ytd_decided"] > 0:
        summary["ytd_win_rate"] = float(summary["ytd_wins"] / summary["ytd_decided"])
    if summary["yesterday_decided"] > 0:
        summary["yesterday_win_rate"] = float(summary["yesterday_wins"] / summary["yesterday_decided"])
    return summary


def _latest_record(log_path: Path) -> Optional[Dict[str, float]]:
    if not log_path.exists():
        return None
    latest_by_game: Dict[str, Dict[str, str]] = {}
    latest_dt_by_game: Dict[str, datetime] = {}
    with log_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            gid = str(row.get("game_id") or "").strip()
            if not gid:
                continue
            dt = _parse_logged_datetime(row.get("date", ""))
            prev_dt = latest_dt_by_game.get(gid)
            if prev_dt is None or dt >= prev_dt:
                latest_dt_by_game[gid] = dt
                latest_by_game[gid] = row

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


def _format_record_and_rate(wins: int, losses: int, pushes: int, win_rate: Optional[float], empty_msg: str) -> Tuple[str, str]:
    record = f"{wins}-{losses}"
    if pushes > 0:
        record = f"{record}-{pushes}"
    rate_text = f"{(win_rate * 100.0):.1f}% win rate" if isinstance(win_rate, float) else empty_msg
    return record, rate_text


def _safe_text(raw_value: object) -> str:
    text = str(raw_value if raw_value is not None else "").strip()
    return html.escape(text) if text else "—"


def _render_summary_card(title: str, record_text: str, rate_text: str) -> None:
    st.markdown(
        (
            "<div class='summary-card'>"
            f"<div class='summary-title'>{html.escape(title)}</div>"
            f"<div class='summary-record'>{html.escape(record_text)}</div>"
            f"<div class='summary-rate'>{html.escape(rate_text)}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _render_predictions_table(rows: List[Dict[str, object]]) -> None:
    headers = ["Matchup", "Pick", "Line", "Pred", "Edge", "Conf%", "Price"]
    header_html = "".join(f"<th>{h}</th>" for h in headers)
    body_html_rows: List[str] = []
    for row in rows:
        pick = str(row.get("Pick") or "").strip().upper()
        pick_class = "pick-over" if pick == "OVER" else "pick-under" if pick == "UNDER" else ""
        pick_html = f"<span class='{pick_class}'>{html.escape(pick or '—')}</span>" if pick_class else _safe_text(pick or "—")
        body_html_rows.append(
            "<tr>"
            f"<td>{_safe_text(row.get('Matchup'))}</td>"
            f"<td>{pick_html}</td>"
            f"<td>{_safe_text(row.get('Line'))}</td>"
            f"<td>{_safe_text(row.get('Pred'))}</td>"
            f"<td>{_safe_text(row.get('Edge'))}</td>"
            f"<td>{_safe_text(row.get('Conf%'))}</td>"
            f"<td>{_safe_text(row.get('Price'))}</td>"
            "</tr>"
        )
    st.markdown(
        (
            "<div class='top-plays-table-wrap'>"
            "<table class='top-plays-table'>"
            f"<thead><tr>{header_html}</tr></thead>"
            f"<tbody>{''.join(body_html_rows)}</tbody>"
            "</table>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_public_app() -> None:
    st.set_page_config(page_title="NHL Over/Under Picks", layout="wide")
    st.markdown(APP_STYLES, unsafe_allow_html=True)
    st.title("NHL Over/Under Picks")

    if st.button("Refresh", type="secondary"):
        st.rerun()

    log_path = APP_ROOT / "bets_log.csv"
    run_rows, run_dt = _latest_run_rows(log_path)
    graded = _graded_summary(log_path)
    if run_dt is not None:
        slate_date = run_dt.strftime("%A, %b %d, %Y")
        last_updated = run_dt.isoformat()
    else:
        today = datetime.now()
        slate_date = today.strftime("%A, %b %d, %Y")
        last_updated = today.isoformat()
    st.markdown(f"<div class='meta-line'>Slate Date: {html.escape(slate_date)}</div>", unsafe_allow_html=True)
    st.markdown("### Top Plays")
    st.markdown(f"<div class='meta-caption'>Last updated: {html.escape(last_updated)}</div>", unsafe_allow_html=True)
    y_w = int(graded.get("yesterday_wins", 0) or 0)
    y_l = int(graded.get("yesterday_losses", 0) or 0)
    y_p = int(graded.get("yesterday_pushes", 0) or 0)
    y_wr = graded.get("yesterday_win_rate")
    y_text, y_delta = _format_record_and_rate(
        wins=y_w,
        losses=y_l,
        pushes=y_p,
        win_rate=y_wr,
        empty_msg="No graded bets yesterday",
    )

    s_w = int(graded.get("ytd_wins", 0) or 0)
    s_l = int(graded.get("ytd_losses", 0) or 0)
    s_p = int(graded.get("ytd_pushes", 0) or 0)
    s_wr = graded.get("ytd_win_rate")
    s_text, s_delta = _format_record_and_rate(
        wins=s_w,
        losses=s_l,
        pushes=s_p,
        win_rate=s_wr,
        empty_msg="No graded bets YTD",
    )

    c1, c2 = st.columns(2)
    with c1:
        _render_summary_card("Yesterday (graded)", y_text, y_delta)
    with c2:
        _render_summary_card("YTD (graded)", s_text, s_delta)

    if run_rows:
        table_rows: List[Dict[str, object]] = []
        for row in run_rows:
            side = str(row.get("side") or "").strip().upper()
            action = str(row.get("action") or "").strip().upper()
            line = _safe_float(row.get("line", ""))
            pred_total = _safe_float(row.get("pred_total", ""))
            edge = _safe_float(row.get("edge", ""))
            confidence = _safe_float(row.get("confidence", ""))
            price = _safe_float(row.get("price", ""))
            table_rows.append({
                "Matchup": str(row.get("matchup") or "").strip(),
                "Pick": side or "—",
                "Line": round(line, 2) if line is not None else None,
                "Pred": round(pred_total, 2) if pred_total is not None else None,
                "Edge": round(edge, 2) if edge is not None else None,
                "Conf%": round(confidence * 100.0, 1) if confidence is not None and confidence <= 1.0 else round(confidence, 1) if confidence is not None else None,
                "Price": int(price) if price is not None else None,
                "_action": action,
                "_edge_abs": abs(edge) if edge is not None else -1.0,
            })
        bet_rows = [r for r in table_rows if str(r.get("_action", "")).upper() == "BET"]
        if bet_rows:
            table_rows = bet_rows
        table_rows.sort(key=lambda r: r.get("_edge_abs", -1.0), reverse=True)
        for row in table_rows:
            row.pop("_edge_abs", None)
            row.pop("_action", None)

        _render_predictions_table(table_rows)
    else:
        st.info("No top plays available yet.")


if __name__ == "__main__":
    render_public_app()

