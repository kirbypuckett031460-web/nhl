import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st
try:
    import pandas as pd
except Exception:  # pragma: no cover - fallback only
    pd = None


APP_ROOT = Path(__file__).resolve().parent


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


def render_public_app() -> None:
    st.set_page_config(page_title="NHL Over/Under Picks", layout="wide")
    st.title("NHL Over/Under Picks")

    if st.button("Refresh", type="secondary"):
        st.rerun()

    log_path = APP_ROOT / "bets_log.csv"
    run_rows, run_dt = _latest_run_rows(log_path)
    if run_dt is not None:
        slate_date = run_dt.strftime("%A, %b %d, %Y")
        last_updated = run_dt.isoformat()
    else:
        today = datetime.now()
        slate_date = today.strftime("%A, %b %d, %Y")
        last_updated = today.isoformat()
    st.write(f"Slate Date: {slate_date}")
    st.markdown("### Top Plays")
    st.caption(f"Last updated: {last_updated}")

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

        if pd is not None:
            frame = pd.DataFrame(table_rows)

            def _pick_style(val: object) -> str:
                txt = str(val or "").strip().upper()
                if txt == "OVER":
                    return "color: #1f8b4c; font-weight: 700;"
                if txt == "UNDER":
                    return "color: #c0392b; font-weight: 700;"
                return ""

            styled = frame.style.applymap(_pick_style, subset=["Pick"])
            st.dataframe(styled, use_container_width=True, hide_index=True)
        else:
            st.dataframe(table_rows, use_container_width=True, hide_index=True)
    else:
        st.info("No top plays available yet.")


if __name__ == "__main__":
    render_public_app()

