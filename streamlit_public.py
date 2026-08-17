import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st
try:
    import pandas as pd
except Exception:  # pragma: no cover - fallback only
    pd = None


APP_ROOT = Path(__file__).resolve().parent

DARK_MODE_CSS = """
<style>
[data-testid="stAppViewContainer"] {
  background-color: #0b1220;
}

[data-testid="stHeader"] {
  background: transparent;
}

[data-testid="stMetric"] {
  background-color: #111827;
  border: 1px solid #1f2937;
  border-radius: 10px;
  padding: 0.5rem 0.75rem;
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


def _split_matchup(matchup: str) -> Tuple[str, str]:
    raw = str(matchup or "").strip()
    if "@" in raw:
        away, home = raw.split("@", 1)
        return away.strip(), home.strip()
    if " at " in raw.lower():
        chunks = raw.split(" at ")
        if len(chunks) == 2:
            return chunks[0].strip(), chunks[1].strip()
    return raw, "—"


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


def _style_pick_cell(val: object) -> str:
    txt = str(val or "").strip().upper()
    if txt == "OVER":
        return "background-color: #166534; color: #dcfce7; font-weight: 700; text-align: center;"
    if txt == "UNDER":
        return "background-color: #991b1b; color: #fee2e2; font-weight: 700; text-align: center;"
    return "text-align: center;"


def _style_edge_cell(val: object) -> str:
    try:
        num = float(val)
    except Exception:
        return ""
    intensity = min(0.75, 0.22 + min(abs(num), 3.0) * 0.16)
    if num >= 0:
        return f"background-color: rgba(16, 185, 129, {intensity:.3f}); color: #ecfeff;"
    return f"background-color: rgba(244, 63, 94, {intensity:.3f}); color: #ffe4e6;"


def _style_conf_cell(val: object) -> str:
    try:
        num = float(val)
    except Exception:
        return ""
    centered = max(-1.0, min(1.0, (num - 50.0) / 50.0))
    intensity = 0.2 + abs(centered) * 0.6
    if centered >= 0:
        return f"background-color: rgba(20, 184, 166, {intensity:.3f}); color: #ecfeff;"
    return f"background-color: rgba(236, 72, 153, {intensity:.3f}); color: #fdf2f8;"


def render_public_app() -> None:
    st.set_page_config(page_title="NHL Over/Under Picks", layout="wide")
    st.markdown(DARK_MODE_CSS, unsafe_allow_html=True)
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
    st.write(f"Slate Date: {slate_date}")
    st.caption(f"Last updated: {last_updated}")
    y_w = int(graded.get("yesterday_wins", 0) or 0)
    y_l = int(graded.get("yesterday_losses", 0) or 0)
    y_p = int(graded.get("yesterday_pushes", 0) or 0)
    y_wr = graded.get("yesterday_win_rate")
    y_text = f"{y_w}-{y_l}"
    if y_p > 0:
        y_text = f"{y_text}-{y_p}"
    y_delta = f"{(y_wr * 100.0):.1f}% win rate" if isinstance(y_wr, float) else "No graded bets yesterday"

    s_w = int(graded.get("ytd_wins", 0) or 0)
    s_l = int(graded.get("ytd_losses", 0) or 0)
    s_p = int(graded.get("ytd_pushes", 0) or 0)
    s_wr = graded.get("ytd_win_rate")
    s_text = f"{s_w}-{s_l}"
    if s_p > 0:
        s_text = f"{s_text}-{s_p}"
    s_delta = f"{(s_wr * 100.0):.1f}% win rate" if isinstance(s_wr, float) else "No graded bets YTD"

    st.caption(f"Yesterday (graded): {y_text} ({y_delta})   |   YTD (graded): {s_text} ({s_delta})")

    if run_rows:
        table_rows: List[Dict[str, object]] = []
        for row in run_rows:
            away, home = _split_matchup(row.get("matchup", ""))
            side = str(row.get("side") or "").strip().upper()
            action = str(row.get("action") or "").strip().upper()
            line = _safe_float(row.get("line", ""))
            pred_total = _safe_float(row.get("pred_total", ""))
            edge = _safe_float(row.get("edge", ""))
            confidence = _safe_float(row.get("confidence", ""))
            conf_pct = None
            if confidence is not None:
                conf_pct = confidence * 100.0 if confidence <= 1.0 else confidence
            table_rows.append({
                "Away": away,
                "Home": home,
                "Line": round(line, 2) if line is not None else None,
                "Model": round(pred_total, 2) if pred_total is not None else None,
                "Pick": side or "—",
                "Edge": round(edge, 2) if edge is not None else None,
                "Confidence": round(conf_pct, 1) if conf_pct is not None else None,
                "_action": action,
                "_edge_abs": abs(edge) if edge is not None else -1.0,
            })
        table_rows.sort(key=lambda r: str(r.get("Away") or ""))

        if pd is not None:
            full_rows = [{k: v for k, v in row.items() if not k.startswith("_")} for row in table_rows]
            frame = pd.DataFrame(full_rows)
            if not frame.empty:
                for col in ("Line", "Model", "Edge", "Confidence"):
                    if col in frame.columns:
                        frame[col] = frame[col].map(lambda x: f"{x:.1f}" if isinstance(x, (int, float)) else x)
            st.markdown("### Full Slate")
            try:
                styled = frame.style.map(_style_pick_cell, subset=["Pick"])
                styled = styled.map(_style_edge_cell, subset=["Edge"])
                styled = styled.map(_style_conf_cell, subset=["Confidence"])
            except Exception:
                styled = frame.style.applymap(_style_pick_cell, subset=["Pick"])
                styled = styled.applymap(_style_edge_cell, subset=["Edge"])
                styled = styled.applymap(_style_conf_cell, subset=["Confidence"])
            st.dataframe(styled, use_container_width=True, hide_index=True)

            top_rows = [row for row in table_rows if str(row.get("_action", "")).upper() == "BET"]
            if not top_rows:
                top_rows = sorted(table_rows, key=lambda r: r.get("_edge_abs", -1.0), reverse=True)[:5]
            else:
                top_rows = sorted(top_rows, key=lambda r: r.get("_edge_abs", -1.0), reverse=True)
            top_rows_clean = [{k: v for k, v in row.items() if not k.startswith("_")} for row in top_rows]
            top_frame = pd.DataFrame(top_rows_clean)
            if not top_frame.empty:
                for col in ("Line", "Model", "Edge", "Confidence"):
                    if col in top_frame.columns:
                        top_frame[col] = top_frame[col].map(lambda x: f"{x:.1f}" if isinstance(x, (int, float)) else x)
            st.markdown("### Top Plays")
            try:
                top_styled = top_frame.style.map(_style_pick_cell, subset=["Pick"])
                top_styled = top_styled.map(_style_edge_cell, subset=["Edge"])
                top_styled = top_styled.map(_style_conf_cell, subset=["Confidence"])
            except Exception:
                top_styled = top_frame.style.applymap(_style_pick_cell, subset=["Pick"])
                top_styled = top_styled.applymap(_style_edge_cell, subset=["Edge"])
                top_styled = top_styled.applymap(_style_conf_cell, subset=["Confidence"])
            st.dataframe(top_styled, use_container_width=True, hide_index=True)
        else:
            full_rows = [{k: v for k, v in row.items() if not k.startswith("_")} for row in table_rows]
            st.markdown("### Full Slate")
            st.dataframe(full_rows, use_container_width=True, hide_index=True)
            top_rows = [row for row in table_rows if str(row.get("_action", "")).upper() == "BET"]
            if not top_rows:
                top_rows = sorted(table_rows, key=lambda r: r.get("_edge_abs", -1.0), reverse=True)[:5]
            else:
                top_rows = sorted(top_rows, key=lambda r: r.get("_edge_abs", -1.0), reverse=True)
            top_rows_clean = [{k: v for k, v in row.items() if not k.startswith("_")} for row in top_rows]
            st.markdown("### Top Plays")
            st.dataframe(top_rows_clean, use_container_width=True, hide_index=True)
    else:
        st.info("No top plays available yet.")


if __name__ == "__main__":
    render_public_app()

