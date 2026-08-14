import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import streamlit as st
import streamlit.components.v1 as components


APP_ROOT = Path(__file__).resolve().parent


def _latest_record(log_path: Path) -> Optional[Dict[str, float]]:
    if not log_path.exists():
        return None
    fmts = ("%Y-%m-%d %H:%M:%S", "%m/%d/%Y %H:%M")
    latest_by_game: Dict[str, Dict[str, str]] = {}
    latest_dt_by_game: Dict[str, datetime] = {}
    with log_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            gid = str(row.get("game_id") or "").strip()
            if not gid:
                continue
            raw_date = str(row.get("date") or "").strip()
            dt = datetime.min
            for fmt in fmts:
                try:
                    dt = datetime.strptime(raw_date, fmt)
                    break
                except Exception:
                    continue
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
    st.title("NHL Over/Under Predictions")
    st.caption("Public dashboard and latest model outputs.")

    st.button("Refresh", type="secondary")

    predictions_image = APP_ROOT / "predictions.png"
    dashboard_html = APP_ROOT / "nhl_real_data_dashboard.html"
    log_path = APP_ROOT / "bets_log.csv"

    if predictions_image.exists():
        mtime = datetime.fromtimestamp(predictions_image.stat().st_mtime)
        st.caption(f"Updated: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        st.image(str(predictions_image), caption="Latest predictions", use_container_width=True)
    else:
        st.info("Predictions image is not available yet.")

    record = _latest_record(log_path)
    if record:
        st.metric(
            "Latest per-game O/U record",
            f"{int(record['wins'])}-{int(record['losses'])}",
            f"{record['win_rate'] * 100:.1f}% win rate",
        )

    if dashboard_html.exists():
        dashboard_text = dashboard_html.read_text(encoding="utf-8", errors="ignore")
        with st.expander("Full dashboard", expanded=True):
            components.html(dashboard_text, height=1200, scrolling=True)
    else:
        st.info("Dashboard HTML is not available yet.")


if __name__ == "__main__":
    render_public_app()

