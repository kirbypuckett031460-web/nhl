"""Market history (odds snapshots -> canonical open/close) utilities."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import glob


def resolve_closing_lines_version(version_dir: str, version_prefix: str) -> Optional[str]:
    """Resolve a specific versioned closing-lines file by prefix (YYYYMMDD or YYYYMMDDTHHMMSSZ)."""
    if not version_dir or not version_prefix:
        return None
    try:
        pattern = os.path.join(version_dir, f"{version_prefix}*_closing_lines.csv")
        matches = sorted(glob.glob(pattern))
        return matches[-1] if matches else None
    except Exception:
        return None


def build_closing_lines_from_odds_history(
    odds_history_path: str,
    output_path: str,
    version_dir: Optional[str] = "data/history/closing_lines",
    require_game_date: bool = False,
) -> Optional[str]:
    """Build a canonical open/close totals+prices table from odds_history snapshots.

    Expected odds_history columns:
    - timestamp
    - game_id
    - book_total
    - (optional) book_over, book_under
    - (optional) game_date (ISO, used as puck drop time)
    - (optional) book / book_key / book_title

    Selection:
    - open: earliest snapshot <= game_date (else earliest snapshot)
    - close: latest snapshot <= game_date (else latest snapshot)
    """
    if not odds_history_path or not os.path.exists(odds_history_path):
        return None
    try:
        oh = pd.read_csv(odds_history_path)
    except Exception:
        return None
    if oh.empty:
        return None
    required = {"timestamp", "game_id", "book_total"}
    if not required.issubset(set(oh.columns)):
        return None

    oh = oh.copy()
    oh["game_id"] = oh["game_id"].astype(str)
    oh["ts"] = pd.to_datetime(oh["timestamp"], errors="coerce")
    oh["book_total"] = pd.to_numeric(oh["book_total"], errors="coerce")
    oh["book_over"] = pd.to_numeric(oh.get("book_over"), errors="coerce")
    oh["book_under"] = pd.to_numeric(oh.get("book_under"), errors="coerce")

    if "game_date" in oh.columns:
        try:
            oh["game_dt"] = pd.to_datetime(oh["game_date"], errors="coerce", utc=True).dt.tz_convert(None)
        except Exception:
            oh["game_dt"] = pd.NaT
    else:
        oh["game_dt"] = pd.NaT

    if "book" not in oh.columns:
        oh["book"] = oh.get("book_key", oh.get("book_title", ""))

    rows: List[Dict[str, Any]] = []
    for gid, grp in oh.dropna(subset=["ts"]).groupby("game_id"):
        g = grp.sort_values("ts")
        try:
            gdt = g["game_dt"].dropna().iloc[-1] if g["game_dt"].notna().any() else None
        except Exception:
            gdt = None
        if require_game_date and (gdt is None or not isinstance(gdt, pd.Timestamp)):
            continue
        if gdt is not None and isinstance(gdt, pd.Timestamp):
            pre = g[g["ts"] <= gdt]
            use_open = pre if not pre.empty else g
            use_close = pre if not pre.empty else g
        else:
            use_open = g
            use_close = g
        open_row = use_open.iloc[0]
        close_row = use_close.iloc[-1]
        rows.append(
            {
                "game_id": str(gid),
                "open_total": open_row.get("book_total"),
                "open_over_price": open_row.get("book_over"),
                "open_under_price": open_row.get("book_under"),
                "open_source": str(open_row.get("book") or open_row.get("book_key") or ""),
                "open_timestamp": open_row.get("timestamp"),
                "closing_total": close_row.get("book_total"),
                "closing_over_price": close_row.get("book_over"),
                "closing_under_price": close_row.get("book_under"),
                "closing_source": str(close_row.get("book") or close_row.get("book_key") or ""),
                "closing_timestamp": close_row.get("timestamp"),
            }
        )
    if not rows:
        return None

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out_df.to_csv(output_path, index=False)

    if version_dir:
        try:
            os.makedirs(version_dir, exist_ok=True)
            stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            version_path = os.path.join(version_dir, f"{stamp}_closing_lines.csv")
            out_df.to_csv(version_path, index=False)
        except Exception:
            pass

    return output_path

