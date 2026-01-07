"""Time-aligned status history attachment utilities."""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def attach_status_history(
    features_df: pd.DataFrame,
    status_history_path: Optional[str],
    game_time_col: str = "date",
) -> pd.DataFrame:
    """Attach time-aligned goalie/injury adjustments for training from a status history CSV.

    Optional leakage guard:
    - If CSV contains `status_timestamp`, entries with status_timestamp > game_time are ignored.

    Expected CSV columns (flexible; best-effort mapping):
    - game_id (optional)
    - date (optional; used with matchup)
    - matchup (AWAY@HOME) (optional; used with date)
    - status_timestamp (optional)
    - home_goalie_adj, away_goalie_adj, injury_penalty_adj (optional)
    - home_goalie_gsax, away_goalie_gsax (optional)
    - home_goalie_prob, away_goalie_prob (optional)
    """
    if features_df is None or not isinstance(features_df, pd.DataFrame) or features_df.empty:
        return features_df
    if not status_history_path:
        return features_df
    if not os.path.exists(status_history_path):
        return features_df

    try:
        hist = pd.read_csv(status_history_path)
    except Exception:
        return features_df
    if hist is None or hist.empty:
        return features_df

    df = features_df.copy()
    hist = hist.copy()

    if "game_id" in df.columns:
        df["game_id"] = df["game_id"].astype(str)
    if "game_id" in hist.columns:
        hist["game_id"] = hist["game_id"].astype(str)

    if "matchup" not in df.columns and {"home_team", "away_team"}.issubset(df.columns):
        df["matchup"] = df["away_team"].astype(str).str.upper() + "@" + df["home_team"].astype(str).str.upper()
    if "matchup" in hist.columns:
        hist["matchup"] = hist["matchup"].astype(str).str.upper().str.strip()

    if game_time_col in df.columns:
        df["_game_dt"] = pd.to_datetime(df[game_time_col], errors="coerce", utc=True).dt.tz_convert(None)
        df["_date_only"] = df["_game_dt"].dt.date
    else:
        df["_game_dt"] = pd.NaT
        df["_date_only"] = pd.NaT

    if "date" in hist.columns:
        hist["_date_only"] = pd.to_datetime(hist["date"], errors="coerce").dt.date
    else:
        hist["_date_only"] = pd.NaT

    if "status_timestamp" in hist.columns:
        hist["_status_dt"] = pd.to_datetime(hist["status_timestamp"], errors="coerce", utc=True).dt.tz_convert(None)
    else:
        hist["_status_dt"] = pd.NaT

    col_map: Dict[str, List[str]] = {
        "home_goalie_adj": ["home_goalie_adj", "hg_adj", "home_adj"],
        "away_goalie_adj": ["away_goalie_adj", "ag_adj", "away_adj"],
        "injury_penalty_adj": ["injury_penalty_adj", "inj_adj", "injury_adj"],
        "home_goalie_gsax": ["home_goalie_gsax", "hg_gsax", "home_gsax"],
        "away_goalie_gsax": ["away_goalie_gsax", "ag_gsax", "away_gsax"],
        "home_goalie_prob": ["home_goalie_prob", "hg_prob", "home_prob"],
        "away_goalie_prob": ["away_goalie_prob", "ag_prob", "away_prob"],
    }
    for target, aliases in col_map.items():
        if target in hist.columns:
            continue
        for alt in aliases:
            if alt in hist.columns:
                hist = hist.rename(columns={alt: target})
                break

    attach_cols = [c for c in col_map.keys() if c in hist.columns]
    if not attach_cols:
        df.drop(columns=["_game_dt", "_date_only"], inplace=True, errors="ignore")
        return df

    for c in attach_cols:
        hist[c] = pd.to_numeric(hist[c], errors="coerce")

    # Join by game_id when possible; else by matchup+date.
    if "game_id" in hist.columns and "game_id" in df.columns and hist["game_id"].notna().any():
        subset = hist[["game_id", "_status_dt"] + attach_cols].dropna(subset=["game_id"])
        merged = df.merge(subset, on="game_id", how="left", suffixes=("", "_hist"))
    elif "matchup" in hist.columns and hist["matchup"].notna().any():
        subset = hist[["matchup", "_date_only", "_status_dt"] + attach_cols].dropna(subset=["matchup"])
        merged = df.merge(subset, left_on=["matchup", "_date_only"], right_on=["matchup", "_date_only"], how="left", suffixes=("", "_hist"))
    else:
        df.drop(columns=["_game_dt", "_date_only"], inplace=True, errors="ignore")
        return df

    # Leakage guard: ignore status updates after puck drop (when timestamps exist).
    if "_status_dt" in merged.columns and "_game_dt" in merged.columns:
        late_mask = merged["_status_dt"].notna() & merged["_game_dt"].notna() & (merged["_status_dt"] > merged["_game_dt"])
        if late_mask.any():
            for c in attach_cols:
                try:
                    merged.loc[late_mask, c] = np.nan
                except Exception:
                    pass

    # Fill into df columns (prefer existing non-zero/non-null).
    for c in attach_cols:
        if c not in merged.columns:
            continue
        if c not in df.columns:
            df[c] = np.nan
        base = pd.to_numeric(merged.get(c), errors="coerce")
        incoming = pd.to_numeric(merged.get(c), errors="coerce")
        # If df has placeholder 0.0 or NaN, fill from incoming.
        try:
            existing = pd.to_numeric(df.get(c), errors="coerce")
            mask = existing.isna() | np.isclose(existing.fillna(0.0), 0.0, atol=1e-12)
            df.loc[mask, c] = incoming.loc[mask]
        except Exception:
            df[c] = base

    df.drop(columns=["_game_dt", "_date_only"], inplace=True, errors="ignore")
    return df

