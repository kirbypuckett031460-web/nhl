"""Thin compatibility wrapper around RealNHLDataCollector.

The historical get_real_data script used random estimates whenever feed data was
missing. That behavior has been removed: this module now simply delegates to the
hardened collector so only verifiable stats are returned.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from real_nhl_data import RealNHLDataCollector


def get_real_nhl_data_simple(
    days_back: int = 30,
    max_games: int = 200,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Return verified NHL games using the shared RealNHLDataCollector."""
    collector = RealNHLDataCollector()
    if end_date is None:
        end_date = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    end_dt = pd.to_datetime(end_date)
    start_dt = end_dt - pd.Timedelta(days=days_back)
    return collector.get_real_nhl_data(
        start_date=start_dt.strftime("%Y-%m-%d"),
        end_date=end_dt.strftime("%Y-%m-%d"),
        max_games=max_games,
    )


__all__ = ["get_real_nhl_data_simple", "RealNHLDataCollector"]
