"""Odds/price helpers shared across the project."""

from __future__ import annotations

from typing import Optional


def american_to_decimal(american: Optional[float], default_decimal: float = 1.9090909091) -> float:
    """Convert American odds to decimal odds.

    Returns default_decimal (≈ -110) when odds are missing/unparseable.
    """
    try:
        if american is None:
            return float(default_decimal)
        a = float(american)
        if a != a:  # NaN
            return float(default_decimal)
        a_int = int(round(a))
        if a_int >= 100:
            return 1.0 + (a_int / 100.0)
        return 1.0 + (100.0 / max(1, abs(a_int)))
    except Exception:
        return float(default_decimal)


def decimal_to_implied_prob(decimal_odds: Optional[float]) -> float:
    try:
        d = float(decimal_odds) if decimal_odds is not None else 0.0
        if d <= 0:
            return 0.0
        return 1.0 / d
    except Exception:
        return 0.0

