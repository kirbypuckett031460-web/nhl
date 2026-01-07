"""Probability and EV utilities for totals markets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.stats import nbinom, poisson

from nhl_model.odds_utils import american_to_decimal


@dataclass
class TotalsProbs:
    over: float
    under: float
    push: float

    def normalized(self) -> "TotalsProbs":
        total = float(self.over + self.under + self.push)
        if not np.isfinite(total) or total <= 0:
            return TotalsProbs(over=0.5, under=0.5, push=0.0)
        return TotalsProbs(over=float(self.over / total), under=float(self.under / total), push=float(self.push / total))


def is_integer_line(line: float) -> bool:
    try:
        lv = float(line)
        return abs(lv - round(lv)) < 1e-9
    except Exception:
        return False


def totals_probs_nb_poisson(mu: float, line: float, nb_k: Optional[float]) -> TotalsProbs:
    """Compute P(over/under/push) for a totals line using NB (if k provided) else Poisson."""
    try:
        mu_f = float(mu)
        if not np.isfinite(mu_f) or mu_f <= 0:
            return TotalsProbs(over=0.0, under=1.0, push=0.0)
        lv = float(line)
        if is_integer_line(lv):
            L = int(round(lv))
            if nb_k is not None and np.isfinite(float(nb_k)) and float(nb_k) > 0:
                k = float(nb_k)
                p = k / (k + mu_f)
                push = float(nbinom.pmf(L, k, p))
                under = float(nbinom.cdf(L - 1, k, p)) if L > 0 else 0.0
                over = float(1.0 - nbinom.cdf(L, k, p))
                return TotalsProbs(over=over, under=under, push=push).normalized()
            push = float(poisson.pmf(L, mu_f))
            under = float(poisson.cdf(L - 1, mu_f)) if L > 0 else 0.0
            over = float(1.0 - poisson.cdf(L, mu_f))
            return TotalsProbs(over=over, under=under, push=push).normalized()
        # Half-lines: push mass is 0. Over means >= ceil(line) -> > floor(line)
        L = int(np.floor(lv))
        if nb_k is not None and np.isfinite(float(nb_k)) and float(nb_k) > 0:
            k = float(nb_k)
            p = k / (k + mu_f)
            under = float(nbinom.cdf(L, k, p))
            over = float(1.0 - under)
            return TotalsProbs(over=over, under=under, push=0.0).normalized()
        under = float(poisson.cdf(L, mu_f))
        over = float(1.0 - under)
        return TotalsProbs(over=over, under=under, push=0.0).normalized()
    except Exception:
        return TotalsProbs(over=0.5, under=0.5, push=0.0)


def side_ev(prob_win: float, prob_push: float, american_price: Optional[float]) -> float:
    """Expected value per 1u stake, including push probability."""
    dec = american_to_decimal(american_price)
    b = dec - 1.0
    # win: +b, loss: -1, push: 0
    prob_loss = max(0.0, 1.0 - float(prob_win) - float(prob_push))
    return float(prob_win * b - prob_loss)


def totals_side_evs(probs: TotalsProbs, over_price: Optional[float], under_price: Optional[float]) -> Tuple[float, float]:
    """Return (EV_over, EV_under) per 1u stake."""
    ev_over = side_ev(probs.over, probs.push, over_price)
    ev_under = side_ev(probs.under, probs.push, under_price)
    return ev_over, ev_under

