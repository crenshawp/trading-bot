"""TradingLab-style multi-horizon momentum signal math.

The source video scores four daily close-to-close horizons (one week, two
weeks, one month, and two months), then combines their signs into a single
``-4..+4`` trend score.  This module keeps that calculation pure so the live
scanner, tests, and any future backtester use exactly the same rules.

Execution policy (weekly cadence, ATR exits, and portfolio sizing) belongs to
the existing scanner/allocation layers.  The video did not define a mechanical
exit, so that policy is intentionally not invented inside the signal math.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from trading_bot import config


@dataclass(frozen=True)
class MomentumReading:
    """One completed multi-horizon evaluation."""

    score: int
    direction: str | None
    position_scale: float
    current_price: float
    horizon_returns_pct: tuple[float, ...]
    annualized_average_move_pct: float


def evaluate(
    closes: Sequence[float],
    *,
    lookbacks: tuple[int, ...] = config.MULTI_HORIZON_LOOKBACKS,
    volatility_window: int = config.MULTI_HORIZON_VOLATILITY_WINDOW,
    annualization_periods: int = config.STOCK_ANNUALIZATION_PERIODS,
) -> MomentumReading:
    """Score closed daily prices and return direction plus exposure fraction.

    Each positive horizon return contributes ``+1`` and each negative return
    contributes ``-1``. An exactly unchanged horizon contributes zero rather
    than manufacturing a directional vote. Position scale is ``abs(score)``
    divided by the number of horizons, reproducing the video's full/half
    mapping for the normal ``+/-4`` and ``+/-2`` cases.

    ``annualized_average_move_pct`` reproduces the video's volatility measure:
    the mean absolute daily percentage move, annualized by square-root-of-time.
    Stocks use ``sqrt(252)``, not the video's crypto-specific ``sqrt(365)``.
    It is retained as auditable signal metadata; the production allocator uses
    the bot's existing ATR-normalized risk model.
    """
    if not lookbacks or any(lb <= 0 for lb in lookbacks):
        raise ValueError("lookbacks must contain positive integers")
    if volatility_window <= 0:
        raise ValueError("volatility_window must be positive")
    if annualization_periods <= 0:
        raise ValueError("annualization_periods must be positive")

    required = max(max(lookbacks) + 1, volatility_window + 1)
    if len(closes) < required:
        raise ValueError(
            f"insufficient closes: {len(closes)} available, {required} required"
        )

    prices = [float(value) for value in closes]
    relevant = prices[-required:]
    if any(not math.isfinite(value) or value <= 0.0 for value in relevant):
        raise ValueError("closes must be finite and positive")

    current = prices[-1]
    horizon_returns: list[float] = []
    votes: list[int] = []
    for lookback in lookbacks:
        past = prices[-1 - lookback]
        horizon_return = (current / past - 1.0) * 100.0
        horizon_returns.append(horizon_return)
        votes.append(1 if horizon_return > 0.0 else -1 if horizon_return < 0.0 else 0)

    score = sum(votes)
    direction = "long" if score > 0 else "short" if score < 0 else None
    position_scale = abs(score) / len(lookbacks)

    volatility_closes = prices[-(volatility_window + 1):]
    absolute_moves = [
        abs(current_close / previous_close - 1.0)
        for previous_close, current_close in zip(
            volatility_closes[:-1], volatility_closes[1:], strict=True,
        )
    ]
    average_move = sum(absolute_moves) / len(absolute_moves)
    annualized_average_move_pct = (
        average_move * math.sqrt(annualization_periods) * 100.0
    )

    return MomentumReading(
        score=score,
        direction=direction,
        position_scale=position_scale,
        current_price=current,
        horizon_returns_pct=tuple(horizon_returns),
        annualized_average_move_pct=annualized_average_move_pct,
    )
