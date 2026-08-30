"""Tests for the TradingLab-style multi-horizon momentum calculation."""

from __future__ import annotations

import math

import pytest

from trading_bot import config, multi_horizon_momentum


def _prices(*, current: float = 100.0) -> list[float]:
    prices = [100.0] * 50
    prices[-1] = current
    return prices


def test_full_long_when_all_four_horizons_are_positive() -> None:
    prices = _prices()
    for lookback in config.MULTI_HORIZON_LOOKBACKS:
        prices[-1 - lookback] = 90.0

    reading = multi_horizon_momentum.evaluate(prices)

    assert reading.score == 4
    assert reading.direction == "long"
    assert reading.position_scale == 1.0
    assert all(value > 0.0 for value in reading.horizon_returns_pct)


def test_half_long_when_three_horizons_rise_and_one_falls() -> None:
    prices = _prices()
    for lookback in config.MULTI_HORIZON_LOOKBACKS[:3]:
        prices[-1 - lookback] = 90.0
    prices[-1 - config.MULTI_HORIZON_LOOKBACKS[3]] = 110.0

    reading = multi_horizon_momentum.evaluate(prices)

    assert reading.score == 2
    assert reading.direction == "long"
    assert reading.position_scale == 0.5


def test_full_short_when_all_four_horizons_are_negative() -> None:
    prices = _prices()
    for lookback in config.MULTI_HORIZON_LOOKBACKS:
        prices[-1 - lookback] = 110.0

    reading = multi_horizon_momentum.evaluate(prices)

    assert reading.score == -4
    assert reading.direction == "short"
    assert reading.position_scale == 1.0


def test_balanced_votes_are_flat() -> None:
    prices = _prices()
    for lookback in config.MULTI_HORIZON_LOOKBACKS[:2]:
        prices[-1 - lookback] = 90.0
    for lookback in config.MULTI_HORIZON_LOOKBACKS[2:]:
        prices[-1 - lookback] = 110.0

    reading = multi_horizon_momentum.evaluate(prices)

    assert reading.score == 0
    assert reading.direction is None
    assert reading.position_scale == 0.0


def test_annualized_average_move_uses_stock_sessions() -> None:
    prices = [100.0 * (1.01**index) for index in range(50)]

    reading = multi_horizon_momentum.evaluate(prices)

    assert reading.annualized_average_move_pct == pytest.approx(
        math.sqrt(config.STOCK_ANNUALIZATION_PERIODS), rel=1e-12,
    )


@pytest.mark.parametrize(
    "prices, match",
    [
        ([100.0] * 42, "insufficient closes"),
        ([100.0] * 49 + [0.0], "finite and positive"),
        ([100.0] * 49 + [float("nan")], "finite and positive"),
    ],
)
def test_invalid_history_is_rejected(prices: list[float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        multi_horizon_momentum.evaluate(prices)
