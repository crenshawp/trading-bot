"""Tests for the dataclass models — frozen invariants and validation constants."""

import dataclasses
from datetime import datetime

import pytest

from trading_bot.models import (
    VALID_ASSET_CLASSES,
    VALID_DIRECTIONS,
    VALID_OUTCOMES,
    Signal,
    Trade,
)


def test_signal_is_frozen() -> None:
    sig = Signal(
        timestamp=datetime(2026, 5, 25, 9, 31),
        ticker="GOOGL",
        asset_class="stock",
        signal_type="ema21_pullback",
        direction="call",
        entry_price=180.0,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        sig.ticker = "BLK"


def test_trade_is_frozen() -> None:
    trade = Trade(signal_id=1, opened_at=datetime(2026, 5, 25))
    with pytest.raises(dataclasses.FrozenInstanceError):
        trade.signal_id = 99


def test_valid_constants_are_frozensets() -> None:
    assert isinstance(VALID_ASSET_CLASSES, frozenset)
    assert isinstance(VALID_DIRECTIONS, frozenset)
    assert isinstance(VALID_OUTCOMES, frozenset)
    assert "stock" in VALID_ASSET_CLASSES and "crypto" in VALID_ASSET_CLASSES
    assert {"call", "put", "long", "short"} <= VALID_DIRECTIONS
    assert {"win", "loss", "breakeven", "open", "expired"} <= VALID_OUTCOMES
