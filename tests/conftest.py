"""Shared pytest fixtures for the trading-bot test suite."""

import os
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest


@pytest.fixture
def railway_env() -> Iterator[None]:
    """Simulate Railway environment."""
    with patch.dict(os.environ, {"RAILWAY_ENVIRONMENT": "production"}):
        yield


@pytest.fixture
def local_env() -> Iterator[None]:
    """Simulate local environment (no RAILWAY_ENVIRONMENT)."""
    env = {k: v for k, v in os.environ.items() if k != "RAILWAY_ENVIRONMENT"}
    with patch.dict(os.environ, env, clear=True):
        yield


@pytest.fixture
def mock_keyring() -> Iterator[dict[str, str]]:
    """In-memory keyring backend for tests."""
    store: dict[str, str] = {}

    def fake_get(service: str, name: str) -> str | None:
        return store.get(f"{service}:{name}")

    def fake_set(service: str, name: str, value: str) -> None:
        store[f"{service}:{name}"] = value

    def fake_delete(service: str, name: str) -> None:
        store.pop(f"{service}:{name}", None)

    with (
        patch("keyring.get_password", side_effect=fake_get),
        patch("keyring.set_password", side_effect=fake_set),
        patch("keyring.delete_password", side_effect=fake_delete),
    ):
        yield store


@pytest.fixture
def tmp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Fresh, initialized database in a per-test temp directory.

    db.py references ``config.DB_PATH`` (not a cached import), so patching
    the source in config is sufficient.
    """
    db_path = tmp_path / "test.db"
    monkeypatch.setattr("trading_bot.config.DB_PATH", db_path)
    from trading_bot import db
    db.init_db()
    yield db_path


@pytest.fixture
def fake_candles() -> Callable[..., pd.DataFrame]:
    """Build a small OHLCV DataFrame with controllable highs and lows.

    Used by outcome resolver tests to drive the TP/SL decision logic without
    hitting yfinance. The index is tz-aware UTC datetimes spaced by the given
    interval. ``closes`` defaults to the midpoint of high/low per candle.
    """
    def _build(
        highs:  list[float],
        lows:   list[float],
        closes: list[float] | None = None,
        start:  datetime | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        if len(highs) != len(lows):
            raise ValueError("highs and lows must be the same length")
        if closes is not None and len(closes) != len(highs):
            raise ValueError("closes must match highs/lows length")

        step = timedelta(hours=1) if interval == "1h" else timedelta(days=1)
        anchor = start if start is not None else datetime(2026, 4, 1, tzinfo=UTC)
        index = [anchor + step * i for i in range(len(highs))]

        opens  = [(h + low) / 2 for h, low in zip(highs, lows, strict=True)]
        closes_ = closes if closes is not None else opens

        return pd.DataFrame(
            {
                "Open":   opens,
                "High":   highs,
                "Low":    lows,
                "Close":  closes_,
                "Volume": [1_000_000] * len(highs),
            },
            index=pd.DatetimeIndex(index),
        )

    return _build
