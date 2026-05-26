"""Tests for trading_bot.regime — Phase 2.1 macro regime detection.

yfinance is mocked throughout; classification, caching, and snapshot logic
are exercised in isolation. The disk cache (~/.regime_cache.json) is
redirected to ``tmp_path`` per test so we don't trample the developer's
real cache.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from trading_bot import db, regime
from trading_bot.models import Signal, Trade

# ────────────────────── fixtures + helpers ──────────────────────


@pytest.fixture(autouse=True)
def isolate_regime_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Path]:
    """Redirect the disk cache + reset the memory cache for every test."""
    cache_path = tmp_path / ".regime_cache.json"
    monkeypatch.setattr("trading_bot.regime._CACHE_PATH", cache_path)
    regime._reset_cache_for_tests()
    yield cache_path
    regime._reset_cache_for_tests()


def _build_spy_df(
    *,
    rows: int = 260,
    start_price: float = 400.0,
    drift: float = 0.5,
    end_date: datetime | None = None,
    overrides: dict[int, float] | None = None,
) -> pd.DataFrame:
    """Build a synthetic SPY OHLC DataFrame with linear-drift closes.

    `drift > 0` produces a rising market (bull); `drift < 0` a falling
    one (bear); `drift == 0` flat (sideways once EMAs converge). The
    `overrides` dict lets a test pin a specific close at a specific index.
    """
    closes = [start_price + i * drift for i in range(rows)]
    if overrides:
        for idx, price in overrides.items():
            closes[idx] = price
    anchor = end_date if end_date is not None else datetime(2026, 5, 26, tzinfo=UTC)
    index = [anchor - timedelta(days=(rows - 1 - i)) for i in range(rows)]
    df = pd.DataFrame(
        {
            "Open":   closes,
            "High":   [c + 1.0 for c in closes],
            "Low":    [c - 1.0 for c in closes],
            "Close":  closes,
            "Volume": [10_000_000] * rows,
        },
        index=pd.DatetimeIndex(index),
    )
    return df


def _patch_yf(monkeypatch: pytest.MonkeyPatch, df: pd.DataFrame | Exception | None,
              ) -> list[dict[str, Any]]:
    """Patch yf.download. Captures call kwargs into a list for assertions."""
    calls: list[dict[str, Any]] = []

    def fake_download(ticker: str, **kwargs: Any) -> pd.DataFrame | None:
        calls.append({"ticker": ticker, **kwargs})
        if isinstance(df, Exception):
            raise df
        return df

    monkeypatch.setattr("trading_bot.regime.yf.download", fake_download)
    return calls


# ────────────────────── classification ──────────────────────


def test_classify_bull() -> None:
    assert regime.classify(spy_close=600, ema50=580, ema200=550, ema50_slope=1.5) == "bull"


def test_classify_bear() -> None:
    assert regime.classify(spy_close=400, ema50=420, ema200=450, ema50_slope=-0.8) == "bear"


def test_classify_sideways_price_above_50_below_200() -> None:
    # Price > 200 EMA but 50 < 200 → mixed → sideways
    assert regime.classify(spy_close=560, ema50=540, ema200=550, ema50_slope=0.5) == "sideways"


def test_classify_sideways_price_below_50_above_200() -> None:
    assert regime.classify(spy_close=540, ema50=560, ema200=550, ema50_slope=-0.5) == "sideways"


def test_classify_sideways_positive_slope_in_bear_setup() -> None:
    # SPY < 200 EMA, 50 < 200, but slope > 0 → not strictly bear
    assert regime.classify(spy_close=540, ema50=545, ema200=560, ema50_slope=0.5) == "sideways"


def test_classify_sideways_negative_slope_in_bull_setup() -> None:
    assert regime.classify(spy_close=600, ema50=595, ema200=580, ema50_slope=-0.5) == "sideways"


def test_classify_sideways_when_above_200_but_50_equal_200() -> None:
    assert regime.classify(spy_close=600, ema50=580, ema200=580, ema50_slope=1.0) == "sideways"


def test_classify_zero_slope_is_sideways() -> None:
    assert regime.classify(spy_close=600, ema50=580, ema200=550, ema50_slope=0.0) == "sideways"


def test_classify_ema50_equals_ema200_is_sideways() -> None:
    assert regime.classify(spy_close=600, ema50=580, ema200=580, ema50_slope=1.0) == "sideways"


# ────────────────────── fetch + caching ──────────────────────


def test_get_current_regime_single_fetch_on_cold_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_yf(monkeypatch, _build_spy_df())
    snap = regime.get_current_regime()
    assert snap.regime in {"bull", "bear", "sideways"}
    assert snap.spy_close > 0
    assert len(calls) == 1
    assert calls[0]["ticker"] == "SPY"


def test_get_current_regime_uses_memory_cache_within_24h(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_yf(monkeypatch, _build_spy_df())
    first = regime.get_current_regime()
    second = regime.get_current_regime()
    assert first == second
    assert len(calls) == 1  # second call did NOT trigger another fetch


def test_force_refresh_bypasses_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_yf(monkeypatch, _build_spy_df())
    regime.get_current_regime()
    regime.get_current_regime(force_refresh=True)
    assert len(calls) == 2


def test_disk_cache_persists_across_process_restart(
    monkeypatch: pytest.MonkeyPatch, isolate_regime_cache: Path,
) -> None:
    _patch_yf(monkeypatch, _build_spy_df())
    first = regime.get_current_regime()
    assert isolate_regime_cache.exists()

    # Simulate a process restart by wiping the in-memory cache only.
    regime._reset_cache_for_tests()

    # Re-patch yfinance to raise — the disk cache should satisfy the call.
    calls = _patch_yf(monkeypatch, RuntimeError("network down"))
    second = regime.get_current_regime()
    assert second == first
    assert calls == []  # yfinance was NOT called


def test_disk_cache_expires_after_24h(
    monkeypatch: pytest.MonkeyPatch, isolate_regime_cache: Path,
) -> None:
    # Hand-roll an old cache file
    old_snap = regime.RegimeSnapshot(
        date="2026-04-01", regime="bull",
        spy_close=600.0, ema50=590.0, ema200=550.0, ema50_slope=1.0,
    )
    old_cached_at = datetime.now(UTC) - timedelta(hours=25)
    payload = {
        "cached_at": old_cached_at.isoformat(),
        "snapshot": {
            "date": old_snap.date, "regime": old_snap.regime,
            "spy_close": old_snap.spy_close, "ema50": old_snap.ema50,
            "ema200": old_snap.ema200, "ema50_slope": old_snap.ema50_slope,
        },
    }
    isolate_regime_cache.write_text(json.dumps(payload), encoding="utf-8")
    regime._reset_cache_for_tests()

    calls = _patch_yf(monkeypatch, _build_spy_df())
    snap = regime.get_current_regime()
    assert len(calls) == 1  # stale disk cache rejected, re-fetched
    assert snap.date != old_snap.date or snap.spy_close != old_snap.spy_close


def test_malformed_yfinance_response_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # DataFrame with no Close column
    bad = pd.DataFrame({"Open": [1.0], "Volume": [1]})
    _patch_yf(monkeypatch, bad)
    with pytest.raises(regime.RegimeFetchError):
        regime.get_current_regime()


def test_empty_yfinance_response_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_yf(monkeypatch, pd.DataFrame())
    with pytest.raises(regime.RegimeFetchError):
        regime.get_current_regime()


def test_insufficient_yfinance_data_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Only 50 rows — far short of 200 EMA + slope window
    _patch_yf(monkeypatch, _build_spy_df(rows=50))
    with pytest.raises(regime.RegimeFetchError, match="insufficient"):
        regime.get_current_regime()


def test_yfinance_exception_raises_regime_fetch_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_yf(monkeypatch, RuntimeError("rate limited"))
    with pytest.raises(regime.RegimeFetchError, match="rate limited"):
        regime.get_current_regime()


def test_multiindex_columns_flattened(monkeypatch: pytest.MonkeyPatch) -> None:
    df = _build_spy_df()
    df.columns = pd.MultiIndex.from_product([df.columns, ["SPY"]])
    _patch_yf(monkeypatch, df)
    snap = regime.get_current_regime()
    assert snap.regime in {"bull", "bear", "sideways"}


def test_disk_cache_corrupt_file_falls_through_to_fetch(
    monkeypatch: pytest.MonkeyPatch, isolate_regime_cache: Path,
) -> None:
    isolate_regime_cache.write_text("{not json", encoding="utf-8")
    calls = _patch_yf(monkeypatch, _build_spy_df())
    snap = regime.get_current_regime()
    assert len(calls) == 1
    assert snap.regime in {"bull", "bear", "sideways"}


# ────────────────────── snapshot_regime_for_date ──────────────────────


def test_snapshot_for_date_returns_snapshot_for_known_date(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = _build_spy_df(rows=400, drift=0.5, end_date=datetime(2026, 5, 26, tzinfo=UTC))
    _patch_yf(monkeypatch, df)
    snap = regime.snapshot_regime_for_date("2026-05-20")
    assert snap is not None
    assert snap.date == "2026-05-20"
    assert snap.regime in {"bull", "bear", "sideways"}


def test_snapshot_for_date_returns_none_for_weekend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Build a DF whose index intentionally skips Saturdays/Sundays.
    rows = 400
    closes = [400.0 + i * 0.5 for i in range(rows)]
    anchor = datetime(2026, 5, 26, tzinfo=UTC)
    # Build a business-day index and trim to rows.
    bdays = pd.bdate_range(end=anchor, periods=rows)
    df = pd.DataFrame(
        {
            "Open":   closes, "High":   [c + 1.0 for c in closes],
            "Low":    [c - 1.0 for c in closes],
            "Close":  closes, "Volume": [1_000_000] * rows,
        },
        index=bdays,
    )
    _patch_yf(monkeypatch, df)

    # Pick a Saturday — bdate_range will have skipped it.
    sat = "2026-05-23"  # 2026-05-23 is a Saturday
    snap = regime.snapshot_regime_for_date(sat)
    assert snap is None


def test_snapshot_for_date_returns_none_when_insufficient_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = _build_spy_df(rows=400, end_date=datetime(2026, 5, 26, tzinfo=UTC))
    _patch_yf(monkeypatch, df)
    # An early date in the synthetic series — idx < 200 → None.
    early_date = (datetime(2026, 5, 26) - timedelta(days=399)).strftime("%Y-%m-%d")
    snap = regime.snapshot_regime_for_date(early_date)
    assert snap is None


def test_snapshot_for_date_rejects_bad_date_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_yf(monkeypatch, _build_spy_df(rows=400))
    with pytest.raises(regime.RegimeFetchError, match="invalid target_date"):
        regime.snapshot_regime_for_date("not-a-date")


# ────────────────────── backfill ──────────────────────


def _seed_closed_trade(
    *,
    ticker: str = "GOOGL",
    asset_class: str = "stock",
    direction: str = "call",
    signal_type: str = "ema21_pullback",
    opened: datetime,
    outcome: str = "win",
) -> int:
    sig_id = db.insert_signal(Signal(
        timestamp=opened,
        ticker=ticker,
        asset_class=asset_class,
        signal_type=signal_type,
        direction=direction,
        entry_price=100.0,
        take_profit=110.0,
        stop_loss=95.0,
    ))
    return db.insert_trade(Trade(
        signal_id=sig_id,
        opened_at=opened,
        closed_at=opened + timedelta(days=2),
        outcome=outcome,
        exit_price=105.0,
        pnl_pct=5.0,
    ))


def test_backfill_populates_trade_regimes(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = _build_spy_df(rows=400, drift=0.8, end_date=datetime(2026, 5, 26, tzinfo=UTC))
    _patch_yf(monkeypatch, df)

    _seed_closed_trade(opened=datetime(2026, 4, 1, tzinfo=UTC))
    _seed_closed_trade(opened=datetime(2026, 4, 5, tzinfo=UTC), ticker="META")
    _seed_closed_trade(opened=datetime(2026, 4, 5, tzinfo=UTC), ticker="AMZN")

    summary = regime.backfill_trade_regimes()
    assert summary["trades_updated"] == 3
    # 2 distinct dates → 2 regime_snapshots rows
    assert summary["dates_snapshotted"] == 2
    assert summary["errors"] == 0


def test_backfill_is_idempotent(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = _build_spy_df(rows=400, end_date=datetime(2026, 5, 26, tzinfo=UTC))
    _patch_yf(monkeypatch, df)
    _seed_closed_trade(opened=datetime(2026, 4, 1, tzinfo=UTC))

    first = regime.backfill_trade_regimes()
    second = regime.backfill_trade_regimes()
    assert first["trades_updated"] == 1
    assert second["trades_updated"] == 0
    assert second["dates_snapshotted"] == 0


def test_backfill_skips_open_trades(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = _build_spy_df(rows=400, end_date=datetime(2026, 5, 26, tzinfo=UTC))
    _patch_yf(monkeypatch, df)
    sid = db.insert_signal(Signal(
        timestamp=datetime(2026, 4, 1, tzinfo=UTC),
        ticker="GOOGL", asset_class="stock", signal_type="ema21_pullback",
        direction="call", entry_price=100.0, take_profit=110.0, stop_loss=95.0,
    ))
    db.insert_trade(Trade(
        signal_id=sid, opened_at=datetime(2026, 4, 1, tzinfo=UTC), outcome="open",
    ))
    summary = regime.backfill_trade_regimes()
    assert summary["trades_updated"] == 0


# ────────────────────── scanner integration ──────────────────────


def test_scanner_tags_new_trade_with_regime(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trading_bot import scanner

    def fake_get_current_regime(force_refresh: bool = False) -> regime.RegimeSnapshot:
        return regime.RegimeSnapshot(
            date="2026-05-26", regime="bull", spy_close=600.0,
            ema50=590.0, ema200=550.0, ema50_slope=1.0,
        )
    monkeypatch.setattr(scanner.regime, "get_current_regime", fake_get_current_regime)

    sid = scanner.log_signal({
        "ticker": "GOOGL", "asset_type": "stock", "direction": "CALL",
        "setup": "EMA21 Pullback", "price": 180.0,
        "take_profit": 185.0, "stop_loss": 178.0,
    })
    assert sid is not None
    trade = db.get_trade_by_signal_id(sid)
    assert trade is not None
    assert trade.market_regime == "bull"


def test_scanner_uses_unknown_when_regime_fetch_fails(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trading_bot import scanner

    def boom(force_refresh: bool = False) -> regime.RegimeSnapshot:
        raise regime.RegimeFetchError("network down")
    monkeypatch.setattr(scanner.regime, "get_current_regime", boom)

    sid = scanner.log_signal({
        "ticker": "GOOGL", "asset_type": "stock", "direction": "CALL",
        "setup": "EMA21 Pullback", "price": 180.0,
        "take_profit": 185.0, "stop_loss": 178.0,
    })
    assert sid is not None, "signal must still be logged even when regime fails"
    trade = db.get_trade_by_signal_id(sid)
    assert trade is not None
    assert trade.market_regime == "unknown"


def test_daily_regime_snapshot_job_inserts_row(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trading_bot import scanner

    snap = regime.RegimeSnapshot(
        date="2026-05-26", regime="bull",
        spy_close=612.45, ema50=598.21, ema200=571.88, ema50_slope=1.23,
    )

    def fake_get(force_refresh: bool = False) -> regime.RegimeSnapshot:
        return snap

    monkeypatch.setattr(scanner.regime, "get_current_regime", fake_get)
    scanner._run_daily_regime_snapshot()

    row = db.get_regime_snapshot("2026-05-26")
    assert row is not None
    assert row["regime"] == "bull"
    assert row["spy_close"] == pytest.approx(612.45)


def test_daily_regime_snapshot_job_is_idempotent(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trading_bot import scanner

    snap = regime.RegimeSnapshot(
        date="2026-05-26", regime="bull",
        spy_close=612.45, ema50=598.21, ema200=571.88, ema50_slope=1.23,
    )

    def fake_get(force_refresh: bool = False) -> regime.RegimeSnapshot:
        return snap

    monkeypatch.setattr(scanner.regime, "get_current_regime", fake_get)
    scanner._run_daily_regime_snapshot()
    scanner._run_daily_regime_snapshot()

    rows = db.get_regime_snapshots(limit=10)
    assert len(rows) == 1


def test_daily_regime_snapshot_job_swallows_errors(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_bot import scanner

    def boom(force_refresh: bool = False) -> regime.RegimeSnapshot:
        raise regime.RegimeFetchError("kaboom")

    monkeypatch.setattr(scanner.regime, "get_current_regime", boom)
    # Should not raise
    scanner._run_daily_regime_snapshot()
    assert "kaboom" in capsys.readouterr().err


# ────────────────────── last_cached_at ──────────────────────


def test_disk_cache_non_dict_payload_falls_through(
    monkeypatch: pytest.MonkeyPatch, isolate_regime_cache: Path,
) -> None:
    isolate_regime_cache.write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")
    calls = _patch_yf(monkeypatch, _build_spy_df())
    snap = regime.get_current_regime()
    assert len(calls) == 1
    assert snap is not None


def test_disk_cache_missing_keys_falls_through(
    monkeypatch: pytest.MonkeyPatch, isolate_regime_cache: Path,
) -> None:
    isolate_regime_cache.write_text(json.dumps({"cached_at": 123}), encoding="utf-8")
    calls = _patch_yf(monkeypatch, _build_spy_df())
    regime.get_current_regime()
    assert len(calls) == 1


def test_disk_cache_invalid_snapshot_falls_through(
    monkeypatch: pytest.MonkeyPatch, isolate_regime_cache: Path,
) -> None:
    payload = {
        "cached_at": datetime.now(UTC).isoformat(),
        "snapshot": {"date": "x", "wrong_field": True},
    }
    isolate_regime_cache.write_text(json.dumps(payload), encoding="utf-8")
    calls = _patch_yf(monkeypatch, _build_spy_df())
    regime.get_current_regime()
    assert len(calls) == 1


def test_disk_cache_write_failure_does_not_raise(
    monkeypatch: pytest.MonkeyPatch,
    isolate_regime_cache: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _patch_yf(monkeypatch, _build_spy_df())

    def boom(_path: Any, *_args: Any, **_kwargs: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", boom)
    regime.get_current_regime()  # must NOT raise
    assert "cache write error" in capsys.readouterr().err


def test_backfill_handles_snapshot_fetch_error(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """If snapshot_regime_for_date raises mid-loop, the bad date is counted as
    an error and the loop continues with remaining trades."""
    _seed_closed_trade(opened=datetime(2026, 4, 1, tzinfo=UTC))

    def boom(date_str: str) -> regime.RegimeSnapshot | None:
        raise regime.RegimeFetchError(f"fetch failed for {date_str}")

    monkeypatch.setattr("trading_bot.regime.snapshot_regime_for_date", boom)
    summary = regime.backfill_trade_regimes()
    assert summary["trades_updated"] == 0
    assert summary["errors"] == 1
    assert "fetch failed" in capsys.readouterr().err


def test_backfill_skips_when_date_has_no_data(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A trade opened on a non-trading date (weekend / holiday) is counted
    as skipped_no_data, not an error."""
    _seed_closed_trade(opened=datetime(2026, 4, 1, tzinfo=UTC))
    monkeypatch.setattr(
        "trading_bot.regime.snapshot_regime_for_date", lambda _d: None,
    )
    summary = regime.backfill_trade_regimes()
    assert summary["trades_updated"] == 0
    assert summary["skipped_no_data"] == 1


def test_last_cached_at_returns_none_when_no_cache() -> None:
    assert regime.last_cached_at() is None


def test_last_cached_at_returns_disk_time_after_warmup(
    monkeypatch: pytest.MonkeyPatch, isolate_regime_cache: Path,
) -> None:
    _patch_yf(monkeypatch, _build_spy_df())
    regime.get_current_regime()
    when = regime.last_cached_at()
    assert when is not None
    assert when.tzinfo is not None

    # Even after wiping memory cache, disk should answer.
    regime._reset_cache_for_tests()
    again = regime.last_cached_at()
    assert again is not None
