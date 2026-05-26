"""Tests for trading_bot.vix — Phase 2.2 VIX context.

yfinance is mocked throughout. The disk cache (.vix_cache.json) is
redirected to ``tmp_path`` per test so we don't trample any real cache.
Structure mirrors ``tests/test_regime.py`` deliberately — same coverage
shape on the volatility axis.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from trading_bot import db, vix
from trading_bot.models import Signal, Trade

# ────────────────────── fixtures + helpers ──────────────────────


@pytest.fixture(autouse=True)
def isolate_vix_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Path]:
    """Redirect the disk cache + reset memory cache for every test."""
    cache_path = tmp_path / ".vix_cache.json"
    monkeypatch.setattr("trading_bot.vix._CACHE_PATH", cache_path)
    vix._reset_cache_for_tests()
    yield cache_path
    vix._reset_cache_for_tests()


def _build_vix_df(
    *,
    closes: list[float] | None = None,
    end_date: datetime | None = None,
) -> pd.DataFrame:
    """Build a synthetic ^VIX OHLC DataFrame with given close values."""
    if closes is None:
        closes = [18.0 + (i % 5) * 0.1 for i in range(30)]
    anchor = end_date if end_date is not None else datetime(2026, 5, 26, tzinfo=UTC)
    n = len(closes)
    # Skip weekends — VIX doesn't trade Sat/Sun.
    idx = pd.bdate_range(end=anchor, periods=n)
    return pd.DataFrame(
        {
            "Open":   closes,
            "High":   [c + 0.5 for c in closes],
            "Low":    [max(0.0, c - 0.5) for c in closes],
            "Close":  closes,
            "Volume": [0] * n,
        },
        index=idx,
    )


def _patch_yf(
    monkeypatch: pytest.MonkeyPatch, df: pd.DataFrame | Exception | None,
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake_download(ticker: str, **kwargs: Any) -> pd.DataFrame | None:
        calls.append({"ticker": ticker, **kwargs})
        if isinstance(df, Exception):
            raise df
        return df

    monkeypatch.setattr("trading_bot.vix.yf.download", fake_download)
    return calls


# ────────────────────── classification ──────────────────────


@pytest.mark.parametrize("level", [0.0, 12.0, 15.5, 19.99])
def test_classify_low(level: float) -> None:
    assert vix.classify(level) == "low"


@pytest.mark.parametrize("level", [20.0, 25.5, 29.99])
def test_classify_elevated(level: float) -> None:
    assert vix.classify(level) == "elevated"


@pytest.mark.parametrize("level", [30.0, 35.0, 39.99])
def test_classify_high(level: float) -> None:
    assert vix.classify(level) == "high"


@pytest.mark.parametrize("level", [40.0, 55.0, 80.0])
def test_classify_extreme(level: float) -> None:
    assert vix.classify(level) == "extreme"


def test_classify_rejects_negative() -> None:
    with pytest.raises(ValueError, match="must be >= 0"):
        vix.classify(-1.0)


# ────────────────────── fetch + caching ──────────────────────


def test_get_current_vix_single_fetch_on_cold_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_yf(monkeypatch, _build_vix_df())
    snap = vix.get_current_vix()
    assert snap.vix_band in {"low", "elevated", "high", "extreme"}
    assert snap.vix_level > 0
    assert len(calls) == 1
    assert calls[0]["ticker"] == "^VIX"


def test_memory_cache_within_1h(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_yf(monkeypatch, _build_vix_df())
    first = vix.get_current_vix()
    second = vix.get_current_vix()
    assert first == second
    assert len(calls) == 1


def test_force_refresh_bypasses_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_yf(monkeypatch, _build_vix_df())
    vix.get_current_vix()
    vix.get_current_vix(force_refresh=True)
    assert len(calls) == 2


def test_disk_cache_persists_across_process_restart(
    monkeypatch: pytest.MonkeyPatch, isolate_vix_cache: Path,
) -> None:
    _patch_yf(monkeypatch, _build_vix_df())
    first = vix.get_current_vix()
    assert isolate_vix_cache.exists()

    vix._reset_cache_for_tests()
    calls = _patch_yf(monkeypatch, RuntimeError("net down"))
    second = vix.get_current_vix()
    assert second == first
    assert calls == []


def test_disk_cache_expires_after_1h(
    monkeypatch: pytest.MonkeyPatch, isolate_vix_cache: Path,
) -> None:
    """A snapshot older than 1h should be discarded and re-fetched."""
    stale_snap = vix.VixSnapshot(
        date="2026-05-25", vix_level=17.0, vix_band="low",
        captured_at=(datetime.now(UTC) - timedelta(hours=2)).isoformat(),
    )
    payload = {
        "cached_at": (datetime.now(UTC) - timedelta(hours=2)).isoformat(),
        "snapshot": {
            "date": stale_snap.date,
            "vix_level": stale_snap.vix_level,
            "vix_band": stale_snap.vix_band,
            "captured_at": stale_snap.captured_at,
        },
    }
    isolate_vix_cache.write_text(json.dumps(payload), encoding="utf-8")
    vix._reset_cache_for_tests()

    calls = _patch_yf(monkeypatch, _build_vix_df())
    vix.get_current_vix()
    assert len(calls) == 1  # stale cache rejected, re-fetched


def test_malformed_yfinance_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    bad = pd.DataFrame({"Open": [1.0], "Volume": [1]})
    _patch_yf(monkeypatch, bad)
    with pytest.raises(vix.VixFetchError):
        vix.get_current_vix()


def test_empty_yfinance_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_yf(monkeypatch, pd.DataFrame())
    with pytest.raises(vix.VixFetchError):
        vix.get_current_vix()


def test_yfinance_exception_raises_vix_fetch_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_yf(monkeypatch, RuntimeError("rate limited"))
    with pytest.raises(vix.VixFetchError, match="rate limited"):
        vix.get_current_vix()


def test_multiindex_columns_flattened(monkeypatch: pytest.MonkeyPatch) -> None:
    df = _build_vix_df()
    df.columns = pd.MultiIndex.from_product([df.columns, ["^VIX"]])
    _patch_yf(monkeypatch, df)
    snap = vix.get_current_vix()
    assert snap.vix_band in {"low", "elevated", "high", "extreme"}


def test_close_column_all_nan_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    df = _build_vix_df()
    df["Close"] = float("nan")
    _patch_yf(monkeypatch, df)
    with pytest.raises(vix.VixFetchError, match="no usable Close"):
        vix.get_current_vix()


def test_disk_cache_corrupt_falls_through(
    monkeypatch: pytest.MonkeyPatch, isolate_vix_cache: Path,
) -> None:
    isolate_vix_cache.write_text("{not json", encoding="utf-8")
    calls = _patch_yf(monkeypatch, _build_vix_df())
    snap = vix.get_current_vix()
    assert len(calls) == 1
    assert snap is not None


def test_disk_cache_non_dict_payload_falls_through(
    monkeypatch: pytest.MonkeyPatch, isolate_vix_cache: Path,
) -> None:
    isolate_vix_cache.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    calls = _patch_yf(monkeypatch, _build_vix_df())
    vix.get_current_vix()
    assert len(calls) == 1


def test_disk_cache_invalid_snapshot_falls_through(
    monkeypatch: pytest.MonkeyPatch, isolate_vix_cache: Path,
) -> None:
    payload = {
        "cached_at": datetime.now(UTC).isoformat(),
        "snapshot": {"date": "x", "wrong_field": True},
    }
    isolate_vix_cache.write_text(json.dumps(payload), encoding="utf-8")
    calls = _patch_yf(monkeypatch, _build_vix_df())
    vix.get_current_vix()
    assert len(calls) == 1


def test_disk_cache_write_failure_does_not_raise(
    monkeypatch: pytest.MonkeyPatch,
    isolate_vix_cache: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _patch_yf(monkeypatch, _build_vix_df())

    def boom(_path: Any, *_args: Any, **_kwargs: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", boom)
    vix.get_current_vix()  # must NOT raise
    assert "cache write error" in capsys.readouterr().err


# ────────────────────── snapshot_vix_for_date ──────────────────────


def test_snapshot_for_known_trading_day(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = _build_vix_df(end_date=datetime(2026, 5, 26, tzinfo=UTC))
    _patch_yf(monkeypatch, df)
    # 2026-05-20 is a Wednesday — guaranteed business day in the bdate_range.
    snap = vix.snapshot_vix_for_date("2026-05-20")
    assert snap is not None
    assert snap.date == "2026-05-20"
    assert snap.vix_band in {"low", "elevated", "high", "extreme"}


def test_snapshot_for_weekend_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = _build_vix_df(end_date=datetime(2026, 5, 26, tzinfo=UTC))
    _patch_yf(monkeypatch, df)
    # 2026-05-23 is a Saturday
    assert vix.snapshot_vix_for_date("2026-05-23") is None


def test_snapshot_for_future_date_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = _build_vix_df(end_date=datetime(2026, 5, 26, tzinfo=UTC))
    _patch_yf(monkeypatch, df)
    assert vix.snapshot_vix_for_date("2030-01-01") is None


def test_snapshot_for_date_rejects_bad_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_yf(monkeypatch, _build_vix_df())
    with pytest.raises(vix.VixFetchError, match="invalid target_date"):
        vix.snapshot_vix_for_date("not-a-date")


# ────────────────────── backfill ──────────────────────


def _seed_closed_trade(
    *,
    ticker: str = "GOOGL",
    asset_class: str = "stock",
    direction: str = "call",
    signal_type: str = "ema21_pullback",
    opened: datetime,
    outcome: str = "win",
    market_regime: str | None = None,
) -> int:
    sig_id = db.insert_signal(Signal(
        timestamp=opened, ticker=ticker, asset_class=asset_class,
        signal_type=signal_type, direction=direction,
        entry_price=100.0, take_profit=110.0, stop_loss=95.0,
    ))
    return db.insert_trade(Trade(
        signal_id=sig_id, opened_at=opened,
        closed_at=opened + timedelta(days=2),
        outcome=outcome, exit_price=105.0, pnl_pct=5.0,
        market_regime=market_regime,
    ))


def test_backfill_populates_trade_vix(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = _build_vix_df(end_date=datetime(2026, 5, 26, tzinfo=UTC))
    _patch_yf(monkeypatch, df)

    base = datetime(2026, 5, 18, tzinfo=UTC)  # a Monday inside bdate window
    _seed_closed_trade(opened=base)
    _seed_closed_trade(opened=base + timedelta(days=1), ticker="META")
    _seed_closed_trade(opened=base + timedelta(days=1), ticker="AMZN")

    summary = vix.backfill_trade_vix()
    assert summary["trades_updated"] == 3
    assert summary["dates_snapshotted"] == 2
    assert summary["errors"] == 0


def test_backfill_is_idempotent(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = _build_vix_df(end_date=datetime(2026, 5, 26, tzinfo=UTC))
    _patch_yf(monkeypatch, df)
    _seed_closed_trade(opened=datetime(2026, 5, 18, tzinfo=UTC))

    first = vix.backfill_trade_vix()
    second = vix.backfill_trade_vix()
    assert first["trades_updated"] == 1
    assert second["trades_updated"] == 0


def test_backfill_skips_open_trades(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = _build_vix_df(end_date=datetime(2026, 5, 26, tzinfo=UTC))
    _patch_yf(monkeypatch, df)
    sid = db.insert_signal(Signal(
        timestamp=datetime(2026, 5, 18, tzinfo=UTC),
        ticker="GOOGL", asset_class="stock", signal_type="ema21_pullback",
        direction="call", entry_price=100.0, take_profit=110.0, stop_loss=95.0,
    ))
    db.insert_trade(Trade(
        signal_id=sid, opened_at=datetime(2026, 5, 18, tzinfo=UTC),
        outcome="open",
    ))
    summary = vix.backfill_trade_vix()
    assert summary["trades_updated"] == 0


def test_backfill_independent_of_regime_backfill(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A trade that already carries a regime tag but no VIX tag is a
    legitimate candidate for VIX backfill. The two axes are independent —
    backfilling one does not touch the other."""
    df = _build_vix_df(end_date=datetime(2026, 5, 26, tzinfo=UTC))
    _patch_yf(monkeypatch, df)

    _seed_closed_trade(
        opened=datetime(2026, 5, 18, tzinfo=UTC),
        market_regime="bull",
    )
    summary = vix.backfill_trade_vix()
    assert summary["trades_updated"] == 1

    # The regime tag survived.
    trade = db.get_trade_by_signal_id(1)
    assert trade is not None
    assert trade.market_regime == "bull"
    assert trade.vix_band in {"low", "elevated", "high", "extreme"}


def test_backfill_handles_snapshot_fetch_error(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _seed_closed_trade(opened=datetime(2026, 5, 18, tzinfo=UTC))

    def boom(_d: str) -> vix.VixSnapshot | None:
        raise vix.VixFetchError("kaboom")

    monkeypatch.setattr("trading_bot.vix.snapshot_vix_for_date", boom)
    summary = vix.backfill_trade_vix()
    assert summary["errors"] == 1
    assert "kaboom" in capsys.readouterr().err


def test_backfill_skips_when_date_has_no_data(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_closed_trade(opened=datetime(2026, 5, 18, tzinfo=UTC))
    monkeypatch.setattr("trading_bot.vix.snapshot_vix_for_date", lambda _d: None)
    summary = vix.backfill_trade_vix()
    assert summary["trades_updated"] == 0
    assert summary["skipped_no_data"] == 1


# ────────────────────── scanner integration ──────────────────────


def test_scanner_tags_new_trade_with_vix(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trading_bot import regime, scanner

    monkeypatch.setattr(
        scanner.regime, "get_current_regime",
        lambda **_: regime.RegimeSnapshot(
            date="2026-05-26", regime="bull", spy_close=600.0,
            ema50=590.0, ema200=550.0, ema50_slope=1.0,
        ),
    )
    monkeypatch.setattr(
        scanner.vix, "get_current_vix",
        lambda **_: vix.VixSnapshot(
            date="2026-05-26", vix_level=18.4, vix_band="low",
            captured_at=datetime.now(UTC).isoformat(),
        ),
    )

    sid = scanner.log_signal({
        "ticker": "GOOGL", "asset_type": "stock", "direction": "CALL",
        "setup": "EMA21 Pullback", "price": 180.0,
        "take_profit": 185.0, "stop_loss": 178.0,
    })
    assert sid is not None
    trade = db.get_trade_by_signal_id(sid)
    assert trade is not None
    assert trade.vix_level == pytest.approx(18.4)
    assert trade.vix_band == "low"


def test_scanner_uses_unknown_when_vix_fetch_fails(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trading_bot import regime, scanner

    monkeypatch.setattr(
        scanner.regime, "get_current_regime",
        lambda **_: regime.RegimeSnapshot(
            date="2026-05-26", regime="bull", spy_close=600.0,
            ema50=590.0, ema200=550.0, ema50_slope=1.0,
        ),
    )

    def boom(**_: object) -> vix.VixSnapshot:
        raise vix.VixFetchError("net down")
    monkeypatch.setattr(scanner.vix, "get_current_vix", boom)

    sid = scanner.log_signal({
        "ticker": "GOOGL", "asset_type": "stock", "direction": "CALL",
        "setup": "EMA21 Pullback", "price": 180.0,
        "take_profit": 185.0, "stop_loss": 178.0,
    })
    assert sid is not None, "signal must still fire when VIX fetch fails"
    trade = db.get_trade_by_signal_id(sid)
    assert trade is not None
    assert trade.vix_band == "unknown"
    assert trade.vix_level is None


def test_daily_vix_snapshot_job_inserts_row(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trading_bot import scanner

    snap = vix.VixSnapshot(
        date="2026-05-26", vix_level=18.4, vix_band="low",
        captured_at=datetime.now(UTC).isoformat(),
    )
    monkeypatch.setattr(scanner.vix, "get_current_vix", lambda **_: snap)
    scanner._run_daily_vix_snapshot()

    row = db.get_vix_snapshot("2026-05-26")
    assert row is not None
    assert row["vix_band"] == "low"
    assert row["vix_level"] == pytest.approx(18.4)


def test_daily_vix_snapshot_job_is_idempotent(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trading_bot import scanner

    snap = vix.VixSnapshot(
        date="2026-05-26", vix_level=18.4, vix_band="low",
        captured_at=datetime.now(UTC).isoformat(),
    )
    monkeypatch.setattr(scanner.vix, "get_current_vix", lambda **_: snap)
    scanner._run_daily_vix_snapshot()
    scanner._run_daily_vix_snapshot()
    assert len(db.get_vix_snapshots(limit=10)) == 1


def test_daily_vix_snapshot_job_swallows_errors(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_bot import scanner

    def boom(**_: object) -> vix.VixSnapshot:
        raise vix.VixFetchError("kaboom")

    monkeypatch.setattr(scanner.vix, "get_current_vix", boom)
    scanner._run_daily_vix_snapshot()  # must NOT raise
    assert "kaboom" in capsys.readouterr().err


# ────────────────────── last_cached_at ──────────────────────


def test_last_cached_at_returns_none_when_no_cache() -> None:
    assert vix.last_cached_at() is None


def test_last_cached_at_returns_disk_time_after_warmup(
    monkeypatch: pytest.MonkeyPatch, isolate_vix_cache: Path,
) -> None:
    _patch_yf(monkeypatch, _build_vix_df())
    vix.get_current_vix()
    when = vix.last_cached_at()
    assert when is not None
    vix._reset_cache_for_tests()
    again = vix.last_cached_at()
    assert again is not None
