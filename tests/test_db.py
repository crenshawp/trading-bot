"""Tests for trading_bot.db — schema, CRUD, validation, and dedupe."""

import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pytest

from trading_bot import db
from trading_bot.models import DailyPerf, Signal, Trade


def _make_signal(**overrides: Any) -> Signal:
    base: dict[str, Any] = {
        "timestamp": datetime(2026, 5, 25, 9, 31),
        "ticker": "GOOGL",
        "asset_class": "stock",
        "signal_type": "ema21_pullback",
        "direction": "call",
        "entry_price": 180.0,
    }
    base.update(overrides)
    return Signal(**base)


# ---- schema / init ----


def test_init_db_is_idempotent(tmp_db: Path) -> None:
    db.init_db()  # tmp_db already called init_db once; second call must not error
    db.init_db()
    assert db.schema_version() == 3


def test_schema_version_is_1_after_init(tmp_db: Path) -> None:
    assert db.schema_version() == 3


# ---- signals ----


def test_insert_signal_round_trip(tmp_db: Path) -> None:
    sid = db.insert_signal(_make_signal())
    assert sid >= 1
    signals = db.get_signals()
    assert len(signals) == 1
    fetched = signals[0]
    assert fetched.ticker == "GOOGL"
    assert fetched.entry_price == 180.0
    assert fetched.id == sid


def test_insert_signal_dedupe_returns_existing_id(tmp_db: Path) -> None:
    sid1 = db.insert_signal(_make_signal())
    sid2 = db.insert_signal(_make_signal())
    assert sid1 == sid2
    assert len(db.get_signals()) == 1


def test_insert_signal_rejects_invalid_asset_class(tmp_db: Path) -> None:
    with pytest.raises(ValueError, match="asset_class"):
        db.insert_signal(_make_signal(asset_class="commodity"))


def test_insert_signal_rejects_invalid_direction(tmp_db: Path) -> None:
    with pytest.raises(ValueError, match="direction"):
        db.insert_signal(_make_signal(direction="hodl"))


def test_get_signals_filters_by_ticker(tmp_db: Path) -> None:
    db.insert_signal(_make_signal(ticker="GOOGL"))
    db.insert_signal(
        _make_signal(ticker="META", timestamp=datetime(2026, 5, 25, 9, 32))
    )
    result = db.get_signals(ticker="META")
    assert len(result) == 1
    assert result[0].ticker == "META"


def test_get_signals_filters_by_since(tmp_db: Path) -> None:
    db.insert_signal(_make_signal(timestamp=datetime(2026, 5, 24, 10, 0)))
    db.insert_signal(
        _make_signal(timestamp=datetime(2026, 5, 25, 10, 0), ticker="META")
    )
    result = db.get_signals(since=datetime(2026, 5, 25))
    assert len(result) == 1
    assert result[0].ticker == "META"


# ---- trades ----


def test_insert_trade_requires_valid_signal_id(tmp_db: Path) -> None:
    bad_trade = Trade(signal_id=9999, opened_at=datetime(2026, 5, 25))
    with pytest.raises(sqlite3.IntegrityError):
        db.insert_trade(bad_trade)


def test_update_trade_modifies_only_specified_fields(tmp_db: Path) -> None:
    sid = db.insert_signal(_make_signal())
    tid = db.insert_trade(Trade(signal_id=sid, opened_at=datetime(2026, 5, 25)))
    db.update_trade(tid, outcome="win", pnl_pct=2.5)
    conn = db.get_connection()
    try:
        row = conn.execute("SELECT * FROM trades WHERE id=?", (tid,)).fetchone()
    finally:
        conn.close()
    assert row is not None
    assert row["outcome"] == "win"
    assert row["pnl_pct"] == 2.5
    assert row["notes"] is None
    assert row["exit_price"] is None


def test_update_trade_rejects_unknown_field(tmp_db: Path) -> None:
    sid = db.insert_signal(_make_signal())
    tid = db.insert_trade(Trade(signal_id=sid, opened_at=datetime(2026, 5, 25)))
    with pytest.raises(ValueError, match="Unknown trade field"):
        db.update_trade(tid, color="red")


def test_update_trade_rejects_invalid_outcome(tmp_db: Path) -> None:
    sid = db.insert_signal(_make_signal())
    tid = db.insert_trade(Trade(signal_id=sid, opened_at=datetime(2026, 5, 25)))
    with pytest.raises(ValueError, match="outcome"):
        db.update_trade(tid, outcome="moonshot")


def test_update_trade_with_no_fields_is_noop(tmp_db: Path) -> None:
    sid = db.insert_signal(_make_signal())
    tid = db.insert_trade(Trade(signal_id=sid, opened_at=datetime(2026, 5, 25)))
    db.update_trade(tid)  # explicit no-op, no exception


def test_update_trade_normalizes_datetime(tmp_db: Path) -> None:
    sid = db.insert_signal(_make_signal())
    tid = db.insert_trade(Trade(signal_id=sid, opened_at=datetime(2026, 5, 25)))
    db.update_trade(tid, closed_at=datetime(2026, 5, 26, 16, 0), outcome="win")
    conn = db.get_connection()
    try:
        row = conn.execute("SELECT * FROM trades WHERE id=?", (tid,)).fetchone()
    finally:
        conn.close()
    assert row["closed_at"] == "2026-05-26T16:00:00"


def test_insert_trade_rejects_invalid_outcome(tmp_db: Path) -> None:
    sid = db.insert_signal(_make_signal())
    with pytest.raises(ValueError, match="outcome"):
        db.insert_trade(
            Trade(signal_id=sid, opened_at=datetime(2026, 5, 25), outcome="hodl")
        )


def test_get_open_trades_returns_unclosed(tmp_db: Path) -> None:
    sid = db.insert_signal(_make_signal())
    open_tid = db.insert_trade(Trade(signal_id=sid, opened_at=datetime(2026, 5, 25)))
    closed_tid = db.insert_trade(
        Trade(
            signal_id=sid,
            opened_at=datetime(2026, 5, 24),
            closed_at=datetime(2026, 5, 24, 16, 0),
            outcome="win",
        )
    )
    open_ids = [t.id for t in db.get_open_trades()]
    assert open_tid in open_ids
    assert closed_tid not in open_ids


# ---- daily performance ----


def test_upsert_daily_perf_updates_existing(tmp_db: Path) -> None:
    today = "2026-05-25"
    db.upsert_daily_performance(DailyPerf(date=today, signals_fired=3))
    db.upsert_daily_performance(DailyPerf(date=today, signals_fired=7, wins=2))
    perf = db.get_daily_performance(date(2026, 5, 25))
    assert perf is not None
    assert perf.signals_fired == 7
    assert perf.wins == 2


def test_get_daily_performance_returns_none_when_absent(tmp_db: Path) -> None:
    assert db.get_daily_performance(date(2026, 1, 1)) is None


# ---- diagnostics ----


def test_get_table_counts_zero_on_fresh_db(tmp_db: Path) -> None:
    counts = db.get_table_counts()
    assert counts == {
        "signals": 0, "trades": 0, "daily_performance": 0,
        "regime_snapshots": 0, "vix_snapshots": 0,
    }


def test_schema_version_returns_zero_on_uninitialized_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("trading_bot.config.DB_PATH", tmp_path / "empty.db")
    # init_db NOT called — schema_version should report 0 gracefully
    assert db.schema_version() == 0
