"""Tests for trading_bot.db — schema, CRUD, validation, and dedupe."""

import sqlite3
from datetime import date, datetime, timedelta
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
    assert db.schema_version() == 30


def test_schema_version_is_1_after_init(tmp_db: Path) -> None:
    assert db.schema_version() == 30


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
        "predictions": 0, "settings": 0, "active_watchlist": 0,
        "discovery_results": 0, "shadow_evaluations": 0,
        "watchlist_transitions": 0,
        "signal_pair_status": 0, "signal_pair_transitions": 0,
        "optimization_runs": 0, "readiness_state": 0,
        "option_positions": 0, "long_term_positions": 0,
        "equity_snapshots": 0, "plan_executions": 0,
        "pending_orders": 0,
    }


# ---- sentiment provenance (cross-cutting audit) ----


def test_sentiment_neutral_success_and_fail_soft_round_trip(tmp_db: Path) -> None:
    successful_id = db.insert_signal(_make_signal(ticker="GOOGL"))
    failed_id = db.insert_signal(_make_signal(ticker="META"))
    db.insert_trade(Trade(
        signal_id=successful_id,
        opened_at=datetime(2026, 6, 1, 10, 0),
        sentiment_score=0.0,
        sentiment_label="neutral",
        sentiment_ok=True,
        sentiment_rationale="Mixed headlines balanced out.",
    ))
    db.insert_trade(Trade(
        signal_id=failed_id,
        opened_at=datetime(2026, 6, 1, 11, 0),
        sentiment_score=0.0,
        sentiment_label="neutral",
        sentiment_ok=False,
        sentiment_rationale="no news data",
    ))

    successful = db.get_trade_by_signal_id(successful_id)
    failed = db.get_trade_by_signal_id(failed_id)
    assert successful is not None and failed is not None
    assert successful.sentiment_ok is True
    assert successful.sentiment_rationale == "Mixed headlines balanced out."
    assert failed.sentiment_ok is False
    assert failed.sentiment_rationale == "no news data"

    recent = {row["ticker"]: row for row in db.get_recent_trade_sentiment()}
    assert recent["GOOGL"]["sentiment_ok"] is True
    assert recent["GOOGL"]["sentiment_rationale"] == "Mixed headlines balanced out."
    assert recent["META"]["sentiment_ok"] is False
    assert recent["META"]["sentiment_rationale"] == "no news data"


def test_legacy_sentiment_label_keeps_unknown_provenance(tmp_db: Path) -> None:
    signal_id = db.insert_signal(_make_signal())
    db.insert_trade(Trade(
        signal_id=signal_id,
        opened_at=datetime(2026, 6, 1, 10, 0),
        sentiment_score=0.0,
        sentiment_label="neutral",
    ))

    trade = db.get_trade_by_signal_id(signal_id)
    assert trade is not None
    assert trade.sentiment_ok is None
    assert trade.sentiment_rationale is None
    rows = db.get_recent_trade_sentiment()
    assert len(rows) == 1
    assert rows[0]["sentiment_ok"] is None
    assert rows[0]["sentiment_rationale"] is None


def test_indicator_unknown_success_and_fail_soft_round_trip(tmp_db: Path) -> None:
    successful_id = db.insert_signal(_make_signal(ticker="GOOGL"))
    failed_id = db.insert_signal(_make_signal(ticker="META"))
    db.insert_trade(Trade(
        signal_id=successful_id,
        opened_at=datetime(2026, 6, 1, 10, 0),
        ind_ok=True,
        ind_vol_regime="unknown",
        ind_concentration="unknown",
    ))
    db.insert_trade(Trade(
        signal_id=failed_id,
        opened_at=datetime(2026, 6, 1, 11, 0),
        ind_ok=False,
        ind_vol_regime="unknown",
        ind_concentration="unknown",
    ))

    successful = db.get_trade_by_signal_id(successful_id)
    failed = db.get_trade_by_signal_id(failed_id)
    assert successful is not None and failed is not None
    assert successful.ind_ok is True
    assert failed.ind_ok is False
    recent = {row["ticker"]: row for row in db.get_recent_trade_indicators()}
    assert recent["GOOGL"]["ind_ok"] is True
    assert recent["META"]["ind_ok"] is False
    assert recent["GOOGL"]["ind_vol_regime"] == "unknown"
    assert recent["META"]["ind_vol_regime"] == "unknown"


def test_legacy_indicator_context_keeps_unknown_provenance(tmp_db: Path) -> None:
    signal_id = db.insert_signal(_make_signal())
    db.insert_trade(Trade(
        signal_id=signal_id,
        opened_at=datetime(2026, 6, 1, 10, 0),
        ind_vol_regime="unknown",
        ind_concentration="unknown",
    ))

    trade = db.get_trade_by_signal_id(signal_id)
    assert trade is not None and trade.ind_ok is None
    rows = db.get_recent_trade_indicators()
    assert len(rows) == 1
    assert rows[0]["ind_ok"] is None


def test_schema_version_returns_zero_on_uninitialized_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("trading_bot.config.DB_PATH", tmp_path / "empty.db")
    # init_db NOT called — schema_version should report 0 gracefully
    assert db.schema_version() == 0


# ---- indicator-family context (Phase 6) ----


def test_get_recent_trade_indicators_only_returns_scored_rows(tmp_db: Path) -> None:
    # An indicator-scored (active stock) trade...
    sid_scored = db.insert_signal(_make_signal(ticker="GOOGL"))
    db.insert_trade(Trade(
        signal_id=sid_scored, opened_at=datetime(2026, 6, 1, 10, 0), outcome="win",
        ind_ok=True,
        ind_atr=2.5, ind_realized_vol=0.018, ind_vol_regime="normal",
        ind_rsi=54.3, ind_adx=27.1, ind_obv=1_234_567.0,
        ind_correlation=0.42, ind_concentration="moderate",
    ))
    # ...and an un-scored (crypto/shadow) trade with ind_vol_regime left NULL.
    sid_plain = db.insert_signal(
        _make_signal(ticker="BTC-USD", asset_class="crypto", direction="long")
    )
    db.insert_trade(Trade(
        signal_id=sid_plain, opened_at=datetime(2026, 6, 1, 11, 0), outcome="loss",
    ))

    rows = db.get_recent_trade_indicators()
    assert len(rows) == 1                       # NULL-regime row excluded
    row = rows[0]
    assert row["ticker"] == "GOOGL"
    assert row["ind_ok"] is True
    assert row["ind_vol_regime"] == "normal"
    assert row["ind_rsi"] == pytest.approx(54.3)
    assert row["ind_adx"] == pytest.approx(27.1)
    assert row["ind_obv"] == pytest.approx(1_234_567.0)
    assert row["ind_correlation"] == pytest.approx(0.42)
    assert row["ind_concentration"] == "moderate"
    assert row["outcome"] == "win"


def test_get_recent_trade_indicators_newest_first_and_limit(tmp_db: Path) -> None:
    for i, ticker in enumerate(["AAA", "BBB", "CCC"]):
        sid = db.insert_signal(_make_signal(ticker=ticker, signal_type=f"s{i}"))
        db.insert_trade(Trade(
            signal_id=sid, opened_at=datetime(2026, 6, 1, 10 + i, 0),
            outcome="open", ind_ok=True, ind_vol_regime="high",
        ))
    rows = db.get_recent_trade_indicators(limit=2)
    assert [r["ticker"] for r in rows] == ["CCC", "BBB"]   # newest first, capped at 2


# ---- risk recommendation context (Phase 7) ----


def test_risk_unknown_success_and_fail_soft_round_trip(tmp_db: Path) -> None:
    successful_id = db.insert_signal(_make_signal(ticker="GOOGL"))
    failed_id = db.insert_signal(_make_signal(ticker="META"))
    db.insert_trade(Trade(
        signal_id=successful_id,
        opened_at=datetime(2026, 6, 1, 10, 0),
        risk_ok=True,
        risk_reason="ok",
        risk_portfolio_verdict="unknown",
        risk_position_verdict="unknown",
        risk_cluster_verdict="unknown",
    ))
    db.insert_trade(Trade(
        signal_id=failed_id,
        opened_at=datetime(2026, 6, 1, 11, 0),
        risk_ok=False,
        risk_reason="size unavailable: missing/zero ATR",
        risk_portfolio_verdict="unknown",
        risk_position_verdict="unknown",
        risk_cluster_verdict="unknown",
    ))

    successful = db.get_trade_by_signal_id(successful_id)
    failed = db.get_trade_by_signal_id(failed_id)
    assert successful is not None and failed is not None
    assert successful.risk_ok is True and successful.risk_reason == "ok"
    assert failed.risk_ok is False
    assert failed.risk_reason == "size unavailable: missing/zero ATR"
    recent = {row["ticker"]: row for row in db.get_recent_trade_risk()}
    assert recent["GOOGL"]["risk_ok"] is True
    assert recent["META"]["risk_ok"] is False
    assert recent["META"]["risk_reason"] == "size unavailable: missing/zero ATR"


def test_legacy_risk_context_keeps_unknown_provenance(tmp_db: Path) -> None:
    signal_id = db.insert_signal(_make_signal())
    db.insert_trade(Trade(
        signal_id=signal_id,
        opened_at=datetime(2026, 6, 1, 10, 0),
        risk_portfolio_verdict="unknown",
        risk_position_verdict="unknown",
        risk_cluster_verdict="unknown",
    ))

    trade = db.get_trade_by_signal_id(signal_id)
    assert trade is not None
    assert trade.risk_ok is None and trade.risk_reason is None
    rows = db.get_recent_trade_risk()
    assert len(rows) == 1
    assert rows[0]["risk_ok"] is None and rows[0]["risk_reason"] is None


def test_get_recent_trade_risk_only_returns_assessed_rows(tmp_db: Path) -> None:
    # A risk-assessed (active stock) trade...
    sid = db.insert_signal(_make_signal(ticker="GOOGL"))
    db.insert_trade(Trade(
        signal_id=sid, opened_at=datetime(2026, 6, 1, 10, 0), outcome="win",
        risk_ok=True, risk_reason="ok",
        risk_recommended_size=20.0, risk_pct=0.6, risk_position_pct=20.0,
        risk_capped=True, risk_total_pct=4.5, risk_portfolio_verdict="ok",
        risk_position_verdict="would-exceed-position",
        risk_cluster_pct=3.0, risk_cluster_verdict="ok",
    ))
    # ...and a crypto/shadow trade with no risk recommendation (verdict NULL).
    sid2 = db.insert_signal(
        _make_signal(ticker="BTC-USD", asset_class="crypto", direction="long")
    )
    db.insert_trade(Trade(
        signal_id=sid2, opened_at=datetime(2026, 6, 1, 11, 0), outcome="loss",
    ))

    rows = db.get_recent_trade_risk()
    assert len(rows) == 1                       # NULL-verdict row excluded
    r = rows[0]
    assert r["ticker"] == "GOOGL"
    assert r["risk_ok"] is True and r["risk_reason"] == "ok"
    assert r["risk_recommended_size"] == pytest.approx(20.0)
    assert r["risk_pct"] == pytest.approx(0.6)
    assert r["risk_capped"] is True
    assert r["risk_portfolio_verdict"] == "ok"
    assert r["risk_position_verdict"] == "would-exceed-position"
    assert r["outcome"] == "win"


def test_get_recent_signal_risk_grades_reads_signals_newest_first(
    tmp_db: Path,
) -> None:
    db.insert_signal(_make_signal(
        ticker="GOOGL", timestamp=datetime(2026, 6, 1, 10, 0),
        earnings_risk="MEDIUM — Earnings in 10 days", news_risk="LOW",
    ))
    db.insert_signal(_make_signal(
        ticker="META", timestamp=datetime(2026, 6, 1, 11, 0),
        earnings_risk="LOW", news_risk="MEDIUM — 1 medium-risk article",
    ))

    rows = db.get_recent_signal_risk_grades(limit=1)

    assert rows == [{
        "timestamp": "2026-06-01T11:00:00",
        "ticker": "META",
        "signal_type": "ema21_pullback",
        "earnings_risk": "LOW",
        "news_risk": "MEDIUM — 1 medium-risk article",
    }]


# ---- self-optimization runs (Phase 9) ----


def test_optimization_runs_round_trip_newest_first(tmp_db: Path) -> None:
    db.insert_optimization_run("2026-01-01T00:00:00", 30, 90, '{"a": 1}')
    db.insert_optimization_run("2026-02-01T00:00:00", 30, 90, '{"b": 2}')
    runs = db.get_optimization_runs()
    assert [r["run_timestamp"] for r in runs] == [
        "2026-02-01T00:00:00", "2026-01-01T00:00:00",
    ]
    latest = db.get_latest_optimization_run()
    assert latest is not None
    assert latest["run_timestamp"] == "2026-02-01T00:00:00"
    assert latest["findings_json"] == '{"b": 2}'


def test_get_latest_optimization_run_none_when_empty(tmp_db: Path) -> None:
    assert db.get_latest_optimization_run() is None


# ---- readiness ledger + resolved counts (Phase 10) ----


def _resolved(track_mode: str, outcome: str, pnl: float | None, idx: int) -> None:
    ts = datetime(2026, 1, 1, 9, 0) + timedelta(minutes=idx)
    sid = db.insert_signal(Signal(
        timestamp=ts, ticker=f"R{idx}", asset_class="stock",
        signal_type="ema21_pullback", direction="call", entry_price=100.0,
    ))
    db.insert_trade(Trade(
        signal_id=sid, opened_at=ts, closed_at=ts + timedelta(days=1),
        outcome=outcome, pnl_pct=pnl, track_mode=track_mode,
    ))


def test_count_resolved_trades_by_track_mode(tmp_db: Path) -> None:
    _resolved("active", "win", 2.0, 0)
    _resolved("active", "loss", -1.0, 1)
    _resolved("active", "expired", None, 2)   # excluded
    _resolved("active", "open", None, 3)      # excluded (open)
    _resolved("shadow", "win", 1.0, 4)
    assert db.count_resolved_trades() == 3              # all win/loss
    assert db.count_resolved_trades(track_mode="active") == 2
    assert db.count_resolved_trades(track_mode="shadow") == 1


def test_readiness_state_round_trip(tmp_db: Path) -> None:
    assert db.get_readiness_state("watchlist_rotation") is None
    db.upsert_readiness_state(
        "watchlist_rotation", status="ready", n_at_crossing=10,
        crossed_at="2026-06-01T00:00:00", announced=True,
    )
    row = db.get_readiness_state("watchlist_rotation")
    assert row is not None
    assert row["status"] == "ready"
    assert row["n_at_crossing"] == 10
    assert row["announced"] is True
    # upsert overwrites
    db.upsert_readiness_state(
        "watchlist_rotation", status="ready", n_at_crossing=10,
        crossed_at="2026-06-01T00:00:00", announced=False,
    )
    assert db.get_readiness_state("watchlist_rotation")["announced"] is False


def test_upsert_readiness_state_rejects_bad_status(tmp_db: Path) -> None:
    with pytest.raises(ValueError, match="status"):
        db.upsert_readiness_state(
            "x", status="bogus", n_at_crossing=None, crossed_at=None, announced=False,
        )
