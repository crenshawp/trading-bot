"""Tests for trading_bot.performance — pure SQL aggregations + daily perf.

These tests seed synthetic signal+trade pairs directly via ``db.insert_*``
and never hit yfinance. Aggregation rules under test:

* Closed = win/loss/expired; open and NULL excluded from every stat.
* Expired counts toward total + avg/best/worst PnL but NOT win-rate math.
* Empty slices return numeric ``None`` (not 0).
"""

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from trading_bot import db, performance
from trading_bot.models import Signal, Trade

# ────────────────────── seeding helper ──────────────────────


def _seed_trade(
    *,
    signal_type: str = "ema21_pullback",
    ticker: str = "GOOGL",
    asset_class: str = "stock",
    direction: str = "call",
    outcome: str | None = "win",
    pnl_pct: float | None = 5.0,
    timestamp: datetime | None = None,
    closed_at: datetime | None = None,
    entry_price: float = 100.0,
    exit_price: float | None = 105.0,
) -> tuple[int, int]:
    """Insert signal + trade for the test scenario. Returns (signal_id, trade_id).

    Distinct timestamps are ensured via microsecond offsets so the UNIQUE
    INDEX on (timestamp, ticker, signal_type) doesn't dedupe scenarios.
    """
    ts = timestamp if timestamp is not None else datetime(2026, 4, 1, tzinfo=UTC)
    sig = Signal(
        timestamp=ts,
        ticker=ticker,
        asset_class=asset_class,
        signal_type=signal_type,
        direction=direction,
        entry_price=entry_price,
        take_profit=110.0,
        stop_loss=95.0,
    )
    sid = db.insert_signal(sig)
    closed_at_val = closed_at if closed_at is not None else (
        ts + timedelta(days=2) if outcome in ("win", "loss", "expired") else None
    )
    trade = Trade(
        signal_id=sid,
        opened_at=ts,
        closed_at=closed_at_val,
        outcome=outcome,
        exit_price=exit_price if outcome in ("win", "loss", "expired") else None,
        pnl_pct=pnl_pct if outcome in ("win", "loss", "expired") else None,
    )
    tid = db.insert_trade(trade)
    return sid, tid


def _bulk_seed(scenarios: list[dict[str, Any]]) -> None:
    """Seed many trades. Each scenario is a kwargs dict for ``_seed_trade``.

    Auto-offsets ``timestamp`` by 1 second per scenario to avoid dedupe
    collisions when callers omit it.
    """
    base = datetime(2026, 4, 1, tzinfo=UTC)
    for i, scenario in enumerate(scenarios):
        scenario.setdefault("timestamp", base + timedelta(seconds=i))
        _seed_trade(**scenario)


# ────────────────────── aggregation correctness ──────────────────────


def test_stats_by_signal_type_groups_correctly(tmp_db: Path) -> None:
    _bulk_seed([
        {"signal_type": "ema21_pullback", "outcome": "win",  "pnl_pct": 5.0},
        {"signal_type": "ema21_pullback", "outcome": "win",  "pnl_pct": 3.0},
        {"signal_type": "ema21_pullback", "outcome": "loss", "pnl_pct": -2.0},
        {"signal_type": "oversold_reversal", "ticker": "BTC-USD",
         "asset_class": "crypto", "direction": "long",
         "outcome": "win",  "pnl_pct": 7.0},
        {"signal_type": "oversold_reversal", "ticker": "BTC-USD",
         "asset_class": "crypto", "direction": "long",
         "outcome": "loss", "pnl_pct": -3.0},
    ])
    stats = performance.stats_by_signal_type()
    by_label = {s.label: s for s in stats}
    assert by_label["ema21_pullback"].total == 3
    assert by_label["ema21_pullback"].wins == 2
    assert by_label["ema21_pullback"].losses == 1
    assert by_label["ema21_pullback"].win_rate == pytest.approx(66.666, rel=1e-3)
    assert by_label["ema21_pullback"].avg_pnl_pct == pytest.approx(2.0)
    assert by_label["oversold_reversal"].total == 2
    # Sorted descending by win_rate
    assert stats[0].win_rate is not None and stats[1].win_rate is not None
    assert stats[0].win_rate >= stats[1].win_rate


def test_stats_by_ticker_filters_by_asset_class(tmp_db: Path) -> None:
    _bulk_seed([
        {"ticker": "GOOGL",   "asset_class": "stock",  "outcome": "win", "pnl_pct": 5.0},
        {"ticker": "META",    "asset_class": "stock",  "outcome": "loss", "pnl_pct": -3.0},
        {"ticker": "BTC-USD", "asset_class": "crypto", "direction": "long",
         "signal_type": "oversold_reversal", "outcome": "win", "pnl_pct": 8.0},
    ])
    stocks_only = performance.stats_by_ticker(asset_class="stock")
    tickers = {s.label for s in stocks_only}
    assert tickers == {"GOOGL", "META"}
    assert "BTC-USD" not in tickers

    crypto_only = performance.stats_by_ticker(asset_class="crypto")
    assert {s.label for s in crypto_only} == {"BTC-USD"}

    all_tickers = performance.stats_by_ticker()
    assert {s.label for s in all_tickers} == {"GOOGL", "META", "BTC-USD"}


def test_stats_by_ticker_rejects_invalid_asset_class(tmp_db: Path) -> None:
    with pytest.raises(ValueError, match="asset_class"):
        performance.stats_by_ticker(asset_class="forex")


def test_stats_by_asset_class_splits_stock_crypto(tmp_db: Path) -> None:
    _bulk_seed([
        {"asset_class": "stock",  "outcome": "win",  "pnl_pct": 5.0},
        {"asset_class": "stock",  "outcome": "loss", "pnl_pct": -2.0},
        {"ticker": "BTC-USD", "asset_class": "crypto", "direction": "long",
         "signal_type": "oversold_reversal", "outcome": "win", "pnl_pct": 8.0},
    ])
    stats = performance.stats_by_asset_class()
    by_label = {s.label: s for s in stats}
    assert by_label["stock"].total == 2
    assert by_label["crypto"].total == 1


def test_stats_overall_aggregates_everything(tmp_db: Path) -> None:
    _bulk_seed([
        {"outcome": "win",     "pnl_pct": 5.0},
        {"outcome": "win",     "pnl_pct": 3.0},
        {"outcome": "loss",    "pnl_pct": -2.0},
        {"outcome": "expired", "pnl_pct": 1.0},
    ])
    stats = performance.stats_overall()
    assert stats.total == 4
    assert stats.wins == 2
    assert stats.losses == 1
    assert stats.expired == 1
    # Win rate excludes expired: 2 / 3
    assert stats.win_rate == pytest.approx(66.666, rel=1e-3)
    # Avg PnL includes expired: (5 + 3 + -2 + 1) / 4 = 1.75
    assert stats.avg_pnl_pct == pytest.approx(1.75)
    assert stats.best_pnl_pct == pytest.approx(5.0)
    assert stats.worst_pnl_pct == pytest.approx(-2.0)


def test_stats_recent_filters_by_date_window(tmp_db: Path) -> None:
    now = datetime.now(UTC)
    # One inside the 30-day window (5 days ago), one outside (40 days ago).
    _seed_trade(
        timestamp=now - timedelta(days=5),
        closed_at=now - timedelta(days=4),
        outcome="win", pnl_pct=5.0,
    )
    _seed_trade(
        timestamp=now - timedelta(days=40),
        closed_at=now - timedelta(days=39),
        outcome="loss", pnl_pct=-3.0,
        signal_type="oversold_reversal", ticker="BTC-USD",
        asset_class="crypto", direction="long",
    )
    recent = performance.stats_recent(days=30)
    assert recent.total == 1
    assert recent.wins == 1


def test_stats_cross_excludes_slices_under_3_trades(tmp_db: Path) -> None:
    # ema21_pullback / META → 3 trades (qualifies)
    # ema21_pullback / AMZN → 2 trades (excluded)
    _bulk_seed([
        {"signal_type": "ema21_pullback", "ticker": "META", "outcome": "win",  "pnl_pct": 5.0},
        {"signal_type": "ema21_pullback", "ticker": "META", "outcome": "win",  "pnl_pct": 3.0},
        {"signal_type": "ema21_pullback", "ticker": "META", "outcome": "loss", "pnl_pct": -2.0},
        {"signal_type": "ema21_pullback", "ticker": "AMZN", "outcome": "win",  "pnl_pct": 4.0},
        {"signal_type": "ema21_pullback", "ticker": "AMZN", "outcome": "win",  "pnl_pct": 2.0},
    ])
    stats = performance.stats_by_signal_type_and_ticker()
    labels = {s.label for s in stats}
    assert "ema21_pullback / META" in labels
    assert "ema21_pullback / AMZN" not in labels


# ────────────────────── aggregation rules ──────────────────────


def test_expired_trades_excluded_from_win_rate(tmp_db: Path) -> None:
    """All-expired slice has win_rate = None (not 0)."""
    _bulk_seed([
        {"outcome": "expired", "pnl_pct": 1.0},
        {"outcome": "expired", "pnl_pct": 2.0},
        {"outcome": "expired", "pnl_pct": -1.0},
    ])
    stats = performance.stats_overall()
    assert stats.total == 3
    assert stats.expired == 3
    assert stats.wins == 0
    assert stats.losses == 0
    assert stats.win_rate is None  # not 0.0
    assert stats.avg_pnl_pct == pytest.approx((1.0 + 2.0 - 1.0) / 3.0)


def test_expired_trades_included_in_avg_pnl(tmp_db: Path) -> None:
    _bulk_seed([
        {"outcome": "win",     "pnl_pct": 10.0},
        {"outcome": "expired", "pnl_pct": -2.0},
    ])
    stats = performance.stats_overall()
    # Avg of 10 and -2 = 4
    assert stats.avg_pnl_pct == pytest.approx(4.0)
    # But win_rate is wins / (wins + losses) = 1 / 1 = 100%
    assert stats.win_rate == pytest.approx(100.0)


def test_open_trades_excluded_from_all_stats(tmp_db: Path) -> None:
    _bulk_seed([
        {"outcome": "win", "pnl_pct": 5.0},
    ])
    # An "open" trade — should not affect any stat
    _seed_trade(
        ticker="META",
        outcome="open",
        pnl_pct=None,
        exit_price=None,
        timestamp=datetime(2026, 5, 1, tzinfo=UTC),
    )
    stats = performance.stats_overall()
    assert stats.total == 1  # only the closed win counts
    assert stats.wins == 1


def test_null_outcome_trades_excluded_from_all_stats(tmp_db: Path) -> None:
    _bulk_seed([
        {"outcome": "win", "pnl_pct": 5.0},
    ])
    # outcome=None — not yet resolved
    _seed_trade(
        ticker="AMZN",
        outcome=None,
        pnl_pct=None,
        exit_price=None,
        timestamp=datetime(2026, 5, 1, tzinfo=UTC),
    )
    stats = performance.stats_overall()
    assert stats.total == 1


def test_empty_slice_returns_none_not_zero(tmp_db: Path) -> None:
    stats = performance.stats_overall()
    assert stats.total == 0
    assert stats.wins == 0
    assert stats.losses == 0
    assert stats.expired == 0
    assert stats.win_rate is None
    assert stats.avg_pnl_pct is None
    assert stats.best_pnl_pct is None
    assert stats.worst_pnl_pct is None


def test_best_and_worst_pnl_correct_with_mixed_outcomes(tmp_db: Path) -> None:
    _bulk_seed([
        {"outcome": "win",     "pnl_pct":  8.0},
        {"outcome": "win",     "pnl_pct":  2.0},
        {"outcome": "loss",    "pnl_pct": -5.0},
        {"outcome": "expired", "pnl_pct":  1.5},
        {"outcome": "expired", "pnl_pct": -3.0},
    ])
    stats = performance.stats_overall()
    assert stats.best_pnl_pct == pytest.approx(8.0)
    assert stats.worst_pnl_pct == pytest.approx(-5.0)


# ────────────────────── daily_performance ──────────────────────


def test_update_daily_performance_creates_row(tmp_db: Path) -> None:
    day = datetime(2026, 4, 1, 14, tzinfo=UTC)
    _seed_trade(
        timestamp=day,
        closed_at=day + timedelta(hours=4),
        outcome="win", pnl_pct=5.0,
    )
    _seed_trade(
        ticker="META", timestamp=day + timedelta(hours=1),
        closed_at=day + timedelta(hours=5),
        outcome="loss", pnl_pct=-2.0,
    )
    result = performance.update_daily_performance(date(2026, 4, 1))
    assert result.date == "2026-04-01"
    assert result.signals_fired == 2
    assert result.trades_opened == 2
    assert result.trades_closed == 2
    assert result.wins == 1
    assert result.losses == 1
    assert result.win_rate == pytest.approx(50.0)
    assert result.total_pnl_pct == pytest.approx(3.0)


def test_update_daily_performance_upserts(tmp_db: Path) -> None:
    day = datetime(2026, 4, 1, 14, tzinfo=UTC)
    _seed_trade(timestamp=day, closed_at=day + timedelta(hours=4),
                outcome="win", pnl_pct=5.0)
    first = performance.update_daily_performance(date(2026, 4, 1))
    # Seed a second signal/trade on the same day
    _seed_trade(ticker="META", timestamp=day + timedelta(hours=1),
                closed_at=day + timedelta(hours=5),
                outcome="loss", pnl_pct=-1.0)
    second = performance.update_daily_performance(date(2026, 4, 1))
    assert first.signals_fired == 1
    assert second.signals_fired == 2
    # Only one row in the table
    conn = db.get_connection()
    try:
        count = conn.execute(
            "SELECT COUNT(*) c FROM daily_performance WHERE date = ?",
            ("2026-04-01",),
        ).fetchone()["c"]
    finally:
        conn.close()
    assert count == 1


def test_update_daily_performance_defaults_to_yesterday(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Freeze "today" by patching datetime.now usage isn't trivial; instead,
    # verify the default branch picks yesterday by checking the date string.
    yesterday = (datetime.now(UTC).date() - timedelta(days=1)).isoformat()
    perf = performance.update_daily_performance()
    assert perf.date == yesterday


def test_update_daily_performance_with_no_activity_uses_zeros(tmp_db: Path) -> None:
    perf = performance.update_daily_performance(date(2025, 1, 1))
    assert perf.signals_fired == 0
    assert perf.trades_opened == 0
    assert perf.trades_closed == 0
    assert perf.wins == 0
    assert perf.losses == 0
    assert perf.win_rate is None
    assert perf.total_pnl_pct is None


def test_backfill_daily_performance_covers_all_dates(tmp_db: Path) -> None:
    _seed_trade(timestamp=datetime(2026, 4, 1, 10, tzinfo=UTC),
                closed_at=datetime(2026, 4, 3, 10, tzinfo=UTC),
                outcome="win", pnl_pct=5.0)
    _seed_trade(ticker="META",
                timestamp=datetime(2026, 4, 5, 10, tzinfo=UTC),
                closed_at=datetime(2026, 4, 5, 14, tzinfo=UTC),
                outcome="loss", pnl_pct=-3.0)
    count = performance.backfill_daily_performance()
    # Distinct dates: signal timestamps 2026-04-01, 2026-04-05;
    #                 closed_at 2026-04-03, 2026-04-05; opened_at same as ts
    # → {04-01, 04-03, 04-05} = 3 dates
    assert count == 3


def test_backfill_daily_performance_is_idempotent(tmp_db: Path) -> None:
    _seed_trade(timestamp=datetime(2026, 4, 1, 10, tzinfo=UTC),
                closed_at=datetime(2026, 4, 1, 14, tzinfo=UTC),
                outcome="win", pnl_pct=5.0)
    first  = performance.backfill_daily_performance()
    second = performance.backfill_daily_performance()
    assert first == second
    conn = db.get_connection()
    try:
        rows = conn.execute("SELECT COUNT(*) c FROM daily_performance").fetchone()["c"]
    finally:
        conn.close()
    assert rows == first


# ────────────────────── empty-database edges ──────────────────────


def test_stats_overall_empty_database(tmp_db: Path) -> None:
    stats = performance.stats_overall()
    assert stats.label == "overall"
    assert stats.total == 0
    assert stats.win_rate is None


def test_stats_by_signal_type_empty_database(tmp_db: Path) -> None:
    assert performance.stats_by_signal_type() == []


def test_stats_by_ticker_empty_database(tmp_db: Path) -> None:
    assert performance.stats_by_ticker() == []


def test_stats_by_asset_class_empty_database(tmp_db: Path) -> None:
    assert performance.stats_by_asset_class() == []


def test_stats_cross_empty_database(tmp_db: Path) -> None:
    assert performance.stats_by_signal_type_and_ticker() == []


# ────────────────────── regime-aware slices (Phase 2.1) ──────────────────────


def _seed_with_regime(
    *, regime: str, ticker: str = "GOOGL", signal_type: str = "ema21_pullback",
    asset_class: str = "stock", direction: str = "call",
    outcome: str = "win", pnl_pct: float = 5.0,
    timestamp: datetime | None = None,
) -> int:
    """Insert a signal + trade with the given market_regime tag."""
    ts = timestamp if timestamp is not None else datetime(2026, 4, 1, tzinfo=UTC)
    sid = db.insert_signal(Signal(
        timestamp=ts, ticker=ticker, asset_class=asset_class,
        signal_type=signal_type, direction=direction,
        entry_price=100.0, take_profit=110.0, stop_loss=95.0,
    ))
    return db.insert_trade(Trade(
        signal_id=sid, opened_at=ts,
        closed_at=ts + timedelta(days=2),
        outcome=outcome, exit_price=105.0, pnl_pct=pnl_pct,
        market_regime=regime,
    ))


def test_stats_by_regime_orders_bull_sideways_bear_unknown(tmp_db: Path) -> None:
    base = datetime(2026, 4, 1, tzinfo=UTC)
    _seed_with_regime(regime="bear",     timestamp=base + timedelta(seconds=1), outcome="loss", pnl_pct=-3.0)
    _seed_with_regime(regime="unknown",  timestamp=base + timedelta(seconds=2))
    _seed_with_regime(regime="bull",     timestamp=base + timedelta(seconds=3))
    _seed_with_regime(regime="sideways", timestamp=base + timedelta(seconds=4))
    result = performance.stats_by_regime()
    assert [s.label for s in result] == ["bull", "sideways", "bear", "unknown"]


def test_stats_by_regime_folds_null_into_unknown(tmp_db: Path) -> None:
    # No market_regime tag — falls into 'unknown' bucket
    ts = datetime(2026, 4, 1, tzinfo=UTC)
    sid = db.insert_signal(Signal(
        timestamp=ts, ticker="GOOGL", asset_class="stock",
        signal_type="ema21_pullback", direction="call",
        entry_price=100.0, take_profit=110.0, stop_loss=95.0,
    ))
    db.insert_trade(Trade(
        signal_id=sid, opened_at=ts, closed_at=ts + timedelta(days=2),
        outcome="win", exit_price=105.0, pnl_pct=5.0,
        market_regime=None,
    ))
    result = performance.stats_by_regime()
    assert len(result) == 1
    assert result[0].label == "unknown"
    assert result[0].wins == 1


def test_stats_by_signal_type_with_regime_builds_matrix(tmp_db: Path) -> None:
    base = datetime(2026, 4, 1, tzinfo=UTC)
    _seed_with_regime(regime="bull", signal_type="ema21_pullback",
                      timestamp=base, outcome="win", pnl_pct=5.0)
    _seed_with_regime(regime="bull", signal_type="ema21_pullback",
                      timestamp=base + timedelta(seconds=1),
                      outcome="win", pnl_pct=3.0)
    _seed_with_regime(regime="sideways", signal_type="ema21_pullback",
                      timestamp=base + timedelta(seconds=2),
                      outcome="loss", pnl_pct=-2.0)
    _seed_with_regime(regime="bull", signal_type="oversold_reversal",
                      ticker="BTC-USD", asset_class="crypto", direction="long",
                      timestamp=base + timedelta(seconds=3),
                      outcome="win", pnl_pct=7.0)

    rows = performance.stats_by_signal_type_with_regime()
    by_label = {r.label: r for r in rows}

    ep = by_label["ema21_pullback"]
    assert ep.overall.total == 3
    assert ep.by_regime["bull"].wins == 2
    assert ep.by_regime["sideways"].losses == 1
    assert "bear" not in ep.by_regime  # no bear trades for this signal

    osr = by_label["oversold_reversal"]
    assert osr.overall.total == 1
    assert osr.by_regime["bull"].wins == 1


def test_stats_by_ticker_with_regime_filters_by_asset_class(tmp_db: Path) -> None:
    base = datetime(2026, 4, 1, tzinfo=UTC)
    _seed_with_regime(regime="bull", ticker="GOOGL",
                      timestamp=base, outcome="win", pnl_pct=5.0)
    _seed_with_regime(regime="bull", ticker="BTC-USD", asset_class="crypto",
                      direction="long", signal_type="oversold_reversal",
                      timestamp=base + timedelta(seconds=1),
                      outcome="win", pnl_pct=7.0)

    stocks = performance.stats_by_ticker_with_regime(asset_class="stock")
    tickers = {r.label for r in stocks}
    assert tickers == {"GOOGL"}

    crypto = performance.stats_by_ticker_with_regime(asset_class="crypto")
    assert {r.label for r in crypto} == {"BTC-USD"}


def test_stats_by_ticker_with_regime_rejects_bad_asset_class(tmp_db: Path) -> None:
    with pytest.raises(ValueError, match="asset_class"):
        performance.stats_by_ticker_with_regime(asset_class="forex")


def test_stats_by_regime_empty_database(tmp_db: Path) -> None:
    assert performance.stats_by_regime() == []


def test_stats_by_signal_type_with_regime_empty(tmp_db: Path) -> None:
    assert performance.stats_by_signal_type_with_regime() == []


# ────────────────────── VIX-aware slices (Phase 2.2) ──────────────────────


def _seed_with_axes(
    *,
    market_regime: str | None = None,
    vix_band: str | None = None,
    vix_level: float | None = None,
    ticker: str = "GOOGL",
    signal_type: str = "ema21_pullback",
    asset_class: str = "stock",
    direction: str = "call",
    outcome: str = "win",
    pnl_pct: float = 5.0,
    timestamp: datetime | None = None,
) -> int:
    """Seed a signal+trade with both regime and VIX tags."""
    ts = timestamp if timestamp is not None else datetime(2026, 4, 1, tzinfo=UTC)
    sid = db.insert_signal(Signal(
        timestamp=ts, ticker=ticker, asset_class=asset_class,
        signal_type=signal_type, direction=direction,
        entry_price=100.0, take_profit=110.0, stop_loss=95.0,
    ))
    return db.insert_trade(Trade(
        signal_id=sid, opened_at=ts,
        closed_at=ts + timedelta(days=2),
        outcome=outcome, exit_price=105.0, pnl_pct=pnl_pct,
        market_regime=market_regime,
        vix_level=vix_level, vix_band=vix_band,
    ))


def test_stats_by_vix_band_orders_low_elevated_high_extreme_unknown(
    tmp_db: Path,
) -> None:
    base = datetime(2026, 4, 1, tzinfo=UTC)
    _seed_with_axes(vix_band="extreme", timestamp=base + timedelta(seconds=1),
                    outcome="loss", pnl_pct=-3.0)
    _seed_with_axes(vix_band="unknown", timestamp=base + timedelta(seconds=2))
    _seed_with_axes(vix_band="low",     timestamp=base + timedelta(seconds=3))
    _seed_with_axes(vix_band="high",    timestamp=base + timedelta(seconds=4))
    _seed_with_axes(vix_band="elevated", timestamp=base + timedelta(seconds=5))
    result = performance.stats_by_vix_band()
    assert [s.label for s in result] == [
        "low", "elevated", "high", "extreme", "unknown",
    ]


def test_stats_by_vix_band_folds_null_into_unknown(tmp_db: Path) -> None:
    _seed_with_axes(vix_band=None)
    result = performance.stats_by_vix_band()
    assert len(result) == 1
    assert result[0].label == "unknown"


def test_stats_by_signal_type_with_vix_builds_matrix(tmp_db: Path) -> None:
    base = datetime(2026, 4, 1, tzinfo=UTC)
    _seed_with_axes(vix_band="low", signal_type="ema21_pullback",
                    timestamp=base, outcome="win", pnl_pct=5.0)
    _seed_with_axes(vix_band="low", signal_type="ema21_pullback",
                    timestamp=base + timedelta(seconds=1),
                    outcome="win", pnl_pct=3.0)
    _seed_with_axes(vix_band="elevated", signal_type="ema21_pullback",
                    timestamp=base + timedelta(seconds=2),
                    outcome="loss", pnl_pct=-2.0)
    _seed_with_axes(vix_band="low", signal_type="oversold_reversal",
                    ticker="BTC-USD", asset_class="crypto", direction="long",
                    timestamp=base + timedelta(seconds=3),
                    outcome="win", pnl_pct=7.0)

    rows = performance.stats_by_signal_type_with_vix()
    by_label = {r.label: r for r in rows}

    ep = by_label["ema21_pullback"]
    assert ep.overall.total == 3
    assert ep.by_vix["low"].wins == 2
    assert ep.by_vix["elevated"].losses == 1
    assert "high" not in ep.by_vix


def test_stats_by_ticker_with_vix_filters_by_asset_class(tmp_db: Path) -> None:
    base = datetime(2026, 4, 1, tzinfo=UTC)
    _seed_with_axes(vix_band="low", ticker="GOOGL", timestamp=base)
    _seed_with_axes(vix_band="elevated", ticker="BTC-USD",
                    asset_class="crypto", direction="long",
                    signal_type="oversold_reversal",
                    timestamp=base + timedelta(seconds=1),
                    outcome="win", pnl_pct=7.0)

    stocks = performance.stats_by_ticker_with_vix(asset_class="stock")
    assert {r.label for r in stocks} == {"GOOGL"}
    crypto = performance.stats_by_ticker_with_vix(asset_class="crypto")
    assert {r.label for r in crypto} == {"BTC-USD"}


def test_stats_by_ticker_with_vix_rejects_bad_asset_class(tmp_db: Path) -> None:
    with pytest.raises(ValueError, match="asset_class"):
        performance.stats_by_ticker_with_vix(asset_class="forex")


def test_stats_by_regime_x_vix_sorts_by_trade_count_desc(tmp_db: Path) -> None:
    base = datetime(2026, 4, 1, tzinfo=UTC)
    # bull/low: 3 trades, bull/elevated: 2, sideways/low: 1
    for i in range(3):
        _seed_with_axes(
            market_regime="bull", vix_band="low",
            timestamp=base + timedelta(seconds=i),
        )
    for i in range(2):
        _seed_with_axes(
            market_regime="bull", vix_band="elevated",
            timestamp=base + timedelta(seconds=10 + i),
        )
    _seed_with_axes(
        market_regime="sideways", vix_band="low",
        timestamp=base + timedelta(seconds=20),
    )

    rows = performance.stats_by_regime_x_vix()
    assert [r.label for r in rows] == [
        "bull / low", "bull / elevated", "sideways / low",
    ]
    assert [r.total for r in rows] == [3, 2, 1]


def test_stats_by_regime_x_vix_omits_empty_buckets(tmp_db: Path) -> None:
    """Only non-empty (regime, vix) buckets show up."""
    _seed_with_axes(market_regime="bull", vix_band="low")
    rows = performance.stats_by_regime_x_vix()
    assert len(rows) == 1
    assert rows[0].label == "bull / low"


def test_stats_by_vix_band_empty(tmp_db: Path) -> None:
    assert performance.stats_by_vix_band() == []


def test_stats_by_regime_x_vix_empty(tmp_db: Path) -> None:
    assert performance.stats_by_regime_x_vix() == []


def test_stats_by_signal_type_with_vix_empty(tmp_db: Path) -> None:
    assert performance.stats_by_signal_type_with_vix() == []
