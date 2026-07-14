"""Tests for trading_bot.outcomes — the trade resolver.

yfinance is mocked throughout; the resolver's correctness depends entirely
on the OHLC decision logic and the same-candle ambiguity rule, both of
which are exercised here.
"""

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from trading_bot import db, outcomes
from trading_bot.models import Signal, Trade

# ────────────────────── helpers ──────────────────────


def _make_signal(
    *,
    direction: str = "call",
    asset_class: str = "stock",
    entry: float = 100.0,
    take_profit: float | None = 110.0,
    stop_loss: float | None = 95.0,
    ticker: str = "GOOGL",
    signal_type: str = "ema21_pullback",
    timestamp: datetime | None = None,
    hold_estimate_days: int | None = None,
) -> int:
    """Insert a Signal and return its id."""
    sig = Signal(
        timestamp=timestamp or datetime(2026, 4, 1, tzinfo=UTC),
        ticker=ticker,
        asset_class=asset_class,
        signal_type=signal_type,
        direction=direction,
        entry_price=entry,
        take_profit=take_profit,
        stop_loss=stop_loss,
        hold_estimate_days=hold_estimate_days,
    )
    return db.insert_signal(sig)


def _make_trade(signal_id: int, opened_at: datetime | None = None) -> int:
    return db.insert_trade(
        Trade(
            signal_id=signal_id,
            opened_at=opened_at or datetime(2026, 4, 1, tzinfo=UTC),
            outcome="open",
        )
    )


def _patch_yf(monkeypatch: pytest.MonkeyPatch, df: pd.DataFrame | None) -> list[str]:
    """Patch yfinance.download to return ``df``. Returns a list that captures
    every ticker requested so tests can assert call counts."""
    calls: list[str] = []

    def fake_download(ticker: str, **_kwargs: object) -> pd.DataFrame | None:
        calls.append(ticker)
        return df

    monkeypatch.setattr("trading_bot.outcomes.yf.download", fake_download)
    return calls


# ────────────────────── resolve_trade — direction × outcome matrix ──────────────────────


def test_resolve_call_hits_take_profit(
    tmp_db: Path,
    fake_candles: Callable[..., pd.DataFrame],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sid = _make_signal(direction="call", entry=100.0, take_profit=110.0, stop_loss=95.0)
    tid = _make_trade(sid)
    # Day 1 quiet, Day 2 spikes high enough to hit TP.
    df = fake_candles(highs=[105, 112], lows=[98, 100])
    _patch_yf(monkeypatch, df)

    trade  = next(t for t in db.get_open_trades() if t.id == tid)
    signal = db.get_signal_by_id(sid)
    assert signal is not None

    resolved = outcomes.resolve_trade(trade, signal, now=datetime(2026, 5, 1, tzinfo=UTC))
    assert resolved.outcome == "win"
    assert resolved.exit_price == 110.0
    assert resolved.pnl_pct == pytest.approx(10.0)


def test_resolve_call_hits_stop_loss(
    tmp_db: Path,
    fake_candles: Callable[..., pd.DataFrame],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sid = _make_signal(direction="call", entry=100.0, take_profit=110.0, stop_loss=95.0)
    tid = _make_trade(sid)
    df = fake_candles(highs=[105, 108], lows=[97, 90])
    _patch_yf(monkeypatch, df)

    trade  = next(t for t in db.get_open_trades() if t.id == tid)
    signal = db.get_signal_by_id(sid)
    assert signal is not None

    resolved = outcomes.resolve_trade(trade, signal, now=datetime(2026, 5, 1, tzinfo=UTC))
    assert resolved.outcome == "loss"
    assert resolved.exit_price == 95.0
    assert resolved.pnl_pct == pytest.approx(-5.0)


def test_resolve_put_hits_take_profit(
    tmp_db: Path,
    fake_candles: Callable[..., pd.DataFrame],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # PUT: profits when price drops; TP is BELOW entry.
    sid = _make_signal(direction="put", entry=100.0, take_profit=90.0, stop_loss=105.0)
    tid = _make_trade(sid)
    df = fake_candles(highs=[102, 103], lows=[98, 89])
    _patch_yf(monkeypatch, df)

    trade  = next(t for t in db.get_open_trades() if t.id == tid)
    signal = db.get_signal_by_id(sid)
    assert signal is not None

    resolved = outcomes.resolve_trade(trade, signal, now=datetime(2026, 5, 1, tzinfo=UTC))
    assert resolved.outcome == "win"
    assert resolved.exit_price == 90.0
    assert resolved.pnl_pct == pytest.approx(10.0)


def test_resolve_put_hits_stop_loss(
    tmp_db: Path,
    fake_candles: Callable[..., pd.DataFrame],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sid = _make_signal(direction="put", entry=100.0, take_profit=90.0, stop_loss=105.0)
    tid = _make_trade(sid)
    df = fake_candles(highs=[103, 108], lows=[97, 95])
    _patch_yf(monkeypatch, df)

    trade  = next(t for t in db.get_open_trades() if t.id == tid)
    signal = db.get_signal_by_id(sid)
    assert signal is not None

    resolved = outcomes.resolve_trade(trade, signal, now=datetime(2026, 5, 1, tzinfo=UTC))
    assert resolved.outcome == "loss"
    assert resolved.exit_price == 105.0
    assert resolved.pnl_pct == pytest.approx(-5.0)


def test_resolve_same_candle_tp_and_sl_marks_as_loss(
    tmp_db: Path,
    fake_candles: Callable[..., pd.DataFrame],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same-candle ambiguity rule: conservatively mark as loss."""
    sid = _make_signal(direction="call", entry=100.0, take_profit=110.0, stop_loss=95.0)
    tid = _make_trade(sid)
    # Single candle that touches both TP and SL — the rule says SL wins.
    df = fake_candles(highs=[111], lows=[94])
    _patch_yf(monkeypatch, df)

    trade  = next(t for t in db.get_open_trades() if t.id == tid)
    signal = db.get_signal_by_id(sid)
    assert signal is not None

    resolved = outcomes.resolve_trade(trade, signal, now=datetime(2026, 5, 1, tzinfo=UTC))
    assert resolved.outcome == "loss"
    assert resolved.exit_price == 95.0


# ────────────────────── hold-window behaviour ──────────────────────


def test_resolve_expired_after_hold_window(
    tmp_db: Path,
    fake_candles: Callable[..., pd.DataFrame],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No TP/SL hit, hold window elapsed → expired with last close as exit."""
    sid = _make_signal(
        direction="call",
        asset_class="stock",
        entry=100.0,
        take_profit=110.0,
        stop_loss=95.0,
        hold_estimate_days=5,
    )
    tid = _make_trade(sid, opened_at=datetime(2026, 4, 1, tzinfo=UTC))
    # 5 candles, all inside the no-touch zone, last close at 102.
    df = fake_candles(
        highs=[105, 104, 103, 106, 105],
        lows=[ 97,  98,  99,  98,  97],
        closes=[100, 99, 100, 101, 102],
    )
    _patch_yf(monkeypatch, df)

    trade  = next(t for t in db.get_open_trades() if t.id == tid)
    signal = db.get_signal_by_id(sid)
    assert signal is not None

    # "now" past the 5-day deadline.
    resolved = outcomes.resolve_trade(
        trade, signal, now=datetime(2026, 5, 1, tzinfo=UTC)
    )
    assert resolved.outcome == "expired"
    assert resolved.exit_price == 102.0
    # PnL for expired: (last_close - entry) / entry × 100 for a call.
    assert resolved.pnl_pct == pytest.approx(2.0)


def test_resolve_still_open_inside_hold_window(
    tmp_db: Path,
    fake_candles: Callable[..., pd.DataFrame],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No TP/SL hit, hold window still active → outcome='open'."""
    sid = _make_signal(direction="call", entry=100.0, take_profit=110.0, stop_loss=95.0,
                       hold_estimate_days=30)
    tid = _make_trade(sid, opened_at=datetime(2026, 4, 1, tzinfo=UTC))
    df = fake_candles(highs=[105, 104], lows=[97, 98])
    _patch_yf(monkeypatch, df)

    trade  = next(t for t in db.get_open_trades() if t.id == tid)
    signal = db.get_signal_by_id(sid)
    assert signal is not None

    # "now" only 3 days in, well inside the 30-day window.
    resolved = outcomes.resolve_trade(
        trade, signal, now=datetime(2026, 4, 4, tzinfo=UTC)
    )
    assert resolved.outcome == "open"
    assert resolved.exit_price is None


# ────────────── forward-only safety proof (Phase 21 critical gate) ──────────


def test_pre_fix_null_estimate_row_resolves_identically_on_default(
    tmp_db: Path,
    fake_candles: Callable[..., pd.DataFrame],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE Phase 21 safety proof: a signal row with NULL hold_estimate_days —
    the shape of EVERY signal fired before the fix, and of every currently-open
    trade's row — resolves exactly as it always has: still open inside the
    30-day stock default, expired at precisely opened_at + 30 days. The fix
    never touches existing rows, so this behavior is unchanged forever."""
    opened = datetime(2026, 4, 1, tzinfo=UTC)
    sid = _make_signal(hold_estimate_days=None)          # pre-Phase-21 row shape
    tid = _make_trade(sid, opened_at=opened)
    # Flat candles that never touch TP (110) or SL (95).
    df = fake_candles(highs=[105, 104, 106], lows=[98, 99, 97], closes=[100, 101, 102])
    _patch_yf(monkeypatch, df)

    trade  = next(t for t in db.get_open_trades() if t.id == tid)
    signal = db.get_signal_by_id(sid)
    assert signal is not None
    assert signal.hold_estimate_days is None

    # Day 29: still open — the unchanged default window is still active.
    still_open = outcomes.resolve_trade(
        trade, signal, now=opened + timedelta(days=29),
    )
    assert still_open.outcome == "open"

    # Past day 30: expired at EXACTLY opened + 30 days — today's behavior.
    resolved = outcomes.resolve_trade(
        trade, signal, now=opened + timedelta(days=45),
    )
    assert resolved.outcome == "expired"
    assert resolved.closed_at == opened + timedelta(days=30)


def test_post_fix_signal_resolves_on_its_own_real_window(
    tmp_db: Path,
    fake_candles: Callable[..., pd.DataFrame],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A signal fired through the REAL fire path (scanner.log_signal) now
    persists its ATR-based estimate and the resolver settles it against THAT
    window — not the 30-day default."""
    from trading_bot import regime, scanner, vix

    def _no_regime(*_a: object, **_k: object) -> object:
        raise regime.RegimeFetchError("offline")

    def _no_vix(*_a: object, **_k: object) -> object:
        raise vix.VixFetchError("offline")

    monkeypatch.setattr("trading_bot.regime.get_current_regime", _no_regime)
    monkeypatch.setattr("trading_bot.vix.get_current_vix", _no_vix)

    sid = scanner.log_signal({
        "ticker": "GOOGL", "asset_type": "stock",
        "trade_type": "📆 SWING TRADE", "direction": "CALL 📈",
        "setup": "EMA21 Pullback", "price": 100.0,
        "take_profit": 110.0, "stop_loss": 95.0,
        "hold_days": "1-2 days",                 # the scanner's ATR estimate
        "confidence": "High",
    })
    assert sid is not None
    signal = db.get_signal_by_id(sid)
    assert signal is not None
    assert signal.hold_estimate_days == 2       # PERSISTED now (outer bound)

    trade = db.get_trade_by_signal_id(sid)
    assert trade is not None
    opened = trade.opened_at
    df = fake_candles(
        highs=[105, 104, 106], lows=[98, 99, 97], closes=[100, 101, 102],
        start=opened,
    )
    _patch_yf(monkeypatch, df)

    # Inside its own 2-day window: still open.
    still_open = outcomes.resolve_trade(
        trade, signal, now=opened + timedelta(days=1),
    )
    assert still_open.outcome == "open"

    # Past ITS window (day 3 of 2) — expired at opened + 2 days, where the
    # old behavior would have held it open for 30.
    resolved = outcomes.resolve_trade(
        trade, signal, now=opened + timedelta(days=3),
    )
    assert resolved.outcome == "expired"
    assert resolved.closed_at == opened + timedelta(days=2)


def test_crypto_signal_still_resolves_on_unchanged_default(
    tmp_db: Path,
    fake_candles: Callable[..., pd.DataFrame],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Phase 21 changes stock behavior only: a crypto signal's hours-shaped
    estimate persists NOTHING, so it keeps resolving on the same 14-day
    crypto default as ever."""
    from trading_bot import regime, scanner, vix

    def _no_regime(*_a: object, **_k: object) -> object:
        raise regime.RegimeFetchError("offline")

    def _no_vix(*_a: object, **_k: object) -> object:
        raise vix.VixFetchError("offline")

    monkeypatch.setattr("trading_bot.regime.get_current_regime", _no_regime)
    monkeypatch.setattr("trading_bot.vix.get_current_vix", _no_vix)

    sid = scanner.log_signal({
        "ticker": "BTC-USD", "asset_type": "crypto",
        "trade_type": "⚡ CRYPTO TRADE", "direction": "LONG 📈",
        "setup": "Oversold Reversal", "price": 50_000.0,
        "take_profit": 51_000.0, "stop_loss": 49_500.0,
        "hold_days": "2-8 hours", "confidence": "High",
    })
    assert sid is not None
    signal = db.get_signal_by_id(sid)
    assert signal is not None
    assert signal.hold_estimate_days is None

    trade = db.get_trade_by_signal_id(sid)
    assert trade is not None
    opened = trade.opened_at
    # Flat candles between SL (49500) and TP (51000) — no touch.
    df = fake_candles(
        highs=[50_500] * 3, lows=[49_800] * 3, closes=[50_000] * 3,
        start=opened, interval="1h",
    )
    _patch_yf(monkeypatch, df)

    resolved = outcomes.resolve_trade(
        trade, signal, now=opened + timedelta(days=15),
    )
    assert resolved.outcome == "expired"
    assert resolved.closed_at == opened + timedelta(days=14)   # crypto default


# ────────────────────── PnL calculation ──────────────────────


def test_pnl_calculation_for_call_win() -> None:
    assert outcomes.calculate_pnl_pct("call", 100.0, 110.0) == pytest.approx(10.0)


def test_pnl_calculation_for_put_win() -> None:
    # PUT win: exit below entry; PnL positive.
    assert outcomes.calculate_pnl_pct("put", 100.0, 90.0) == pytest.approx(10.0)


def test_pnl_calculation_for_loss() -> None:
    # CALL loss: exit below entry; PnL negative.
    assert outcomes.calculate_pnl_pct("call", 100.0, 95.0) == pytest.approx(-5.0)
    # SHORT loss: exit above entry; PnL negative.
    assert outcomes.calculate_pnl_pct("short", 100.0, 105.0) == pytest.approx(-5.0)


def test_pnl_rejects_unknown_direction() -> None:
    with pytest.raises(ValueError, match="Unknown direction"):
        outcomes.calculate_pnl_pct("hodl", 100.0, 105.0)


# ────────────────────── resolve_all_open_trades ──────────────────────


def test_resolve_all_open_trades_skips_already_closed(
    tmp_db: Path,
    fake_candles: Callable[..., pd.DataFrame],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Closed trades stay closed; only open trades get updated."""
    sid = _make_signal(direction="call")
    # Already-closed trade (manually set outcome).
    closed_tid = db.insert_trade(
        Trade(
            signal_id=sid,
            opened_at=datetime(2026, 4, 1, tzinfo=UTC),
            closed_at=datetime(2026, 4, 2, tzinfo=UTC),
            outcome="win",
            exit_price=110.0,
            pnl_pct=10.0,
        )
    )
    df = fake_candles(highs=[200], lows=[50])  # would otherwise hit TP+SL
    _patch_yf(monkeypatch, df)

    result = outcomes.resolve_all_open_trades(now=datetime(2026, 5, 1, tzinfo=UTC))
    # Should report nothing happened — no OPEN trades to resolve.
    assert result == {"wins": 0, "losses": 0, "expired": 0, "still_open": 0}

    # The closed trade is untouched.
    closed = db.get_open_trades()
    assert all(t.id != closed_tid for t in closed)


def test_resolve_all_open_trades_caches_yfinance_calls(
    tmp_db: Path,
    fake_candles: Callable[..., pd.DataFrame],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three open trades on the same ticker → exactly one yfinance call."""
    sid_a = _make_signal(direction="call", ticker="META",
                         timestamp=datetime(2026, 4, 1, tzinfo=UTC))
    sid_b = _make_signal(direction="call", ticker="META", signal_type="trend_acceleration",
                         timestamp=datetime(2026, 4, 1, 0, 1, tzinfo=UTC))
    sid_c = _make_signal(direction="call", ticker="META", signal_type="higher_high_breakout",
                         timestamp=datetime(2026, 4, 1, 0, 2, tzinfo=UTC))
    _make_trade(sid_a)
    _make_trade(sid_b)
    _make_trade(sid_c)

    df = fake_candles(highs=[112], lows=[100])  # all hit TP
    calls = _patch_yf(monkeypatch, df)

    outcomes.resolve_all_open_trades(now=datetime(2026, 5, 1, tzinfo=UTC))
    assert calls.count("META") == 1, f"expected 1 yfinance call for META, got {calls}"


def test_resolve_all_open_trades_handles_yfinance_exception(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A yfinance exception is logged but doesn't crash the resolver loop."""
    sid = _make_signal(direction="call", hold_estimate_days=1)
    _make_trade(sid)

    def boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("yfinance simulated failure")

    monkeypatch.setattr("trading_bot.outcomes.yf.download", boom)
    result = outcomes.resolve_all_open_trades(now=datetime(2026, 5, 1, tzinfo=UTC))
    # Trade marked expired (no data, past deadline) — exact bucket depends on
    # _fetch_candles' fallback path; just verify we didn't crash.
    assert sum(result.values()) == 1


# ────────────────────── backfill ──────────────────────


def test_backfill_creates_trades_for_orphan_signals(
    tmp_db: Path,
    fake_candles: Callable[..., pd.DataFrame],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Three signals with no trades yet.
    _make_signal(timestamp=datetime(2026, 4, 1, tzinfo=UTC), ticker="META")
    _make_signal(timestamp=datetime(2026, 4, 1, 0, 1, tzinfo=UTC), ticker="GOOGL")
    _make_signal(timestamp=datetime(2026, 4, 1, 0, 2, tzinfo=UTC), ticker="AMZN")

    df = fake_candles(highs=[112], lows=[100])  # everyone hits TP
    _patch_yf(monkeypatch, df)

    # Freeze "now" via patch on the helper so the resolver sees past-deadline.
    monkeypatch.setattr(outcomes, "_now_utc", lambda: datetime(2026, 5, 30, tzinfo=UTC))

    created = outcomes.backfill_signals_without_trades()
    assert created == 3
    # Each signal now has exactly one trade.
    for sid in (1, 2, 3):
        assert db.get_trade_by_signal_id(sid) is not None


def test_backfill_is_idempotent(
    tmp_db: Path,
    fake_candles: Callable[..., pd.DataFrame],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _make_signal()
    df = fake_candles(highs=[112], lows=[100])
    _patch_yf(monkeypatch, df)
    monkeypatch.setattr(outcomes, "_now_utc", lambda: datetime(2026, 5, 30, tzinfo=UTC))

    first  = outcomes.backfill_signals_without_trades()
    second = outcomes.backfill_signals_without_trades()
    assert first == 1
    assert second == 0
    # Still exactly one trade for the one signal.
    open_or_closed = db.get_signals_without_trades()
    assert open_or_closed == []


# ────────────────────── summary ──────────────────────


def test_summary_aggregates_by_signal_type(
    tmp_db: Path,
) -> None:
    # Two ema21_pullback (1 win, 1 loss), one oversold_reversal (1 win).
    s1 = _make_signal(signal_type="ema21_pullback",
                      timestamp=datetime(2026, 4, 1, tzinfo=UTC))
    s2 = _make_signal(signal_type="ema21_pullback",
                      timestamp=datetime(2026, 4, 1, 0, 1, tzinfo=UTC))
    s3 = _make_signal(signal_type="oversold_reversal", asset_class="crypto",
                      timestamp=datetime(2026, 4, 1, 0, 2, tzinfo=UTC),
                      ticker="BTC-USD")

    db.insert_trade(Trade(signal_id=s1, opened_at=datetime(2026, 4, 1, tzinfo=UTC),
                          closed_at=datetime(2026, 4, 2, tzinfo=UTC),
                          outcome="win", exit_price=110.0, pnl_pct=10.0))
    db.insert_trade(Trade(signal_id=s2, opened_at=datetime(2026, 4, 1, tzinfo=UTC),
                          closed_at=datetime(2026, 4, 2, tzinfo=UTC),
                          outcome="loss", exit_price=95.0, pnl_pct=-5.0))
    db.insert_trade(Trade(signal_id=s3, opened_at=datetime(2026, 4, 1, tzinfo=UTC),
                          closed_at=datetime(2026, 4, 1, 1, tzinfo=UTC),
                          outcome="win", exit_price=110.0, pnl_pct=20.0))

    report = outcomes.summary()
    by_type = {entry["signal_type"]: entry for entry in report["by_signal_type"]}

    assert by_type["ema21_pullback"]["wins"] == 1
    assert by_type["ema21_pullback"]["losses"] == 1
    assert by_type["ema21_pullback"]["win_rate"] == pytest.approx(50.0)
    assert by_type["ema21_pullback"]["avg_pnl"] == pytest.approx(2.5)  # (10 + -5) / 2

    assert by_type["oversold_reversal"]["wins"] == 1
    assert by_type["oversold_reversal"]["losses"] == 0
    assert by_type["oversold_reversal"]["win_rate"] == pytest.approx(100.0)
    assert by_type["oversold_reversal"]["avg_pnl"] == pytest.approx(20.0)

    assert report["open"] == 0
    assert report["expired"] == 0


def test_summary_handles_empty_database(tmp_db: Path) -> None:
    report = outcomes.summary()
    assert report == {"by_signal_type": [], "open": 0, "expired": 0}


# ────────────────────── edge cases ──────────────────────


def test_resolve_trade_already_closed_returns_unchanged(tmp_db: Path) -> None:
    sid = _make_signal()
    closed = Trade(
        signal_id=sid,
        opened_at=datetime(2026, 4, 1, tzinfo=UTC),
        closed_at=datetime(2026, 4, 2, tzinfo=UTC),
        outcome="win",
        exit_price=110.0,
        pnl_pct=10.0,
        id=42,
    )
    signal = db.get_signal_by_id(sid)
    assert signal is not None
    result = outcomes.resolve_trade(closed, signal)
    assert result is closed  # short-circuited, same object


def test_resolve_trade_without_tp_sl_stays_open(
    tmp_db: Path,
    fake_candles: Callable[..., pd.DataFrame],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sid = _make_signal(take_profit=None, stop_loss=None)
    tid = _make_trade(sid)
    _patch_yf(monkeypatch, fake_candles(highs=[200], lows=[50]))

    trade  = next(t for t in db.get_open_trades() if t.id == tid)
    signal = db.get_signal_by_id(sid)
    assert signal is not None

    resolved = outcomes.resolve_trade(trade, signal, now=datetime(2026, 5, 1, tzinfo=UTC))
    assert resolved.outcome == "open"
