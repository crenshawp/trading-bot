"""Tests for the long-term buy-and-hold system (Phase 14, stocks + crypto).

Mocked data only, no live network. Fundamental screen (fail-open / red-flag),
long-horizon technical confirmation, earnings wait-window, candidate generation
(crypto SWING never executes), diversification sizing, and the protective exit
watcher are the core.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from trading_bot import config, db, indicators, long_term, risk
from trading_bot.broker.fake import FakeBroker
from trading_bot.earnings import EarningsInfo
from trading_bot.long_term import LongTermCandidate, LongTermPosition

# ───────────────────────── config sanity ────────────────────────────────────


def test_longterm_config_defaults() -> None:
    assert config.EARNINGS_WAIT_DAYS == 5
    assert config.MAX_POSITION_WEIGHT_PCT == 15.0
    assert config.TREND_PERIOD == 200
    assert config.TREND_BREAKDOWN_DAYS == 3
    assert config.MAX_DRAWDOWN_STOP_PCT == 25.0


def test_crypto_swing_signals_are_data_only_and_unrouted() -> None:
    # The crypto SWING signals must be data-only AND absent from the routing map.
    # Phase 23 sealed overbought_reversal (the detector's third setup) — it was
    # missing, letting route_pool's crypto fallback send a SHORT to execution.
    assert "oversold_reversal" in config.DATA_ONLY_SIGNAL_TYPES
    assert "momentum_breakout" in config.DATA_ONLY_SIGNAL_TYPES
    assert "overbought_reversal" in config.DATA_ONLY_SIGNAL_TYPES
    assert "oversold_reversal" not in config.SIGNAL_TYPE_TO_POOL
    assert "momentum_breakout" not in config.SIGNAL_TYPE_TO_POOL
    assert "overbought_reversal" not in config.SIGNAL_TYPE_TO_POOL
    # The long-term signals ARE routed, to their pools.
    assert config.SIGNAL_TYPE_TO_POOL[config.LONGTERM_STOCK_SIGNAL] == config.POOL_LONG_TERM
    assert config.SIGNAL_TYPE_TO_POOL[config.LONGTERM_CRYPTO_SIGNAL] == config.POOL_CRYPTO


# ───────────────────────── neutral types ────────────────────────────────────


def test_long_term_candidate_defaults() -> None:
    c = LongTermCandidate(
        ticker="AAPL", asset_class="stock",
        signal_type=config.LONGTERM_STOCK_SIGNAL, entry_price=190.0,
    )
    assert c.entry_rationale == ""
    assert c.signal_type == "long_term_stock"


def test_long_term_position_defaults() -> None:
    p = LongTermPosition(
        ticker="AAPL", asset_class="stock", entry_price=190.0,
        entry_date=datetime(2026, 1, 1, tzinfo=UTC), qty=10.0,
    )
    assert p.status == "open"
    assert p.exit_reason is None and p.exit_price is None


def test_module_exposes_types() -> None:
    assert long_term.LongTermCandidate is LongTermCandidate
    assert long_term.LongTermPosition is LongTermPosition


# ───────────────────────── fundamental screen (fail-open) ───────────────────


def test_screen_fundamentals_fail_open_on_missing_data() -> None:
    # Missing EITHER datum -> fail open (not blocked).
    assert long_term.screen_fundamentals(None, 300.0)[0] is False
    assert long_term.screen_fundamentals(-0.5, None)[0] is False
    assert long_term.screen_fundamentals(None, None)[0] is False


def test_screen_fundamentals_blocks_only_genuine_red_flag() -> None:
    # BOTH bad (earnings growth -30% AND debt/equity 250 == 2.5x) -> blocked.
    blocked, reason = long_term.screen_fundamentals(-0.30, 250.0)
    assert blocked is True and "red flag" in reason


def test_screen_fundamentals_passes_clean_and_single_bad() -> None:
    # Clean fundamentals -> not blocked.
    assert long_term.screen_fundamentals(0.15, 80.0)[0] is False
    # Only ONE bad datum -> not blocked (loose filter needs both).
    assert long_term.screen_fundamentals(-0.30, 80.0)[0] is False    # only earnings bad
    assert long_term.screen_fundamentals(0.15, 250.0)[0] is False    # only debt bad


def test_fundamental_red_flag_uses_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(long_term, "_fetch_fundamentals", lambda _t: (-0.4, 300.0))
    assert long_term.fundamental_red_flag("BADCO")[0] is True
    monkeypatch.setattr(long_term, "_fetch_fundamentals", lambda _t: (None, None))
    assert long_term.fundamental_red_flag("UNKNOWN")[0] is False   # fail-open


def test_fetch_fundamentals_failsoft(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(_t: str) -> object:
        raise RuntimeError("yf down")

    monkeypatch.setattr("trading_bot.long_term.yf.Ticker", boom)
    assert long_term._fetch_fundamentals("AAPL") == (None, None)


# ───────────────────────── technical confirmation (pure) ────────────────────


def test_sma_trend_primitive() -> None:
    got = indicators.sma(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]), 3)
    assert got.iloc[-1] == 4.0                      # (3 + 4 + 5) / 3


def test_confirm_conditions_all_pass() -> None:
    ok, reason = long_term._confirm_conditions(
        110.0, 100.0, 25.0, 55.0, adx_min=20.0, rsi_max=70.0,
    )
    assert ok is True and "price" in reason


def test_confirm_conditions_each_single_failure() -> None:
    # price below trend
    ok, r = long_term._confirm_conditions(90.0, 100.0, 25.0, 55.0, adx_min=20.0, rsi_max=70.0)
    assert ok is False and "trend" in r
    # ADX too weak
    ok, r = long_term._confirm_conditions(110.0, 100.0, 10.0, 55.0, adx_min=20.0, rsi_max=70.0)
    assert ok is False and "ADX" in r
    # RSI overbought
    ok, r = long_term._confirm_conditions(110.0, 100.0, 25.0, 80.0, adx_min=20.0, rsi_max=70.0)
    assert ok is False and "RSI" in r


def test_confirm_conditions_insufficient_data() -> None:
    assert long_term._confirm_conditions(
        None, 100.0, 25.0, 55.0, adx_min=20.0, rsi_max=70.0,
    ) == (False, "insufficient data")


def test_confirm_technical_entry_wires_indicators(
    fake_candles: Callable[..., pd.DataFrame],
) -> None:
    # A pure uptrend: price is above the short trend and ADX is strong, but RSI
    # pins near 100 (all gains) -> the RSI ceiling fails. Deterministic wiring.
    closes = [100.0 + i for i in range(30)]
    df = fake_candles([c + 1 for c in closes], [c - 1 for c in closes], closes)
    ok, reason = long_term.confirm_technical_entry(df, trend_period=5)
    assert ok is False and "RSI" in reason


def test_confirm_technical_entry_insufficient_data(
    fake_candles: Callable[..., pd.DataFrame],
) -> None:
    df = fake_candles([101.0, 102.0], [99.0, 100.0], [100.0, 101.0])
    ok, reason = long_term.confirm_technical_entry(df, trend_period=200)
    assert ok is False and reason == "insufficient data"


# ───────────────────────── earnings wait-window (soft) ──────────────────────

_NOW = datetime(2026, 1, 10, tzinfo=UTC)


def _patch_earnings(
    monkeypatch: pytest.MonkeyPatch, when: datetime | None,
) -> None:
    monkeypatch.setattr(
        long_term.earnings, "next_earnings_date",
        lambda t: EarningsInfo(ticker=t, earnings_date=when),
    )


def test_earnings_wait_window_skips_when_within(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_earnings(monkeypatch, datetime(2026, 1, 12, tzinfo=UTC))   # 2 days out
    in_window, reason = long_term.in_earnings_wait_window("AAPL", now=_NOW)
    assert in_window is True and "retry later" in reason


def test_earnings_wait_window_proceeds_when_outside_or_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_earnings(monkeypatch, datetime(2026, 2, 1, tzinfo=UTC))    # far out
    assert long_term.in_earnings_wait_window("AAPL", now=_NOW)[0] is False
    _patch_earnings(monkeypatch, None)                               # unknown
    assert long_term.in_earnings_wait_window("AAPL", now=_NOW)[0] is False


# ───────────────────────── candidate generation ─────────────────────────────


def _df(close: float) -> pd.DataFrame:
    return pd.DataFrame({
        "Open": [close], "High": [close], "Low": [close], "Close": [close],
        "Volume": [1_000_000],
    })


def _stub_gates(
    monkeypatch: pytest.MonkeyPatch, *, confirmed: bool = True,
    blocked: bool = False, in_wait: bool = False,
) -> None:
    monkeypatch.setattr(long_term, "fundamental_red_flag", lambda _t: (blocked, ""))
    monkeypatch.setattr(long_term, "confirm_technical_entry", lambda _df, **k: (confirmed, "ok"))
    monkeypatch.setattr(long_term, "in_earnings_wait_window", lambda _t, **k: (in_wait, ""))


def test_generate_candidates_emits_clean_stock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_gates(monkeypatch)
    cands = long_term.generate_candidates(
        lambda _t: _df(190.0), stock_universe=("AAPL",), crypto_universe=(),
    )
    assert len(cands) == 1
    assert cands[0].signal_type == config.LONGTERM_STOCK_SIGNAL
    assert cands[0].entry_price == 190.0


def test_generate_candidates_skips_fundamental_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_gates(monkeypatch, blocked=True)
    assert long_term.generate_candidates(
        lambda _t: _df(190.0), stock_universe=("BADCO",), crypto_universe=(),
    ) == []


def test_generate_candidates_skips_technical_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_gates(monkeypatch, confirmed=False)
    assert long_term.generate_candidates(
        lambda _t: _df(190.0), stock_universe=("AAPL",), crypto_universe=("BTC-USD",),
    ) == []


def test_generate_candidates_skips_earnings_wait_retry_later(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_gates(monkeypatch, in_wait=True)
    # Stock is skipped (earnings window), but crypto has NO earnings gate.
    cands = long_term.generate_candidates(
        lambda _t: _df(100.0), stock_universe=("AAPL",), crypto_universe=("BTC-USD",),
    )
    assert [c.ticker for c in cands] == ["BTC-USD"]


def test_generate_candidates_crypto_is_technical_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Even with a fundamental block flagged, crypto ignores it (technical-only).
    _stub_gates(monkeypatch, blocked=True)
    cands = long_term.generate_candidates(
        lambda _t: _df(64000.0), stock_universe=(), crypto_universe=("BTC-USD",),
    )
    assert len(cands) == 1
    assert cands[0].signal_type == config.LONGTERM_CRYPTO_SIGNAL


def test_generate_candidates_never_emits_crypto_swing_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_gates(monkeypatch)
    cands = long_term.generate_candidates(
        lambda _t: _df(100.0),
        stock_universe=("AAPL",), crypto_universe=("BTC-USD", "ETH-USD"),
    )
    # Every emitted candidate carries ONLY a long-term signal type — the crypto
    # SWING signals can never be produced here.
    assert all(
        c.signal_type in (config.LONGTERM_STOCK_SIGNAL, config.LONGTERM_CRYPTO_SIGNAL)
        for c in cands
    )
    assert all(c.signal_type not in config.DATA_ONLY_SIGNAL_TYPES for c in cands)


def test_generate_candidates_failsoft_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_gates(monkeypatch)

    def boom(_t: str) -> pd.DataFrame | None:
        raise RuntimeError("fetch down")

    assert long_term.generate_candidates(
        boom, stock_universe=("AAPL",), crypto_universe=(),
    ) == []


def test_to_allocation_candidate() -> None:
    lt = LongTermCandidate(
        ticker="AAPL", asset_class="stock",
        signal_type=config.LONGTERM_STOCK_SIGNAL, entry_price=190.0,
    )
    ac = long_term.to_allocation_candidate(lt)
    assert ac.ticker == "AAPL" and ac.direction == "long"
    assert ac.atr is None                    # no ATR/stop sizing for long-term
    assert ac.signal_type == config.LONGTERM_STOCK_SIGNAL


# ───────────────────────── diversification-weighted sizing ──────────────────


def test_position_size_long_term_cap_and_fractional_qty() -> None:
    s = risk.position_size_long_term(10_000.0, 200.0)   # 15% of $10k = $1500
    assert s.ok is True
    assert s.dollars == 1_500.0
    assert s.qty == 7.5                                 # $1500 / $200 fractional


def test_position_size_long_term_is_pool_capital_aware() -> None:
    small = risk.position_size_long_term(2_000.0, 100.0)
    big = risk.position_size_long_term(20_000.0, 100.0)
    assert small.dollars == 300.0                       # 15% of 2k
    assert big.dollars == 3_000.0                       # 15% of 20k


def test_position_size_long_term_custom_weight_cap() -> None:
    s = risk.position_size_long_term(10_000.0, 100.0, max_weight_pct=10.0)
    assert s.dollars == 1_000.0                         # cap enforced at 10%


def test_position_size_long_term_failsoft() -> None:
    assert risk.position_size_long_term(0.0, 100.0).ok is False
    assert risk.position_size_long_term(10_000.0, 0.0).ok is False
    assert risk.position_size_long_term(10_000.0, None).ok is False


# ───────────────────────── protective exit decision (pure) ──────────────────


def test_evaluate_exit_drawdown_stop() -> None:
    # Entry 100, price 74 -> 26% drawdown >= 25% stop.
    d = long_term.evaluate_protective_exit(100.0, 74.0, 0)
    assert d.action == "close" and d.reason == "drawdown_stop"


def test_evaluate_exit_trend_breakdown() -> None:
    # 3 consecutive closes below trend (breakdown_days default 3), mild drawdown.
    d = long_term.evaluate_protective_exit(100.0, 98.0, 3)
    assert d.action == "close" and d.reason == "trend_breakdown"


def test_evaluate_exit_holds_when_neither() -> None:
    # Small drawdown, only 2 closes below trend -> hold (not premature).
    d = long_term.evaluate_protective_exit(100.0, 95.0, 2)
    assert d.action == "hold" and d.reason == "holding"


def test_evaluate_exit_has_no_profit_taking_path() -> None:
    # A large GAIN never triggers a close — protective exits ONLY this phase.
    d = long_term.evaluate_protective_exit(100.0, 150.0, 0)
    assert d.action == "hold"
    # No profit-taking reason exists in the module's exit vocabulary.
    assert not hasattr(long_term, "EXIT_TAKE_PROFIT")


def test_consecutive_closes_below_trend_counts_trailing_run(
    fake_candles: Callable[..., pd.DataFrame],
) -> None:
    # SMA(3); the last two closes dip below the rising trend.
    closes = [10.0, 11.0, 12.0, 13.0, 14.0, 11.0, 10.0]
    df = fake_candles([c + 1 for c in closes], [c - 1 for c in closes], closes)
    assert long_term._consecutive_closes_below_trend(df, trend_period=3) == 2


# ───────────────────────── exit watcher (auto-executing) ────────────────────


def _seed_position(ticker: str, entry: float, qty: float = 10.0) -> int:
    return db.insert_long_term_position(LongTermPosition(
        ticker=ticker, asset_class="stock", entry_price=entry,
        entry_date=datetime(2026, 1, 1, tzinfo=UTC), qty=qty, status="open",
    ))


def _flat_df(price: float, n: int = 5) -> pd.DataFrame:
    return pd.DataFrame({
        "Open": [price] * n, "High": [price] * n, "Low": [price] * n,
        "Close": [price] * n, "Volume": [1_000_000] * n,
    })


def test_watcher_closes_on_drawdown_stop(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_position("AAPL", 100.0)
    b = FakeBroker()
    calls: list[tuple[str, float, str, dict[str, object]]] = []
    original = b.submit_order

    def spy(symbol: str, qty: float, side: str, **kw: object) -> object:
        calls.append((symbol, qty, side, kw))
        return original(symbol, qty, side, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(b, "submit_order", spy)
    actions = long_term.watch_long_term_positions(
        b, price_fetch=lambda _t: _flat_df(70.0),   # 30% drawdown
        now=datetime(2026, 1, 5, tzinfo=UTC),
    )
    assert actions[0].action == "close" and actions[0].reason == "drawdown_stop"
    # A SELL closing order via the equity path (no options).
    _sym, _qty, side, _kw = calls[0]
    assert side == "sell"
    # The position is recorded closed with the exit reason.
    assert db.get_open_long_term_positions() == []
    closed = db.get_long_term_positions()[0]
    assert closed.status == "closed" and closed.exit_reason == "drawdown_stop"


def test_watcher_closes_on_trend_breakdown(tmp_db: Path) -> None:
    _seed_position("MSFT", 100.0)
    # A series that ends with 3+ closes below its short trend, small drawdown.
    closes = [100.0, 102.0, 104.0, 106.0, 108.0, 101.0, 100.0, 99.0]
    df = pd.DataFrame({
        "Open": closes, "High": [c + 1 for c in closes],
        "Low": [c - 1 for c in closes], "Close": closes,
        "Volume": [1_000_000] * len(closes),
    })
    actions = long_term.watch_long_term_positions(
        FakeBroker(), price_fetch=lambda _t: df, now=datetime(2026, 1, 5, tzinfo=UTC),
        trend_period=3,
    )
    assert actions[0].action == "close" and actions[0].reason == "trend_breakdown"


def test_watcher_holds_when_no_exit(tmp_db: Path) -> None:
    _seed_position("AAPL", 100.0)
    actions = long_term.watch_long_term_positions(
        FakeBroker(), price_fetch=lambda _t: _flat_df(101.0),
        now=datetime(2026, 1, 5, tzinfo=UTC), trend_period=3,
    )
    assert actions[0].action == "hold"
    assert len(db.get_open_long_term_positions()) == 1   # still open, nothing sold


def test_watcher_one_error_does_not_block_others(tmp_db: Path) -> None:
    _seed_position("GOOD", 100.0)
    _seed_position("BAD", 100.0)

    def fetch(ticker: str) -> pd.DataFrame | None:
        if ticker == "BAD":
            raise RuntimeError("price feed down")
        return _flat_df(70.0)                # GOOD hits the drawdown stop

    actions = long_term.watch_long_term_positions(
        FakeBroker(), price_fetch=fetch, now=datetime(2026, 1, 5, tzinfo=UTC),
    )
    by_ticker = {a.position.ticker: a.action for a in actions}
    assert by_ticker["GOOD"] == "close"     # processed despite the other's error
    assert by_ticker["BAD"] == "error"


# ───────────── swing-fallback rows keep swing exits (Phase 19) ──────────────


def _seed_swing_fallback(
    ticker: str = "META", entry: float = 480.0, qty: float = 20.0, *,
    tp: float | None = 500.0, sl: float | None = 465.0,
    deadline: datetime | None = None, direction: str = "long",
) -> int:
    return db.insert_long_term_position(LongTermPosition(
        ticker=ticker, asset_class="stock", entry_price=entry,
        entry_date=datetime(2026, 1, 1, tzinfo=UTC), qty=qty, status="open",
        source="swing_fallback", direction=direction, tp=tp, sl=sl,
        deadline=deadline,
    ))


def test_swing_fallback_closes_on_its_own_take_profit(tmp_db: Path) -> None:
    """+4.4% would be a HOLD under long-term rules — the swing TP closes it."""
    _seed_swing_fallback(tp=500.0, sl=465.0)
    actions = long_term.watch_long_term_positions(
        FakeBroker(), price_fetch=lambda _t: _flat_df(501.0),
        now=datetime(2026, 1, 3, tzinfo=UTC),
    )
    assert actions[0].action == "close" and actions[0].reason == "take_profit"
    closed = db.get_long_term_positions()[0]
    assert closed.status == "closed" and closed.exit_reason == "take_profit"


def test_swing_fallback_closes_on_its_own_stop_loss(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """-3.3% is nowhere near the 25% long-term drawdown stop — the swing SL
    closes it anyway, with a SELL via the equity path."""
    _seed_swing_fallback(tp=500.0, sl=465.0)
    b = FakeBroker()
    calls: list[str] = []
    original = b.submit_order

    def spy(symbol: str, qty: float, side: str, **kw: object) -> object:
        calls.append(side)
        return original(symbol, qty, side, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(b, "submit_order", spy)
    actions = long_term.watch_long_term_positions(
        b, price_fetch=lambda _t: _flat_df(464.0),
        now=datetime(2026, 1, 3, tzinfo=UTC),
    )
    assert actions[0].action == "close" and actions[0].reason == "stop_loss"
    assert calls == ["sell"]


def test_swing_fallback_never_evaluates_longterm_rules(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 27% drawdown WOULD trip the long-term drawdown stop — but a
    swing-fallback position holds because its own SL is lower, and the
    long-term trigger functions are never even called for it."""
    _seed_swing_fallback(entry=480.0, tp=1_000.0, sl=100.0)
    protective_calls: list[object] = []
    trend_calls: list[object] = []
    original_protective = long_term.evaluate_protective_exit
    original_trend = long_term._consecutive_closes_below_trend
    monkeypatch.setattr(
        long_term, "evaluate_protective_exit",
        lambda *a, **k: (protective_calls.append(a), original_protective(*a, **k))[1],
    )
    monkeypatch.setattr(
        long_term, "_consecutive_closes_below_trend",
        lambda *a, **k: (trend_calls.append(a), original_trend(*a, **k))[1],
    )

    actions = long_term.watch_long_term_positions(
        FakeBroker(), price_fetch=lambda _t: _flat_df(350.0),   # -27% from entry
        now=datetime(2026, 1, 3, tzinfo=UTC),
    )
    assert actions[0].action == "hold"           # swing SL (100) not hit
    assert protective_calls == []                # long-term rules never ran
    assert trend_calls == []
    assert len(db.get_open_long_term_positions()) == 1


def test_swing_fallback_hold_deadline_closes_like_a_swing(tmp_db: Path) -> None:
    _seed_swing_fallback(
        tp=500.0, sl=465.0, deadline=datetime(2026, 1, 4, tzinfo=UTC),
    )
    actions = long_term.watch_long_term_positions(
        FakeBroker(), price_fetch=lambda _t: _flat_df(481.0),   # between TP/SL
        now=datetime(2026, 1, 5, tzinfo=UTC),                   # past deadline
    )
    assert actions[0].action == "close" and actions[0].reason == "hold_deadline"
    closed = db.get_long_term_positions()[0]
    assert closed.exit_reason == "hold_deadline"


def test_swing_fallback_short_closes_with_buy(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A put-signal fallback SOLD shares; its stop sits ABOVE entry and the
    close must BUY the shares back."""
    _seed_swing_fallback(direction="short", tp=460.0, sl=495.0)
    b = FakeBroker()
    calls: list[str] = []
    original = b.submit_order

    def spy(symbol: str, qty: float, side: str, **kw: object) -> object:
        calls.append(side)
        return original(symbol, qty, side, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(b, "submit_order", spy)
    actions = long_term.watch_long_term_positions(
        b, price_fetch=lambda _t: _flat_df(496.0),   # above the short's stop
        now=datetime(2026, 1, 3, tzinfo=UTC),
    )
    assert actions[0].action == "close" and actions[0].reason == "stop_loss"
    assert calls == ["buy"]


def test_swing_fallback_without_levels_holds(tmp_db: Path) -> None:
    """Today's execute path passes deadline=None (no hold window on a
    PlannedOrder yet); with no TP/SL either, the watcher holds — it never
    falls back to the long-term rules."""
    _seed_swing_fallback(tp=None, sl=None, deadline=None)
    actions = long_term.watch_long_term_positions(
        FakeBroker(), price_fetch=lambda _t: _flat_df(350.0),   # -27%
        now=datetime(2026, 1, 3, tzinfo=UTC),
    )
    assert actions[0].action == "hold"


def test_mixed_book_neither_path_affects_the_other(tmp_db: Path) -> None:
    """Regression for genuine long-term behavior: in a mixed book, the
    long_term row still closes on its drawdown stop while the swing-fallback
    row (down just as far but above its own SL) holds."""
    _seed_position("HOLD5", 100.0)                      # genuine long-term
    _seed_swing_fallback(ticker="SWNG", entry=100.0, tp=200.0, sl=10.0)

    actions = long_term.watch_long_term_positions(
        FakeBroker(), price_fetch=lambda _t: _flat_df(70.0),   # -30% for both
        now=datetime(2026, 1, 5, tzinfo=UTC),
    )
    by_ticker = {a.position.ticker: (a.action, a.reason) for a in actions}
    assert by_ticker["HOLD5"] == ("close", "drawdown_stop")   # unchanged rules
    assert by_ticker["SWNG"] == ("hold", "holding")           # its own SL wins


def test_cli_longterm_positions_shows_source(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_bot import __main__ as m
    _seed_position("AAPL", 100.0)
    _seed_swing_fallback(ticker="META", tp=500.0, sl=465.0)
    monkeypatch.setattr(m.long_term, "fetch_daily_candles", lambda _t: _flat_df(99.0))
    m.cmd_longterm_positions()
    out = capsys.readouterr().out
    assert "long_term" in out
    assert "swing_fallback" in out
    assert "tp=" in out and "sl=" in out    # the levels the watcher applies


# ───────────────────────── position persistence round-trip ──────────────────


def test_long_term_position_db_round_trip(tmp_db: Path) -> None:
    pid = _seed_position("AAPL", 190.0, qty=5.0)
    assert len(db.get_open_long_term_positions()) == 1
    db.update_long_term_position(
        pid, status="closed", exit_price=140.0,
        exit_date=datetime(2026, 2, 1, tzinfo=UTC), exit_reason="drawdown_stop",
    )
    assert db.get_open_long_term_positions() == []
    closed = db.get_long_term_positions()[0]
    assert closed.status == "closed" and closed.exit_price == 140.0


def test_update_long_term_position_rejects_unknown_field(tmp_db: Path) -> None:
    pid = _seed_position("AAPL", 190.0)
    with pytest.raises(ValueError, match="Unknown long_term_position field"):
        db.update_long_term_position(pid, entry_price=999.0)


# ───────────────────────── long-term CLI ────────────────────────────────────


def test_cli_longterm_candidates(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_bot import __main__ as m
    monkeypatch.setattr(
        m.long_term, "generate_candidates",
        lambda _fetch: [LongTermCandidate(
            ticker="AAPL", asset_class="stock",
            signal_type=config.LONGTERM_STOCK_SIGNAL, entry_price=190.0,
            entry_rationale="ok",
        )],
    )
    m.cmd_longterm_candidates()
    out = capsys.readouterr().out
    assert "LONG-TERM ENTRY CANDIDATES" in out
    assert "AAPL" in out
    assert "Logged 1 candidate(s) to signals" in out
    assert "Nothing is submitted" in out
    # Phase 17: the candidate was persisted as a fired signal (no trades row —
    # long-term outcomes are tracked in long_term_positions, not the resolver).
    (sig,) = db.get_signals()
    assert sig.ticker == "AAPL"
    assert sig.signal_type == config.LONGTERM_STOCK_SIGNAL
    assert sig.id is not None
    assert db.get_trade_by_signal_id(sig.id) is None


def test_cli_longterm_candidates_empty(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_bot import __main__ as m
    monkeypatch.setattr(m.long_term, "generate_candidates", lambda _fetch: [])
    m.cmd_longterm_candidates()
    assert "(none)" in capsys.readouterr().out


def test_cli_longterm_positions(
    monkeypatch: pytest.MonkeyPatch, tmp_db: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_bot import __main__ as m
    _seed_position("AAPL", 100.0)
    monkeypatch.setattr(m.long_term, "fetch_daily_candles", lambda _t: _flat_df(80.0))
    m.cmd_longterm_positions()
    out = capsys.readouterr().out
    assert "AAPL" in out
    assert "20.0" in out                 # 20% drawdown from 100 -> 80


def test_cli_longterm_positions_empty(
    tmp_db: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_bot import __main__ as m
    m.cmd_longterm_positions()
    assert "(none)" in capsys.readouterr().out
