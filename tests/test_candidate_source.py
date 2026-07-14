"""Tests for trading_bot.candidate_source (Phase 17 — live signal wiring).

Mocked data only, no live network. The core disciplines under test: the
recency / considered-once / track boundaries are enforced by the query itself;
crypto swing signals flow through as candidates but are still dropped by
build_plan's EXISTING data-only filter (the two-layer guarantee, now proven
against live-sourced data); marking happens on inclusion in an execute-bound
pull, never on execution; and the crypto symbol translation at the Alpaca
boundary.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from trading_bot import allocation, candidate_source, config, db
from trading_bot.broker import alpaca
from trading_bot.broker.base import AccountInfo
from trading_bot.broker.fake import FakeBroker
from trading_bot.broker.options import OptionChainResult
from trading_bot.models import Signal, Trade

_NOW = datetime(2026, 7, 10, 15, 0, tzinfo=UTC)
_ACCOUNT = AccountInfo(ok=True, equity=100_000.0, cash=100_000.0)


def _seed_signal(
    *,
    ticker: str = "META",
    signal_type: str = "ema21_pullback",
    asset_class: str = "stock",
    direction: str = "call",
    entry: float = 480.0,
    atr: float | None = 8.0,
    rsi: float | None = 55.0,
    earnings_risk: bool = False,
    timestamp: datetime = _NOW,
    hold_days: int | None = None,
) -> int:
    return db.insert_signal(Signal(
        timestamp=timestamp, ticker=ticker, asset_class=asset_class,
        signal_type=signal_type, direction=direction, entry_price=entry,
        atr=atr, rsi=rsi, earnings_risk=earnings_risk,
        hold_estimate_days=hold_days,
    ))


def _seed_trade(signal_id: int, **overrides: Any) -> int:
    fields: dict[str, Any] = {
        "signal_id": signal_id, "opened_at": _NOW, "outcome": "open",
        "track_mode": "active", "ind_rsi": 52.0, "ind_adx": 27.0,
        "ind_obv": 1_200_000.0, "ind_vol_regime": "low",
        "ind_concentration": "diversified", "sentiment_score": 0.4,
    }
    fields.update(overrides)
    return db.insert_trade(Trade(**fields))


def _live(**kwargs: Any) -> list[allocation.Candidate]:
    kwargs.setdefault("now", _NOW)
    return candidate_source.live_candidates(**kwargs)


# ───────────────────────── recency window ────────────────────────────────────


def test_fresh_signal_is_returned(tmp_db: Path) -> None:
    sid = _seed_signal()
    _seed_trade(sid)
    (candidate,) = _live()
    assert candidate.ticker == "META"
    assert candidate.signal_type == "ema21_pullback"


def test_stale_signal_outside_window_is_excluded(tmp_db: Path) -> None:
    stale = _NOW - timedelta(hours=config.CANDIDATE_RECENCY_HOURS, minutes=1)
    sid = _seed_signal(timestamp=stale)
    _seed_trade(sid)
    assert _live() == []


def test_signal_exactly_at_window_edge_is_included(tmp_db: Path) -> None:
    edge = _NOW - timedelta(hours=config.CANDIDATE_RECENCY_HOURS)
    sid = _seed_signal(timestamp=edge)
    _seed_trade(sid)
    assert len(_live()) == 1


# ───────────────────────── considered-once boundary ──────────────────────────


def test_considered_signal_is_excluded_by_the_query(tmp_db: Path) -> None:
    sid = _seed_signal()
    _seed_trade(sid)
    db.mark_signals_considered([sid], _NOW)
    assert _live() == []


def test_mark_on_pull_not_on_execution(tmp_db: Path) -> None:
    """A planned-but-never-executed signal is consumed by the pull itself."""
    sid = _seed_signal()
    _seed_trade(sid)

    pulled = _live(mark_considered=True)      # pulled into an execute-bound plan
    assert len(pulled) == 1                   # ... which we then never execute

    signal = db.get_signal_by_id(sid)
    assert signal is not None
    assert signal.considered_at == _NOW       # stamped at PULL time
    assert _live(now=_NOW + timedelta(minutes=5)) == []   # never re-offered


def test_preview_pull_does_not_consume(tmp_db: Path) -> None:
    sid = _seed_signal()
    _seed_trade(sid)
    assert len(_live(mark_considered=False)) == 1
    signal = db.get_signal_by_id(sid)
    assert signal is not None
    assert signal.considered_at is None
    assert len(_live()) == 1                  # still offered to the real pull


def test_first_considered_stamp_wins(tmp_db: Path) -> None:
    sid = _seed_signal()
    _seed_trade(sid)
    assert db.mark_signals_considered([sid], _NOW) == 1
    later = _NOW + timedelta(hours=1)
    assert db.mark_signals_considered([sid], later) == 0   # already stamped
    signal = db.get_signal_by_id(sid)
    assert signal is not None
    assert signal.considered_at == _NOW


# ───────────────────────── track / outcome boundary ──────────────────────────


def test_shadow_signal_is_excluded(tmp_db: Path) -> None:
    sid = _seed_signal(ticker="SHDW")
    _seed_trade(sid, track_mode="shadow")
    assert _live() == []


def test_resolved_trade_signal_is_excluded(tmp_db: Path) -> None:
    sid = _seed_signal()
    _seed_trade(sid, outcome="win", closed_at=_NOW)
    assert _live() == []


def test_signal_without_trade_row_is_included(tmp_db: Path) -> None:
    """Long-term entries have no trades row — the LEFT JOIN keeps them."""
    _seed_signal(
        ticker="BTC-USD", signal_type="long_term_crypto", asset_class="crypto",
        direction="long", entry=64_000.0, atr=None, rsi=None,
    )
    (candidate,) = _live()
    assert candidate.signal_type == "long_term_crypto"
    assert candidate.ticker_active is True    # permissive defaults, per Phase 14
    assert candidate.pair_enabled is True
    assert candidate.atr is None


# ───────────────────────── candidate shape mapping ───────────────────────────


def test_candidate_maps_trade_context_and_gates(tmp_db: Path) -> None:
    db.add_to_active_watchlist("META", "seed")
    sid = _seed_signal(earnings_risk=True)
    _seed_trade(sid)

    (c,) = _live()
    assert c.direction == "call"
    assert c.asset_class == "stock"
    assert c.entry == 480.0
    assert c.atr == 8.0
    assert c.earnings_blackout is True        # from signals.earnings_risk
    assert c.rsi == 52.0                      # trade ind_rsi preferred
    assert c.adx == 27.0
    assert c.obv == 1_200_000.0
    assert c.vol_regime == "low"
    assert c.sentiment_score == 0.4
    assert c.concentration == "diversified"
    assert c.ticker_active is True


def test_candidate_rsi_falls_back_to_signal_column(tmp_db: Path) -> None:
    sid = _seed_signal(rsi=61.0)
    _seed_trade(sid, ind_rsi=None)
    (c,) = _live()
    assert c.rsi == 61.0


def test_benched_ticker_maps_inactive(tmp_db: Path) -> None:
    db.add_to_active_watchlist("META", "seed")
    db.set_watchlist_status("META", "benched")
    sid = _seed_signal()
    _seed_trade(sid)
    (c,) = _live()
    assert c.ticker_active is False           # build_plan will drop it


def test_muted_pair_maps_disabled(tmp_db: Path) -> None:
    db.set_signal_pair_status("META", "ema21_pullback", "muted")
    sid = _seed_signal()
    _seed_trade(sid)
    (c,) = _live()
    assert c.pair_enabled is False            # build_plan will drop it


def test_dedupe_keeps_newest_signal_per_pair(tmp_db: Path) -> None:
    older = _seed_signal(entry=470.0, timestamp=_NOW - timedelta(hours=2))
    newer = _seed_signal(entry=480.0, timestamp=_NOW - timedelta(hours=1))
    _seed_trade(older)
    _seed_trade(newer)

    (c,) = _live(mark_considered=True)
    assert c.entry == 480.0                   # the re-fired setup supersedes

    # BOTH rows were pulled, so BOTH are consumed — the superseded older
    # duplicate is never re-offered on its own next cycle.
    for sid in (older, newer):
        signal = db.get_signal_by_id(sid)
        assert signal is not None
        assert signal.considered_at is not None


# ───────────────────────── hold-window carrying (Phase 20) ──────────────────


def test_swing_candidate_carries_resolver_hold_deadline(tmp_db: Path) -> None:
    """The signal's own hold estimate becomes the carried deadline — fire
    timestamp + the Phase 1 resolver window, nothing invented."""
    sid = _seed_signal(hold_days=3)
    _seed_trade(sid)
    (c,) = _live()
    assert c.hold_deadline == _NOW + timedelta(days=3)


def test_swing_candidate_without_estimate_uses_resolver_default(
    tmp_db: Path,
) -> None:
    """Live signals don't persist an estimate today — the carried deadline is
    then the SAME asset-class default the resolver has settled trades on since
    Phase 1 (30 days for stock)."""
    from trading_bot import outcomes

    sid = _seed_signal(hold_days=None)
    _seed_trade(sid)
    (c,) = _live()
    expected = _NOW + outcomes.hold_window_for(None, "stock")
    assert c.hold_deadline == expected == _NOW + timedelta(days=30)


def test_long_term_candidates_carry_no_hold_deadline(tmp_db: Path) -> None:
    """LONG_TERM / CRYPTO pools have no time stop by design."""
    _seed_signal(
        ticker="AAPL", signal_type="long_term_stock", direction="long",
        atr=None, rsi=None,
    )
    _seed_signal(
        ticker="BTC-USD", signal_type="long_term_crypto", asset_class="crypto",
        direction="long", entry=64_000.0, atr=None, rsi=None,
    )
    candidates = _live()
    assert len(candidates) == 2
    assert all(c.hold_deadline is None for c in candidates)


def test_hold_deadline_survives_allocation_into_planned_order(
    tmp_db: Path,
) -> None:
    sid = _seed_signal(hold_days=2)
    _seed_trade(sid)
    result = allocation.build_plan(_live(), _ACCOUNT)
    (order,) = result.plan.orders
    assert order.hold_deadline == _NOW + timedelta(days=2)   # pass-through


def test_fired_signal_carries_tightened_window_downstream(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Phase 21 integration: a signal fired through the REAL fire path
    (scanner.log_signal) now persists its estimate, so the Phase 20
    hold_deadline reflects the signal's own 5-day intent — tightened from the
    30-day default every pre-fix signal fell back to."""
    from trading_bot import regime, scanner, vix

    def _no_regime(*_a: object, **_k: object) -> object:
        raise regime.RegimeFetchError("offline")

    def _no_vix(*_a: object, **_k: object) -> object:
        raise vix.VixFetchError("offline")

    monkeypatch.setattr("trading_bot.regime.get_current_regime", _no_regime)
    monkeypatch.setattr("trading_bot.vix.get_current_vix", _no_vix)

    sid = scanner.log_signal({
        "ticker": "META", "asset_type": "stock",
        "trade_type": "📆 SWING TRADE", "direction": "CALL 📈",
        "setup": "EMA21 Pullback", "price": 480.0,
        "take_profit": 496.0, "stop_loss": 468.0,
        "hold_days": "3-5 days", "confidence": "High",
    })
    assert sid is not None
    signal = db.get_signal_by_id(sid)
    assert signal is not None

    (c,) = candidate_source.live_candidates(now=signal.timestamp)
    assert c.hold_deadline == signal.timestamp + timedelta(days=5)   # not +30


# ───────────────────────── ATR forward-only proof (Phase 22) ─────────────────


def test_old_null_atr_row_is_still_filtered_unsizeable(tmp_db: Path) -> None:
    """THE Phase 22 forward-only proof for atr: a pre-fix signal row (NULL
    atr — the shape of every existing row) still produces an unsizeable
    candidate that build_plan filters out, exactly as today. No backfill,
    no behavior change for existing data."""
    sid = _seed_signal(atr=None)
    _seed_trade(sid)
    (candidate,) = _live()
    assert candidate.atr is None
    result = allocation.build_plan([candidate], _ACCOUNT)
    assert result.plan.orders == []
    (skip,) = result.skipped
    assert skip.reason == "unsizeable"          # identical to pre-fix behavior


def test_new_fired_signal_persists_real_atr_and_sizes(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A signal fired through the REAL path now carries its ATR into a
    sizeable candidate — the gap that kept every live SWING signal out of
    execution is closed."""
    from trading_bot import regime, scanner, vix

    def _no_regime(*_a: object, **_k: object) -> object:
        raise regime.RegimeFetchError("offline")

    def _no_vix(*_a: object, **_k: object) -> object:
        raise vix.VixFetchError("offline")

    monkeypatch.setattr("trading_bot.regime.get_current_regime", _no_regime)
    monkeypatch.setattr("trading_bot.vix.get_current_vix", _no_vix)

    sid = scanner.log_signal({
        "ticker": "META", "asset_type": "stock",
        "trade_type": "📆 SWING TRADE", "direction": "CALL 📈",
        "setup": "EMA21 Pullback", "price": 480.0,
        "take_profit": 496.0, "stop_loss": 468.0,
        "hold_days": "1-2 days", "atr": 8.0, "confidence": "High",
    })
    assert sid is not None
    signal = db.get_signal_by_id(sid)
    assert signal is not None
    assert signal.atr == 8.0                    # persisted at fire

    (candidate,) = candidate_source.live_candidates(now=signal.timestamp)
    assert candidate.atr == 8.0
    result = allocation.build_plan([candidate], _ACCOUNT)
    (order,) = result.plan.orders               # sizeable and PLANNED
    assert order.ticker == "META"
    assert order.qty > 0


def test_mixed_old_and_new_signals_only_new_is_plannable(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Section 4's downstream regression: with an old NULL-atr row and a
    newly fired signal side by side, the plan contains ONLY the new one —
    the old row keeps today's unsizeable-filtered behavior, untouched."""
    from trading_bot import regime, scanner, vix

    def _no_regime(*_a: object, **_k: object) -> object:
        raise regime.RegimeFetchError("offline")

    def _no_vix(*_a: object, **_k: object) -> object:
        raise vix.VixFetchError("offline")

    monkeypatch.setattr("trading_bot.regime.get_current_regime", _no_regime)
    monkeypatch.setattr("trading_bot.vix.get_current_vix", _no_vix)

    new_sid = scanner.log_signal({
        "ticker": "META", "asset_type": "stock",
        "trade_type": "📆 SWING TRADE", "direction": "CALL 📈",
        "setup": "EMA21 Pullback", "price": 480.0,
        "take_profit": 496.0, "stop_loss": 468.0,
        "hold_days": "1-2 days", "atr": 8.0, "confidence": "High",
    })
    assert new_sid is not None
    new_signal = db.get_signal_by_id(new_sid)
    assert new_signal is not None

    old_sid = _seed_signal(                      # pre-fix shape: NULL atr
        ticker="GOOGL", entry=175.0, atr=None, timestamp=new_signal.timestamp,
    )
    _seed_trade(old_sid, opened_at=new_signal.timestamp)

    candidates = candidate_source.live_candidates(now=new_signal.timestamp)
    result = allocation.build_plan(candidates, _ACCOUNT)

    (order,) = result.plan.orders
    assert order.ticker == "META"                # the new signal is plannable
    assert [(s.ticker, s.reason) for s in result.skipped] == [
        ("GOOGL", "unsizeable"),                 # the old row: unchanged
    ]


# ───────────────────────── two-layer crypto-swing guarantee (live data) ──────


def test_crypto_swing_signals_are_sourced_but_never_planned(tmp_db: Path) -> None:
    """The live equivalent of the sample-based data-only test: real crypto
    swing signals ARE candidates (data flows keep working) but build_plan's
    EXISTING filter still drops them before routing."""
    for signal_type in ("oversold_reversal", "momentum_breakout"):
        sid = _seed_signal(
            ticker="BTC-USD", signal_type=signal_type, asset_class="crypto",
            direction="long", entry=64_000.0, atr=1_500.0,
        )
        _seed_trade(sid, ind_vol_regime="normal", sentiment_score=0.2)

    candidates = _live()
    assert {c.signal_type for c in candidates} == {
        "oversold_reversal", "momentum_breakout",
    }                                          # sourced — data present

    result = allocation.build_plan(candidates, _ACCOUNT)
    assert result.plan.orders == []            # never planned
    assert len(result.skipped) == 2
    assert all("data-only" in s.reason for s in result.skipped)


def test_live_swing_signal_reaches_the_plan(tmp_db: Path) -> None:
    sid = _seed_signal()
    _seed_trade(sid)
    result = allocation.build_plan(_live(), _ACCOUNT)
    (order,) = result.plan.orders
    assert order.ticker == "META"
    assert order.pool == config.POOL_SWING


def test_sample_candidates_still_works_standalone(tmp_db: Path) -> None:
    cands = allocation.sample_candidates()
    assert len(cands) >= 4                     # demos/tests unaffected
    result = allocation.build_plan(cands, _ACCOUNT)
    assert result.plan.orders                  # the fixture still plans


# ───────────────────────── crypto symbol translation ─────────────────────────


def test_to_alpaca_symbol_translates_crypto_pairs() -> None:
    assert alpaca.to_alpaca_symbol("BTC-USD") == "BTC/USD"
    assert alpaca.to_alpaca_symbol("ETH-USD") == "ETH/USD"
    assert alpaca.to_alpaca_symbol("BNB-USD") == "BNB/USD"


def test_to_alpaca_symbol_leaves_non_crypto_untouched() -> None:
    assert alpaca.to_alpaca_symbol("AAPL") == "AAPL"
    assert alpaca.to_alpaca_symbol("BRK-B") == "BRK-B"          # not a -USD pair
    assert alpaca.to_alpaca_symbol("AAPL260116C00190000") == "AAPL260116C00190000"
    assert alpaca.to_alpaca_symbol("btc-usd") == "btc-usd"      # not wire-format


class _FakeResp:
    def __init__(self, payload: object, status: int = 200) -> None:
        self.status_code = status
        self.content = b"{}"
        self._payload = payload

    def json(self) -> object:
        return self._payload


def test_alpaca_submit_sends_translated_crypto_symbol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wire body carries BTC/USD while callers keep passing BTC-USD."""
    monkeypatch.setattr(
        "trading_bot.broker.alpaca.secrets.get_secret", lambda _n: "k",
    )
    captured: list[dict[str, Any]] = []

    def fake(method: str, url: str, **kw: Any) -> _FakeResp:
        captured.append({"method": method, "url": url, **kw})
        return _FakeResp({"id": "abc", "status": "accepted", "symbol": "BTC/USD"})

    monkeypatch.setattr("trading_bot.broker.alpaca.requests.request", fake)
    result = alpaca.AlpacaBroker().submit_order(
        "BTC-USD", 0.05, "buy", limit_price=64_000.0, time_in_force="day",
    )
    assert result.ok is True
    assert captured[0]["json"]["symbol"] == "BTC/USD"    # translated at the wire


def test_alpaca_submit_leaves_stock_symbol_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "trading_bot.broker.alpaca.secrets.get_secret", lambda _n: "k",
    )
    captured: list[dict[str, Any]] = []

    def fake(method: str, url: str, **kw: Any) -> _FakeResp:
        captured.append({"method": method, "url": url, **kw})
        return _FakeResp({"id": "abc", "status": "accepted", "symbol": "AAPL"})

    monkeypatch.setattr("trading_bot.broker.alpaca.requests.request", fake)
    alpaca.AlpacaBroker().submit_order("AAPL", 1.0, "buy", limit_price=190.0)
    assert captured[0]["json"]["symbol"] == "AAPL"


# ───────────────────────── CLI wiring (live default) ─────────────────────────


def test_cli_allocate_plan_live_shows_seeded_signal_without_consuming(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_bot import __main__ as m
    now = datetime.now(UTC)
    sid = _seed_signal(timestamp=now)
    _seed_trade(sid, opened_at=now)
    monkeypatch.setattr(m.broker, "AlpacaBroker", lambda: FakeBroker())

    m.cmd_allocate_plan()
    out = capsys.readouterr().out
    assert "live fired signals" in out
    assert "META" in out
    signal = db.get_signal_by_id(sid)
    assert signal is not None
    assert signal.considered_at is None        # the review step never consumes


def test_cli_execute_confirm_live_submits_and_marks_considered(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_bot import __main__ as m

    class FakeChainClient:
        def get_option_chain(self, _underlying: str) -> OptionChainResult:
            return OptionChainResult(ok=False, reason="options not enabled")

    now = datetime.now(UTC)
    sid = _seed_signal(timestamp=now)
    _seed_trade(sid, opened_at=now)
    broker = FakeBroker()
    monkeypatch.setattr(m.broker, "AlpacaBroker", lambda: broker)
    monkeypatch.setattr(m.broker, "AlpacaOptionsClient", FakeChainClient)

    m.cmd_allocate_execute(confirm=True)
    out = capsys.readouterr().out
    assert "PLAN EXECUTION" in out
    assert broker._orders                      # a paper order went out
    signal = db.get_signal_by_id(sid)
    assert signal is not None
    assert signal.considered_at is not None    # consumed by the confirm pull

    # A second confirm run finds nothing new to submit.
    orders_after_first = len(broker._orders)
    m.cmd_allocate_execute(confirm=True)
    assert len(broker._orders) == orders_after_first
