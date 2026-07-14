"""Tests for trading_bot.plan_execution (Phase 16 — plan → broker bridge)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from trading_bot import config, db, long_term, risk_of_ruin
from trading_bot import plan_execution as pe
from trading_bot.allocation import ExecutionPlan, PlannedOrder
from trading_bot.broker.base import STATUS_REJECTED, OrderResult
from trading_bot.broker.fake import FakeBroker
from trading_bot.broker.options import OptionChainResult, OptionContract
from trading_bot.models import PlanExecution

# 11:00 ET on a July (EDT, UTC-4) trading day.
_NOW = datetime(2026, 7, 10, 15, 0, tzinfo=UTC)


def _execution(
    *,
    ticker: str = "META",
    pool: str = "SWING",
    status: str = pe.STATUS_SUBMITTED,
    executed_at: datetime = _NOW,
    order_ref: str | None = "fake-1",
) -> PlanExecution:
    return PlanExecution(
        plan_id=pe.make_plan_id(executed_at),
        executed_at=executed_at,
        ticker=ticker,
        pool=pool,
        status=status,
        signal_type="ema21_pullback",
        side="buy",
        qty=2.0,
        vehicle="option_full",
        order_ref=order_ref,
        reason="ok",
    )


# ───────────────────────── audit-table roundtrip ─────────────────────────────


def test_insert_and_read_roundtrip(tmp_db: Path) -> None:
    row_id = db.insert_plan_execution(_execution())
    rows = db.get_plan_executions()
    assert len(rows) == 1
    got = rows[0]
    assert got.id == row_id
    assert got.ticker == "META"
    assert got.pool == "SWING"
    assert got.status == "submitted"
    assert got.signal_type == "ema21_pullback"
    assert got.side == "buy"
    assert got.qty == 2.0
    assert got.vehicle == "option_full"
    assert got.order_ref == "fake-1"
    assert got.reason == "ok"
    assert got.executed_at == _NOW
    assert got.plan_id == pe.make_plan_id(_NOW)


def test_insert_invalid_status_raises(tmp_db: Path) -> None:
    with pytest.raises(ValueError, match="Invalid plan-execution status"):
        db.insert_plan_execution(_execution(status="filled"))


def test_get_plan_executions_newest_first(tmp_db: Path) -> None:
    earlier = datetime(2026, 7, 10, 14, 0, tzinfo=UTC)
    db.insert_plan_execution(_execution(ticker="AAA", executed_at=earlier))
    db.insert_plan_execution(_execution(ticker="BBB", executed_at=_NOW))
    rows = db.get_plan_executions()
    assert [r.ticker for r in rows] == ["BBB", "AAA"]


# ───────────────────────── idempotency guard ─────────────────────────────────


def test_already_executed_blocks_same_cycle(tmp_db: Path) -> None:
    db.insert_plan_execution(_execution())
    assert pe.already_executed("META", "SWING", _NOW) is True


def test_already_executed_is_per_ticker_and_pool(tmp_db: Path) -> None:
    db.insert_plan_execution(_execution())
    assert pe.already_executed("GOOGL", "SWING", _NOW) is False
    assert pe.already_executed("META", "LONG_TERM", _NOW) is False


def test_already_executed_ignores_non_submitted(tmp_db: Path) -> None:
    for status in (pe.STATUS_REJECTED, pe.STATUS_ERROR, pe.STATUS_SKIPPED):
        db.insert_plan_execution(_execution(status=status, order_ref=None))
    assert pe.already_executed("META", "SWING", _NOW) is False


def test_already_executed_prior_cycle_does_not_block(tmp_db: Path) -> None:
    # 22:00 ET the previous day — a prior cycle, so today may execute.
    yesterday_et = datetime(2026, 7, 10, 2, 0, tzinfo=UTC)
    db.insert_plan_execution(_execution(executed_at=yesterday_et))
    assert pe.already_executed("META", "SWING", _NOW) is False


def test_cycle_start_is_midnight_et() -> None:
    start = pe.cycle_start(_NOW)
    # Midnight ET on 2026-07-10 is 04:00 UTC (EDT).
    assert start == datetime(2026, 7, 10, 4, 0, tzinfo=UTC)


def test_make_plan_id_derives_from_timestamp() -> None:
    assert pe.make_plan_id(_NOW) == "plan-20260710T150000Z"


# ───────────────────────── authorization gate (Phase 15) ─────────────────────


def test_check_authorization_permissive_by_default(tmp_db: Path) -> None:
    check = pe.check_authorization()
    assert check.authorized is True
    assert "authorized" in check.reason


def test_check_authorization_refuses_when_revoked(tmp_db: Path) -> None:
    risk_of_ruin.revoke(config.ENTRY_CAPABILITY, "tier1: 7 consecutive losses")
    check = pe.check_authorization()
    assert check.authorized is False
    assert "REVOKED" in check.reason
    assert "tier1: 7 consecutive losses" in check.reason
    assert "risk state" in check.reason


def test_check_authorization_restored_after_reauthorize(tmp_db: Path) -> None:
    risk_of_ruin.revoke(config.ENTRY_CAPABILITY, "tier1: drawdown")
    assert pe.check_authorization().authorized is False
    risk_of_ruin.authorize(config.ENTRY_CAPABILITY)
    assert pe.check_authorization().authorized is True


# ───────────────────────── routing fixtures ──────────────────────────────────


def _swing_order(
    *, ticker: str = "META", qty: float = 10.0, entry: float = 480.0,
    est_cost: float = 600.0, dollar_risk: float = 150.0, side: str = "buy",
    rank: int = 1, hold_deadline: datetime | None = None,
) -> PlannedOrder:
    return PlannedOrder(
        rank=rank, pool=config.POOL_SWING, tier="HIGH", ticker=ticker,
        signal_type="ema21_pullback", side=side, qty=qty, entry=entry,
        est_cost=est_cost, dollar_risk=dollar_risk, score=0.8,
        hold_deadline=hold_deadline,
    )


def _long_term_order(
    *, ticker: str = "AAPL", pool: str = config.POOL_LONG_TERM,
    qty: float = 45.0, entry: float = 100.0, rank: int = 2,
    hold_deadline: datetime | None = None,
) -> PlannedOrder:
    signal = "long_term_crypto" if pool == config.POOL_CRYPTO else "long_term_stock"
    return PlannedOrder(
        rank=rank, pool=pool, tier="NORMAL", ticker=ticker, signal_type=signal,
        side="buy", qty=qty, entry=entry, est_cost=qty * entry,
        dollar_risk=qty * entry, score=0.4, hold_deadline=hold_deadline,
    )


def _contract(*, delta: float = 0.70, strike: float = 480.0) -> OptionContract:
    expiry = (_NOW + timedelta(days=30)).date().isoformat()
    return OptionContract(
        symbol=f"META-call-{strike:g}", underlying="META", option_type="call",
        strike=strike, expiry=expiry, delta=delta, theta=-0.05, vega=0.1,
        gamma=0.01, open_interest=500, bid=4.9, ask=5.1, mid=5.0,
    )


def _chain_fetch(_underlying: str) -> OptionChainResult:
    return OptionChainResult(ok=True, contracts=[_contract()])


class SpyBroker(FakeBroker):
    """FakeBroker that counts submissions and can reject specific symbols."""

    def __init__(self, *, reject_symbols: frozenset[str] = frozenset()) -> None:
        super().__init__()
        self.submissions: list[str] = []
        self._reject_symbols = reject_symbols

    def submit_order(
        self, symbol: str, qty: float, side: str, **kwargs: object,
    ) -> OrderResult:
        self.submissions.append(symbol)
        if symbol in self._reject_symbols:
            return OrderResult(
                ok=False, status=STATUS_REJECTED, symbol=symbol,
                reason="insufficient buying power",
            )
        return super().submit_order(symbol, qty, side, **kwargs)  # type: ignore[arg-type]


# ───────────────────────── execute_plan: authorization refusal ───────────────


def test_execute_plan_refuses_when_unauthorized_zero_broker_calls(
    tmp_db: Path,
) -> None:
    risk_of_ruin.revoke(config.ENTRY_CAPABILITY, "tier1: 7 consecutive losses")
    broker = SpyBroker()
    plan = ExecutionPlan(orders=[_swing_order(), _long_term_order()])

    run = pe.execute_plan(broker, plan, now=_NOW, option_chain_fetch=_chain_fetch)

    assert run.ok is False
    assert "REVOKED" in run.note
    assert run.executions == []
    assert broker.submissions == []          # ZERO broker calls
    assert db.get_plan_executions() == []    # nothing recorded either


# ───────────────────────── execute_plan: SWING routing (Phase 13) ────────────


def test_swing_order_routes_through_options_hierarchy(tmp_db: Path) -> None:
    broker = SpyBroker()
    run = pe.execute_plan(
        broker, ExecutionPlan(orders=[_swing_order()]), now=_NOW,
        option_chain_fetch=_chain_fetch,
    )

    assert run.ok is True
    (execution,) = run.executions
    assert execution.status == "submitted"
    assert execution.vehicle == "option_full"
    assert execution.order_ref is not None
    assert broker.submissions == ["META-call-480"]   # the OCC symbol, not shares

    # The Phase 13 path recorded the option position for the exit watcher,
    # with the TP/SL levels recovered from the order's own fields:
    # stop = dollar_risk/qty = 15 -> sl 465; tp distance = 15 x (2/1.5) = 20.
    (pos,) = db.get_open_option_positions()
    assert pos.underlying == "META"
    assert pos.tp == 500.0
    assert pos.sl == 465.0


def test_swing_order_falls_back_to_shares_without_options(tmp_db: Path) -> None:
    broker = SpyBroker()
    run = pe.execute_plan(
        broker, ExecutionPlan(orders=[_swing_order()]), now=_NOW,
        option_chain_fetch=None, options_available=False,
    )
    (execution,) = run.executions
    assert execution.status == "submitted"
    assert execution.vehicle == "shares"
    assert broker.submissions == ["META"]            # fractional shares fallback
    assert db.get_open_option_positions() == []      # no contract to record


def test_swing_chain_fetch_error_falls_back_to_shares(tmp_db: Path) -> None:
    def boom(_underlying: str) -> OptionChainResult:
        raise RuntimeError("chain API down")

    broker = SpyBroker()
    run = pe.execute_plan(
        broker, ExecutionPlan(orders=[_swing_order()]), now=_NOW,
        option_chain_fetch=boom,
    )
    (execution,) = run.executions
    assert execution.status == "submitted"
    assert execution.vehicle == "shares"


# ───────────────────────── execute_plan: LONG_TERM / CRYPTO routing ──────────


def test_long_term_order_routes_through_entry_submission(tmp_db: Path) -> None:
    broker = SpyBroker()
    run = pe.execute_plan(
        broker, ExecutionPlan(orders=[_long_term_order()]), now=_NOW,
    )
    (execution,) = run.executions
    assert execution.status == "submitted"
    assert execution.vehicle == "shares"
    assert broker.submissions == ["AAPL"]

    (pos,) = db.get_open_long_term_positions()
    assert pos.ticker == "AAPL"
    assert pos.asset_class == "stock"
    assert pos.qty == 45.0
    assert pos.entry_price == 100.0


def test_crypto_pool_order_records_crypto_asset_class(tmp_db: Path) -> None:
    broker = SpyBroker()
    order = _long_term_order(
        ticker="BTC-USD", pool=config.POOL_CRYPTO, qty=0.05, entry=64_000.0,
    )
    run = pe.execute_plan(broker, ExecutionPlan(orders=[order]), now=_NOW)
    assert run.executions[0].status == "submitted"
    (pos,) = db.get_open_long_term_positions()
    assert pos.asset_class == "crypto"


# ───────────────────────── idempotency: never double-submit ──────────────────


def test_execute_twice_does_not_double_submit(tmp_db: Path) -> None:
    broker = SpyBroker()
    plan = ExecutionPlan(orders=[_swing_order(), _long_term_order()])

    first = pe.execute_plan(broker, plan, now=_NOW, option_chain_fetch=_chain_fetch)
    assert [e.status for e in first.executions] == ["submitted", "submitted"]
    submissions_after_first = list(broker.submissions)

    second = pe.execute_plan(
        broker, plan, now=_NOW + timedelta(minutes=5),
        option_chain_fetch=_chain_fetch,
    )
    assert [e.status for e in second.executions] == ["skipped", "skipped"]
    assert all(e.reason == pe.SKIP_ALREADY_EXECUTED for e in second.executions)
    assert broker.submissions == submissions_after_first   # no new broker calls
    assert len(db.get_open_option_positions()) == 1
    assert len(db.get_open_long_term_positions()) == 1


def test_rejected_attempt_may_be_retried_same_cycle(tmp_db: Path) -> None:
    rejecting = SpyBroker(reject_symbols=frozenset({"AAPL"}))
    plan = ExecutionPlan(orders=[_long_term_order()])
    first = pe.execute_plan(rejecting, plan, now=_NOW)
    assert first.executions[0].status == "rejected"

    accepting = SpyBroker()
    second = pe.execute_plan(accepting, plan, now=_NOW + timedelta(minutes=5))
    assert second.executions[0].status == "submitted"   # rejection never blocks


# ───────────────────────── fail-soft batch ───────────────────────────────────


def test_one_rejection_does_not_block_the_batch(tmp_db: Path) -> None:
    broker = SpyBroker(reject_symbols=frozenset({"AAPL"}))
    plan = ExecutionPlan(orders=[
        _long_term_order(ticker="AAPL", rank=1),
        _long_term_order(ticker="MSFT", rank=2),
    ])
    run = pe.execute_plan(broker, plan, now=_NOW)

    assert [e.status for e in run.executions] == ["rejected", "submitted"]
    assert run.executions[0].reason == "insufficient buying power"
    assert broker.submissions == ["AAPL", "MSFT"]    # the batch continued
    (pos,) = db.get_open_long_term_positions()       # only the accepted order
    assert pos.ticker == "MSFT"


def test_unknown_pool_is_error_never_submitted(tmp_db: Path) -> None:
    broker = SpyBroker()
    weird = PlannedOrder(
        rank=1, pool="MARGIN", tier="HIGH", ticker="XYZ", signal_type="x",
        side="buy", qty=1.0, entry=10.0, est_cost=10.0, dollar_risk=10.0,
        score=0.1,
    )
    run = pe.execute_plan(broker, ExecutionPlan(orders=[weird]), now=_NOW)
    assert run.executions[0].status == "error"
    assert "unknown pool" in run.executions[0].reason
    assert broker.submissions == []


# ───────────────────────── audit trail completeness ──────────────────────────


def test_every_outcome_is_recorded_in_plan_executions(tmp_db: Path) -> None:
    broker = SpyBroker(reject_symbols=frozenset({"AAPL"}))
    plan = ExecutionPlan(orders=[
        _swing_order(rank=1),                        # submitted (option)
        _long_term_order(ticker="AAPL", rank=2),     # rejected
        _long_term_order(ticker="MSFT", rank=3),     # submitted
    ])
    run = pe.execute_plan(broker, plan, now=_NOW, option_chain_fetch=_chain_fetch)

    rows = db.get_plan_executions()
    assert len(rows) == 3
    assert {r.status for r in rows} == {"submitted", "rejected"}
    assert all(r.plan_id == run.plan_id for r in rows)
    by_ticker = {r.ticker: r for r in rows}
    assert by_ticker["META"].order_ref is not None
    assert by_ticker["META"].vehicle == "option_full"
    assert by_ticker["AAPL"].status == "rejected"
    assert by_ticker["AAPL"].order_ref is None
    assert by_ticker["MSFT"].status == "submitted"

    # And the skip on a re-run is recorded too.
    pe.execute_plan(
        broker, ExecutionPlan(orders=[_swing_order()]),
        now=_NOW + timedelta(minutes=1), option_chain_fetch=_chain_fetch,
    )
    statuses = [r.status for r in db.get_plan_executions()]
    assert "skipped" in statuses


# ───────────────────────── hold-window plumbing (Phase 20) ───────────────────


def test_shares_fallback_persists_carried_hold_deadline(tmp_db: Path) -> None:
    """The live path no longer records deadline=None: the shares fallback
    inherits the ORIGINAL swing signal's settlement deadline."""
    deadline = _NOW + timedelta(days=2)
    broker = SpyBroker()
    run = pe.execute_plan(
        broker,
        ExecutionPlan(orders=[_swing_order(hold_deadline=deadline)]),
        now=_NOW,                       # no chain -> shares fallback
    )
    assert run.executions[0].status == "submitted"
    (pos,) = db.get_open_long_term_positions()
    assert pos.source == "swing_fallback"
    assert pos.deadline == deadline     # the REAL time stop, carried through


def test_watcher_hold_deadline_fires_on_carried_value(tmp_db: Path) -> None:
    """End to end: execute -> tracked with the carried deadline -> the Phase 19
    watcher time-stops it once that moment passes."""
    import pandas as pd

    deadline = _NOW + timedelta(days=2)
    pe.execute_plan(
        SpyBroker(),
        ExecutionPlan(orders=[_swing_order(hold_deadline=deadline)]),
        now=_NOW,
    )
    flat = pd.DataFrame({
        "Open": [480.0] * 5, "High": [480.0] * 5, "Low": [480.0] * 5,
        "Close": [480.0] * 5, "Volume": [1_000_000] * 5,   # between TP and SL
    })
    (action,) = long_term.watch_long_term_positions(
        SpyBroker(), price_fetch=lambda _t: flat,
        now=deadline + timedelta(hours=1),
    )
    assert action.action == "close" and action.reason == "hold_deadline"


def test_shares_fallback_missing_deadline_fails_soft(
    tmp_db: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """A SWING order without a carried window degrades to today's behavior
    (no time stop), logged — the fire path never raises."""
    run = pe.execute_plan(
        SpyBroker(),
        ExecutionPlan(orders=[_swing_order(hold_deadline=None)]),
        now=_NOW,
    )
    assert run.executions[0].status == "submitted"    # fired anyway
    (pos,) = db.get_open_long_term_positions()
    assert pos.deadline is None
    assert "no hold window carried" in capsys.readouterr().err


def test_option_position_deadline_untouched_by_carried_window(
    tmp_db: Path,
) -> None:
    """Phase 13's domain: even with a carried window on the order, an OPTION
    position's deadline stays exactly as before (None from this path)."""
    run = pe.execute_plan(
        SpyBroker(),
        ExecutionPlan(orders=[_swing_order(hold_deadline=_NOW + timedelta(days=2))]),
        now=_NOW, option_chain_fetch=_chain_fetch,     # full option fits
    )
    assert run.executions[0].vehicle == "option_full"
    (opt,) = db.get_open_option_positions()
    assert opt.deadline is None
    assert db.get_open_long_term_positions() == []


def test_fired_signal_window_reaches_shares_fallback_deadline(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Phase 21 end to end: a signal fired through scanner.log_signal carries
    its now-persisted estimate all the way to the shares-fallback position's
    deadline — fire → candidate → plan → executed time stop, no defaults."""
    from trading_bot import allocation, candidate_source, regime, scanner, vix
    from trading_bot.broker.base import AccountInfo

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
    # Separate PRE-EXISTING gap (out of Phase 21 scope): log_signal doesn't
    # persist atr either, and Phase 7 sizing needs it. Supply it directly so
    # this test exercises the hold-window plumbing, not the atr gap.
    conn = db.get_connection()
    try:
        conn.execute("UPDATE signals SET atr = ? WHERE id = ?", (8.0, sid))
        conn.commit()
    finally:
        conn.close()
    signal = db.get_signal_by_id(sid)
    assert signal is not None

    candidates = candidate_source.live_candidates(now=signal.timestamp)
    result = allocation.build_plan(
        candidates, AccountInfo(ok=True, equity=100_000.0, cash=100_000.0),
    )
    run = pe.execute_plan(
        SpyBroker(), result.plan, now=signal.timestamp,   # no chain -> shares
    )
    assert run.executions[0].status == "submitted"
    (pos,) = db.get_open_long_term_positions()
    assert pos.source == "swing_fallback"
    assert pos.deadline == signal.timestamp + timedelta(days=5)   # not +30


def test_long_term_entry_never_acquires_a_deadline(tmp_db: Path) -> None:
    """LOCKED DESIGN regression: even an adversarial LONG_TERM order carrying
    a hold_deadline produces a genuine long-term position with deadline=None —
    the entry path never reads the field."""
    run = pe.execute_plan(
        SpyBroker(),
        ExecutionPlan(orders=[
            _long_term_order(hold_deadline=_NOW + timedelta(days=2)),
        ]),
        now=_NOW,
    )
    assert run.executions[0].status == "submitted"
    (pos,) = db.get_open_long_term_positions()
    assert pos.source == "long_term"
    assert pos.deadline is None            # unconditionally, by design
    assert (pos.tp, pos.sl) == (None, None)


# ───────────────────────── exit-level recovery ───────────────────────────────


def test_swing_exit_levels_bullish_and_bearish() -> None:
    buy = _swing_order()                      # entry 480, stop distance 15
    assert pe._swing_exit_levels(buy) == (500.0, 465.0)
    sell = _swing_order(side="sell")
    assert pe._swing_exit_levels(sell) == (460.0, 495.0)


def test_swing_exit_levels_unsizeable_is_none() -> None:
    order = _swing_order(qty=0.0, dollar_risk=0.0)
    assert pe._swing_exit_levels(order) == (None, None)


# ───────────────────────── Phase 14 entry submission path ────────────────────


def test_submit_long_term_entry_records_open_position(tmp_db: Path) -> None:
    broker = SpyBroker()
    order, position_id = long_term.submit_long_term_entry(
        broker, ticker="GOOGL", asset_class="stock", qty=2.5,
        entry_price=175.0, now=_NOW,
    )
    assert order.ok is True
    assert position_id is not None
    (pos,) = db.get_open_long_term_positions()
    assert pos.id == position_id
    assert (pos.ticker, pos.qty, pos.entry_price) == ("GOOGL", 2.5, 175.0)


def test_submit_long_term_entry_rejected_records_nothing(tmp_db: Path) -> None:
    broker = SpyBroker(reject_symbols=frozenset({"GOOGL"}))
    order, position_id = long_term.submit_long_term_entry(
        broker, ticker="GOOGL", asset_class="stock", qty=2.5,
        entry_price=175.0, now=_NOW,
    )
    assert order.ok is False
    assert position_id is None
    assert db.get_open_long_term_positions() == []


# ───────────────────────── CLI: --confirm is mandatory ───────────────────────


def test_cli_execute_refuses_without_confirm(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_bot import __main__ as m
    broker = SpyBroker()
    monkeypatch.setattr(m.broker, "AlpacaBroker", lambda: broker)
    m.cmd_allocate_execute(confirm=False)
    out = capsys.readouterr().out
    assert "REFUSED: --confirm is required" in out
    assert "ALLOCATION PLAN" in out            # the plan is shown for review
    assert broker.submissions == []            # NOTHING was submitted
    assert db.get_plan_executions() == []


def test_cli_execute_with_confirm_submits(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_bot import __main__ as m

    class FakeChainClient:
        def get_option_chain(self, _underlying: str) -> OptionChainResult:
            return OptionChainResult(ok=False, reason="options not enabled")

    broker = SpyBroker()
    monkeypatch.setattr(m.broker, "AlpacaBroker", lambda: broker)
    monkeypatch.setattr(m.broker, "AlpacaOptionsClient", FakeChainClient)
    # Phase 17: the sample fixture stays available behind --sample.
    m.cmd_allocate_execute(confirm=True, sample=True)
    out = capsys.readouterr().out
    assert "PLAN EXECUTION" in out
    assert "Submitted" in out
    assert broker.submissions != []            # paper orders actually went out
    rows = db.get_plan_executions()
    assert rows and all(r.status == "submitted" for r in rows)
    # The data-only crypto swing signals were filtered by the plan, never routed.
    assert all(r.pool != config.POOL_CRYPTO for r in rows)


def test_cli_execute_with_confirm_refuses_when_unauthorized(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_bot import __main__ as m
    risk_of_ruin.revoke(config.ENTRY_CAPABILITY, "tier1: drawdown 21.0%")
    broker = SpyBroker()
    monkeypatch.setattr(m.broker, "AlpacaBroker", lambda: broker)
    m.cmd_allocate_execute(confirm=True)
    out = capsys.readouterr().out
    assert "EXECUTION REFUSED" in out
    assert broker.submissions == []
    assert db.get_plan_executions() == []
