"""Tests for options execution — selection, cost, hierarchy, exit watcher.

Pure logic and mocked network only. The multiplier cost, the full→undersized→
shares hierarchy, and the manual exit watcher are the core; strike selection is
checked against hand-built contract fixtures.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from trading_bot import config, db
from trading_bot import options_execution as oe
from trading_bot.broker.base import (
    ORDER_TYPE_LIMIT,
    STATUS_CANCELED,
    STATUS_FILLED,
    TIF_DAY,
    OrderResult,
)
from trading_bot.broker.fake import FakeBroker
from trading_bot.broker.options import OptionContract
from trading_bot.models import OptionPosition, PendingOrder

_REF = date(2026, 1, 1)
_OPENED = datetime(2026, 1, 1, tzinfo=UTC)


def _contract(
    delta: float, *, option_type: str = "call", oi: int | None = 500,
    bid: float | None = 4.9, ask: float | None = 5.1, dte: int = 30,
    strike: float = 150.0,
) -> OptionContract:
    expiry = (_REF + timedelta(days=dte)).isoformat()
    mid = (bid + ask) / 2.0 if bid is not None and ask is not None else None
    return OptionContract(
        symbol=f"AAPL-{option_type}-{strike}-{dte}", underlying="AAPL",
        option_type=option_type, strike=strike, expiry=expiry, delta=delta,
        theta=-0.05, vega=0.1, gamma=0.01, open_interest=oi,
        bid=bid, ask=ask, mid=mid,
    )


# ───────────────────────── strike selection (four fixture cases) ─────────────


def test_select_contract_qualifying() -> None:
    got = oe.select_contract("call", [_contract(0.70)], ref_date=_REF)
    assert got is not None
    assert got.delta == 0.70


def test_select_contract_none_in_band() -> None:
    # delta 0.40 is below the 0.65-0.75 target band.
    assert oe.select_contract("call", [_contract(0.40)], ref_date=_REF) is None


def test_select_contract_in_band_but_illiquid() -> None:
    # In-band delta, but open interest below the floor -> excluded.
    assert oe.select_contract("call", [_contract(0.70, oi=10)], ref_date=_REF) is None
    # Wide bid-ask spread -> excluded.
    wide = _contract(0.70, bid=4.0, ask=6.0)   # spread 40% of mid
    assert oe.select_contract("call", [wide], ref_date=_REF) is None


def test_select_contract_in_band_but_insufficient_dte() -> None:
    # In-band + liquid, but only 5 DTE (< MIN_DTE 14) -> excluded.
    assert oe.select_contract("call", [_contract(0.70, dte=5)], ref_date=_REF) is None


def test_select_contract_rejects_leaps_outside_swing_window() -> None:
    assert oe.select_contract(
        "call", [_contract(0.70, dte=config.MAX_DTE + 1)], ref_date=_REF,
    ) is None


def test_select_contract_prefers_target_dte_when_delta_ties() -> None:
    got = oe.select_contract(
        "call",
        [_contract(0.70, dte=55), _contract(0.70, dte=config.TARGET_DTE)],
        ref_date=_REF,
    )
    assert got is not None
    assert got.expiry == (_REF + timedelta(days=config.TARGET_DTE)).isoformat()


# ───────────────────────── selection orientation + preference ────────────────


def test_select_contract_put_orientation() -> None:
    # A put's delta is negative; |delta| in band qualifies. Side 'short' -> put.
    put = _contract(-0.70, option_type="put")
    call = _contract(0.70, option_type="call")
    got = oe.select_contract("short", [call, put], ref_date=_REF)
    assert got is not None and got.option_type == "put"


def test_select_contract_prefers_delta_closest_to_centre() -> None:
    # Band centre is 0.70; among 0.66 and 0.70, prefer 0.70.
    got = oe.select_contract(
        "call", [_contract(0.66, strike=155.0), _contract(0.70, strike=150.0)],
        ref_date=_REF,
    )
    assert got is not None and got.delta == 0.70


def test_select_contract_undersized_band_widens_down() -> None:
    # delta 0.55 fails the full band (0.65-0.75) but passes the undersized floor.
    c = _contract(0.55)
    assert oe.select_contract("call", [c], ref_date=_REF) is None
    widened = oe.select_contract(
        "call", [c], ref_date=_REF, delta_low=config.UNDERSIZED_DELTA_FLOOR,
    )
    assert widened is not None and widened.delta == 0.55


# ───────────────────────── cost model (multiplier correctness) ───────────────


def test_cost_for_contract_applies_multiplier() -> None:
    c = _contract(0.70, bid=4.9, ask=5.1)   # mid == 5.0
    # Entries submit at the ask, so sizing must use that same executable price.
    assert oe.cost_for_contract(c, 2) == 1020.0
    assert oe.cost_for_contract(c, 1) == 510.0


def test_cost_for_contract_none_without_premium() -> None:
    no_price = OptionContract(
        symbol="X", underlying="AAPL", option_type="call", strike=150.0,
        expiry="2026-02-01",
    )
    assert oe.cost_for_contract(no_price, 1) is None


# ───────────────────────── execution hierarchy ──────────────────────────────


def test_choose_execution_full_option() -> None:
    # ask 5.1 -> $510/contract; $600 capital fits one full-band contract.
    contracts = [_contract(0.70)]
    dec = oe.choose_execution(
        "call", 600.0, "AAPL", 100.0, contracts, ref_date=_REF,
    )
    assert dec.vehicle == oe.VEHICLE_OPTION_FULL
    assert dec.qty == 1.0 and dec.est_cost == 510.0
    assert dec.side == "buy"


def test_choose_execution_undersized_when_no_full_fit() -> None:
    # Only a 0.55-delta contract exists (out of the full band). It fits capital
    # in the widened undersized band.
    contracts = [_contract(0.55)]
    dec = oe.choose_execution(
        "call", 600.0, "AAPL", 100.0, contracts, ref_date=_REF,
    )
    assert dec.vehicle == oe.VEHICLE_OPTION_UNDERSIZED
    assert dec.qty == 1.0


def test_choose_execution_falls_back_to_shares_when_no_contract_fits() -> None:
    # A full-band contract exists but costs $500/contract; only $300 allocated ->
    # no whole contract fits (never undersized to a fraction) -> shares.
    contracts = [_contract(0.70)]
    dec = oe.choose_execution(
        "call", 300.0, "AAPL", 100.0, contracts, ref_date=_REF,
    )
    assert dec.vehicle == oe.VEHICLE_SHARES
    assert dec.qty == 3.0            # $300 / $100 underlying = 3 fractional shares
    assert dec.symbol == "AAPL"


def test_option_risk_budget_is_separate_from_shares_position_value() -> None:
    dec = oe.choose_execution(
        "call",
        300.0,  # maximum premium loss
        "AAPL",
        100.0,
        [_contract(0.70)],  # one contract costs $500, so it cannot fit
        ref_date=_REF,
        shares_capital=2_000.0,
    )
    assert dec.vehicle == oe.VEHICLE_SHARES
    assert dec.est_cost == 2_000.0
    assert dec.qty == 20.0


def test_option_market_hours_guard() -> None:
    assert oe.is_option_market_open(datetime(2026, 7, 10, 15, 0, tzinfo=UTC))
    assert not oe.is_option_market_open(datetime(2026, 7, 11, 15, 0, tzinfo=UTC))
    assert not oe.is_option_market_open(datetime(2026, 7, 10, 22, 0, tzinfo=UTC))


def test_choose_execution_options_unavailable_routes_to_shares() -> None:
    contracts = [_contract(0.70)]
    dec = oe.choose_execution(
        "call", 600.0, "AAPL", 100.0, contracts, ref_date=_REF,
        options_available=False,
    )
    assert dec.vehicle == oe.VEHICLE_SHARES


def test_choose_execution_empty_chain_routes_to_shares() -> None:
    dec = oe.choose_execution("call", 600.0, "AAPL", 100.0, [], ref_date=_REF)
    assert dec.vehicle == oe.VEHICLE_SHARES


def test_choose_execution_short_shares_fallback_sells() -> None:
    dec = oe.choose_execution("short", 600.0, "AAPL", 100.0, [], ref_date=_REF)
    assert dec.vehicle == oe.VEHICLE_SHARES and dec.side == "sell"


def test_choose_execution_shares_unsizeable_without_price() -> None:
    dec = oe.choose_execution("call", 600.0, "AAPL", None, [], ref_date=_REF)
    assert dec.vehicle == oe.VEHICLE_NONE and dec.qty == 0.0


# ───────────────────────── order submission (Phase 11 path reuse) ────────────


def _decision_full() -> oe.ExecutionDecision:
    return oe.choose_execution(
        "call", 600.0, "AAPL", 100.0, [_contract(0.70)], ref_date=_REF,
    )


def test_submit_execution_order_is_limit_with_no_bracket_attrs(
    monkeypatch: pytest.MonkeyPatch, tmp_db: Path,
) -> None:
    b = FakeBroker()
    calls: list[tuple[str, float, str, dict[str, object]]] = []
    original = b.submit_order

    def spy(symbol: str, qty: float, side: str, **kw: object) -> object:
        calls.append((symbol, qty, side, kw))
        return original(symbol, qty, side, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(b, "submit_order", spy)
    order = oe.submit_execution_order(b, _decision_full())
    assert order.ok is True
    symbol, _qty, side, kw = calls[0]
    assert side == "buy" and kw["order_type"] == "limit"
    # NO bracket / OTO attributes are EVER sent on an options order.
    for forbidden in ("order_class", "take_profit", "stop_loss", "bracket", "otoco"):
        assert forbidden not in kw


def test_submit_execution_order_structured_rejection(tmp_db: Path) -> None:
    b = FakeBroker(reject_reason="options not permitted at this level")
    order = oe.submit_execution_order(b, _decision_full())
    assert order.ok is False
    assert order.status == "rejected"
    assert "options not permitted" in order.reason


# ───────────────────────── persistence (option_positions, schema v17) ────────


def test_execute_decision_records_option_position(tmp_db: Path) -> None:
    order, pid = oe.execute_decision(
        FakeBroker(auto_fill=True), _decision_full(), opened_at=_OPENED, signal_id=7,
        tp=110.0, sl=95.0, deadline=datetime(2026, 1, 20, tzinfo=UTC),
    )
    assert order.ok is True and pid is not None
    positions = db.get_open_option_positions()
    assert len(positions) == 1
    p = positions[0]
    assert p.symbol == _contract(0.70).symbol
    assert p.vehicle == "option_full"
    assert p.delta_entry == 0.70
    assert p.premium_entry == 5.1            # broker-reported limit fill
    assert p.multiplier == 100               # the multiplier is persisted
    assert p.tp == 110.0 and p.sl == 95.0 and p.outcome == "open"
    assert p.signal_id == 7
    pending = db.get_pending_order(order.order_id)
    assert pending is not None
    assert pending.signal_id == 7


def test_execute_decision_shares_records_swing_fallback_not_option(
    tmp_db: Path,
) -> None:
    # $300 capital / $500 per contract -> shares fallback -> no option row;
    # Phase 19: the position is tracked in the long-term lifecycle book with
    # source='swing_fallback' instead of vanishing.
    dec = oe.choose_execution(
        "call", 300.0, "AAPL", 100.0, [_contract(0.70)], ref_date=_REF,
    )
    assert dec.vehicle == oe.VEHICLE_SHARES
    order, pid = oe.execute_decision(
        FakeBroker(auto_fill=True), dec, opened_at=_OPENED
    )
    assert order.ok is True and pid is not None
    assert db.get_open_option_positions() == []
    (pos,) = db.get_open_long_term_positions()
    assert pos.id == pid
    assert pos.source == "swing_fallback"


def test_shares_fallback_persists_original_swing_exit_data(tmp_db: Path) -> None:
    deadline = datetime(2026, 1, 3, tzinfo=UTC)
    dec = oe.choose_execution("call", 300.0, "AAPL", 100.0, [], ref_date=_REF)
    order, pid = oe.execute_decision(
        FakeBroker(auto_fill=True), dec, opened_at=_OPENED, tp=110.0, sl=95.0,
        deadline=deadline,
    )
    assert order.ok is True and pid is not None
    (pos,) = db.get_open_long_term_positions()
    assert pos.ticker == "AAPL"
    assert pos.direction == "long"
    assert pos.qty == 3.0                    # $300 / $100
    assert pos.entry_price == 100.0          # est_cost / qty
    assert (pos.tp, pos.sl) == (110.0, 95.0)  # ORIGINAL swing levels, persisted
    assert pos.deadline == deadline


def test_short_shares_fallback_records_short_direction(tmp_db: Path) -> None:
    dec = oe.choose_execution("short", 300.0, "AAPL", 100.0, [], ref_date=_REF)
    assert dec.side == "sell"
    order, _pid = oe.execute_decision(
        FakeBroker(auto_fill=True), dec, opened_at=_OPENED, tp=90.0, sl=105.0,
    )
    assert order.ok is True
    (pos,) = db.get_open_long_term_positions()
    assert pos.direction == "short"          # its close must BUY


def test_rejected_shares_fallback_records_nothing(tmp_db: Path) -> None:
    dec = oe.choose_execution("call", 300.0, "AAPL", 100.0, [], ref_date=_REF)
    order, pid = oe.execute_decision(
        FakeBroker(reject_reason="market closed"), dec, opened_at=_OPENED,
    )
    assert order.ok is False and pid is None
    assert db.get_open_long_term_positions() == []


def test_update_option_position_close_round_trip(tmp_db: Path) -> None:
    pid = db.insert_option_position(OptionPosition(
        symbol="X260116C00150000", underlying="X", option_type="call",
        strike=150.0, expiry="2026-01-16", contracts=1.0, opened_at=_OPENED,
        outcome="open",
    ))
    assert len(db.get_open_option_positions()) == 1
    db.update_option_position(
        pid, outcome="win", closed_at=datetime(2026, 1, 10, tzinfo=UTC),
        exit_price=7.0, pnl_dollars=200.0,
    )
    assert db.get_open_option_positions() == []          # no longer open
    closed = db.get_option_positions()[0]
    assert closed.outcome == "win" and closed.pnl_dollars == 200.0


def test_update_option_position_rejects_unknown_field(tmp_db: Path) -> None:
    pid = db.insert_option_position(OptionPosition(
        symbol="X260116C00150000", underlying="X", option_type="call",
        strike=150.0, expiry="2026-01-16", contracts=1.0, opened_at=_OPENED,
    ))
    with pytest.raises(ValueError, match="Unknown option_position field"):
        db.update_option_position(pid, strike=999.0)


# ───────────────────────── exit decision (pure) ─────────────────────────────

_LATE = datetime(2026, 2, 1, tzinfo=UTC)
_DEADLINE = datetime(2026, 1, 20, tzinfo=UTC)


def test_evaluate_exit_call_take_profit_and_stop() -> None:
    tp = oe.evaluate_option_exit("call", 110.0, 110.0, 95.0, _OPENED, None)
    assert tp.action == "close" and tp.reason == "take_profit"
    sl = oe.evaluate_option_exit("call", 95.0, 110.0, 95.0, _OPENED, None)
    assert sl.action == "close" and sl.reason == "stop_loss"


def test_evaluate_exit_put_orientation() -> None:
    # A put is bearish: take-profit BELOW, stop-loss ABOVE.
    tp = oe.evaluate_option_exit("put", 90.0, 90.0, 105.0, _OPENED, None)
    assert tp.reason == "take_profit"
    sl = oe.evaluate_option_exit("put", 105.0, 90.0, 105.0, _OPENED, None)
    assert sl.reason == "stop_loss"


def test_evaluate_exit_hold_deadline() -> None:
    d = oe.evaluate_option_exit("call", 100.0, 110.0, 95.0, _LATE, _DEADLINE)
    assert d.action == "close" and d.reason == "hold_deadline"


def test_evaluate_exit_not_premature() -> None:
    # Price between SL and TP, before the deadline -> hold.
    d = oe.evaluate_option_exit("call", 100.0, 110.0, 95.0, _OPENED, _LATE)
    assert d.action == "hold" and d.reason == "holding"


def test_evaluate_exit_missing_price_holds() -> None:
    d = oe.evaluate_option_exit("call", None, 110.0, 95.0, _OPENED, None)
    assert d.action == "hold"


# ───────────────────────── exit watcher (driver) ────────────────────────────


def _open_position(symbol: str, underlying: str) -> int:
    position_id = db.insert_option_position(OptionPosition(
        symbol=symbol, underlying=underlying, option_type="call", strike=150.0,
        expiry="2026-01-16", contracts=2.0, opened_at=_OPENED, premium_entry=5.0,
        tp=110.0, sl=95.0, deadline=_DEADLINE, multiplier=100, outcome="open",
    ))
    db.insert_pending_order(PendingOrder(
        client_order_id=f"entry-option-{position_id}",
        broker_order_id=f"broker-entry-option-{position_id}",
        ticker=underlying,
        broker_symbol=symbol,
        asset_class="stock",
        vehicle="option_full",
        target_position_kind="option",
        side="buy",
        requested_qty=2.0,
        requested_limit_price=5.0,
        submitted_at=_OPENED,
        intent_payload_json=json.dumps({
            "intent_kind": "option",
            "option_type": "call",
            "strike": 150.0,
            "expiry": "2026-01-16",
            "multiplier": 100,
            "delta_entry": None,
            "theta": None,
            "vega": None,
            "gamma": None,
            "tp": 110.0,
            "sl": 95.0,
            "deadline": _DEADLINE.isoformat(),
        }, sort_keys=True),
        lifecycle_status=STATUS_FILLED,
        broker_status=STATUS_FILLED,
        filled_qty=2.0,
        filled_avg_price=5.0,
        last_fill_at=_OPENED,
        last_fill_time_source="broker",
        last_refreshed_at=_OPENED,
        terminal_reason=STATUS_FILLED,
        terminal_at=_OPENED,
        position_kind="option",
        position_id=position_id,
    ))
    return position_id


def test_watcher_closes_on_take_profit_with_sell_no_bracket(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _open_position("AAPL260116C00150000", "AAPL")
    b = FakeBroker(auto_fill=True)
    calls: list[tuple[str, float, str, dict[str, object]]] = []
    original = b.submit_order

    def spy(symbol: str, qty: float, side: str, **kw: object) -> object:
        calls.append((symbol, qty, side, kw))
        return original(symbol, qty, side, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(b, "submit_order", spy)
    actions = oe.watch_open_option_positions(
        b, underlying_price_fetch=lambda _u: 111.0,
        option_price_fetch=lambda _s: 8.0, now=_OPENED,
    )
    assert len(actions) == 1
    assert actions[0].action == "close" and actions[0].reason == "take_profit"
    # A closing SELL LIMIT, no bracket attributes.
    _sym, _qty, side, kw = calls[0]
    assert side == "sell" and kw["order_type"] == "limit"
    for forbidden in ("order_class", "take_profit", "stop_loss", "bracket"):
        assert forbidden not in kw
    # Contract-aware PnL: (8 - 5) * 100 * 2 == 600.
    assert actions[0].pnl_dollars == 600.0
    assert db.get_open_option_positions() == []
    closed = db.get_option_positions()[0]
    assert closed.outcome == "win" and closed.pnl_dollars == 600.0


def test_watcher_holds_when_no_exit(tmp_db: Path) -> None:
    _open_position("AAPL260116C00150000", "AAPL")
    actions = oe.watch_open_option_positions(
        FakeBroker(), underlying_price_fetch=lambda _u: 100.0,
        option_price_fetch=lambda _s: 5.0, now=_OPENED,
    )
    assert actions[0].action == "hold"
    assert len(db.get_open_option_positions()) == 1   # still open, no order sent


def test_watcher_one_error_does_not_block_others(tmp_db: Path) -> None:
    _open_position("GOOD260116C00150000", "GOOD")
    _open_position("BAD260116C00150000", "BAD")

    def fetch(underlying: str) -> float | None:
        if underlying == "BAD":
            raise RuntimeError("price feed down")
        return 111.0                       # GOOD hits take-profit

    actions = oe.watch_open_option_positions(
        FakeBroker(auto_fill=True), underlying_price_fetch=fetch,
        option_price_fetch=lambda _s: 8.0, now=_OPENED,
    )
    by_symbol = {a.position.underlying: a.action for a in actions}
    assert by_symbol["GOOD"] == "close"    # processed despite the other's error
    assert by_symbol["BAD"] == "error"     # errored, but did not block GOOD


def test_watcher_accepted_unfilled_stays_open_and_restart_does_not_resubmit(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    position_id = _open_position("AAPL260116C00150000", "AAPL")
    results: list[bool] = []
    monkeypatch.setattr(
        oe.risk_of_ruin,
        "record_broker_result",
        lambda ok, status=None: results.append(ok),
    )

    first = oe.watch_open_option_positions(
        FakeBroker(auto_fill=False),
        underlying_price_fetch=lambda _u: 111.0,
        option_price_fetch=lambda _s: 8.0,
        now=_OPENED,
    )
    assert first[0].action == "close"
    position = db.get_option_position(position_id)
    assert position is not None
    assert position.outcome == "open"
    assert position.closed_at is None
    assert position.exit_price is None
    assert position.exit_reason is None
    assert position.pnl_dollars is None

    restart_calls: list[object] = []
    restarted_broker = FakeBroker(auto_fill=True)
    monkeypatch.setattr(
        restarted_broker,
        "submit_order",
        lambda *args, **kwargs: restart_calls.append((args, kwargs)),
    )
    second = oe.watch_open_option_positions(
        restarted_broker,
        underlying_price_fetch=lambda _u: 111.0,
        option_price_fetch=lambda _s: 9.0,
        now=_OPENED + timedelta(minutes=5),
    )
    assert second[0].action == "hold"
    assert restart_calls == []
    exits = db.get_pending_exit_orders_for_position("option", position_id)
    assert len(exits) == 1
    assert exits[0].filled_qty == 0.0
    assert exits[0].fees_dollars is None
    assert results == [True]


def test_watcher_terminal_partial_retries_only_remaining_and_uses_actual_vwap(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    position_id = _open_position("AAPL260116C00150000", "AAPL")
    quantities: list[float] = []
    snapshots = [
        (STATUS_CANCELED, 1.0, 6.0, "2026-01-02T15:00:00+00:00"),
        (STATUS_FILLED, 1.0, 8.0, "2026-01-03T16:00:00+00:00"),
    ]
    broker = FakeBroker()

    def submit(
        symbol: str,
        qty: float,
        side: str,
        *,
        order_type: str = ORDER_TYPE_LIMIT,
        limit_price: float | None = None,
        time_in_force: str = TIF_DAY,
        client_order_id: str | None = None,
    ) -> OrderResult:
        status, filled_qty, fill_price, filled_at = snapshots.pop(0)
        quantities.append(qty)
        return OrderResult(
            ok=True,
            status=status,
            order_id=f"exit-{len(quantities)}",
            client_order_id=client_order_id,
            symbol=symbol,
            qty=qty,
            filled_qty=filled_qty,
            filled_avg_price=fill_price,
            side=side,
            order_type=order_type,
            time_in_force=time_in_force,
            limit_price=limit_price,
            submitted_at=_OPENED.isoformat(),
            filled_at=filled_at,
            updated_at=filled_at,
            raw_status=status,
        )

    monkeypatch.setattr(broker, "submit_order", submit)
    first = oe.watch_open_option_positions(
        broker,
        underlying_price_fetch=lambda _u: 111.0,
        option_price_fetch=lambda _s: 99.0,
        now=_OPENED,
    )
    assert first[0].action == "close"
    partial = db.get_option_position(position_id)
    assert partial is not None and partial.outcome == "open"

    second = oe.watch_open_option_positions(
        broker,
        underlying_price_fetch=lambda _u: 111.0,
        option_price_fetch=lambda _s: 101.0,
        now=_OPENED + timedelta(days=1),
    )
    assert second[0].action == "close"
    assert quantities == [2.0, 1.0]
    closed = db.get_option_position(position_id)
    assert closed is not None
    assert closed.outcome == "win"
    assert closed.exit_reason == "take_profit"
    assert closed.exit_price == 7.0
    assert closed.closed_at == datetime(2026, 1, 3, 16, 0, tzinfo=UTC)
    assert closed.pnl_dollars == 400.0


def test_watcher_trigger_reason_does_not_override_actual_economic_outcome(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    position_id = _open_position("AAPL260116C00150000", "AAPL")
    broker = FakeBroker()
    fill_at = "2026-01-04T17:30:00+00:00"

    def submit(
        symbol: str,
        qty: float,
        side: str,
        **kwargs: object,
    ) -> OrderResult:
        return OrderResult(
            ok=True,
            status=STATUS_FILLED,
            order_id="actual-loss",
            client_order_id=str(kwargs["client_order_id"]),
            symbol=symbol,
            qty=qty,
            filled_qty=qty,
            filled_avg_price=4.0,
            side=side,
            order_type=ORDER_TYPE_LIMIT,
            time_in_force=TIF_DAY,
            limit_price=float(kwargs["limit_price"]),
            submitted_at=_OPENED.isoformat(),
            filled_at=fill_at,
            updated_at=fill_at,
            raw_status=STATUS_FILLED,
        )

    monkeypatch.setattr(broker, "submit_order", submit)
    oe.watch_open_option_positions(
        broker,
        underlying_price_fetch=lambda _u: 111.0,
        option_price_fetch=lambda _s: 8.0,
        now=_OPENED,
    )

    closed = db.get_option_position(position_id)
    assert closed is not None
    assert closed.exit_reason == "take_profit"
    assert closed.outcome == "loss"
    assert closed.exit_price == 4.0
    assert closed.closed_at == datetime(2026, 1, 4, 17, 30, tzinfo=UTC)
    assert closed.pnl_dollars == -200.0


def test_watcher_refuses_legacy_row_but_continues_linked_positions(
    tmp_db: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    legacy_id = db.insert_option_position(OptionPosition(
        symbol="LEGACY260116C00150000",
        underlying="LEGACY",
        option_type="call",
        strike=150.0,
        expiry="2026-01-16",
        contracts=2.0,
        opened_at=_OPENED,
        premium_entry=5.0,
        tp=110.0,
        sl=95.0,
        outcome="open",
    ))
    linked_id = _open_position("GOOD260116C00150000", "GOOD")

    actions = oe.watch_open_option_positions(
        FakeBroker(auto_fill=True),
        underlying_price_fetch=lambda _u: 111.0,
        option_price_fetch=lambda _s: 8.0,
        now=_OPENED,
    )
    by_underlying = {action.position.underlying: action for action in actions}
    assert by_underlying["LEGACY"].action == "error"
    assert by_underlying["GOOD"].action == "close"
    legacy = db.get_option_position(legacy_id)
    linked = db.get_option_position(linked_id)
    assert legacy is not None and legacy.outcome == "open"
    assert linked is not None and linked.outcome == "win"
    assert "lifecycle exit refused" in capsys.readouterr().err


# ─────────────── unknown exit price is refused, never priced at entry ────────


def test_watcher_refuses_close_when_option_price_unknown(tmp_db: Path) -> None:
    """A triggered exit with no available option price must NOT be submitted at
    the entry premium. A close priced at entry rests unfillable for exactly the
    losing positions an exit exists to close, while the book records a close as
    sent. The position stays open and the error is surfaced instead."""
    _open_position("AAPL260116C00150000", "AAPL")
    b = FakeBroker(auto_fill=True)
    actions = oe.watch_open_option_positions(
        b,
        underlying_price_fetch=lambda _u: 111.0,   # take-profit HAS triggered
        option_price_fetch=lambda _s: None,        # ...but the price is unknown
        now=_OPENED,
    )
    assert len(actions) == 1
    assert actions[0].action == "error"
    assert "no current option price" in actions[0].reason
    # Nothing was submitted and the position is still open for the next pass.
    assert len(db.get_open_option_positions()) == 1
    assert db.get_pending_exit_orders_for_position("option", 1) == []


def test_watcher_still_closes_when_price_is_known(tmp_db: Path) -> None:
    """Positive control for the guard above — a known price still closes."""
    _open_position("AAPL260116C00150000", "AAPL")
    actions = oe.watch_open_option_positions(
        FakeBroker(auto_fill=True),
        underlying_price_fetch=lambda _u: 111.0,
        option_price_fetch=lambda _s: 8.0,
        now=_OPENED,
    )
    assert actions[0].action == "close"
    assert db.get_open_option_positions() == []
