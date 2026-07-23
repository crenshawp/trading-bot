"""Tests for options execution — selection, cost, hierarchy, exit watcher.

Pure logic and mocked network only. The multiplier cost, the full→undersized→
shares hierarchy, and the manual exit watcher are the core; strike selection is
checked against hand-built contract fixtures.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from trading_bot import config, db
from trading_bot import options_execution as oe
from trading_bot.broker.fake import FakeBroker
from trading_bot.broker.options import OptionContract
from trading_bot.models import OptionPosition

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
    # 5.0 premium * 100 multiplier * 2 contracts == 1000. A 100x bug would give
    # 10 (no multiplier) or 100000 — this pins the multiplier in place.
    assert oe.cost_for_contract(c, 2) == 1000.0
    assert oe.cost_for_contract(c, 1) == 500.0


def test_cost_for_contract_none_without_premium() -> None:
    no_price = OptionContract(
        symbol="X", underlying="AAPL", option_type="call", strike=150.0,
        expiry="2026-02-01",
    )
    assert oe.cost_for_contract(no_price, 1) is None


# ───────────────────────── execution hierarchy ──────────────────────────────


def test_choose_execution_full_option() -> None:
    # mid 5.0 -> $500/contract; $600 capital fits one full-band contract.
    contracts = [_contract(0.70)]
    dec = oe.choose_execution(
        "call", 600.0, "AAPL", 100.0, contracts, ref_date=_REF,
    )
    assert dec.vehicle == oe.VEHICLE_OPTION_FULL
    assert dec.qty == 1.0 and dec.est_cost == 500.0
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
    return db.insert_option_position(OptionPosition(
        symbol=symbol, underlying=underlying, option_type="call", strike=150.0,
        expiry="2026-01-16", contracts=2.0, opened_at=_OPENED, premium_entry=5.0,
        tp=110.0, sl=95.0, deadline=_DEADLINE, multiplier=100, outcome="open",
    ))


def test_watcher_closes_on_take_profit_with_sell_no_bracket(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _open_position("AAPL260116C00150000", "AAPL")
    b = FakeBroker()
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
        FakeBroker(), underlying_price_fetch=fetch,
        option_price_fetch=lambda _s: 8.0, now=_OPENED,
    )
    by_symbol = {a.position.underlying: a.action for a in actions}
    assert by_symbol["GOOD"] == "close"    # processed despite the other's error
    assert by_symbol["BAD"] == "error"     # errored, but did not block GOOD
