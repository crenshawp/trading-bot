"""Dormant atomic exit-fill aggregation and materialization tests."""

from __future__ import annotations

import dataclasses
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from trading_bot import db, order_lifecycle
from trading_bot.broker.base import (
    STATUS_CANCELED,
    STATUS_FILLED,
    STATUS_PARTIALLY_FILLED,
)
from trading_bot.models import LongTermPosition, OptionPosition, PendingOrder

_ENTRY_AT = datetime(2026, 7, 20, 14, 30, tzinfo=UTC)
_ENTRY_FILL_AT = datetime(2026, 7, 20, 14, 31, tzinfo=UTC)
_EXIT_AT = datetime(2026, 7, 23, 15, 0, tzinfo=UTC)
_OPTION_SYMBOL = "GOOGL260918C00190000"


def _option_entry_payload(option_type: str, multiplier: int) -> str:
    return json.dumps(
        {
            "intent_kind": "option",
            "option_type": option_type,
            "strike": 190.0,
            "expiry": "2026-09-18",
            "multiplier": multiplier,
            "delta_entry": None,
            "theta": None,
            "vega": None,
            "gamma": None,
            "tp": 205.0,
            "sl": 175.0,
            "deadline": None,
        },
        sort_keys=True,
    )


def _share_entry_payload(source: str, direction: str) -> str:
    payload: dict[str, Any] = {
        "intent_kind": "long_term" if source == "long_term" else "shares_fallback",
        "source": source,
        "direction": direction,
    }
    if source == "swing_fallback":
        payload.update({"tp": 120.0, "sl": 80.0, "deadline": None})
    return json.dumps(payload, sort_keys=True)


def _insert_option(
    *,
    option_type: str = "call",
    qty: float = 2.0,
    entry_price: float = 4.0,
    multiplier: int = 100,
    entry_status: str = STATUS_FILLED,
) -> int:
    position_id = db.insert_option_position(
        OptionPosition(
            symbol=_OPTION_SYMBOL,
            underlying="GOOGL",
            option_type=option_type,
            strike=190.0,
            expiry="2026-09-18",
            contracts=qty,
            opened_at=_ENTRY_FILL_AT,
            multiplier=multiplier,
            premium_entry=entry_price,
            outcome="open",
        )
    )
    terminal = entry_status in {STATUS_FILLED, STATUS_CANCELED}
    db.insert_pending_order(
        PendingOrder(
            client_order_id=f"entry-option-{position_id}",
            broker_order_id=f"broker-entry-option-{position_id}",
            ticker="GOOGL",
            broker_symbol=_OPTION_SYMBOL,
            asset_class="stock",
            vehicle="option_full",
            target_position_kind="option",
            side="buy",
            requested_qty=qty,
            requested_limit_price=entry_price + 1.0,
            submitted_at=_ENTRY_AT,
            intent_payload_json=_option_entry_payload(option_type, multiplier),
            lifecycle_status=entry_status,
            broker_status=entry_status,
            filled_qty=qty,
            filled_avg_price=entry_price,
            last_fill_at=_ENTRY_FILL_AT,
            last_fill_time_source="broker",
            last_refreshed_at=_ENTRY_FILL_AT,
            terminal_reason=entry_status if terminal else None,
            terminal_at=_ENTRY_FILL_AT if terminal else None,
            position_kind="option",
            position_id=position_id,
        )
    )
    return position_id


def _insert_shares(
    *,
    source: str = "long_term",
    direction: str = "long",
    qty: float = 3.0,
    entry_price: float = 100.0,
) -> int:
    position_id = db.insert_long_term_position(
        LongTermPosition(
            ticker="GOOGL",
            asset_class="stock",
            entry_price=entry_price,
            entry_date=_ENTRY_FILL_AT,
            qty=qty,
            source=source,
            direction=direction,
            tp=120.0 if source == "swing_fallback" else None,
            sl=80.0 if source == "swing_fallback" else None,
        )
    )
    db.insert_pending_order(
        PendingOrder(
            client_order_id=f"entry-shares-{position_id}",
            broker_order_id=f"broker-entry-shares-{position_id}",
            ticker="GOOGL",
            broker_symbol="GOOGL",
            asset_class="stock",
            vehicle="shares",
            target_position_kind="long_term",
            side="sell" if direction == "short" else "buy",
            requested_qty=qty,
            requested_limit_price=entry_price + 1.0,
            submitted_at=_ENTRY_AT,
            intent_payload_json=_share_entry_payload(source, direction),
            lifecycle_status=STATUS_FILLED,
            broker_status=STATUS_FILLED,
            filled_qty=qty,
            filled_avg_price=entry_price,
            last_fill_at=_ENTRY_FILL_AT,
            last_fill_time_source="broker",
            last_refreshed_at=_ENTRY_FILL_AT,
            terminal_reason=STATUS_FILLED,
            terminal_at=_ENTRY_FILL_AT,
            position_kind="long_term",
            position_id=position_id,
        )
    )
    return position_id


def _insert_exit(
    position_kind: str,
    position_id: int,
    label: str,
    *,
    requested_qty: float,
    filled_qty: float,
    fill_price: float | None,
    status: str,
    exit_reason: str,
    fill_at: datetime | None,
    submitted_offset: int = 0,
    requested_limit_price: float = 999.0,
) -> int:
    terminal = status in {STATUS_FILLED, STATUS_CANCELED}
    if position_kind == "option":
        position = db.get_option_position(position_id)
        assert position is not None
        ticker = position.underlying
        broker_symbol = position.symbol
        asset_class = "stock"
        vehicle = position.vehicle
        side = "sell"
    else:
        position = db.get_long_term_position(position_id)
        assert position is not None
        ticker = broker_symbol = position.ticker
        asset_class = position.asset_class
        vehicle = "shares"
        side = "buy" if position.direction == "short" else "sell"
    return db.insert_pending_order(
        PendingOrder(
            client_order_id=f"exit-{label}",
            broker_order_id=f"broker-exit-{label}",
            order_role="exit",
            ticker=ticker,
            broker_symbol=broker_symbol,
            asset_class=asset_class,
            vehicle=vehicle,
            target_position_kind=position_kind,
            closes_position_kind=position_kind,
            closes_position_id=position_id,
            side=side,
            requested_qty=requested_qty,
            requested_limit_price=requested_limit_price,
            submitted_at=_EXIT_AT + timedelta(minutes=submitted_offset),
            intent_payload_json=json.dumps(
                {
                    "intent_kind": "position_exit",
                    "exit_reason": exit_reason,
                },
                sort_keys=True,
            ),
            lifecycle_status=status,
            broker_status=status,
            filled_qty=filled_qty,
            filled_avg_price=fill_price,
            last_fill_at=fill_at,
            last_fill_time_source="broker" if fill_at is not None else None,
            last_refreshed_at=_EXIT_AT + timedelta(minutes=submitted_offset),
            terminal_reason=status if terminal else None,
            terminal_at=(
                _EXIT_AT + timedelta(minutes=submitted_offset)
                if terminal else None
            ),
        )
    )


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_option_full_close_uses_actual_fills_reason_and_economic_outcome(
    tmp_db: Path,
    option_type: str,
) -> None:
    position_id = _insert_option(option_type=option_type, qty=2.0, entry_price=4.0)
    final_time = _EXIT_AT + timedelta(minutes=1)
    _insert_exit(
        "option",
        position_id,
        option_type,
        requested_qty=2.0,
        filled_qty=2.0,
        fill_price=3.0,
        status=STATUS_FILLED,
        exit_reason="take_profit",
        fill_at=final_time,
        requested_limit_price=9.5,
    )

    result = order_lifecycle.materialize_position_exit_fills(
        "option", position_id
    )

    assert result.action == order_lifecycle.EXIT_FILL_CLOSED
    assert result.exit_vwap == 3.0
    assert result.pnl_dollars == -200.0
    assert result.outcome == "loss"
    assert result.exit_reason == "take_profit"
    position = db.get_option_position(position_id)
    assert position is not None
    assert position.exit_price == 3.0
    assert position.closed_at == final_time
    assert position.exit_reason == "take_profit"
    assert position.pnl_dollars == -200.0
    assert position.outcome == "loss"
    (exit_order,) = db.get_pending_exit_orders_for_position("option", position_id)
    assert exit_order.requested_limit_price == 9.5
    assert exit_order.fees_dollars is None


@pytest.mark.parametrize(
    ("source", "direction", "exit_price", "expected_pnl"),
    [
        ("long_term", "long", 110.0, 30.0),
        ("swing_fallback", "long", 110.0, 30.0),
        ("swing_fallback", "short", 90.0, 30.0),
    ],
)
def test_share_full_close_preserves_direction_semantics_for_later_query(
    tmp_db: Path,
    source: str,
    direction: str,
    exit_price: float,
    expected_pnl: float,
) -> None:
    position_id = _insert_shares(source=source, direction=direction)
    final_time = _EXIT_AT + timedelta(minutes=2)
    _insert_exit(
        "long_term",
        position_id,
        f"{source}-{direction}",
        requested_qty=3.0,
        filled_qty=3.0,
        fill_price=exit_price,
        status=STATUS_FILLED,
        exit_reason="hold_deadline",
        fill_at=final_time,
    )

    result = order_lifecycle.materialize_position_exit_fills(
        "long_term", position_id
    )

    assert result.action == order_lifecycle.EXIT_FILL_CLOSED
    assert result.pnl_dollars == expected_pnl
    assert result.outcome == "win"
    position = db.get_long_term_position(position_id)
    assert position is not None
    assert position.status == "closed"
    assert position.exit_price == exit_price
    assert position.exit_date == final_time
    assert position.exit_reason == "hold_deadline"
    assert position.source == source
    assert position.direction == direction


def test_multiple_terminal_partial_attempts_aggregate_vwap_and_final_reason(
    tmp_db: Path,
) -> None:
    position_id = _insert_shares(qty=3.0, entry_price=100.0)
    first_time = _EXIT_AT + timedelta(minutes=1)
    second_time = _EXIT_AT + timedelta(minutes=2)
    _insert_exit(
        "long_term",
        position_id,
        "partial-1",
        requested_qty=3.0,
        filled_qty=1.0,
        fill_price=110.0,
        status=STATUS_CANCELED,
        exit_reason="take_profit",
        fill_at=first_time,
        submitted_offset=0,
    )
    _insert_exit(
        "long_term",
        position_id,
        "partial-2",
        requested_qty=2.0,
        filled_qty=1.0,
        fill_price=120.0,
        status=STATUS_CANCELED,
        exit_reason="stop_loss",
        fill_at=second_time,
        submitted_offset=1,
    )

    partial = order_lifecycle.materialize_position_exit_fills(
        "long_term", position_id
    )
    assert partial.action == order_lifecycle.EXIT_FILL_OPEN
    assert partial.exited_qty == 2.0
    assert partial.remaining_qty == 1.0
    assert partial.exit_vwap == 115.0
    assert order_lifecycle.remaining_position_quantity(
        "long_term", position_id
    ) == 1.0
    position = db.get_long_term_position(position_id)
    assert position is not None and position.status == "open"

    final_time = _EXIT_AT + timedelta(minutes=3)
    final_id = _insert_exit(
        "long_term",
        position_id,
        "partial-3",
        requested_qty=1.0,
        filled_qty=1.0,
        fill_price=130.0,
        status=STATUS_FILLED,
        exit_reason="hold_deadline",
        fill_at=final_time,
        submitted_offset=2,
    )
    closed = order_lifecycle.materialize_position_exit_fills(
        "long_term", position_id
    )

    assert closed.action == order_lifecycle.EXIT_FILL_CLOSED
    assert closed.exited_qty == 3.0
    assert closed.remaining_qty == 0.0
    assert closed.exit_vwap == 120.0
    assert closed.exit_reason == "hold_deadline"
    assert closed.final_fill_at == final_time
    assert closed.final_exit_pending_order_id == final_id
    assert closed.pnl_dollars == 60.0


def test_repeated_partial_snapshot_uses_one_rows_latest_cumulative_fill(
    tmp_db: Path,
) -> None:
    position_id = _insert_shares(qty=3.0, entry_price=100.0)
    exit_id = _insert_exit(
        "long_term",
        position_id,
        "cumulative",
        requested_qty=3.0,
        filled_qty=1.0,
        fill_price=110.0,
        status=STATUS_PARTIALLY_FILLED,
        exit_reason="take_profit",
        fill_at=_EXIT_AT + timedelta(minutes=1),
    )
    first = order_lifecycle.position_exit_fill_state("long_term", position_id)
    assert first.exited_qty == 1.0 and first.remaining_qty == 2.0

    final_time = _EXIT_AT + timedelta(minutes=2)
    db.update_pending_order(
        exit_id,
        lifecycle_status=STATUS_FILLED,
        broker_status=STATUS_FILLED,
        filled_qty=3.0,
        filled_avg_price=120.0,
        last_fill_at=final_time,
        last_fill_time_source="broker",
        last_refreshed_at=final_time,
        terminal_reason=STATUS_FILLED,
        terminal_at=final_time,
    )
    closed = order_lifecycle.materialize_position_exit_fills(
        "long_term", position_id
    )

    assert closed.action == order_lifecycle.EXIT_FILL_CLOSED
    assert closed.exited_qty == 3.0
    assert closed.exit_vwap == 120.0
    assert closed.pnl_dollars == 60.0


def test_zero_fill_terminal_and_terminal_partial_leave_position_open(
    tmp_db: Path,
) -> None:
    position_id = _insert_option(qty=2.0, entry_price=4.0)
    _insert_exit(
        "option",
        position_id,
        "zero",
        requested_qty=2.0,
        filled_qty=0.0,
        fill_price=None,
        status=STATUS_CANCELED,
        exit_reason="stop_loss",
        fill_at=None,
    )
    zero = order_lifecycle.materialize_position_exit_fills("option", position_id)
    assert zero.action == order_lifecycle.EXIT_FILL_OPEN
    assert zero.exited_qty == 0.0
    assert zero.remaining_qty == 2.0
    assert zero.exit_vwap is None
    position = db.get_option_position(position_id)
    assert position is not None and position.outcome == "open"

    _insert_exit(
        "option",
        position_id,
        "terminal-partial",
        requested_qty=2.0,
        filled_qty=0.5,
        fill_price=4.5,
        status=STATUS_CANCELED,
        exit_reason="stop_loss",
        fill_at=_EXIT_AT + timedelta(minutes=2),
        submitted_offset=1,
    )
    partial = order_lifecycle.materialize_position_exit_fills(
        "option", position_id
    )
    assert partial.action == order_lifecycle.EXIT_FILL_OPEN
    assert partial.remaining_qty == 1.5
    assert db.get_option_position(position_id) == position


def test_full_actual_quantity_with_nonterminal_exit_does_not_close(
    tmp_db: Path,
) -> None:
    position_id = _insert_option(qty=2.0)
    _insert_exit(
        "option",
        position_id,
        "nonterminal-full",
        requested_qty=2.0,
        filled_qty=2.0,
        fill_price=5.0,
        status=STATUS_PARTIALLY_FILLED,
        exit_reason="take_profit",
        fill_at=_EXIT_AT + timedelta(minutes=1),
    )

    result = order_lifecycle.materialize_position_exit_fills(
        "option", position_id
    )

    assert result.action == order_lifecycle.EXIT_FILL_PENDING
    assert result.remaining_qty == 0.0
    position = db.get_option_position(position_id)
    assert position is not None and position.outcome == "open"
    assert position.closed_at is None


def test_cent_rounding_can_make_small_actual_gain_breakeven(tmp_db: Path) -> None:
    position_id = _insert_option(qty=1.0, entry_price=1.0, multiplier=100)
    _insert_exit(
        "option",
        position_id,
        "cent-breakeven",
        requested_qty=1.0,
        filled_qty=1.0,
        fill_price=1.00004,
        status=STATUS_FILLED,
        exit_reason="hold_deadline",
        fill_at=_EXIT_AT + timedelta(minutes=1),
    )

    result = order_lifecycle.materialize_position_exit_fills(
        "option", position_id
    )

    assert result.pnl_dollars == 0.0
    assert result.outcome == "breakeven"
    position = db.get_option_position(position_id)
    assert position is not None and position.outcome == "breakeven"


def test_oversell_is_integrity_unknown_and_mutates_nothing(tmp_db: Path) -> None:
    position_id = _insert_option(qty=2.0)
    _insert_exit(
        "option",
        position_id,
        "oversell-1",
        requested_qty=1.5,
        filled_qty=1.5,
        fill_price=5.0,
        status=STATUS_CANCELED,
        exit_reason="take_profit",
        fill_at=_EXIT_AT + timedelta(minutes=1),
    )
    _insert_exit(
        "option",
        position_id,
        "oversell-2",
        requested_qty=1.0,
        filled_qty=1.0,
        fill_price=5.5,
        status=STATUS_FILLED,
        exit_reason="take_profit",
        fill_at=_EXIT_AT + timedelta(minutes=2),
        submitted_offset=1,
    )

    result = order_lifecycle.materialize_position_exit_fills(
        "option", position_id
    )

    assert result.action == order_lifecycle.EXIT_FILL_INTEGRITY_UNKNOWN
    assert result.remaining_qty is None
    assert "oversells" in result.reason
    position = db.get_option_position(position_id)
    assert position is not None and position.outcome == "open"


def test_nonterminal_entry_conflict_and_unlinked_legacy_are_excluded(
    tmp_db: Path,
) -> None:
    nonterminal_id = _insert_option(
        qty=2.0,
        entry_status=STATUS_PARTIALLY_FILLED,
    )
    nonterminal = order_lifecycle.position_exit_fill_state(
        "option", nonterminal_id
    )
    assert nonterminal.action == order_lifecycle.EXIT_FILL_INTEGRITY_UNKNOWN
    assert "not terminal" in nonterminal.reason

    legacy_id = db.insert_option_position(
        OptionPosition(
            symbol=_OPTION_SYMBOL,
            underlying="GOOGL",
            option_type="call",
            strike=190.0,
            expiry="2026-09-18",
            contracts=1.0,
            opened_at=_ENTRY_FILL_AT,
            premium_entry=4.0,
            outcome="open",
        )
    )
    legacy = order_lifecycle.materialize_position_exit_fills("option", legacy_id)
    assert legacy.action == order_lifecycle.EXIT_FILL_INTEGRITY_UNKNOWN
    assert "exactly one" in legacy.reason
    assert order_lifecycle.remaining_position_quantity("option", legacy_id) is None


def test_conflicting_typed_quantity_is_integrity_unknown(tmp_db: Path) -> None:
    position_id = _insert_shares(qty=3.0)
    conn = db.get_connection()
    try:
        conn.execute(
            "UPDATE long_term_positions SET qty = 4.0 WHERE id = ?",
            (position_id,),
        )
        conn.commit()
    finally:
        conn.close()

    result = order_lifecycle.position_exit_fill_state(
        "long_term", position_id
    )
    assert result.action == order_lifecycle.EXIT_FILL_INTEGRITY_UNKNOWN
    assert "does not match" in result.reason


def test_multiple_entry_links_are_integrity_unknown(tmp_db: Path) -> None:
    position_id = _insert_option(qty=1.0)
    original = db.get_pending_order(f"broker-entry-option-{position_id}")
    assert original is not None
    db.insert_pending_order(
        dataclasses.replace(
            original,
            id=None,
            client_order_id="duplicate-entry-link",
            broker_order_id="duplicate-entry-link-broker",
        )
    )

    result = order_lifecycle.position_exit_fill_state("option", position_id)
    assert result.action == order_lifecycle.EXIT_FILL_INTEGRITY_UNKNOWN
    assert "exactly one" in result.reason


@pytest.mark.parametrize(
    ("corruption_sql", "match"),
    [
        (
            "UPDATE pending_orders SET last_fill_at = 'not-a-time' "
            "WHERE client_order_id = 'exit-corrupt'",
            "malformed",
        ),
        (
            "UPDATE pending_orders SET filled_avg_price = 1e999 "
            "WHERE client_order_id = 'exit-corrupt'",
            "finite",
        ),
    ],
)
def test_invalid_exit_price_or_time_is_integrity_unknown(
    tmp_db: Path,
    corruption_sql: str,
    match: str,
) -> None:
    position_id = _insert_option(qty=1.0)
    _insert_exit(
        "option",
        position_id,
        "corrupt",
        requested_qty=1.0,
        filled_qty=1.0,
        fill_price=5.0,
        status=STATUS_FILLED,
        exit_reason="take_profit",
        fill_at=_EXIT_AT + timedelta(minutes=1),
    )
    conn = db.get_connection()
    try:
        conn.execute(corruption_sql)
        conn.commit()
    finally:
        conn.close()

    result = order_lifecycle.materialize_position_exit_fills(
        "option", position_id
    )
    assert result.action == order_lifecycle.EXIT_FILL_INTEGRITY_UNKNOWN
    assert match in result.reason
    position = db.get_option_position(position_id)
    assert position is not None and position.outcome == "open"


def test_unsupported_exit_payload_is_integrity_unknown(tmp_db: Path) -> None:
    position_id = _insert_option(qty=1.0)
    _insert_exit(
        "option",
        position_id,
        "payload",
        requested_qty=1.0,
        filled_qty=1.0,
        fill_price=5.0,
        status=STATUS_FILLED,
        exit_reason="take_profit",
        fill_at=_EXIT_AT + timedelta(minutes=1),
    )
    conn = db.get_connection()
    try:
        conn.execute("DROP TRIGGER trg_pending_orders_immutable_intent")
        conn.execute(
            "UPDATE pending_orders SET intent_payload_version = 2 "
            "WHERE client_order_id = 'exit-payload'"
        )
        conn.commit()
    finally:
        conn.close()

    result = order_lifecycle.materialize_position_exit_fills(
        "option", position_id
    )
    assert result.action == order_lifecycle.EXIT_FILL_INTEGRITY_UNKNOWN
    assert "unsupported intent payload version" in result.reason


def test_exact_replay_is_noop_and_preserves_summary(tmp_db: Path) -> None:
    position_id = _insert_option(qty=1.0, entry_price=4.0)
    _insert_exit(
        "option",
        position_id,
        "replay",
        requested_qty=1.0,
        filled_qty=1.0,
        fill_price=5.0,
        status=STATUS_FILLED,
        exit_reason="take_profit",
        fill_at=_EXIT_AT + timedelta(minutes=1),
    )

    first = order_lifecycle.materialize_position_exit_fills("option", position_id)
    before = db.get_option_position(position_id)
    second = order_lifecycle.materialize_position_exit_fills("option", position_id)
    after = db.get_option_position(position_id)

    assert first.action == order_lifecycle.EXIT_FILL_CLOSED
    assert second.action == order_lifecycle.EXIT_FILL_UNCHANGED
    assert second.reason == "exact close-summary replay"
    assert before == after


def test_atomic_update_failure_rolls_back_complete_close_summary(
    tmp_db: Path,
) -> None:
    position_id = _insert_option(qty=1.0)
    _insert_exit(
        "option",
        position_id,
        "rollback",
        requested_qty=1.0,
        filled_qty=1.0,
        fill_price=5.0,
        status=STATUS_FILLED,
        exit_reason="take_profit",
        fill_at=_EXIT_AT + timedelta(minutes=1),
    )
    conn = db.get_connection()
    try:
        conn.execute(
            "CREATE TRIGGER fail_exit_close BEFORE UPDATE ON option_positions "
            "BEGIN SELECT RAISE(ABORT, 'simulated close failure'); END"
        )
        conn.commit()
    finally:
        conn.close()

    result = order_lifecycle.materialize_position_exit_fills(
        "option", position_id
    )

    assert result.action == order_lifecycle.EXIT_FILL_INTEGRITY_UNKNOWN
    assert "simulated close failure" in result.reason
    position = db.get_option_position(position_id)
    assert position is not None
    assert position.outcome == "open"
    assert position.closed_at is None
    assert position.exit_price is None
    assert position.exit_reason is None
    assert position.pnl_dollars is None


def test_database_open_failure_returns_integrity_unknown(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    position_id = _insert_option(qty=1.0)

    def fail_connection() -> sqlite3.Connection:
        raise sqlite3.OperationalError("database unavailable")

    monkeypatch.setattr(db, "get_connection", fail_connection)
    result = order_lifecycle.materialize_position_exit_fills(
        "option", position_id
    )

    assert result.action == order_lifecycle.EXIT_FILL_INTEGRITY_UNKNOWN
    assert result.remaining_qty is None
    assert "database unavailable" in result.reason
