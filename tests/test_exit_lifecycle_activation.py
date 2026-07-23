"""Final activation proofs for restart-safe exit reconciliation."""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from trading_bot import db, options_execution, order_lifecycle
from trading_bot.broker.base import (
    ORDER_TYPE_LIMIT,
    STATUS_CANCELED,
    STATUS_FILLED,
    STATUS_NEW,
    STATUS_PARTIALLY_FILLED,
    TIF_DAY,
    OrderResult,
)
from trading_bot.broker.fake import FakeBroker
from trading_bot.models import OptionPosition, PendingOrder

_ENTRY_AT = datetime(2026, 7, 22, 14, 30, tzinfo=UTC)
_EXIT_AT = datetime(2026, 7, 23, 15, 0, tzinfo=UTC)
_OPTION_SYMBOL = "GOOGL260918C00190000"


def _option_payload() -> str:
    return json.dumps(
        {
            "intent_kind": "option",
            "option_type": "call",
            "strike": 190.0,
            "expiry": "2026-09-18",
            "multiplier": 100,
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


def _insert_linked_option(
    *,
    entry_qty: float = 2.0,
    requested_qty: float | None = None,
    entry_price: float = 5.0,
    entry_status: str = STATUS_FILLED,
    broker_order_id: str | None = None,
) -> int:
    position_id = db.insert_option_position(
        OptionPosition(
            symbol=_OPTION_SYMBOL,
            underlying="GOOGL",
            option_type="call",
            strike=190.0,
            expiry="2026-09-18",
            contracts=entry_qty,
            opened_at=_ENTRY_AT,
            premium_entry=entry_price,
            outcome="open",
            vehicle="option_full",
        )
    )
    terminal = entry_status in {STATUS_FILLED, STATUS_CANCELED}
    entry_id = broker_order_id or f"entry-broker-{position_id}"
    db.insert_pending_order(
        PendingOrder(
            client_order_id=f"entry-client-{position_id}",
            broker_order_id=entry_id,
            ticker="GOOGL",
            broker_symbol=_OPTION_SYMBOL,
            asset_class="stock",
            vehicle="option_full",
            target_position_kind="option",
            side="buy",
            requested_qty=requested_qty or entry_qty,
            requested_limit_price=entry_price,
            submitted_at=_ENTRY_AT,
            intent_payload_json=_option_payload(),
            lifecycle_status=entry_status,
            broker_status=entry_status,
            filled_qty=entry_qty,
            filled_avg_price=entry_price,
            last_fill_at=_ENTRY_AT,
            last_fill_time_source="broker",
            last_refreshed_at=_ENTRY_AT,
            terminal_reason=entry_status if terminal else None,
            terminal_at=_ENTRY_AT if terminal else None,
            position_kind="option",
            position_id=position_id,
        )
    )
    return position_id


def _snapshot(
    status: str,
    filled_qty: float,
    fill_price: float,
    fill_at: datetime,
) -> OrderResult:
    return OrderResult(
        ok=True,
        status=status,
        order_id="placeholder",
        filled_qty=filled_qty,
        filled_avg_price=fill_price,
        filled_at=fill_at.isoformat(),
        updated_at=fill_at.isoformat(),
        raw_status=status,
    )


class RefreshOnlyBroker(FakeBroker):
    def __init__(self, snapshot: OrderResult | None = None) -> None:
        super().__init__()
        self.snapshot = snapshot
        self.get_calls = 0
        self.submit_calls = 0

    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        *,
        order_type: str = ORDER_TYPE_LIMIT,
        limit_price: float | None = None,
        time_in_force: str = TIF_DAY,
        client_order_id: str | None = None,
    ) -> OrderResult:
        self.submit_calls += 1
        raise AssertionError("reconciliation must never submit")

    def get_order(self, order_id: str) -> OrderResult:
        self.get_calls += 1
        assert self.snapshot is not None
        return dataclasses.replace(self.snapshot, order_id=order_id)


class TerminalPartialBroker(FakeBroker):
    def __init__(self) -> None:
        super().__init__()
        self.quantities: list[float] = []

    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        *,
        order_type: str = ORDER_TYPE_LIMIT,
        limit_price: float | None = None,
        time_in_force: str = TIF_DAY,
        client_order_id: str | None = None,
    ) -> OrderResult:
        self.quantities.append(qty)
        first = len(self.quantities) == 1
        status = STATUS_CANCELED if first else STATUS_FILLED
        fill_at = _EXIT_AT + timedelta(minutes=len(self.quantities))
        return OrderResult(
            ok=True,
            status=status,
            order_id=f"terminal-partial-{len(self.quantities)}",
            client_order_id=client_order_id,
            symbol=symbol,
            qty=qty,
            filled_qty=1.0,
            filled_avg_price=6.0 if first else 8.0,
            side=side,
            order_type=order_type,
            time_in_force=time_in_force,
            limit_price=limit_price,
            submitted_at=_EXIT_AT.isoformat(),
            filled_at=fill_at.isoformat(),
            updated_at=fill_at.isoformat(),
            raw_status=status,
        )


def test_exit_reconciliation_survives_unfilled_partial_restart_full_and_replay(
    tmp_db: Path,
) -> None:
    position_id = _insert_linked_option()
    submit_broker = FakeBroker(auto_fill=False)
    submitted = order_lifecycle.submit_position_exit(
        submit_broker,
        position_kind="option",
        position_id=position_id,
        requested_qty=2.0,
        requested_limit_price=99.0,
        exit_reason="emergency_shutdown",
        submitted_at=_EXIT_AT,
    )
    assert submitted.status == STATUS_NEW

    unfilled = order_lifecycle.reconcile_pending_orders(
        submit_broker, observed_at=_EXIT_AT
    )
    assert unfilled.exit_materializations == ()
    current = db.get_option_position(position_id)
    assert current is not None and current.outcome == "open"

    partial_broker = RefreshOnlyBroker(
        _snapshot(
            STATUS_PARTIALLY_FILLED,
            1.0,
            6.0,
            _EXIT_AT + timedelta(minutes=1),
        )
    )
    partial = order_lifecycle.reconcile_pending_orders(
        partial_broker, observed_at=_EXIT_AT + timedelta(minutes=1)
    )
    assert partial.exit_materializations[0].action == order_lifecycle.EXIT_FILL_OPEN
    current = db.get_option_position(position_id)
    assert current is not None and current.outcome == "open"

    restarted_broker = RefreshOnlyBroker(
        _snapshot(
            STATUS_FILLED,
            2.0,
            7.0,
            _EXIT_AT + timedelta(minutes=2),
        )
    )
    completed = order_lifecycle.reconcile_pending_orders(
        restarted_broker, observed_at=_EXIT_AT + timedelta(minutes=2)
    )
    assert completed.exit_materializations[0].action == order_lifecycle.EXIT_FILL_CLOSED
    closed = db.get_option_position(position_id)
    assert closed is not None
    assert closed.outcome == "win"
    assert closed.exit_reason == "emergency_shutdown"
    assert closed.exit_price == 7.0
    assert closed.closed_at == _EXIT_AT + timedelta(minutes=2)
    assert closed.pnl_dollars == 400.0

    replay_broker = RefreshOnlyBroker()
    replay = order_lifecycle.reconcile_pending_orders(
        replay_broker, observed_at=_EXIT_AT + timedelta(minutes=3)
    )
    assert replay.refreshes == ()
    assert replay.exit_materializations[0].action == (
        order_lifecycle.EXIT_FILL_UNCHANGED
    )
    assert replay_broker.get_calls == replay_broker.submit_calls == 0
    replayed = db.get_option_position(position_id)
    assert replayed == closed


def test_terminal_partial_emergency_retries_only_actual_remaining_quantity(
    tmp_db: Path,
) -> None:
    position_id = _insert_linked_option()
    broker = TerminalPartialBroker()
    position = db.get_option_position(position_id)
    assert position is not None

    first, first_pnl = options_execution.close_option_position(
        broker,
        position,
        exit_price=99.0,
        now=_EXIT_AT,
        reason="emergency_shutdown",
    )
    assert first is not None and first.status == STATUS_CANCELED
    assert first_pnl is None
    partial = db.get_option_position(position_id)
    assert partial is not None and partial.outcome == "open"

    second, second_pnl = options_execution.close_option_position(
        broker,
        position,
        exit_price=101.0,
        now=_EXIT_AT + timedelta(minutes=2),
        reason="emergency_shutdown",
    )
    assert second is not None and second.status == STATUS_FILLED
    assert broker.quantities == [2.0, 1.0]
    closed = db.get_option_position(position_id)
    assert closed is not None
    assert closed.outcome == "win"
    assert closed.exit_reason == "emergency_shutdown"
    assert closed.exit_price == 7.0
    assert closed.pnl_dollars == second_pnl == 400.0


def test_terminal_exit_update_before_materialization_recovers_without_io(
    tmp_db: Path,
) -> None:
    position_id = _insert_linked_option()
    submit_broker = FakeBroker(auto_fill=False)
    order_lifecycle.submit_position_exit(
        submit_broker,
        position_kind="option",
        position_id=position_id,
        requested_qty=2.0,
        requested_limit_price=9.0,
        exit_reason="stop_loss",
        submitted_at=_EXIT_AT,
    )
    persisted_only = RefreshOnlyBroker(
        _snapshot(STATUS_FILLED, 2.0, 4.0, _EXIT_AT + timedelta(minutes=1))
    )
    refreshed = order_lifecycle.refresh_pending_orders(
        persisted_only, observed_at=_EXIT_AT + timedelta(minutes=1)
    )
    assert refreshed[0].lifecycle_status == STATUS_FILLED
    current = db.get_option_position(position_id)
    assert current is not None and current.outcome == "open"

    restarted = RefreshOnlyBroker()
    recovered = order_lifecycle.reconcile_pending_orders(
        restarted, observed_at=_EXIT_AT + timedelta(minutes=2)
    )
    assert recovered.refreshes == ()
    assert recovered.exit_materializations[0].action == (
        order_lifecycle.EXIT_FILL_CLOSED
    )
    assert restarted.get_calls == restarted.submit_calls == 0
    closed = db.get_option_position(position_id)
    assert closed is not None
    assert closed.outcome == "loss"
    assert closed.exit_price == 4.0
    assert closed.pnl_dollars == -200.0


def test_emergency_freezes_growing_entry_before_submitting_actual_quantity(
    tmp_db: Path,
) -> None:
    position_id = _insert_linked_option(
        entry_qty=1.0,
        requested_qty=2.0,
        entry_status=STATUS_PARTIALLY_FILLED,
        broker_order_id="growing-entry",
    )
    broker = FakeBroker(auto_fill=True)
    later_fill_at = _ENTRY_AT + timedelta(minutes=5)
    broker._orders["growing-entry"] = OrderResult(
        ok=True,
        status=STATUS_PARTIALLY_FILLED,
        order_id="growing-entry",
        client_order_id=f"entry-client-{position_id}",
        symbol=_OPTION_SYMBOL,
        qty=2.0,
        filled_qty=1.5,
        filled_avg_price=5.2,
        side="buy",
        order_type=ORDER_TYPE_LIMIT,
        time_in_force=TIF_DAY,
        limit_price=5.0,
        submitted_at=_ENTRY_AT.isoformat(),
        filled_at=later_fill_at.isoformat(),
        updated_at=later_fill_at.isoformat(),
        raw_status=STATUS_PARTIALLY_FILLED,
    )
    position = db.get_option_position(position_id)
    assert position is not None

    order, pnl = options_execution.close_option_position(
        broker,
        position,
        exit_price=6.0,
        now=_EXIT_AT,
        reason="emergency_shutdown",
    )

    assert order is not None and order.status == STATUS_FILLED
    assert order.qty == 1.5
    entry = db.get_pending_order("growing-entry")
    assert entry is not None
    assert entry.lifecycle_status == STATUS_CANCELED
    assert entry.filled_qty == 1.5
    closed = db.get_option_position(position_id)
    assert closed is not None
    assert closed.contracts == 1.5
    assert closed.premium_entry == 5.2
    assert closed.exit_price == 6.0
    assert closed.pnl_dollars == pnl == 120.0
    assert closed.exit_reason == "emergency_shutdown"
