"""Final activation proofs for restart-safe exit reconciliation."""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from trading_bot import db, options_execution, order_lifecycle
from trading_bot.broker import alpaca
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
    tp: float | None = None,
    sl: float | None = None,
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
            tp=tp,
            sl=sl,
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



def _growing_entry_broker(position_id: int, broker_order_id: str) -> FakeBroker:
    """A broker whose linked entry is still working with 1.5 of 2 filled."""
    broker = FakeBroker(auto_fill=True)
    later_fill_at = _ENTRY_AT + timedelta(minutes=5)
    broker._orders[broker_order_id] = OrderResult(
        ok=True,
        status=STATUS_PARTIALLY_FILLED,
        order_id=broker_order_id,
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
    return broker


def test_option_watcher_stop_loss_fires_while_the_entry_is_still_working(
    tmp_db: Path,
) -> None:
    """The ROUTINE watcher must close a triggered position whose entry is
    still growing — not merely the emergency closer.

    A partially-filled entry is an ordinary state: the typed position exists
    and is exposed the moment the first contract fills. Before the freeze was
    added here, the exit submitter's materializer refused outright ("linked
    entry order is not terminal"), so TP, SL and the hold deadline were ALL
    unreachable for that position and the watcher merely re-errored each hour.
    """
    position_id = _insert_linked_option(
        entry_qty=1.0,
        requested_qty=2.0,
        entry_status=STATUS_PARTIALLY_FILLED,
        broker_order_id="working-entry",
        tp=205.0,
        sl=175.0,
    )
    broker = _growing_entry_broker(position_id, "working-entry")
    position = db.get_option_position(position_id)
    assert position is not None

    actions = options_execution.watch_open_option_positions(
        broker,
        underlying_price_fetch=lambda _underlying: 170.0,  # below sl=175
        option_price_fetch=lambda _symbol: 6.0,
        now=_EXIT_AT,
        positions=[position],
    )

    assert len(actions) == 1
    assert actions[0].action == "close"
    assert actions[0].reason == options_execution.EXIT_STOP_LOSS

    # The working entry was cancelled and its final fill materialized, so the
    # exit went out for the quantity actually held.
    entry = db.get_pending_order("working-entry")
    assert entry is not None
    assert entry.lifecycle_status == STATUS_CANCELED
    assert entry.filled_qty == 1.5
    assert actions[0].order is not None
    assert actions[0].order.qty == 1.5

    closed = db.get_option_position(position_id)
    assert closed is not None
    assert closed.exit_reason == options_execution.EXIT_STOP_LOSS
    assert closed.exit_price == 6.0


def test_option_watcher_leaves_a_settled_entry_untouched(tmp_db: Path) -> None:
    """The freeze is a no-op on the ordinary fully-filled path.

    freeze_position_entry_intent short-circuits to "already_terminal" for a
    settled entry, so no cancel is issued and the close is unchanged — this
    pins that the fix did not disturb the common case.
    """
    position_id = _insert_linked_option(
        entry_qty=2.0, entry_status=STATUS_FILLED, tp=205.0, sl=175.0,
    )
    broker = FakeBroker(auto_fill=True)
    position = db.get_option_position(position_id)
    assert position is not None

    actions = options_execution.watch_open_option_positions(
        broker,
        underlying_price_fetch=lambda _underlying: 170.0,
        option_price_fetch=lambda _symbol: 6.0,
        now=_EXIT_AT,
        positions=[position],
    )

    assert len(actions) == 1
    assert actions[0].action == "close"
    assert actions[0].order is not None
    assert actions[0].order.qty == 2.0
    entry = db.get_pending_order(f"entry-broker-{position_id}")
    assert entry is not None
    assert entry.lifecycle_status == STATUS_FILLED  # never cancelled


class _PendingCancelBroker(FakeBroker):
    """Models Alpaca's real cancel semantics, which FakeBroker does not.

    ``DELETE /v2/orders/{id}`` returning 204 means the cancel was QUEUED, not
    applied. A ``GET`` immediately afterwards very commonly still shows the
    order working, with Alpaca's raw status ``pending_cancel``. FakeBroker
    instead swaps in a fully CANCELED order, i.e. it only ever models the
    already-applied case — which is why this path had no coverage.
    """

    def cancel_order(self, order_id: str) -> OrderResult:
        order = self._orders[order_id]
        # 204: accepted, not yet applied. The stored order is left WORKING,
        # carrying whatever the neutral map makes of Alpaca's raw status.
        self._orders[order_id] = dataclasses.replace(
            order, status=alpaca.map_status("pending_cancel"), raw_status="pending_cancel"
        )
        return OrderResult(ok=True, status=STATUS_CANCELED, order_id=order_id)


def test_pending_cancel_does_not_terminalize_a_still_working_entry(
    tmp_db: Path,
) -> None:
    """A cancel REQUEST must not close the ledger on an order that can still fill.

    Alpaca returns ``pending_cancel`` when the cancel was accepted but not yet
    applied — the order is still working at the exchange. Mapping that onto a
    terminal status stamped ``terminal_at``, and a locally-terminal row is never
    read from the broker again, so the freeze materialized the position at
    whatever had filled at that instant and any later fill was lost forever.

    The correct behaviour was already written — ``freeze_position_entry_intent``
    has an explicit "entry remains nonterminal after cancel" branch — the status
    map simply prevented it from ever being reached.
    """
    position_id = _insert_linked_option(
        entry_qty=1.0,
        requested_qty=2.0,
        entry_status=STATUS_PARTIALLY_FILLED,
        broker_order_id="pending-cancel-entry",
    )
    broker = _PendingCancelBroker(auto_fill=True)
    broker._orders["pending-cancel-entry"] = dataclasses.replace(
        OrderResult(
            ok=True,
            status=STATUS_PARTIALLY_FILLED,
            order_id="pending-cancel-entry",
            client_order_id=f"entry-client-{position_id}",
            symbol=_OPTION_SYMBOL,
            qty=2.0,
            filled_qty=1.0,
            filled_avg_price=5.0,
            side="buy",
            order_type=ORDER_TYPE_LIMIT,
            time_in_force=TIF_DAY,
            limit_price=5.0,
            submitted_at=_ENTRY_AT.isoformat(),
            updated_at=_ENTRY_AT.isoformat(),
        ),
        # What Alpaca actually reports right after an accepted cancel request.
        raw_status="pending_cancel",
    )

    result = order_lifecycle.freeze_position_entry_intent(
        broker, position_kind="option", position_id=position_id, observed_at=_EXIT_AT
    )

    # Not frozen, and explicitly reported as still pending — the honest answer.
    assert result.frozen is False
    assert result.action == "pending"

    entry = db.get_pending_order("pending-cancel-entry")
    assert entry is not None
    # The load-bearing assertion: the row stays live, so the next refresh cycle
    # still reads it from the broker and the eventual real fill is captured.
    assert entry.terminal_at is None
    assert entry.lifecycle_status != STATUS_CANCELED
