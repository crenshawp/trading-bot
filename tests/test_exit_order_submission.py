"""Dormant restart-safe exit-order submission tests."""

from __future__ import annotations

import dataclasses
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from trading_bot import db, long_term, options_execution, order_lifecycle
from trading_bot.broker.base import (
    ORDER_LOOKUP_NOT_FOUND,
    ORDER_LOOKUP_UNAVAILABLE,
    ORDER_TYPE_LIMIT,
    STATUS_ERROR,
    STATUS_FILLED,
    STATUS_NEW,
    STATUS_PARTIALLY_FILLED,
    STATUS_REJECTED,
    TIF_DAY,
    OrderLookupResult,
    OrderResult,
)
from trading_bot.broker.fake import FakeBroker
from trading_bot.models import LongTermPosition, OptionPosition, PendingOrder

_ENTRY_AT = datetime(2026, 7, 22, 14, 30, tzinfo=UTC)
_FILL_AT = datetime(2026, 7, 22, 14, 31, tzinfo=UTC)
_EXIT_AT = datetime(2026, 7, 23, 15, 0, tzinfo=UTC)
_OPTION_SYMBOL = "GOOGL260918C00190000"


def _option_entry_payload() -> str:
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


def _share_entry_payload(*, source: str, direction: str) -> str:
    payload: dict[str, Any] = {
        "intent_kind": "long_term" if source == "long_term" else "shares_fallback",
        "source": source,
        "direction": direction,
    }
    if source == "swing_fallback":
        payload.update({"tp": 190.0, "sl": 160.0, "deadline": None})
    return json.dumps(payload, sort_keys=True)


def _insert_linked_option(*, qty: float = 2.0) -> int:
    position_id = db.insert_option_position(
        OptionPosition(
            symbol=_OPTION_SYMBOL,
            underlying="GOOGL",
            option_type="call",
            strike=190.0,
            expiry="2026-09-18",
            contracts=qty,
            opened_at=_FILL_AT,
            premium_entry=4.2,
            outcome="open",
            vehicle="option_full",
        )
    )
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
            requested_limit_price=4.25,
            submitted_at=_ENTRY_AT,
            intent_payload_json=_option_entry_payload(),
            lifecycle_status=STATUS_FILLED,
            broker_status=STATUS_FILLED,
            filled_qty=qty,
            filled_avg_price=4.2,
            last_fill_at=_FILL_AT,
            last_fill_time_source="broker",
            last_refreshed_at=_FILL_AT,
            terminal_reason=STATUS_FILLED,
            terminal_at=_FILL_AT,
            position_kind="option",
            position_id=position_id,
        )
    )
    return position_id


def _insert_linked_shares(
    *,
    source: str = "long_term",
    direction: str = "long",
    qty: float = 3.0,
) -> int:
    position_id = db.insert_long_term_position(
        LongTermPosition(
            ticker="GOOGL",
            asset_class="stock",
            entry_price=174.5,
            entry_date=_FILL_AT,
            qty=qty,
            source=source,
            direction=direction,
            tp=190.0 if source == "swing_fallback" else None,
            sl=160.0 if source == "swing_fallback" else None,
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
            requested_limit_price=175.0,
            submitted_at=_ENTRY_AT,
            intent_payload_json=_share_entry_payload(
                source=source,
                direction=direction,
            ),
            lifecycle_status=STATUS_FILLED,
            broker_status=STATUS_FILLED,
            filled_qty=qty,
            filled_avg_price=174.5,
            last_fill_at=_FILL_AT,
            last_fill_time_source="broker",
            last_refreshed_at=_FILL_AT,
            terminal_reason=STATUS_FILLED,
            terminal_at=_FILL_AT,
            position_kind="long_term",
            position_id=position_id,
        )
    )
    return position_id


class InspectingBroker(FakeBroker):
    """Assert the immutable exit row exists before returning a snapshot."""

    def __init__(self, snapshot: OrderResult) -> None:
        super().__init__()
        self.snapshot = snapshot
        self.submit_calls = 0
        self.client_order_ids: list[str] = []

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
        assert client_order_id is not None
        pending = db.get_pending_order_by_client_order_id(client_order_id)
        assert pending is not None
        assert pending.lifecycle_status == "prepared"
        assert pending.order_role == "exit"
        self.client_order_ids.append(client_order_id)
        return dataclasses.replace(
            self.snapshot,
            client_order_id=client_order_id,
            symbol=symbol,
            qty=qty,
            side=side,
            order_type=order_type,
            limit_price=limit_price,
            time_in_force=time_in_force,
        )


class CountingFakeBroker(FakeBroker):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
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
        return super().submit_order(
            symbol,
            qty,
            side,
            order_type=order_type,
            limit_price=limit_price,
            time_in_force=time_in_force,
            client_order_id=client_order_id,
        )


class LostResponseBroker(CountingFakeBroker):
    def __init__(self, lookup_outcome: str) -> None:
        super().__init__()
        self.lookup_outcome = lookup_outcome
        self.lookup_calls = 0

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
        accepted = super().submit_order(
            symbol,
            qty,
            side,
            order_type=order_type,
            limit_price=limit_price,
            time_in_force=time_in_force,
            client_order_id=client_order_id,
        )
        assert accepted.ok
        return OrderResult(
            ok=False,
            status=STATUS_ERROR,
            client_order_id=client_order_id,
            symbol=symbol,
            reason="response lost after provider acceptance",
        )

    def get_order_by_client_order_id(
        self, client_order_id: str,
    ) -> OrderLookupResult:
        self.lookup_calls += 1
        if self.lookup_outcome == ORDER_LOOKUP_NOT_FOUND:
            return OrderLookupResult(
                outcome=ORDER_LOOKUP_NOT_FOUND,
                reason="HTTP 404",
            )
        if self.lookup_outcome == ORDER_LOOKUP_UNAVAILABLE:
            return OrderLookupResult(
                outcome=ORDER_LOOKUP_UNAVAILABLE,
                reason="lookup unavailable",
            )
        return super().get_order_by_client_order_id(client_order_id)


@pytest.mark.parametrize(
    ("shape", "position_kind", "expected_symbol", "expected_side", "expected_vehicle"),
    [
        ("option", "option", _OPTION_SYMBOL, "sell", "option_full"),
        ("long", "long_term", "GOOGL", "sell", "shares"),
        ("short", "long_term", "GOOGL", "buy", "shares"),
    ],
)
def test_exit_intent_exists_before_io_and_uses_exact_client_id_for_all_shapes(
    tmp_db: Path,
    shape: str,
    position_kind: str,
    expected_symbol: str,
    expected_side: str,
    expected_vehicle: str,
) -> None:
    if shape == "option":
        position_id = _insert_linked_option()
    else:
        position_id = _insert_linked_shares(
            source="swing_fallback" if shape == "short" else "long_term",
            direction="short" if shape == "short" else "long",
        )
    broker = InspectingBroker(
        OrderResult(
            ok=True,
            status=STATUS_NEW,
            order_id=f"exit-{shape}",
            raw_status="accepted",
            filled_qty=0.0,
        )
    )

    result = order_lifecycle.submit_position_exit(
        broker,
        position_kind=position_kind,
        position_id=position_id,
        requested_qty=1.0,
        requested_limit_price=5.0 if shape == "option" else 176.0,
        exit_reason="take_profit",
        submitted_at=_EXIT_AT,
    )

    assert result.ok is True
    assert broker.submit_calls == 1
    assert result.client_order_id == broker.client_order_ids[0]
    pending = db.get_pending_order(result.order_id)
    assert pending is not None
    assert pending.client_order_id == broker.client_order_ids[0]
    assert pending.order_role == "exit"
    assert pending.closes_position_kind == position_kind
    assert pending.closes_position_id == position_id
    assert pending.position_kind is None and pending.position_id is None
    assert pending.broker_symbol == expected_symbol
    assert pending.side == expected_side
    assert pending.vehicle == expected_vehicle
    assert json.loads(pending.intent_payload_json) == {
        "intent_kind": "position_exit",
        "exit_reason": "take_profit",
    }
    if position_kind == "option":
        position = db.get_option_position(position_id)
        assert position is not None and position.outcome == "open"
        assert position.closed_at is None and position.exit_price is None
    else:
        position = db.get_long_term_position(position_id)
        assert position is not None and position.status == "open"
        assert position.exit_date is None and position.exit_price is None


@pytest.mark.parametrize(
    (
        "status", "filled_qty", "filled_avg_price", "filled_at", "updated_at",
        "expected_source", "expected_time", "terminal",
    ),
    [
        (STATUS_NEW, 0.0, None, None, "2026-07-23T15:00:01+00:00", None, None, False),
        (
            STATUS_PARTIALLY_FILLED, 1.0, 4.4,
            "2026-07-23T15:00:02+00:00", None,
            "broker", datetime(2026, 7, 23, 15, 0, 2, tzinfo=UTC), False,
        ),
        (
            STATUS_FILLED, 2.0, 4.5, None, None,
            "observed", _EXIT_AT, True,
        ),
    ],
)
def test_immediate_exit_snapshots_store_fill_time_provenance_and_null_fees(
    tmp_db: Path,
    status: str,
    filled_qty: float,
    filled_avg_price: float | None,
    filled_at: str | None,
    updated_at: str | None,
    expected_source: str | None,
    expected_time: datetime | None,
    terminal: bool,
) -> None:
    position_id = _insert_linked_option()
    broker = InspectingBroker(
        OrderResult(
            ok=True,
            status=status,
            order_id=f"snapshot-{status}",
            raw_status=status,
            filled_qty=filled_qty,
            filled_avg_price=filled_avg_price,
            filled_at=filled_at,
            updated_at=updated_at,
        )
    )

    order_lifecycle.submit_position_exit(
        broker,
        position_kind="option",
        position_id=position_id,
        requested_qty=2.0,
        requested_limit_price=4.5,
        exit_reason="hold_deadline",
        submitted_at=_EXIT_AT,
    )

    pending = db.get_pending_order(f"snapshot-{status}")
    assert pending is not None
    assert pending.filled_qty == filled_qty
    assert pending.filled_avg_price == filled_avg_price
    assert pending.last_fill_time_source == expected_source
    assert pending.last_fill_at == expected_time
    assert pending.fees_dollars is None
    assert (pending.terminal_at is not None) is terminal
    position = db.get_option_position(position_id)
    assert position is not None and position.outcome == "open"


def test_provider_updated_at_is_broker_provenance_and_malformed_time_falls_back(
    tmp_db: Path,
) -> None:
    first_id = _insert_linked_option()
    updated_at = "2026-07-23T15:00:03+00:00"
    order_lifecycle.submit_position_exit(
        InspectingBroker(OrderResult(
            ok=True,
            status=STATUS_PARTIALLY_FILLED,
            order_id="updated-at",
            raw_status=STATUS_PARTIALLY_FILLED,
            filled_qty=1.0,
            filled_avg_price=4.4,
            updated_at=updated_at,
        )),
        position_kind="option",
        position_id=first_id,
        requested_qty=2.0,
        requested_limit_price=4.5,
        exit_reason="stop_loss",
        submitted_at=_EXIT_AT,
    )
    first = db.get_pending_order("updated-at")
    assert first is not None
    assert first.last_fill_at == datetime.fromisoformat(updated_at)
    assert first.last_fill_time_source == "broker"

    second_id = _insert_linked_option()
    order_lifecycle.submit_position_exit(
        InspectingBroker(OrderResult(
            ok=True,
            status=STATUS_PARTIALLY_FILLED,
            order_id="malformed-fill-time",
            raw_status=STATUS_PARTIALLY_FILLED,
            filled_qty=1.0,
            filled_avg_price=4.4,
            filled_at="not-a-time",
            updated_at="also-not-a-time",
        )),
        position_kind="option",
        position_id=second_id,
        requested_qty=2.0,
        requested_limit_price=4.5,
        exit_reason="stop_loss",
        submitted_at=_EXIT_AT,
    )
    second = db.get_pending_order("malformed-fill-time")
    assert second is not None
    assert second.last_fill_at == _EXIT_AT
    assert second.last_fill_time_source == "observed"


@pytest.mark.parametrize(
    ("result", "expected_status"),
    [
        (
            OrderResult(
                ok=False,
                status=STATUS_REJECTED,
                reason="market closed",
            ),
            "abandoned",
        ),
        (
            OrderResult(
                ok=False,
                status=STATUS_REJECTED,
                order_id="bound-rejection",
                raw_status=STATUS_REJECTED,
                reason="market closed",
            ),
            STATUS_REJECTED,
        ),
        (
            OrderResult(
                ok=False,
                status=STATUS_ERROR,
                reason="response ambiguous",
            ),
            "prepared",
        ),
    ],
)
def test_rejected_bound_and_ambiguous_submit_outcomes_are_durable(
    tmp_db: Path,
    result: OrderResult,
    expected_status: str,
) -> None:
    position_id = _insert_linked_option()
    broker = InspectingBroker(result)

    returned = order_lifecycle.submit_position_exit(
        broker,
        position_kind="option",
        position_id=position_id,
        requested_qty=2.0,
        requested_limit_price=4.5,
        exit_reason="take_profit",
        submitted_at=_EXIT_AT,
    )

    pending = db.get_pending_order_by_client_order_id(
        broker.client_order_ids[0]
    )
    assert returned.status == result.status
    assert pending is not None and pending.lifecycle_status == expected_status
    assert pending.order_role == "exit"


def test_bind_failure_recovers_same_exit_row_without_resubmit(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    position_id = _insert_linked_option()
    broker = CountingFakeBroker()
    observed_results: list[bool] = []
    real_update = db.update_pending_order

    def fail_bind(pending_order_id: int, **fields: Any) -> None:
        if "broker_order_id" in fields:
            raise sqlite3.OperationalError("simulated bind failure")
        real_update(pending_order_id, **fields)

    monkeypatch.setattr(db, "update_pending_order", fail_bind)
    with pytest.raises(order_lifecycle.OrderIntentCaptureError, match="could not be bound"):
        order_lifecycle.submit_position_exit(
            broker,
            position_kind="option",
            position_id=position_id,
            requested_qty=2.0,
            requested_limit_price=4.5,
            exit_reason="take_profit",
            submitted_at=_EXIT_AT,
            broker_result_observer=observed_results.append,
        )
    assert observed_results == [True]
    (prepared,) = db.get_pending_exit_orders_for_position("option", position_id)
    assert prepared.lifecycle_status == "prepared"
    client_order_id = prepared.client_order_id
    assert client_order_id is not None

    monkeypatch.setattr(db, "update_pending_order", real_update)
    recovery = order_lifecycle.recover_prepared_order(
        broker,
        client_order_id,
        observed_at=_EXIT_AT,
    )

    assert recovery.action == order_lifecycle.RECOVERY_BOUND
    assert broker.submit_calls == 1
    bound = db.get_pending_order_by_client_order_id(client_order_id)
    assert bound is not None and bound.lifecycle_status == STATUS_NEW
    assert bound.order_role == "exit"


@pytest.mark.parametrize(
    ("lookup_outcome", "expected_action", "expected_status"),
    [
        (ORDER_LOOKUP_NOT_FOUND, order_lifecycle.RECOVERY_ABANDONED, "abandoned"),
        (ORDER_LOOKUP_UNAVAILABLE, order_lifecycle.RECOVERY_UNCHANGED, "prepared"),
    ],
)
def test_lost_response_recovery_never_resubmits_and_404_abandons(
    tmp_db: Path,
    lookup_outcome: str,
    expected_action: str,
    expected_status: str,
) -> None:
    position_id = _insert_linked_option()
    broker = LostResponseBroker(lookup_outcome)
    result = order_lifecycle.submit_position_exit(
        broker,
        position_kind="option",
        position_id=position_id,
        requested_qty=2.0,
        requested_limit_price=4.5,
        exit_reason="take_profit",
        submitted_at=_EXIT_AT,
    )
    assert result.status == STATUS_ERROR
    (prepared,) = db.get_pending_exit_orders_for_position("option", position_id)
    assert prepared.client_order_id is not None

    recovery = order_lifecycle.recover_prepared_order(
        broker,
        prepared.client_order_id,
        observed_at=_EXIT_AT,
    )

    assert recovery.action == expected_action
    assert broker.submit_calls == 1
    assert broker.lookup_calls == 1
    current = db.get_pending_order_by_client_order_id(prepared.client_order_id)
    assert current is not None and current.lifecycle_status == expected_status


@pytest.mark.parametrize("qty", [0.0, -1.0, float("nan"), float("inf"), 3.0])
def test_exit_submission_rejects_invalid_remaining_quantity(
    tmp_db: Path, qty: float,
) -> None:
    position_id = _insert_linked_option(qty=2.0)
    with pytest.raises(ValueError, match="requested_qty"):
        order_lifecycle.submit_position_exit(
            FakeBroker(),
            position_kind="option",
            position_id=position_id,
            requested_qty=qty,
            requested_limit_price=4.5,
            exit_reason="take_profit",
            submitted_at=_EXIT_AT,
        )
    assert db.get_pending_exit_orders_for_position("option", position_id) == []


def test_exit_submission_rejects_invalid_target_role_and_legacy_unlinked_position(
    tmp_db: Path,
) -> None:
    with pytest.raises(ValueError, match="position kind"):
        order_lifecycle.submit_position_exit(
            FakeBroker(),
            position_kind="trade",
            position_id=1,
            requested_qty=1.0,
            requested_limit_price=1.0,
            exit_reason="take_profit",
            submitted_at=_EXIT_AT,
        )
    with pytest.raises(ValueError, match="position_id"):
        order_lifecycle.submit_position_exit(
            FakeBroker(),
            position_kind="option",
            position_id=0,
            requested_qty=1.0,
            requested_limit_price=1.0,
            exit_reason="take_profit",
            submitted_at=_EXIT_AT,
        )

    legacy_id = db.insert_option_position(
        OptionPosition(
            symbol=_OPTION_SYMBOL,
            underlying="GOOGL",
            option_type="call",
            strike=190.0,
            expiry="2026-09-18",
            contracts=1.0,
            opened_at=_FILL_AT,
            premium_entry=4.2,
            outcome="open",
        )
    )
    with pytest.raises(ValueError, match="broker-tracked entry"):
        order_lifecycle.submit_position_exit(
            FakeBroker(),
            position_kind="option",
            position_id=legacy_id,
            requested_qty=1.0,
            requested_limit_price=4.5,
            exit_reason="take_profit",
            submitted_at=_EXIT_AT,
        )

    linked_id = _insert_linked_option()
    with pytest.raises(order_lifecycle.OrderIntentCaptureError, match="role"):
        order_lifecycle.prepare_order_intent(
            ticker="GOOGL",
            broker_symbol=_OPTION_SYMBOL,
            asset_class="stock",
            vehicle="option_full",
            target_position_kind="option",
            side="sell",
            requested_qty=1.0,
            requested_limit_price=4.5,
            submitted_at=_EXIT_AT,
            intent_payload={
                "intent_kind": "position_exit",
                "exit_reason": "take_profit",
            },
            order_role="liquidate",
            closes_position_kind="option",
            closes_position_id=linked_id,
        )
    with pytest.raises(order_lifecycle.OrderIntentCaptureError, match="prepared"):
        order_lifecycle.prepare_order_intent(
            ticker="GOOGL",
            broker_symbol=_OPTION_SYMBOL,
            asset_class="stock",
            vehicle="option_full",
            target_position_kind="option",
            side="sell",
            requested_qty=1.0,
            requested_limit_price=4.5,
            submitted_at=_EXIT_AT,
            intent_payload={
                "intent_kind": "position_exit",
                "exit_reason": "take_profit",
            },
            order_role="entry",
            closes_position_kind="option",
            closes_position_id=linked_id,
        )


def test_second_call_is_blocked_while_exit_attempt_is_ambiguous(tmp_db: Path) -> None:
    position_id = _insert_linked_option()
    broker = LostResponseBroker(ORDER_LOOKUP_UNAVAILABLE)
    order_lifecycle.submit_position_exit(
        broker,
        position_kind="option",
        position_id=position_id,
        requested_qty=2.0,
        requested_limit_price=4.5,
        exit_reason="stop_loss",
        submitted_at=_EXIT_AT,
    )

    with pytest.raises(ValueError, match="nonterminal exit attempt"):
        order_lifecycle.submit_position_exit(
            broker,
            position_kind="option",
            position_id=position_id,
            requested_qty=2.0,
            requested_limit_price=4.5,
            exit_reason="stop_loss",
            submitted_at=_EXIT_AT,
        )
    assert broker.submit_calls == 1


def test_emergency_direct_closers_remain_unwired_until_17a5(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def dormant_must_not_run(*_args: object, **_kwargs: object) -> OrderResult:
        raise AssertionError("emergency direct closers must remain unchanged")

    monkeypatch.setattr(order_lifecycle, "submit_position_exit", dormant_must_not_run)
    monkeypatch.setattr(
        options_execution.risk_of_ruin,
        "record_broker_result",
        lambda _ok: None,
    )
    option_order, _ = options_execution.close_option_position(
        FakeBroker(),
        OptionPosition(
            symbol=_OPTION_SYMBOL,
            underlying="GOOGL",
            option_type="call",
            strike=190.0,
            expiry="2026-09-18",
            contracts=1.0,
            opened_at=_FILL_AT,
            premium_entry=4.2,
        ),
        exit_price=4.5,
        now=_EXIT_AT,
        outcome="win",
    )
    long_order = long_term.close_long_term_position(
        FakeBroker(),
        LongTermPosition(
            ticker="GOOGL",
            asset_class="stock",
            entry_price=174.5,
            entry_date=_FILL_AT,
            qty=1.0,
        ),
        exit_price=176.0,
        now=_EXIT_AT,
        reason="trend_breakdown",
    )

    assert option_order.ok is True and long_order.ok is True
    assert db.get_pending_orders() == []
