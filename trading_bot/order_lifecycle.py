"""Durable capture of accepted broker-order materialization intent.

This module is deliberately limited to submit-time capture. Later lifecycle
refresh and fill materialization remain separate capabilities; until their
final cutover, the existing eager position writers stay active.
"""

from __future__ import annotations

import dataclasses
import json
import math
import sqlite3
import sys
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from trading_bot import db
from trading_bot.broker.base import (
    ORDER_LOOKUP_FOUND,
    ORDER_LOOKUP_NOT_FOUND,
    ORDER_TYPE_LIMIT,
    STATUS_CANCELED,
    STATUS_ERROR,
    STATUS_FILLED,
    STATUS_PARTIALLY_FILLED,
    STATUS_REJECTED,
    STATUS_UNKNOWN,
    TIF_DAY,
    Broker,
    OrderResult,
)
from trading_bot.models import (
    TERMINAL_PENDING_ORDER_STATUSES,
    VALID_PENDING_ORDER_STATUSES,
    PendingOrder,
    is_recoverable_pending_order_client_id,
)


class OrderIntentCaptureError(RuntimeError):
    """An accepted broker order was not safely represented in the ledger."""


_IMMUTABLE_INTENT_FIELDS: tuple[str, ...] = (
    "broker_order_id",
    "ticker",
    "broker_symbol",
    "asset_class",
    "vehicle",
    "target_position_kind",
    "side",
    "requested_qty",
    "requested_limit_price",
    "submitted_at",
    "signal_id",
    "intent_payload_version",
    "intent_payload_json",
)

_CLIENT_ORDER_ID_PREFIX = "tradingbot-"

RECOVERY_BOUND = "bound"
RECOVERY_ABANDONED = "abandoned"
RECOVERY_UNCHANGED = "unchanged"
RECOVERY_REFUSED = "refused"


@dataclass(frozen=True)
class PreparedOrderRecoveryResult:
    """Outcome of one lookup-only prepared-order recovery attempt."""

    client_order_id: str
    action: str
    reason: str = ""
    broker_order_id: str | None = None


def _immutable_intent(order: PendingOrder) -> tuple[object, ...]:
    return tuple(getattr(order, field) for field in _IMMUTABLE_INTENT_FIELDS)


def _broker_order_id(order: OrderResult) -> str:
    order_id = order.order_id.strip() if order.order_id is not None else ""
    if not order_id:
        raise OrderIntentCaptureError(
            "broker accepted the order without an order ID; "
            "materialization intent was not durably captured"
        )
    return order_id


def _submitted_at(order: OrderResult, fallback: datetime) -> datetime:
    """Prefer the broker timestamp, but fail soft to the known acceptance time."""
    if order.submitted_at is None:
        return fallback
    try:
        parsed = datetime.fromisoformat(order.submitted_at)
    except ValueError:
        print(
            "  order lifecycle: malformed broker submitted_at; "
            "using local acceptance time",
            file=sys.stderr,
        )
        return fallback
    if parsed.tzinfo is None:
        print(
            "  order lifecycle: timezone-less broker submitted_at; "
            "using local acceptance time",
            file=sys.stderr,
        )
        return fallback
    return parsed


def _positive_fill_state(order: OrderResult) -> tuple[float, float | None]:
    try:
        filled_qty = float(order.filled_qty)
    except (TypeError, ValueError) as exc:
        raise OrderIntentCaptureError(
            "accepted broker order returned a malformed filled quantity; "
            "materialization intent was not durably captured"
        ) from exc
    if not math.isfinite(filled_qty) or filled_qty < 0:
        raise OrderIntentCaptureError(
            "accepted broker order returned an invalid filled quantity; "
            "materialization intent was not durably captured"
        )

    filled_avg_price = order.filled_avg_price
    if filled_avg_price is not None:
        try:
            filled_avg_price = float(filled_avg_price)
        except (TypeError, ValueError) as exc:
            raise OrderIntentCaptureError(
                "accepted broker order returned a malformed average fill price; "
                "materialization intent was not durably captured"
            ) from exc
        if not math.isfinite(filled_avg_price) or filled_avg_price <= 0:
            raise OrderIntentCaptureError(
                "accepted broker order returned an invalid average fill price; "
                "materialization intent was not durably captured"
            )

    if (filled_qty > 0) != (filled_avg_price is not None):
        raise OrderIntentCaptureError(
            "accepted broker order returned an incomplete fill snapshot; "
            "materialization intent was not durably captured"
        )
    return filled_qty, filled_avg_price


def _initial_lifecycle(
    order: OrderResult,
    *,
    requested_qty: float,
    observed_at: datetime,
) -> tuple[str, str | None, float, float | None, str | None, datetime | None]:
    """Normalize a usable submit response without inventing broker truth."""
    filled_qty, filled_avg_price = _positive_fill_state(order)
    raw_status = order.raw_status.strip() if order.raw_status else None
    status = "expired" if raw_status and raw_status.lower() == "expired" else order.status
    if status == STATUS_ERROR:
        raise OrderIntentCaptureError(
            "an accepted broker order cannot carry transport-error status; "
            "materialization intent was not durably captured"
        )
    if status not in VALID_PENDING_ORDER_STATUSES:
        status = STATUS_UNKNOWN

    if filled_qty == 0 and status in (STATUS_PARTIALLY_FILLED, STATUS_FILLED):
        raise OrderIntentCaptureError(
            f"accepted broker order status {status!r} has no usable fill; "
            "materialization intent was not durably captured"
        )

    if status not in (STATUS_CANCELED, STATUS_REJECTED, "expired") and filled_qty > 0:
        status = (
            STATUS_FILLED
            if status == STATUS_FILLED or filled_qty >= requested_qty
            else STATUS_PARTIALLY_FILLED
        )

    broker_status = raw_status or order.status or None
    if status in TERMINAL_PENDING_ORDER_STATUSES:
        terminal_reason = order.reason.strip() or broker_status or status
        return (
            status,
            broker_status,
            filled_qty,
            filled_avg_price,
            terminal_reason,
            observed_at,
        )
    return status, broker_status, filled_qty, filled_avg_price, None, None


def _insert_idempotently(order: PendingOrder) -> int:
    """Insert once; an exact immutable-intent replay returns the original id."""
    try:
        return db.insert_pending_order(order)
    except sqlite3.IntegrityError as exc:
        try:
            existing = db.get_pending_order(order.broker_order_id)
        except Exception as lookup_exc:
            raise OrderIntentCaptureError(
                f"accepted broker order {order.broker_order_id!r} hit a ledger "
                "conflict and the durable row could not be verified"
            ) from lookup_exc
        if existing is None:
            raise OrderIntentCaptureError(
                f"accepted broker order {order.broker_order_id!r} was not "
                "durably captured"
            ) from exc
        if _immutable_intent(existing) != _immutable_intent(order):
            raise OrderIntentCaptureError(
                f"broker order ID {order.broker_order_id!r} is already captured "
                "with different immutable materialization intent"
            ) from exc
        if existing.id is None:
            raise OrderIntentCaptureError(
                f"accepted broker order {order.broker_order_id!r} has no durable row id"
            ) from exc
        return existing.id
    except Exception as exc:
        raise OrderIntentCaptureError(
            f"accepted broker order {order.broker_order_id!r} intent was not "
            f"durably captured: {exc}"
        ) from exc


def generate_client_order_id() -> str:
    """Return one Alpaca-safe, globally unique bot submission identity.

    The fixed ASCII prefix plus UUID hex is 43 characters, below Alpaca's
    48-character client-order-ID limit and contains only letters, digits, and a
    hyphen. A recovery path never calls this function.
    """
    return f"{_CLIENT_ORDER_ID_PREFIX}{uuid.uuid4().hex}"


def prepare_order_intent(
    *,
    ticker: str,
    broker_symbol: str,
    asset_class: str,
    vehicle: str,
    target_position_kind: str,
    side: str,
    requested_qty: float,
    requested_limit_price: float | None,
    submitted_at: datetime,
    intent_payload: Mapping[str, Any],
    signal_id: int | None = None,
    client_order_id: str | None = None,
) -> PendingOrder:
    """Persist immutable intent before any broker submission can occur."""
    resolved_client_order_id = client_order_id or generate_client_order_id()
    pending = PendingOrder(
        client_order_id=resolved_client_order_id,
        broker_order_id=None,
        ticker=ticker,
        broker_symbol=broker_symbol,
        asset_class=asset_class,
        vehicle=vehicle,
        target_position_kind=target_position_kind,
        side=side,
        requested_qty=requested_qty,
        requested_limit_price=requested_limit_price,
        submitted_at=submitted_at,
        signal_id=signal_id,
        intent_payload_json=json.dumps(intent_payload, sort_keys=True),
        lifecycle_status="prepared",
    )
    try:
        pending_id = db.insert_pending_order(pending)
    except Exception as exc:
        raise OrderIntentCaptureError(
            "order intent was not durably prepared before submission: "
            f"{exc}"
        ) from exc
    return dataclasses.replace(pending, id=pending_id)


def _bind_prepared_snapshot(
    pending: PendingOrder,
    order: OrderResult,
    *,
    observed_at: datetime,
) -> int:
    """Bind one usable cumulative broker snapshot to its prepared row."""
    if pending.id is None or pending.client_order_id is None:
        raise OrderIntentCaptureError("prepared order has no durable identity")
    order_id = _broker_order_id(order)
    if (
        order.client_order_id is not None
        and order.client_order_id != pending.client_order_id
    ):
        raise OrderIntentCaptureError(
            "broker response client order ID does not match prepared intent"
        )
    (
        lifecycle_status,
        broker_status,
        filled_qty,
        filled_avg_price,
        terminal_reason,
        terminal_at,
    ) = _initial_lifecycle(
        order,
        requested_qty=pending.requested_qty,
        observed_at=observed_at,
    )
    try:
        db.update_pending_order(
            pending.id,
            broker_order_id=order_id,
            lifecycle_status=lifecycle_status,
            broker_status=broker_status,
            filled_qty=filled_qty,
            filled_avg_price=filled_avg_price,
            last_refreshed_at=observed_at,
            terminal_reason=terminal_reason,
            terminal_at=terminal_at,
        )
    except Exception as exc:
        raise OrderIntentCaptureError(
            f"broker order {order_id!r} was accepted but its prepared intent "
            f"could not be bound: {exc}"
        ) from exc
    return pending.id


def _abandon_prepared_order(
    pending: PendingOrder,
    *,
    reason: str,
    observed_at: datetime,
) -> None:
    """Terminalize one definitively unbound prepared intent."""
    if pending.id is None:
        raise OrderIntentCaptureError("prepared order has no durable row id")
    terminal_reason = reason.strip() or "prepared order was not accepted"
    try:
        db.update_pending_order(
            pending.id,
            lifecycle_status="abandoned",
            last_refreshed_at=observed_at,
            terminal_reason=terminal_reason,
            terminal_at=observed_at,
        )
    except Exception as exc:
        raise OrderIntentCaptureError(
            "prepared order could not be marked abandoned: "
            f"{exc}"
        ) from exc


def submit_prepared_order(
    broker: Broker,
    *,
    ticker: str,
    broker_symbol: str,
    asset_class: str,
    vehicle: str,
    target_position_kind: str,
    side: str,
    requested_qty: float,
    requested_limit_price: float | None,
    submitted_at: datetime,
    intent_payload: Mapping[str, Any],
    signal_id: int | None = None,
    order_type: str = ORDER_TYPE_LIMIT,
    time_in_force: str = TIF_DAY,
) -> OrderResult:
    """Prepare, submit once with the same client ID, and bind a usable reply.

    Transport or malformed ambiguity intentionally leaves the row ``prepared``
    for lookup-only recovery. A definitive unbound rejection becomes terminal
    ``abandoned``; a rejection carrying a broker order ID binds as ``rejected``.
    No branch retries submission or generates a replacement identity.
    """
    pending = prepare_order_intent(
        ticker=ticker,
        broker_symbol=broker_symbol,
        asset_class=asset_class,
        vehicle=vehicle,
        target_position_kind=target_position_kind,
        side=side,
        requested_qty=requested_qty,
        requested_limit_price=requested_limit_price,
        submitted_at=submitted_at,
        signal_id=signal_id,
        intent_payload=intent_payload,
    )
    assert pending.client_order_id is not None
    try:
        order = broker.submit_order(
            broker_symbol,
            requested_qty,
            side,
            order_type=order_type,
            limit_price=requested_limit_price,
            time_in_force=time_in_force,
            client_order_id=pending.client_order_id,
        )
    except Exception as exc:  # noqa: BLE001 - broker boundary remains fail-soft
        print(f"  order lifecycle: broker submit error ({exc})", file=sys.stderr)
        return OrderResult(
            ok=False,
            status=STATUS_ERROR,
            symbol=broker_symbol,
            client_order_id=pending.client_order_id,
            reason=str(exc),
        )

    has_broker_order_id = bool(order.order_id and order.order_id.strip())
    if order.status == STATUS_REJECTED and not has_broker_order_id:
        _abandon_prepared_order(
            pending,
            reason=f"submission rejected: {order.reason or 'rejected'}",
            observed_at=submitted_at,
        )
        return order
    if not order.ok and order.status != STATUS_REJECTED:
        return order
    _bind_prepared_snapshot(pending, order, observed_at=submitted_at)
    return order


def recover_prepared_order(
    broker: Broker,
    client_order_id: str,
    *,
    observed_at: datetime | None = None,
) -> PreparedOrderRecoveryResult:
    """Recover one prepared intent by lookup only; this never submits."""
    if not is_recoverable_pending_order_client_id(client_order_id):
        return PreparedOrderRecoveryResult(
            client_order_id=client_order_id,
            action=RECOVERY_REFUSED,
            reason="synthetic legacy client order ID is not provider-recoverable",
        )
    pending = db.get_pending_order_by_client_order_id(client_order_id)
    if pending is None:
        return PreparedOrderRecoveryResult(
            client_order_id=client_order_id,
            action=RECOVERY_UNCHANGED,
            reason="prepared order not found in local ledger",
        )
    if pending.lifecycle_status != "prepared":
        return PreparedOrderRecoveryResult(
            client_order_id=client_order_id,
            action=RECOVERY_UNCHANGED,
            reason=f"local lifecycle is already {pending.lifecycle_status}",
            broker_order_id=pending.broker_order_id,
        )
    moment = observed_at or datetime.now(UTC)
    try:
        lookup = broker.get_order_by_client_order_id(client_order_id)
    except Exception as exc:  # noqa: BLE001 - broker boundary remains fail-soft
        print(
            f"  order lifecycle: client-ID lookup error ({exc})",
            file=sys.stderr,
        )
        return PreparedOrderRecoveryResult(
            client_order_id=client_order_id,
            action=RECOVERY_UNCHANGED,
            reason=str(exc),
        )
    if lookup.outcome == ORDER_LOOKUP_NOT_FOUND:
        _abandon_prepared_order(
            pending,
            reason=f"client order ID not found: {lookup.reason or 'HTTP 404'}",
            observed_at=moment,
        )
        return PreparedOrderRecoveryResult(
            client_order_id=client_order_id,
            action=RECOVERY_ABANDONED,
            reason=lookup.reason,
        )
    if lookup.outcome != ORDER_LOOKUP_FOUND or lookup.order is None:
        return PreparedOrderRecoveryResult(
            client_order_id=client_order_id,
            action=RECOVERY_UNCHANGED,
            reason=lookup.reason or "client order ID lookup unavailable",
        )
    try:
        _bind_prepared_snapshot(pending, lookup.order, observed_at=moment)
    except OrderIntentCaptureError as exc:
        print(f"  order lifecycle: recovery snapshot unusable ({exc})", file=sys.stderr)
        return PreparedOrderRecoveryResult(
            client_order_id=client_order_id,
            action=RECOVERY_UNCHANGED,
            reason=str(exc),
        )
    return PreparedOrderRecoveryResult(
        client_order_id=client_order_id,
        action=RECOVERY_BOUND,
        broker_order_id=lookup.order.order_id,
    )


def recover_prepared_orders(
    broker: Broker,
    *,
    observed_at: datetime | None = None,
) -> list[PreparedOrderRecoveryResult]:
    """Run lookup-only recovery for every prepared row in submission order."""
    client_order_ids: list[str] = []
    for order in db.get_nonterminal_pending_orders():
        if order.lifecycle_status != "prepared":
            continue
        client_order_id = order.client_order_id
        if client_order_id is not None:
            client_order_ids.append(client_order_id)
    return [
        recover_prepared_order(
            broker,
            client_order_id,
            observed_at=observed_at,
        )
        for client_order_id in client_order_ids
    ]


def capture_accepted_order(
    order: OrderResult,
    *,
    ticker: str,
    broker_symbol: str,
    asset_class: str,
    vehicle: str,
    target_position_kind: str,
    side: str,
    requested_qty: float,
    requested_limit_price: float | None,
    accepted_at: datetime,
    intent_payload: Mapping[str, Any],
    signal_id: int | None = None,
) -> int:
    """Durably capture one accepted submit snapshot and immutable intent.

    Duplicate calls with the same broker ID and immutable intent return the
    existing row ID. A conflicting replay or any inability to prove durable
    capture fails loudly so an accepted order is never reported as safely
    tracked when it is not.
    """
    if not order.ok:
        raise ValueError("only broker-accepted orders can be captured")
    order_id = _broker_order_id(order)
    (
        lifecycle_status,
        broker_status,
        filled_qty,
        filled_avg_price,
        terminal_reason,
        terminal_at,
    ) = _initial_lifecycle(
        order,
        requested_qty=requested_qty,
        observed_at=accepted_at,
    )
    pending = PendingOrder(
        broker_order_id=order_id,
        ticker=ticker,
        broker_symbol=broker_symbol,
        asset_class=asset_class,
        vehicle=vehicle,
        target_position_kind=target_position_kind,
        side=side,
        requested_qty=requested_qty,
        requested_limit_price=requested_limit_price,
        submitted_at=_submitted_at(order, accepted_at),
        signal_id=signal_id,
        intent_payload_json=json.dumps(intent_payload, sort_keys=True),
        lifecycle_status=lifecycle_status,
        broker_status=broker_status,
        filled_qty=filled_qty,
        filled_avg_price=filled_avg_price,
        last_refreshed_at=accepted_at,
        terminal_reason=terminal_reason,
        terminal_at=terminal_at,
    )
    return _insert_idempotently(pending)
