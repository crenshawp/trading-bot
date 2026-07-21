"""Durable capture of accepted broker-order materialization intent.

This module is deliberately limited to submit-time capture. Later lifecycle
refresh and fill materialization remain separate capabilities; until their
final cutover, the existing eager position writers stay active.
"""

from __future__ import annotations

import json
import math
import sqlite3
import sys
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from trading_bot import db
from trading_bot.broker.base import (
    STATUS_CANCELED,
    STATUS_ERROR,
    STATUS_FILLED,
    STATUS_PARTIALLY_FILLED,
    STATUS_REJECTED,
    STATUS_UNKNOWN,
    OrderResult,
)
from trading_bot.models import (
    TERMINAL_PENDING_ORDER_STATUSES,
    VALID_PENDING_ORDER_STATUSES,
    PendingOrder,
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
        terminal_reason = broker_status or status
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
