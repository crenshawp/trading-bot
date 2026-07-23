"""Restart-safe capture, refresh, and fill materialization for broker orders.

Entry and dormant exit paths durably prepare intent before submission and only
materialize broker-reported cumulative fills.  Scanner reconciliation refreshes
existing orders lookup-only, then replays every durable positive entry fill so
a process crash between either step is harmless and retryable.
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
from datetime import UTC, date, datetime
from typing import Any

from trading_bot import db
from trading_bot.broker.base import (
    ORDER_LOOKUP_FOUND,
    ORDER_LOOKUP_NOT_FOUND,
    ORDER_TYPE_LIMIT,
    STATUS_CANCELED,
    STATUS_ERROR,
    STATUS_FILLED,
    STATUS_NEW,
    STATUS_PARTIALLY_FILLED,
    STATUS_REJECTED,
    STATUS_UNKNOWN,
    TIF_DAY,
    Broker,
    OrderResult,
)
from trading_bot.models import (
    PENDING_ORDER_INTENT_VERSION,
    TERMINAL_PENDING_ORDER_STATUSES,
    VALID_PENDING_ORDER_INTENT_KINDS,
    VALID_PENDING_ORDER_STATUSES,
    LongTermPosition,
    OptionPosition,
    PendingOrder,
    is_recoverable_pending_order_client_id,
)


class OrderIntentCaptureError(RuntimeError):
    """An accepted broker order was not safely represented in the ledger."""


_IMMUTABLE_INTENT_FIELDS: tuple[str, ...] = (
    "broker_order_id",
    "order_role",
    "ticker",
    "broker_symbol",
    "asset_class",
    "vehicle",
    "target_position_kind",
    "closes_position_kind",
    "closes_position_id",
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

RECOVERY_REASON_MALFORMED = "malformed"
RECOVERY_REASON_NOT_FOUND = "not_found"
RECOVERY_REASON_STATE = "state"
RECOVERY_REASON_UNAVAILABLE = "unavailable"

REFRESH_UPDATED = "updated"
REFRESH_UNCHANGED = "unchanged"
REFRESH_UNAVAILABLE = "unavailable"
REFRESH_MALFORMED = "malformed"
REFRESH_STALE = "stale"
REFRESH_SKIPPED = "skipped"
REFRESH_FAILED = "failed"

_BOUND_BROKER_STATUSES: frozenset[str] = frozenset(
    {
        STATUS_NEW,
        STATUS_PARTIALLY_FILLED,
        STATUS_FILLED,
        STATUS_CANCELED,
        STATUS_REJECTED,
        STATUS_UNKNOWN,
    }
)
_NONTERMINAL_STATUS_RANK = {
    STATUS_UNKNOWN: 0,
    STATUS_NEW: 1,
    STATUS_PARTIALLY_FILLED: 2,
}


@dataclass(frozen=True)
class PreparedOrderRecoveryResult:
    """Outcome of one lookup-only prepared-order recovery attempt."""

    client_order_id: str
    action: str
    reason: str = ""
    broker_order_id: str | None = None
    reason_kind: str = ""


@dataclass(frozen=True)
class PendingOrderRefreshResult:
    """Durable outcome of one prepared or bound order refresh attempt."""

    pending_order_id: int | None
    client_order_id: str | None
    broker_order_id: str | None
    action: str
    previous_status: str
    lifecycle_status: str
    filled_qty: float
    reason: str = ""


@dataclass(frozen=True)
class _LifecycleSnapshot:
    lifecycle_status: str
    broker_status: str | None
    filled_qty: float
    filled_avg_price: float | None
    last_fill_at: datetime | None
    last_fill_time_source: str | None
    terminal_reason: str | None
    terminal_at: datetime | None


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


def _provider_fill_time(
    order: OrderResult,
    *,
    filled_qty: float,
    observed_at: datetime,
) -> tuple[datetime | None, str | None]:
    """Return usable provider fill/update time, else durable observation time."""
    if filled_qty == 0:
        return None, None
    for field, raw_value in (
        ("filled_at", order.filled_at),
        ("updated_at", order.updated_at),
    ):
        if raw_value is None:
            continue
        if not isinstance(raw_value, str):
            print(
                f"  order lifecycle: malformed broker {field}; "
                "using local observation time",
                file=sys.stderr,
            )
            continue
        try:
            parsed = datetime.fromisoformat(raw_value)
        except ValueError:
            print(
                f"  order lifecycle: malformed broker {field}; "
                "using local observation time",
                file=sys.stderr,
            )
            continue
        if parsed.tzinfo is None:
            print(
                f"  order lifecycle: timezone-less broker {field}; "
                "using local observation time",
                file=sys.stderr,
            )
            continue
        return parsed, "broker"
    return observed_at, "observed"


def _initial_lifecycle(
    order: OrderResult,
    *,
    requested_qty: float,
    observed_at: datetime,
) -> _LifecycleSnapshot:
    """Normalize a usable submit response without inventing broker truth."""
    filled_qty, filled_avg_price = _positive_fill_state(order)
    last_fill_at, last_fill_time_source = _provider_fill_time(
        order,
        filled_qty=filled_qty,
        observed_at=observed_at,
    )
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
        return _LifecycleSnapshot(
            lifecycle_status=status,
            broker_status=broker_status,
            filled_qty=filled_qty,
            filled_avg_price=filled_avg_price,
            last_fill_at=last_fill_at,
            last_fill_time_source=last_fill_time_source,
            terminal_reason=terminal_reason,
            terminal_at=observed_at,
        )
    return _LifecycleSnapshot(
        lifecycle_status=status,
        broker_status=broker_status,
        filled_qty=filled_qty,
        filled_avg_price=filled_avg_price,
        last_fill_at=last_fill_at,
        last_fill_time_source=last_fill_time_source,
        terminal_reason=None,
        terminal_at=None,
    )


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
    order_role: str = "entry",
    closes_position_kind: str | None = None,
    closes_position_id: int | None = None,
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
        order_role=order_role,
        closes_position_kind=closes_position_kind,
        closes_position_id=closes_position_id,
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
    snapshot = _initial_lifecycle(
        order,
        requested_qty=pending.requested_qty,
        observed_at=observed_at,
    )
    try:
        db.update_pending_order(
            pending.id,
            broker_order_id=order_id,
            lifecycle_status=snapshot.lifecycle_status,
            broker_status=snapshot.broker_status,
            filled_qty=snapshot.filled_qty,
            filled_avg_price=snapshot.filled_avg_price,
            last_fill_at=snapshot.last_fill_at,
            last_fill_time_source=snapshot.last_fill_time_source,
            last_refreshed_at=observed_at,
            terminal_reason=snapshot.terminal_reason,
            terminal_at=snapshot.terminal_at,
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
    order_role: str = "entry",
    closes_position_kind: str | None = None,
    closes_position_id: int | None = None,
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
        order_role=order_role,
        closes_position_kind=closes_position_kind,
        closes_position_id=closes_position_id,
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

    if order.client_order_id is None:
        # The submitted identity is locally certain even when a broker response
        # omits the echo. Preserve it so immediate handling can reload the exact
        # prepared/abandoned row without inferring identity from order content.
        order = dataclasses.replace(
            order,
            client_order_id=pending.client_order_id,
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


@dataclass(frozen=True)
class _PositionExitSubmission:
    ticker: str
    broker_symbol: str
    asset_class: str
    vehicle: str
    side: str
    held_qty: float


def _terminal_linked_entry_qty(position_kind: str, position_id: int) -> float:
    """Return terminal actual entry quantity for one broker-linked position."""
    conn = db.get_connection()
    try:
        rows = conn.execute(
            "SELECT lifecycle_status, broker_order_id, filled_qty, "
            "filled_avg_price, last_fill_at, last_fill_time_source "
            "FROM pending_orders WHERE order_role = 'entry' "
            "AND position_kind = ? AND position_id = ? ORDER BY id ASC",
            (position_kind, position_id),
        ).fetchall()
    finally:
        conn.close()
    if len(rows) != 1:
        raise ValueError(
            "exit target must link to exactly one broker-tracked entry order"
        )
    row = rows[0]
    if row["lifecycle_status"] not in TERMINAL_PENDING_ORDER_STATUSES:
        raise ValueError("exit target entry order must be terminal before exit")
    filled_qty = float(row["filled_qty"])
    if (
        not row["broker_order_id"]
        or not math.isfinite(filled_qty)
        or filled_qty <= 0
        or row["filled_avg_price"] is None
        or float(row["filled_avg_price"]) <= 0
        or row["last_fill_at"] is None
        or row["last_fill_time_source"] not in {"broker", "observed"}
    ):
        raise ValueError("exit target entry order lacks usable broker fill truth")
    return filled_qty


def _position_exit_submission(
    position_kind: str,
    position_id: int,
) -> _PositionExitSubmission:
    """Validate one open typed position and derive its exact closing shape."""
    if position_kind == "option":
        option_position = db.get_option_position(position_id)
        if option_position is None or option_position.outcome not in (None, "open"):
            raise ValueError("exit target option position must exist and be open")
        submission = _PositionExitSubmission(
            ticker=option_position.underlying,
            broker_symbol=option_position.symbol,
            asset_class="stock",
            vehicle=option_position.vehicle,
            side="sell",
            held_qty=float(option_position.contracts),
        )
    elif position_kind == "long_term":
        long_position = db.get_long_term_position(position_id)
        if long_position is None or long_position.status != "open":
            raise ValueError("exit target long-term position must exist and be open")
        submission = _PositionExitSubmission(
            ticker=long_position.ticker,
            broker_symbol=long_position.ticker,
            asset_class=long_position.asset_class,
            vehicle="shares",
            side="buy" if long_position.direction == "short" else "sell",
            held_qty=float(long_position.qty),
        )
    else:
        raise ValueError(f"invalid exit target position kind {position_kind!r}")
    if not math.isfinite(submission.held_qty) or submission.held_qty <= 0:
        raise ValueError("exit target position must carry a positive holding")
    linked_qty = _terminal_linked_entry_qty(position_kind, position_id)
    if not _same_number(linked_qty, submission.held_qty):
        raise ValueError(
            "exit target position quantity does not match terminal entry truth"
        )
    return submission


def submit_position_exit(
    broker: Broker,
    *,
    position_kind: str,
    position_id: int,
    requested_qty: float,
    requested_limit_price: float | None,
    exit_reason: str,
    submitted_at: datetime,
    order_type: str = ORDER_TYPE_LIMIT,
    time_in_force: str = TIF_DAY,
) -> OrderResult:
    """Dormantly prepare and submit one restart-safe typed position exit.

    The typed open position and its terminal entry fill are validated before a
    prepared ``order_role='exit'`` row is written.  Broker I/O occurs only after
    that immutable row exists, always with its exact client order ID.  This
    primitive is intentionally not called by any watcher or emergency path yet.
    """
    if position_id <= 0:
        raise ValueError("exit target position_id must be positive")
    if (
        isinstance(requested_qty, bool)
        or not math.isfinite(requested_qty)
        or requested_qty <= 0
    ):
        raise ValueError("exit requested_qty must be positive and finite")
    if not isinstance(exit_reason, str) or not exit_reason.strip():
        raise ValueError("exit_reason must be a non-empty string")
    if submitted_at.tzinfo is None:
        raise ValueError("exit submitted_at must be timezone-aware")
    submission = _position_exit_submission(position_kind, position_id)
    if requested_qty > submission.held_qty and not _same_number(
        requested_qty, submission.held_qty
    ):
        raise ValueError("exit requested_qty exceeds the linked open holding")
    existing_attempts = db.get_pending_exit_orders_for_position(
        position_kind, position_id
    )
    if any(attempt.terminal_at is None for attempt in existing_attempts):
        raise ValueError("exit target already has a nonterminal exit attempt")
    return submit_prepared_order(
        broker,
        ticker=submission.ticker,
        broker_symbol=submission.broker_symbol,
        asset_class=submission.asset_class,
        vehicle=submission.vehicle,
        target_position_kind=position_kind,
        side=submission.side,
        requested_qty=requested_qty,
        requested_limit_price=requested_limit_price,
        submitted_at=submitted_at,
        intent_payload={
            "intent_kind": "position_exit",
            "exit_reason": exit_reason.strip(),
        },
        order_type=order_type,
        time_in_force=time_in_force,
        order_role="exit",
        closes_position_kind=position_kind,
        closes_position_id=position_id,
    )


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
            reason_kind=RECOVERY_REASON_STATE,
        )
    pending = db.get_pending_order_by_client_order_id(client_order_id)
    if pending is None:
        return PreparedOrderRecoveryResult(
            client_order_id=client_order_id,
            action=RECOVERY_UNCHANGED,
            reason="prepared order not found in local ledger",
            reason_kind=RECOVERY_REASON_STATE,
        )
    if pending.lifecycle_status != "prepared":
        return PreparedOrderRecoveryResult(
            client_order_id=client_order_id,
            action=RECOVERY_UNCHANGED,
            reason=f"local lifecycle is already {pending.lifecycle_status}",
            broker_order_id=pending.broker_order_id,
            reason_kind=RECOVERY_REASON_STATE,
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
            reason_kind=RECOVERY_REASON_UNAVAILABLE,
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
            reason_kind=RECOVERY_REASON_NOT_FOUND,
        )
    if lookup.outcome != ORDER_LOOKUP_FOUND or lookup.order is None:
        return PreparedOrderRecoveryResult(
            client_order_id=client_order_id,
            action=RECOVERY_UNCHANGED,
            reason=lookup.reason or "client order ID lookup unavailable",
            reason_kind=RECOVERY_REASON_UNAVAILABLE,
        )
    try:
        _bind_prepared_snapshot(pending, lookup.order, observed_at=moment)
    except OrderIntentCaptureError as exc:
        print(f"  order lifecycle: recovery snapshot unusable ({exc})", file=sys.stderr)
        return PreparedOrderRecoveryResult(
            client_order_id=client_order_id,
            action=RECOVERY_UNCHANGED,
            reason=str(exc),
            reason_kind=RECOVERY_REASON_MALFORMED,
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


def _refresh_result(
    previous: PendingOrder,
    current: PendingOrder,
    *,
    action: str,
    reason: str = "",
) -> PendingOrderRefreshResult:
    return PendingOrderRefreshResult(
        pending_order_id=current.id,
        client_order_id=current.client_order_id,
        broker_order_id=current.broker_order_id,
        action=action,
        previous_status=previous.lifecycle_status,
        lifecycle_status=current.lifecycle_status,
        filled_qty=current.filled_qty,
        reason=reason,
    )


def _log_refresh_issue(order: PendingOrder, category: str, reason: str) -> None:
    identity = order.broker_order_id or order.client_order_id or "unidentified"
    print(
        f"  order lifecycle: refresh {category} for {identity} ({reason})",
        file=sys.stderr,
    )


def _current_pending_order(order: PendingOrder) -> PendingOrder | None:
    """Reload one caller-supplied row before performing external work."""
    if order.client_order_id is None:
        return None
    return db.get_pending_order_by_client_order_id(order.client_order_id)


def _normalize_refresh_snapshot(
    pending: PendingOrder,
    order: OrderResult,
    *,
    observed_at: datetime,
) -> _LifecycleSnapshot:
    """Validate one bound-order read and normalize its cumulative truth."""
    expected_order_id = pending.broker_order_id
    if order.order_id is not None and not isinstance(order.order_id, str):
        raise OrderIntentCaptureError("broker refresh returned a malformed order ID")
    returned_order_id = order.order_id.strip() if order.order_id is not None else ""
    if expected_order_id is None:
        raise OrderIntentCaptureError("bound refresh row has no broker order ID")
    if not returned_order_id:
        raise OrderIntentCaptureError("broker refresh response has no order ID")
    if returned_order_id != expected_order_id:
        raise OrderIntentCaptureError(
            f"broker refresh returned order ID {returned_order_id!r}, "
            f"expected {expected_order_id!r}"
        )
    if not isinstance(order.status, str) or order.status not in _BOUND_BROKER_STATUSES:
        raise OrderIntentCaptureError(
            f"broker refresh returned invalid lifecycle status {order.status!r}"
        )
    if order.raw_status is not None and not isinstance(order.raw_status, str):
        raise OrderIntentCaptureError("broker refresh returned a malformed raw status")
    if not isinstance(order.reason, str):
        raise OrderIntentCaptureError("broker refresh returned a malformed reason")
    if isinstance(order.filled_qty, bool) or isinstance(order.filled_avg_price, bool):
        raise OrderIntentCaptureError("broker refresh returned a malformed fill snapshot")
    raw_status = order.raw_status.strip() if order.raw_status else None
    if order.status == STATUS_UNKNOWN and raw_status is None:
        raise OrderIntentCaptureError(
            "broker refresh returned neither a recognized nor raw lifecycle status"
        )
    return _initial_lifecycle(
        order,
        requested_qty=pending.requested_qty,
        observed_at=observed_at,
    )


def _same_number(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def _apply_bound_refresh(
    pending: PendingOrder,
    order: OrderResult,
    *,
    observed_at: datetime,
) -> PendingOrderRefreshResult:
    try:
        snapshot = _normalize_refresh_snapshot(
            pending,
            order,
            observed_at=observed_at,
        )
    except OrderIntentCaptureError as exc:
        reason = str(exc)
        _log_refresh_issue(pending, REFRESH_MALFORMED, reason)
        return _refresh_result(
            pending,
            pending,
            action=REFRESH_MALFORMED,
            reason=reason,
        )

    quantity_is_equal = _same_number(snapshot.filled_qty, pending.filled_qty)
    if snapshot.filled_qty < pending.filled_qty and not quantity_is_equal:
        reason = (
            f"filled_qty regression {snapshot.filled_qty} < {pending.filled_qty}"
        )
        _log_refresh_issue(pending, REFRESH_STALE, reason)
        return _refresh_result(
            pending,
            pending,
            action=REFRESH_STALE,
            reason=reason,
        )
    if quantity_is_equal and pending.filled_qty > 0:
        assert pending.filled_avg_price is not None
        assert snapshot.filled_avg_price is not None
        if not _same_number(snapshot.filled_avg_price, pending.filled_avg_price):
            reason = (
                "average fill price changed without a cumulative quantity increase"
            )
            _log_refresh_issue(pending, REFRESH_MALFORMED, reason)
            return _refresh_result(
                pending,
                pending,
                action=REFRESH_MALFORMED,
                reason=reason,
            )

    if snapshot.lifecycle_status not in TERMINAL_PENDING_ORDER_STATUSES:
        previous_rank = _NONTERMINAL_STATUS_RANK[pending.lifecycle_status]
        incoming_rank = _NONTERMINAL_STATUS_RANK[snapshot.lifecycle_status]
        if incoming_rank < previous_rank:
            reason = (
                f"lifecycle regression {pending.lifecycle_status} -> "
                f"{snapshot.lifecycle_status}"
            )
            _log_refresh_issue(pending, REFRESH_STALE, reason)
            return _refresh_result(
                pending,
                pending,
                action=REFRESH_STALE,
                reason=reason,
            )

    effective_filled_qty = pending.filled_qty if quantity_is_equal else snapshot.filled_qty
    effective_avg_price = (
        pending.filled_avg_price if quantity_is_equal else snapshot.filled_avg_price
    )
    effective_last_fill_at = (
        pending.last_fill_at if quantity_is_equal else snapshot.last_fill_at
    )
    effective_last_fill_time_source = (
        pending.last_fill_time_source
        if quantity_is_equal
        else snapshot.last_fill_time_source
    )
    state_changed = (
        snapshot.lifecycle_status != pending.lifecycle_status
        or snapshot.broker_status != pending.broker_status
        or not quantity_is_equal
        or effective_avg_price != pending.filled_avg_price
        or snapshot.terminal_reason != pending.terminal_reason
        or snapshot.terminal_at != pending.terminal_at
    )
    if not state_changed:
        return _refresh_result(
            pending,
            pending,
            action=REFRESH_UNCHANGED,
            reason="repeated cumulative broker snapshot",
        )

    assert pending.id is not None
    try:
        db.update_pending_order(
            pending.id,
            lifecycle_status=snapshot.lifecycle_status,
            broker_status=snapshot.broker_status,
            filled_qty=effective_filled_qty,
            filled_avg_price=effective_avg_price,
            last_fill_at=effective_last_fill_at,
            last_fill_time_source=effective_last_fill_time_source,
            last_refreshed_at=observed_at,
            terminal_reason=snapshot.terminal_reason,
            terminal_at=snapshot.terminal_at,
        )
    except (sqlite3.Error, TypeError, ValueError) as exc:
        reason = f"durable lifecycle update failed: {exc}"
        _log_refresh_issue(pending, REFRESH_FAILED, reason)
        return _refresh_result(
            pending,
            pending,
            action=REFRESH_FAILED,
            reason=reason,
        )
    current = dataclasses.replace(
        pending,
        lifecycle_status=snapshot.lifecycle_status,
        broker_status=snapshot.broker_status,
        filled_qty=effective_filled_qty,
        filled_avg_price=effective_avg_price,
        last_fill_at=effective_last_fill_at,
        last_fill_time_source=effective_last_fill_time_source,
        last_refreshed_at=observed_at,
        terminal_reason=snapshot.terminal_reason,
        terminal_at=snapshot.terminal_at,
    )
    return _refresh_result(pending, current, action=REFRESH_UPDATED)


def _refresh_prepared_order(
    broker: Broker,
    pending: PendingOrder,
    *,
    observed_at: datetime,
) -> PendingOrderRefreshResult:
    client_order_id = pending.client_order_id
    assert client_order_id is not None
    try:
        recovery = recover_prepared_order(
            broker,
            client_order_id,
            observed_at=observed_at,
        )
    except (OrderIntentCaptureError, sqlite3.Error, TypeError, ValueError) as exc:
        reason = f"prepared-order recovery failed: {exc}"
        _log_refresh_issue(pending, REFRESH_FAILED, reason)
        return _refresh_result(
            pending,
            pending,
            action=REFRESH_FAILED,
            reason=reason,
        )
    current = db.get_pending_order_by_client_order_id(client_order_id) or pending
    if recovery.action in (RECOVERY_BOUND, RECOVERY_ABANDONED):
        return _refresh_result(pending, current, action=REFRESH_UPDATED)
    if recovery.reason_kind == RECOVERY_REASON_UNAVAILABLE:
        _log_refresh_issue(pending, REFRESH_UNAVAILABLE, recovery.reason)
        return _refresh_result(
            pending,
            current,
            action=REFRESH_UNAVAILABLE,
            reason=recovery.reason,
        )
    if recovery.reason_kind == RECOVERY_REASON_MALFORMED:
        _log_refresh_issue(pending, REFRESH_MALFORMED, recovery.reason)
        return _refresh_result(
            pending,
            current,
            action=REFRESH_MALFORMED,
            reason=recovery.reason,
        )
    action = REFRESH_SKIPPED if recovery.action == RECOVERY_REFUSED else REFRESH_UNCHANGED
    return _refresh_result(
        pending,
        current,
        action=action,
        reason=recovery.reason,
    )


def refresh_pending_order(
    broker: Broker,
    pending: PendingOrder,
    *,
    observed_at: datetime | None = None,
) -> PendingOrderRefreshResult:
    """Refresh one durable row without submitting or materializing a position."""
    current = _current_pending_order(pending)
    if current is None:
        return _refresh_result(
            pending,
            pending,
            action=REFRESH_SKIPPED,
            reason="pending order is no longer present in the durable ledger",
        )
    if current.lifecycle_status in TERMINAL_PENDING_ORDER_STATUSES:
        return _refresh_result(
            current,
            current,
            action=REFRESH_SKIPPED,
            reason=f"local lifecycle is already {current.lifecycle_status}",
        )
    moment = observed_at or datetime.now(UTC)
    if current.lifecycle_status == "prepared":
        return _refresh_prepared_order(
            broker,
            current,
            observed_at=moment,
        )
    broker_order_id = current.broker_order_id
    assert broker_order_id is not None
    try:
        snapshot = broker.get_order(broker_order_id)
    except Exception as exc:  # noqa: BLE001 - broker boundary must remain fail-soft
        reason = str(exc) or "broker order read raised"
        _log_refresh_issue(current, REFRESH_UNAVAILABLE, reason)
        return _refresh_result(
            current,
            current,
            action=REFRESH_UNAVAILABLE,
            reason=reason,
        )
    if not snapshot.ok or snapshot.status == STATUS_ERROR:
        reason = (
            snapshot.reason.strip()
            if isinstance(snapshot.reason, str) and snapshot.reason.strip()
            else "broker order read unavailable"
        )
        _log_refresh_issue(current, REFRESH_UNAVAILABLE, reason)
        return _refresh_result(
            current,
            current,
            action=REFRESH_UNAVAILABLE,
            reason=reason,
        )
    return _apply_bound_refresh(current, snapshot, observed_at=moment)


def refresh_pending_orders(
    broker: Broker,
    *,
    observed_at: datetime | None = None,
) -> list[PendingOrderRefreshResult]:
    """Refresh every nonterminal ledger row once, isolating per-row failures."""
    moment = observed_at or datetime.now(UTC)
    try:
        pending_orders = db.get_nonterminal_pending_orders()
    except sqlite3.Error as exc:
        print(
            f"  order lifecycle: pending-order refresh query failed ({exc})",
            file=sys.stderr,
        )
        return []
    results: list[PendingOrderRefreshResult] = []
    for pending in pending_orders:
        try:
            result = refresh_pending_order(broker, pending, observed_at=moment)
        except (sqlite3.Error, TypeError, ValueError, RuntimeError) as exc:
            reason = f"refresh attempt failed: {exc}"
            _log_refresh_issue(pending, REFRESH_FAILED, reason)
            result = _refresh_result(
                pending,
                pending,
                action=REFRESH_FAILED,
                reason=reason,
            )
        results.append(result)
    return results


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
    snapshot = _initial_lifecycle(
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
        lifecycle_status=snapshot.lifecycle_status,
        broker_status=snapshot.broker_status,
        filled_qty=snapshot.filled_qty,
        filled_avg_price=snapshot.filled_avg_price,
        last_fill_at=snapshot.last_fill_at,
        last_fill_time_source=snapshot.last_fill_time_source,
        last_refreshed_at=accepted_at,
        terminal_reason=snapshot.terminal_reason,
        terminal_at=snapshot.terminal_at,
    )
    return _insert_idempotently(pending)


# ── fill materialization (dormant until 02e cutover) ─────────────────────────
#
# Turns a durable pending-order row's cumulative broker fill into the correct
# option / long-term / shares-fallback position, idempotently.  Nothing here
# polls or submits: it reads the fill state that :func:`refresh_pending_order`
# already persisted and materializes it.  It stays unscheduled until 02e.

MATERIALIZE_CREATED = "created"
MATERIALIZE_ADOPTED = "adopted"
MATERIALIZE_UPDATED = "updated"
MATERIALIZE_UNCHANGED = "unchanged"
MATERIALIZE_SKIPPED = "skipped"
MATERIALIZE_FAILED = "failed"


class FillMaterializationError(RuntimeError):
    """A usable broker fill could not be truthfully materialized."""


@dataclass(frozen=True)
class FillMaterializationResult:
    """Durable outcome of one pending-order fill materialization attempt."""

    pending_order_id: int | None
    client_order_id: str | None
    broker_order_id: str | None
    action: str
    position_kind: str | None
    position_id: int | None
    filled_qty: float
    reason: str = ""


def _materialize_result(
    pending: PendingOrder,
    *,
    action: str,
    position_kind: str | None = None,
    position_id: int | None = None,
    reason: str = "",
) -> FillMaterializationResult:
    return FillMaterializationResult(
        pending_order_id=pending.id,
        client_order_id=pending.client_order_id,
        broker_order_id=pending.broker_order_id,
        action=action,
        position_kind=position_kind if position_kind is not None else pending.position_kind,
        position_id=position_id if position_id is not None else pending.position_id,
        filled_qty=pending.filled_qty,
        reason=reason,
    )


def _log_materialize_issue(pending: PendingOrder, reason: str) -> None:
    identity = pending.broker_order_id or pending.client_order_id or "unidentified"
    print(
        f"  order lifecycle: materialize issue for {identity} ({reason})",
        file=sys.stderr,
    )


def _decode_intent_payload(pending: PendingOrder) -> dict[str, Any]:
    """Decode and shallow-validate the immutable version-1 intent payload."""
    if pending.intent_payload_version != PENDING_ORDER_INTENT_VERSION:
        raise FillMaterializationError(
            "unsupported intent payload version "
            f"{pending.intent_payload_version}; expected {PENDING_ORDER_INTENT_VERSION}"
        )
    try:
        payload = json.loads(pending.intent_payload_json)
    except (TypeError, json.JSONDecodeError) as exc:
        raise FillMaterializationError("intent payload is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise FillMaterializationError("intent payload must be a JSON object")
    intent_kind = payload.get("intent_kind")
    if intent_kind not in VALID_PENDING_ORDER_INTENT_KINDS:
        raise FillMaterializationError(
            f"unsupported intent_kind {intent_kind!r}"
        )
    return payload


def _payload_number(payload: Mapping[str, Any], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FillMaterializationError(
            f"intent field {key!r} must be numeric or null"
        )
    number = float(value)
    if not math.isfinite(number):
        raise FillMaterializationError(
            f"intent field {key!r} must be finite or null"
        )
    return number


def _payload_deadline(payload: Mapping[str, Any]) -> datetime | None:
    raw = payload.get("deadline")
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise FillMaterializationError("intent deadline must be an ISO datetime or null")
    try:
        return datetime.fromisoformat(raw)
    except ValueError as exc:
        raise FillMaterializationError("intent deadline is not an ISO datetime") from exc


def _build_option_position(
    pending: PendingOrder,
    payload: Mapping[str, Any],
    *,
    filled_qty: float,
    filled_avg_price: float,
    opened_at: datetime,
) -> OptionPosition:
    """Build the option position from immutable intent + actual broker fill."""
    if payload.get("intent_kind") != "option":
        raise FillMaterializationError("option target requires an option intent")
    if pending.target_position_kind != "option" or pending.vehicle not in (
        "option_full", "option_undersized",
    ):
        raise FillMaterializationError("option intent requires an option target/vehicle")
    option_type = payload.get("option_type")
    if option_type not in ("call", "put"):
        raise FillMaterializationError("option intent option_type must be 'call' or 'put'")
    strike = _payload_number(payload, "strike")
    if strike is None or strike <= 0:
        raise FillMaterializationError("option intent strike must be positive")
    expiry = payload.get("expiry")
    if not isinstance(expiry, str) or not expiry.strip():
        raise FillMaterializationError("option intent expiry must be a date string")
    try:
        date.fromisoformat(expiry)
    except ValueError as exc:
        raise FillMaterializationError("option intent expiry must be an ISO date") from exc
    multiplier = payload.get("multiplier")
    if isinstance(multiplier, bool) or not isinstance(multiplier, int) or multiplier <= 0:
        raise FillMaterializationError("option intent multiplier must be a positive integer")
    return OptionPosition(
        symbol=pending.broker_symbol,
        underlying=pending.ticker,
        option_type=option_type,
        strike=strike,
        expiry=expiry,
        contracts=filled_qty,
        opened_at=opened_at,
        multiplier=multiplier,
        signal_id=pending.signal_id,
        order_id=pending.broker_order_id,
        premium_entry=filled_avg_price,
        delta_entry=_payload_number(payload, "delta_entry"),
        theta=_payload_number(payload, "theta"),
        vega=_payload_number(payload, "vega"),
        gamma=_payload_number(payload, "gamma"),
        tp=_payload_number(payload, "tp"),
        sl=_payload_number(payload, "sl"),
        deadline=_payload_deadline(payload),
        outcome="open",
        vehicle=pending.vehicle,
    )


def _build_long_term_position(
    pending: PendingOrder,
    payload: Mapping[str, Any],
    *,
    intent_kind: str,
    filled_qty: float,
    filled_avg_price: float,
    opened_at: datetime,
) -> LongTermPosition:
    """Build the long-term / shares-fallback position from intent + fill."""
    if intent_kind not in ("long_term", "shares_fallback"):
        raise FillMaterializationError(
            "long_term target requires a long_term or shares_fallback intent"
        )
    if pending.target_position_kind != "long_term" or pending.vehicle != "shares":
        raise FillMaterializationError(
            "share intent requires a long_term target and shares vehicle"
        )
    expected_source = "long_term" if intent_kind == "long_term" else "swing_fallback"
    source = payload.get("source")
    if source != expected_source:
        raise FillMaterializationError(
            f"{intent_kind} intent source must be {expected_source!r}"
        )
    direction = payload.get("direction")
    if direction not in ("long", "short"):
        raise FillMaterializationError("share intent direction must be 'long' or 'short'")
    expected_direction = "long" if pending.side == "buy" else "short"
    if direction != expected_direction:
        raise FillMaterializationError("share intent direction must match the submitted side")
    if intent_kind == "long_term" and (direction != "long" or pending.side != "buy"):
        raise FillMaterializationError("long_term intent must be a long buy order")
    is_fallback = intent_kind == "shares_fallback"
    return LongTermPosition(
        ticker=pending.ticker,
        asset_class=pending.asset_class,
        entry_price=filled_avg_price,
        entry_date=opened_at,
        qty=filled_qty,
        status="open",
        source=source,
        direction=direction,
        tp=_payload_number(payload, "tp") if is_fallback else None,
        sl=_payload_number(payload, "sl") if is_fallback else None,
        deadline=_payload_deadline(payload) if is_fallback else None,
    )


def _create_position(
    pending: PendingOrder,
    payload: Mapping[str, Any],
    intent_kind: str,
    *,
    filled_avg_price: float,
    observed_at: datetime,
) -> FillMaterializationResult:
    """Insert and atomically link the first materialized position."""
    assert pending.id is not None
    if pending.target_position_kind == "option":
        option_pos = _build_option_position(
            pending,
            payload,
            filled_qty=pending.filled_qty,
            filled_avg_price=filled_avg_price,
            opened_at=observed_at,
        )
        if not is_recoverable_pending_order_client_id(pending.client_order_id):
            position_id = db.adopt_legacy_option_position(pending.id, option_pos)
            if position_id is None:
                reason = (
                    "legacy option position identity is not uniquely provable; "
                    "skipped for manual review"
                )
                _log_materialize_issue(pending, reason)
                return _materialize_result(
                    pending,
                    action=MATERIALIZE_SKIPPED,
                    reason=reason,
                )
            return _materialize_result(
                pending,
                action=MATERIALIZE_ADOPTED,
                position_kind="option",
                position_id=position_id,
            )
        position_id = db.materialize_new_option_position(pending.id, option_pos)
        return _materialize_result(
            pending,
            action=MATERIALIZE_CREATED,
            position_kind="option",
            position_id=position_id,
        )
    if not is_recoverable_pending_order_client_id(pending.client_order_id):
        reason = (
            "legacy long-term/share position has no durable broker identity; "
            "skipped for manual review"
        )
        _log_materialize_issue(pending, reason)
        return _materialize_result(
            pending,
            action=MATERIALIZE_SKIPPED,
            reason=reason,
        )
    long_term_pos = _build_long_term_position(
        pending,
        payload,
        intent_kind=intent_kind,
        filled_qty=pending.filled_qty,
        filled_avg_price=filled_avg_price,
        opened_at=observed_at,
    )
    position_id = db.materialize_new_long_term_position(pending.id, long_term_pos)
    return _materialize_result(
        pending,
        action=MATERIALIZE_CREATED,
        position_kind="long_term",
        position_id=position_id,
    )


def _update_position(
    pending: PendingOrder,
    *,
    filled_avg_price: float,
) -> FillMaterializationResult:
    """Advance an already-linked position to a larger cumulative fill, or no-op."""
    assert pending.id is not None
    assert pending.position_id is not None
    kind = pending.position_kind
    if kind != pending.target_position_kind:
        raise FillMaterializationError(
            "linked position kind does not match the pending-order target"
        )
    try:
        if kind == "option":
            changed = db.update_option_position_fill(
                pending.id,
                pending.position_id,
                contracts=pending.filled_qty,
                premium_entry=filled_avg_price,
            )
        else:
            changed = db.update_long_term_position_fill(
                pending.id,
                pending.position_id,
                qty=pending.filled_qty,
                entry_price=filled_avg_price,
            )
    except ValueError as exc:
        raise FillMaterializationError(str(exc)) from exc
    if not changed:
        return _materialize_result(
            pending,
            action=MATERIALIZE_UNCHANGED,
            position_kind=kind,
            position_id=pending.position_id,
            reason="position already reflects the cumulative fill",
        )
    return _materialize_result(
        pending,
        action=MATERIALIZE_UPDATED,
        position_kind=kind,
        position_id=pending.position_id,
    )


def _materialize_current(
    pending: PendingOrder,
    *,
    observed_at: datetime,
) -> FillMaterializationResult:
    if pending.filled_qty == 0:
        return _materialize_result(
            pending,
            action=MATERIALIZE_SKIPPED,
            reason="no usable fill to materialize",
        )
    if not math.isfinite(pending.filled_qty) or pending.filled_qty < 0:
        raise FillMaterializationError("cumulative fill quantity is invalid")
    filled_avg_price = pending.filled_avg_price
    if (
        filled_avg_price is None
        or not math.isfinite(filled_avg_price)
        or filled_avg_price <= 0
    ):
        raise FillMaterializationError("usable fill is missing a positive average price")
    payload = _decode_intent_payload(pending)
    intent_kind = str(payload["intent_kind"])
    if pending.position_id is None:
        return _create_position(
            pending,
            payload,
            intent_kind,
            filled_avg_price=filled_avg_price,
            observed_at=observed_at,
        )
    return _update_position(pending, filled_avg_price=filled_avg_price)


def materialize_pending_order_fill(
    pending: PendingOrder,
    *,
    observed_at: datetime | None = None,
) -> FillMaterializationResult:
    """Materialize one durable row's cumulative fill into its position.

    Reloads the row first so replays and restarts act on durable truth, never a
    stale in-memory copy. Zero fill creates nothing; the first positive fill
    inserts and links the correct position; a later higher cumulative fill
    updates that same row; an exact snapshot is a no-op. Inconsistent intent or
    a regressed cumulative fill fails without mutating anything.
    """
    try:
        current = _current_pending_order(pending)
    except sqlite3.Error as exc:
        reason = f"durable materialization lookup failed: {exc}"
        _log_materialize_issue(pending, reason)
        return _materialize_result(pending, action=MATERIALIZE_FAILED, reason=reason)
    if current is None:
        return _materialize_result(
            pending,
            action=MATERIALIZE_SKIPPED,
            reason="pending order is no longer present in the durable ledger",
        )
    moment = observed_at or datetime.now(UTC)
    try:
        return _materialize_current(current, observed_at=moment)
    except FillMaterializationError as exc:
        reason = str(exc)
        _log_materialize_issue(current, reason)
        return _materialize_result(current, action=MATERIALIZE_FAILED, reason=reason)
    except (sqlite3.Error, TypeError, ValueError, RuntimeError) as exc:
        reason = f"durable materialization failed: {exc}"
        _log_materialize_issue(current, reason)
        return _materialize_result(current, action=MATERIALIZE_FAILED, reason=reason)


def materialize_submitted_order_fill(
    order: OrderResult,
    *,
    observed_at: datetime | None = None,
) -> FillMaterializationResult | None:
    """Opportunistically materialize a submit reply's already-durable fill.

    This helper never infers quantity or price from requested values.  A new or
    accepted-unfilled response therefore returns a skipped result with no
    position.  Lookup/materialization failure is fail-soft because the durable
    pending row remains available to the scanner reconciliation cycle.
    """
    try:
        pending = (
            db.get_pending_order_by_client_order_id(order.client_order_id)
            if order.client_order_id is not None
            else db.get_pending_order(order.order_id)
        )
    except sqlite3.Error as exc:
        print(
            f"  order lifecycle: immediate materialization lookup failed ({exc})",
            file=sys.stderr,
        )
        return None
    if pending is None:
        identity = order.order_id or order.client_order_id or "unidentified"
        print(
            f"  order lifecycle: no durable intent found for {identity}; "
            "immediate materialization deferred",
            file=sys.stderr,
        )
        return None
    return materialize_pending_order_fill(pending, observed_at=observed_at)


def materialize_pending_order_fills(
    *,
    observed_at: datetime | None = None,
) -> list[FillMaterializationResult]:
    """Materialize every fill-bearing row once, including terminal rows."""
    moment = observed_at or datetime.now(UTC)
    try:
        pending_orders = db.get_pending_orders_with_fills()
    except sqlite3.Error as exc:
        print(
            f"  order lifecycle: fill materialization query failed ({exc})",
            file=sys.stderr,
        )
        return []
    return [
        materialize_pending_order_fill(pending, observed_at=moment)
        for pending in pending_orders
    ]


@dataclass(frozen=True)
class PendingOrderReconciliationResult:
    """One ordered refresh-then-materialize scanner pass."""

    refreshes: tuple[PendingOrderRefreshResult, ...]
    materializations: tuple[FillMaterializationResult, ...]


def reconcile_pending_orders(
    broker: Broker,
    *,
    observed_at: datetime | None = None,
) -> PendingOrderReconciliationResult:
    """Refresh accepted orders, then replay all durable fills exactly once.

    Prepared rows are lookup-only through ``recover_prepared_order``; this
    function has no submission call.  Materialization is a separate second
    phase and still runs if the refresh phase encounters an unexpected durable
    read failure, allowing a prior terminal/full/partial snapshot to recover
    after a crash between lifecycle persistence and position materialization.
    """
    moment = observed_at or datetime.now(UTC)
    try:
        refreshes = refresh_pending_orders(broker, observed_at=moment)
    except Exception as exc:  # noqa: BLE001 - phase isolation preserves fill replay
        print(
            f"  order lifecycle: reconciliation refresh phase failed ({exc})",
            file=sys.stderr,
        )
        refreshes = []
    try:
        materializations = materialize_pending_order_fills(observed_at=moment)
    except Exception as exc:  # noqa: BLE001 - scanner cycle must remain fail-soft
        print(
            f"  order lifecycle: reconciliation materialization phase failed ({exc})",
            file=sys.stderr,
        )
        materializations = []
    return PendingOrderReconciliationResult(
        refreshes=tuple(refreshes),
        materializations=tuple(materializations),
    )
