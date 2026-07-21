"""Dormant pending-order lifecycle refresh; all broker behavior is mocked."""

from __future__ import annotations

import dataclasses
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from trading_bot import db, order_lifecycle
from trading_bot.broker.base import (
    ORDER_LOOKUP_FOUND,
    ORDER_LOOKUP_NOT_FOUND,
    ORDER_LOOKUP_UNAVAILABLE,
    ORDER_TYPE_LIMIT,
    STATUS_CANCELED,
    STATUS_ERROR,
    STATUS_FILLED,
    STATUS_NEW,
    STATUS_PARTIALLY_FILLED,
    STATUS_REJECTED,
    STATUS_UNKNOWN,
    TIF_DAY,
    OrderLookupResult,
    OrderResult,
)
from trading_bot.broker.fake import FakeBroker
from trading_bot.models import PendingOrder

_SUBMITTED_AT = datetime(2026, 7, 21, 14, 31, tzinfo=UTC)
_REFRESHED_AT = datetime(2026, 7, 21, 14, 35, tzinfo=UTC)


class RefreshBroker(FakeBroker):
    """Serve injected order snapshots and count every broker method used."""

    def __init__(
        self,
        *,
        snapshots: dict[str, list[OrderResult]] | None = None,
        lookups: dict[str, OrderLookupResult] | None = None,
    ) -> None:
        super().__init__()
        self.snapshots = snapshots or {}
        self.lookups = lookups or {}
        self.get_order_calls: list[str] = []
        self.lookup_calls: list[str] = []
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
        return OrderResult(
            ok=False,
            status=STATUS_ERROR,
            symbol=symbol,
            qty=qty,
            side=side,
            order_type=order_type,
            limit_price=limit_price,
            time_in_force=time_in_force,
            client_order_id=client_order_id,
            reason="refresh must never submit",
        )

    def get_order(self, order_id: str) -> OrderResult:
        self.get_order_calls.append(order_id)
        snapshots = self.snapshots.get(order_id)
        if not snapshots:
            return OrderResult(
                ok=False,
                status=STATUS_ERROR,
                order_id=order_id,
                reason="HTTP 404 order not found",
            )
        if len(snapshots) == 1:
            return snapshots[0]
        return snapshots.pop(0)

    def get_order_by_client_order_id(
        self,
        client_order_id: str,
    ) -> OrderLookupResult:
        self.lookup_calls.append(client_order_id)
        return self.lookups.get(
            client_order_id,
            OrderLookupResult(
                outcome=ORDER_LOOKUP_UNAVAILABLE,
                reason="lookup fixture missing",
            ),
        )


def _intent_payload() -> str:
    return json.dumps(
        {
            "intent_kind": "long_term",
            "source": "long_term",
            "direction": "long",
        },
        sort_keys=True,
    )


def _insert_bound(
    broker_order_id: str,
    *,
    lifecycle_status: str = STATUS_NEW,
    broker_status: str | None = None,
    filled_qty: float = 0.0,
    filled_avg_price: float | None = None,
    last_refreshed_at: datetime | None = _SUBMITTED_AT,
) -> PendingOrder:
    pending = PendingOrder(
        client_order_id=f"tradingbot-{broker_order_id}",
        broker_order_id=broker_order_id,
        ticker="GOOGL",
        broker_symbol="GOOGL",
        asset_class="stock",
        vehicle="shares",
        target_position_kind="long_term",
        side="buy",
        requested_qty=3.0,
        requested_limit_price=175.0,
        submitted_at=_SUBMITTED_AT,
        intent_payload_json=_intent_payload(),
        lifecycle_status=lifecycle_status,
        broker_status=broker_status or lifecycle_status,
        filled_qty=filled_qty,
        filled_avg_price=filled_avg_price,
        last_refreshed_at=last_refreshed_at,
    )
    db.insert_pending_order(pending)
    inserted = db.get_pending_order(broker_order_id)
    assert inserted is not None
    return inserted


def _insert_prepared(client_order_id: str) -> PendingOrder:
    pending = PendingOrder(
        client_order_id=client_order_id,
        broker_order_id=None,
        ticker="GOOGL",
        broker_symbol="GOOGL",
        asset_class="stock",
        vehicle="shares",
        target_position_kind="long_term",
        side="buy",
        requested_qty=3.0,
        requested_limit_price=175.0,
        submitted_at=_SUBMITTED_AT,
        intent_payload_json=_intent_payload(),
        lifecycle_status="prepared",
    )
    db.insert_pending_order(pending)
    inserted = db.get_pending_order_by_client_order_id(client_order_id)
    assert inserted is not None
    return inserted


def _snapshot(
    broker_order_id: str,
    *,
    status: str,
    filled_qty: float = 0.0,
    filled_avg_price: float | None = None,
    raw_status: str | None = None,
    reason: str = "",
) -> OrderResult:
    return OrderResult(
        ok=True,
        status=status,
        reason=reason,
        order_id=broker_order_id,
        symbol="GOOGL",
        qty=3.0,
        filled_qty=filled_qty,
        filled_avg_price=filled_avg_price,
        side="buy",
        raw_status=raw_status or status,
    )


def test_multiple_partial_fills_then_full_fill_advance_cumulative_truth(
    tmp_db: Path,
) -> None:
    order_id = "multi-fill-1"
    pending = _insert_bound(order_id)
    broker = RefreshBroker(
        snapshots={
            order_id: [
                _snapshot(
                    order_id,
                    status=STATUS_PARTIALLY_FILLED,
                    filled_qty=1.0,
                    filled_avg_price=174.0,
                ),
                _snapshot(
                    order_id,
                    status=STATUS_PARTIALLY_FILLED,
                    filled_qty=2.0,
                    filled_avg_price=174.5,
                ),
                _snapshot(
                    order_id,
                    status=STATUS_FILLED,
                    filled_qty=3.0,
                    filled_avg_price=175.0,
                ),
            ]
        }
    )

    first = order_lifecycle.refresh_pending_order(
        broker, pending, observed_at=_REFRESHED_AT
    )
    second_pending = db.get_pending_order(order_id)
    assert second_pending is not None
    second = order_lifecycle.refresh_pending_order(
        broker,
        second_pending,
        observed_at=_REFRESHED_AT + timedelta(minutes=1),
    )
    third_pending = db.get_pending_order(order_id)
    assert third_pending is not None
    third = order_lifecycle.refresh_pending_order(
        broker,
        third_pending,
        observed_at=_REFRESHED_AT + timedelta(minutes=2),
    )

    assert [first.action, second.action, third.action] == [
        order_lifecycle.REFRESH_UPDATED,
        order_lifecycle.REFRESH_UPDATED,
        order_lifecycle.REFRESH_UPDATED,
    ]
    assert [first.filled_qty, second.filled_qty, third.filled_qty] == [1.0, 2.0, 3.0]
    filled = db.get_pending_order(order_id)
    assert filled is not None
    assert filled.lifecycle_status == STATUS_FILLED
    assert filled.filled_avg_price == 175.0
    assert filled.terminal_at == _REFRESHED_AT + timedelta(minutes=2)
    assert filled.position_id is None
    assert db.get_open_long_term_positions() == []
    assert broker.submit_calls == 0


def test_immediate_full_fill_is_terminal_without_materializing_position(
    tmp_db: Path,
) -> None:
    order_id = "full-fill-1"
    pending = _insert_bound(order_id)
    broker = RefreshBroker(
        snapshots={
            order_id: [
                _snapshot(
                    order_id,
                    status=STATUS_FILLED,
                    filled_qty=3.0,
                    filled_avg_price=176.0,
                )
            ]
        }
    )

    result = order_lifecycle.refresh_pending_order(
        broker, pending, observed_at=_REFRESHED_AT
    )

    assert result.action == order_lifecycle.REFRESH_UPDATED
    assert result.lifecycle_status == STATUS_FILLED
    refreshed = db.get_pending_order(order_id)
    assert refreshed is not None
    assert refreshed.filled_qty == 3.0
    assert refreshed.position_id is None
    assert broker.submit_calls == 0


@pytest.mark.parametrize(
    ("status", "raw_status", "expected_status"),
    [
        (STATUS_CANCELED, STATUS_CANCELED, STATUS_CANCELED),
        (STATUS_REJECTED, STATUS_REJECTED, STATUS_REJECTED),
        (STATUS_CANCELED, "expired", "expired"),
    ],
)
def test_zero_fill_terminal_states_create_no_position(
    tmp_db: Path,
    status: str,
    raw_status: str,
    expected_status: str,
) -> None:
    order_id = f"zero-{expected_status}"
    pending = _insert_bound(order_id)
    broker = RefreshBroker(
        snapshots={
            order_id: [
                _snapshot(
                    order_id,
                    status=status,
                    raw_status=raw_status,
                    reason=f"provider {raw_status}",
                )
            ]
        }
    )

    result = order_lifecycle.refresh_pending_order(
        broker, pending, observed_at=_REFRESHED_AT
    )

    assert result.lifecycle_status == expected_status
    terminal = db.get_pending_order(order_id)
    assert terminal is not None
    assert terminal.filled_qty == 0
    assert terminal.filled_avg_price is None
    assert terminal.position_id is None
    assert terminal.terminal_reason == f"provider {raw_status}"
    assert db.get_open_long_term_positions() == []


@pytest.mark.parametrize(
    ("status", "raw_status", "expected_status"),
    [
        (STATUS_CANCELED, STATUS_CANCELED, STATUS_CANCELED),
        (STATUS_REJECTED, STATUS_REJECTED, STATUS_REJECTED),
        (STATUS_CANCELED, "expired", "expired"),
    ],
)
def test_partial_fill_terminal_states_preserve_cumulative_fill_for_02d(
    tmp_db: Path,
    status: str,
    raw_status: str,
    expected_status: str,
) -> None:
    order_id = f"partial-{expected_status}"
    pending = _insert_bound(
        order_id,
        lifecycle_status=STATUS_PARTIALLY_FILLED,
        filled_qty=1.0,
        filled_avg_price=174.25,
    )
    broker = RefreshBroker(
        snapshots={
            order_id: [
                _snapshot(
                    order_id,
                    status=status,
                    raw_status=raw_status,
                    filled_qty=1.0,
                    filled_avg_price=174.25,
                )
            ]
        }
    )

    result = order_lifecycle.refresh_pending_order(
        broker, pending, observed_at=_REFRESHED_AT
    )

    assert result.lifecycle_status == expected_status
    terminal = db.get_pending_order(order_id)
    assert terminal is not None
    assert terminal.filled_qty == 1.0
    assert terminal.filled_avg_price == 174.25
    assert terminal.position_id is None
    assert db.get_open_long_term_positions() == []


def test_unavailable_and_malformed_reads_retain_state_with_distinct_results(
    tmp_db: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    unavailable_id = "unavailable-1"
    malformed_id = "malformed-1"
    unavailable = _insert_bound(unavailable_id)
    malformed = _insert_bound(malformed_id)
    broker = RefreshBroker(
        snapshots={
            unavailable_id: [
                OrderResult(
                    ok=False,
                    status=STATUS_ERROR,
                    order_id=unavailable_id,
                    reason="simulated outage",
                )
            ],
            malformed_id: [
                _snapshot(
                    malformed_id,
                    status=STATUS_PARTIALLY_FILLED,
                    filled_qty=1.0,
                    filled_avg_price=None,
                )
            ],
        }
    )

    unavailable_result = order_lifecycle.refresh_pending_order(broker, unavailable)
    malformed_result = order_lifecycle.refresh_pending_order(broker, malformed)

    assert unavailable_result.action == order_lifecycle.REFRESH_UNAVAILABLE
    assert unavailable_result.reason == "simulated outage"
    assert malformed_result.action == order_lifecycle.REFRESH_MALFORMED
    assert "incomplete fill snapshot" in malformed_result.reason
    assert db.get_pending_order(unavailable_id) == unavailable
    assert db.get_pending_order(malformed_id) == malformed
    stderr = capsys.readouterr().err
    assert "refresh unavailable" in stderr
    assert "refresh malformed" in stderr
    assert broker.submit_calls == 0


@pytest.mark.parametrize(
    ("incoming", "reason_fragment"),
    [
        (
            dataclasses.replace(
                _snapshot("malformed-fields", status=STATUS_NEW),
                order_id=7,  # type: ignore[arg-type] - deliberate malformed runtime value
            ),
            "malformed order ID",
        ),
        (
            dataclasses.replace(
                _snapshot("malformed-fields", status=STATUS_NEW),
                order_id=None,
            ),
            "no order ID",
        ),
        (
            _snapshot("wrong-order-id", status=STATUS_NEW),
            "expected 'malformed-fields'",
        ),
        (
            dataclasses.replace(
                _snapshot("malformed-fields", status=STATUS_NEW),
                status="prepared",
            ),
            "invalid lifecycle status",
        ),
        (
            dataclasses.replace(
                _snapshot("malformed-fields", status=STATUS_NEW),
                raw_status=4,  # type: ignore[arg-type] - deliberate malformed runtime value
            ),
            "malformed raw status",
        ),
        (
            dataclasses.replace(
                _snapshot("malformed-fields", status=STATUS_NEW),
                reason=None,  # type: ignore[arg-type] - deliberate malformed runtime value
            ),
            "malformed reason",
        ),
        (
            dataclasses.replace(
                _snapshot("malformed-fields", status=STATUS_NEW),
                filled_qty=True,
                filled_avg_price=1.0,
            ),
            "malformed fill snapshot",
        ),
        (
            dataclasses.replace(
                _snapshot("malformed-fields", status=STATUS_UNKNOWN),
                raw_status=None,
            ),
            "neither a recognized nor raw lifecycle status",
        ),
    ],
)
def test_malformed_bound_snapshot_shapes_never_change_durable_truth(
    tmp_db: Path,
    incoming: OrderResult,
    reason_fragment: str,
) -> None:
    order_id = "malformed-fields"
    pending = _insert_bound(order_id)

    result = order_lifecycle.refresh_pending_order(
        RefreshBroker(snapshots={order_id: [incoming]}),
        pending,
    )

    assert result.action == order_lifecycle.REFRESH_MALFORMED
    assert reason_fragment in result.reason
    assert db.get_pending_order(order_id) == pending


def test_repeated_and_regressed_snapshots_are_idempotent(
    tmp_db: Path,
) -> None:
    order_id = "stale-1"
    pending = _insert_bound(
        order_id,
        lifecycle_status=STATUS_PARTIALLY_FILLED,
        filled_qty=2.0,
        filled_avg_price=174.5,
    )
    broker = RefreshBroker(
        snapshots={
            order_id: [
                _snapshot(
                    order_id,
                    status=STATUS_PARTIALLY_FILLED,
                    filled_qty=2.0,
                    filled_avg_price=174.5,
                ),
                _snapshot(
                    order_id,
                    status=STATUS_PARTIALLY_FILLED,
                    filled_qty=1.0,
                    filled_avg_price=174.0,
                ),
            ]
        }
    )

    repeated = order_lifecycle.refresh_pending_order(
        broker, pending, observed_at=_REFRESHED_AT
    )
    unchanged = db.get_pending_order(order_id)
    assert unchanged is not None
    regressed = order_lifecycle.refresh_pending_order(
        broker,
        unchanged,
        observed_at=_REFRESHED_AT + timedelta(minutes=1),
    )

    assert repeated.action == order_lifecycle.REFRESH_UNCHANGED
    assert regressed.action == order_lifecycle.REFRESH_STALE
    assert "filled_qty regression" in regressed.reason
    assert db.get_pending_order(order_id) == pending
    assert broker.submit_calls == 0


@pytest.mark.parametrize(
    "incoming",
    [
        _snapshot(
            "inconsistent-1",
            status=STATUS_PARTIALLY_FILLED,
            filled_qty=1.0,
            filled_avg_price=175.0,
        ),
        _snapshot("inconsistent-1", status=STATUS_FILLED),
    ],
)
def test_inconsistent_average_or_status_never_overwrites_prior_truth(
    tmp_db: Path,
    incoming: OrderResult,
) -> None:
    order_id = "inconsistent-1"
    if incoming.status == STATUS_FILLED:
        pending = _insert_bound(order_id)
    else:
        pending = _insert_bound(
            order_id,
            lifecycle_status=STATUS_PARTIALLY_FILLED,
            filled_qty=1.0,
            filled_avg_price=174.0,
        )
    broker = RefreshBroker(snapshots={order_id: [incoming]})

    result = order_lifecycle.refresh_pending_order(broker, pending)

    assert result.action == order_lifecycle.REFRESH_MALFORMED
    assert db.get_pending_order(order_id) == pending


def test_normalized_status_regression_is_stale_even_at_equal_quantity(
    tmp_db: Path,
) -> None:
    order_id = "status-regression-1"
    pending = _insert_bound(order_id)
    incoming = _snapshot(
        order_id,
        status=STATUS_UNKNOWN,
        raw_status="provider_mystery",
    )

    result = order_lifecycle.refresh_pending_order(
        RefreshBroker(snapshots={order_id: [incoming]}),
        pending,
    )

    assert result.action == order_lifecycle.REFRESH_STALE
    assert "lifecycle regression" in result.reason
    assert db.get_pending_order(order_id) == pending


def test_bound_order_404_is_unavailable_not_terminal_or_empty(
    tmp_db: Path,
) -> None:
    order_id = "bound-404"
    pending = _insert_bound(order_id)
    broker = RefreshBroker()

    result = order_lifecycle.refresh_pending_order(
        broker, pending, observed_at=_REFRESHED_AT
    )

    assert result.action == order_lifecycle.REFRESH_UNAVAILABLE
    assert "404" in result.reason
    assert db.get_pending_order(order_id) == pending
    assert db.get_nonterminal_pending_orders() == [pending]
    assert broker.get_order_calls == [order_id]
    assert broker.lookup_calls == []
    assert broker.submit_calls == 0


def test_broker_exception_is_fail_soft_and_retains_bound_state(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order_id = "bound-raises"
    pending = _insert_bound(order_id)
    broker = RefreshBroker()

    def raise_read(_: str) -> OrderResult:
        raise RuntimeError("simulated broker exception")

    monkeypatch.setattr(broker, "get_order", raise_read)
    result = order_lifecycle.refresh_pending_order(broker, pending)

    assert result.action == order_lifecycle.REFRESH_UNAVAILABLE
    assert result.reason == "simulated broker exception"
    assert db.get_pending_order(order_id) == pending
    assert broker.submit_calls == 0


def test_durable_update_failure_is_reported_without_partial_state(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order_id = "update-fails"
    pending = _insert_bound(order_id)
    incoming = _snapshot(
        order_id,
        status=STATUS_PARTIALLY_FILLED,
        filled_qty=1.0,
        filled_avg_price=174.0,
    )

    def fail_update(_: int, **fields: object) -> None:
        assert fields["filled_qty"] == 1.0
        raise sqlite3.OperationalError("simulated durable write failure")

    monkeypatch.setattr(order_lifecycle.db, "update_pending_order", fail_update)
    result = order_lifecycle.refresh_pending_order(
        RefreshBroker(snapshots={order_id: [incoming]}),
        pending,
    )

    assert result.action == order_lifecycle.REFRESH_FAILED
    assert "simulated durable write failure" in result.reason
    assert db.get_pending_order(order_id) == pending


@pytest.mark.parametrize(
    ("outcome", "expected_action", "expected_status", "expected_bound"),
    [
        (
            ORDER_LOOKUP_FOUND,
            order_lifecycle.REFRESH_UPDATED,
            STATUS_NEW,
            True,
        ),
        (
            ORDER_LOOKUP_NOT_FOUND,
            order_lifecycle.REFRESH_UPDATED,
            "abandoned",
            False,
        ),
        (
            ORDER_LOOKUP_UNAVAILABLE,
            order_lifecycle.REFRESH_UNAVAILABLE,
            "prepared",
            False,
        ),
    ],
)
def test_prepared_refresh_delegates_lookup_only_recovery(
    tmp_db: Path,
    outcome: str,
    expected_action: str,
    expected_status: str,
    expected_bound: bool,
) -> None:
    client_order_id = f"tradingbot-prepared-{outcome}"
    pending = _insert_prepared(client_order_id)
    order = dataclasses.replace(
        _snapshot("recovered-1", status=STATUS_NEW),
        client_order_id=client_order_id,
    )
    lookup = OrderLookupResult(
        outcome=outcome,
        reason=("simulated lookup outage" if outcome == ORDER_LOOKUP_UNAVAILABLE else "HTTP 404"),
        order=order if outcome == ORDER_LOOKUP_FOUND else None,
    )
    broker = RefreshBroker(lookups={client_order_id: lookup})

    result = order_lifecycle.refresh_pending_order(
        broker, pending, observed_at=_REFRESHED_AT
    )

    assert result.action == expected_action
    assert result.lifecycle_status == expected_status
    refreshed = db.get_pending_order_by_client_order_id(client_order_id)
    assert refreshed is not None
    assert (refreshed.broker_order_id is not None) is expected_bound
    assert broker.lookup_calls == [client_order_id]
    assert broker.get_order_calls == []
    assert broker.submit_calls == 0


def test_prepared_malformed_lookup_snapshot_retains_prepared_state(
    tmp_db: Path,
) -> None:
    client_order_id = "tradingbot-prepared-malformed"
    pending = _insert_prepared(client_order_id)
    malformed = dataclasses.replace(
        _snapshot("prepared-malformed", status=STATUS_FILLED),
        client_order_id=client_order_id,
    )
    broker = RefreshBroker(
        lookups={
            client_order_id: OrderLookupResult(
                outcome=ORDER_LOOKUP_FOUND,
                order=malformed,
            )
        }
    )

    result = order_lifecycle.refresh_pending_order(broker, pending)

    assert result.action == order_lifecycle.REFRESH_MALFORMED
    assert result.lifecycle_status == "prepared"
    assert db.get_pending_order_by_client_order_id(client_order_id) == pending
    assert broker.submit_calls == 0


def test_prepared_terminal_write_failure_is_reported_without_resubmit(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client_order_id = "tradingbot-prepared-write-failure"
    pending = _insert_prepared(client_order_id)
    broker = RefreshBroker(
        lookups={
            client_order_id: OrderLookupResult(
                outcome=ORDER_LOOKUP_NOT_FOUND,
                reason="HTTP 404",
            )
        }
    )

    def fail_update(_: int, **fields: object) -> None:
        assert fields["lifecycle_status"] == "abandoned"
        raise sqlite3.OperationalError("simulated abandon write failure")

    monkeypatch.setattr(order_lifecycle.db, "update_pending_order", fail_update)
    result = order_lifecycle.refresh_pending_order(broker, pending)

    assert result.action == order_lifecycle.REFRESH_FAILED
    assert "simulated abandon write failure" in result.reason
    assert db.get_pending_order_by_client_order_id(client_order_id) == pending
    assert broker.submit_calls == 0


def test_terminal_or_missing_local_rows_are_skipped_without_broker_reads(
    tmp_db: Path,
) -> None:
    order_id = "already-terminal"
    pending = _insert_bound(order_id)
    broker = RefreshBroker(
        snapshots={
            order_id: [
                _snapshot(
                    order_id,
                    status=STATUS_FILLED,
                    filled_qty=3.0,
                    filled_avg_price=175.0,
                )
            ]
        }
    )
    order_lifecycle.refresh_pending_order(broker, pending)
    terminal = db.get_pending_order(order_id)
    assert terminal is not None

    skipped_terminal = order_lifecycle.refresh_pending_order(broker, terminal)
    missing = dataclasses.replace(terminal, client_order_id="tradingbot-not-local")
    skipped_missing = order_lifecycle.refresh_pending_order(broker, missing)

    assert skipped_terminal.action == order_lifecycle.REFRESH_SKIPPED
    assert skipped_missing.action == order_lifecycle.REFRESH_SKIPPED
    assert broker.get_order_calls == [order_id]
    assert broker.lookup_calls == []
    assert broker.submit_calls == 0


def test_batch_refresh_returns_one_result_per_row_without_submissions(
    tmp_db: Path,
) -> None:
    prepared_client_id = "tradingbot-batch-prepared"
    _insert_prepared(prepared_client_id)
    partial = _insert_bound("batch-partial")
    unavailable = _insert_bound("batch-unavailable")
    recovered = dataclasses.replace(
        _snapshot("batch-recovered", status=STATUS_NEW),
        client_order_id=prepared_client_id,
    )
    broker = RefreshBroker(
        snapshots={
            "batch-partial": [
                _snapshot(
                    "batch-partial",
                    status=STATUS_PARTIALLY_FILLED,
                    filled_qty=1.0,
                    filled_avg_price=174.0,
                )
            ],
            "batch-unavailable": [
                OrderResult(
                    ok=False,
                    status=STATUS_ERROR,
                    order_id="batch-unavailable",
                    reason="batch outage",
                )
            ],
        },
        lookups={
            prepared_client_id: OrderLookupResult(
                outcome=ORDER_LOOKUP_FOUND,
                order=recovered,
            )
        },
    )

    results = order_lifecycle.refresh_pending_orders(
        broker, observed_at=_REFRESHED_AT
    )
    by_client_id = {result.client_order_id: result for result in results}

    assert len(results) == 3
    assert by_client_id[prepared_client_id].lifecycle_status == STATUS_NEW
    assert by_client_id[partial.client_order_id].filled_qty == 1.0
    assert by_client_id[unavailable.client_order_id].action == (
        order_lifecycle.REFRESH_UNAVAILABLE
    )
    assert broker.lookup_calls == [prepared_client_id]
    assert set(broker.get_order_calls) == {"batch-partial", "batch-unavailable"}
    assert broker.submit_calls == 0
    assert db.get_open_long_term_positions() == []
