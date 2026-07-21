"""Submit-time pending-order capture; all broker behavior is mocked."""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from trading_bot import db, long_term, order_lifecycle
from trading_bot import options_execution as oe
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
from trading_bot.broker.options import OptionContract

_ACCEPTED_AT = datetime(2026, 7, 20, 15, 1, tzinfo=UTC)
_SUBMITTED_AT = datetime(2026, 7, 20, 15, 0, 58, tzinfo=UTC)
_DEADLINE = datetime(2026, 8, 3, 20, 0, tzinfo=UTC)


class SnapshotBroker(FakeBroker):
    """Return one injected accepted submit snapshot without a live call."""

    def __init__(self, snapshot: OrderResult) -> None:
        super().__init__()
        self.snapshot = snapshot
        self.submit_calls = 0
        self.submitted_client_order_ids: list[str | None] = []

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
        self.submitted_client_order_ids.append(client_order_id)
        return dataclasses.replace(
            self.snapshot,
            symbol=symbol,
            qty=qty,
            side=side,
            order_type=order_type,
            limit_price=limit_price,
            time_in_force=time_in_force,
            client_order_id=client_order_id,
        )


class LostResponseBroker(FakeBroker):
    """Accept once internally, then surface a lost response to the caller."""

    def __init__(self, lookup_outcome: str = "found") -> None:
        super().__init__()
        self.lookup_outcome = lookup_outcome
        self.submit_calls = 0
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
        self.submit_calls += 1
        accepted = super().submit_order(
            symbol,
            qty,
            side,
            order_type=order_type,
            limit_price=limit_price,
            time_in_force=time_in_force,
            client_order_id=client_order_id,
        )
        assert accepted.ok is True
        return OrderResult(
            ok=False,
            status=STATUS_ERROR,
            reason="response lost after provider acceptance",
            client_order_id=client_order_id,
            symbol=symbol,
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
                reason="simulated lookup outage",
            )
        return super().get_order_by_client_order_id(client_order_id)


def _snapshot(
    *,
    order_id: str | None = "accepted-order-1",
    status: str = STATUS_NEW,
    raw_status: str = "accepted",
    filled_qty: float = 0.0,
    filled_avg_price: float | None = None,
) -> OrderResult:
    return OrderResult(
        ok=True,
        status=status,
        order_id=order_id,
        submitted_at=_SUBMITTED_AT.isoformat(),
        raw_status=raw_status,
        filled_qty=filled_qty,
        filled_avg_price=filled_avg_price,
    )


def _option_decision() -> oe.ExecutionDecision:
    contract = OptionContract(
        symbol="GOOGL260918C00190000",
        underlying="GOOGL",
        option_type="call",
        strike=190.0,
        expiry="2026-09-18",
        delta=0.68,
        theta=-0.04,
        vega=0.12,
        gamma=0.03,
        open_interest=500,
        bid=4.9,
        ask=5.1,
        mid=5.0,
    )
    return oe.ExecutionDecision(
        vehicle=oe.VEHICLE_OPTION_FULL,
        side="buy",
        symbol=contract.symbol,
        qty=2.0,
        est_cost=1_000.0,
        dollar_risk=1_000.0,
        contract=contract,
        reason="ok",
    )


@pytest.mark.parametrize(
    ("status", "raw_status", "filled_qty", "filled_avg", "expected_terminal"),
    [
        (STATUS_NEW, "accepted", 0.0, None, False),
        (STATUS_PARTIALLY_FILLED, "partially_filled", 1.0, 5.05, False),
        (STATUS_FILLED, "filled", 2.0, 5.0, True),
    ],
)
def test_option_capture_preserves_unfilled_partial_and_filled_submit_snapshots(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
    raw_status: str,
    filled_qty: float,
    filled_avg: float | None,
    expected_terminal: bool,
) -> None:
    def selection_must_not_run(*_args: object, **_kwargs: object) -> oe.ExecutionDecision:
        raise AssertionError("capture must use the selected decision")

    monkeypatch.setattr(oe, "choose_execution", selection_must_not_run)
    snapshot = _snapshot(
        status=status,
        raw_status=raw_status,
        filled_qty=filled_qty,
        filled_avg_price=filled_avg,
    )
    order, position_id = oe.execute_decision(
        SnapshotBroker(snapshot),
        _option_decision(),
        opened_at=_ACCEPTED_AT,
        signal_id=17,
        tp=205.0,
        sl=175.0,
        deadline=_DEADLINE,
    )

    assert order.ok is True and position_id is not None
    (pending,) = db.get_pending_orders()
    assert pending.broker_order_id == "accepted-order-1"
    assert pending.ticker == "GOOGL"
    assert pending.broker_symbol == "GOOGL260918C00190000"
    assert pending.asset_class == "stock"
    assert pending.vehicle == "option_full"
    assert pending.target_position_kind == "option"
    assert pending.requested_qty == 2.0
    assert pending.requested_limit_price == 5.1
    assert pending.submitted_at == _ACCEPTED_AT
    assert pending.signal_id == 17
    assert pending.lifecycle_status == status
    assert pending.broker_status == raw_status
    assert pending.filled_qty == filled_qty
    assert pending.filled_avg_price == filled_avg
    assert pending.last_refreshed_at == _ACCEPTED_AT
    assert (pending.terminal_at is not None) is expected_terminal
    assert (pending.terminal_reason is not None) is expected_terminal
    assert pending.position_id is None
    assert json.loads(pending.intent_payload_json) == {
        "intent_kind": "option",
        "option_type": "call",
        "strike": 190.0,
        "expiry": "2026-09-18",
        "multiplier": 100,
        "delta_entry": 0.68,
        "theta": -0.04,
        "vega": 0.12,
        "gamma": 0.03,
        "tp": 205.0,
        "sl": 175.0,
        "deadline": _DEADLINE.isoformat(),
    }

    # 02b intentionally leaves the legacy requested-value writer active.
    (legacy_position,) = db.get_open_option_positions()
    assert legacy_position.contracts == 2.0
    assert legacy_position.premium_entry == 5.0


def test_shares_fallback_capture_preserves_swing_intent(tmp_db: Path) -> None:
    decision = oe.ExecutionDecision(
        vehicle=oe.VEHICLE_SHARES,
        side="sell",
        symbol="META",
        qty=1.5,
        est_cost=750.0,
        dollar_risk=750.0,
        contract=None,
        reason="ok",
    )
    order, position_id = oe.execute_decision(
        SnapshotBroker(_snapshot(order_id="shares-fallback-1")),
        decision,
        opened_at=_ACCEPTED_AT,
        signal_id=23,
        tp=460.0,
        sl=510.0,
        deadline=_DEADLINE,
    )

    assert order.ok is True and position_id is not None
    pending = db.get_pending_order("shares-fallback-1")
    assert pending is not None
    assert pending.ticker == pending.broker_symbol == "META"
    assert pending.vehicle == "shares"
    assert pending.target_position_kind == "long_term"
    assert pending.side == "sell"
    assert pending.requested_limit_price == 500.0
    assert pending.signal_id == 23
    assert json.loads(pending.intent_payload_json) == {
        "intent_kind": "shares_fallback",
        "source": "swing_fallback",
        "direction": "short",
        "tp": 460.0,
        "sl": 510.0,
        "deadline": _DEADLINE.isoformat(),
    }
    (legacy_position,) = db.get_open_long_term_positions()
    assert legacy_position.source == "swing_fallback"
    assert legacy_position.direction == "short"


def test_long_term_capture_preserves_buy_and_hold_intent(tmp_db: Path) -> None:
    snapshot = _snapshot(
        order_id="long-term-1",
        status=STATUS_PARTIALLY_FILLED,
        raw_status="partially_filled",
        filled_qty=0.5,
        filled_avg_price=174.5,
    )
    order, position_id = long_term.submit_long_term_entry(
        SnapshotBroker(snapshot),
        ticker="GOOGL",
        asset_class="stock",
        qty=2.5,
        entry_price=175.0,
        now=_ACCEPTED_AT,
    )

    assert order.ok is True and position_id is not None
    pending = db.get_pending_order("long-term-1")
    assert pending is not None
    assert pending.ticker == pending.broker_symbol == "GOOGL"
    assert pending.vehicle == "shares"
    assert pending.target_position_kind == "long_term"
    assert pending.side == "buy"
    assert pending.requested_qty == 2.5
    assert pending.requested_limit_price == 175.0
    assert pending.lifecycle_status == "partially_filled"
    assert pending.filled_qty == 0.5
    assert pending.filled_avg_price == 174.5
    assert json.loads(pending.intent_payload_json) == {
        "intent_kind": "long_term",
        "source": "long_term",
        "direction": "long",
    }
    (legacy_position,) = db.get_open_long_term_positions()
    assert legacy_position.source == "long_term"
    assert legacy_position.qty == 2.5
    assert legacy_position.entry_price == 175.0


def test_duplicate_capture_is_idempotent_and_conflicting_intent_fails(
    tmp_db: Path,
) -> None:
    snapshot = _snapshot(order_id="retry-1")
    kwargs = {
        "ticker": "GOOGL",
        "broker_symbol": "GOOGL",
        "asset_class": "stock",
        "vehicle": "shares",
        "target_position_kind": "long_term",
        "side": "buy",
        "requested_qty": 2.5,
        "requested_limit_price": 175.0,
        "accepted_at": _ACCEPTED_AT,
        "intent_payload": {
            "intent_kind": "long_term",
            "source": "long_term",
            "direction": "long",
        },
    }

    first_id = order_lifecycle.capture_accepted_order(snapshot, **kwargs)
    retry_id = order_lifecycle.capture_accepted_order(snapshot, **kwargs)

    assert retry_id == first_id
    assert len(db.get_pending_orders()) == 1
    with pytest.raises(
        order_lifecycle.OrderIntentCaptureError,
        match="different immutable materialization intent",
    ):
        order_lifecycle.capture_accepted_order(snapshot, **{**kwargs, "ticker": "META"})
    assert len(db.get_pending_orders()) == 1


@pytest.mark.parametrize("vehicle", ["option", "shares_fallback", "long_term"])
def test_accepted_order_without_broker_id_is_not_reported_as_captured(
    tmp_db: Path,
    vehicle: str,
) -> None:
    broker = SnapshotBroker(_snapshot(order_id=None))
    with pytest.raises(
        order_lifecycle.OrderIntentCaptureError,
        match="without an order ID",
    ):
        if vehicle == "long_term":
            long_term.submit_long_term_entry(
                broker,
                ticker="GOOGL",
                asset_class="stock",
                qty=2.5,
                entry_price=175.0,
                now=_ACCEPTED_AT,
            )
        else:
            decision = _option_decision()
            if vehicle == "shares_fallback":
                decision = dataclasses.replace(
                    decision,
                    vehicle=oe.VEHICLE_SHARES,
                    side="buy",
                    symbol="GOOGL",
                    qty=2.5,
                    est_cost=437.5,
                    dollar_risk=437.5,
                    contract=None,
                )
            oe.execute_decision(broker, decision, opened_at=_ACCEPTED_AT)

    (prepared,) = db.get_pending_orders()
    assert prepared.lifecycle_status == "prepared"
    assert prepared.broker_order_id is None
    assert prepared.client_order_id is not None
    assert db.get_open_option_positions() == []
    assert db.get_open_long_term_positions() == []


def test_capture_persistence_failure_is_fail_loud_before_legacy_insert(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def disk_full(_pending: object) -> int:
        raise RuntimeError("disk full")

    monkeypatch.setattr(order_lifecycle.db, "insert_pending_order", disk_full)
    with pytest.raises(
        order_lifecycle.OrderIntentCaptureError,
        match="not durably prepared before submission: disk full",
    ):
        oe.execute_decision(
            SnapshotBroker(_snapshot(order_id="not-durable-1")),
            _option_decision(),
            opened_at=_ACCEPTED_AT,
        )

    assert db.get_pending_orders() == []
    assert db.get_open_option_positions() == []


@pytest.mark.parametrize(
    ("snapshot", "match"),
    [
        (_snapshot(filled_qty=-1.0), "invalid filled quantity"),
        (_snapshot(filled_qty=1.0), "incomplete fill snapshot"),
        (
            _snapshot(filled_qty=1.0, filled_avg_price=0.0),
            "invalid average fill price",
        ),
        (
            _snapshot(status=STATUS_PARTIALLY_FILLED, raw_status="partially_filled"),
            "has no usable fill",
        ),
        (_snapshot(status=STATUS_ERROR, raw_status="error"), "transport-error status"),
    ],
)
def test_malformed_accepted_snapshot_remains_prepared_for_lookup(
    tmp_db: Path,
    snapshot: OrderResult,
    match: str,
) -> None:
    with pytest.raises(order_lifecycle.OrderIntentCaptureError, match=match):
        oe.execute_decision(
            SnapshotBroker(snapshot),
            _option_decision(),
            opened_at=_ACCEPTED_AT,
        )
    (prepared,) = db.get_pending_orders()
    assert prepared.lifecycle_status == "prepared"
    assert prepared.broker_order_id is None
    assert db.get_open_option_positions() == []


@pytest.mark.parametrize("submitted_at", [None, "not-a-time", "2026-07-20T15:00:58"])
def test_broker_time_cannot_rewrite_prepared_submission_time(
    tmp_db: Path,
    capsys: pytest.CaptureFixture[str],
    submitted_at: str | None,
) -> None:
    snapshot = dataclasses.replace(_snapshot(), submitted_at=submitted_at)
    oe.execute_decision(
        SnapshotBroker(snapshot),
        _option_decision(),
        opened_at=_ACCEPTED_AT,
    )

    (pending,) = db.get_pending_orders()
    assert pending.submitted_at == _ACCEPTED_AT
    assert "broker submitted_at" not in capsys.readouterr().err


def test_unknown_submit_status_is_preserved_as_unknown(tmp_db: Path) -> None:
    snapshot = _snapshot(status="provider_new_state", raw_status="held")
    oe.execute_decision(
        SnapshotBroker(snapshot),
        _option_decision(),
        opened_at=_ACCEPTED_AT,
    )
    (pending,) = db.get_pending_orders()
    assert pending.lifecycle_status == "unknown"
    assert pending.broker_status == "held"


def test_nonaccepted_result_cannot_be_captured(tmp_db: Path) -> None:
    with pytest.raises(ValueError, match="only broker-accepted"):
        order_lifecycle.capture_accepted_order(
            dataclasses.replace(_snapshot(), ok=False),
            ticker="GOOGL",
            broker_symbol="GOOGL",
            asset_class="stock",
            vehicle="shares",
            target_position_kind="long_term",
            side="buy",
            requested_qty=2.5,
            requested_limit_price=175.0,
            accepted_at=_ACCEPTED_AT,
            intent_payload={
                "intent_kind": "long_term",
                "source": "long_term",
                "direction": "long",
            },
        )
    assert db.get_pending_orders() == []


def test_prepared_row_exists_before_submit_and_exact_client_id_binds(
    tmp_db: Path,
) -> None:
    class InspectingBroker(SnapshotBroker):
        observed_prepared = False

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
            assert client_order_id is not None
            prepared = db.get_pending_order_by_client_order_id(client_order_id)
            assert prepared is not None
            assert prepared.lifecycle_status == "prepared"
            assert prepared.broker_order_id is None
            self.observed_prepared = True
            return super().submit_order(
                symbol,
                qty,
                side,
                order_type=order_type,
                limit_price=limit_price,
                time_in_force=time_in_force,
                client_order_id=client_order_id,
            )

    broker = InspectingBroker(_snapshot(order_id="atomic-bind-1"))
    order, position_id = oe.execute_decision(
        broker,
        _option_decision(),
        opened_at=_ACCEPTED_AT,
    )

    assert broker.observed_prepared is True
    assert broker.submit_calls == 1
    (submitted_client_order_id,) = broker.submitted_client_order_ids
    assert submitted_client_order_id is not None
    assert submitted_client_order_id.startswith("tradingbot-")
    assert len(submitted_client_order_id) == 43
    assert submitted_client_order_id.replace("-", "").isalnum()
    assert order.client_order_id == submitted_client_order_id
    assert position_id is not None
    pending = db.get_pending_order_by_client_order_id(submitted_client_order_id)
    assert pending is not None
    assert pending.broker_order_id == "atomic-bind-1"
    assert pending.lifecycle_status == "new"


def test_db_bind_failure_after_acceptance_leaves_recoverable_prepared_row(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker = FakeBroker()
    original_update = order_lifecycle.db.update_pending_order

    def fail_bind(pending_order_id: int, **fields: object) -> None:
        if fields.get("broker_order_id") is not None:
            raise RuntimeError("simulated bind failure")
        original_update(pending_order_id, **fields)

    monkeypatch.setattr(order_lifecycle.db, "update_pending_order", fail_bind)
    with pytest.raises(
        order_lifecycle.OrderIntentCaptureError,
        match="could not be bound: simulated bind failure",
    ):
        oe.execute_decision(
            broker,
            _option_decision(),
            opened_at=_ACCEPTED_AT,
        )

    (prepared,) = db.get_pending_orders()
    assert prepared.lifecycle_status == "prepared"
    assert prepared.broker_order_id is None
    assert prepared.client_order_id is not None
    lookup = broker.get_order_by_client_order_id(prepared.client_order_id)
    assert lookup.order is not None
    assert lookup.order.order_id == "fake-1"
    assert db.get_open_option_positions() == []


def test_lost_accept_response_recovers_same_row_by_client_id_without_resubmit(
    tmp_db: Path,
) -> None:
    broker = LostResponseBroker()
    order, position_id = oe.execute_decision(
        broker,
        _option_decision(),
        opened_at=_ACCEPTED_AT,
    )
    assert order.status == STATUS_ERROR
    assert position_id is None
    (prepared,) = db.get_pending_orders()
    assert prepared.lifecycle_status == "prepared"
    assert prepared.client_order_id is not None
    prepared_id = prepared.id

    recovery = order_lifecycle.recover_prepared_order(
        broker,
        prepared.client_order_id,
        observed_at=_ACCEPTED_AT,
    )

    assert recovery.action == order_lifecycle.RECOVERY_BOUND
    assert recovery.broker_order_id == "fake-1"
    bound = db.get_pending_order_by_client_order_id(prepared.client_order_id)
    assert bound is not None
    assert bound.id == prepared_id
    assert bound.broker_order_id == "fake-1"
    assert bound.lifecycle_status == "new"
    assert broker.submit_calls == 1
    assert broker.lookup_calls == 1
    assert db.get_open_option_positions() == []


def test_lost_accept_response_then_not_found_abandons_without_resubmit(
    tmp_db: Path,
) -> None:
    broker = LostResponseBroker(ORDER_LOOKUP_NOT_FOUND)
    oe.execute_decision(
        broker,
        _option_decision(),
        opened_at=_ACCEPTED_AT,
    )
    (prepared,) = db.get_pending_orders()
    assert prepared.client_order_id is not None
    prepared_id = prepared.id

    recovery = order_lifecycle.recover_prepared_order(
        broker,
        prepared.client_order_id,
        observed_at=_ACCEPTED_AT,
    )

    assert recovery.action == order_lifecycle.RECOVERY_ABANDONED
    abandoned = db.get_pending_order_by_client_order_id(prepared.client_order_id)
    assert abandoned is not None
    assert abandoned.id == prepared_id
    assert abandoned.broker_order_id is None
    assert abandoned.lifecycle_status == "abandoned"
    assert abandoned.terminal_at == _ACCEPTED_AT
    assert "not found" in (abandoned.terminal_reason or "")
    assert broker.submit_calls == 1
    assert db.get_open_option_positions() == []


def test_unavailable_client_id_lookup_leaves_prepared_without_resubmit(
    tmp_db: Path,
) -> None:
    broker = LostResponseBroker(ORDER_LOOKUP_UNAVAILABLE)
    oe.execute_decision(
        broker,
        _option_decision(),
        opened_at=_ACCEPTED_AT,
    )
    (prepared,) = db.get_pending_orders()
    assert prepared.client_order_id is not None

    recovery = order_lifecycle.recover_prepared_order(
        broker,
        prepared.client_order_id,
        observed_at=_ACCEPTED_AT,
    )

    assert recovery.action == order_lifecycle.RECOVERY_UNCHANGED
    unchanged = db.get_pending_order_by_client_order_id(prepared.client_order_id)
    assert unchanged == prepared
    assert broker.submit_calls == 1
    assert broker.lookup_calls == 1


def test_legacy_client_id_recovery_is_refused_before_broker_lookup(
    tmp_db: Path,
) -> None:
    order_id = "legacy-bound-order"
    order_lifecycle.capture_accepted_order(
        _snapshot(order_id=order_id),
        ticker="GOOGL",
        broker_symbol="GOOGL",
        asset_class="stock",
        vehicle="shares",
        target_position_kind="long_term",
        side="buy",
        requested_qty=2.5,
        requested_limit_price=175.0,
        accepted_at=_ACCEPTED_AT,
        intent_payload={
            "intent_kind": "long_term",
            "source": "long_term",
            "direction": "long",
        },
    )
    client_order_id = f"legacy-broker-{order_id}"
    broker = LostResponseBroker()

    recovery = order_lifecycle.recover_prepared_order(
        broker,
        client_order_id,
        observed_at=_ACCEPTED_AT,
    )

    assert recovery.action == order_lifecycle.RECOVERY_REFUSED
    assert broker.lookup_calls == 0
    assert broker.submit_calls == 0


@pytest.mark.parametrize(
    ("snapshot", "expected_status", "expected_bound"),
    [
        (
            OrderResult(
                ok=False,
                status=STATUS_REJECTED,
                reason="local validation rejected",
            ),
            "abandoned",
            False,
        ),
        (
            dataclasses.replace(
                _snapshot(order_id="provider-rejected-1", status=STATUS_REJECTED),
                ok=False,
                reason="broker rejected",
                raw_status="rejected",
            ),
            "rejected",
            True,
        ),
    ],
)
def test_submission_rejections_preserve_v25_bound_unbound_invariants(
    tmp_db: Path,
    snapshot: OrderResult,
    expected_status: str,
    expected_bound: bool,
) -> None:
    order, position_id = oe.execute_decision(
        SnapshotBroker(snapshot),
        _option_decision(),
        opened_at=_ACCEPTED_AT,
    )

    assert order.ok is False
    assert position_id is None
    (pending,) = db.get_pending_orders()
    assert pending.lifecycle_status == expected_status
    assert (pending.broker_order_id is not None) is expected_bound
    assert pending.terminal_at == _ACCEPTED_AT
    assert pending.terminal_reason is not None
    assert db.get_open_option_positions() == []
