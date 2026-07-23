"""Durable idempotency tests for untyped emergency broker-position closes."""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from trading_bot import db, order_lifecycle
from trading_bot import risk_of_ruin as ror
from trading_bot.broker.base import (
    ORDER_LOOKUP_FOUND,
    ORDER_LOOKUP_UNAVAILABLE,
    ORDER_TYPE_LIMIT,
    STATUS_CANCELED,
    STATUS_ERROR,
    STATUS_FILLED,
    STATUS_NEW,
    TIF_DAY,
    OrderLookupResult,
    OrderResult,
)
from trading_bot.broker.fake import FakeBroker

_TS = datetime(2026, 7, 23, 14, 0, tzinfo=UTC)


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def __call__(self, title: str, message: str) -> bool:
        self.calls.append((title, message))
        return True


class _RecordingBroker(FakeBroker):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.submissions: list[tuple[str, float, str, str | None]] = []

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
        self.submissions.append((symbol, qty, side, client_order_id))
        return super().submit_order(
            symbol,
            qty,
            side,
            order_type=order_type,
            limit_price=limit_price,
            time_in_force=time_in_force,
            client_order_id=client_order_id,
        )


class _ClosingBroker(_RecordingBroker):
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
        order = super().submit_order(
            symbol,
            qty,
            side,
            order_type=order_type,
            limit_price=limit_price,
            time_in_force=time_in_force,
            client_order_id=client_order_id,
        )
        if order.ok:
            self._positions.pop(symbol, None)
        return order


class _LostResponseBroker(_RecordingBroker):
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
        super().submit_order(
            symbol,
            qty,
            side,
            order_type=order_type,
            limit_price=limit_price,
            time_in_force=time_in_force,
            client_order_id=client_order_id,
        )
        return OrderResult(
            ok=False,
            status=STATUS_ERROR,
            symbol=symbol,
            client_order_id=client_order_id,
            reason="response lost after broker acceptance",
        )


class _PartialTerminalBroker(_RecordingBroker):
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
        if not self.submissions:
            self.submissions.append((symbol, qty, side, client_order_id))
            self._seq += 1
            order_id = f"fake-{self._seq}"
            order = OrderResult(
                ok=True,
                status=STATUS_CANCELED,
                order_id=order_id,
                client_order_id=client_order_id,
                symbol=symbol,
                qty=qty,
                filled_qty=1.0,
                filled_avg_price=limit_price,
                side=side,
                order_type=order_type,
                time_in_force=time_in_force,
                limit_price=limit_price,
                submitted_at=_TS.isoformat(),
                filled_at=_TS.isoformat(),
                updated_at=_TS.isoformat(),
                raw_status=STATUS_CANCELED,
            )
            self._orders[order_id] = order
            return order
        return super().submit_order(
            symbol,
            qty,
            side,
            order_type=order_type,
            limit_price=limit_price,
            time_in_force=time_in_force,
            client_order_id=client_order_id,
        )


class _LookupBroker(_RecordingBroker):
    def __init__(self, lookup: OrderLookupResult) -> None:
        super().__init__()
        self.lookup = lookup

    def get_order_by_client_order_id(
        self, client_order_id: str,
    ) -> OrderLookupResult:
        return self.lookup


def _prepare_generic(client_order_id: str = "generic-prepared-1") -> None:
    order_lifecycle.prepare_order_intent(
        ticker="NVDA",
        broker_symbol="NVDA",
        asset_class="stock",
        vehicle="shares",
        target_position_kind="long_term",
        side="sell",
        requested_qty=3.0,
        requested_limit_price=120.0,
        submitted_at=_TS,
        intent_payload={
            "intent_kind": "generic_emergency",
            "reason": "emergency_shutdown",
        },
        client_order_id=client_order_id,
        order_role="generic_emergency",
    )


def _submit_generic(
    broker: FakeBroker,
    *,
    qty: float = 3.0,
    now: datetime = _TS,
) -> OrderResult | None:
    return order_lifecycle.submit_generic_emergency_close(
        broker,
        broker_symbol="NVDA",
        side="sell",
        broker_position_qty=qty,
        requested_limit_price=120.0,
        reason="emergency_shutdown",
        submitted_at=now,
    )


def test_repeated_emergency_pass_suppresses_accepted_unfilled_duplicate(
    tmp_db: Path,
) -> None:
    broker = _RecordingBroker(auto_fill=False)
    broker.set_position("NVDA", 3.0, avg_entry_price=120.0)

    first = ror.emergency_shutdown(
        broker, trigger="test", notifier=_Recorder(), now=_TS
    )
    second = ror.emergency_shutdown(
        broker, trigger="test", notifier=_Recorder(), now=_TS + timedelta(minutes=1)
    )

    assert first.status == second.status == "holding"
    assert first.closed == second.closed == []
    assert "awaiting actual close fills" in first.pending[0]
    assert "close already pending" in second.pending[0]
    assert len(broker.submissions) == 1
    assert broker.submissions[0][:3] == ("NVDA", 3.0, "sell")
    assert broker.submissions[0][3] is not None
    attempts = db.get_generic_emergency_orders_for_symbol("NVDA")
    assert len(attempts) == 1
    assert attempts[0].lifecycle_status == STATUS_NEW


def test_restart_recovers_lost_submit_response_by_client_id_without_resubmit(
    tmp_db: Path,
) -> None:
    first_broker = _LostResponseBroker(auto_fill=False)
    first_broker.set_position("NVDA", 3.0, avg_entry_price=120.0)
    first = ror.emergency_shutdown(
        first_broker, trigger="test", notifier=_Recorder(), now=_TS
    )
    (prepared,) = db.get_generic_emergency_orders_for_symbol("NVDA")
    assert first.status == "holding"
    assert prepared.lifecycle_status == "prepared"
    assert prepared.client_order_id == first_broker.submissions[0][3]

    restarted = _RecordingBroker(auto_fill=False)
    restarted._positions = first_broker._positions.copy()
    restarted._orders = first_broker._orders.copy()
    restarted._seq = first_broker._seq
    second = ror.emergency_shutdown(
        restarted,
        trigger="test",
        notifier=_Recorder(),
        now=_TS + timedelta(minutes=1),
    )

    assert second.status == "holding"
    assert restarted.submissions == []
    (recovered,) = db.get_generic_emergency_orders_for_symbol("NVDA")
    assert recovered.lifecycle_status == STATUS_NEW
    assert recovered.broker_order_id == "fake-1"


def test_terminal_partial_retry_is_capped_at_unfilled_remainder(
    tmp_db: Path,
) -> None:
    broker = _PartialTerminalBroker(auto_fill=False)
    broker.set_position("NVDA", 3.0, avg_entry_price=120.0)

    first = ror.emergency_shutdown(
        broker, trigger="test", notifier=_Recorder(), now=_TS
    )
    second = ror.emergency_shutdown(
        broker, trigger="test", notifier=_Recorder(), now=_TS + timedelta(minutes=1)
    )

    assert first.status == second.status == "holding"
    assert [submission[1] for submission in broker.submissions] == [3.0, 2.0]
    attempts = db.get_generic_emergency_orders_for_symbol("NVDA")
    assert len(attempts) == 2
    assert attempts[0].filled_qty == 1.0
    assert attempts[0].terminal_at is not None
    assert attempts[1].terminal_at is None


def test_zero_fill_terminal_attempt_allows_full_retry(
    tmp_db: Path,
) -> None:
    broker = _RecordingBroker(reject_reason="market closed")
    broker.set_position("NVDA", 3.0, avg_entry_price=120.0)

    first = ror.emergency_shutdown(
        broker, trigger="test", notifier=_Recorder(), now=_TS
    )
    broker.reject_reason = None
    second = ror.emergency_shutdown(
        broker, trigger="test", notifier=_Recorder(), now=_TS + timedelta(minutes=1)
    )

    assert first.status == second.status == "holding"
    assert [submission[1] for submission in broker.submissions] == [3.0, 3.0]
    attempts = db.get_generic_emergency_orders_for_symbol("NVDA")
    assert [attempt.lifecycle_status for attempt in attempts] == [
        "abandoned",
        STATUS_NEW,
    ]


@pytest.mark.parametrize(
    "lookup",
    [
        OrderLookupResult(
            outcome=ORDER_LOOKUP_UNAVAILABLE,
            reason="provider unavailable",
        ),
        OrderLookupResult(
            outcome=ORDER_LOOKUP_FOUND,
            order=OrderResult(
                ok=True,
                status=STATUS_NEW,
                order_id=None,
                client_order_id="generic-prepared-1",
                symbol="NVDA",
                filled_qty=0.0,
            ),
        ),
    ],
    ids=["unavailable", "malformed"],
)
def test_ambiguous_prepared_lookup_never_blind_resubmits(
    tmp_db: Path, lookup: OrderLookupResult,
) -> None:
    _prepare_generic()
    broker = _LookupBroker(lookup)

    result = _submit_generic(broker)

    assert result is None
    assert broker.submissions == []
    (prepared,) = db.get_generic_emergency_orders_for_symbol("NVDA")
    assert prepared.lifecycle_status == "prepared"


def test_definitive_404_abandons_then_later_pass_uses_new_identity(
    tmp_db: Path,
) -> None:
    _prepare_generic()
    broker = _RecordingBroker(auto_fill=False)

    first = _submit_generic(broker)
    (abandoned,) = db.get_generic_emergency_orders_for_symbol("NVDA")
    second = _submit_generic(broker, now=_TS + timedelta(minutes=1))

    assert first is None
    assert abandoned.lifecycle_status == "abandoned"
    assert broker.submissions and second is not None
    attempts = db.get_generic_emergency_orders_for_symbol("NVDA")
    assert len(attempts) == 2
    assert attempts[0].client_order_id != attempts[1].client_order_id


def test_terminal_old_attempt_does_not_block_genuinely_new_later_position(
    tmp_db: Path,
) -> None:
    broker = _ClosingBroker(auto_fill=True)
    broker.set_position("NVDA", 3.0, avg_entry_price=120.0)

    first = ror.emergency_shutdown(
        broker, trigger="first", notifier=_Recorder(), now=_TS
    )
    broker.set_position("NVDA", 3.0, avg_entry_price=130.0)
    second = ror.emergency_shutdown(
        broker,
        trigger="second",
        notifier=_Recorder(),
        now=_TS + timedelta(days=1),
    )

    assert first.status == second.status == "halted"
    assert [submission[1] for submission in broker.submissions] == [3.0, 3.0]
    attempts = db.get_generic_emergency_orders_for_symbol("NVDA")
    assert len(attempts) == 2
    assert all(attempt.lifecycle_status == STATUS_FILLED for attempt in attempts)


def test_scanner_refreshes_generic_fill_without_typed_materialization_or_tier1(
    tmp_db: Path,
) -> None:
    broker = _RecordingBroker(auto_fill=False)
    submitted = _submit_generic(broker)
    assert submitted is not None and submitted.order_id is not None
    broker._orders[submitted.order_id] = dataclasses.replace(
        submitted,
        status=STATUS_FILLED,
        filled_qty=3.0,
        filled_avg_price=120.0,
        filled_at=(_TS + timedelta(seconds=10)).isoformat(),
        updated_at=(_TS + timedelta(seconds=10)).isoformat(),
        raw_status=STATUS_FILLED,
    )

    result = order_lifecycle.reconcile_pending_orders(
        broker, observed_at=_TS + timedelta(seconds=10)
    )
    (generic,) = db.get_generic_emergency_orders_for_symbol("NVDA")

    assert generic.lifecycle_status == STATUS_FILLED
    assert generic.order_role == "generic_emergency"
    assert generic.closes_position_kind is None
    assert generic.closes_position_id is None
    assert generic.position_kind is None
    assert generic.position_id is None
    assert json.loads(generic.intent_payload_json) == {
        "intent_kind": "generic_emergency",
        "reason": "emergency_shutdown",
    }
    assert result.materializations == ()
    assert result.exit_materializations == ()
    direct = order_lifecycle.materialize_pending_order_fill(generic)
    assert direct.action == order_lifecycle.MATERIALIZE_SKIPPED
    assert "not an entry materialization" in direct.reason
    assert db.get_pending_orders_with_fills() == []
    assert db.get_pending_exit_orders_with_fills() == []
    assert db.get_broker_execution_outcome_candidates() == []
    assert ror._ordered_broker_execution_outcomes() == []
    assert ror.consecutive_losses() == 0

    conn = db.get_connection()
    try:
        counts = {
            table: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in (
                "signals",
                "trades",
                "option_positions",
                "long_term_positions",
            )
        }
    finally:
        conn.close()
    assert counts == {
        "signals": 0,
        "trades": 0,
        "option_positions": 0,
        "long_term_positions": 0,
    }


def test_schema_unique_guard_rejects_second_live_generic_intent(
    tmp_db: Path,
) -> None:
    _prepare_generic("generic-live-1")

    with pytest.raises(order_lifecycle.OrderIntentCaptureError):
        _prepare_generic("generic-live-2")

    (only_attempt,) = db.get_generic_emergency_orders_for_symbol("NVDA")
    assert only_attempt.client_order_id == "generic-live-1"
