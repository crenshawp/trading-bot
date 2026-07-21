"""Final lifecycle cutover: restart/redeploy and scanner scheduling proofs."""

from __future__ import annotations

import dataclasses
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest
import schedule

from trading_bot import db, order_lifecycle, risk_of_ruin, scanner
from trading_bot.broker.base import (
    ORDER_LOOKUP_FOUND,
    ORDER_LOOKUP_NOT_FOUND,
    ORDER_LOOKUP_UNAVAILABLE,
    ORDER_TYPE_LIMIT,
    STATUS_CANCELED,
    STATUS_FILLED,
    STATUS_NEW,
    STATUS_PARTIALLY_FILLED,
    TIF_DAY,
    OrderLookupResult,
    OrderResult,
)
from trading_bot.broker.fake import FakeBroker
from trading_bot.models import LongTermPosition, OptionPosition, PendingOrder

_NOW = datetime(2026, 7, 21, 15, 0, tzinfo=UTC)
_OPTION_SYMBOL = "GOOGL260918C00190000"


def _option_payload() -> dict[str, object]:
    return {
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
        "deadline": None,
    }


def _insert_option_pending(
    order_id: str,
    *,
    lifecycle_status: str = STATUS_NEW,
    filled_qty: float = 0.0,
    filled_avg_price: float | None = None,
    legacy: bool = False,
) -> PendingOrder:
    terminal = lifecycle_status in {STATUS_FILLED, STATUS_CANCELED}
    pending = PendingOrder(
        client_order_id=None if legacy else f"tradingbot-test-{order_id}",
        broker_order_id=order_id,
        ticker="GOOGL",
        broker_symbol=_OPTION_SYMBOL,
        asset_class="stock",
        vehicle="option_full",
        target_position_kind="option",
        side="buy",
        requested_qty=2.0,
        requested_limit_price=5.1,
        submitted_at=_NOW,
        intent_payload_json=json.dumps(_option_payload(), sort_keys=True),
        lifecycle_status=lifecycle_status,
        broker_status=lifecycle_status,
        filled_qty=filled_qty,
        filled_avg_price=filled_avg_price,
        last_refreshed_at=_NOW,
        terminal_reason=lifecycle_status if terminal else None,
        terminal_at=_NOW if terminal else None,
    )
    pending_id = db.insert_pending_order(pending)
    stored = db.get_pending_order(order_id)
    assert stored is not None and stored.id == pending_id
    return stored


def _insert_legacy_long_term_pending(order_id: str) -> PendingOrder:
    pending = PendingOrder(
        client_order_id=None,
        broker_order_id=order_id,
        ticker="GOOGL",
        broker_symbol="GOOGL",
        asset_class="stock",
        vehicle="shares",
        target_position_kind="long_term",
        side="buy",
        requested_qty=2.0,
        requested_limit_price=175.0,
        submitted_at=_NOW,
        intent_payload_json=json.dumps(
            {
                "intent_kind": "long_term",
                "source": "long_term",
                "direction": "long",
            },
            sort_keys=True,
        ),
        lifecycle_status=STATUS_FILLED,
        broker_status=STATUS_FILLED,
        filled_qty=2.0,
        filled_avg_price=174.5,
        last_refreshed_at=_NOW,
        terminal_reason=STATUS_FILLED,
        terminal_at=_NOW,
    )
    db.insert_pending_order(pending)
    stored = db.get_pending_order(order_id)
    assert stored is not None
    return stored


def _prepare_option(client_order_id: str) -> PendingOrder:
    return order_lifecycle.prepare_order_intent(
        ticker="GOOGL",
        broker_symbol=_OPTION_SYMBOL,
        asset_class="stock",
        vehicle="option_full",
        target_position_kind="option",
        side="buy",
        requested_qty=2.0,
        requested_limit_price=5.1,
        submitted_at=_NOW,
        intent_payload=_option_payload(),
        client_order_id=client_order_id,
    )


class LookupOnlyBroker(FakeBroker):
    def __init__(self, lookup: OrderLookupResult) -> None:
        super().__init__()
        self.lookup = lookup
        self.lookup_calls = 0
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
        raise AssertionError("scanner recovery must never submit")

    def get_order_by_client_order_id(
        self, client_order_id: str,
    ) -> OrderLookupResult:
        self.lookup_calls += 1
        return self.lookup


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


def _snapshot(
    status: str,
    filled_qty: float,
    filled_avg_price: float | None,
    *,
    reason: str = "",
) -> OrderResult:
    return OrderResult(
        ok=True,
        status=status,
        order_id="broker-order",
        raw_status=status,
        filled_qty=filled_qty,
        filled_avg_price=filled_avg_price,
        reason=reason,
    )


def test_scanner_redeploy_recovers_lost_acceptance_and_materializes_fill(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepare_option("tradingbot-redeploy-success")
    accepted = dataclasses.replace(
        _snapshot(STATUS_FILLED, 2.0, 5.0),
        order_id="accepted-before-crash",
        client_order_id=prepared.client_order_id,
    )
    broker = LookupOnlyBroker(
        OrderLookupResult(outcome=ORDER_LOOKUP_FOUND, order=accepted)
    )
    monkeypatch.setattr("trading_bot.broker.AlpacaBroker", lambda: broker)

    scanner._run_order_lifecycle_cycle()

    recovered = db.get_pending_order_by_client_order_id(
        "tradingbot-redeploy-success"
    )
    assert recovered is not None
    assert recovered.broker_order_id == "accepted-before-crash"
    assert recovered.lifecycle_status == STATUS_FILLED
    assert recovered.position_id is not None
    (position,) = db.get_open_option_positions()
    assert (position.contracts, position.premium_entry) == (2.0, 5.0)
    assert broker.lookup_calls == 1
    assert broker.submit_calls == 0


def test_scanner_redeploy_404_abandons_prepared_without_resubmission(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_option("tradingbot-redeploy-404")
    broker = LookupOnlyBroker(
        OrderLookupResult(outcome=ORDER_LOOKUP_NOT_FOUND, reason="HTTP 404")
    )
    monkeypatch.setattr("trading_bot.broker.AlpacaBroker", lambda: broker)

    scanner._run_order_lifecycle_cycle()

    recovered = db.get_pending_order_by_client_order_id("tradingbot-redeploy-404")
    assert recovered is not None
    assert recovered.lifecycle_status == "abandoned"
    assert recovered.broker_order_id is None
    assert recovered.position_id is None
    assert db.get_open_option_positions() == []
    assert broker.lookup_calls == 1
    assert broker.submit_calls == 0


def test_scanner_redeploy_unavailable_remains_lookup_only_retryable(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_option("tradingbot-redeploy-unavailable")
    broker = LookupOnlyBroker(
        OrderLookupResult(
            outcome=ORDER_LOOKUP_UNAVAILABLE,
            reason="simulated outage",
        )
    )
    monkeypatch.setattr("trading_bot.broker.AlpacaBroker", lambda: broker)

    scanner._run_order_lifecycle_cycle()
    scanner._run_order_lifecycle_cycle()

    recovered = db.get_pending_order_by_client_order_id(
        "tradingbot-redeploy-unavailable"
    )
    assert recovered is not None
    assert recovered.lifecycle_status == "prepared"
    assert recovered.broker_order_id is None
    assert broker.lookup_calls == 2
    assert broker.submit_calls == 0


def test_restart_between_partial_fills_updates_one_position_idempotently(
    tmp_db: Path,
) -> None:
    _insert_option_pending("partial-restart")
    first_process = RefreshOnlyBroker(
        _snapshot(STATUS_PARTIALLY_FILLED, 1.0, 5.0)
    )
    first = order_lifecycle.reconcile_pending_orders(
        first_process, observed_at=_NOW
    )
    assert first.materializations[0].action == order_lifecycle.MATERIALIZE_CREATED
    first_position_id = first.materializations[0].position_id

    restarted_process = RefreshOnlyBroker(
        _snapshot(STATUS_FILLED, 2.0, 5.2)
    )
    second = order_lifecycle.reconcile_pending_orders(
        restarted_process, observed_at=_NOW
    )
    assert second.materializations[0].action == order_lifecycle.MATERIALIZE_UPDATED
    assert second.materializations[0].position_id == first_position_id

    repeated_process = RefreshOnlyBroker()
    third = order_lifecycle.reconcile_pending_orders(
        repeated_process, observed_at=_NOW
    )
    assert third.refreshes == ()
    assert third.materializations[0].action == order_lifecycle.MATERIALIZE_UNCHANGED
    assert repeated_process.get_calls == 0
    assert first_process.submit_calls == restarted_process.submit_calls == 0
    (position,) = db.get_open_option_positions()
    assert position.id == first_position_id
    assert (position.contracts, position.premium_entry) == (2.0, 5.2)
    assert len(db.get_pending_orders()) == 1


def test_restart_after_terminal_lifecycle_update_materializes_partial_fill(
    tmp_db: Path,
) -> None:
    _insert_option_pending("terminal-crash")
    before_crash = RefreshOnlyBroker(
        _snapshot(
            STATUS_CANCELED,
            1.0,
            4.9,
            reason="canceled after partial fill",
        )
    )
    refreshed = order_lifecycle.refresh_pending_orders(
        before_crash, observed_at=_NOW
    )
    assert refreshed[0].lifecycle_status == STATUS_CANCELED
    assert db.get_open_option_positions() == []

    restarted_process = RefreshOnlyBroker()
    recovered = order_lifecycle.reconcile_pending_orders(
        restarted_process, observed_at=_NOW
    )
    assert recovered.refreshes == ()
    assert recovered.materializations[0].action == order_lifecycle.MATERIALIZE_CREATED
    assert restarted_process.get_calls == 0
    (position,) = db.get_open_option_positions()
    assert (position.contracts, position.premium_entry) == (1.0, 4.9)


def test_materialization_failure_keeps_fill_retryable_without_resubmit(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pending = _insert_option_pending(
        "materialize-retry",
        lifecycle_status=STATUS_FILLED,
        filled_qty=2.0,
        filled_avg_price=5.0,
    )
    broker = RefreshOnlyBroker()
    original = db.materialize_new_option_position

    def fail_once(_pending_order_id: int, _pos: OptionPosition) -> int:
        raise RuntimeError("simulated position write failure")

    monkeypatch.setattr(db, "materialize_new_option_position", fail_once)
    failed = order_lifecycle.reconcile_pending_orders(broker, observed_at=_NOW)
    assert failed.materializations[0].action == order_lifecycle.MATERIALIZE_FAILED
    durable = db.get_pending_order(pending.broker_order_id)
    assert durable is not None
    assert durable.filled_qty == 2.0 and durable.position_id is None
    assert db.get_open_option_positions() == []
    assert broker.submit_calls == 0

    monkeypatch.setattr(db, "materialize_new_option_position", original)
    retried = order_lifecycle.reconcile_pending_orders(broker, observed_at=_NOW)
    assert retried.materializations[0].action == order_lifecycle.MATERIALIZE_CREATED
    assert len(db.get_open_option_positions()) == 1
    assert broker.submit_calls == 0


def test_legacy_option_row_adopts_unique_broker_identity_without_duplicate(
    tmp_db: Path,
) -> None:
    eager_position_id = db.insert_option_position(
        OptionPosition(
            symbol=_OPTION_SYMBOL,
            underlying="GOOGL",
            option_type="call",
            strike=190.0,
            expiry="2026-09-18",
            contracts=2.0,
            opened_at=_NOW,
            order_id="legacy-option",
            premium_entry=5.1,
            delta_entry=0.68,
            theta=-0.04,
            vega=0.12,
            gamma=0.03,
            tp=205.0,
            sl=175.0,
            outcome="open",
            vehicle="option_full",
        )
    )
    pending = _insert_option_pending(
        "legacy-option",
        lifecycle_status=STATUS_PARTIALLY_FILLED,
        filled_qty=1.0,
        filled_avg_price=4.9,
        legacy=True,
    )

    adopted = order_lifecycle.materialize_pending_order_fill(pending)
    replayed = order_lifecycle.materialize_pending_order_fill(pending)

    assert adopted.action == order_lifecycle.MATERIALIZE_ADOPTED
    assert adopted.position_id == eager_position_id
    assert replayed.action == order_lifecycle.MATERIALIZE_UNCHANGED
    (position,) = db.get_open_option_positions()
    assert position.id == eager_position_id
    assert (position.contracts, position.premium_entry) == (1.0, 4.9)
    linked = db.get_pending_order("legacy-option")
    assert linked is not None and linked.position_id == eager_position_id


def test_unprovable_legacy_rows_skip_for_manual_review_without_duplicates(
    tmp_db: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    eager_long_id = db.insert_long_term_position(
        LongTermPosition(
            ticker="GOOGL",
            asset_class="stock",
            entry_price=175.0,
            entry_date=_NOW,
            qty=2.0,
            status="open",
        )
    )
    legacy_long = _insert_legacy_long_term_pending("legacy-long")
    legacy_option = _insert_option_pending(
        "legacy-option-without-position",
        lifecycle_status=STATUS_FILLED,
        filled_qty=2.0,
        filled_avg_price=5.0,
        legacy=True,
    )

    long_result = order_lifecycle.materialize_pending_order_fill(legacy_long)
    option_result = order_lifecycle.materialize_pending_order_fill(legacy_option)

    assert long_result.action == order_lifecycle.MATERIALIZE_SKIPPED
    assert option_result.action == order_lifecycle.MATERIALIZE_SKIPPED
    assert "manual review" in long_result.reason
    assert "manual review" in option_result.reason
    (long_position,) = db.get_open_long_term_positions()
    assert long_position.id == eager_long_id
    assert db.get_open_option_positions() == []
    stored_long = db.get_pending_order("legacy-long")
    assert stored_long is not None and stored_long.position_id is None
    assert "manual review" in capsys.readouterr().err


def test_scanner_hook_invokes_one_fail_soft_reconciliation_pass(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    broker = RefreshOnlyBroker()
    monkeypatch.setattr("trading_bot.broker.AlpacaBroker", lambda: broker)
    calls = 0

    def fail_soft_once(_broker: FakeBroker) -> object:
        nonlocal calls
        calls += 1
        raise RuntimeError("simulated reconciliation failure")

    monkeypatch.setattr(order_lifecycle, "reconcile_pending_orders", fail_soft_once)
    scanner._run_order_lifecycle_cycle()

    assert calls == 1
    assert broker.submit_calls == 0
    assert "order lifecycle cycle error" in capsys.readouterr().err


def test_main_runs_boot_recovery_and_registers_one_hourly_hook(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schedule.clear()
    boot_calls = 0
    state_calls = 0

    def lifecycle_hook() -> None:
        nonlocal boot_calls
        boot_calls += 1

    def state() -> str:
        nonlocal state_calls
        state_calls += 1
        return risk_of_ruin.STATE_NORMAL if state_calls == 1 else risk_of_ruin.STATE_HALTED

    monkeypatch.setattr(scanner, "_load_secrets", lambda: None)
    monkeypatch.setattr(scanner, "_active_stock_watchlist", lambda: [])
    monkeypatch.setattr(scanner, "scan_stocks", lambda: None)
    monkeypatch.setattr(scanner, "_run_shadow_scan", lambda: None)
    monkeypatch.setattr(scanner, "scan_crypto", lambda: None)
    monkeypatch.setattr(scanner, "send_morning_report", lambda: None)
    monkeypatch.setattr(scanner, "_run_order_lifecycle_cycle", lifecycle_hook)
    monkeypatch.setattr(scanner, "_run_readiness_evaluation", lambda: None)
    monkeypatch.setattr(scanner, "_run_risk_cycle", lambda: None)
    monkeypatch.setattr(scanner, "_run_evaluator_cycle", lambda: None)
    monkeypatch.setattr(scanner, "_seed_prediction_defaults", lambda: None)
    monkeypatch.setattr(risk_of_ruin, "get_state", state)
    monkeypatch.setattr(
        risk_of_ruin,
        "run_guarded",
        lambda _fn, *, on_catastrophic: None,
    )

    try:
        with patch.object(sys, "argv", ["tradingbot"]):
            scanner.main()
        lifecycle_jobs = [
            job
            for job in schedule.jobs
            if getattr(job.job_func, "func", job.job_func) is lifecycle_hook
        ]
        assert boot_calls == 1
        assert len(lifecycle_jobs) == 1
        assert lifecycle_jobs[0].interval == 1
        assert lifecycle_jobs[0].unit == "hours"
    finally:
        schedule.clear()
