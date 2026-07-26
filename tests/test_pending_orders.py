"""Durable pending-order schema, model, migration, and CRUD tests."""

from __future__ import annotations

import dataclasses
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from trading_bot import db
from trading_bot.models import (
    LEGACY_PENDING_ORDER_CLIENT_ID_PREFIX,
    PENDING_ORDER_GENERIC_EMERGENCY_INTENT_VERSION,
    PENDING_ORDER_INTENT_VERSION,
    PENDING_ORDER_POSITION_EXIT_INTENT_VERSION,
    TERMINAL_PENDING_ORDER_STATUSES,
    VALID_PENDING_ORDER_FILL_TIME_SOURCES,
    VALID_PENDING_ORDER_INTENT_KINDS,
    VALID_PENDING_ORDER_POSITION_KINDS,
    VALID_PENDING_ORDER_ROLES,
    VALID_PENDING_ORDER_STATUSES,
    VALID_PENDING_ORDER_VEHICLES,
    OptionPosition,
    PendingOrder,
    Signal,
    Trade,
    is_recoverable_pending_order_client_id,
)


def _option_payload(**overrides: Any) -> str:
    payload: dict[str, Any] = {
        "intent_kind": "option",
        "option_type": "call",
        "strike": 190.0,
        "expiry": "2026-09-18",
        "multiplier": 100,
        "delta_entry": 0.68,
        "theta": -0.04,
        "vega": 0.12,
        "gamma": 0.03,
        "tp": 195.0,
        "sl": 175.0,
        "deadline": "2026-08-03T20:00:00+00:00",
    }
    payload.update(overrides)
    return json.dumps(payload, sort_keys=True)


def _option_payload_without(field: str) -> str:
    payload = json.loads(_option_payload())
    del payload[field]
    return json.dumps(payload, sort_keys=True)


def _share_payload(
    *,
    intent_kind: str = "long_term",
    source: str = "long_term",
    direction: str = "long",
    **overrides: Any,
) -> str:
    payload: dict[str, Any] = {
        "intent_kind": intent_kind,
        "source": source,
        "direction": direction,
    }
    if intent_kind == "shares_fallback":
        payload.update({"tp": 195.0, "sl": 175.0, "deadline": None})
    payload.update(overrides)
    return json.dumps(payload, sort_keys=True)


def _exit_payload(exit_reason: str = "take_profit", **overrides: Any) -> str:
    payload: dict[str, Any] = {
        "intent_kind": "position_exit",
        "exit_reason": exit_reason,
    }
    payload.update(overrides)
    return json.dumps(payload, sort_keys=True)


def _pending_order(**overrides: Any) -> PendingOrder:
    base: dict[str, Any] = {
        "client_order_id": "bot-order-1",
        "broker_order_id": "broker-order-1",
        "ticker": "GOOGL",
        "broker_symbol": "GOOGL260918C00190000",
        "asset_class": "stock",
        "vehicle": "option_full",
        "target_position_kind": "option",
        "side": "buy",
        "requested_qty": 2.0,
        "requested_limit_price": 4.25,
        "submitted_at": datetime(2026, 7, 20, 15, 0, tzinfo=UTC),
        "intent_payload_json": _option_payload(),
        "signal_id": 17,
        "broker_status": "accepted",
    }
    base.update(overrides)
    if (
        base.get("filled_qty", 0) > 0
        and "last_fill_at" not in overrides
        and "last_fill_time_source" not in overrides
    ):
        base["last_fill_at"] = (
            base.get("last_refreshed_at")
            or base.get("terminal_at")
            or base["submitted_at"]
        )
        base["last_fill_time_source"] = "observed"
    return PendingOrder(**base)


def test_pending_order_model_is_frozen_and_vocabularies_are_explicit() -> None:
    order = _pending_order()
    with pytest.raises(dataclasses.FrozenInstanceError):
        order.ticker = "META"

    assert PENDING_ORDER_INTENT_VERSION == 1
    assert PENDING_ORDER_POSITION_EXIT_INTENT_VERSION == 1
    assert PENDING_ORDER_GENERIC_EMERGENCY_INTENT_VERSION == 1
    assert {
        "option", "long_term", "shares_fallback", "position_exit",
        "generic_emergency",
    } == VALID_PENDING_ORDER_INTENT_KINDS
    assert {"entry", "exit", "generic_emergency"} == VALID_PENDING_ORDER_ROLES
    assert {"broker", "observed"} == VALID_PENDING_ORDER_FILL_TIME_SOURCES
    assert {
        "prepared", "new", "partially_filled", "filled", "canceled",
        "rejected", "expired", "unknown", "abandoned",
    } == VALID_PENDING_ORDER_STATUSES
    assert {
        "filled", "canceled", "rejected", "expired", "abandoned",
    } == TERMINAL_PENDING_ORDER_STATUSES
    assert {
        "option_full", "option_undersized", "shares",
    } == VALID_PENDING_ORDER_VEHICLES
    assert {"option", "long_term"} == VALID_PENDING_ORDER_POSITION_KINDS


def test_pending_order_schema_has_exact_columns_and_constraints(tmp_db: Path) -> None:
    conn = db.get_connection()
    try:
        columns = {
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(pending_orders)").fetchall()
        }
        indexes = {
            str(row["name"])
            for row in conn.execute("PRAGMA index_list(pending_orders)").fetchall()
        }
        triggers = {
            str(row["name"])
            for row in conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'trigger' AND tbl_name = 'pending_orders'"
            ).fetchall()
        }
    finally:
        conn.close()

    assert columns == {
        "id", "client_order_id", "broker_order_id", "order_role", "ticker",
        "broker_symbol", "asset_class", "vehicle", "target_position_kind",
        "closes_position_kind", "closes_position_id", "side", "requested_qty",
        "requested_limit_price", "submitted_at", "signal_id",
        "intent_payload_version", "intent_payload_json", "lifecycle_status",
        "broker_status", "filled_qty", "filled_avg_price", "last_fill_at",
        "last_fill_time_source", "fees_dollars", "last_refreshed_at",
        "terminal_reason", "terminal_at", "position_kind", "position_id",
    }
    assert "idx_pending_orders_nonterminal" in indexes
    assert "idx_pending_orders_exit_target" in indexes
    assert "idx_pending_orders_generic_nonterminal" in indexes
    assert "idx_pending_orders_typed_exit_nonterminal" in indexes
    assert triggers == {
        "trg_pending_orders_immutable_intent",
        "trg_pending_orders_broker_id_bind_once",
        "trg_pending_orders_filled_qty_monotonic",
        "trg_pending_orders_position_link_once",
        "trg_pending_orders_fill_time_on_insert",
        "trg_pending_orders_fill_time_on_update",
    }


def test_typed_exit_unique_index_is_authoritative_and_allows_terminal_retry(
    tmp_db: Path,
) -> None:
    first_id = db.insert_pending_order(_pending_order(
        client_order_id="typed-exit-first",
        broker_order_id=None,
        order_role="exit",
        closes_position_kind="option",
        closes_position_id=41,
        side="sell",
        intent_payload_json=_exit_payload(),
        lifecycle_status="prepared",
        broker_status=None,
    ))

    with pytest.raises(db.PendingExitAlreadyExistsError) as exc_info:
        db.insert_pending_order(_pending_order(
            client_order_id="typed-exit-racing",
            broker_order_id=None,
            order_role="exit",
            closes_position_kind="option",
            closes_position_id=41,
            side="sell",
            intent_payload_json=_exit_payload("stop_loss"),
            lifecycle_status="prepared",
            broker_status=None,
        ))

    assert exc_info.value.position_kind == "option"
    assert exc_info.value.position_id == 41
    assert exc_info.value.pending_order_id == first_id
    db.update_pending_order(
        first_id,
        lifecycle_status="abandoned",
        last_refreshed_at=datetime(2026, 7, 20, 15, 1, tzinfo=UTC),
        terminal_reason="definitively not submitted",
        terminal_at=datetime(2026, 7, 20, 15, 1, tzinfo=UTC),
    )
    retry_id = db.insert_pending_order(_pending_order(
        client_order_id="typed-exit-retry",
        broker_order_id=None,
        order_role="exit",
        closes_position_kind="option",
        closes_position_id=41,
        side="sell",
        intent_payload_json=_exit_payload("stop_loss"),
        lifecycle_status="prepared",
        broker_status=None,
    ))
    assert retry_id > first_id


def test_v30_typed_exit_duplicate_migration_fails_closed_then_restarts_cleanly(
    tmp_db: Path,
) -> None:
    conn = db.get_connection()
    try:
        conn.execute("DROP INDEX idx_pending_orders_typed_exit_nonterminal")
        conn.execute("UPDATE schema_version SET version = 30")
        conn.commit()
    finally:
        conn.close()

    duplicate_ids = [
        db.insert_pending_order(_pending_order(
            client_order_id=f"bot-duplicate-{suffix}",
            broker_order_id=None,
            order_role="exit",
            closes_position_kind="option",
            closes_position_id=73,
            side="sell",
            intent_payload_json=_exit_payload(reason),
            lifecycle_status="prepared",
            broker_status=None,
        ))
        for suffix, reason in (("a", "take_profit"), ("b", "stop_loss"))
    ]
    conn = db.get_connection()
    try:
        before = [
            tuple(row)
            for row in conn.execute(
                "SELECT * FROM pending_orders ORDER BY id"
            ).fetchall()
        ]
    finally:
        conn.close()

    with pytest.raises(RuntimeError) as exc_info:
        db.init_db()
    assert (
        "reconcile duplicate nonterminal exits against broker truth: "
        f"option/73 pending_order_ids={duplicate_ids}"
    ) in str(exc_info.value)

    conn = db.get_connection()
    try:
        after = [
            tuple(row)
            for row in conn.execute(
                "SELECT * FROM pending_orders ORDER BY id"
            ).fetchall()
        ]
        index = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index' "
            "AND name = 'idx_pending_orders_typed_exit_nonterminal'"
        ).fetchone()
    finally:
        conn.close()
    assert after == before
    assert index is None
    assert db.schema_version() == 30

    db.update_pending_order(
        duplicate_ids[1],
        lifecycle_status="abandoned",
        last_refreshed_at=datetime(2026, 7, 20, 15, 2, tzinfo=UTC),
        terminal_reason="operator reconciled as never submitted",
        terminal_at=datetime(2026, 7, 20, 15, 2, tzinfo=UTC),
    )
    db.init_db()
    assert db.schema_version() == 31
    assert len(db.get_pending_exit_orders_for_position("option", 73)) == 2


def test_v25_sentiment_migration_preserves_pending_order_and_legacy_trade(
    tmp_db: Path,
) -> None:
    pending_id = db.insert_pending_order(_pending_order())
    pending_before = db.get_pending_order("broker-order-1")
    signal_id = db.insert_signal(Signal(
        timestamp=datetime(2026, 7, 20, 9, 31),
        ticker="GOOGL",
        asset_class="stock",
        signal_type="ema21_pullback",
        direction="call",
        entry_price=180.0,
    ))
    trade_id = db.insert_trade(Trade(
        signal_id=signal_id,
        opened_at=datetime(2026, 7, 20, 9, 31),
        sentiment_score=0.0,
        sentiment_label="neutral",
    ))

    conn = db.get_connection()
    try:
        conn.execute("ALTER TABLE trades DROP COLUMN sentiment_ok")
        conn.execute("ALTER TABLE trades DROP COLUMN sentiment_rationale")
        conn.execute("UPDATE schema_version SET version = 25")
        conn.commit()
    finally:
        conn.close()

    db.init_db()
    db.init_db()

    assert db.schema_version() == 31
    assert db.get_pending_order("broker-order-1") == pending_before
    assert pending_before is not None and pending_before.id == pending_id
    legacy = db.get_trade_by_signal_id(signal_id)
    assert legacy is not None and legacy.id == trade_id
    assert legacy.sentiment_label == "neutral"
    assert legacy.sentiment_ok is None
    assert legacy.sentiment_rationale is None


def test_v26_indicator_migration_preserves_pending_order_and_legacy_trade(
    tmp_db: Path,
) -> None:
    pending_id = db.insert_pending_order(_pending_order())
    pending_before = db.get_pending_order("broker-order-1")
    signal_id = db.insert_signal(Signal(
        timestamp=datetime(2026, 7, 20, 9, 31),
        ticker="GOOGL",
        asset_class="stock",
        signal_type="ema21_pullback",
        direction="call",
        entry_price=180.0,
    ))
    trade_id = db.insert_trade(Trade(
        signal_id=signal_id,
        opened_at=datetime(2026, 7, 20, 9, 31),
        ind_vol_regime="unknown",
        ind_concentration="unknown",
    ))

    conn = db.get_connection()
    try:
        conn.execute("ALTER TABLE trades DROP COLUMN ind_ok")
        conn.execute("UPDATE schema_version SET version = 26")
        conn.commit()
    finally:
        conn.close()

    db.init_db()
    db.init_db()

    assert db.schema_version() == 31
    assert db.get_pending_order("broker-order-1") == pending_before
    assert pending_before is not None and pending_before.id == pending_id
    legacy = db.get_trade_by_signal_id(signal_id)
    assert legacy is not None and legacy.id == trade_id
    assert legacy.ind_vol_regime == "unknown"
    assert legacy.ind_ok is None


def test_v27_risk_migration_preserves_pending_order_and_legacy_trade(
    tmp_db: Path,
) -> None:
    pending_id = db.insert_pending_order(_pending_order())
    pending_before = db.get_pending_order("broker-order-1")
    signal_id = db.insert_signal(Signal(
        timestamp=datetime(2026, 7, 20, 9, 31),
        ticker="GOOGL",
        asset_class="stock",
        signal_type="ema21_pullback",
        direction="call",
        entry_price=180.0,
    ))
    trade_id = db.insert_trade(Trade(
        signal_id=signal_id,
        opened_at=datetime(2026, 7, 20, 9, 31),
        risk_portfolio_verdict="unknown",
        risk_position_verdict="unknown",
        risk_cluster_verdict="unknown",
    ))

    conn = db.get_connection()
    try:
        conn.execute("ALTER TABLE trades DROP COLUMN risk_ok")
        conn.execute("ALTER TABLE trades DROP COLUMN risk_reason")
        conn.execute("UPDATE schema_version SET version = 27")
        conn.commit()
    finally:
        conn.close()

    db.init_db()
    db.init_db()

    assert db.schema_version() == 31
    assert db.get_pending_order("broker-order-1") == pending_before
    assert pending_before is not None and pending_before.id == pending_id
    legacy = db.get_trade_by_signal_id(signal_id)
    assert legacy is not None and legacy.id == trade_id
    assert legacy.risk_portfolio_verdict == "unknown"
    assert legacy.risk_ok is None
    assert legacy.risk_reason is None


def test_option_pending_order_round_trip_preserves_full_intent(tmp_db: Path) -> None:
    order = _pending_order()
    pending_id = db.insert_pending_order(order)

    fetched = db.get_pending_order("broker-order-1")
    assert fetched == dataclasses.replace(order, id=pending_id)
    assert db.get_pending_order("missing") is None
    assert db.get_pending_order_by_client_order_id("bot-order-1") == fetched
    assert db.get_pending_order_by_client_order_id("missing") is None
    assert db.get_pending_orders() == [fetched]
    assert db.get_nonterminal_pending_orders() == [fetched]


@pytest.mark.parametrize(
    ("broker_order_id", "intent_kind", "side", "direction", "source"),
    [
        ("long-term-1", "long_term", "buy", "long", "long_term"),
        (
            "shares-fallback-1", "shares_fallback", "sell", "short",
            "swing_fallback",
        ),
    ],
)
def test_share_intents_round_trip_without_rerunning_selection(
    tmp_db: Path,
    broker_order_id: str,
    intent_kind: str,
    side: str,
    direction: str,
    source: str,
) -> None:
    payload: dict[str, Any] = {
        "intent_kind": intent_kind,
        "source": source,
        "direction": direction,
    }
    if intent_kind == "shares_fallback":
        payload.update({"tp": 195.0, "sl": 175.0, "deadline": None})
    order = _pending_order(
        broker_order_id=broker_order_id,
        broker_symbol="GOOGL",
        vehicle="shares",
        target_position_kind="long_term",
        side=side,
        requested_qty=1.75,
        requested_limit_price=182.25,
        intent_payload_json=json.dumps(payload, sort_keys=True),
    )

    pending_id = db.insert_pending_order(order)
    fetched = db.get_pending_order(broker_order_id)

    assert fetched == dataclasses.replace(order, id=pending_id)
    assert json.loads(fetched.intent_payload_json) == payload


def test_exit_role_round_trip_uses_typed_close_target_not_entry_link(
    tmp_db: Path,
) -> None:
    exit_order = _pending_order(
        client_order_id="bot-exit-1",
        broker_order_id="broker-exit-1",
        order_role="exit",
        side="sell",
        closes_position_kind="option",
        closes_position_id=41,
        intent_payload_json=_exit_payload("take_profit"),
        lifecycle_status="partially_filled",
        broker_status="partially_filled",
        filled_qty=0.5,
        filled_avg_price=4.5,
        last_fill_at=datetime(2026, 7, 20, 15, 2, tzinfo=UTC),
        last_fill_time_source="broker",
    )

    pending_id = db.insert_pending_order(exit_order)
    fetched = db.get_pending_order("broker-exit-1")

    assert fetched == dataclasses.replace(exit_order, id=pending_id)
    assert fetched is not None
    assert fetched.position_kind is None and fetched.position_id is None
    assert fetched.fees_dollars is None
    assert db.get_pending_exit_orders_for_position("option", 41) == [fetched]
    assert db.get_pending_orders_with_fills() == []


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"closes_position_kind": "option", "closes_position_id": 1}, "entry"),
        (
            {
                "order_role": "exit", "side": "sell",
                "intent_payload_json": _exit_payload(),
            },
            "close target",
        ),
        (
            {
                "order_role": "exit", "side": "sell",
                "closes_position_kind": "long_term", "closes_position_id": 1,
                "intent_payload_json": _exit_payload(),
            },
            "match",
        ),
        (
            {
                "order_role": "exit", "side": "buy",
                "closes_position_kind": "option", "closes_position_id": 1,
                "intent_payload_json": _exit_payload(),
            },
            "sell",
        ),
        (
            {
                "order_role": "exit", "side": "sell",
                "closes_position_kind": "option", "closes_position_id": 1,
            },
            "position_exit",
        ),
        ({"intent_payload_json": _exit_payload()}, "exit-role"),
        (
            {
                "order_role": "exit", "side": "sell",
                "closes_position_kind": "option", "closes_position_id": 1,
                "intent_payload_json": _exit_payload(" "),
            },
            "exit_reason",
        ),
        (
            {
                "order_role": "exit", "side": "sell",
                "closes_position_kind": "option", "closes_position_id": 1,
                "intent_payload_json": _exit_payload(),
                "intent_payload_version": 2,
            },
            "Unsupported",
        ),
    ],
)
def test_entry_exit_shapes_and_versioned_exit_payload_are_validated(
    tmp_db: Path, overrides: dict[str, Any], match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        db.insert_pending_order(_pending_order(**overrides))


def test_fill_time_provenance_and_unknown_fees_are_enforced(tmp_db: Path) -> None:
    observed_at = datetime(2026, 7, 20, 15, 2, tzinfo=UTC)
    observed = _pending_order(
        client_order_id="bot-observed-fill",
        broker_order_id="observed-fill",
        lifecycle_status="partially_filled",
        broker_status="partially_filled",
        filled_qty=1.0,
        filled_avg_price=4.2,
        last_refreshed_at=observed_at,
    )
    pending_id = db.insert_pending_order(observed)
    stored = db.get_pending_order("observed-fill")
    assert stored is not None and stored.id == pending_id
    assert stored.last_fill_at == observed_at
    assert stored.last_fill_time_source == "observed"
    assert stored.fees_dollars is None
    broker_at = datetime(2026, 7, 20, 15, 1, 30, tzinfo=UTC)
    broker_fill = _pending_order(
        client_order_id="bot-broker-fill",
        broker_order_id="broker-fill",
        lifecycle_status="partially_filled",
        broker_status="partially_filled",
        filled_qty=1.0,
        filled_avg_price=4.15,
        last_fill_at=broker_at,
        last_fill_time_source="broker",
    )
    db.insert_pending_order(broker_fill)
    stored_broker = db.get_pending_order("broker-fill")
    assert stored_broker is not None
    assert stored_broker.last_fill_at == broker_at
    assert stored_broker.last_fill_time_source == "broker"
    assert stored_broker.fees_dollars is None
    conn = db.get_connection()
    try:
        with pytest.raises(sqlite3.IntegrityError, match="fill-time provenance"):
            conn.execute(
                "UPDATE pending_orders SET last_fill_at = NULL, "
                "last_fill_time_source = NULL WHERE id = ?",
                (pending_id,),
            )
    finally:
        conn.close()

    with pytest.raises(ValueError, match="requires last_fill_at"):
        db.insert_pending_order(
            _pending_order(
                client_order_id="bot-missing-fill-time",
                broker_order_id="missing-fill-time",
                lifecycle_status="partially_filled",
                broker_status="partially_filled",
                filled_qty=1.0,
                filled_avg_price=4.2,
                last_fill_at=None,
                last_fill_time_source=None,
            )
        )
    with pytest.raises(ValueError, match="zero-fill"):
        db.insert_pending_order(
            _pending_order(
                client_order_id="bot-zero-fill-time",
                broker_order_id="zero-fill-time",
                last_fill_at=observed_at,
                last_fill_time_source="observed",
            )
        )
    with pytest.raises(ValueError, match="zero-fill"):
        db.insert_pending_order(
            _pending_order(
                client_order_id="bot-zero-fee",
                broker_order_id="zero-fee",
                fees_dollars=0.0,
            )
        )


def test_exit_close_target_and_entry_position_link_are_immutable(tmp_db: Path) -> None:
    exit_order = _pending_order(
        client_order_id="bot-exit-immutable",
        broker_order_id="exit-immutable",
        order_role="exit",
        side="sell",
        closes_position_kind="option",
        closes_position_id=41,
        intent_payload_json=_exit_payload("stop_loss"),
    )
    pending_id = db.insert_pending_order(exit_order)
    conn = db.get_connection()
    try:
        with pytest.raises(sqlite3.IntegrityError, match="intent is immutable"):
            conn.execute(
                "UPDATE pending_orders SET closes_position_id = 42 WHERE id = ?",
                (pending_id,),
            )
        with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint"):
            conn.execute(
                "UPDATE pending_orders SET position_kind = 'option', "
                "position_id = 41 WHERE id = ?",
                (pending_id,),
            )
    finally:
        conn.close()


def test_option_position_round_trip_and_update_preserve_exit_reason(
    tmp_db: Path,
) -> None:
    position = OptionPosition(
        symbol="GOOGL260918C00190000",
        underlying="GOOGL",
        option_type="call",
        strike=190.0,
        expiry="2026-09-18",
        contracts=1.0,
        opened_at=datetime(2026, 7, 20, 15, 0, tzinfo=UTC),
        exit_reason="take_profit",
    )
    position_id = db.insert_option_position(position)
    assert db.get_option_position(position_id) == dataclasses.replace(
        position, id=position_id
    )

    db.update_option_position(position_id, exit_reason="emergency_shutdown")
    updated = db.get_option_position(position_id)
    assert updated is not None
    assert updated.exit_reason == "emergency_shutdown"


def test_v28_option_position_migration_preserves_row_and_adds_null_exit_reason(
    tmp_db: Path,
) -> None:
    position = OptionPosition(
        symbol="GOOGL260918C00190000",
        underlying="GOOGL",
        option_type="call",
        strike=190.0,
        expiry="2026-09-18",
        contracts=1.0,
        opened_at=datetime(2026, 7, 20, 15, 0, tzinfo=UTC),
        premium_entry=4.2,
    )
    position_id = db.insert_option_position(position)
    conn = db.get_connection()
    try:
        conn.execute("ALTER TABLE option_positions DROP COLUMN exit_reason")
        conn.execute("UPDATE schema_version SET version = 28")
        conn.commit()
    finally:
        conn.close()

    db.init_db()
    db.init_db()

    assert db.get_option_position(position_id) == dataclasses.replace(
        position, id=position_id
    )
    assert db.schema_version() == 31


def test_duplicate_broker_order_id_is_rejected(tmp_db: Path) -> None:
    db.insert_pending_order(_pending_order())
    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
        db.insert_pending_order(
            _pending_order(client_order_id="bot-order-2", ticker="META")
        )


def test_prepared_rows_allow_multiple_null_broker_ids_with_unique_client_ids(
    tmp_db: Path,
) -> None:
    first = _pending_order(
        client_order_id="bot-prepared-1",
        broker_order_id=None,
        lifecycle_status="prepared",
        broker_status=None,
    )
    second = dataclasses.replace(first, client_order_id="bot-prepared-2")

    first_id = db.insert_pending_order(first)
    second_id = db.insert_pending_order(second)

    assert first_id != second_id
    assert db.get_pending_order_by_client_order_id(
        "bot-prepared-1"
    ) == dataclasses.replace(first, id=first_id)
    assert db.get_pending_order_by_client_order_id(
        "bot-prepared-2"
    ) == dataclasses.replace(second, id=second_id)
    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
        db.insert_pending_order(dataclasses.replace(second, client_order_id="bot-prepared-1"))


def test_only_bound_legacy_path_synthesizes_recovery_ineligible_client_id(
    tmp_db: Path,
) -> None:
    pending_id = db.insert_pending_order(_pending_order(client_order_id=None))
    fetched = db.get_pending_order("broker-order-1")

    assert fetched is not None
    assert fetched.id == pending_id
    assert fetched.client_order_id == (
        f"{LEGACY_PENDING_ORDER_CLIENT_ID_PREFIX}broker-broker-order-1"
    )
    assert not is_recoverable_pending_order_client_id(fetched.client_order_id)

    with pytest.raises(ValueError, match="explicit before submit"):
        db.insert_pending_order(
            _pending_order(
                client_order_id=None,
                broker_order_id=None,
                lifecycle_status="prepared",
                broker_status=None,
            )
        )
    with pytest.raises(ValueError, match="namespace is reserved"):
        db.insert_pending_order(
            _pending_order(
                client_order_id="legacy-prepared-1",
                broker_order_id=None,
                lifecycle_status="prepared",
                broker_status=None,
            )
        )


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"intent_payload_json": "not-json"}, "valid JSON"),
        ({"intent_payload_version": 2}, "Unsupported"),
        (
            {"intent_payload_json": _option_payload(theta="not-a-number")},
            "theta",
        ),
        (
            {"vehicle": "shares", "target_position_kind": "long_term"},
            "option intent",
        ),
        ({"side": "sell"}, "buy order"),
        ({"lifecycle_status": "accepted"}, "status"),
        ({"requested_qty": 0.0}, "requested_qty"),
    ],
)
def test_insert_pending_order_rejects_unusable_contracts(
    tmp_db: Path, overrides: dict[str, Any], match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        db.insert_pending_order(_pending_order(**overrides))


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"intent_payload_json": "[]"}, "JSON object"),
        ({"intent_payload_json": "{}"}, "intent_kind"),
        (
            {"intent_payload_json": _option_payload_without("theta")},
            "missing fields",
        ),
        ({"intent_payload_json": _option_payload(option_type="straddle")}, "option_type"),
        ({"intent_payload_json": _option_payload(strike=0)}, "strike"),
        ({"intent_payload_json": _option_payload(multiplier=100.0)}, "multiplier"),
        ({"intent_payload_json": _option_payload(expiry="soon")}, "expiry"),
        (
            {"intent_payload_json": _share_payload()},
            "share intent requires",
        ),
        (
            {
                "vehicle": "shares",
                "target_position_kind": "long_term",
                "intent_payload_json": json.dumps({"intent_kind": "long_term"}),
            },
            "missing fields",
        ),
        (
            {
                "vehicle": "shares",
                "target_position_kind": "long_term",
                "intent_payload_json": _share_payload(source="swing_fallback"),
            },
            "source",
        ),
        (
            {
                "vehicle": "shares",
                "target_position_kind": "long_term",
                "intent_payload_json": _share_payload(direction="sideways"),
            },
            "direction",
        ),
        (
            {
                "vehicle": "shares",
                "target_position_kind": "long_term",
                "intent_payload_json": _share_payload(direction="short"),
            },
            "submitted side",
        ),
        (
            {
                "vehicle": "shares",
                "target_position_kind": "long_term",
                "side": "sell",
                "intent_payload_json": _share_payload(direction="short"),
            },
            "buy/long",
        ),
        ({"intent_payload_json": _option_payload(deadline=17)}, "deadline"),
        ({"intent_payload_json": _option_payload(deadline="tomorrow")}, "deadline"),
    ],
)
def test_insert_pending_order_rejects_incomplete_version_one_payloads(
    tmp_db: Path, overrides: dict[str, Any], match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        db.insert_pending_order(_pending_order(**overrides))


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"broker_order_id": " "}, "broker_order_id"),
        ({"asset_class": "future"}, "asset_class"),
        ({"vehicle": "spread"}, "vehicle"),
        ({"target_position_kind": "trade"}, "target position kind"),
        ({"side": "hold"}, "side"),
        ({"requested_limit_price": 0.0}, "requested_limit_price"),
        ({"signal_id": 0}, "signal_id"),
        ({"filled_qty": -1.0}, "filled_qty"),
        ({"filled_qty": 1.0}, "appear together"),
        ({"lifecycle_status": "filled"}, "terminal_reason"),
        ({"terminal_reason": "premature"}, "nonterminal"),
        (
            {
                "lifecycle_status": "partially_filled",
                "filled_qty": 1.0,
                "filled_avg_price": 4.2,
                "position_kind": "trade",
                "position_id": 3,
            },
            "position kind",
        ),
        (
            {
                "lifecycle_status": "partially_filled",
                "filled_qty": 1.0,
                "filled_avg_price": 4.2,
                "position_kind": "option",
                "position_id": 0,
            },
            "position_id",
        ),
        (
            {
                "lifecycle_status": "partially_filled",
                "filled_qty": 1.0,
                "filled_avg_price": 4.2,
                "position_kind": "long_term",
                "position_id": 3,
            },
            "match its target",
        ),
    ],
)
def test_insert_pending_order_rejects_invalid_normalized_fields(
    tmp_db: Path, overrides: dict[str, Any], match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        db.insert_pending_order(_pending_order(**overrides))


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"broker_order_id": "broker-order-1"}, "must be unbound"),
        ({"broker_status": "pending"}, "broker status"),
        (
            {"filled_qty": 1.0, "filled_avg_price": 4.2},
            "cannot carry a fill",
        ),
        (
            {"last_refreshed_at": datetime(2026, 7, 20, 15, 1, tzinfo=UTC)},
            "refresh time",
        ),
        (
            {
                "filled_qty": 1.0,
                "filled_avg_price": 4.2,
                "position_kind": "option",
                "position_id": 1,
            },
            "cannot carry a fill",
        ),
    ],
)
def test_prepared_insert_enforces_pre_submit_invariants(
    tmp_db: Path, overrides: dict[str, Any], match: str,
) -> None:
    prepared_fields: dict[str, Any] = {
        "client_order_id": "bot-prepared-1",
        "broker_order_id": None,
        "lifecycle_status": "prepared",
        "broker_status": None,
    }
    prepared_fields.update(overrides)
    order = _pending_order(**prepared_fields)
    with pytest.raises(ValueError, match=match):
        db.insert_pending_order(order)


def test_prepared_order_binds_broker_id_once_and_id_becomes_immutable(
    tmp_db: Path,
) -> None:
    prepared = _pending_order(
        client_order_id="bot-prepared-1",
        broker_order_id=None,
        lifecycle_status="prepared",
        broker_status=None,
    )
    pending_id = db.insert_pending_order(prepared)
    observed_at = datetime(2026, 7, 20, 15, 1, tzinfo=UTC)

    with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint"):
        db.update_pending_order(
            pending_id,
            broker_order_id="broker-bound-too-early",
        )
    db.update_pending_order(
        pending_id,
        broker_order_id="broker-bound-1",
        lifecycle_status="new",
        broker_status="accepted",
        last_refreshed_at=observed_at,
    )

    bound = db.get_pending_order_by_client_order_id("bot-prepared-1")
    assert bound is not None
    assert bound.broker_order_id == "broker-bound-1"
    assert bound.lifecycle_status == "new"
    assert bound.last_refreshed_at == observed_at
    with pytest.raises(sqlite3.IntegrityError, match="may only bind once"):
        db.update_pending_order(pending_id, broker_order_id="broker-rebound-2")
    with pytest.raises(sqlite3.IntegrityError, match="may only bind once"):
        db.update_pending_order(pending_id, broker_order_id=None)


def test_abandoned_is_unbound_zero_fill_terminal_with_reason_and_time(
    tmp_db: Path,
) -> None:
    terminal_at = datetime(2026, 7, 20, 15, 5, tzinfo=UTC)
    prepared = _pending_order(
        client_order_id="bot-abandoned-1",
        broker_order_id=None,
        lifecycle_status="prepared",
        broker_status=None,
    )
    pending_id = db.insert_pending_order(prepared)
    with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint"):
        db.update_pending_order(pending_id, lifecycle_status="abandoned")
    db.update_pending_order(
        pending_id,
        lifecycle_status="abandoned",
        terminal_reason="client_order_id not found",
        terminal_at=terminal_at,
    )
    abandoned = dataclasses.replace(
        prepared,
        lifecycle_status="abandoned",
        terminal_reason="client_order_id not found",
        terminal_at=terminal_at,
    )

    assert db.get_pending_order_by_client_order_id(
        "bot-abandoned-1"
    ) == dataclasses.replace(abandoned, id=pending_id)
    assert db.get_nonterminal_pending_orders() == []
    with pytest.raises(ValueError, match="terminal_reason"):
        db.insert_pending_order(
            dataclasses.replace(
                abandoned,
                client_order_id="bot-abandoned-2",
                terminal_reason=None,
                terminal_at=None,
            )
        )
    with pytest.raises(ValueError, match="broker status"):
        db.insert_pending_order(
            dataclasses.replace(
                abandoned,
                client_order_id="bot-abandoned-3",
                broker_status="not_found",
            )
        )


def test_cumulative_fill_update_links_once_and_terminalizes(tmp_db: Path) -> None:
    pending_id = db.insert_pending_order(_pending_order())
    refreshed_at = datetime(2026, 7, 20, 15, 2, tzinfo=UTC)
    db.update_pending_order(
        pending_id,
        lifecycle_status="partially_filled",
        broker_status="partially_filled",
        filled_qty=1.0,
        filled_avg_price=4.20,
        last_refreshed_at=refreshed_at,
        position_kind="option",
        position_id=41,
    )
    partial = db.get_pending_order("broker-order-1")
    assert partial is not None
    assert partial.lifecycle_status == "partially_filled"
    assert partial.filled_qty == 1.0
    assert partial.filled_avg_price == 4.20
    assert partial.last_refreshed_at == refreshed_at
    assert partial.position_kind == "option"
    assert partial.position_id == 41
    assert db.get_nonterminal_pending_orders() == [partial]

    terminal_at = datetime(2026, 7, 20, 15, 4, tzinfo=UTC)
    db.update_pending_order(
        pending_id,
        lifecycle_status="filled",
        broker_status="filled",
        filled_qty=2.0,
        filled_avg_price=4.30,
        last_refreshed_at=terminal_at,
        terminal_reason="filled",
        terminal_at=terminal_at,
    )
    filled = db.get_pending_order("broker-order-1")
    assert filled is not None
    assert filled.lifecycle_status == "filled"
    assert filled.filled_qty == 2.0
    assert filled.filled_avg_price == 4.30
    assert filled.position_id == 41
    assert filled.terminal_reason == "filled"
    assert filled.terminal_at == terminal_at
    assert db.get_nonterminal_pending_orders() == []


def test_zero_fill_cancel_has_no_position_and_partial_cancel_preserves_link(
    tmp_db: Path,
) -> None:
    terminal_at = datetime(2026, 7, 20, 20, 0, tzinfo=UTC)
    zero = _pending_order(
        client_order_id="bot-zero-cancel",
        broker_order_id="zero-cancel",
        lifecycle_status="canceled",
        broker_status="expired",
        terminal_reason="expired",
        terminal_at=terminal_at,
    )
    partial = _pending_order(
        client_order_id="bot-partial-cancel",
        broker_order_id="partial-cancel",
        lifecycle_status="canceled",
        broker_status="canceled",
        filled_qty=1.0,
        filled_avg_price=4.20,
        terminal_reason="canceled",
        terminal_at=terminal_at,
        position_kind="option",
        position_id=88,
    )

    db.insert_pending_order(zero)
    db.insert_pending_order(partial)

    assert db.get_pending_order("zero-cancel") == dataclasses.replace(zero, id=1)
    assert db.get_pending_order("partial-cancel") == dataclasses.replace(partial, id=2)
    assert db.get_nonterminal_pending_orders() == []


def test_position_link_requires_usable_fill(tmp_db: Path) -> None:
    with pytest.raises(ValueError, match="usable fill"):
        db.insert_pending_order(
            _pending_order(position_kind="option", position_id=9)
        )


def test_update_whitelist_and_database_triggers_protect_durable_truth(
    tmp_db: Path,
) -> None:
    pending_id = db.insert_pending_order(
        _pending_order(
            lifecycle_status="partially_filled",
            broker_status="partially_filled",
            filled_qty=1.0,
            filled_avg_price=4.2,
            position_kind="option",
            position_id=20,
        )
    )
    with pytest.raises(ValueError, match="Unknown pending_order field"):
        db.update_pending_order(pending_id, ticker="META")
    with pytest.raises(sqlite3.IntegrityError, match="cannot decrease"):
        db.update_pending_order(pending_id, filled_qty=0.5, filled_avg_price=4.1)
    with pytest.raises(sqlite3.IntegrityError, match="position link is immutable"):
        db.update_pending_order(pending_id, position_id=21)

    conn = db.get_connection()
    try:
        with pytest.raises(sqlite3.IntegrityError, match="intent is immutable"):
            conn.execute(
                "UPDATE pending_orders SET ticker = 'META' WHERE id = ?",
                (pending_id,),
            )
        with pytest.raises(sqlite3.IntegrityError, match="intent is immutable"):
            conn.execute(
                "UPDATE pending_orders SET client_order_id = ? WHERE id = ?",
                ("bot-order-rebound", pending_id),
            )
    finally:
        conn.close()


def test_terminal_status_requires_reason_and_time_atomically(tmp_db: Path) -> None:
    pending_id = db.insert_pending_order(_pending_order())
    with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint"):
        db.update_pending_order(pending_id, lifecycle_status="canceled")


@pytest.mark.parametrize(
    ("fields", "match"),
    [
        ({"lifecycle_status": "accepted"}, "status"),
        ({"filled_qty": -1.0}, "filled_qty"),
        ({"filled_avg_price": 0.0}, "filled_avg_price"),
        ({"position_kind": "trade"}, "position kind"),
        ({"position_id": 0}, "position_id"),
    ],
)
def test_update_pending_order_rejects_invalid_lifecycle_fields(
    tmp_db: Path, fields: dict[str, Any], match: str,
) -> None:
    pending_id = db.insert_pending_order(_pending_order())
    with pytest.raises(ValueError, match=match):
        db.update_pending_order(pending_id, **fields)


def test_update_pending_order_with_no_fields_is_noop(tmp_db: Path) -> None:
    pending_id = db.insert_pending_order(_pending_order())
    db.update_pending_order(pending_id)
    assert db.get_pending_order("broker-order-1") is not None


def test_v23_database_migrates_without_rewriting_existing_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy_path = tmp_path / "legacy-v23.db"
    conn = sqlite3.connect(legacy_path)
    try:
        conn.execute("CREATE TABLE schema_version (version INTEGER NOT NULL)")
        conn.execute("INSERT INTO schema_version (version) VALUES (23)")
        conn.execute(
            "CREATE TABLE settings ("
            "key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
            ("sentinel", "preserved", "2026-07-20T00:00:00+00:00"),
        )
        conn.commit()
    finally:
        conn.close()
    monkeypatch.setattr("trading_bot.config.DB_PATH", legacy_path)

    db.init_db()
    db.init_db()

    assert db.schema_version() == 31
    migrated = db.get_connection()
    try:
        sentinel = migrated.execute(
            "SELECT value FROM settings WHERE key = 'sentinel'"
        ).fetchone()
        table = migrated.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name = 'pending_orders'"
        ).fetchone()
    finally:
        migrated.close()
    assert sentinel is not None and sentinel["value"] == "preserved"
    assert table is not None
    assert db.get_pending_orders() == []


def _create_v24_pending_orders(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE pending_orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            broker_order_id TEXT NOT NULL UNIQUE,
            ticker TEXT NOT NULL,
            broker_symbol TEXT NOT NULL,
            asset_class TEXT NOT NULL CHECK (asset_class IN ('stock','crypto')),
            vehicle TEXT NOT NULL CHECK (
                vehicle IN ('option_full','option_undersized','shares')
            ),
            target_position_kind TEXT NOT NULL CHECK (
                target_position_kind IN ('option','long_term')
            ),
            side TEXT NOT NULL CHECK (side IN ('buy','sell')),
            requested_qty REAL NOT NULL CHECK (requested_qty > 0),
            requested_limit_price REAL CHECK (
                requested_limit_price IS NULL OR requested_limit_price > 0
            ),
            submitted_at TEXT NOT NULL,
            signal_id INTEGER,
            intent_payload_version INTEGER NOT NULL DEFAULT 1 CHECK (
                intent_payload_version >= 1
            ),
            intent_payload_json TEXT NOT NULL,
            lifecycle_status TEXT NOT NULL DEFAULT 'new' CHECK (
                lifecycle_status IN (
                    'new','partially_filled','filled','canceled','rejected',
                    'expired','unknown'
                )
            ),
            broker_status TEXT,
            filled_qty REAL NOT NULL DEFAULT 0 CHECK (filled_qty >= 0),
            filled_avg_price REAL,
            last_refreshed_at TEXT,
            terminal_reason TEXT,
            terminal_at TEXT,
            position_kind TEXT CHECK (position_kind IN ('option','long_term')),
            position_id INTEGER,
            CHECK (
                (target_position_kind = 'option'
                 AND vehicle IN ('option_full','option_undersized')
                 AND side = 'buy')
                OR (target_position_kind = 'long_term' AND vehicle = 'shares')
            ),
            CHECK (
                (filled_qty = 0 AND filled_avg_price IS NULL)
                OR (filled_qty > 0 AND filled_avg_price > 0)
            ),
            CHECK (
                (lifecycle_status IN ('filled','canceled','rejected','expired')
                 AND terminal_at IS NOT NULL
                 AND TRIM(COALESCE(terminal_reason, '')) <> '')
                OR (lifecycle_status IN ('new','partially_filled','unknown')
                    AND terminal_at IS NULL AND terminal_reason IS NULL)
            ),
            CHECK (
                (position_kind IS NULL AND position_id IS NULL)
                OR (position_kind IS NOT NULL AND position_id > 0)
            ),
            CHECK (position_kind IS NULL OR position_kind = target_position_kind),
            CHECK (
                position_id IS NULL OR (filled_qty > 0 AND filled_avg_price > 0)
            )
        );
        CREATE INDEX idx_pending_orders_nonterminal
            ON pending_orders(submitted_at) WHERE terminal_at IS NULL;
        CREATE TRIGGER trg_pending_orders_immutable_intent
        BEFORE UPDATE OF
            broker_order_id, ticker, broker_symbol, asset_class, vehicle,
            target_position_kind, side, requested_qty, requested_limit_price,
            submitted_at, signal_id, intent_payload_version, intent_payload_json
        ON pending_orders
        BEGIN
            SELECT RAISE(
                ABORT, 'pending order materialization intent is immutable'
            );
        END;
        CREATE TRIGGER trg_pending_orders_filled_qty_monotonic
        BEFORE UPDATE OF filled_qty ON pending_orders
        WHEN NEW.filled_qty < OLD.filled_qty
        BEGIN
            SELECT RAISE(ABORT, 'pending order filled_qty cannot decrease');
        END;
        CREATE TRIGGER trg_pending_orders_position_link_once
        BEFORE UPDATE OF position_kind, position_id ON pending_orders
        WHEN OLD.position_id IS NOT NULL AND (
            NEW.position_id IS NOT OLD.position_id
            OR NEW.position_kind IS NOT OLD.position_kind
        )
        BEGIN
            SELECT RAISE(
                ABORT, 'pending order position link is immutable once set'
            );
        END;
        """
    )


def test_v28_to_v30_migration_preserves_every_legacy_ledger_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy_path = tmp_path / "legacy-v28.db"
    conn = sqlite3.connect(legacy_path)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("CREATE TABLE schema_version (version INTEGER NOT NULL)")
        conn.execute("INSERT INTO schema_version (version) VALUES (28)")
        _create_v24_pending_orders(conn)
        db._migrate_to_v25(conn)
        conn.execute(
            """
            INSERT INTO pending_orders (
                id, client_order_id, broker_order_id, ticker, broker_symbol,
                asset_class, vehicle, target_position_kind, side, requested_qty,
                requested_limit_price, submitted_at, signal_id,
                intent_payload_version, intent_payload_json, lifecycle_status,
                broker_status, filled_qty, filled_avg_price, last_refreshed_at,
                terminal_reason, terminal_at, position_kind, position_id
            ) VALUES (
                5, 'v28-client-5', 'v28-broker-5', 'GOOGL',
                'GOOGL260918C00190000', 'stock', 'option_full', 'option',
                'buy', 2.0, 4.25, '2026-07-20T15:00:00+00:00', 17, 1, ?,
                'partially_filled', 'partially_filled', 1.0, 4.2, NULL,
                NULL, NULL, 'option', 88
            )
            """,
            (_option_payload(),),
        )
        before_row = conn.execute(
            "SELECT * FROM pending_orders WHERE id = 5"
        ).fetchone()
        assert before_row is not None
        legacy_columns = before_row.keys()
        legacy_values = {key: before_row[key] for key in legacy_columns}
        conn.commit()
    finally:
        conn.close()
    monkeypatch.setattr("trading_bot.config.DB_PATH", legacy_path)

    db.init_db()
    db.init_db()

    migrated = db.get_connection()
    try:
        after_row = migrated.execute(
            "SELECT * FROM pending_orders WHERE id = 5"
        ).fetchone()
    finally:
        migrated.close()
    assert after_row is not None
    assert {key: after_row[key] for key in legacy_values} == legacy_values
    assert after_row["order_role"] == "entry"
    assert after_row["closes_position_kind"] is None
    assert after_row["closes_position_id"] is None
    assert after_row["last_fill_at"] == "2026-07-20T15:00:00+00:00"
    assert after_row["last_fill_time_source"] == "observed"
    assert after_row["fees_dollars"] is None
    assert db.schema_version() == 31


def test_v24_pending_order_rows_migrate_collision_free_and_preserve_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy_path = tmp_path / "legacy-v24.db"
    conn = sqlite3.connect(legacy_path)
    try:
        conn.execute("CREATE TABLE schema_version (version INTEGER NOT NULL)")
        conn.execute("INSERT INTO schema_version (version) VALUES (24)")
        _create_v24_pending_orders(conn)
        rows = (
            (
                7, "legacy-bound-7", "GOOGL", "GOOGL260918C00190000",
                "stock", "option_full", "option", "buy", 2.0, 4.25,
                "2026-07-20T15:00:00+00:00", 17, 1, _option_payload(),
                "new", "accepted", 0.0, None, None, None, None, None, None,
            ),
            (
                11, "legacy-bound-11", "GOOGL", "GOOGL260918C00190000",
                "stock", "option_full", "option", "buy", 2.0, 4.25,
                "2026-07-20T15:00:00+00:00", 17, 1, _option_payload(),
                "canceled", "canceled", 1.0, 4.2,
                "2026-07-20T15:04:00+00:00", "canceled",
                "2026-07-20T15:04:00+00:00", "option", 88,
            ),
            (
                15, "legacy-bound-15", "GOOGL", "GOOGL260918C00190000",
                "stock", "option_full", "option", "buy", 2.0, 4.25,
                "2026-07-20T15:10:00+00:00", 17, 1, _option_payload(),
                "partially_filled", "partially_filled", 0.5, 4.1,
                None, None, None, None, None,
            ),
        )
        conn.executemany(
            """
            INSERT INTO pending_orders (
                id, broker_order_id, ticker, broker_symbol, asset_class,
                vehicle, target_position_kind, side, requested_qty,
                requested_limit_price, submitted_at, signal_id,
                intent_payload_version, intent_payload_json, lifecycle_status,
                broker_status, filled_qty, filled_avg_price, last_refreshed_at,
                terminal_reason, terminal_at, position_kind, position_id
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?
            )
            """,
            rows,
        )
        conn.commit()
    finally:
        conn.close()
    monkeypatch.setattr("trading_bot.config.DB_PATH", legacy_path)

    db.init_db()
    db.init_db()

    assert db.schema_version() == 31
    first = db.get_pending_order("legacy-bound-7")
    second = db.get_pending_order("legacy-bound-11")
    third = db.get_pending_order("legacy-bound-15")
    assert first is not None and second is not None and third is not None
    assert first.client_order_id == "legacy-v24-7"
    assert second.client_order_id == "legacy-v24-11"
    assert first.id == 7 and second.id == 11
    assert first.broker_status == "accepted" and first.filled_qty == 0
    assert first.order_role == "entry"
    assert first.closes_position_kind is None and first.closes_position_id is None
    assert first.last_fill_at is None and first.last_fill_time_source is None
    assert first.fees_dollars is None
    assert second.lifecycle_status == "canceled"
    assert second.filled_qty == 1.0 and second.filled_avg_price == 4.2
    assert second.position_kind == "option" and second.position_id == 88
    assert second.order_role == "entry"
    assert second.last_fill_at == datetime(
        2026, 7, 20, 15, 4, tzinfo=UTC
    )
    assert second.last_fill_time_source == "observed"
    assert second.fees_dollars is None
    assert third.last_fill_at == datetime(
        2026, 7, 20, 15, 10, tzinfo=UTC
    )
    assert third.last_fill_time_source == "observed"
    assert third.fees_dollars is None
    assert not is_recoverable_pending_order_client_id(first.client_order_id)
    assert not is_recoverable_pending_order_client_id(second.client_order_id)
    assert db.insert_pending_order(
        _pending_order(
            client_order_id="bot-post-migration-1",
            broker_order_id=None,
            lifecycle_status="prepared",
            broker_status=None,
        )
    ) == 16

    migrated = db.get_connection()
    try:
        indexes = {
            str(row["name"])
            for row in migrated.execute(
                "PRAGMA index_list(pending_orders)"
            ).fetchall()
        }
        triggers = {
            str(row["name"])
            for row in migrated.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'trigger' AND tbl_name = 'pending_orders'"
            ).fetchall()
        }
    finally:
        migrated.close()
    assert "idx_pending_orders_nonterminal" in indexes
    assert "idx_pending_orders_exit_target" in indexes
    assert triggers == {
        "trg_pending_orders_immutable_intent",
        "trg_pending_orders_broker_id_bind_once",
        "trg_pending_orders_filled_qty_monotonic",
        "trg_pending_orders_position_link_once",
        "trg_pending_orders_fill_time_on_insert",
        "trg_pending_orders_fill_time_on_update",
    }
