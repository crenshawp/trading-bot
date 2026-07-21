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
    PENDING_ORDER_INTENT_VERSION,
    TERMINAL_PENDING_ORDER_STATUSES,
    VALID_PENDING_ORDER_INTENT_KINDS,
    VALID_PENDING_ORDER_POSITION_KINDS,
    VALID_PENDING_ORDER_STATUSES,
    VALID_PENDING_ORDER_VEHICLES,
    PendingOrder,
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


def _pending_order(**overrides: Any) -> PendingOrder:
    base: dict[str, Any] = {
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
    return PendingOrder(**base)


def test_pending_order_model_is_frozen_and_vocabularies_are_explicit() -> None:
    order = _pending_order()
    with pytest.raises(dataclasses.FrozenInstanceError):
        order.ticker = "META"

    assert PENDING_ORDER_INTENT_VERSION == 1
    assert {
        "option", "long_term", "shares_fallback",
    } == VALID_PENDING_ORDER_INTENT_KINDS
    assert {
        "new", "partially_filled", "filled", "canceled", "rejected",
        "expired", "unknown",
    } == VALID_PENDING_ORDER_STATUSES
    assert {
        "filled", "canceled", "rejected", "expired",
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
        "id", "broker_order_id", "ticker", "broker_symbol", "asset_class",
        "vehicle", "target_position_kind", "side", "requested_qty",
        "requested_limit_price", "submitted_at", "signal_id",
        "intent_payload_version", "intent_payload_json", "lifecycle_status",
        "broker_status", "filled_qty", "filled_avg_price", "last_refreshed_at",
        "terminal_reason", "terminal_at", "position_kind", "position_id",
    }
    assert "idx_pending_orders_nonterminal" in indexes
    assert triggers == {
        "trg_pending_orders_immutable_intent",
        "trg_pending_orders_filled_qty_monotonic",
        "trg_pending_orders_position_link_once",
    }


def test_option_pending_order_round_trip_preserves_full_intent(tmp_db: Path) -> None:
    order = _pending_order()
    pending_id = db.insert_pending_order(order)

    fetched = db.get_pending_order("broker-order-1")
    assert fetched == dataclasses.replace(order, id=pending_id)
    assert db.get_pending_order("missing") is None
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


def test_duplicate_broker_order_id_is_rejected(tmp_db: Path) -> None:
    db.insert_pending_order(_pending_order())
    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
        db.insert_pending_order(_pending_order(ticker="META"))


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
        broker_order_id="zero-cancel",
        lifecycle_status="canceled",
        broker_status="expired",
        terminal_reason="expired",
        terminal_at=terminal_at,
    )
    partial = _pending_order(
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

    assert db.schema_version() == 24
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
