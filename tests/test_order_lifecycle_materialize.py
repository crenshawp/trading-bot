"""Dormant fill materialization: cumulative broker fills → positions.

Every scenario is driven purely off durable ledger state — no broker is
polled or submitted here. The materializer reads the fill snapshot that the
refresh step already persisted and turns it into the correct option /
long-term / shares-fallback position, idempotently and atomically.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from trading_bot import db, order_lifecycle
from trading_bot.models import OptionPosition, PendingOrder

_SUBMITTED_AT = datetime(2026, 7, 21, 14, 31, tzinfo=UTC)
_REFRESHED_AT = datetime(2026, 7, 21, 14, 35, tzinfo=UTC)
_OPENED_AT = datetime(2026, 7, 21, 14, 40, tzinfo=UTC)
_OCC_SYMBOL = "GOOGL260821C00180000"
_DEADLINE = "2026-08-01T20:00:00+00:00"


def _option_payload() -> str:
    return json.dumps(
        {
            "intent_kind": "option",
            "option_type": "call",
            "strike": 180.0,
            "expiry": "2026-08-21",
            "multiplier": 100,
            "delta_entry": 0.55,
            "theta": -0.03,
            "vega": 0.10,
            "gamma": 0.02,
            "tp": 190.0,
            "sl": 170.0,
            "deadline": _DEADLINE,
        },
        sort_keys=True,
    )


def _long_term_payload() -> str:
    return json.dumps(
        {"intent_kind": "long_term", "source": "long_term", "direction": "long"},
        sort_keys=True,
    )


def _shares_fallback_payload(direction: str) -> str:
    return json.dumps(
        {
            "intent_kind": "shares_fallback",
            "source": "swing_fallback",
            "direction": direction,
            "tp": 190.0,
            "sl": 170.0,
            "deadline": _DEADLINE,
        },
        sort_keys=True,
    )


def _insert_pending(
    broker_order_id: str,
    *,
    target_position_kind: str,
    vehicle: str,
    side: str,
    intent_payload_json: str,
    lifecycle_status: str = "partially_filled",
    filled_qty: float = 1.0,
    filled_avg_price: float | None = 2.5,
    terminal_reason: str | None = None,
    terminal_at: datetime | None = None,
    ticker: str = "GOOGL",
    broker_symbol: str = "GOOGL",
    asset_class: str = "stock",
    requested_qty: float = 3.0,
    signal_id: int | None = None,
) -> PendingOrder:
    """Insert one bound, fill-carrying ledger row (post-refresh state)."""
    broker_status = (
        None if lifecycle_status in ("prepared", "abandoned") else lifecycle_status
    )
    pending = PendingOrder(
        client_order_id=f"tradingbot-{broker_order_id}",
        broker_order_id=broker_order_id,
        ticker=ticker,
        broker_symbol=broker_symbol,
        asset_class=asset_class,
        vehicle=vehicle,
        target_position_kind=target_position_kind,
        side=side,
        requested_qty=requested_qty,
        requested_limit_price=175.0,
        submitted_at=_SUBMITTED_AT,
        signal_id=signal_id,
        intent_payload_json=intent_payload_json,
        lifecycle_status=lifecycle_status,
        broker_status=broker_status,
        filled_qty=filled_qty,
        filled_avg_price=filled_avg_price,
        last_refreshed_at=_REFRESHED_AT,
        terminal_reason=terminal_reason,
        terminal_at=terminal_at,
    )
    db.insert_pending_order(pending)
    inserted = db.get_pending_order(broker_order_id)
    assert inserted is not None
    return inserted


def _raw_insert_pending(
    *,
    broker_order_id: str,
    intent_payload_json: str,
    intent_payload_version: int = 1,
    target_position_kind: str = "option",
    vehicle: str = "option_full",
    side: str = "buy",
    broker_symbol: str = _OCC_SYMBOL,
    filled_qty: float = 1.0,
    filled_avg_price: float = 2.5,
) -> str:
    """Insert a ledger row bypassing Python validation.

    Table CHECK constraints still hold; only the JSON/version *content* rules
    (enforced in Python at insert) are skipped, so we can simulate a corrupt or
    forward-version payload reaching the dormant materializer.
    """
    conn = db.get_connection()
    try:
        conn.execute(
            "INSERT INTO pending_orders ("
            "client_order_id, broker_order_id, ticker, broker_symbol, asset_class, "
            "vehicle, target_position_kind, side, requested_qty, "
            "requested_limit_price, submitted_at, signal_id, "
            "intent_payload_version, intent_payload_json, lifecycle_status, "
            "broker_status, filled_qty, filled_avg_price, last_refreshed_at, "
            "terminal_reason, terminal_at, position_kind, position_id"
            ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                f"tradingbot-{broker_order_id}",
                broker_order_id,
                "GOOGL",
                broker_symbol,
                "stock",
                vehicle,
                target_position_kind,
                side,
                3.0,
                175.0,
                _SUBMITTED_AT.isoformat(),
                None,
                intent_payload_version,
                intent_payload_json,
                "partially_filled",
                "partially_filled",
                filled_qty,
                filled_avg_price,
                _REFRESHED_AT.isoformat(),
                None,
                None,
                None,
                None,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return f"tradingbot-{broker_order_id}"


def _require_pending(broker_order_id: str) -> PendingOrder:
    pending = db.get_pending_order(broker_order_id)
    assert pending is not None
    return pending


# ── first-fill creation ──────────────────────────────────────────────────────


def test_option_first_fill_creates_and_links_position(tmp_db: Path) -> None:
    pending = _insert_pending(
        "opt-1",
        target_position_kind="option",
        vehicle="option_full",
        side="buy",
        intent_payload_json=_option_payload(),
        broker_symbol=_OCC_SYMBOL,
        filled_qty=2.0,
        filled_avg_price=3.25,
        signal_id=29,
    )

    result = order_lifecycle.materialize_pending_order_fill(
        pending, observed_at=_OPENED_AT
    )

    assert result.action == order_lifecycle.MATERIALIZE_CREATED
    assert result.position_kind == "option"
    assert result.position_id is not None

    pos = db.get_option_position(result.position_id)
    assert pos is not None
    assert pos.symbol == _OCC_SYMBOL
    assert pos.underlying == "GOOGL"
    assert pos.option_type == "call"
    assert pos.strike == 180.0
    assert pos.expiry == "2026-08-21"
    assert pos.contracts == 2.0                 # actual broker cumulative qty
    assert pos.premium_entry == 3.25            # broker average, not the quote
    assert pos.multiplier == 100
    assert pos.delta_entry == 0.55
    assert pos.tp == 190.0
    assert pos.sl == 170.0
    assert pos.deadline == datetime.fromisoformat(_DEADLINE)
    assert pos.order_id == "opt-1"
    assert pos.signal_id == 29
    assert pos.opened_at == _OPENED_AT
    assert pos.vehicle == "option_full"
    assert pos.outcome == "open"

    linked = db.get_pending_order("opt-1")
    assert linked is not None
    assert linked.position_kind == "option"
    assert linked.position_id == result.position_id


def test_long_term_first_fill_creates_position(tmp_db: Path) -> None:
    pending = _insert_pending(
        "lt-1",
        target_position_kind="long_term",
        vehicle="shares",
        side="buy",
        intent_payload_json=_long_term_payload(),
        filled_qty=3.0,
        filled_avg_price=175.0,
    )

    result = order_lifecycle.materialize_pending_order_fill(
        pending, observed_at=_OPENED_AT
    )

    assert result.action == order_lifecycle.MATERIALIZE_CREATED
    assert result.position_kind == "long_term"
    assert result.position_id is not None

    pos = db.get_long_term_position(result.position_id)
    assert pos is not None
    assert pos.source == "long_term"
    assert pos.direction == "long"
    assert pos.qty == 3.0
    assert pos.entry_price == 175.0
    assert pos.entry_date == _OPENED_AT
    assert pos.status == "open"
    assert pos.tp is None
    assert pos.sl is None
    assert pos.deadline is None


def test_shares_fallback_long_creates_swing_position(tmp_db: Path) -> None:
    pending = _insert_pending(
        "sf-long",
        target_position_kind="long_term",
        vehicle="shares",
        side="buy",
        intent_payload_json=_shares_fallback_payload("long"),
        filled_qty=4.0,
        filled_avg_price=181.0,
    )

    result = order_lifecycle.materialize_pending_order_fill(
        pending, observed_at=_OPENED_AT
    )
    assert result.action == order_lifecycle.MATERIALIZE_CREATED
    assert result.position_id is not None

    pos = db.get_long_term_position(result.position_id)
    assert pos is not None
    assert pos.source == "swing_fallback"
    assert pos.direction == "long"
    assert pos.qty == 4.0
    assert pos.entry_price == 181.0
    assert pos.tp == 190.0
    assert pos.sl == 170.0
    assert pos.deadline == datetime.fromisoformat(_DEADLINE)


def test_shares_fallback_short_records_short_direction(tmp_db: Path) -> None:
    pending = _insert_pending(
        "sf-short",
        target_position_kind="long_term",
        vehicle="shares",
        side="sell",
        intent_payload_json=_shares_fallback_payload("short"),
        filled_qty=2.0,
        filled_avg_price=181.0,
    )

    result = order_lifecycle.materialize_pending_order_fill(pending)
    assert result.action == order_lifecycle.MATERIALIZE_CREATED
    assert result.position_id is not None

    pos = db.get_long_term_position(result.position_id)
    assert pos is not None
    assert pos.direction == "short"
    assert pos.source == "swing_fallback"


def test_immediate_full_fill_creates_then_replay_is_noop(tmp_db: Path) -> None:
    pending = _insert_pending(
        "full-1",
        target_position_kind="long_term",
        vehicle="shares",
        side="buy",
        intent_payload_json=_long_term_payload(),
        lifecycle_status="filled",
        filled_qty=3.0,
        filled_avg_price=175.0,
        terminal_reason="filled",
        terminal_at=_REFRESHED_AT,
    )

    created = order_lifecycle.materialize_pending_order_fill(pending)
    assert created.action == order_lifecycle.MATERIALIZE_CREATED

    replay = order_lifecycle.materialize_pending_order_fill(
        _require_pending("full-1")
    )
    assert replay.action == order_lifecycle.MATERIALIZE_UNCHANGED
    assert replay.position_id == created.position_id
    assert len(db.get_long_term_positions()) == 1


# ── cumulative advancement ───────────────────────────────────────────────────


def test_multiple_partials_advance_same_position_row(tmp_db: Path) -> None:
    pending = _insert_pending(
        "multi-1",
        target_position_kind="long_term",
        vehicle="shares",
        side="buy",
        intent_payload_json=_long_term_payload(),
        filled_qty=1.0,
        filled_avg_price=174.0,
    )
    first = order_lifecycle.materialize_pending_order_fill(pending)
    assert first.action == order_lifecycle.MATERIALIZE_CREATED
    position_id = first.position_id
    assert position_id is not None

    db.update_pending_order(_pk("multi-1"), filled_qty=2.0, filled_avg_price=174.5)
    second = order_lifecycle.materialize_pending_order_fill(
        _require_pending("multi-1")
    )
    assert second.action == order_lifecycle.MATERIALIZE_UPDATED
    assert second.position_id == position_id

    db.update_pending_order(_pk("multi-1"), filled_qty=3.0, filled_avg_price=175.0)
    third = order_lifecycle.materialize_pending_order_fill(
        _require_pending("multi-1")
    )
    assert third.action == order_lifecycle.MATERIALIZE_UPDATED
    assert third.position_id == position_id

    pos = db.get_long_term_position(position_id)
    assert pos is not None
    assert pos.qty == 3.0
    assert pos.entry_price == 175.0
    assert len(db.get_long_term_positions()) == 1   # never a second row


def test_option_higher_fill_updates_contracts_and_premium(tmp_db: Path) -> None:
    pending = _insert_pending(
        "opt-adv",
        target_position_kind="option",
        vehicle="option_full",
        side="buy",
        intent_payload_json=_option_payload(),
        broker_symbol=_OCC_SYMBOL,
        filled_qty=1.0,
        filled_avg_price=3.0,
    )
    created = order_lifecycle.materialize_pending_order_fill(pending)
    assert created.position_id is not None

    db.update_pending_order(_pk("opt-adv"), filled_qty=2.0, filled_avg_price=3.5)
    updated = order_lifecycle.materialize_pending_order_fill(
        _require_pending("opt-adv")
    )
    assert updated.action == order_lifecycle.MATERIALIZE_UPDATED

    pos = db.get_option_position(created.position_id)
    assert pos is not None
    assert pos.contracts == 2.0
    assert pos.premium_entry == 3.5
    assert len(db.get_option_positions()) == 1


def test_fill_change_during_update_never_applies_a_stale_snapshot(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pending = _insert_pending(
        "update-snapshot",
        target_position_kind="long_term",
        vehicle="shares",
        side="buy",
        intent_payload_json=_long_term_payload(),
        filled_qty=1.0,
        filled_avg_price=174.0,
    )
    created = order_lifecycle.materialize_pending_order_fill(pending)
    assert created.position_id is not None
    pending_id = _pk("update-snapshot")
    db.update_pending_order(pending_id, filled_qty=2.0, filled_avg_price=174.5)
    original = db.update_long_term_position_fill

    def _advance_before_update(
        pending_order_id: int,
        position_id: int,
        *,
        qty: float,
        entry_price: float,
    ) -> bool:
        db.update_pending_order(
            pending_order_id, filled_qty=3.0, filled_avg_price=175.0
        )
        return original(
            pending_order_id,
            position_id,
            qty=qty,
            entry_price=entry_price,
        )

    monkeypatch.setattr(db, "update_long_term_position_fill", _advance_before_update)
    failed = order_lifecycle.materialize_pending_order_fill(
        _require_pending("update-snapshot")
    )

    assert failed.action == order_lifecycle.MATERIALIZE_FAILED
    assert "fill changed" in failed.reason
    position = db.get_long_term_position(created.position_id)
    assert position is not None
    assert position.qty == 1.0
    assert position.entry_price == 174.0

    monkeypatch.setattr(db, "update_long_term_position_fill", original)
    retried = order_lifecycle.materialize_pending_order_fill(
        _require_pending("update-snapshot")
    )
    assert retried.action == order_lifecycle.MATERIALIZE_UPDATED
    position = db.get_long_term_position(created.position_id)
    assert position is not None
    assert position.qty == 3.0
    assert position.entry_price == 175.0


# ── terminal / zero-fill truth ───────────────────────────────────────────────


def test_terminal_partial_fill_materializes_the_partial_holding(tmp_db: Path) -> None:
    pending = _insert_pending(
        "term-part",
        target_position_kind="long_term",
        vehicle="shares",
        side="buy",
        intent_payload_json=_long_term_payload(),
        lifecycle_status="canceled",
        filled_qty=1.0,
        filled_avg_price=174.0,
        terminal_reason="canceled after partial fill",
        terminal_at=_REFRESHED_AT,
    )

    result = order_lifecycle.materialize_pending_order_fill(pending)
    assert result.action == order_lifecycle.MATERIALIZE_CREATED
    assert result.position_id is not None

    pos = db.get_long_term_position(result.position_id)
    assert pos is not None
    assert pos.qty == 1.0                # the ACTUAL partial holding is preserved
    assert pos.entry_price == 174.0


def test_zero_fill_creates_no_position(tmp_db: Path) -> None:
    pending = _insert_pending(
        "zero-new",
        target_position_kind="long_term",
        vehicle="shares",
        side="buy",
        intent_payload_json=_long_term_payload(),
        lifecycle_status="new",
        filled_qty=0.0,
        filled_avg_price=None,
    )

    result = order_lifecycle.materialize_pending_order_fill(pending)
    assert result.action == order_lifecycle.MATERIALIZE_SKIPPED
    assert result.position_id is None
    assert db.get_long_term_positions() == []

    linked = db.get_pending_order("zero-new")
    assert linked is not None
    assert linked.position_id is None


@pytest.mark.parametrize("lifecycle_status", ["canceled", "rejected", "expired"])
def test_terminal_zero_fill_creates_no_position(
    tmp_db: Path, lifecycle_status: str
) -> None:
    pending = _insert_pending(
        f"zero-{lifecycle_status}",
        target_position_kind="option",
        vehicle="option_full",
        side="buy",
        intent_payload_json=_option_payload(),
        broker_symbol=_OCC_SYMBOL,
        lifecycle_status=lifecycle_status,
        filled_qty=0.0,
        filled_avg_price=None,
        terminal_reason=f"{lifecycle_status} unfilled",
        terminal_at=_REFRESHED_AT,
    )

    result = order_lifecycle.materialize_pending_order_fill(pending)
    assert result.action == order_lifecycle.MATERIALIZE_SKIPPED
    assert db.get_option_positions() == []


# ── idempotency / restart safety ─────────────────────────────────────────────


def test_restart_style_replay_never_duplicates_position(tmp_db: Path) -> None:
    pending = _insert_pending(
        "restart-1",
        target_position_kind="option",
        vehicle="option_full",
        side="buy",
        intent_payload_json=_option_payload(),
        broker_symbol=_OCC_SYMBOL,
        filled_qty=2.0,
        filled_avg_price=3.0,
        signal_id=31,
    )
    first = order_lifecycle.materialize_pending_order_fill(pending)
    assert first.action == order_lifecycle.MATERIALIZE_CREATED

    # A restart re-reads the SAME durable ledger row and reprocesses it.
    for _ in range(3):
        replay = order_lifecycle.materialize_pending_order_fill(
            _require_pending("restart-1")
        )
        assert replay.action == order_lifecycle.MATERIALIZE_UNCHANGED
        assert replay.position_id == first.position_id
    assert len(db.get_option_positions()) == 1
    position = db.get_option_position(first.position_id)
    assert position is not None
    assert position.signal_id == 31


def test_missing_ledger_row_is_skipped(tmp_db: Path) -> None:
    ghost = PendingOrder(
        client_order_id="tradingbot-ghost",
        broker_order_id="ghost",
        ticker="GOOGL",
        broker_symbol="GOOGL",
        asset_class="stock",
        vehicle="shares",
        target_position_kind="long_term",
        side="buy",
        requested_qty=3.0,
        requested_limit_price=175.0,
        submitted_at=_SUBMITTED_AT,
        intent_payload_json=_long_term_payload(),
        lifecycle_status="partially_filled",
        filled_qty=1.0,
        filled_avg_price=174.0,
    )
    result = order_lifecycle.materialize_pending_order_fill(ghost)
    assert result.action == order_lifecycle.MATERIALIZE_SKIPPED
    assert db.get_long_term_positions() == []


# ── failure without mutation ─────────────────────────────────────────────────


def test_regressed_cumulative_fill_fails_without_mutation(tmp_db: Path) -> None:
    pending = _insert_pending(
        "regress-1",
        target_position_kind="long_term",
        vehicle="shares",
        side="buy",
        intent_payload_json=_long_term_payload(),
        filled_qty=2.0,
        filled_avg_price=174.0,
    )
    created = order_lifecycle.materialize_pending_order_fill(pending)
    assert created.position_id is not None

    # Simulate a position row that somehow got ahead of the ledger snapshot.
    conn = db.get_connection()
    try:
        conn.execute(
            "UPDATE long_term_positions SET qty = 5.0 WHERE id = ?",
            (created.position_id,),
        )
        conn.commit()
    finally:
        conn.close()

    result = order_lifecycle.materialize_pending_order_fill(
        _require_pending("regress-1")
    )
    assert result.action == order_lifecycle.MATERIALIZE_FAILED
    pos = db.get_long_term_position(created.position_id)
    assert pos is not None
    assert pos.qty == 5.0    # untouched


def test_diverging_average_at_same_qty_fails_without_mutation(tmp_db: Path) -> None:
    pending = _insert_pending(
        "diverge-1",
        target_position_kind="long_term",
        vehicle="shares",
        side="buy",
        intent_payload_json=_long_term_payload(),
        filled_qty=2.0,
        filled_avg_price=174.0,
    )
    created = order_lifecycle.materialize_pending_order_fill(pending)
    assert created.position_id is not None

    # Same cumulative quantity, but a different average — inconsistent.
    db.update_pending_order(_pk("diverge-1"), filled_avg_price=175.0)
    result = order_lifecycle.materialize_pending_order_fill(
        _require_pending("diverge-1")
    )
    assert result.action == order_lifecycle.MATERIALIZE_FAILED
    pos = db.get_long_term_position(created.position_id)
    assert pos is not None
    assert pos.entry_price == 174.0     # untouched


def test_unsupported_intent_version_fails_without_mutation(tmp_db: Path) -> None:
    _raw_insert_pending(
        broker_order_id="ver-2",
        intent_payload_json=_option_payload(),
        intent_payload_version=2,
    )
    pending = db.get_pending_order("ver-2")
    assert pending is not None

    result = order_lifecycle.materialize_pending_order_fill(pending)
    assert result.action == order_lifecycle.MATERIALIZE_FAILED
    assert "version" in result.reason
    assert db.get_option_positions() == []


def test_malformed_payload_json_fails_without_mutation(tmp_db: Path) -> None:
    _raw_insert_pending(
        broker_order_id="bad-json",
        intent_payload_json="{not valid json",
    )
    pending = db.get_pending_order("bad-json")
    assert pending is not None

    result = order_lifecycle.materialize_pending_order_fill(pending)
    assert result.action == order_lifecycle.MATERIALIZE_FAILED
    assert db.get_option_positions() == []


def test_inconsistent_intent_for_target_fails_without_mutation(tmp_db: Path) -> None:
    # Option target, but the payload is a long-term intent (missing option fields).
    _raw_insert_pending(
        broker_order_id="mismatch-1",
        intent_payload_json=_long_term_payload(),
        target_position_kind="option",
        vehicle="option_full",
        side="buy",
    )
    pending = db.get_pending_order("mismatch-1")
    assert pending is not None

    result = order_lifecycle.materialize_pending_order_fill(pending)
    assert result.action == order_lifecycle.MATERIALIZE_FAILED
    assert db.get_option_positions() == []


def test_cross_kind_payload_cannot_spoof_an_option_intent(tmp_db: Path) -> None:
    payload = json.loads(_option_payload())
    payload.update(
        {"intent_kind": "long_term", "source": "long_term", "direction": "long"}
    )
    _raw_insert_pending(
        broker_order_id="cross-kind-option",
        intent_payload_json=json.dumps(payload, sort_keys=True),
        target_position_kind="option",
        vehicle="option_full",
        side="buy",
    )

    result = order_lifecycle.materialize_pending_order_fill(
        _require_pending("cross-kind-option")
    )

    assert result.action == order_lifecycle.MATERIALIZE_FAILED
    assert "option intent" in result.reason
    assert db.get_option_positions() == []


def test_cross_kind_payload_cannot_spoof_a_share_intent(tmp_db: Path) -> None:
    payload = json.loads(_option_payload())
    payload.update({"source": "swing_fallback", "direction": "long"})
    _raw_insert_pending(
        broker_order_id="cross-kind-share",
        intent_payload_json=json.dumps(payload, sort_keys=True),
        target_position_kind="long_term",
        vehicle="shares",
        side="buy",
        broker_symbol="GOOGL",
    )

    result = order_lifecycle.materialize_pending_order_fill(
        _require_pending("cross-kind-share")
    )

    assert result.action == order_lifecycle.MATERIALIZE_FAILED
    assert "long_term or shares_fallback intent" in result.reason
    assert db.get_long_term_positions() == []


def test_link_failure_rolls_back_the_inserted_position(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pending = _insert_pending(
        "atomic-1",
        target_position_kind="option",
        vehicle="option_full",
        side="buy",
        intent_payload_json=_option_payload(),
        broker_symbol=_OCC_SYMBOL,
        filled_qty=1.0,
        filled_avg_price=3.0,
    )

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise sqlite3.OperationalError("simulated ledger-link failure")

    monkeypatch.setattr(db, "_link_pending_order_position", _boom)

    result = order_lifecycle.materialize_pending_order_fill(pending)

    assert result.action == order_lifecycle.MATERIALIZE_FAILED
    # The insert must have rolled back with the failed link — no orphan holding.
    assert db.get_option_positions() == []
    linked = db.get_pending_order("atomic-1")
    assert linked is not None
    assert linked.position_id is None


def test_first_fill_change_during_link_rolls_back_and_retries_current_truth(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pending = _insert_pending(
        "atomic-snapshot",
        target_position_kind="option",
        vehicle="option_full",
        side="buy",
        intent_payload_json=_option_payload(),
        broker_symbol=_OCC_SYMBOL,
        filled_qty=1.0,
        filled_avg_price=3.0,
    )
    original = db.materialize_new_option_position

    def _advance_before_link(pending_order_id: int, pos: OptionPosition) -> int:
        db.update_pending_order(
            pending_order_id, filled_qty=2.0, filled_avg_price=3.5
        )
        return original(pending_order_id, pos)

    monkeypatch.setattr(db, "materialize_new_option_position", _advance_before_link)
    failed = order_lifecycle.materialize_pending_order_fill(pending)

    assert failed.action == order_lifecycle.MATERIALIZE_FAILED
    assert "fill changed" in failed.reason
    assert db.get_option_positions() == []
    assert _require_pending("atomic-snapshot").position_id is None

    monkeypatch.setattr(db, "materialize_new_option_position", original)
    retried = order_lifecycle.materialize_pending_order_fill(
        _require_pending("atomic-snapshot")
    )
    assert retried.action == order_lifecycle.MATERIALIZE_CREATED
    assert retried.position_id is not None
    position = db.get_option_position(retried.position_id)
    assert position is not None
    assert position.contracts == 2.0
    assert position.premium_entry == 3.5


# ── batch entry point ────────────────────────────────────────────────────────


def test_batch_materializes_every_fill_bearing_row(tmp_db: Path) -> None:
    _insert_pending(
        "batch-opt",
        target_position_kind="option",
        vehicle="option_full",
        side="buy",
        intent_payload_json=_option_payload(),
        broker_symbol=_OCC_SYMBOL,
        filled_qty=1.0,
        filled_avg_price=3.0,
    )
    _insert_pending(
        "batch-lt",
        target_position_kind="long_term",
        vehicle="shares",
        side="buy",
        intent_payload_json=_long_term_payload(),
        filled_qty=3.0,
        filled_avg_price=175.0,
    )
    _insert_pending(
        "batch-zero",
        target_position_kind="long_term",
        vehicle="shares",
        side="buy",
        intent_payload_json=_long_term_payload(),
        lifecycle_status="new",
        filled_qty=0.0,
        filled_avg_price=None,
    )

    results = order_lifecycle.materialize_pending_order_fills(observed_at=_OPENED_AT)

    actions = sorted(r.action for r in results)
    assert actions == [
        order_lifecycle.MATERIALIZE_CREATED,
        order_lifecycle.MATERIALIZE_CREATED,
    ]
    assert len(db.get_option_positions()) == 1
    assert len(db.get_long_term_positions()) == 1


def _pk(broker_order_id: str) -> int:
    """Resolve a ledger row's primary key from its broker order id."""
    row = db.get_pending_order(broker_order_id)
    assert row is not None and row.id is not None
    return row.id


def _option_payload_with(**overrides: object) -> str:
    payload: dict[str, object] = {
        "intent_kind": "option",
        "option_type": "call",
        "strike": 180.0,
        "expiry": "2026-08-21",
        "multiplier": 100,
        "delta_entry": 0.55,
        "theta": -0.03,
        "vega": 0.10,
        "gamma": 0.02,
        "tp": 190.0,
        "sl": 170.0,
        "deadline": _DEADLINE,
    }
    payload.update(overrides)
    return json.dumps(payload, sort_keys=True)


@pytest.mark.parametrize(
    "payload",
    [
        _option_payload_with(option_type="spread"),
        _option_payload_with(strike="cheap"),
        _option_payload_with(strike=-1.0),
        _option_payload_with(expiry=""),
        _option_payload_with(expiry="not-an-iso-date"),
        _option_payload_with(multiplier=0),
        _option_payload_with(tp=float("inf")),
        _option_payload_with(deadline="not-a-datetime"),
    ],
)
def test_malformed_option_intent_fails_without_mutation(
    tmp_db: Path, payload: str
) -> None:
    broker_order_id = f"bad-opt-{abs(hash(payload)) % 10_000}"
    _raw_insert_pending(
        broker_order_id=broker_order_id,
        intent_payload_json=payload,
        target_position_kind="option",
        vehicle="option_full",
        side="buy",
    )
    pending = db.get_pending_order(broker_order_id)
    assert pending is not None

    result = order_lifecycle.materialize_pending_order_fill(pending)
    assert result.action == order_lifecycle.MATERIALIZE_FAILED
    assert db.get_option_positions() == []


def test_share_direction_mismatch_fails_without_mutation(tmp_db: Path) -> None:
    # A buy order whose payload claims a short direction is inconsistent.
    _raw_insert_pending(
        broker_order_id="dir-bad",
        intent_payload_json=_shares_fallback_payload("short"),
        target_position_kind="long_term",
        vehicle="shares",
        side="buy",
    )
    pending = db.get_pending_order("dir-bad")
    assert pending is not None

    result = order_lifecycle.materialize_pending_order_fill(pending)
    assert result.action == order_lifecycle.MATERIALIZE_FAILED
    assert db.get_long_term_positions() == []


def test_share_source_mismatch_fails_without_mutation(tmp_db: Path) -> None:
    bad = json.dumps(
        {"intent_kind": "long_term", "source": "swing_fallback", "direction": "long"},
        sort_keys=True,
    )
    _raw_insert_pending(
        broker_order_id="src-bad",
        intent_payload_json=bad,
        target_position_kind="long_term",
        vehicle="shares",
        side="buy",
    )
    pending = db.get_pending_order("src-bad")
    assert pending is not None

    result = order_lifecycle.materialize_pending_order_fill(pending)
    assert result.action == order_lifecycle.MATERIALIZE_FAILED
    assert db.get_long_term_positions() == []


def test_missing_linked_option_row_fails_without_mutation(tmp_db: Path) -> None:
    pending = _insert_pending(
        "orphan-link",
        target_position_kind="option",
        vehicle="option_full",
        side="buy",
        intent_payload_json=_option_payload(),
        broker_symbol=_OCC_SYMBOL,
        filled_qty=1.0,
        filled_avg_price=3.0,
    )
    created = order_lifecycle.materialize_pending_order_fill(pending)
    assert created.position_id is not None

    conn = db.get_connection()
    try:
        conn.execute(
            "DELETE FROM option_positions WHERE id = ?", (created.position_id,)
        )
        conn.commit()
    finally:
        conn.close()

    db.update_pending_order(_pk("orphan-link"), filled_qty=2.0, filled_avg_price=3.5)
    result = order_lifecycle.materialize_pending_order_fill(
        _require_pending("orphan-link")
    )
    assert result.action == order_lifecycle.MATERIALIZE_FAILED
    assert "linked option position row is missing" in result.reason


def test_missing_linked_long_term_row_fails_without_mutation(tmp_db: Path) -> None:
    pending = _insert_pending(
        "orphan-lt",
        target_position_kind="long_term",
        vehicle="shares",
        side="buy",
        intent_payload_json=_long_term_payload(),
        filled_qty=1.0,
        filled_avg_price=174.0,
    )
    created = order_lifecycle.materialize_pending_order_fill(pending)
    assert created.position_id is not None

    conn = db.get_connection()
    try:
        conn.execute(
            "DELETE FROM long_term_positions WHERE id = ?", (created.position_id,)
        )
        conn.commit()
    finally:
        conn.close()

    db.update_pending_order(_pk("orphan-lt"), filled_qty=2.0, filled_avg_price=174.5)
    result = order_lifecycle.materialize_pending_order_fill(
        _require_pending("orphan-lt")
    )
    assert result.action == order_lifecycle.MATERIALIZE_FAILED
    assert "linked long-term position row is missing" in result.reason
