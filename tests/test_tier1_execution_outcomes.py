"""Tier-1 loss counting from exact broker execution lifecycle truth."""

from __future__ import annotations

import dataclasses
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from trading_bot import config, db, order_lifecycle
from trading_bot import risk_of_ruin as ror
from trading_bot.broker.base import STATUS_FILLED, STATUS_PARTIALLY_FILLED
from trading_bot.models import (
    BrokerExecutionOutcome,
    BrokerExecutionOutcomeCandidate,
    LongTermPosition,
    OptionPosition,
    PendingOrder,
    Signal,
    Trade,
)

_BASE = datetime(2026, 7, 1, 14, 0, tzinfo=UTC)


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def __call__(self, title: str, message: str) -> bool:
        self.calls.append((title, message))
        return True


def _entry_payload(kind: str, source: str, direction: str) -> str:
    if kind == "option":
        payload: dict[str, Any] = {
            "intent_kind": "option",
            "option_type": "call",
            "strike": 100.0,
            "expiry": "2026-12-18",
            "multiplier": 100,
            "delta_entry": None,
            "theta": None,
            "vega": None,
            "gamma": None,
            "tp": None,
            "sl": None,
            "deadline": None,
        }
    else:
        payload = {
            "intent_kind": (
                "long_term" if source == "long_term" else "shares_fallback"
            ),
            "source": source,
            "direction": direction,
        }
        if source == "swing_fallback":
            payload.update({"tp": 120.0, "sl": 80.0, "deadline": None})
    return json.dumps(payload, sort_keys=True)


def _insert_entry(
    label: str,
    *,
    kind: str = "long_term",
    source: str = "long_term",
    direction: str = "long",
    qty: float = 1.0,
    entry_price: float = 100.0,
) -> int:
    entered_at = _BASE - timedelta(days=1)
    ticker = f"T{label}"
    if kind == "option":
        symbol = f"OPT-{label}"
        position_id = db.insert_option_position(
            OptionPosition(
                symbol=symbol,
                underlying=ticker,
                option_type="call",
                strike=100.0,
                expiry="2026-12-18",
                contracts=qty,
                opened_at=entered_at,
                premium_entry=entry_price,
                outcome="open",
            )
        )
        broker_symbol = symbol
        asset_class = "stock"
        vehicle = "option_full"
        side = "buy"
    else:
        position_id = db.insert_long_term_position(
            LongTermPosition(
                ticker=ticker,
                asset_class="stock",
                entry_price=entry_price,
                entry_date=entered_at,
                qty=qty,
                source=source,
                direction=direction,
                tp=120.0 if source == "swing_fallback" else None,
                sl=80.0 if source == "swing_fallback" else None,
            )
        )
        broker_symbol = ticker
        asset_class = "stock"
        vehicle = "shares"
        side = "sell" if direction == "short" else "buy"
    db.insert_pending_order(
        PendingOrder(
            client_order_id=f"entry-{label}",
            broker_order_id=f"broker-entry-{label}",
            ticker=ticker,
            broker_symbol=broker_symbol,
            asset_class=asset_class,
            vehicle=vehicle,
            target_position_kind=kind,
            side=side,
            requested_qty=qty,
            requested_limit_price=entry_price + 50.0,
            submitted_at=entered_at - timedelta(minutes=1),
            intent_payload_json=_entry_payload(kind, source, direction),
            lifecycle_status=STATUS_FILLED,
            broker_status=STATUS_FILLED,
            filled_qty=qty,
            filled_avg_price=entry_price,
            last_fill_at=entered_at,
            last_fill_time_source="broker",
            terminal_reason=STATUS_FILLED,
            terminal_at=entered_at,
            position_kind=kind,
            position_id=position_id,
        )
    )
    return position_id


def _insert_exit(
    label: str,
    position_id: int,
    *,
    kind: str = "long_term",
    direction: str = "long",
    qty: float = 1.0,
    requested_qty: float | None = None,
    fill_price: float = 90.0,
    fill_at: datetime,
    status: str = STATUS_FILLED,
    submitted_offset: int = 0,
    requested_price: float = 999.0,
) -> int:
    if kind == "option":
        position = db.get_option_position(position_id)
        assert position is not None
        ticker = position.underlying
        broker_symbol = position.symbol
        vehicle = position.vehicle
        side = "sell"
    else:
        position = db.get_long_term_position(position_id)
        assert position is not None
        ticker = broker_symbol = position.ticker
        vehicle = "shares"
        side = "buy" if direction == "short" else "sell"
    terminal = status == STATUS_FILLED
    return db.insert_pending_order(
        PendingOrder(
            client_order_id=f"exit-{label}",
            broker_order_id=f"broker-exit-{label}",
            order_role="exit",
            ticker=ticker,
            broker_symbol=broker_symbol,
            asset_class="stock",
            vehicle=vehicle,
            target_position_kind=kind,
            closes_position_kind=kind,
            closes_position_id=position_id,
            side=side,
            requested_qty=requested_qty if requested_qty is not None else qty,
            requested_limit_price=requested_price,
            submitted_at=fill_at - timedelta(minutes=1) + timedelta(
                seconds=submitted_offset
            ),
            intent_payload_json=json.dumps(
                {"intent_kind": "position_exit", "exit_reason": "risk_exit"},
                sort_keys=True,
            ),
            lifecycle_status=status,
            broker_status=status,
            filled_qty=qty,
            filled_avg_price=fill_price,
            last_fill_at=fill_at,
            last_fill_time_source="broker",
            terminal_reason=STATUS_FILLED if terminal else None,
            terminal_at=fill_at if terminal else None,
        )
    )


def _insert_closed(
    label: str,
    *,
    kind: str = "long_term",
    source: str = "long_term",
    direction: str = "long",
    outcome: str = "loss",
    fill_at: datetime = _BASE,
) -> tuple[int, int]:
    entry_price = 5.0 if kind == "option" else 100.0
    position_id = _insert_entry(
        label,
        kind=kind,
        source=source,
        direction=direction,
        entry_price=entry_price,
    )
    delta = {"loss": -1.0, "win": 1.0, "breakeven": 0.0}[outcome]
    if direction == "short":
        delta = -delta
    exit_id = _insert_exit(
        label,
        position_id,
        kind=kind,
        direction=direction,
        fill_price=entry_price + delta,
        fill_at=fill_at,
    )
    result = order_lifecycle.materialize_position_exit_fills(kind, position_id)
    assert result.action == order_lifecycle.EXIT_FILL_CLOSED
    assert result.outcome == outcome
    return position_id, exit_id


def _event(
    outcome: str,
    offset: int,
    *,
    kind: str = "option",
) -> BrokerExecutionOutcome:
    return BrokerExecutionOutcome(
        position_kind=kind,
        position_id=offset + 1,
        outcome=outcome,
        final_fill_at=_BASE - timedelta(minutes=offset),
        final_exit_pending_order_id=1000 - offset,
    )


@pytest.mark.parametrize(
    ("kind", "source", "direction"),
    [
        ("option", "option", "long"),
        ("long_term", "long_term", "long"),
        ("long_term", "swing_fallback", "long"),
        ("long_term", "swing_fallback", "short"),
    ],
)
def test_exact_typed_execution_sources_count(
    tmp_db: Path,
    kind: str,
    source: str,
    direction: str,
) -> None:
    _insert_closed(
        f"{kind}-{source}-{direction}",
        kind=kind,
        source=source,
        direction=direction,
    )
    (candidate,) = db.get_broker_execution_outcome_candidates()
    assert candidate.position_kind == kind
    assert candidate.position_source == source
    assert candidate.position_direction == direction
    assert ror.consecutive_losses() == 1


def test_models_are_immutable() -> None:
    candidate = BrokerExecutionOutcomeCandidate(
        "option", 1, True, True, "option", "long", _BASE, 4.0,
        "risk_exit", "loss", -100.0, _BASE, 2,
    )
    outcome = _event("loss", 0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        candidate.__setattr__("position_id", 2)
    with pytest.raises(dataclasses.FrozenInstanceError):
        outcome.__setattr__("outcome", "win")


def test_trades_and_unlinked_legacy_closes_are_excluded(tmp_db: Path) -> None:
    signal_id = db.insert_signal(
        Signal(
            timestamp=_BASE,
            ticker="PAPER",
            asset_class="stock",
            signal_type="ema21_pullback",
            direction="call",
            entry_price=100.0,
        )
    )
    db.insert_trade(
        Trade(
            signal_id=signal_id,
            opened_at=_BASE,
            closed_at=_BASE + timedelta(hours=1),
            outcome="loss",
            track_mode="active",
        )
    )
    legacy_id = _insert_entry("legacy", kind="option", entry_price=5.0)
    db.update_option_position(
        legacy_id,
        closed_at=_BASE,
        exit_price=4.0,
        exit_reason="legacy",
        outcome="loss",
        pnl_dollars=-100.0,
    )
    assert db.get_broker_execution_outcome_candidates() == []
    assert ror.consecutive_losses() == 0


def test_multi_exit_uses_actual_aggregate_not_requested_price(tmp_db: Path) -> None:
    position_id = _insert_entry("multi", kind="option", qty=2.0, entry_price=5.0)
    first_at = _BASE
    _insert_exit(
        "multi-a", position_id, kind="option", qty=1.0, requested_qty=1.0,
        fill_price=4.0, fill_at=first_at, requested_price=500.0,
    )
    final_at = _BASE + timedelta(minutes=1)
    final_id = _insert_exit(
        "multi-b", position_id, kind="option", qty=1.0, requested_qty=1.0,
        fill_price=6.0, fill_at=final_at, requested_price=600.0,
        submitted_offset=1,
    )
    result = order_lifecycle.materialize_position_exit_fills("option", position_id)
    assert result.exit_vwap == 5.0
    assert result.outcome == "breakeven"
    (outcome,) = ror._ordered_broker_execution_outcomes()
    assert outcome.outcome == "breakeven"
    assert outcome.final_fill_at == final_at
    assert outcome.final_exit_pending_order_id == final_id
    assert ror.consecutive_losses() == 0


def test_open_and_full_nonterminal_exits_emit_no_event(tmp_db: Path) -> None:
    partial_id = _insert_entry("partial", qty=2.0)
    _insert_exit(
        "partial", partial_id, qty=1.0, requested_qty=2.0,
        fill_price=90.0, fill_at=_BASE,
    )
    pending_id = _insert_entry("pending")
    _insert_exit(
        "pending", pending_id, fill_price=90.0,
        fill_at=_BASE + timedelta(minutes=1), status=STATUS_PARTIALLY_FILLED,
    )
    assert len(db.get_broker_execution_outcome_candidates()) == 2
    assert ror._ordered_broker_execution_outcomes() == []
    assert ror.consecutive_losses() == 0


def test_ready_close_is_unknown_until_typed_summary_materializes(tmp_db: Path) -> None:
    position_id = _insert_entry("ready")
    _insert_exit("ready", position_id, fill_price=90.0, fill_at=_BASE)
    state = order_lifecycle.position_exit_fill_state("long_term", position_id)
    assert state.action == order_lifecycle.EXIT_FILL_READY
    assert ror.consecutive_losses() is None


def test_typed_summary_mismatch_is_unknown(tmp_db: Path) -> None:
    position_id, _ = _insert_closed("mismatch")
    conn = db.get_connection()
    try:
        conn.execute(
            "UPDATE long_term_positions SET exit_price = ? WHERE id = ?",
            (999.0, position_id),
        )
        conn.commit()
    finally:
        conn.close()
    assert ror.consecutive_losses() is None


def test_missing_target_is_unknown(tmp_db: Path) -> None:
    db.insert_pending_order(
        PendingOrder(
            client_order_id="exit-missing",
            broker_order_id="broker-exit-missing",
            order_role="exit",
            ticker="MISSING",
            broker_symbol="MISSING",
            asset_class="stock",
            vehicle="shares",
            target_position_kind="long_term",
            closes_position_kind="long_term",
            closes_position_id=999,
            side="sell",
            requested_qty=1.0,
            requested_limit_price=99.0,
            submitted_at=_BASE - timedelta(minutes=1),
            intent_payload_json=json.dumps(
                {"intent_kind": "position_exit", "exit_reason": "risk_exit"}
            ),
            lifecycle_status=STATUS_FILLED,
            broker_status=STATUS_FILLED,
            filled_qty=1.0,
            filled_avg_price=90.0,
            last_fill_at=_BASE,
            last_fill_time_source="broker",
            terminal_reason=STATUS_FILLED,
            terminal_at=_BASE,
        )
    )
    assert ror.consecutive_losses() is None


def test_oversold_integrity_and_malformed_time_are_unknown(tmp_db: Path) -> None:
    oversold_id = _insert_entry("oversold")
    _insert_exit(
        "oversold", oversold_id, qty=2.0, requested_qty=2.0,
        fill_price=90.0, fill_at=_BASE,
    )
    assert ror.consecutive_losses() is None

    conn = db.get_connection()
    try:
        conn.execute("DELETE FROM pending_orders")
        conn.execute("DELETE FROM long_term_positions")
        conn.commit()
    finally:
        conn.close()
    malformed_id = _insert_entry("malformed")
    exit_id = _insert_exit(
        "malformed", malformed_id, fill_price=90.0, fill_at=_BASE,
    )
    conn = db.get_connection()
    try:
        conn.execute(
            "UPDATE pending_orders SET last_fill_at = 'not-a-time' WHERE id = ?",
            (exit_id,),
        )
        conn.commit()
    finally:
        conn.close()
    assert ror.consecutive_losses() is None


def test_deterministic_actual_fill_order_ties(monkeypatch: pytest.MonkeyPatch) -> None:
    candidates = [
        BrokerExecutionOutcomeCandidate(
            "long_term", 1, True, True, "long_term", "long", _BASE, 90.0,
            "risk_exit", None, None, _BASE, 42,
        ),
        BrokerExecutionOutcomeCandidate(
            "option", 1, True, True, "option", "long", _BASE, 4.0,
            "risk_exit", "loss", -100.0, _BASE, 42,
        ),
        BrokerExecutionOutcomeCandidate(
            "option", 2, True, True, "option", "long", _BASE, 4.0,
            "risk_exit", "loss", -100.0, _BASE, 42,
        ),
    ]
    monkeypatch.setattr(db, "get_broker_execution_outcome_candidates", lambda: candidates)

    def state(kind: str, position_id: int) -> order_lifecycle.ExitFillMaterializationResult:
        return order_lifecycle.ExitFillMaterializationResult(
            kind, position_id, order_lifecycle.EXIT_FILL_UNCHANGED, True,
            entry_qty=1.0, exited_qty=1.0, remaining_qty=0.0,
            exit_vwap=4.0 if kind == "option" else 90.0,
            final_fill_at=_BASE, exit_reason="risk_exit", pnl_dollars=-100.0,
            outcome="loss", final_exit_pending_order_id=42,
        )

    monkeypatch.setattr(order_lifecycle, "position_exit_fill_state", state)
    ordered = ror._ordered_broker_execution_outcomes()
    assert [(event.position_kind, event.position_id) for event in ordered] == [
        ("option", 2), ("option", 1), ("long_term", 1),
    ]


def test_unbounded_query_sees_many_newer_neutral_events(tmp_db: Path) -> None:
    for index in range(6):
        _insert_closed(
            f"loss-{index}", outcome="loss", fill_at=_BASE + timedelta(minutes=index)
        )
    for index in range(25):
        _insert_closed(
            f"neutral-{index}", outcome="breakeven",
            fill_at=_BASE + timedelta(hours=1, minutes=index),
        )
    assert len(db.get_broker_execution_outcome_candidates()) == 31
    assert ror.consecutive_losses() == 6


@pytest.mark.parametrize(
    ("stream", "expected"),
    [
        ([], 0),
        (["win", "loss"], 0),
        (["loss", "loss", "win"], 2),
        (["breakeven", "loss", "win"], 1),
        (["expired", "loss", "win"], 1),
    ],
)
def test_counter_win_loss_breakeven_and_exact_expiry_semantics(
    monkeypatch: pytest.MonkeyPatch,
    stream: list[str],
    expected: int,
) -> None:
    monkeypatch.setattr(
        ror,
        "_ordered_broker_execution_outcomes",
        lambda: [_event(value, index) for index, value in enumerate(stream)],
    )
    assert ror.consecutive_losses() == expected


@pytest.mark.parametrize(
    ("stream", "expected"),
    [
        (["loss"] * 6 + ["unknown"], None),
        (["loss"] * 7 + ["unknown"], 7),
        (["loss", "loss", "win", "unknown"], 2),
        (["unknown", "loss"] * 4, None),
    ],
)
def test_unknown_before_after_threshold_and_older_than_win(
    monkeypatch: pytest.MonkeyPatch,
    stream: list[str],
    expected: int | None,
) -> None:
    monkeypatch.setattr(
        ror,
        "_ordered_broker_execution_outcomes",
        lambda: [_event(value, index) for index, value in enumerate(stream)],
    )
    assert ror.consecutive_losses() == expected


def test_db_and_analyzer_failures_log_and_return_unknown(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def db_boom() -> list[BrokerExecutionOutcomeCandidate]:
        raise sqlite3.OperationalError("db unavailable")

    monkeypatch.setattr(db, "get_broker_execution_outcome_candidates", db_boom)
    assert ror.consecutive_losses() is None
    assert "db unavailable" in capsys.readouterr().err

    candidate = BrokerExecutionOutcomeCandidate(
        "option", 1, True, True, "option", "long", _BASE, 4.0,
        "risk_exit", "loss", -100.0, _BASE, 2,
    )
    monkeypatch.setattr(db, "get_broker_execution_outcome_candidates", lambda: [candidate])

    def analyzer_boom(_kind: str, _position_id: int) -> Any:
        raise RuntimeError("analyzer failed")

    monkeypatch.setattr(order_lifecycle, "position_exit_fill_state", analyzer_boom)
    assert ror.consecutive_losses() is None
    assert "analyzer failed" in capsys.readouterr().err


def test_unknown_losses_do_not_block_independent_drawdown(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ror, "consecutive_losses", lambda: None)
    db.insert_equity_snapshot(10_000.0, _BASE)
    db.insert_equity_snapshot(8_000.0, _BASE + timedelta(minutes=1))
    result = ror.evaluate_tier1(notifier=_Recorder())
    assert result.tripped is True
    assert result.trigger == "drawdown"
    assert result.losses is None


@pytest.mark.parametrize("state", [ror.STATE_PAUSED, ror.STATE_HOLDING, ror.STATE_HALTED])
def test_unknown_preserves_existing_non_normal_state(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    state: str,
) -> None:
    ror._set_state(state)
    ror.revoke(config.ENTRY_CAPABILITY, f"existing {state}")
    monkeypatch.setattr(ror, "consecutive_losses", lambda: None)
    recorder = _Recorder()
    result = ror.evaluate_tier1(notifier=recorder)
    assert result.state == state
    assert result.tripped is False
    assert ror.get_state() == state
    assert ror.revoke_reason(config.ENTRY_CAPABILITY) == f"existing {state}"
    assert recorder.calls == []


@pytest.mark.parametrize(("losses", "trips"), [(6, False), (7, True)])
def test_exact_six_and_seven_loss_threshold(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    losses: int,
    trips: bool,
) -> None:
    monkeypatch.setattr(
        ror,
        "_ordered_broker_execution_outcomes",
        lambda: [_event("loss", index) for index in range(losses)],
    )
    result = ror.evaluate_tier1(notifier=_Recorder())
    assert result.tripped is trips
    assert result.losses == losses


def test_execution_outcome_read_has_no_broker_call_or_mutation(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    position_id, _ = _insert_closed("readonly")
    before = db.get_long_term_position(position_id)

    def network_boom(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("network must not be called")

    monkeypatch.setattr(ror.requests, "get", network_boom, raising=False)
    monkeypatch.setattr(ror.requests, "post", network_boom)
    assert ror.consecutive_losses() == 1
    assert db.get_long_term_position(position_id) == before
    assert len(db.get_pending_exit_orders_for_position("long_term", position_id)) == 1


def test_cli_renders_unknown_loss_truth(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_bot import __main__ as main

    monkeypatch.setattr(ror, "consecutive_losses", lambda: None)
    main.cmd_risk_ror_status()
    assert "Consecutive losses:    unknown / 7 limit" in capsys.readouterr().out


# ───────── operator re-authorization must actually clear the Tier-1 pause ────


def test_reauthorize_breaks_the_tier1_loss_deadlock(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pause must not come straight back on the next hourly cycle.

    The loss streak is DERIVED from closed positions, so `reauthorize` — which
    zeroes both Tier-2 counters — had nothing to zero here. `evaluate_tier1`
    re-read the same seven losses, re-revoked entry and re-paused within the
    hour. That is unrecoverable, not merely annoying: the only thing that
    breaks a streak is a WIN, a win needs a new position, and new positions are
    exactly what the pause revokes.
    """
    losses = [_event("loss", index) for index in range(7)]
    monkeypatch.setattr(ror, "_ordered_broker_execution_outcomes", lambda: losses)

    assert ror.consecutive_losses() == config.MAX_CONSECUTIVE_LOSSES
    assert ror.evaluate_tier1(notifier=_Recorder()).tripped is True
    assert ror.get_state() == ror.STATE_PAUSED

    ok, _msg = ror.reauthorize(config.ROR_REAUTHORIZE_TOKEN)
    assert ok is True

    # The next hourly cycle, with the SAME closed book and nothing new traded.
    assert ror.consecutive_losses() == 0
    assert ror.evaluate_tier1(notifier=_Recorder()).tripped is False
    assert ror.get_state() == ror.STATE_NORMAL
    assert ror.is_entry_authorized() is True


def test_a_fresh_loss_run_after_reauthorization_still_trips(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The breaker is acknowledged, not disabled: MAX_CONSECUTIVE_LOSSES NEW
    real-execution losses trip it again exactly as before."""
    book = [_event("loss", index) for index in range(7)]
    monkeypatch.setattr(ror, "_ordered_broker_execution_outcomes", lambda: book)
    ror.evaluate_tier1(notifier=_Recorder())
    ror.reauthorize(config.ROR_REAUTHORIZE_TOKEN)
    assert ror.consecutive_losses() == 0

    # Seven newer losses arrive (offset -1.. -7 -> newer than the whole book).
    fresh = [_event("loss", -(index + 1)) for index in range(7)]
    book[:0] = fresh
    assert ror.consecutive_losses() == config.MAX_CONSECUTIVE_LOSSES
    assert ror.evaluate_tier1(notifier=_Recorder()).tripped is True
    assert ror.get_state() == ror.STATE_PAUSED


def test_acknowledgement_only_hides_events_at_or_older_than_the_watermark(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single newer loss is still counted; the acknowledged ones are not."""
    book = [_event("loss", index) for index in range(7)]
    monkeypatch.setattr(ror, "_ordered_broker_execution_outcomes", lambda: book)
    ror.reauthorize(config.ROR_REAUTHORIZE_TOKEN)

    book.insert(0, _event("loss", -1))
    assert ror.consecutive_losses() == 1


def test_unreadable_acknowledgement_falls_back_to_counting_everything(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Fail STRICTER: a mangled watermark must never silently mute the breaker."""
    from trading_bot import settings

    book = [_event("loss", index) for index in range(7)]
    monkeypatch.setattr(ror, "_ordered_broker_execution_outcomes", lambda: book)
    settings.set(ror._TIER1_LOSS_ACK_KEY, "not-a-timestamp|nope")

    assert ror.consecutive_losses() == config.MAX_CONSECUTIVE_LOSSES
    assert "unreadable tier1 acknowledgement" in capsys.readouterr().err


def test_a_naive_acknowledgement_is_ignored_rather_than_crashing_the_cycle(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Regression: a timezone-less watermark PARSES, then poisons the compare.

    `datetime.fromisoformat("2026-08-01T10:00:00")` raises no ValueError, so it
    slipped past the unreadable-value guard and reached
    `(fill_at, order_id) <= acknowledged` — where every `final_fill_at` is
    guaranteed tz-aware. That raises TypeError, which is NOT caught in
    `consecutive_losses` (its try wraps only the outcomes fetch), so it
    propagated through `evaluate_tier1` and aborted the whole hourly risk cycle:
    no Tier-1 evaluation, no reconciliation, no catastrophic check, and no
    STATE_HOLDING retry of pending emergency closes — every hour, until the
    settings row was repaired by hand.

    Treated like any other unusable value: ignored, so the FULL streak counts
    (stricter, never looser).
    """
    from trading_bot import settings

    book = [_event("loss", index) for index in range(7)]
    monkeypatch.setattr(ror, "_ordered_broker_execution_outcomes", lambda: book)
    settings.set(ror._TIER1_LOSS_ACK_KEY, "2026-08-01T10:00:00|1000")

    assert ror.consecutive_losses() == config.MAX_CONSECUTIVE_LOSSES
    assert "has no timezone" in capsys.readouterr().err


def test_a_tz_aware_acknowledgement_still_bounds_the_streak(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Control: the normal write path stores a tz-aware stamp, which must keep
    working exactly as before — the fix must not reject a valid watermark."""
    from trading_bot import settings

    book = [_event("loss", index) for index in range(7)]
    ack = book[2]
    assert ack.final_fill_at is not None
    settings.set(
        ror._TIER1_LOSS_ACK_KEY,
        f"{ack.final_fill_at.isoformat()}|{ack.final_exit_pending_order_id}",
    )
    monkeypatch.setattr(ror, "_ordered_broker_execution_outcomes", lambda: book)

    # Only the two events newer than the acknowledged one still count.
    assert ror.consecutive_losses() == 2


def test_reauthorize_survives_an_unreadable_outcome_set(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """It must still restore state. The streak is indeterminate in this case,
    so `consecutive_losses` returns None and cannot trip anyway."""
    def boom() -> list[BrokerExecutionOutcome]:
        raise sqlite3.OperationalError("db unavailable")

    monkeypatch.setattr(ror, "_ordered_broker_execution_outcomes", boom)
    ror._set_state(ror.STATE_PAUSED)
    ror.revoke(config.ENTRY_CAPABILITY, "tier1: test")

    ok, _msg = ror.reauthorize(config.ROR_REAUTHORIZE_TOKEN)

    assert ok is True
    assert ror.get_state() == ror.STATE_NORMAL
    assert ror.is_entry_authorized() is True
    assert "could not read execution outcomes" in capsys.readouterr().err
    assert ror.consecutive_losses() is None
