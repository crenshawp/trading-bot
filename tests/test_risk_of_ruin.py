"""Tests for the risk-of-ruin safety layer (Phase 15).

Mocked data only, no live network. Tier 1 (pause via authorization revoke),
catastrophic detection, and the Tier 2 emergency shutdown orchestrator are the
core; equity snapshots drive the drawdown check so unrealized losses count.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from trading_bot import allocation, config, db
from trading_bot import risk_of_ruin as ror
from trading_bot.broker.base import STATUS_FILLED, AccountInfo, Position
from trading_bot.broker.fake import FakeBroker
from trading_bot.models import (
    LongTermPosition,
    OptionPosition,
    PendingOrder,
    Signal,
    Trade,
)

_TS = datetime(2026, 1, 1, 9, 0, tzinfo=UTC)


class _Recorder:
    """A fake notifier recording (title, message) calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def __call__(self, title: str, message: str) -> bool:
        self.calls.append((title, message))
        return True


# ───────────────────────── config sanity ────────────────────────────────────


def test_risk_of_ruin_config_defaults() -> None:
    assert config.MAX_CONSECUTIVE_LOSSES == 7
    assert config.MAX_DRAWDOWN_PCT == 20.0
    assert config.MAX_CONSECUTIVE_BROKER_ERRORS == 3
    assert config.ENTRY_CAPABILITY == "new_position_entry"


# ───────────────────────── equity snapshots (drawdown source) ───────────────


def test_equity_snapshot_round_trip(tmp_db: Path) -> None:
    db.insert_equity_snapshot(10_000.0, _TS)
    db.insert_equity_snapshot(11_000.0, _TS + timedelta(hours=1))
    db.insert_equity_snapshot(9_000.0, _TS + timedelta(hours=2))
    assert db.get_equity_peak() == 11_000.0        # running max
    assert db.get_latest_equity() == 9_000.0       # most recent


def test_equity_snapshot_empty_returns_none(tmp_db: Path) -> None:
    assert db.get_equity_peak() is None
    assert db.get_latest_equity() is None


# ───────────────────────── recent resolved outcomes (streak source) ─────────


def _seed_resolved(
    outcomes: list[str],
    *,
    track_mode: str = "active",
    signal_type: str = "ema21_pullback",
    asset_class: str = "stock",
    ticker_prefix: str = "T",
    start_minute: int = 0,
) -> None:
    """Seed resolved trades chronologically. ``start_minute`` orders separate
    seeding calls against each other (later minutes close later)."""
    for i, outcome in enumerate(outcomes):
        ts = _TS + timedelta(minutes=start_minute + i)
        sid = db.insert_signal(Signal(
            timestamp=ts, ticker=f"{ticker_prefix}{i}", asset_class=asset_class,
            signal_type=signal_type, direction="call", entry_price=100.0,
        ))
        db.insert_trade(Trade(
            signal_id=sid, opened_at=ts,
            closed_at=ts + timedelta(days=1),
            outcome=outcome, pnl_pct=1.0 if outcome == "win" else -1.0,
            track_mode=track_mode,
        ))


def test_get_recent_resolved_outcomes_newest_first(tmp_db: Path) -> None:
    _seed_resolved(["win", "loss", "loss"])        # chronological order
    assert db.get_recent_resolved_outcomes() == ["loss", "loss", "win"]


def test_get_recent_resolved_outcomes_excludes_shadow_and_open(tmp_db: Path) -> None:
    _seed_resolved(["loss"], track_mode="shadow")
    _seed_resolved(["win"])
    sid = db.insert_signal(Signal(
        timestamp=_TS + timedelta(hours=5), ticker="OPEN", asset_class="stock",
        signal_type="ema21_pullback", direction="call", entry_price=100.0,
    ))
    db.insert_trade(Trade(signal_id=sid, opened_at=_TS, outcome="open"))
    assert db.get_recent_resolved_outcomes() == ["win"]


# ───────────────────────── authorization ledger (Phase 10 pattern) ──────────


def test_authorization_defaults_to_authorized(tmp_db: Path) -> None:
    assert ror.is_authorized("new_position_entry") is True   # no row == authorized
    assert ror.is_entry_authorized() is True
    assert ror.revoke_reason("new_position_entry") is None


def test_revoke_and_authorize_round_trip(tmp_db: Path) -> None:
    ror.revoke("new_position_entry", "tier1: test")
    assert ror.is_entry_authorized() is False
    assert ror.revoke_reason("new_position_entry") == "tier1: test"
    ror.authorize("new_position_entry")
    assert ror.is_entry_authorized() is True
    assert ror.revoke_reason("new_position_entry") is None


# ───────────────────────── Tier 1: consecutive losses ───────────────────────


def test_consecutive_losses_ignores_trade_tracking_rows(tmp_db: Path) -> None:
    _seed_resolved(["loss", "win", "loss", "loss"])   # chronological
    assert ror.consecutive_losses() == 0


def test_tier1_does_not_trip_at_six_losses(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_resolved(["win"] + ["loss"] * 6)
    monkeypatch.setattr(ror, "consecutive_losses", lambda: 6)
    rec = _Recorder()
    result = ror.evaluate_tier1(notifier=rec)
    assert result.tripped is False
    assert ror.is_entry_authorized() is True
    assert rec.calls == []


def test_tier1_trips_at_exactly_seven_losses(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_resolved(["win"] + ["loss"] * 7)
    monkeypatch.setattr(ror, "consecutive_losses", lambda: 7)
    rec = _Recorder()
    result = ror.evaluate_tier1(notifier=rec)
    assert result.tripped is True and result.trigger == "consecutive_losses"
    assert ror.is_entry_authorized() is False          # entry revoked
    assert ror.get_state() == ror.STATE_PAUSED
    assert len(rec.calls) == 1
    assert "PAUSED" in rec.calls[0][0]


def test_get_recent_resolved_outcomes_excludes_each_data_only_type(
    tmp_db: Path,
) -> None:
    """Query-level proof: every data-only signal type is excluded at the
    source; real stock swing outcomes still flow."""
    for i, signal_type in enumerate(sorted(config.DATA_ONLY_SIGNAL_TYPES)):
        _seed_resolved(
            ["loss"], signal_type=signal_type, asset_class="crypto",
            ticker_prefix=f"X{i}-", start_minute=i,
        )
    _seed_resolved(["win"], ticker_prefix="REAL-", start_minute=50)

    assert db.get_recent_resolved_outcomes() == ["win"]


# ─────────────── Tier 1 scope (Phase 23): data-only losses never count ──────


def test_data_only_losing_streak_does_not_trip_tier1(tmp_db: Path) -> None:
    """The observed failure mode: a losing streak living entirely in the
    data-only crypto swing signals — active-track but permanently barred from
    execution — must NOT pause real entry. All three detector setups covered,
    well past the 7-loss threshold."""
    for i, signal_type in enumerate(
        ("oversold_reversal", "momentum_breakout", "overbought_reversal")
    ):
        _seed_resolved(
            ["loss"] * 4, signal_type=signal_type, asset_class="crypto",
            ticker_prefix=f"C{i}-", start_minute=i * 10,
        )

    assert ror.consecutive_losses() == 0            # invisible to the breaker
    rec = _Recorder()
    result = ror.evaluate_tier1(notifier=rec)
    assert result.tripped is False
    assert ror.is_entry_authorized() is True        # real entry NOT paused
    assert ror.get_state() == ror.STATE_NORMAL
    assert rec.calls == []


def test_active_trade_tracking_loss_streak_does_not_trip(tmp_db: Path) -> None:
    """Active trade tracking is not broker execution truth."""
    _seed_resolved(["win"] + ["loss"] * 7)
    rec = _Recorder()
    result = ror.evaluate_tier1(notifier=rec)
    assert result.tripped is False
    assert result.losses == 0
    assert ror.is_entry_authorized() is True
    assert ror.get_state() == ror.STATE_NORMAL
    assert rec.calls == []


def test_mixed_trade_tracking_streaks_are_all_excluded(tmp_db: Path) -> None:
    """Neither stock nor data-only trade rows are execution outcomes."""
    _seed_resolved(
        ["loss"] * 6, ticker_prefix="REAL-", start_minute=0,      # real: 6
    )
    _seed_resolved(
        ["loss"] * 5, signal_type="oversold_reversal", asset_class="crypto",
        ticker_prefix="DATA-", start_minute=100,                  # newer, data-only
    )

    assert ror.consecutive_losses() == 0
    result = ror.evaluate_tier1(notifier=_Recorder())
    assert result.tripped is False                  # 6 < 7 — no trip
    assert ror.is_entry_authorized() is True

    # More trade-tracking rows still cannot create a broker loss event.
    _seed_resolved(["loss"], ticker_prefix="REAL7-", start_minute=200)
    assert ror.consecutive_losses() == 0
    result = ror.evaluate_tier1(notifier=_Recorder())
    assert result.tripped is False


def test_pre_existing_pause_is_not_auto_cleared_by_scope_fix(
    tmp_db: Path,
) -> None:
    """A pause tripped BEFORE this fix (e.g. by data-only noise) stays paused:
    the scope change alters what COUNTS, never the persisted state. Clearing
    still requires the explicit operator token, per Phase 15's design."""
    # Simulate the pre-fix trip: revoked entry + paused state, and a book
    # whose only losses are data-only (which the scoped counter now ignores).
    ror.revoke("new_position_entry", "tier1: 13 consecutive losses")
    ror._set_state(ror.STATE_PAUSED)
    _seed_resolved(
        ["loss"] * 8, signal_type="oversold_reversal", asset_class="crypto",
        ticker_prefix="D-",
    )
    assert ror.consecutive_losses() == 0            # scoped counter is calm...

    result = ror.evaluate_tier1(notifier=_Recorder())
    assert result.state == ror.STATE_PAUSED         # ...but the pause HOLDS
    assert ror.is_entry_authorized() is False       # still revoked
    assert ror.get_state() == ror.STATE_PAUSED

    # Only the explicit operator token clears it — exactly as before.
    ok, _msg = ror.reauthorize("wrong-token")
    assert ok is False
    assert ror.get_state() == ror.STATE_PAUSED
    ok, _msg = ror.reauthorize(config.ROR_REAUTHORIZE_TOKEN)
    assert ok is True
    assert ror.is_entry_authorized() is True
    assert ror.get_state() == ror.STATE_NORMAL


# ───────────────────────── Tier 1: drawdown (incl. unrealized) ──────────────


def test_tier1_does_not_trip_below_20_pct_drawdown(tmp_db: Path) -> None:
    db.insert_equity_snapshot(10_000.0, _TS)
    db.insert_equity_snapshot(8_100.0, _TS + timedelta(hours=1))   # 19%
    result = ror.evaluate_tier1(notifier=_Recorder())
    assert result.tripped is False


def test_tier1_trips_at_exactly_20_pct_drawdown_from_unrealized(tmp_db: Path) -> None:
    # No closed trades at all — the drawdown comes from account EQUITY falling
    # (an open position's unrealized loss), which the snapshots capture.
    assert db.get_recent_resolved_outcomes() == []
    db.insert_equity_snapshot(10_000.0, _TS)
    db.insert_equity_snapshot(8_000.0, _TS + timedelta(hours=1))   # exactly 20%
    rec = _Recorder()
    result = ror.evaluate_tier1(notifier=rec)
    assert result.tripped is True and result.trigger == "drawdown"
    assert result.drawdown_pct == 20.0
    assert ror.is_entry_authorized() is False
    assert len(rec.calls) == 1


def test_tier1_leaves_existing_positions_untouched(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Open option + long-term positions exist; a Tier-1 trip must NOT close them
    # (no close calls occur — Tier 1 never touches a broker or a position row).
    db.insert_option_position(OptionPosition(
        symbol="AAPL260116C00150000", underlying="AAPL", option_type="call",
        strike=150.0, expiry="2026-01-16", contracts=1.0, opened_at=_TS,
        outcome="open",
    ))
    db.insert_long_term_position(LongTermPosition(
        ticker="META", asset_class="stock", entry_price=480.0, entry_date=_TS,
        qty=5.0, status="open",
    ))
    _seed_resolved(["loss"] * 7)
    monkeypatch.setattr(ror, "consecutive_losses", lambda: 7)
    result = ror.evaluate_tier1(notifier=_Recorder())
    assert result.tripped is True
    assert len(db.get_open_option_positions()) == 1     # untouched
    assert len(db.get_open_long_term_positions()) == 1  # untouched


def test_tier1_does_not_retrip_when_already_paused(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_resolved(["loss"] * 7)
    monkeypatch.setattr(ror, "consecutive_losses", lambda: 7)
    rec = _Recorder()
    first = ror.evaluate_tier1(notifier=rec)
    second = ror.evaluate_tier1(notifier=rec)
    assert first.tripped is True
    assert second.tripped is False                      # no re-trip spam
    assert len(rec.calls) == 1


# ───────────────────────── allocation gate (entries-empty) ──────────────────


def test_build_plan_entries_empty_when_entry_revoked() -> None:
    acct = AccountInfo(ok=True, equity=100_000.0, cash=100_000.0)
    candidate = allocation.Candidate(
        ticker="META", signal_type="ema21_pullback", direction="call",
        asset_class="stock", entry=480.0, atr=8.0, expectancy=0.9,
    )
    res = allocation.build_plan([candidate], acct, entry_authorized=False)
    assert res.ok is True
    assert res.plan.orders == []                        # entries-empty
    assert res.skipped[0].stage == "risk"
    assert "entries-paused" in res.skipped[0].reason


# ───────────────────────── catastrophic detection: broker errors ─────────────


def test_broker_error_streak_trips_at_three_consecutive(tmp_db: Path) -> None:
    ror.record_broker_result(False)
    ror.record_broker_result(False)
    assert ror.check_catastrophic() is None            # 2 < 3: calm
    ror.record_broker_result(False)
    reason = ror.check_catastrophic()
    assert reason is not None and "3 consecutive broker errors" in reason


def test_broker_error_streak_resets_on_intervening_success(tmp_db: Path) -> None:
    ror.record_broker_result(False)
    ror.record_broker_result(False)
    ror.record_broker_result(True)                     # success resets
    ror.record_broker_result(False)
    assert ror.broker_error_streak() == 1
    assert ror.check_catastrophic() is None


def test_order_paths_feed_the_broker_error_counter(tmp_db: Path) -> None:
    # A structured rejection through the options order path increments the
    # streak; a successful submit resets it (the Phase 15 wiring, end to end).
    from datetime import date

    from trading_bot import options_execution as oe
    from trading_bot.broker.fake import FakeBroker
    from trading_bot.broker.options import OptionContract

    contract = OptionContract(
        symbol="AAPL260130C00150000", underlying="AAPL", option_type="call",
        strike=150.0, expiry="2026-01-30", delta=0.70, open_interest=500,
        bid=4.9, ask=5.1, mid=5.0,
    )
    decision = oe.choose_execution(
        "call", 600.0, "AAPL", 100.0, [contract], ref_date=date(2026, 1, 1),
    )
    oe.submit_execution_order(FakeBroker(reject_reason="market closed"), decision)
    assert ror.broker_error_streak() == 1
    oe.submit_execution_order(FakeBroker(), decision)
    assert ror.broker_error_streak() == 0              # success resets


# ───────────────────────── catastrophic detection: unreconcilable ────────────


def test_reconcile_divergence_streak_trips_after_persistence(tmp_db: Path) -> None:
    ror.record_reconcile_result(False)
    ror.record_reconcile_result(False)
    assert ror.check_catastrophic() is None
    ror.record_reconcile_result(False)
    reason = ror.check_catastrophic()
    assert reason is not None and "unreconcilable" in reason


def test_reconcile_divergence_streak_resets_on_clean_check(tmp_db: Path) -> None:
    ror.record_reconcile_result(False)
    ror.record_reconcile_result(False)
    ror.record_reconcile_result(True)                  # clean reconcile resets
    assert ror.reconcile_divergence_streak() == 0
    assert ror.check_catastrophic() is None


# ───────────────────────── scanner reconciliation wiring ────────────────────


def _stub_scanner_risk_cycle(
    monkeypatch: pytest.MonkeyPatch, reconcile_fn: Any,
) -> tuple[Any, FakeBroker]:
    from trading_bot import scanner

    broker = FakeBroker()
    monkeypatch.setattr("trading_bot.broker.AlpacaBroker", lambda: broker)
    monkeypatch.setattr("trading_bot.broker.reconcile", reconcile_fn)
    monkeypatch.setattr(ror, "record_equity_snapshot", lambda _equity: 1)
    monkeypatch.setattr(ror, "evaluate_tier1", lambda: None)
    return scanner, broker


def test_risk_cycle_reconciles_once_and_counts_multiple_divergences_once(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trading_bot.broker.reconcile import Divergence, ReconciliationReport

    calls = 0

    def reconcile_once(_broker: object) -> ReconciliationReport:
        nonlocal calls
        calls += 1
        return ReconciliationReport(ok=True, divergences=[
            Divergence("internal_only", "AAPL", "missing at broker"),
            Divergence("broker_only", "META", "missing internally"),
        ])

    scanner, _broker = _stub_scanner_risk_cycle(monkeypatch, reconcile_once)

    scanner._run_risk_cycle()

    assert calls == 1
    assert ror.reconcile_divergence_streak() == 1


def test_risk_cycle_clean_reconcile_resets_and_ignores_open_order_count(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trading_bot.broker.reconcile import ReconciliationReport

    ror.record_reconcile_result(False)
    ror.record_reconcile_result(False)
    scanner, _broker = _stub_scanner_risk_cycle(
        monkeypatch,
        lambda _broker: ReconciliationReport(
            ok=True, divergences=[], broker_open_orders=9,
        ),
    )

    scanner._run_risk_cycle()

    assert ror.reconcile_divergence_streak() == 0


def test_risk_cycle_reconciles_alpaca_short_shares_fallback_without_tier2(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trading_bot import scanner
    from trading_bot.broker import alpaca

    db.insert_long_term_position(LongTermPosition(
        ticker="TSLA", asset_class="stock", entry_price=240.0,
        entry_date=_TS, qty=3.0, source="swing_fallback", direction="short",
    ))
    ror.record_reconcile_result(False)
    ror.record_reconcile_result(False)

    client = alpaca.AlpacaBroker()
    monkeypatch.setattr(
        client, "get_account",
        lambda: AccountInfo(ok=True, equity=10_000.0),
    )

    def mocked_request(method: str, path: str, **_kwargs: Any) -> alpaca._Response:
        assert method == "GET"
        body: list[dict[str, str]] = (
            [{"symbol": "TSLA", "qty": "-3", "side": "short"}]
            if path == "/v2/positions" else []
        )
        return alpaca._Response(ok=True, status_code=200, body=body, error="")

    monkeypatch.setattr(client, "_request", mocked_request)
    monkeypatch.setattr("trading_bot.broker.AlpacaBroker", lambda: client)
    monkeypatch.setattr(ror, "record_equity_snapshot", lambda _equity: 1)
    monkeypatch.setattr(ror, "evaluate_tier1", lambda: None)
    triggered: list[str] = []
    monkeypatch.setattr(
        ror, "emergency_shutdown",
        lambda _broker, *, trigger: triggered.append(trigger),
    )

    scanner._run_risk_cycle()

    assert client.get_positions().positions[0].qty == 3.0
    assert ror.reconcile_divergence_streak() == 0
    assert triggered == []


def test_risk_cycle_unavailable_reconcile_preserves_streak(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_bot.broker.reconcile import ReconciliationReport

    ror.record_reconcile_result(False)
    ror.record_reconcile_result(False)
    scanner, _broker = _stub_scanner_risk_cycle(
        monkeypatch,
        lambda _broker: ReconciliationReport(ok=False, note="broker down"),
    )

    scanner._run_risk_cycle()

    assert ror.reconcile_divergence_streak() == 2
    assert "risk reconcile unavailable" in capsys.readouterr().err


def test_risk_cycle_reconcile_exception_preserves_streak(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ror.record_reconcile_result(False)

    def unavailable(_broker: object) -> None:
        raise RuntimeError("malformed broker response")

    scanner, _broker = _stub_scanner_risk_cycle(monkeypatch, unavailable)

    scanner._run_risk_cycle()

    assert ror.reconcile_divergence_streak() == 1
    assert "risk reconcile error" in capsys.readouterr().err


def test_risk_cycle_malformed_report_preserves_streak(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ror.record_reconcile_result(False)
    scanner, _broker = _stub_scanner_risk_cycle(
        monkeypatch,
        lambda _broker: object(),
    )

    scanner._run_risk_cycle()

    assert ror.reconcile_divergence_streak() == 1
    assert "malformed report" in capsys.readouterr().err


def test_third_cycle_divergence_triggers_tier2_in_same_cycle(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trading_bot.broker.reconcile import Divergence, ReconciliationReport

    ror.record_reconcile_result(False)
    ror.record_reconcile_result(False)
    scanner, broker = _stub_scanner_risk_cycle(
        monkeypatch,
        lambda _broker: ReconciliationReport(ok=True, divergences=[
            Divergence("qty_mismatch", "AAPL", "quantity differs"),
        ]),
    )
    triggered: list[tuple[object, str]] = []
    monkeypatch.setattr(
        ror, "emergency_shutdown",
        lambda b, *, trigger, **_kw: triggered.append((b, trigger)),
    )

    scanner._run_risk_cycle()

    assert ror.reconcile_divergence_streak() == 3
    assert len(triggered) == 1 and triggered[0][0] is broker
    assert "3 consecutive checks" in triggered[0][1]


def test_risk_cycle_reconciliation_ignores_signal_and_shadow_trades(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trading_bot import scanner
    from trading_bot.broker import reconcile as real_reconcile

    for index, track_mode in enumerate(("active", "shadow")):
        timestamp = _TS + timedelta(minutes=index)
        signal_id = db.insert_signal(Signal(
            timestamp=timestamp, ticker=f"SIG{index}", asset_class="stock",
            signal_type="ema21_pullback", direction="call", entry_price=100.0,
        ))
        db.insert_trade(Trade(
            signal_id=signal_id, opened_at=timestamp, outcome="open",
            track_mode=track_mode,
        ))
    ror.record_reconcile_result(False)
    broker = FakeBroker()
    calls = 0

    def reconcile_once(b: object):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        return real_reconcile(b)  # type: ignore[arg-type]

    monkeypatch.setattr("trading_bot.broker.AlpacaBroker", lambda: broker)
    monkeypatch.setattr("trading_bot.broker.reconcile", reconcile_once)
    monkeypatch.setattr(ror, "record_equity_snapshot", lambda _equity: 1)
    monkeypatch.setattr(ror, "evaluate_tier1", lambda: None)

    scanner._run_risk_cycle()

    assert calls == 1
    assert ror.reconcile_divergence_streak() == 0


# ───────────────────────── top-level guard (unhandled exception) ─────────────


def test_run_guarded_success_calls_no_handler(tmp_db: Path) -> None:
    triggered: list[str] = []
    ok = ror.run_guarded(lambda: None, on_catastrophic=triggered.append)
    assert ok is True and triggered == []


def test_run_guarded_routes_unhandled_exception_without_crashing(
    tmp_db: Path,
) -> None:
    triggered: list[str] = []

    def boom() -> None:
        raise RuntimeError("core loop blew up")

    ok = ror.run_guarded(boom, on_catastrophic=triggered.append)
    assert ok is False                                 # process did NOT crash
    assert len(triggered) == 1
    assert "core loop blew up" in triggered[0]


def test_run_guarded_default_handler_revokes_and_records(tmp_db: Path) -> None:
    def boom() -> None:
        raise ValueError("kaboom")

    ok = ror.run_guarded(boom)                         # default handler
    assert ok is False
    assert ror.is_entry_authorized() is False          # fail-safe hard revoke
    assert ror.get_state() == ror.STATE_HOLDING


def test_run_guarded_survives_a_failing_handler(tmp_db: Path) -> None:
    def boom() -> None:
        raise RuntimeError("primary")

    def bad_handler(_reason: str) -> None:
        raise RuntimeError("handler also broken")

    ok = ror.run_guarded(boom, on_catastrophic=bad_handler)
    assert ok is False                                 # still no crash


# ───────────────────────── Tier 2: emergency shutdown ───────────────────────


class _ClosingFakeBroker(FakeBroker):
    """A FakeBroker whose accepted SELLs actually clear the broker-side position,
    so reconciliation can confirm a flat book."""

    def submit_order(self, symbol: str, qty: float, side: str, **kw: object):  # type: ignore[no-untyped-def]
        order = super().submit_order(symbol, qty, side, **kw)  # type: ignore[arg-type]
        if order.ok and side == "sell":
            self._positions.pop(symbol, None)
        return order


def _seed_open_option() -> int:
    position_id = db.insert_option_position(OptionPosition(
        symbol="AAPL260116C00150000", underlying="AAPL", option_type="call",
        strike=150.0, expiry="2026-01-16", contracts=2.0, opened_at=_TS,
        premium_entry=5.0, outcome="open",
    ))
    db.insert_pending_order(PendingOrder(
        client_order_id=f"risk-entry-option-{position_id}",
        broker_order_id=f"risk-broker-entry-option-{position_id}",
        ticker="AAPL",
        broker_symbol="AAPL260116C00150000",
        asset_class="stock",
        vehicle="option_full",
        target_position_kind="option",
        side="buy",
        requested_qty=2.0,
        requested_limit_price=5.0,
        submitted_at=_TS,
        intent_payload_json=json.dumps({
            "intent_kind": "option",
            "option_type": "call",
            "strike": 150.0,
            "expiry": "2026-01-16",
            "multiplier": 100,
            "delta_entry": None,
            "theta": None,
            "vega": None,
            "gamma": None,
            "tp": None,
            "sl": None,
            "deadline": None,
        }, sort_keys=True),
        lifecycle_status=STATUS_FILLED,
        broker_status=STATUS_FILLED,
        filled_qty=2.0,
        filled_avg_price=5.0,
        last_fill_at=_TS,
        last_fill_time_source="broker",
        last_refreshed_at=_TS,
        terminal_reason=STATUS_FILLED,
        terminal_at=_TS,
        position_kind="option",
        position_id=position_id,
    ))
    return position_id


def _seed_open_long_term() -> int:
    position_id = db.insert_long_term_position(LongTermPosition(
        ticker="META", asset_class="stock", entry_price=480.0, entry_date=_TS,
        qty=5.0, status="open",
    ))
    db.insert_pending_order(PendingOrder(
        client_order_id=f"risk-entry-long-{position_id}",
        broker_order_id=f"risk-broker-entry-long-{position_id}",
        ticker="META",
        broker_symbol="META",
        asset_class="stock",
        vehicle="shares",
        target_position_kind="long_term",
        side="buy",
        requested_qty=5.0,
        requested_limit_price=480.0,
        submitted_at=_TS,
        intent_payload_json=json.dumps({
            "intent_kind": "long_term",
            "source": "long_term",
            "direction": "long",
        }, sort_keys=True),
        lifecycle_status=STATUS_FILLED,
        broker_status=STATUS_FILLED,
        filled_qty=5.0,
        filled_avg_price=480.0,
        last_fill_at=_TS,
        last_fill_time_source="broker",
        last_refreshed_at=_TS,
        terminal_reason=STATUS_FILLED,
        terminal_at=_TS,
        position_kind="long_term",
        position_id=position_id,
    ))
    return position_id


def test_emergency_shutdown_calls_all_three_closers_then_halts(
    tmp_db: Path,
) -> None:
    _seed_open_option()
    _seed_open_long_term()
    b = _ClosingFakeBroker(auto_fill=True)
    b.set_position("NVDA", 3.0, avg_entry_price=120.0)   # broker-side equity
    rec = _Recorder()

    result = ror.emergency_shutdown(
        b, trigger="test trigger", notifier=rec, now=_TS,
        option_price_fetch=lambda _s: 6.0,
        long_term_price_fetch=lambda _t: 470.0,
    )

    assert result.status == "halted"
    # ALL THREE closers were invoked — one closed entry per book.
    kinds = {c.split(" ")[0] for c in result.closed}
    assert kinds == {"option", "long-term", "equity"}
    # Internal books are closed, broker book flat, state persisted halted.
    assert db.get_open_option_positions() == []
    assert db.get_open_long_term_positions() == []
    assert ror.get_state() == ror.STATE_HALTED
    assert ror.is_entry_authorized() is False
    # The emergency Pushover fired with the required title.
    assert any(
        "BOT HAS ENCOUNTERED A SEVERE ERROR SHUT DOWN INITIATED" in t
        for t, _m in rec.calls
    )


def test_emergency_shutdown_market_closed_defers_and_retries(
    tmp_db: Path,
) -> None:
    _seed_open_option()
    rec = _Recorder()
    closed_market = FakeBroker(reject_reason="market closed")

    first = ror.emergency_shutdown(
        closed_market, trigger="test", notifier=rec, now=_TS,
        option_price_fetch=lambda _s: 6.0,
    )
    # Deferred, NOT skipped: holding state, position still open, NOT halted.
    assert first.status == "holding"
    assert first.pending and "market closed" in first.pending[0]
    assert len(db.get_open_option_positions()) == 1     # row NOT faked closed
    assert ror.get_state() == ror.STATE_HOLDING

    # Next cycle, market open: the retry completes and halts.
    second = ror.emergency_shutdown(
        _ClosingFakeBroker(auto_fill=True), trigger="test", notifier=rec, now=_TS,
        option_price_fetch=lambda _s: 6.0,
    )
    assert second.status == "halted"
    assert db.get_open_option_positions() == []
    assert ror.get_state() == ror.STATE_HALTED


def test_emergency_shutdown_reconcile_failure_never_claims_closure(
    tmp_db: Path, monkeypatch,
) -> None:
    from trading_bot.broker.reconcile import ReconciliationReport

    _seed_open_option()
    rec = _Recorder()
    monkeypatch.setattr(
        ror, "reconcile",
        lambda _b: ReconciliationReport(ok=False, note="broker down"),
    )
    result = ror.emergency_shutdown(
        _ClosingFakeBroker(auto_fill=True), trigger="test", notifier=rec, now=_TS,
        option_price_fetch=lambda _s: 6.0,
    )
    assert result.status == "unconfirmed"               # no false "closed" claim
    assert any(
        "UNABLE TO CONFIRM CLOSURE" in t and "MANUAL INTERVENTION" in t
        for t, _m in rec.calls
    )
    # No "severe error shutdown" success message was sent.
    assert not any("SHUT DOWN INITIATED" in t for t, _m in rec.calls)


def test_emergency_shutdown_holds_while_broker_still_shows_positions(
    tmp_db: Path,
) -> None:
    # A plain FakeBroker never clears its seeded position on sell, so the
    # reconcile-confirmation step keeps the bot ALIVE in holding — a full halt
    # only happens after confirmed closure.
    b = FakeBroker()
    b.set_position("NVDA", 3.0, avg_entry_price=120.0)
    result = ror.emergency_shutdown(b, trigger="test", notifier=_Recorder(), now=_TS)
    assert result.status == "holding"
    assert result.closed == []
    assert ror.get_state() == ror.STATE_HOLDING         # alive, not halted


def test_generic_acceptance_is_not_reported_closed_while_order_is_open(
    tmp_db: Path,
) -> None:
    broker = _ClosingFakeBroker(auto_fill=False)
    broker.set_position("NVDA", 3.0, avg_entry_price=120.0)

    result = ror.emergency_shutdown(
        broker, trigger="test", notifier=_Recorder(), now=_TS
    )

    assert result.status == "holding"
    assert result.closed == []
    assert "awaiting actual close fills" in result.pending[0]


def test_generic_short_close_uses_positive_market_value_magnitude(
    tmp_db: Path,
) -> None:
    broker = _ClosingFakeBroker(auto_fill=False)
    broker._positions["TSLA"] = Position(
        symbol="TSLA", qty=3.0, side="short", market_value=-720.0,
    )

    ror.emergency_shutdown(
        broker, trigger="test", notifier=_Recorder(), now=_TS,
    )

    (order,) = db.get_generic_emergency_orders_for_symbol("TSLA")
    assert order.side == "buy"
    assert order.requested_qty == 3.0
    assert order.requested_limit_price == 240.0


def test_emergency_restart_does_not_duplicate_nonterminal_typed_exit(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    position_id = _seed_open_option()
    first = ror.emergency_shutdown(
        FakeBroker(auto_fill=False),
        trigger="test",
        notifier=_Recorder(),
        now=_TS,
        option_price_fetch=lambda _s: 6.0,
    )
    assert first.status == "holding"

    restart_calls: list[object] = []
    restarted = FakeBroker(auto_fill=True)
    monkeypatch.setattr(
        restarted,
        "submit_order",
        lambda *args, **kwargs: restart_calls.append((args, kwargs)),
    )
    second = ror.emergency_shutdown(
        restarted,
        trigger="test",
        notifier=_Recorder(),
        now=_TS + timedelta(minutes=1),
        option_price_fetch=lambda _s: 6.0,
    )

    assert second.status == "holding"
    assert restart_calls == []
    assert len(db.get_pending_exit_orders_for_position("option", position_id)) == 1
    open_position = db.get_option_position(position_id)
    assert open_position is not None and open_position.outcome == "open"


# ───────────────────────── re-authorization (token-gated) ───────────────────


def test_reauthorize_requires_exact_token(tmp_db: Path) -> None:
    ror.revoke(config.ENTRY_CAPABILITY, "tier1: test")
    ok, message = ror.reauthorize("wrong-token")
    assert ok is False and "invalid confirmation token" in message
    assert ror.is_entry_authorized() is False           # nothing changed


def test_reauthorize_clears_tier1_pause(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_resolved(["loss"] * 7)
    monkeypatch.setattr(ror, "consecutive_losses", lambda: 7)
    ror.evaluate_tier1(notifier=_Recorder())
    assert ror.get_state() == ror.STATE_PAUSED
    ok, _ = ror.reauthorize(config.ROR_REAUTHORIZE_TOKEN)
    assert ok is True
    assert ror.get_state() == ror.STATE_NORMAL
    assert ror.is_entry_authorized() is True


def test_reauthorize_clears_tier2_halt_and_resets_detectors(tmp_db: Path) -> None:
    ror.record_broker_result(False)
    ror.emergency_shutdown(FakeBroker(), trigger="test", notifier=_Recorder(), now=_TS)
    assert ror.get_state() == ror.STATE_HALTED
    ok, _ = ror.reauthorize(config.ROR_REAUTHORIZE_TOKEN)
    assert ok is True
    assert ror.get_state() == ror.STATE_NORMAL
    assert ror.is_entry_authorized() is True
    assert ror.broker_error_streak() == 0               # detectors reset


# ───────────────────────── manual kill switch (identical path) ──────────────


def test_kill_switch_refuses_wrong_token(tmp_db: Path) -> None:
    _seed_open_option()
    result = ror.kill_switch("wrong", FakeBroker(), notifier=_Recorder(), now=_TS)
    assert result is None                               # nothing happened
    assert len(db.get_open_option_positions()) == 1
    assert ror.get_state() == ror.STATE_NORMAL


def test_kill_switch_runs_the_identical_shutdown_path(
    tmp_db: Path, monkeypatch,
) -> None:
    # The kill switch delegates to emergency_shutdown ITSELF — same function
    # object, no separate/weaker code path.
    calls: list[str] = []
    original = ror.emergency_shutdown

    def spy(broker, **kw):  # type: ignore[no-untyped-def]
        calls.append(kw["trigger"])
        return original(broker, **kw)

    monkeypatch.setattr(ror, "emergency_shutdown", spy)
    result = ror.kill_switch(
        config.ROR_KILLSWITCH_TOKEN, FakeBroker(), notifier=_Recorder(), now=_TS,
    )
    assert calls == ["manual kill switch"]              # the one orchestrator ran
    assert result is not None and result.status == "halted"


def test_kill_switch_and_auto_trigger_produce_identical_behavior(
    tmp_db: Path,
) -> None:
    # Same setup → the manual kill switch and an auto-detected trigger yield the
    # same status, the same closed book, and the same persisted end state.
    _seed_open_option()
    rec_auto = _Recorder()
    broker = _ClosingFakeBroker(auto_fill=True)
    auto = ror.emergency_shutdown(
        broker, trigger="3 consecutive broker errors", notifier=rec_auto,
        now=_TS, option_price_fetch=lambda _s: 6.0,
    )
    auto_state = ror.get_state()

    # Reset the world and repeat via the kill switch.
    ok, _ = ror.reauthorize(config.ROR_REAUTHORIZE_TOKEN)
    assert ok
    _seed_open_option()
    rec_manual = _Recorder()
    manual = ror.kill_switch(
        config.ROR_KILLSWITCH_TOKEN, broker, notifier=rec_manual, now=_TS,
        option_price_fetch=lambda _s: 6.0,
    )

    assert manual is not None
    assert manual.status == auto.status == "halted"
    assert manual.closed == auto.closed                 # same closures
    assert ror.get_state() == auto_state == ror.STATE_HALTED
    assert (
        [t for t, _m in rec_manual.calls] == [t for t, _m in rec_auto.calls]
    )                                                   # same notifications


# ───────────────────────── Pushover notifier (fail-soft) ────────────────────


def test_pushover_notify_success(monkeypatch) -> None:
    monkeypatch.setattr(
        "trading_bot.risk_of_ruin.secrets.get_secret", lambda _n: "key",
    )

    class _Resp:
        status_code = 200

    monkeypatch.setattr(
        "trading_bot.risk_of_ruin.requests.post", lambda *a, **k: _Resp(),
    )
    assert ror._pushover_notify("t", "m") is True


def test_pushover_notify_failsoft(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "trading_bot.risk_of_ruin.secrets.get_secret", lambda _n: None,
    )
    assert ror._pushover_notify("t", "m") is False        # creds unset

    monkeypatch.setattr(
        "trading_bot.risk_of_ruin.secrets.get_secret", lambda _n: "key",
    )

    class _Resp:
        status_code = 500

    monkeypatch.setattr(
        "trading_bot.risk_of_ruin.requests.post", lambda *a, **k: _Resp(),
    )
    assert ror._pushover_notify("t", "m") is False        # non-2xx

    def boom(*_a: object, **_k: object) -> object:
        raise RuntimeError("net down")

    monkeypatch.setattr("trading_bot.risk_of_ruin.requests.post", boom)
    assert ror._pushover_notify("t", "m") is False        # exception swallowed
    assert "error" in capsys.readouterr().err


# ───────────────────────── risk CLI (status / reauthorize / killswitch) ─────


def test_cli_risk_ror_status(tmp_db: Path, capsys) -> None:
    from trading_bot import __main__ as m
    db.insert_equity_snapshot(10_000.0, _TS)
    db.insert_equity_snapshot(9_000.0, _TS + timedelta(hours=1))
    m.cmd_risk_ror_status()
    out = capsys.readouterr().out
    assert "RISK-OF-RUIN STATUS" in out
    assert "Entry authorized:      yes" in out
    assert "10.0%" in out                               # current drawdown


def test_cli_risk_ror_status_shows_revoked_and_trigger(tmp_db: Path, capsys) -> None:
    from trading_bot import __main__ as m
    ror.revoke(config.ENTRY_CAPABILITY, "tier1: test")
    for _ in range(3):
        ror.record_broker_result(False)
    m.cmd_risk_ror_status()
    out = capsys.readouterr().out
    assert "Entry authorized:      NO" in out
    assert "tier1: test" in out
    assert "CATASTROPHIC TRIGGER" in out


def test_cli_risk_reauthorize(tmp_db: Path, capsys) -> None:
    import pytest

    from trading_bot import __main__ as m
    ror.revoke(config.ENTRY_CAPABILITY, "tier1: test")
    with pytest.raises(SystemExit):
        m.cmd_risk_reauthorize("wrong")                 # wrong token → exit 1
    assert ror.is_entry_authorized() is False
    m.cmd_risk_reauthorize(config.ROR_REAUTHORIZE_TOKEN)
    assert ror.is_entry_authorized() is True
    assert "re-authorized" in capsys.readouterr().out


def test_cli_risk_killswitch(tmp_db: Path, monkeypatch, capsys) -> None:
    import pytest

    from trading_bot import __main__ as m
    monkeypatch.setattr(m.broker, "AlpacaBroker", lambda: FakeBroker())
    monkeypatch.setattr(ror, "_pushover_notify", lambda _t, _m: True)
    with pytest.raises(SystemExit):
        m.cmd_risk_killswitch("wrong")                  # refused, exit 1
    assert ror.get_state() == ror.STATE_NORMAL
    m.cmd_risk_killswitch(config.ROR_KILLSWITCH_TOKEN)
    out = capsys.readouterr().out
    assert "KILL SWITCH: halted" in out
    assert ror.get_state() == ror.STATE_HALTED
