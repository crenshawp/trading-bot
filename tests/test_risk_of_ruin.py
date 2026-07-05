"""Tests for the risk-of-ruin safety layer (Phase 15).

Mocked data only, no live network. Tier 1 (pause via authorization revoke),
catastrophic detection, and the Tier 2 emergency shutdown orchestrator are the
core; equity snapshots drive the drawdown check so unrealized losses count.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from trading_bot import allocation, config, db
from trading_bot import risk_of_ruin as ror
from trading_bot.broker.base import AccountInfo
from trading_bot.models import LongTermPosition, OptionPosition, Signal, Trade

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


def _seed_resolved(outcomes: list[str], *, track_mode: str = "active") -> None:
    for i, outcome in enumerate(outcomes):
        ts = _TS + timedelta(minutes=i)
        sid = db.insert_signal(Signal(
            timestamp=ts, ticker=f"T{i}", asset_class="stock",
            signal_type="ema21_pullback", direction="call", entry_price=100.0,
        ))
        db.insert_trade(Trade(
            signal_id=sid, opened_at=ts, closed_at=ts + timedelta(days=1),
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


def test_consecutive_losses_counts_trailing_run(tmp_db: Path) -> None:
    _seed_resolved(["loss", "win", "loss", "loss"])   # chronological
    assert ror.consecutive_losses() == 2              # trailing run only


def test_tier1_does_not_trip_at_six_losses(tmp_db: Path) -> None:
    _seed_resolved(["win"] + ["loss"] * 6)
    rec = _Recorder()
    result = ror.evaluate_tier1(notifier=rec)
    assert result.tripped is False
    assert ror.is_entry_authorized() is True
    assert rec.calls == []


def test_tier1_trips_at_exactly_seven_losses(tmp_db: Path) -> None:
    _seed_resolved(["win"] + ["loss"] * 7)
    rec = _Recorder()
    result = ror.evaluate_tier1(notifier=rec)
    assert result.tripped is True and result.trigger == "consecutive_losses"
    assert ror.is_entry_authorized() is False          # entry revoked
    assert ror.get_state() == ror.STATE_PAUSED
    assert len(rec.calls) == 1
    assert "PAUSED" in rec.calls[0][0]


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


def test_tier1_leaves_existing_positions_untouched(tmp_db: Path) -> None:
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
    result = ror.evaluate_tier1(notifier=_Recorder())
    assert result.tripped is True
    assert len(db.get_open_option_positions()) == 1     # untouched
    assert len(db.get_open_long_term_positions()) == 1  # untouched


def test_tier1_does_not_retrip_when_already_paused(tmp_db: Path) -> None:
    _seed_resolved(["loss"] * 7)
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
