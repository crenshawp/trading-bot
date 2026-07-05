"""Tests for the risk-of-ruin safety layer (Phase 15).

Mocked data only, no live network. Tier 1 (pause via authorization revoke),
catastrophic detection, and the Tier 2 emergency shutdown orchestrator are the
core; equity snapshots drive the drawdown check so unrealized losses count.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from trading_bot import config, db
from trading_bot.models import Signal, Trade

_TS = datetime(2026, 1, 1, 9, 0, tzinfo=UTC)


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
