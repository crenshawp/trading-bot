"""Tests for trading_bot.readiness — the unified readiness gate (Phase 10).

Synthetic data only, Pushover mocked. The central correctness requirements:
thresholds equal the existing constants (behavior preserved), readiness flips
exactly at the threshold, and the crossing notification fires exactly once.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from trading_bot import config, db, readiness
from trading_bot.models import Signal, Trade

_TS = datetime(2026, 1, 1, 9, 0)


def _seed_resolved(
    n: int, *, outcome: str = "win", pnl: float | None = 2.0,
    track_mode: str = "active", start: int = 0,
) -> int:
    """Insert n resolved trades (each its own signal). Returns next free idx."""
    idx = start
    for _ in range(n):
        ts = _TS + timedelta(minutes=idx)
        sid = db.insert_signal(Signal(
            timestamp=ts, ticker=f"T{idx}", asset_class="stock",
            signal_type="ema21_pullback", direction="call", entry_price=100.0,
        ))
        db.insert_trade(Trade(
            signal_id=sid, opened_at=ts, closed_at=ts + timedelta(days=1),
            outcome=outcome, pnl_pct=pnl, track_mode=track_mode,
        ))
        idx += 1
    return idx


# ───────────────────────── registry / centralization ────────────────────────────


def test_registry_thresholds_equal_centralized_constants() -> None:
    caps = {c.name: c for c in readiness.REGISTRY}
    assert caps["watchlist_rotation"].threshold == config.SM_MIN_CLOSED_SIGNALS
    assert caps["pair_gating"].threshold == config.SP_MIN_CLOSED_SIGNALS
    assert caps["shadow_promotion"].threshold == config.MIN_SHADOW_SIGNALS
    assert caps["self_optimization"].threshold == config.SO_MIN_SAMPLE


def test_min_shadow_signals_back_compat_alias() -> None:
    from trading_bot import shadow_discovery
    assert shadow_discovery.MIN_SHADOW_SIGNALS == config.MIN_SHADOW_SIGNALS


def test_capability_lookup_unknown_raises() -> None:
    assert readiness.capability("watchlist_rotation").kind == "deterministic"
    with pytest.raises(KeyError, match="unknown capability"):
        readiness.capability("does_not_exist")


def test_scope_count_rejects_unknown_scope() -> None:
    with pytest.raises(ValueError, match="unknown readiness scope"):
        readiness._scope_count("bogus")


# ───────────────────────── scope counts ─────────────────────────────────────────


def test_resolved_count_by_scope(tmp_db: Path) -> None:
    idx = _seed_resolved(3, outcome="win", track_mode="active")
    idx = _seed_resolved(2, outcome="loss", pnl=-1.0, track_mode="active", start=idx)
    idx = _seed_resolved(1, outcome="expired", pnl=None, track_mode="active", start=idx)
    idx = _seed_resolved(4, outcome="win", track_mode="shadow", start=idx)
    # 'all' scope (watchlist_rotation): 5 active + 4 shadow = 9 (expired excluded)
    assert readiness.resolved_count("watchlist_rotation") == 9
    # 'active' scope (self_optimization): 5
    assert readiness.resolved_count("self_optimization") == 5
    # 'shadow' scope (shadow_promotion): 4
    assert readiness.resolved_count("shadow_promotion") == 4


# ───────────────────────── readiness crossing ───────────────────────────────────


def test_is_ready_flips_exactly_at_threshold(tmp_db: Path) -> None:
    threshold = config.SM_MIN_CLOSED_SIGNALS        # watchlist_rotation, scope 'all'
    _seed_resolved(threshold - 1)
    assert readiness.is_ready("watchlist_rotation") is False    # n-1: not ready
    _seed_resolved(1, start=threshold)
    assert readiness.is_ready("watchlist_rotation") is True     # n: ready


def test_is_ready_scopes_are_independent(tmp_db: Path) -> None:
    # 10 active resolved makes the 'all'-scope cap ready but not the shadow one.
    _seed_resolved(config.SM_MIN_CLOSED_SIGNALS, track_mode="active")
    assert readiness.is_ready("watchlist_rotation") is True
    assert readiness.is_ready("shadow_promotion") is False      # no shadow trades
