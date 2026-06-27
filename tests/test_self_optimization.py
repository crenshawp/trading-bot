"""Tests for trading_bot.self_optimization — Phase 9 degradation + feature eval.

Synthetic data only, no network. The discipline under test: every finding
carries n, the sample floor gates actionability, and the win_rate/expectancy
math is identical to the live gating subsystems.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from trading_bot import db
from trading_bot import self_optimization as so
from trading_bot.models import Signal, Trade
from trading_bot.signal_pairs import _windowed_stats


def _seed_resolved(
    tmp_db: Path,
    *,
    ticker: str = "GOOGL",
    signal_type: str = "ema21_pullback",
    outcome: str = "win",
    pnl: float | None = 2.0,
    closed_at: datetime | None = None,
    track_mode: str = "active",
    **context: object,
) -> int:
    """Insert one resolved trade (and its signal) with optional stored context."""
    sid = db.insert_signal(Signal(
        timestamp=datetime(2026, 1, 1, 9, 31), ticker=ticker, asset_class="stock",
        signal_type=signal_type, direction="call", entry_price=100.0,
    ))
    closed = closed_at if closed_at is not None else datetime(2026, 6, 1, 16, 0)
    return db.insert_trade(Trade(
        signal_id=sid, opened_at=datetime(2026, 5, 30), closed_at=closed,
        outcome=outcome, pnl_pct=pnl, track_mode=track_mode, **context,
    ))


# ───────────────────────── canonical stat reuse ─────────────────────────────────


def test_stat_for_matches_signal_pairs_definition() -> None:
    rows = [
        {"outcome": "win", "pnl_pct": 3.0},
        {"outcome": "win", "pnl_pct": 1.0},
        {"outcome": "loss", "pnl_pct": -2.0},
    ]
    stat = so.stat_for(rows)
    # Identical to the live gating helper on the same (outcome, pnl) pairs.
    pairs = [(r["outcome"], r["pnl_pct"]) for r in rows]
    assert (stat.n, stat.win_rate, stat.expectancy) == _windowed_stats(pairs)
    # And the hand-computed values: 2 wins / 3 = 66.67%, mean pnl = 0.6667.
    assert stat.n == 3
    assert stat.win_rate == pytest.approx(66.6667, abs=1e-3)
    assert stat.expectancy == pytest.approx(0.6667, abs=1e-3)


def test_stat_for_empty_is_zero_n_and_none() -> None:
    stat = so.stat_for([])
    assert stat == so.Stat(0, None, None)


def test_stat_for_excludes_nothing_extra_expired_never_reaches_here() -> None:
    # Expired rows would not be returned by the query, but if present the win_rate
    # denominator is wins+losses only (expired ignored) — matching the gating math.
    rows = [
        {"outcome": "win", "pnl_pct": 2.0},
        {"outcome": "loss", "pnl_pct": -1.0},
        {"outcome": "expired", "pnl_pct": 0.0},
    ]
    stat = so.stat_for(rows)
    assert stat.n == 2                       # wins + losses only
    assert stat.win_rate == pytest.approx(50.0)


# ───────────────────────── resolved-context query ───────────────────────────────


def test_resolved_context_rows_only_active_win_loss(tmp_db: Path) -> None:
    _seed_resolved(tmp_db, ticker="GOOGL", outcome="win", pnl=2.0)
    _seed_resolved(tmp_db, ticker="META", outcome="loss", pnl=-1.0)
    _seed_resolved(tmp_db, ticker="AMZN", outcome="expired", pnl=None)   # excluded
    _seed_resolved(tmp_db, ticker="TSLA", outcome="win", pnl=1.0,
                   track_mode="shadow")                                  # excluded
    # an open trade (no closed_at) is excluded too
    sid = db.insert_signal(Signal(
        timestamp=datetime(2026, 1, 1), ticker="NVDA", asset_class="stock",
        signal_type="ema21_pullback", direction="call", entry_price=100.0,
    ))
    db.insert_trade(Trade(signal_id=sid, opened_at=datetime(2026, 5, 30),
                          outcome="open", track_mode="active"))

    rows = so.resolved_context_rows()
    tickers = {r["ticker"] for r in rows}
    assert tickers == {"GOOGL", "META"}     # active win/loss only


def test_resolved_context_rows_window_filters_by_closed_at(tmp_db: Path) -> None:
    _seed_resolved(tmp_db, ticker="OLD", closed_at=datetime(2026, 1, 15, 16, 0))
    _seed_resolved(tmp_db, ticker="NEW", closed_at=datetime(2026, 6, 15, 16, 0))
    rows = so.resolved_context_rows(since=datetime(2026, 6, 1))
    assert {r["ticker"] for r in rows} == {"NEW"}


# ───────────────────────── degradation detection ────────────────────────────────


def _rows(
    n: int, pnl: float | None, *, signal_type: str = "ema21_pullback",
    ticker: str = "GOOGL", outcome: str = "win",
) -> list[dict[str, object]]:
    """n synthetic resolved rows with a fixed pnl (expectancy == pnl)."""
    return [
        {"signal_type": signal_type, "ticker": ticker,
         "outcome": outcome, "pnl_pct": pnl}
        for _ in range(n)
    ]


def _overall(findings: list[so.DegradationFinding]) -> so.DegradationFinding:
    return next(f for f in findings if f.scope == "overall")


def test_degradation_fires_when_drop_and_sample_clear_floors() -> None:
    findings = so.compute_degradation(
        _rows(30, 2.0), _rows(30, 1.0), min_sample=30, meaningful_delta=0.15,
    )
    overall = _overall(findings)
    assert overall.verdict == "degraded"
    assert overall.delta == pytest.approx(1.0)       # 2.0 -> 1.0
    assert overall.recent_n == 30 and overall.baseline_n == 30


def test_degradation_does_not_fire_on_real_drop_with_small_sample() -> None:
    # A 1.0 expectancy drop, but only 20 recent trades — variance, not signal.
    findings = so.compute_degradation(
        _rows(30, 2.0), _rows(20, 1.0), min_sample=30, meaningful_delta=0.15,
    )
    overall = _overall(findings)
    assert overall.verdict == "insufficient_sample"
    assert "not actionable" in overall.note


def test_degradation_does_not_fire_when_drop_below_delta() -> None:
    findings = so.compute_degradation(
        _rows(30, 2.0), _rows(30, 1.9), min_sample=30, meaningful_delta=0.15,
    )
    overall = _overall(findings)
    assert overall.verdict == "stable"
    assert overall.delta == pytest.approx(0.1)


def test_degradation_boundary_drop_equal_to_delta_fires() -> None:
    findings = so.compute_degradation(
        _rows(30, 2.0), _rows(30, 1.85), min_sample=30, meaningful_delta=0.15,
    )
    assert _overall(findings).verdict == "degraded"   # delta == meaningful_delta


def test_degradation_improvement_is_stable_not_degraded() -> None:
    findings = so.compute_degradation(
        _rows(30, 1.0), _rows(30, 2.0), min_sample=30, meaningful_delta=0.15,
    )
    overall = _overall(findings)
    assert overall.verdict == "stable"
    assert overall.delta == pytest.approx(-1.0)       # expectancy improved


def test_degradation_insufficient_baseline_when_no_baseline() -> None:
    findings = so.compute_degradation(
        [], _rows(30, 1.0), min_sample=30, meaningful_delta=0.15,
    )
    overall = _overall(findings)
    assert overall.verdict == "insufficient_baseline"
    assert overall.baseline_n == 0


def test_degradation_insufficient_when_recent_has_no_measurable_expectancy() -> None:
    # 30 resolved trades clear the sample floor, but none carry a pnl -> no
    # measurable expectancy -> not actionable rather than a false "degraded".
    findings = so.compute_degradation(
        _rows(30, 2.0), _rows(30, None), min_sample=30, meaningful_delta=0.15,
    )
    overall = _overall(findings)
    assert overall.verdict == "insufficient_sample"
    assert overall.recent_expectancy is None
    assert "no measurable expectancy" in overall.note


def test_degradation_emits_per_setup_and_per_ticker_scopes() -> None:
    baseline = _rows(2, 2.0, signal_type="ema21_pullback", ticker="GOOGL")
    recent = _rows(2, 1.0, signal_type="trend_continuation", ticker="META")
    scopes = {f.scope for f in so.compute_degradation(baseline, recent)}
    assert "overall" in scopes
    assert {"setup:ema21_pullback", "setup:trend_continuation"} <= scopes
    assert {"ticker:GOOGL", "ticker:META"} <= scopes


def test_detect_degradation_splits_recent_and_baseline_windows(tmp_db: Path) -> None:
    # now=2026-07-01, degrade=30d -> recent>=2026-06-01; baseline 90d window.
    _seed_resolved(tmp_db, ticker="REC", closed_at=datetime(2026, 6, 15, 16, 0))
    _seed_resolved(tmp_db, ticker="BASE", closed_at=datetime(2026, 5, 1, 16, 0))
    _seed_resolved(tmp_db, ticker="OLD", closed_at=datetime(2026, 1, 1, 16, 0))
    findings = so.detect_degradation(
        now=datetime(2026, 7, 1), degrade_window_days=30, baseline_window_days=90,
    )
    overall = _overall(findings)
    assert overall.recent_n == 1       # only REC
    assert overall.baseline_n == 1     # only BASE (OLD is before the baseline cut)
