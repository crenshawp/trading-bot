"""Tests for trading_bot.self_optimization — Phase 9 degradation + feature eval.

Synthetic data only, no network. The discipline under test: every finding
carries n, the sample floor gates actionability, and the win_rate/expectancy
math is identical to the live gating subsystems.
"""

from __future__ import annotations

import json
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


# ───────────────────────── feature evaluation ───────────────────────────────────


def _frow(**kw: object) -> dict[str, object]:
    base: dict[str, object] = {
        "outcome": "win", "pnl_pct": 1.0, "signal_type": "s", "ticker": "T",
        "sentiment_label": None, "ind_vol_regime": None, "ind_rsi": None,
        "ind_adx": None, "ind_obv": None, "ind_concentration": None,
        "risk_portfolio_verdict": None,
    }
    base.update(kw)
    return base


def _feature(evals: list[so.FeatureEvaluation], name: str) -> so.FeatureEvaluation:
    return next(e for e in evals if e.feature == name)


def test_feature_evaluations_cover_all_seven_features() -> None:
    names = [e.feature for e in so.compute_feature_evaluations([])]
    assert names == [
        "sentiment", "vol_regime", "rsi", "adx", "obv",
        "concentration", "portfolio_verdict",
    ]


def test_sentiment_buckets_hand_computed() -> None:
    rows = (
        [_frow(sentiment_label="bearish", outcome="win", pnl_pct=2.0)]
        + [_frow(sentiment_label="bearish", outcome="loss", pnl_pct=-1.0)] * 2
        + [_frow(sentiment_label="bullish", outcome="win", pnl_pct=2.0)] * 2
        + [_frow(sentiment_label="bullish", outcome="loss", pnl_pct=-1.0)]
    )
    sent = _feature(so.compute_feature_evaluations(rows, min_sample=3), "sentiment")
    buckets = {b.label: b for b in sent.buckets}
    assert "neutral" not in buckets                      # no neutral rows
    assert buckets["bearish"].n == 3
    assert buckets["bearish"].win_rate == pytest.approx(33.333, abs=1e-2)
    assert buckets["bearish"].expectancy == pytest.approx(0.0)        # mean(2,-1,-1)
    assert buckets["bullish"].win_rate == pytest.approx(66.667, abs=1e-2)
    assert buckets["bullish"].expectancy == pytest.approx(1.0)        # mean(2,2,-1)
    assert sent.actionable is True                       # both buckets n>=3
    assert "bullish 67% (n=3) vs bearish 33% (n=3)" in sent.note


def test_rsi_bands_assign_correctly_incl_boundaries() -> None:
    rows = [
        _frow(ind_rsi=30.0), _frow(ind_rsi=40.0), _frow(ind_rsi=60.0),
        _frow(ind_rsi=61.0), _frow(ind_rsi=70.0),
    ]
    rsi = _feature(so.compute_feature_evaluations(rows, min_sample=1), "rsi")
    counts = {b.label: b.n for b in rsi.buckets}
    assert counts == {"low(<40)": 1, "mid(40-60)": 2, "high(>60)": 2}  # 40,60 -> mid


def test_adx_bands_and_obv_sign_buckets() -> None:
    rows = [
        _frow(ind_adx=19.0, ind_obv=-5.0),
        _frow(ind_adx=20.0, ind_obv=0.0),
        _frow(ind_adx=41.0, ind_obv=5.0),
    ]
    evals = so.compute_feature_evaluations(rows, min_sample=1)
    adx = {b.label: b.n for b in _feature(evals, "adx").buckets}
    obv = {b.label: b.n for b in _feature(evals, "obv").buckets}
    assert adx == {"weak(<20)": 1, "moderate(20-40)": 1, "strong(>40)": 1}
    assert obv == {"negative": 1, "zero": 1, "positive": 1}


def test_concentration_and_verdict_buckets_skip_unknown() -> None:
    rows = [
        _frow(ind_concentration="concentrated", risk_portfolio_verdict="ok"),
        _frow(ind_concentration="diversified",
              risk_portfolio_verdict="would-exceed-portfolio"),
        _frow(ind_concentration="unknown", risk_portfolio_verdict="unknown"),
    ]
    evals = so.compute_feature_evaluations(rows, min_sample=1)
    conc = {b.label for b in _feature(evals, "concentration").buckets}
    verd = {b.label for b in _feature(evals, "portfolio_verdict").buckets}
    assert conc == {"concentrated", "diversified"}          # 'unknown' skipped
    assert verd == {"ok", "would-exceed-portfolio"}


def test_feature_under_min_sample_is_not_actionable() -> None:
    rows = (
        [_frow(ind_vol_regime="low", outcome="win", pnl_pct=1.0)] * 2
        + [_frow(ind_vol_regime="high", outcome="loss", pnl_pct=-1.0)] * 2
    )
    vr = _feature(so.compute_feature_evaluations(rows, min_sample=30), "vol_regime")
    assert all(b.actionable is False for b in vr.buckets)   # each n=2 < 30
    assert vr.actionable is False
    assert "NOT actionable at n<30" in vr.note


def test_feature_with_no_data_notes_insufficient() -> None:
    sent = _feature(so.compute_feature_evaluations([]), "sentiment")
    assert sent.buckets == []
    assert sent.actionable is False
    assert "need >= 2 buckets" in sent.note


def test_evaluate_features_reads_from_db(tmp_db: Path) -> None:
    _seed_resolved(tmp_db, ticker="GOOGL", outcome="win", pnl=2.0,
                   sentiment_label="bullish")
    _seed_resolved(tmp_db, ticker="META", outcome="loss", pnl=-1.0,
                   sentiment_label="bearish")
    sent = _feature(so.evaluate_features(), "sentiment")
    assert {b.label for b in sent.buckets} == {"bullish", "bearish"}


# ───────────────────────── payload / persist / render ───────────────────────────


def test_build_persist_and_render_round_trip(tmp_db: Path) -> None:
    degradations = so.compute_degradation(
        _rows(30, 2.0), _rows(30, 1.0), min_sample=30, meaningful_delta=0.15,
    )
    features = so.compute_feature_evaluations(
        [_frow(sentiment_label="bullish", outcome="win", pnl_pct=2.0)], min_sample=1,
    )
    payload = so.build_payload(
        degradations, features, run_timestamp="2026-07-01T00:00:00",
        degrade_window_days=30, baseline_window_days=90,
    )
    run_id = so.persist_run(payload)
    assert run_id >= 1

    latest = db.get_latest_optimization_run()
    assert latest is not None
    assert latest["degrade_window_days"] == 30
    stored = json.loads(latest["findings_json"])
    assert stored["run_timestamp"] == "2026-07-01T00:00:00"

    text = so.render_report(stored)            # renders from the persisted dict
    assert "SELF-OPTIMIZATION REPORT" in text
    assert "FLAGS ONLY" in text
    assert "DEGRADATION" in text
    assert "degraded" in text                  # the overall finding
    assert "FEATURE EVALUATION" in text
    assert "sentiment" in text


def test_render_report_handles_empty_findings() -> None:
    payload = so.build_payload(
        [], [], run_timestamp="2026-07-01T00:00:00",
        degrade_window_days=30, baseline_window_days=90,
    )
    text = so.render_report(payload)
    assert "no resolved trades in the window" in text


def test_run_optimization_builds_payload_from_db(tmp_db: Path) -> None:
    _seed_resolved(tmp_db, ticker="GOOGL", outcome="win", pnl=2.0,
                   sentiment_label="bullish", closed_at=datetime(2026, 6, 20, 16, 0))
    payload = so.run_optimization(
        now=datetime(2026, 7, 1), degrade_window_days=30, baseline_window_days=90,
    )
    assert payload["run_timestamp"] == "2026-07-01T00:00:00"
    assert payload["degrade_window_days"] == 30
    assert payload["baseline_window_days"] == 90
    assert any(f["scope"] == "overall" for f in payload["degradations"])
    sent = next(f for f in payload["features"] if f["feature"] == "sentiment")
    assert any(b["label"] == "bullish" for b in sent["buckets"])
