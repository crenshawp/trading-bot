"""Consolidated Phase 24 earnings/news persistence and gate regressions.

All inputs are local and deterministic. No earnings, news, market-data, or
broker network call is permitted by these tests.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from trading_bot import allocation, candidate_source, db, regime, scanner, vix
from trading_bot.models import Signal, Trade


def _detector_frame() -> pd.DataFrame:
    row = {
        "Close": 100.0,
        "RSI": 50.0,
        "ATR": 2.0,
        "Volume": 100.0,
        "Vol_MA20": 100.0,
        "EMA21": 100.0,
        "EMA50": 90.0,
        "Slope": 0.0,
        "ROC_Accel": 1.0,
        "Slope_Accel": 0.0,
        "High_20": 110.0,
    }
    return pd.DataFrame([row, row])


def _offline_log_context(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_regime(*_args: object, **_kwargs: object) -> object:
        raise regime.RegimeFetchError("offline Phase 24 test")

    def no_vix(*_args: object, **_kwargs: object) -> object:
        raise vix.VixFetchError("offline Phase 24 test")

    monkeypatch.setattr(regime, "get_current_regime", no_regime)
    monkeypatch.setattr(vix, "get_current_vix", no_vix)


@pytest.mark.parametrize(
    ("earnings_grade", "news_grade", "expected_setup"),
    [
        ("HIGH — Earnings in 2 days", "LOW", "Earnings Risk"),
        ("LOW", "HIGH — 2 high-risk articles", "High Risk News"),
    ],
)
def test_high_detector_grades_still_stop_before_persistence(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    earnings_grade: str,
    news_grade: str,
    expected_setup: str,
) -> None:
    monkeypatch.setattr(scanner, "check_earnings_risk", lambda _t: earnings_grade)
    monkeypatch.setattr(scanner, "check_news_risk", lambda _t: news_grade)

    signals = scanner.detect_stock_signals("META", _detector_frame())

    assert len(signals) == 1
    assert signals[0]["setup"] == expected_setup
    assert scanner.log_signal(signals[0]) is None
    assert db.get_signals() == []


def test_soft_grades_flow_fire_to_candidate_and_reporting_without_new_gate(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _offline_log_context(monkeypatch)
    earnings_grade = "MEDIUM — Earnings in 10 days"
    news_grade = "MEDIUM — 1 medium-risk article"
    monkeypatch.setattr(scanner, "check_earnings_risk", lambda _t: earnings_grade)
    monkeypatch.setattr(scanner, "check_news_risk", lambda _t: news_grade)

    [fired] = scanner.detect_stock_signals("META", _detector_frame())
    signal_id = scanner.log_signal(fired)
    assert signal_id is not None

    persisted = db.get_signal_by_id(signal_id)
    assert persisted is not None
    assert persisted.earnings_risk == earnings_grade
    assert persisted.news_risk == news_grade

    now = datetime.now(UTC) + timedelta(seconds=1)
    [candidate] = candidate_source.live_candidates(now=now)
    assert candidate.earnings_blackout is False
    assert "news_risk" not in candidate.__dataclass_fields__

    [reported] = db.get_recent_signal_risk_grades(limit=1)
    assert reported["earnings_risk"] == earnings_grade
    assert reported["news_risk"] == news_grade


def test_stored_high_earnings_grade_remains_the_only_allocation_blackout(
    tmp_db: Path,
) -> None:
    now = datetime.now(UTC)
    signal_id = db.insert_signal(Signal(
        timestamp=now,
        ticker="LLY",
        asset_class="stock",
        signal_type="ema21_pullback",
        direction="call",
        entry_price=950.0,
        atr=15.0,
        earnings_risk="HIGH — synthetic defense-in-depth row",
        news_risk="LOW",
    ))
    db.insert_trade(Trade(
        signal_id=signal_id,
        opened_at=now,
        outcome="open",
        track_mode="active",
    ))

    [candidate] = candidate_source.live_candidates(now=now + timedelta(seconds=1))

    assert candidate.earnings_blackout is True
    assert allocation.filter_reason(candidate, 30_000.0) == "earnings_blackout"


def test_legacy_zero_rows_are_not_backfilled_or_behaviorally_changed(
    tmp_db: Path,
) -> None:
    now = datetime.now(UTC)
    signal_id = db.insert_signal(Signal(
        timestamp=now,
        ticker="GOOGL",
        asset_class="stock",
        signal_type="ema21_pullback",
        direction="call",
        entry_price=180.0,
        atr=3.0,
    ))
    db.insert_trade(Trade(
        signal_id=signal_id,
        opened_at=now,
        outcome="open",
        track_mode="active",
    ))
    conn = db.get_connection()
    try:
        conn.execute(
            "UPDATE signals SET earnings_risk = 0, news_risk = 0 WHERE id = ?",
            (signal_id,),
        )
        conn.commit()
        raw = conn.execute(
            "SELECT earnings_risk, news_risk FROM signals WHERE id = ?",
            (signal_id,),
        ).fetchone()
    finally:
        conn.close()

    assert str(raw["earnings_risk"]) == "0"
    assert str(raw["news_risk"]) == "0"
    persisted = db.get_signal_by_id(signal_id)
    assert persisted is not None
    assert persisted.earnings_risk == "UNKNOWN"
    assert persisted.news_risk == "UNKNOWN"
    [candidate] = candidate_source.live_candidates(now=now + timedelta(seconds=1))
    assert candidate.earnings_blackout is False

    # Reads normalize for display but never mutate the legacy storage.
    db.get_recent_signal_risk_grades(limit=1)
    conn = db.get_connection()
    try:
        unchanged = conn.execute(
            "SELECT earnings_risk, news_risk FROM signals WHERE id = ?",
            (signal_id,),
        ).fetchone()
    finally:
        conn.close()
    assert str(unchanged["earnings_risk"]) == "0"
    assert str(unchanged["news_risk"]) == "0"
