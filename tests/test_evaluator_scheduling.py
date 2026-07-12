"""Tests for daily evaluator scheduling + transition notifications (Phase 15.5).

Mocked data + mocked Pushover only, no live network. The evaluators auto-act on
their deterministic capabilities (Phase 10); this phase adds the missing daily
call site and a per-event transition notification — no evaluator logic changes.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from trading_bot import db
from trading_bot import evaluator_scheduling as es
from trading_bot.models import Signal, Trade

_TS = datetime(2026, 6, 1, 9, 40, tzinfo=UTC)


def _seed_resolved_pair(
    ticker: str, signal_type: str, *, wins: int, losses: int,
    asset_class: str = "crypto",
) -> None:
    """Seed resolved trades for one (ticker, signal_type) pair."""
    idx = 0
    for outcome, count, pnl in (("win", wins, 2.0), ("loss", losses, -2.0)):
        for _ in range(count):
            ts = _TS - timedelta(days=1, minutes=idx)
            sid = db.insert_signal(Signal(
                timestamp=ts, ticker=ticker, asset_class=asset_class,
                signal_type=signal_type, direction="long", entry_price=100.0,
            ))
            db.insert_trade(Trade(
                signal_id=sid, opened_at=ts, closed_at=ts + timedelta(hours=1),
                outcome=outcome, pnl_pct=pnl, track_mode="active",
            ))
            idx += 1


def _losing_pair() -> None:
    # 11 resolved, all losses -> negative expectancy -> a MUTE transition, and
    # >= the pair-gating readiness threshold (10) so it auto-acts.
    _seed_resolved_pair("BNB-USD", "oversold_reversal", wins=0, losses=11)


# ───────────────────────── run_daily_evaluators ─────────────────────────────


def test_run_daily_evaluators_calls_both_in_real_mode(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, bool] = {}

    def fake_pairs(*, dry_run: bool, now: object = None):  # type: ignore[no-untyped-def]
        calls["pairs_dry_run"] = dry_run
        return es.signal_pairs.PairRun(evaluated_at=_TS, evaluations=[])

    def fake_watchlist(*, dry_run: bool, now: object = None):  # type: ignore[no-untyped-def]
        calls["watchlist_dry_run"] = dry_run
        return es.watchlist_state.StateRun(evaluated_at=_TS, evaluations=[])

    monkeypatch.setattr(es.signal_pairs, "evaluate_signal_pairs", fake_pairs)
    monkeypatch.setattr(es.watchlist_state, "evaluate_watchlist", fake_watchlist)

    es.run_daily_evaluators(now=_TS)
    # BOTH evaluators ran, in REAL (non-dry-run) mode.
    assert calls == {"pairs_dry_run": False, "watchlist_dry_run": False}


def test_run_daily_evaluators_no_transitions(
    tmp_db: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    # Empty DB -> no traded pairs, no watchlist rotation -> zero transitions.
    result = es.run_daily_evaluators(now=_TS)
    assert result.total_transitions == 0
    assert result.lines == []
    assert result.notified is False
    assert "no transitions" in capsys.readouterr().err


def test_run_daily_evaluators_reports_a_pair_mute(tmp_db: Path) -> None:
    _losing_pair()
    result = es.run_daily_evaluators(now=_TS)
    assert result.pair_transitions == 1
    assert any("BNB-USD/oversold_reversal" in line for line in result.lines)
    assert any("enabled -> muted" in line for line in result.lines)
    # The transition actually applied (the evaluator auto-acted).
    assert db.get_signal_pair_status("BNB-USD", "oversold_reversal") == "muted"


def test_transition_line_includes_expectancy_and_reason(tmp_db: Path) -> None:
    _losing_pair()
    result = es.run_daily_evaluators(now=_TS)
    line = next(line for line in result.lines if "BNB-USD" in line)
    assert "expectancy" in line and "n=11" in line
    assert "mute bar" in line          # the evaluator's own reason string
