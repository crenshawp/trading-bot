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


# ───────────────────────── notify_transitions ──────────────────────────────


class _Recorder:
    """A fake notifier recording (title, message) calls."""

    def __init__(self, *, succeed: bool = True) -> None:
        self.calls: list[tuple[str, str]] = []
        self.succeed = succeed

    def __call__(self, title: str, message: str) -> bool:
        self.calls.append((title, message))
        return self.succeed


def test_notify_fires_once_with_content_on_transitions(tmp_db: Path) -> None:
    _losing_pair()
    result = es.run_daily_evaluators(now=_TS)
    rec = _Recorder()
    notified = es.notify_transitions(result, notifier=rec)
    assert notified is True
    assert len(rec.calls) == 1                         # exactly one push
    title, body = rec.calls[0]
    assert "1 pair" in title
    assert "BNB-USD/oversold_reversal" in body
    assert "enabled -> muted" in body


def test_notify_combines_both_evaluators_into_one_push(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Force both evaluators to report a transition; the notification must be a
    # SINGLE combined push, not two separate ones.
    pair_ev = es.signal_pairs.PairEvaluation(
        ticker="BNB-USD", signal_type="oversold_reversal", status="enabled",
        closed_count=11, win_rate=0.0, expectancy=-2.0, decision="mute",
        reason="expectancy -2.000 <= mute bar 0.0",
    )
    wl_ev = es.watchlist_state.TickerEvaluation(
        ticker="TSLA", status="active", closed_count=12, win_rate=40.0,
        expectancy=-0.5, decision="demote", reason="expectancy -0.500 <= demote bar 0.0",
    )
    monkeypatch.setattr(
        es.signal_pairs, "evaluate_signal_pairs",
        lambda **k: es.signal_pairs.PairRun(evaluated_at=_TS, evaluations=[pair_ev]),
    )
    monkeypatch.setattr(
        es.watchlist_state, "evaluate_watchlist",
        lambda **k: es.watchlist_state.StateRun(evaluated_at=_TS, evaluations=[wl_ev]),
    )
    result = es.run_daily_evaluators(now=_TS)
    rec = _Recorder()
    es.notify_transitions(result, notifier=rec)
    assert result.pair_transitions == 1 and result.watchlist_transitions == 1
    assert len(rec.calls) == 1                         # ONE combined push
    _title, body = rec.calls[0]
    assert "PAIR BNB-USD/oversold_reversal" in body
    assert "WATCHLIST TSLA" in body


def test_notify_sends_nothing_on_zero_transitions(
    tmp_db: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    result = es.run_daily_evaluators(now=_TS)       # empty DB -> no transitions
    rec = _Recorder()
    notified = es.notify_transitions(result, notifier=rec)
    assert notified is False
    assert rec.calls == []                             # Pushover NOT called
    assert "no notification" in capsys.readouterr().err


def test_notify_is_failsoft_on_send_error(tmp_db: Path) -> None:
    _losing_pair()
    result = es.run_daily_evaluators(now=_TS)

    def boom(_title: str, _message: str) -> bool:
        raise RuntimeError("pushover down")

    # A send failure is swallowed — does NOT raise or block the cycle.
    assert es.notify_transitions(result, notifier=boom) is False


def test_notify_reports_send_failure_status(tmp_db: Path) -> None:
    _losing_pair()
    result = es.run_daily_evaluators(now=_TS)
    rec = _Recorder(succeed=False)                     # non-2xx / creds unset
    assert es.notify_transitions(result, notifier=rec) is False
    assert len(rec.calls) == 1                         # attempted, just not delivered


def test_run_and_notify_end_to_end(tmp_db: Path) -> None:
    _losing_pair()
    rec = _Recorder()
    result = es.run_and_notify(now=_TS, notifier=rec)
    assert result.notified is True
    assert result.pair_transitions == 1
    assert len(rec.calls) == 1


def test_default_notifier_uses_readiness_pushover(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The default notifier reuses the readiness Pushover client (not a new one).
    captured: list[tuple[str, str]] = []
    monkeypatch.setattr(
        es.readiness, "_pushover_notify",
        lambda title, message: captured.append((title, message)) or True,
    )
    assert es._default_notifier("t", "m") is True
    assert captured == [("t", "m")]


# ───────────────────────── scanner hook (additive, not duplicating) ─────────


def test_scanner_hook_calls_run_and_notify(monkeypatch: pytest.MonkeyPatch) -> None:
    from trading_bot import scanner
    called: list[bool] = []
    monkeypatch.setattr(
        scanner.evaluator_scheduling, "run_and_notify",
        lambda **k: called.append(True) or es.EvaluatorRunResult(0, 0),
    )
    scanner._run_evaluator_cycle()
    assert called == [True]


def test_scanner_hook_does_not_replace_morning_report_or_daily_perf() -> None:
    # The new daily hook is ADDITIVE — the existing daily triggers still exist as
    # distinct, independent callables.
    from trading_bot import scanner
    assert callable(scanner.send_morning_report)
    assert callable(scanner._run_daily_perf_update)
    assert callable(scanner._run_evaluator_cycle)
    assert scanner._run_evaluator_cycle is not scanner._run_daily_perf_update
    assert scanner._run_evaluator_cycle is not scanner.send_morning_report
