"""Tests for trading_bot.signal_pairs — the (ticker, signal_type) gate evaluator.

Covers mute/enable transitions, the hysteresis dead band, insufficient-sample
no-op (stays default-enabled), default-enabled lookups, evaluation regardless
of ticker status, per-pair error isolation, transitions, and dry-run. Data is
inserted directly; no network.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from trading_bot import db, signal_pairs
from trading_bot.models import Signal, Trade

NOW = datetime(2026, 6, 30, tzinfo=UTC)
_IDX = [0]


def _seed(
    ticker: str,
    signal_type: str,
    *,
    wins: int = 0,
    losses: int = 0,
    win_pnl: float = 2.0,
    loss_pnl: float = 1.0,
    track_mode: str = "active",
) -> None:
    rows = [("win", win_pnl)] * wins + [("loss", -loss_pnl)] * losses
    for outcome, pnl in rows:
        ts = datetime(2026, 6, 1, tzinfo=UTC) + timedelta(minutes=_IDX[0])
        _IDX[0] += 1
        sid = db.insert_signal(
            Signal(timestamp=ts, ticker=ticker, asset_class="stock",
                   signal_type=signal_type, direction="call", entry_price=100.0)
        )
        db.insert_trade(
            Trade(signal_id=sid, opened_at=NOW - timedelta(days=6),
                  closed_at=NOW - timedelta(days=5), outcome=outcome,
                  pnl_pct=pnl, track_mode=track_mode)
        )


def _by_pair(run: signal_pairs.PairRun) -> dict[tuple[str, str], signal_pairs.PairEvaluation]:
    return {(e.ticker, e.signal_type): e for e in run.evaluations}


# ───────────────────────── default-enabled ─────────────────────────


def test_default_enabled_when_no_row(tmp_db: Path) -> None:
    assert db.get_signal_pair_status("GOOGL", "ema21_pullback") == "enabled"


def test_get_resolved_outcomes_filters_by_signal_type(tmp_db: Path) -> None:
    _seed("GOOGL", "ema21_pullback", wins=1)
    _seed("GOOGL", "trend_continuation", losses=1)
    cutoff = NOW - timedelta(days=60)
    ema = db.get_resolved_outcomes("GOOGL", cutoff, signal_type="ema21_pullback")
    assert [o for o, _ in ema] == ["win"]


# ───────────────────────── transitions ─────────────────────────


def test_mute_enabled_pair_on_negative_expectancy(tmp_db: Path) -> None:
    _seed("GOOGL", "ema21_pullback", wins=2, losses=8)  # expectancy -0.4

    run = signal_pairs.evaluate_signal_pairs(now=NOW)
    ev = _by_pair(run)[("GOOGL", "ema21_pullback")]
    assert ev.decision == "mute"
    assert ev.new_status == "muted"
    assert db.get_signal_pair_status("GOOGL", "ema21_pullback") == "muted"
    trans = db.get_signal_pair_transitions()
    assert trans[0]["ticker"] == "GOOGL"
    assert trans[0]["from_status"] == "enabled"
    assert trans[0]["to_status"] == "muted"


def test_enable_recovered_muted_pair(tmp_db: Path) -> None:
    db.set_signal_pair_status("META", "overbought_reversal", "muted")
    # Recovery trades fire as shadow but are still real outcomes.
    _seed("META", "overbought_reversal", wins=8, losses=2, track_mode="shadow")

    run = signal_pairs.evaluate_signal_pairs(now=NOW)
    ev = _by_pair(run)[("META", "overbought_reversal")]
    assert ev.decision == "enable"
    assert db.get_signal_pair_status("META", "overbought_reversal") == "enabled"


def test_dead_band_is_a_noop(tmp_db: Path) -> None:
    # Expectancy 0.03: above MUTE (0.0), below ENABLE (0.05) -> nobody moves.
    _seed("GOOGL", "ema21_pullback", wins=10, win_pnl=0.03)
    run = signal_pairs.evaluate_signal_pairs(now=NOW)
    assert _by_pair(run)[("GOOGL", "ema21_pullback")].decision == "hold"
    assert db.get_signal_pair_transitions() == []


def test_insufficient_sample_stays_default_enabled(tmp_db: Path) -> None:
    _seed("GOOGL", "ema21_pullback", wins=1, losses=4)  # 5 < SP_MIN (10)
    run = signal_pairs.evaluate_signal_pairs(now=NOW)
    ev = _by_pair(run)[("GOOGL", "ema21_pullback")]
    assert ev.decision == "hold"
    assert "insufficient sample" in ev.reason
    assert db.get_signal_pair_status("GOOGL", "ema21_pullback") == "enabled"


def test_evaluator_runs_on_benched_tickers_pairs(tmp_db: Path) -> None:
    # The pair belongs to a benched ticker — the evaluator must still mute it so
    # its status is correct when the ticker re-activates.
    db.add_to_active_watchlist("BEN", "seed")
    db.set_watchlist_status("BEN", "benched")
    _seed("BEN", "ema21_pullback", wins=2, losses=8, track_mode="shadow")

    run = signal_pairs.evaluate_signal_pairs(now=NOW)
    ev = _by_pair(run)[("BEN", "ema21_pullback")]
    assert ev.decision == "mute"
    assert db.get_signal_pair_status("BEN", "ema21_pullback") == "muted"


def test_dry_run_changes_nothing(tmp_db: Path) -> None:
    _seed("GOOGL", "ema21_pullback", wins=2, losses=8)  # would mute
    run = signal_pairs.evaluate_signal_pairs(dry_run=True, now=NOW)
    assert _by_pair(run)[("GOOGL", "ema21_pullback")].decision == "mute"
    assert db.get_signal_pair_status("GOOGL", "ema21_pullback") == "enabled"
    assert db.get_signal_pair_transitions() == []


def test_per_pair_error_holds_and_logs_not_silent(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _seed("OK", "ema21_pullback", wins=8, losses=2)
    _seed("ERR", "ema21_pullback", wins=2, losses=8)

    real = signal_pairs.windowed_stats_for_pair

    def flaky(ticker: str, signal_type: str, *, now: datetime | None = None) -> tuple[int, float | None, float | None]:
        if ticker == "ERR":
            raise RuntimeError("stats boom")
        return real(ticker, signal_type, now=now)

    monkeypatch.setattr(signal_pairs, "windowed_stats_for_pair", flaky)
    run = signal_pairs.evaluate_signal_pairs(now=NOW)

    ev = _by_pair(run)[("ERR", "ema21_pullback")]
    assert ev.decision == "hold"
    assert "error" in ev.reason
    assert "ERR" in capsys.readouterr().err
    assert all(t["ticker"] != "ERR" for t in db.get_signal_pair_transitions())


# ───────────────────────── pair_stats ─────────────────────────


def test_pair_stats_reports_status_and_window(tmp_db: Path) -> None:
    _seed("GOOGL", "ema21_pullback", wins=8, losses=2)
    db.set_signal_pair_status("GOOGL", "ema21_pullback", "muted")
    stats = {(s.ticker, s.signal_type): s for s in signal_pairs.pair_stats(now=NOW)}
    s = stats[("GOOGL", "ema21_pullback")]
    assert s.status == "muted"
    assert s.closed_count == 10
    assert s.expectancy == pytest.approx(1.4)


def test_window_excludes_old_outcomes(tmp_db: Path) -> None:
    # Old (outside the window) trades must not count.
    old_close = NOW - timedelta(days=70)
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    sid = db.insert_signal(
        Signal(timestamp=ts, ticker="GOOGL", asset_class="stock",
               signal_type="ema21_pullback", direction="call", entry_price=100.0)
    )
    db.insert_trade(
        Trade(signal_id=sid, opened_at=old_close - timedelta(days=1),
              closed_at=old_close, outcome="loss", pnl_pct=-5.0,
              track_mode="active")
    )
    closed, _wr, _exp = signal_pairs.windowed_stats_for_pair(
        "GOOGL", "ema21_pullback", now=NOW
    )
    assert closed == 0
