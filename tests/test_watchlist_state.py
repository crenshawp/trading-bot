"""Tests for trading_bot.watchlist_state — the active/benched state machine.

Covers the transition rules (demote / recover), the hysteresis dead band, the
insufficient-sample no-op, the MIN_ACTIVE floor, per-ticker error isolation,
transition recording, and dry-run. Data is inserted directly; no network.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from trading_bot import config, db, watchlist_state
from trading_bot.models import Signal, Trade

NOW = datetime(2026, 6, 30, tzinfo=UTC)


def _seed(
    ticker: str,
    *,
    wins: int = 0,
    losses: int = 0,
    win_pnl: float = 2.0,
    loss_pnl: float = 1.0,
    track_mode: str = "active",
    start_idx: int = 0,
) -> int:
    """Insert ``wins`` + ``losses`` resolved trades closed inside the window.
    Losses are stored with negative pnl. Returns the next free idx."""
    idx = start_idx
    rows = [("win", win_pnl)] * wins + [("loss", -loss_pnl)] * losses
    for outcome, pnl in rows:
        ts = datetime(2026, 6, 1, tzinfo=UTC) + timedelta(minutes=idx)
        sid = db.insert_signal(
            Signal(timestamp=ts, ticker=ticker, asset_class="stock",
                   signal_type="ema21_pullback", direction="call",
                   entry_price=100.0)
        )
        db.insert_trade(
            Trade(signal_id=sid, opened_at=NOW - timedelta(days=6),
                  closed_at=NOW - timedelta(days=5), outcome=outcome,
                  pnl_pct=pnl, track_mode=track_mode)
        )
        idx += 1
    return idx


def _add(ticker: str, status: str) -> None:
    db.add_to_active_watchlist(ticker, "seed")
    if status != "active":
        db.set_watchlist_status(ticker, status)


def _by_ticker(run: watchlist_state.StateRun) -> dict[str, watchlist_state.TickerEvaluation]:
    return {e.ticker: e for e in run.evaluations}


# ───────────────────────── transitions ─────────────────────────


def test_demote_active_on_negative_expectancy(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "SM_MIN_ACTIVE", 0)  # isolate from the floor
    _add("BAD", "active")
    _seed("BAD", wins=2, losses=8)  # expectancy (4-8)/10 = -0.4

    run = watchlist_state.evaluate_watchlist(now=NOW)
    ev = _by_ticker(run)["BAD"]
    assert ev.decision == "demote"
    assert ev.new_status == "benched"
    # Applied + recorded.
    statuses = {e["ticker"]: e["status"] for e in db.get_watchlist_entries()}
    assert statuses["BAD"] == "benched"
    trans = db.get_watchlist_transitions()
    assert trans[0]["ticker"] == "BAD"
    assert trans[0]["from_status"] == "active"
    assert trans[0]["to_status"] == "benched"


def test_recover_benched_on_positive_expectancy(tmp_db: Path) -> None:
    _add("REC", "benched")
    _seed("REC", wins=8, losses=2, track_mode="shadow")  # expectancy +1.4

    run = watchlist_state.evaluate_watchlist(now=NOW)
    ev = _by_ticker(run)["REC"]
    assert ev.decision == "recover"
    assert ev.new_status == "active"
    statuses = {e["ticker"]: e["status"] for e in db.get_watchlist_entries()}
    assert statuses["REC"] == "active"


def test_dead_band_is_a_noop(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "SM_MIN_ACTIVE", 0)
    # Expectancy 0.03: above DEMOTE (0.0), below PROMOTE (0.05) -> nobody moves.
    _add("ACT", "active")
    _seed("ACT", wins=10, win_pnl=0.03)
    _add("BEN", "benched")
    _seed("BEN", wins=10, win_pnl=0.03, track_mode="shadow", start_idx=100)

    run = watchlist_state.evaluate_watchlist(now=NOW)
    by = _by_ticker(run)
    assert by["ACT"].decision == "hold"
    assert by["BEN"].decision == "hold"
    assert db.get_watchlist_transitions() == []


def test_insufficient_sample_is_a_noop(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "SM_MIN_ACTIVE", 0)
    _add("BAD", "active")
    _seed("BAD", wins=1, losses=4)  # 5 < SM_MIN_CLOSED_SIGNALS (10)

    run = watchlist_state.evaluate_watchlist(now=NOW)
    ev = _by_ticker(run)["BAD"]
    assert ev.decision == "hold"
    assert "insufficient sample" in ev.reason
    assert db.get_watchlist_transitions() == []


def test_min_active_floor_holds_a_demotion(tmp_db: Path) -> None:
    # SM_MIN_ACTIVE defaults to 5. With 6 active and 2 demote candidates, only
    # the single worst-expectancy one may go; the other is held by the floor.
    idx = 0
    for t in ("G1", "G2", "G3", "G4"):
        _add(t, "active")
        idx = _seed(t, wins=8, losses=2, start_idx=idx)        # exp +1.4 (hold)
    _add("WORST", "active")
    idx = _seed("WORST", wins=1, losses=9, start_idx=idx)       # exp -0.7
    _add("BAD", "active")
    idx = _seed("BAD", wins=3, losses=7, start_idx=idx)         # exp -0.1

    run = watchlist_state.evaluate_watchlist(now=NOW)
    by = _by_ticker(run)
    assert by["WORST"].decision == "demote"   # worst expectancy demoted
    assert by["BAD"].decision == "hold"        # held by the floor
    assert "active floor" in by["BAD"].reason
    # Resulting active count stays at the floor (5).
    assert len(run.active_after) == 5
    statuses = {e["ticker"]: e["status"] for e in db.get_watchlist_entries()}
    assert statuses["WORST"] == "benched"
    assert statuses["BAD"] == "active"


def test_per_ticker_error_holds_and_logs_not_silent(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _add("OK", "active")
    _seed("OK", wins=8, losses=2)
    _add("ERR", "active")

    real = watchlist_state.windowed_stats_for

    def flaky(ticker: str, *, now: datetime | None = None) -> tuple[int, float | None, float | None]:
        if ticker == "ERR":
            raise RuntimeError("stats boom")
        return real(ticker, now=now)

    monkeypatch.setattr(watchlist_state, "windowed_stats_for", flaky)
    run = watchlist_state.evaluate_watchlist(now=NOW)

    ev = _by_ticker(run)["ERR"]
    assert ev.decision == "hold"
    assert "error" in ev.reason
    assert "ERR" in capsys.readouterr().err
    # The error ticker caused no transition.
    assert all(t["ticker"] != "ERR" for t in db.get_watchlist_transitions())


def test_dry_run_changes_nothing(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "SM_MIN_ACTIVE", 0)
    _add("BAD", "active")
    _seed("BAD", wins=2, losses=8)  # would demote

    run = watchlist_state.evaluate_watchlist(dry_run=True, now=NOW)
    assert _by_ticker(run)["BAD"].decision == "demote"   # verdict still computed
    statuses = {e["ticker"]: e["status"] for e in db.get_watchlist_entries()}
    assert statuses["BAD"] == "active"                    # but not applied
    assert db.get_watchlist_transitions() == []


# ───────────────────────── readiness gate (Phase 10) ─────────────────────────


def test_dormant_when_capability_not_ready(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(config, "SM_MIN_ACTIVE", 0)
    _add("BAD", "active")
    _seed("BAD", wins=2, losses=8)  # 10 trades -> per-ticker gate passes, would demote
    monkeypatch.setattr("trading_bot.readiness.is_ready", lambda _name: False)

    run = watchlist_state.evaluate_watchlist(now=NOW)
    assert _by_ticker(run)["BAD"].decision == "demote"    # decision still visible
    statuses = {e["ticker"]: e["status"] for e in db.get_watchlist_entries()}
    assert statuses["BAD"] == "active"                     # but NOT applied (dormant)
    assert db.get_watchlist_transitions() == []
    assert "dormant" in capsys.readouterr().err


def test_acts_when_capability_ready(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "SM_MIN_ACTIVE", 0)
    _add("BAD", "active")
    _seed("BAD", wins=2, losses=8)
    monkeypatch.setattr("trading_bot.readiness.is_ready", lambda _name: True)

    watchlist_state.evaluate_watchlist(now=NOW)
    statuses = {e["ticker"]: e["status"] for e in db.get_watchlist_entries()}
    assert statuses["BAD"] == "benched"                    # applied when ready
