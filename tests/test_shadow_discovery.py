"""Tests for trading_bot.shadow_discovery — live-shadow promotion evaluator.

Plus the resolved-shadow DB query and the reporting filter that keeps shadow
trades out of headline stats. All data is inserted directly; no network.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from trading_bot import db, outcomes, performance, shadow_discovery
from trading_bot.models import Signal, Trade

# Fixed evaluation clock so window math is deterministic.
NOW = datetime(2026, 6, 30, tzinfo=UTC)


def _add_trade(
    ticker: str,
    outcome: str,
    pnl: float | None,
    *,
    idx: int,
    track_mode: str = "shadow",
    closed_at: datetime | None = None,
) -> None:
    """Insert a (signal, trade) pair. ``idx`` keeps each signal's timestamp
    unique so the signals dedupe index doesn't collapse them."""
    ts = datetime(2026, 6, 1, tzinfo=UTC) + timedelta(minutes=idx)
    sid = db.insert_signal(
        Signal(
            timestamp=ts,
            ticker=ticker,
            asset_class="stock",
            signal_type="ema21_pullback",
            direction="call",
            entry_price=100.0,
        )
    )
    ca = closed_at if closed_at is not None else NOW - timedelta(days=5)
    db.insert_trade(
        Trade(
            signal_id=sid,
            opened_at=ca - timedelta(days=1),
            closed_at=ca,
            outcome=outcome,
            pnl_pct=pnl,
            track_mode=track_mode,
        )
    )


def _seed(ticker: str, wins: int, losses: int, *, win_pnl: float, loss_pnl: float,
          track_mode: str = "shadow", start_idx: int = 0) -> int:
    """Seed ``wins`` winning + ``losses`` losing resolved trades. Returns next idx.

    ``win_pnl`` / ``loss_pnl`` are positive magnitudes; losses are stored as
    negative pnl_pct (a losing trade exits below entry)."""
    idx = start_idx
    for _ in range(wins):
        _add_trade(ticker, "win", win_pnl, idx=idx, track_mode=track_mode)
        idx += 1
    for _ in range(losses):
        _add_trade(ticker, "loss", -loss_pnl, idx=idx, track_mode=track_mode)
        idx += 1
    return idx


# ───────────────────────── _compute_stats ─────────────────────────


def test_compute_stats_math() -> None:
    rows = [("win", 2.0)] * 7 + [("loss", -1.0)] * 3
    closed, win_rate, expectancy = shadow_discovery._compute_stats(rows)
    assert closed == 10
    assert win_rate == pytest.approx(70.0)
    # (7*2 - 3*1) / 10 = 1.1
    assert expectancy == pytest.approx(1.1)


def test_compute_stats_empty() -> None:
    closed, win_rate, expectancy = shadow_discovery._compute_stats([])
    assert closed == 0
    assert win_rate is None
    assert expectancy is None


# ───────────────────────── evaluate_ticker (the gate) ─────────────────────────


def test_evaluate_ticker_eligible(tmp_db: Path) -> None:
    _seed("AAA", wins=7, losses=3, win_pnl=2.0, loss_pnl=1.0)
    ev = shadow_discovery.evaluate_ticker("AAA", now=NOW)
    assert ev.closed_count == 10
    assert ev.win_rate == pytest.approx(70.0)
    assert ev.expectancy == pytest.approx(1.1)
    assert ev.eligible is True


def test_evaluate_ticker_below_min_sample_not_eligible(tmp_db: Path) -> None:
    # 9 strong trades — positive expectancy but under MIN_SHADOW_SIGNALS (10).
    _seed("AAA", wins=7, losses=2, win_pnl=2.0, loss_pnl=1.0)
    ev = shadow_discovery.evaluate_ticker("AAA", now=NOW)
    assert ev.closed_count == 9
    assert ev.expectancy is not None and ev.expectancy > 0
    assert ev.eligible is False


def test_evaluate_ticker_below_expectancy_not_eligible(tmp_db: Path) -> None:
    # 10 trades, but expectancy 0.0 (< PROMOTE_EXPECTANCY 0.05).
    _seed("AAA", wins=5, losses=5, win_pnl=1.0, loss_pnl=1.0)
    ev = shadow_discovery.evaluate_ticker("AAA", now=NOW)
    assert ev.closed_count == 10
    assert ev.expectancy == pytest.approx(0.0)
    assert ev.eligible is False


def test_evaluate_ticker_ignores_active_open_expired_and_old(tmp_db: Path) -> None:
    # 10 qualifying shadow trades …
    idx = _seed("AAA", wins=8, losses=2, win_pnl=2.0, loss_pnl=1.0)
    # … plus noise that must NOT count:
    _add_trade("AAA", "win", 9.0, idx=idx, track_mode="active")          # active
    idx += 1
    _add_trade("AAA", "win", 9.0, idx=idx, track_mode="shadow",
               closed_at=NOW - timedelta(days=70))                       # too old
    idx += 1
    # open shadow trade (no outcome) — excluded by the win/loss filter
    ts = datetime(2026, 6, 1, tzinfo=UTC) + timedelta(minutes=idx)
    sid = db.insert_signal(Signal(timestamp=ts, ticker="AAA", asset_class="stock",
                                  signal_type="ema21_pullback", direction="call",
                                  entry_price=100.0))
    db.insert_trade(Trade(signal_id=sid, opened_at=ts, outcome="open",
                          track_mode="shadow"))

    ev = shadow_discovery.evaluate_ticker("AAA", now=NOW)
    assert ev.closed_count == 10  # only the 10 recent resolved shadow trades


def test_get_resolved_shadow_outcomes_query(tmp_db: Path) -> None:
    _add_trade("AAA", "win", 2.0, idx=0, track_mode="shadow")
    _add_trade("AAA", "loss", -1.0, idx=1, track_mode="shadow")
    _add_trade("AAA", "win", 5.0, idx=2, track_mode="active")   # excluded
    rows = db.get_resolved_shadow_outcomes("AAA", NOW - timedelta(days=60))
    assert sorted(o for o, _ in rows) == ["loss", "win"]


# ──────────────────── evaluate_shadow_universe (promotion) ────────────────────


def test_evaluate_universe_promotes_eligible_and_persists(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(shadow_discovery, "SHADOW_UNIVERSE", ["AAA", "BBB"])
    _seed("AAA", wins=8, losses=2, win_pnl=2.0, loss_pnl=1.0)   # eligible
    _seed("BBB", wins=2, losses=2, win_pnl=2.0, loss_pnl=1.0, start_idx=100)  # too few

    evals = shadow_discovery.evaluate_shadow_universe(now=NOW)

    by_ticker = {e.ticker: e for e in evals}
    assert by_ticker["AAA"].eligible is True
    assert by_ticker["AAA"].promoted is True
    assert by_ticker["BBB"].eligible is False
    assert by_ticker["BBB"].promoted is False

    watchlist = db.get_active_watchlist()
    assert "AAA" in watchlist
    assert "BBB" not in watchlist
    # The run is persisted for the record.
    persisted = {r["ticker"]: r for r in db.get_shadow_evaluations()}
    assert persisted["AAA"]["promoted"] is True
    assert persisted["BBB"]["promoted"] is False


def test_evaluate_universe_skips_active_watchlist_names(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(shadow_discovery, "SHADOW_UNIVERSE", ["AAA", "BBB"])
    db.add_to_active_watchlist("AAA", "seed")  # already live — not a candidate
    _seed("AAA", wins=9, losses=1, win_pnl=2.0, loss_pnl=1.0)

    evals = shadow_discovery.evaluate_shadow_universe(now=NOW)
    assert [e.ticker for e in evals] == ["BBB"]


def test_evaluate_universe_dry_run_does_not_promote_or_persist(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(shadow_discovery, "SHADOW_UNIVERSE", ["AAA"])
    _seed("AAA", wins=8, losses=2, win_pnl=2.0, loss_pnl=1.0)

    evals = shadow_discovery.evaluate_shadow_universe(dry_run=True, now=NOW)
    assert evals[0].eligible is True
    assert evals[0].promoted is False          # eligible but not acted on
    assert db.get_active_watchlist() == []     # nothing promoted
    assert db.get_shadow_evaluations() == []    # nothing persisted


def test_evaluate_universe_per_ticker_error_is_logged_not_silent(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(shadow_discovery, "SHADOW_UNIVERSE", ["AAA", "BBB"])
    _seed("BBB", wins=8, losses=2, win_pnl=2.0, loss_pnl=1.0)  # eligible

    real = db.get_resolved_shadow_outcomes

    def flaky(ticker: str, since: datetime) -> list[tuple[str, float | None]]:
        if ticker == "AAA":
            raise RuntimeError("db hiccup")
        return real(ticker, since)

    monkeypatch.setattr(db, "get_resolved_shadow_outcomes", flaky)
    evals = shadow_discovery.evaluate_shadow_universe(now=NOW)

    # AAA errored (skipped, no promotion); BBB still evaluated + promoted.
    assert [e.ticker for e in evals] == ["BBB"]
    assert "AAA" in capsys.readouterr().err
    assert "BBB" in db.get_active_watchlist()


# ──────────────────── reporting filter (shadow excluded) ────────────────────


def test_reporting_excludes_shadow_by_default(tmp_db: Path) -> None:
    _add_trade("AAA", "win", 2.0, idx=0, track_mode="active")
    _add_trade("AAA", "loss", -1.0, idx=1, track_mode="shadow")

    # stats_overall defaults to active only.
    assert performance.stats_overall().total == 1
    assert performance.stats_overall().win_rate == pytest.approx(100.0)
    # The shadow view and the unfiltered view see more.
    assert performance.stats_overall(track_mode="shadow").total == 1
    assert performance.stats_overall(track_mode=None).total == 2

    # outcomes.summary likewise defaults to active.
    active_decided = sum(
        e["wins"] + e["losses"]
        for e in outcomes.summary()["by_signal_type"]
    )
    shadow_decided = sum(
        e["wins"] + e["losses"]
        for e in outcomes.summary(track_mode="shadow")["by_signal_type"]
    )
    assert active_decided == 1
    assert shadow_decided == 1


# ───────────────────────── readiness gate (Phase 10) ─────────────────────────


def test_shadow_dormant_when_capability_not_ready(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(shadow_discovery, "SHADOW_UNIVERSE", ["AAA"])
    _seed("AAA", wins=8, losses=2, win_pnl=2.0, loss_pnl=1.0)   # eligible (10 shadow)
    monkeypatch.setattr("trading_bot.readiness.is_ready", lambda _name: False)

    evals = shadow_discovery.evaluate_shadow_universe(now=NOW)
    by_ticker = {e.ticker: e for e in evals}
    assert by_ticker["AAA"].eligible is True        # standing still computed
    assert by_ticker["AAA"].promoted is False       # but NOT promoted (dormant)
    assert "AAA" not in db.get_active_watchlist()
    assert db.get_shadow_evaluations() == []        # dormant -> no persist (like dry-run)
    assert "dormant" in capsys.readouterr().err


def test_shadow_acts_when_capability_ready(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(shadow_discovery, "SHADOW_UNIVERSE", ["AAA"])
    _seed("AAA", wins=8, losses=2, win_pnl=2.0, loss_pnl=1.0)
    monkeypatch.setattr("trading_bot.readiness.is_ready", lambda _name: True)

    shadow_discovery.evaluate_shadow_universe(now=NOW)
    assert "AAA" in db.get_active_watchlist()        # promoted when ready
