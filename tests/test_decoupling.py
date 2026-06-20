"""Phase 3.3 — shadow promotion and benched recovery must never overlap.

Shadow promotion (entry) owns tickers NOT on the watchlist; the state
evaluator (management) owns tickers ON the watchlist (active OR benched). No
ticker may be processed by both paths in the same cycle.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from trading_bot import db, shadow_discovery, watchlist_state
from trading_bot.models import Signal, Trade

NOW = datetime(2026, 6, 30, tzinfo=UTC)


def _seed_resolved(ticker: str, wins: int, losses: int, *, track_mode: str,
                   start_idx: int) -> int:
    idx = start_idx
    for outcome, pnl in ([("win", 2.0)] * wins + [("loss", -1.0)] * losses):
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


def test_benched_ticker_is_owned_only_by_state_evaluator(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A benched ticker that is ALSO in the shadow universe and has eligible
    recovery stats must be evaluated by the state machine and ignored by shadow
    promotion — never both."""
    monkeypatch.setattr(shadow_discovery, "SHADOW_UNIVERSE", ["BEN", "FRESH"])

    # BEN is on the watchlist but benched, with strong recovery stats.
    db.add_to_active_watchlist("BEN", "shadow")
    db.set_watchlist_status("BEN", "benched")
    idx = _seed_resolved("BEN", wins=8, losses=2, track_mode="shadow", start_idx=0)
    # FRESH is a genuine off-watchlist shadow candidate with eligible stats.
    _seed_resolved("FRESH", wins=8, losses=2, track_mode="shadow", start_idx=idx)

    # Both paths read the SAME watchlist snapshot (dry-run, no mutation) so we
    # observe one cycle's ownership split rather than a sequential hand-off.
    shadow_run = shadow_discovery.evaluate_shadow_universe(dry_run=True, now=NOW)
    shadow_tickers = {e.ticker for e in shadow_run}

    state_run = watchlist_state.evaluate_watchlist(dry_run=True, now=NOW)
    state_tickers = {e.ticker for e in state_run.evaluations}

    # BEN is owned by the state evaluator, untouched by shadow.
    assert "BEN" in state_tickers
    assert "BEN" not in shadow_tickers
    # FRESH is owned by shadow, untouched by the state evaluator.
    assert "FRESH" in shadow_tickers
    assert "FRESH" not in state_tickers
    # The two paths are disjoint this cycle.
    assert shadow_tickers.isdisjoint(state_tickers)


def test_add_to_active_watchlist_cannot_repromote_benched(tmp_db: Path) -> None:
    """Second guard layer: even a direct promotion attempt is a no-op on a
    ticker already present, and never disturbs its benched status."""
    db.add_to_active_watchlist("BEN", "shadow")
    db.set_watchlist_status("BEN", "benched")

    assert db.add_to_active_watchlist("BEN", "shadow") is False
    statuses = {e["ticker"]: e["status"] for e in db.get_watchlist_entries()}
    assert statuses["BEN"] == "benched"
