"""Offline demo of the Phase 3.1-LIVE live-shadow promotion loop.

Run with: ``python demo_shadow.py``

Seeds a throwaway database with RESOLVED shadow trades for a few tickers, runs
the promotion evaluator, and shows that:

* an eligible ticker (enough resolved shadow signals + positive expectancy) is
  promoted into the active watchlist (source='shadow'),
* a thin-sample / negative-expectancy ticker is withheld,
* shadow trades are excluded from the default (active) win-rate reporting.

No network, no real watchlist — everything runs against a temp SQLite file.
"""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

from trading_bot import config

# Point the whole stack at a throwaway DB BEFORE importing db-backed modules.
config.DB_PATH = Path(tempfile.mkdtemp()) / "demo_shadow.db"

from trading_bot import db, outcomes, performance, shadow_discovery  # noqa: E402
from trading_bot.models import Signal, Trade  # noqa: E402

NOW = datetime(2026, 6, 30, tzinfo=UTC)


def add_trade(ticker: str, outcome: str, pnl: float, idx: int) -> None:
    """Insert a resolved shadow (signal, trade) pair closed 5 days ago."""
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
    db.insert_trade(
        Trade(
            signal_id=sid,
            opened_at=NOW - timedelta(days=6),
            closed_at=NOW - timedelta(days=5),
            outcome=outcome,
            pnl_pct=pnl,
            track_mode="shadow",
        )
    )


def main() -> None:
    db.init_db()

    # WINNER: 8 wins / 2 losses of shadow signals -> strong positive expectancy.
    idx = 0
    for _ in range(8):
        add_trade("WINNER", "win", 2.0, idx)
        idx += 1
    for _ in range(2):
        add_trade("WINNER", "loss", -1.0, idx)
        idx += 1
    # THIN: only 4 resolved signals -> below the minimum sample.
    for outcome, pnl in [("win", 2.0), ("win", 2.0), ("loss", -1.0), ("win", 2.0)]:
        add_trade("THIN", outcome, pnl, idx)
        idx += 1
    # One ACTIVE trade so we can show reporting excludes shadow by default.
    ats = datetime(2026, 6, 5, tzinfo=UTC)
    asid = db.insert_signal(
        Signal(timestamp=ats, ticker="LIVE", asset_class="stock",
               signal_type="ema21_pullback", direction="call", entry_price=100.0)
    )
    db.insert_trade(Trade(signal_id=asid, opened_at=ats,
                          closed_at=ats + timedelta(days=1), outcome="loss",
                          pnl_pct=-1.0, track_mode="active"))

    # Restrict the universe to our demo tickers and evaluate.
    shadow_discovery.SHADOW_UNIVERSE = ["WINNER", "THIN"]

    print("Before evaluation, active watchlist:", db.get_active_watchlist())
    evals = shadow_discovery.evaluate_shadow_universe(now=NOW)
    print()
    for ev in evals:
        verdict = "PROMOTED" if ev.promoted else (
            "eligible" if ev.eligible else "withheld"
        )
        print(
            f"  {ev.ticker:<8} resolved={ev.closed_count:<3} "
            f"win_rate={ev.win_rate}  expectancy={ev.expectancy}  -> {verdict}"
        )
    print()
    print("After evaluation, active watchlist:", db.get_active_watchlist())

    print()
    active = performance.stats_overall()
    shadow = performance.stats_overall(track_mode="shadow")
    print(f"Reporting (default=active): {active.total} closed trades")
    print(f"Reporting (shadow view):    {shadow.total} closed trades")
    print(f"outcomes.summary active rows: {len(outcomes.summary()['by_signal_type'])}")


if __name__ == "__main__":
    main()
