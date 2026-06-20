"""Offline demo of the Phase 3.3 watchlist state machine (active<->benched).

Run with: ``python demo_watchlist.py``

Seeds a throwaway database with a watchlist whose tickers have different recent
resolved-trade records, then runs the state evaluator to show:

* an underperformer demoted active -> benched,
* a turned-around benched ticker recovered benched -> active,
* the MIN_ACTIVE floor holding a demotion so the live set never shrinks too far,
* hysteresis dead-band and insufficient-sample names left untouched.

No network — everything runs against a temp SQLite file.
"""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

from trading_bot import config

config.DB_PATH = Path(tempfile.mkdtemp()) / "demo_watchlist.db"

from trading_bot import db, watchlist_state  # noqa: E402
from trading_bot.models import Signal, Trade  # noqa: E402

NOW = datetime(2026, 6, 30, tzinfo=UTC)
_IDX = [0]


def seed(ticker: str, wins: int, losses: int, *, win_pnl: float = 2.0,
         loss_pnl: float = 1.0, track_mode: str = "active") -> None:
    rows = [("win", win_pnl)] * wins + [("loss", -loss_pnl)] * losses
    for outcome, pnl in rows:
        ts = datetime(2026, 6, 1, tzinfo=UTC) + timedelta(minutes=_IDX[0])
        _IDX[0] += 1
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


def add(ticker: str, status: str) -> None:
    db.add_to_active_watchlist(ticker, "seed")
    if status != "active":
        db.set_watchlist_status(ticker, status)


def main() -> None:
    db.init_db()

    # 3 healthy active names (hold), 1 turning-around benched name (recover),
    # and 2 active underperformers. With 3 holds + 2 demote candidates = 5
    # active and 1 recovery, the MIN_ACTIVE floor (5) only lets the single
    # worst-expectancy one go this cycle and holds the other.
    for t in ("AAA", "BBB", "CCC"):
        add(t, "active")
        seed(t, 8, 2)                       # expectancy +1.4 -> hold
    add("WORST", "active")
    seed("WORST", 1, 9)                     # expectancy -0.7 -> demote (worst)
    add("BAD", "active")
    seed("BAD", 3, 7)                       # expectancy -0.1 -> held by floor
    add("COMEBACK", "benched")
    seed("COMEBACK", 8, 2, track_mode="shadow")  # expectancy +1.4 -> recover

    print("Before:", {e["ticker"]: e["status"] for e in db.get_watchlist_entries()})
    print()

    run = watchlist_state.evaluate_watchlist(now=NOW)
    for ev in run.evaluations:
        if ev.decision != "hold" or "floor" in ev.reason:
            print(f"  {ev.ticker:<8} {ev.status} -> {ev.new_status:<8} "
                  f"expectancy={ev.expectancy}  {ev.reason}")
    print()
    print("After active :", run.active_after)
    print("After benched:", run.benched_after)
    print()
    print("Transitions recorded:",
          [(t["ticker"], t["from_status"], t["to_status"])
           for t in db.get_watchlist_transitions()])


if __name__ == "__main__":
    main()
