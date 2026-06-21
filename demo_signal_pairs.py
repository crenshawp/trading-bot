"""Offline demo of the Phase 4 per-(ticker, signal_type) gate.

Run with: ``python demo_signal_pairs.py``

Seeds a throwaway database with resolved trades for several (ticker, setup)
pairs, then runs the per-pair evaluator to show:

* a losing pair MUTED (it will fire as shadow, no alert, but keep collecting),
* a recovered (previously muted) pair re-ENABLED,
* a winning pair and a thin-sample pair left untouched (default-enabled),
* and that muting one setup on a ticker leaves its other setups alerting.

No network — everything runs against a temp SQLite file.
"""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

from trading_bot import config

config.DB_PATH = Path(tempfile.mkdtemp()) / "demo_signal_pairs.db"

from trading_bot import db, signal_pairs  # noqa: E402
from trading_bot.models import Signal, Trade  # noqa: E402

NOW = datetime(2026, 6, 30, tzinfo=UTC)
_IDX = [0]


def seed(ticker: str, signal_type: str, wins: int, losses: int, *,
         win_pnl: float = 2.0, loss_pnl: float = 1.0,
         track_mode: str = "active") -> None:
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


def main() -> None:
    db.init_db()

    # GOOGL/ema21_pullback is losing -> mute. GOOGL/trend_continuation wins ->
    # stays enabled (so GOOGL still alerts on trend_continuation).
    seed("GOOGL", "ema21_pullback", 2, 8)
    seed("GOOGL", "trend_continuation", 8, 2)
    # META/overbought_reversal was muted earlier and has now recovered -> enable.
    db.set_signal_pair_status("META", "overbought_reversal", "muted")
    seed("META", "overbought_reversal", 8, 2, track_mode="shadow")
    # AMZN/higher_high_breakout has too few resolved signals -> stays enabled.
    seed("AMZN", "higher_high_breakout", 2, 2)

    print("Before:", db.get_signal_pair_statuses() or "(all default-enabled)")
    print()

    run = signal_pairs.evaluate_signal_pairs(now=NOW)
    for ev in run.evaluations:
        verb = ev.decision if ev.decision != "hold" else "hold"
        print(f"  {ev.ticker}/{ev.signal_type:<22} {ev.status:<8} -> "
              f"{ev.new_status:<8} expectancy={ev.expectancy}  {verb}: {ev.reason}")
    print()

    print("After (explicit rows):", db.get_signal_pair_statuses())
    print("Effective GOOGL/ema21_pullback     :",
          db.get_signal_pair_status("GOOGL", "ema21_pullback"))
    print("Effective GOOGL/trend_continuation :",
          db.get_signal_pair_status("GOOGL", "trend_continuation"))
    print("Transitions:",
          [(t["ticker"], t["signal_type"], t["from_status"], t["to_status"])
           for t in db.get_signal_pair_transitions()])


if __name__ == "__main__":
    main()
