"""Offline demo of Phase 15.5 evaluator scheduling + transition notifications.

Run with: ``python demo_evaluator_scheduling.py``

No network whatsoever: a throwaway SQLite database and a captured notifier. It
shows the two daily-run outcomes:

* a TRANSITION day — a pair that has run negative-expectancy (the exact incident
  shape) is muted by the daily evaluator run, and ONE combined Pushover fires;
* a NO-OP day — nothing changed, so NOTHING is sent (no daily noise).

Production is identical — only the notifier is a double, and the scanner's
existing daily-task mechanism (not shown here) is what invokes ``run_and_notify``
once per day.
"""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

from trading_bot import config

config.DB_PATH = Path(tempfile.mkdtemp()) / "demo_evalsched.db"

from trading_bot import db  # noqa: E402
from trading_bot import evaluator_scheduling as es  # noqa: E402
from trading_bot.models import Signal, Trade  # noqa: E402

_TS = datetime(2026, 6, 1, 9, 40, tzinfo=UTC)
_NOTES: list[tuple[str, str]] = []


def _notifier(title: str, message: str) -> bool:
    _NOTES.append((title, message))
    return True


def _seed_losing_pair() -> None:
    """11 resolved losses on a crypto pair — negative expectancy, past the
    pair-gating readiness threshold, exactly the incident shape."""
    for i in range(11):
        ts = _TS - timedelta(days=1, minutes=i)
        sid = db.insert_signal(Signal(
            timestamp=ts, ticker="BNB-USD", asset_class="crypto",
            signal_type="oversold_reversal", direction="long", entry_price=100.0,
        ))
        db.insert_trade(Trade(
            signal_id=sid, opened_at=ts, closed_at=ts + timedelta(hours=1),
            outcome="loss", pnl_pct=-2.0, track_mode="active",
        ))


def main() -> None:
    db.init_db()

    print("=== TRANSITION day: a negative-expectancy pair is muted ===")
    _seed_losing_pair()
    result = es.run_and_notify(now=_TS, notifier=_notifier)
    print(f"  pair transitions: {result.pair_transitions}  "
          f"watchlist transitions: {result.watchlist_transitions}  "
          f"notified: {result.notified}")
    for line in result.lines:
        print(f"    {line}")
    print(f"  BNB-USD/oversold_reversal status now: "
          f"{db.get_signal_pair_status('BNB-USD', 'oversold_reversal')}")
    print(f"  Pushover notifications sent: {len(_NOTES)}")
    for title, _msg in _NOTES:
        print(f"    [{title}]")

    print("\n=== NO-OP day: nothing changed -> nothing sent ===")
    before = len(_NOTES)
    result2 = es.run_and_notify(now=_TS + timedelta(days=1), notifier=_notifier)
    print(f"  pair transitions: {result2.pair_transitions}  "
          f"watchlist transitions: {result2.watchlist_transitions}  "
          f"notified: {result2.notified}")
    print(f"  Pushover notifications this run: {len(_NOTES) - before}  "
          "(silence on a no-op day)")


if __name__ == "__main__":
    main()
