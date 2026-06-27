"""Offline demo of the Phase 10 unified readiness gate.

Run with: ``python demo_readiness.py``

Synthetic data, no network (the notifier is captured, not sent). It shows:

* a capability that stays WARMING below its threshold (no notification);
* a threshold crossing where a DETERMINISTIC capability auto-ACTIVATES and an ML
  capability summons a human BUILD — each notifying EXACTLY ONCE;
* a re-run that does NOT re-notify (the crossing is a one-time edge event).

To keep it cheap, the demo swaps in a small-threshold registry (the real ML gate
is 500+). The behavior is identical to production — only the numbers shrink.
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from trading_bot import config

config.DB_PATH = Path(tempfile.mkdtemp()) / "demo_readiness.db"

from trading_bot import db, readiness  # noqa: E402
from trading_bot.models import Signal, Trade  # noqa: E402

# A demo registry with tiny thresholds (production: 10 / 500). Same machinery.
readiness.REGISTRY = (
    readiness.Capability(
        "watchlist_rotation", "deterministic", 3, "all",
        "active<->benched ticker rotation",
    ),
    readiness.Capability(
        "ml_pattern_recognition", "ml", 3, "all", "pattern recognition model",
    ),
)

_NOTES: list[tuple[str, str]] = []


def _notifier(title: str, message: str) -> bool:
    """Capture the notification instead of sending it. Returns success."""
    _NOTES.append((title, message))
    return True


def _seed(n: int, start: int = 0) -> None:
    for i in range(start, start + n):
        ts = datetime(2026, 1, 1, 9, 0) + timedelta(minutes=i)
        sid = db.insert_signal(Signal(
            timestamp=ts, ticker=f"T{i}", asset_class="stock",
            signal_type="ema21_pullback", direction="call", entry_price=100.0,
        ))
        db.insert_trade(Trade(
            signal_id=sid, opened_at=ts, closed_at=ts + timedelta(days=1),
            outcome="win", pnl_pct=2.0, track_mode="active",
        ))


def _show() -> None:
    for cap in readiness.REGISTRY:
        n = readiness.resolved_count(cap.name)
        ready = readiness.is_ready(cap.name)
        state = db.get_readiness_state(cap.name)
        announced = bool(state["announced"]) if state else False
        print(f"    {cap.name:<24} {cap.kind:<13} n={n}/{cap.threshold}  "
              f"status={'ready' if ready else 'warming':<7}  announced={announced}")


def main() -> None:
    db.init_db()
    now = datetime(2026, 6, 30, 12, 0)

    print("=== Below threshold: warming, no notification ===")
    _seed(2)
    readiness.evaluate_readiness(notifier=_notifier, now=now)
    _show()
    print(f"  notifications so far: {len(_NOTES)}")

    print("\n=== Crossing: deterministic auto-activates, ML summons a build ===")
    _seed(1, start=2)   # now 3 resolved -> both cross
    readiness.evaluate_readiness(notifier=_notifier, now=now)
    _show()
    for title, message in _NOTES:
        print(f"  [{title}] {message}")

    print("\n=== Re-run: one-time edge event, no re-notification ===")
    before = len(_NOTES)
    readiness.evaluate_readiness(notifier=_notifier, now=now)
    print(f"  new notifications this pass: {len(_NOTES) - before}")


if __name__ == "__main__":
    main()
