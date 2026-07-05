"""Offline demo of the Phase 15 risk-of-ruin safety layer.

Run with: ``python demo_risk_of_ruin.py``

No network whatsoever: a throwaway SQLite database, a :class:`FakeBroker`, and a
captured notifier. It shows:

* a TIER-1 pause — seven consecutive losses revoke ``new_position_entry`` (the
  allocation plan goes entries-empty; open positions stay untouched);
* operator re-authorization with the confirmation token;
* a full TIER-2 emergency shutdown — every open position (option, long-term,
  broker-side equity) closed via the EXISTING closers, closure confirmed by
  reconciliation, then the halt with the emergency notification;
* the kill switch refusing a wrong token.

Production behavior is identical — only the broker and notifier are doubles.
"""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

from trading_bot import config

config.DB_PATH = Path(tempfile.mkdtemp()) / "demo_ror.db"

from trading_bot import allocation, db  # noqa: E402
from trading_bot import risk_of_ruin as ror  # noqa: E402
from trading_bot.broker.base import AccountInfo  # noqa: E402
from trading_bot.broker.fake import FakeBroker  # noqa: E402
from trading_bot.models import (  # noqa: E402
    LongTermPosition,
    OptionPosition,
    Signal,
    Trade,
)

_TS = datetime(2026, 1, 1, 9, 0, tzinfo=UTC)
_NOTES: list[tuple[str, str]] = []


def _notifier(title: str, message: str) -> bool:
    _NOTES.append((title, message))
    return True


class _ClosingFakeBroker(FakeBroker):
    """Accepted SELLs clear the broker-side position so reconcile confirms flat."""

    def submit_order(self, symbol, qty, side, **kw):  # type: ignore[no-untyped-def]
        order = super().submit_order(symbol, qty, side, **kw)
        if order.ok and side == "sell":
            self._positions.pop(symbol, None)
        return order


def _seed_losses(n: int) -> None:
    for i in range(n):
        ts = _TS + timedelta(minutes=i)
        sid = db.insert_signal(Signal(
            timestamp=ts, ticker=f"T{i}", asset_class="stock",
            signal_type="ema21_pullback", direction="call", entry_price=100.0,
        ))
        db.insert_trade(Trade(
            signal_id=sid, opened_at=ts, closed_at=ts + timedelta(days=1),
            outcome="loss", pnl_pct=-1.0, track_mode="active",
        ))


def main() -> None:
    db.init_db()

    print("=== TIER 1: seven consecutive losses -> pause new entries ===")
    _seed_losses(7)
    result = ror.evaluate_tier1(notifier=_notifier)
    print(f"  tripped={result.tripped} trigger={result.trigger} "
          f"state={result.state}")
    print(f"  entry authorized: {ror.is_entry_authorized()}")

    print("\n=== The allocation plan is now entries-empty ===")
    candidate = allocation.Candidate(
        ticker="META", signal_type="ema21_pullback", direction="call",
        asset_class="stock", entry=480.0, atr=8.0, expectancy=0.9,
    )
    plan = allocation.build_plan(
        [candidate], AccountInfo(ok=True, equity=100_000.0, cash=100_000.0),
        entry_authorized=ror.is_entry_authorized(),
    )
    print(f"  orders={len(plan.plan.orders)}  "
          f"skipped={[(s.ticker, s.reason) for s in plan.skipped]}")

    print("\n=== Operator re-authorization (token-gated) ===")
    ok, message = ror.reauthorize(config.ROR_REAUTHORIZE_TOKEN)
    print(f"  {message}")

    print("\n=== TIER 2: emergency shutdown closes EVERYTHING, then halts ===")
    db.insert_option_position(OptionPosition(
        symbol="AAPL260116C00150000", underlying="AAPL", option_type="call",
        strike=150.0, expiry="2026-01-16", contracts=2.0, opened_at=_TS,
        premium_entry=5.0, outcome="open",
    ))
    db.insert_long_term_position(LongTermPosition(
        ticker="META", asset_class="stock", entry_price=480.0, entry_date=_TS,
        qty=5.0, status="open",
    ))
    broker = _ClosingFakeBroker()
    broker.set_position("NVDA", 3.0, avg_entry_price=120.0)
    shutdown = ror.emergency_shutdown(
        broker, trigger="3 consecutive broker errors", notifier=_notifier,
        now=_TS, option_price_fetch=lambda _s: 6.0,
        long_term_price_fetch=lambda _t: 470.0,
    )
    print(f"  status={shutdown.status}")
    for line in shutdown.closed:
        print(f"  closed: {line}")
    print(f"  open option positions:    {len(db.get_open_option_positions())}")
    print(f"  open long-term positions: {len(db.get_open_long_term_positions())}")
    print(f"  state: {ror.get_state()}")

    print("\n=== Kill switch refuses a wrong token ===")
    refused = ror.kill_switch("wrong-token", broker, notifier=_notifier)
    print(f"  result: {refused}")

    print("\n=== Notifications fired ===")
    for title, _message in _NOTES:
        print(f"  [{title}]")


if __name__ == "__main__":
    main()
