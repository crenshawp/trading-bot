"""Offline demo of the Phase 11 broker execution interface.

Run with: ``python demo_broker.py``

No network whatsoever: it uses the in-memory :class:`FakeBroker` and a throwaway
SQLite database. It shows:

* the account / positions read path returning neutral types;
* a LIMIT order submission (LIMIT is the default — no naive market orders);
* a structured rejection that never raises;
* the order lifecycle (an auto-filled order becomes a position);
* broker-authoritative reconciliation surfacing ``internal_only`` and
  ``broker_only`` divergences — reported, never auto-resolved. Phase 18 scope:
  only BROKER-TRACKED positions (option / long-term rows recorded on accepted
  orders) are compared; the scanner's signal-tracking trades are out of scope
  and produce no divergences;
* an unreachable broker: reconciliation reports ``ok=False`` and flags nothing.

The real ``AlpacaBroker`` speaks to Alpaca's PAPER endpoint with the same neutral
interface; only the in-memory double changes here.
"""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path

from trading_bot import config

config.DB_PATH = Path(tempfile.mkdtemp()) / "demo_broker.db"

from trading_bot import broker, db  # noqa: E402
from trading_bot.broker.fake import FakeBroker  # noqa: E402
from trading_bot.models import LongTermPosition, Signal, Trade  # noqa: E402


def _seed_signal_tracking(ticker: str) -> None:
    """Open a SIGNAL-TRACKING trade (yfinance-resolved, never sent to a broker).

    Phase 18: these are OUT of reconciliation's scope — seeded here to show
    they no longer flood the report with false divergences."""
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    sid = db.insert_signal(Signal(
        timestamp=ts, ticker=ticker, asset_class="stock",
        signal_type="ema21_pullback", direction="call", entry_price=100.0,
    ))
    db.insert_trade(Trade(
        signal_id=sid, opened_at=ts, outcome="open", track_mode="active",
    ))


def _seed_broker_tracked(ticker: str, qty: float) -> None:
    """Open a BROKER-TRACKED position (a Phase 16 long-term entry — recorded
    only after a broker-accepted order). This is what reconciliation compares."""
    db.insert_long_term_position(LongTermPosition(
        ticker=ticker, asset_class="stock", entry_price=100.0,
        entry_date=datetime(2026, 1, 1, tzinfo=UTC), qty=qty, status="open",
    ))


def main() -> None:
    db.init_db()
    b = FakeBroker(buying_power=25_000.0, cash=25_000.0, equity=25_000.0)

    print("=== Account (paper) ===")
    acct = b.get_account()
    print(f"  ok={acct.ok}  buying_power=${acct.buying_power:,.2f}  "
          f"status={acct.status}")

    print("\n=== Submit a LIMIT order (the default order type) ===")
    order = b.submit_order("AAPL", 10, broker.SIDE_BUY, limit_price=190.0)
    print(f"  ok={order.ok}  status={order.status}  id={order.order_id}  "
          f"type={order.order_type}")

    print("\n=== A structured rejection (never raises) ===")
    rej = FakeBroker(reject_reason="insufficient buying power").submit_order(
        "AAPL", 100_000, broker.SIDE_BUY, limit_price=190.0,
    )
    print(f"  ok={rej.ok}  status={rej.status}  reason={rej.reason!r}")

    print("\n=== Order lifecycle: an auto-filled order becomes a position ===")
    filled = FakeBroker(auto_fill=True)
    done = filled.submit_order("MSFT", 5, broker.SIDE_BUY, limit_price=400.0)
    print(f"  order status={done.status}  filled_qty={done.filled_qty}")
    for p in filled.get_positions().positions:
        print(f"  position: {p.symbol} qty={p.qty} side={p.side}")

    print("\n=== Reconciliation (broker-tracked scope, Phase 18) ===")
    for ticker in ("META", "GOOGL", "GS"):   # scanner signal rows — OUT of scope
        _seed_signal_tracking(ticker)
    _seed_broker_tracked("AAPL", qty=10.0)   # a real broker-routed position
    recon = FakeBroker()
    recon.set_position("TSLA", 4)            # the broker holds untracked TSLA
    report = broker.reconcile(recon)
    print(f"  internal={report.internal_symbols}  broker={report.broker_symbols}")
    print("  (3 signal-tracking trades seeded - none compared, none flagged)")
    for d in report.divergences:
        print(f"    [{d.kind}] {d.symbol}: {d.detail}")

    print("\n=== Broker unavailable: cannot reconcile, nothing flagged ===")
    down = broker.reconcile(FakeBroker(fail=True))
    print(f"  ok={down.ok}  divergences={[d.kind for d in down.divergences]}")


if __name__ == "__main__":
    main()
