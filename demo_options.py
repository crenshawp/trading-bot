"""Offline demo of the Phase 13 options execution layer (single-leg, PAPER).

Run with: ``python demo_options.py``

No network whatsoever: hand-built contract fixtures + a :class:`FakeBroker` + a
throwaway SQLite database. It shows the full->undersized->shares execution
hierarchy, submitting + recording an option position, and the manual exit
watcher closing a position when the underlying hits take-profit. The real path
uses ``AlpacaOptionsClient`` for the chain and ``AlpacaBroker`` for orders
against the PAPER endpoint — only the data sources change here.
"""

from __future__ import annotations

import tempfile
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from trading_bot import config

config.DB_PATH = Path(tempfile.mkdtemp()) / "demo_options.db"

from trading_bot import db  # noqa: E402
from trading_bot import options_execution as oe  # noqa: E402
from trading_bot.broker.fake import FakeBroker  # noqa: E402
from trading_bot.broker.options import OptionContract  # noqa: E402

_REF = date(2026, 1, 1)


def _c(
    delta: float, *, option_type: str = "call", oi: int = 500,
    bid: float = 4.9, ask: float = 5.1, dte: int = 30, strike: float = 150.0,
) -> OptionContract:
    expiry = (_REF + timedelta(days=dte)).isoformat()
    return OptionContract(
        symbol=f"AAPL-{option_type}-{strike:g}", underlying="AAPL",
        option_type=option_type, strike=strike, expiry=expiry, delta=delta,
        theta=-0.05, vega=0.1, gamma=0.01, open_interest=oi, bid=bid, ask=ask,
        mid=(bid + ask) / 2.0,
    )


def main() -> None:
    db.init_db()
    broker = FakeBroker()
    chain = [_c(0.70), _c(0.55, strike=155.0)]

    print("=== Execution hierarchy (full -> undersized -> shares) ===")
    scenarios = [
        ("full-band option fits", 600.0, chain, True),
        ("only undersized fits", 600.0, [_c(0.55)], True),
        ("nothing fits -> shares", 300.0, [_c(0.70)], True),
        ("options unavailable -> shares", 600.0, chain, False),
    ]
    for label, capital, contracts, avail in scenarios:
        dec = oe.choose_execution(
            "call", capital, "AAPL", 100.0, contracts, ref_date=_REF,
            options_available=avail,
        )
        print(
            f"  {label:<32} -> vehicle={dec.vehicle:<18} qty={dec.qty:>8.4f}  "
            f"est=${dec.est_cost:>7,.0f}  {dec.symbol}"
        )

    print("\n=== Submit + record an option position (no bracket order) ===")
    dec = oe.choose_execution("call", 600.0, "AAPL", 100.0, chain, ref_date=_REF)
    order, pid = oe.execute_decision(
        broker, dec, opened_at=datetime(2026, 1, 1, tzinfo=UTC),
        tp=110.0, sl=95.0, deadline=datetime(2026, 1, 20, tzinfo=UTC),
    )
    print(f"  order ok={order.ok} status={order.status} position_id={pid}")
    print(f"  open option positions: {len(db.get_open_option_positions())}")

    print("\n=== Exit watcher: underlying hits take-profit -> close ===")
    actions = oe.watch_open_option_positions(
        broker, underlying_price_fetch=lambda _u: 111.0,
        option_price_fetch=lambda _s: 8.0,
        now=datetime(2026, 1, 5, tzinfo=UTC),
    )
    for a in actions:
        print(
            f"  {a.position.symbol}: action={a.action} reason={a.reason} "
            f"pnl=${(a.pnl_dollars or 0.0):,.0f}"
        )
    print(f"  open positions remaining: {len(db.get_open_option_positions())}")


if __name__ == "__main__":
    main()
