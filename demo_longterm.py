"""Offline demo of the Phase 14 long-term buy-and-hold system (stocks + crypto).

Run with: ``python demo_longterm.py``

No network whatsoever: an injected candle fetch + a throwaway SQLite database +
a :class:`FakeBroker`. It shows entry-candidate generation (feeding the SAME
Phase 12 allocation plan) and the AUTO-EXECUTING protective exit watcher closing
a position on a drawdown stop. The entry gates (fundamental / technical /
earnings) are stubbed here so the flow runs offline; production wires them to
yfinance. Still PAPER ONLY — candidate generation submits nothing; only the
protective exit watcher acts (closing).
"""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from trading_bot import config

config.DB_PATH = Path(tempfile.mkdtemp()) / "demo_longterm.db"

from trading_bot import db  # noqa: E402
from trading_bot import long_term as lt  # noqa: E402
from trading_bot.broker.fake import FakeBroker  # noqa: E402
from trading_bot.models import LongTermPosition  # noqa: E402


def _flat_df(price: float, n: int = 5) -> pd.DataFrame:
    return pd.DataFrame({
        "Open": [price] * n, "High": [price] * n, "Low": [price] * n,
        "Close": [price] * n, "Volume": [1_000_000] * n,
    })


def main() -> None:
    db.init_db()

    # Stub the entry gates so the demo runs offline (production wires yfinance).
    lt.fundamental_red_flag = lambda _t: (False, "clean")          # type: ignore[assignment]
    lt.confirm_technical_entry = lambda _df, **_k: (True, "above trend, ADX strong, RSI ok")  # type: ignore[assignment]
    lt.in_earnings_wait_window = lambda _t, **_k: (False, "no earnings soon")  # type: ignore[assignment]

    print("=== Entry candidate generation (feeds the allocation plan) ===")
    prices = {"META": 480.0, "BTC-USD": 64_000.0}
    candidates = lt.generate_candidates(
        lambda t: _flat_df(prices[t]),
        stock_universe=("META",), crypto_universe=("BTC-USD",),
    )
    for c in candidates:
        print(f"  {c.ticker:<8} {c.asset_class:<6} {c.signal_type:<16} "
              f"entry=${c.entry_price:,.0f}  ({c.entry_rationale})")
    print("  crypto SWING signals (oversold_reversal / momentum_breakout) are "
          "data-only - they never appear here.")

    print("\n=== Seed open positions, then run the protective exit watcher ===")
    for ticker, entry in (("META", 480.0), ("AMZN", 200.0)):
        db.insert_long_term_position(LongTermPosition(
            ticker=ticker, asset_class="stock", entry_price=entry,
            entry_date=datetime(2026, 1, 1, tzinfo=UTC), qty=5.0, status="open",
        ))
    # META holds (small move); AMZN has fallen 30% -> drawdown stop.
    current = {"META": 470.0, "AMZN": 140.0}
    actions = lt.watch_long_term_positions(
        FakeBroker(), price_fetch=lambda t: _flat_df(current[t]),
        now=datetime(2026, 3, 1, tzinfo=UTC),
    )
    for a in actions:
        print(f"  {a.position.ticker:<6} action={a.action:<6} reason={a.reason}")

    print("\n=== Resulting open book ===")
    for p in db.get_open_long_term_positions():
        print(f"  {p.ticker:<6} still open @ ${p.entry_price:,.0f}")
    print("  (no profit-taking exit exists this phase - protective closes only)")


if __name__ == "__main__":
    main()
