"""Offline demo of Phase 17 live signal wiring (signal → plan → paper order).

Run with: ``python demo_live_candidates.py``

No network whatsoever: hand-seeded fired signals in a throwaway SQLite database
+ a :class:`FakeBroker`. It shows the live candidate source pulling REAL fired
signals (not the sample fixture) into a plan — including a crypto SWING signal
that is sourced but dropped by build_plan's data-only filter — then an executed
long-term crypto entry whose ticker translates to Alpaca's wire format at the
broker boundary. The real path is ``python -m trading_bot allocate plan`` /
``allocate execute --confirm`` against Alpaca PAPER.
"""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path

from trading_bot import config

config.DB_PATH = Path(tempfile.mkdtemp()) / "demo_live_candidates.db"

from trading_bot import allocation, candidate_source, db, long_term  # noqa: E402
from trading_bot import plan_execution as pe  # noqa: E402
from trading_bot.broker.alpaca import to_alpaca_symbol  # noqa: E402
from trading_bot.broker.fake import FakeBroker  # noqa: E402
from trading_bot.models import LongTermCandidate, Signal, Trade  # noqa: E402

_NOW = datetime(2026, 7, 10, 15, 0, tzinfo=UTC)


def _seed_fired_signals() -> None:
    """Seed what a real scan cycle leaves behind: a swing stock signal (with
    its fire-time trade context), a data-only crypto SWING signal, and a
    long-term crypto entry persisted by the Phase 17 candidate logger."""
    swing = db.insert_signal(Signal(
        timestamp=_NOW, ticker="META", asset_class="stock",
        signal_type="ema21_pullback", direction="call", entry_price=480.0,
        atr=8.0, rsi=55.0,
    ))
    db.insert_trade(Trade(
        signal_id=swing, opened_at=_NOW, outcome="open", track_mode="active",
        ind_rsi=52.0, ind_adx=27.0, ind_obv=1_200_000.0, ind_vol_regime="low",
        ind_concentration="diversified", sentiment_score=0.4,
    ))

    crypto_swing = db.insert_signal(Signal(
        timestamp=_NOW, ticker="BTC-USD", asset_class="crypto",
        signal_type="momentum_breakout", direction="long", entry_price=64_000.0,
        atr=1_500.0,
    ))
    db.insert_trade(Trade(
        signal_id=crypto_swing, opened_at=_NOW, outcome="open",
        track_mode="active",
    ))

    long_term.persist_candidates([LongTermCandidate(
        ticker="BTC-USD", asset_class="crypto",
        signal_type=config.LONGTERM_CRYPTO_SIGNAL, entry_price=64_000.0,
        entry_rationale="crypto long-term: price > trend, ADX ok, RSI ok",
    )], now=_NOW)


def main() -> None:
    db.init_db()
    _seed_fired_signals()
    broker = FakeBroker()

    print("=== Live candidate source: real fired signals, not samples ===")
    candidates = candidate_source.live_candidates(now=_NOW, mark_considered=True)
    for c in candidates:
        print(f"  {c.ticker:<8} {c.signal_type:<18} {c.asset_class:<7} "
              f"entry={c.entry:,.0f}")

    print("\n=== build_plan: the EXISTING data-only filter still drops "
          "crypto swing ===")
    result = allocation.build_plan(candidates, broker.get_account())
    for o in result.plan.orders:
        print(f"  planned: {o.ticker:<8} {o.pool:<10} qty={o.qty:.4f} "
              f"est=${o.est_cost:,.0f}")
    for s in result.skipped:
        print(f"  skipped: {s.ticker:<8} {s.signal_type:<18} [{s.stage}] "
              f"{s.reason}")

    print("\n=== execute: the long-term crypto entry submits (paper) ===")
    run = pe.execute_plan(broker, result.plan, now=_NOW)
    for e in run.executions:
        wire = to_alpaca_symbol(e.order.ticker)
        print(f"  {e.order.ticker:<8} {e.order.pool:<10} -> {e.status:<10} "
              f"ref={e.order_ref or '-':<8} wire-symbol={wire}")
    print(f"  open long-term positions: {len(db.get_open_long_term_positions())}")

    print("\n=== considered-tracking: the same signals are never re-planned ===")
    again = candidate_source.live_candidates(now=_NOW)
    print(f"  candidates on a second pull: {len(again)} (all consumed at pull)")

    print("\n=== symbol translation at the Alpaca boundary ===")
    for symbol in ("BTC-USD", "ETH-USD", "AAPL", "AAPL260116C00190000"):
        print(f"  {symbol:<22} -> {to_alpaca_symbol(symbol)}")


if __name__ == "__main__":
    main()
