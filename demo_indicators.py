"""Offline demo of the Phase 6 advanced indicator families.

Run with: ``python demo_indicators.py``

Everything external is mocked (yfinance, regime/VIX, earnings, news/LLM) so it
runs with no network. It shows:

* the five INDEPENDENT families computed on a sample candle frame
  (volatility/ATR + regime, momentum/RSI, trend-strength/ADX, volume/OBV),
* the cross-asset correlation family classifying an active set as concentrated
  vs diversified,
* a fired stock signal getting that indicator context attached AFTER the
  alert-gating checks and persisted alongside the trade, and
* the persisted context via the same query the ``indicators status`` CLI uses.

The families are ADVISORY only — they never create or block a signal this phase.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from trading_bot import config

# The fired-signal alert text contains emoji; make stdout UTF-8 so the DRY-RUN
# print doesn't choke on a Windows cp1252 console.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

config.DB_PATH = Path(tempfile.mkdtemp()) / "demo_indicators.db"

from trading_bot import db, indicators, regime, scanner, vix  # noqa: E402


def _frame(closes: list[float]) -> pd.DataFrame:
    """A minimal OHLCV frame in the scanner's candle structure."""
    return pd.DataFrame(
        {
            "Open": closes,
            "High": [c + 2.0 for c in closes],
            "Low": [c - 2.0 for c in closes],
            "Close": closes,
            "Volume": [1_000_000 + i * 1000 for i in range(len(closes))],
        },
        index=pd.date_range("2026-01-01", periods=len(closes), freq="D"),
    )


def _signal() -> dict[str, object]:
    return {
        "ticker": "GOOGL", "asset_type": "stock", "trade_type": "SWING TRADE",
        "direction": "CALL", "setup": "EMA21 Pullback",
        "detail": "Pullback to EMA21 in uptrend.", "price": 180.0,
        "take_profit": 187.0, "stop_loss": 177.0, "hold_days": "3-5 days",
        "confidence": "High",
    }


def main() -> None:
    db.init_db()
    scanner.DRY_RUN = True  # print the alert instead of posting

    rising = [100.0 + i + (2.0 if i % 2 else 0.0) for i in range(40)]

    print("=== The five families on a sample candle frame ===")
    ctx = indicators.compute_context(_frame(rising), at=-1)
    print(f"  Volatility : ATR={ctx.atr:.2f}  realized_vol={ctx.realized_vol:.4f}  "
          f"regime={ctx.vol_regime}")
    print(f"  Momentum   : RSI={ctx.rsi:.1f}")
    print(f"  Trend      : ADX={ctx.adx:.1f}  (strength, direction-agnostic)")
    print(f"  Volume     : OBV={ctx.obv:,.0f}")

    print("\n=== Correlation family: concentrated vs diversified ===")
    together = {  # three names that move as one -> hidden single-bet
        "GOOGL": _frame(rising),
        "MSFT": _frame([c * 1.01 for c in rising]),
        "AMZN": _frame([c * 0.99 for c in rising]),
    }
    indicators.clear_correlation_cache()
    avg, label = indicators.correlation_concentration(
        "GOOGL", list(together), lambda t: together[t]["Close"],
    )
    print(f"  move-together set : avg_corr={avg:+.2f} -> {label}")

    swing = [100.0 + (10.0 if i % 2 else 0.0) for i in range(40)]   # 100,110,100,110
    mirror = [100.0 + (0.0 if i % 2 else 10.0) for i in range(40)]  # 110,100,110,100
    spread = {"GOOGL": _frame(swing), "HEDGE": _frame(mirror)}      # exact mirror
    indicators.clear_correlation_cache()
    avg2, label2 = indicators.correlation_concentration(
        "GOOGL", list(spread), lambda t: spread[t]["Close"],
    )
    print(f"  diversified set   : avg_corr={avg2:+.2f} -> {label2}")

    print("\n=== Fired signal -> indicator context attached + persisted ===")
    active = {  # the live active set the correlation universe is drawn from
        "GOOGL": _frame(rising),
        "MSFT": _frame([c * 1.01 for c in rising]),
        "AMZN": _frame([c * 0.99 for c in rising]),
    }
    indicators.clear_correlation_cache()
    with (
        patch.object(scanner, "_active_stock_watchlist_entries",
                     return_value=[(t, "active") for t in active]),
        patch.object(scanner, "get_stock_data", side_effect=lambda t: active.get(t)),
        patch.object(scanner, "_score_signal_sentiment", return_value=None),
        patch("trading_bot.earnings.is_in_blackout",
              return_value=(False, "no earnings in window")),
        patch("trading_bot.regime.get_current_regime",
              side_effect=regime.RegimeFetchError("offline")),
        patch("trading_bot.vix.get_current_vix",
              side_effect=vix.VixFetchError("offline")),
    ):
        scanner._emit_active_signal(_signal(), active["GOOGL"])

    print("\n=== Persisted context (what `indicators status` shows) ===")
    for row in db.get_recent_trade_indicators():
        print(f"  {row['ticker']} {row['signal_type']}  "
              f"vol={row['ind_vol_regime']}  RSI={row['ind_rsi']:.1f}  "
              f"ADX={row['ind_adx']:.1f}  OBV={row['ind_obv']:,.0f}  "
              f"corr={row['ind_correlation']:+.2f} ({row['ind_concentration']})")


if __name__ == "__main__":
    main()
