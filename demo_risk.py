"""Offline demo of the Phase 7 advisory risk framework.

Run with: ``python demo_risk.py``

Everything external is mocked (yfinance, regime/VIX, earnings, news/LLM) so it
runs with no network. It shows:

* volatility-normalized sizing — a wider-ATR (noisier) name sized smaller for the
  same dollar risk, and the per-position cap binding;
* portfolio-level verdicts — an ok book vs one whose total risk would exceed the
  limit, and the correlated-cluster check;
* a fired signal getting a size + portfolio verdicts attached AFTER the alert
  gates and persisted alongside the trade (with a pre-existing open position in
  the book), surfaced via the same query the ``risk status`` CLI uses; and
* the open-book exposure the ``risk exposure`` CLI reports.

ADVISORY ONLY — nothing here sizes, blocks, or shrinks a real position; there is
no capital. Every number is a recommendation recorded next to the outcome.
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

config.DB_PATH = Path(tempfile.mkdtemp()) / "demo_risk.db"

from trading_bot import db, regime, risk, scanner, vix  # noqa: E402
from trading_bot.__main__ import cmd_risk_exposure  # noqa: E402
from trading_bot.models import Signal, Trade  # noqa: E402


def _frame(closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": closes,
            "High": [c + 2.0 for c in closes],
            "Low": [c - 2.0 for c in closes],
            "Close": closes,
            "Volume": [1_000_000] * len(closes),
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

    print("=== Volatility-normalized sizing ($10k notional, 1% risk) ===")
    calm = risk.position_size(entry=180.0, atr=8.0, account=10_000.0,
                              risk_per_trade_pct=1.0)
    noisy = risk.position_size(entry=180.0, atr=16.0, account=10_000.0,
                               risk_per_trade_pct=1.0)
    print(f"  ATR  8 -> ~{calm.recommended_size:.2f} units ({calm.risk_pct:.2f}% risk)")
    print(f"  ATR 16 -> ~{noisy.recommended_size:.2f} units ({noisy.risk_pct:.2f}% risk)")
    print("  -> double the ATR, half the size (same dollar risk).")
    tight = risk.position_size(entry=180.0, atr=2.0, account=10_000.0,
                               risk_per_trade_pct=1.0)
    print(f"  ATR  2 -> ~{tight.recommended_size:.2f} units, capped at the "
          f"{config.MAX_POSITION_PCT:.0f}% per-position limit "
          f"({tight.risk_pct:.2f}% risk vs 1% requested)")

    print("\n=== Portfolio verdicts vs the open book ===")
    candidate = risk.position_size(entry=180.0, atr=8.0, account=10_000.0,
                                   risk_per_trade_pct=1.0)
    light = risk.portfolio_risk(candidate, "diversified",
                                [risk.OpenPosition(1.0), risk.OpenPosition(1.5)])
    print(f"  light book  -> total {light.total_risk_pct:.2f}%  "
          f"portfolio={light.portfolio_verdict}")
    heavy = risk.portfolio_risk(
        candidate, "concentrated",
        [risk.OpenPosition(3.0, "concentrated"), risk.OpenPosition(2.5, "concentrated")],
    )
    print(f"  heavy book  -> total {heavy.total_risk_pct:.2f}%  "
          f"portfolio={heavy.portfolio_verdict}  cluster={heavy.cluster_verdict}")

    print("\n=== Fired signal -> risk attached + persisted ===")
    # A pre-existing open ACTIVE position already carries 2.5% recorded risk.
    seed_sid = db.insert_signal(Signal(
        timestamp=pd.Timestamp("2026-05-01", tz="UTC").to_pydatetime(),
        ticker="MSFT", asset_class="stock", signal_type="ema21_pullback",
        direction="call", entry_price=400.0,
    ))
    db.insert_trade(Trade(
        signal_id=seed_sid, opened_at=pd.Timestamp("2026-05-01", tz="UTC").to_pydatetime(),
        outcome="open", track_mode="active", risk_pct=2.5,
        ind_concentration="concentrated",
    ))

    rising = [100.0 + i + (2.0 if i % 2 else 0.0) for i in range(40)]
    active = {  # the correlated active set the candidate is drawn from
        "GOOGL": _frame(rising),
        "MSFT": _frame([c * 1.01 for c in rising]),
        "AMZN": _frame([c * 0.99 for c in rising]),
    }
    scanner.indicators.clear_correlation_cache()
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

    print("\n=== Persisted recommendation (what `risk status` shows) ===")
    for row in db.get_recent_trade_risk():
        size = row["risk_recommended_size"]
        size_txt = f"{size:.2f}" if isinstance(size, (int, float)) else "n/a"
        print(f"  {row['ticker']} {row['signal_type']}  size={size_txt}  "
              f"risk={row['risk_pct']:.2f}%  total={row['risk_total_pct']:.2f}%  "
              f"portfolio={row['risk_portfolio_verdict']}  "
              f"cluster={row['risk_cluster_verdict']}")

    print("\n=== Open-book exposure (what `risk exposure` shows) ===")
    cmd_risk_exposure()


if __name__ == "__main__":
    main()
