"""Offline demo of the Phase 5 news / sentiment / earnings-blackout flow.

Run with: ``python demo_sentiment.py``

Everything external is mocked (NewsAPI, the LLM, yfinance earnings, regime/VIX)
so it runs with no network. It shows:

* a fired stock signal enriched with advisory sentiment (and a heavy-news flag),
  persisted alongside the trade,
* the SAME signal SUPPRESSED when earnings fall inside the hold window (no alert,
  no trade — the one hard gate),
* and the persisted sentiment context via the same query the CLI uses.
"""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from trading_bot import config

config.DB_PATH = Path(tempfile.mkdtemp()) / "demo_sentiment.db"

from trading_bot import db, regime, scanner, vix  # noqa: E402
from trading_bot.news_client import NewsResult  # noqa: E402


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
    scanner.DRY_RUN = True  # print alerts instead of posting

    headlines = NewsResult(
        "GOOGL",
        ["GOOGL beats on cloud growth", "Analysts raise GOOGL target"] * 5,
        ok=True,
    )
    llm_reply = '{"score": 0.62, "rationale": "Strong cloud results and upgrades."}'

    # Mock every external: news, LLM, earnings, and regime/VIX tagging.
    with (
        patch("trading_bot.news_client.fetch_headlines", return_value=headlines),
        patch("trading_bot.sentiment._call_llm", return_value=llm_reply),
        patch("trading_bot.regime.get_current_regime",
              side_effect=regime.RegimeFetchError("offline")),
        patch("trading_bot.vix.get_current_vix",
              side_effect=vix.VixFetchError("offline")),
    ):
        print("=== Case 1: no earnings in window -> enriched alert + trade ===")
        with patch("trading_bot.earnings.is_in_blackout",
                   return_value=(False, "no earnings in window")):
            scanner._emit_active_signal(_signal())

        print("\n=== Case 2: earnings inside hold window -> SUPPRESSED (hard gate) ===")
        soon = (datetime.now(UTC) + timedelta(days=2)).isoformat()
        with patch("trading_bot.earnings.is_in_blackout",
                   return_value=(True, f"earnings {soon} within 5d hold window")):
            scanner._emit_active_signal(_signal())
            print("  (suppressed — see the earnings-blackout log line above)")

    print("\n=== Persisted sentiment (what `sentiment status` shows) ===")
    for row in db.get_recent_trade_sentiment():
        print(f"  {row['ticker']} {row['signal_type']}  "
              f"score={row['sentiment_score']:+.2f} ({row['sentiment_label']})  "
              f"heavy_news={row['heavy_news']} headlines={row['headline_count']}")


if __name__ == "__main__":
    main()
