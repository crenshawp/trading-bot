"""Offline demo of the Phase 25 yfinance-vs-Alpaca comparison (diagnostic only).

Run with: ``python demo_marketdata_compare.py``

No network whatsoever: both "sources" are fabricated fixtures. It shows a
clean within-tolerance match and a flagged discrepancy side by side, plus the
bar-closure rule ignoring a still-forming daily bar on both sides. The real
path is ``python -m trading_bot marketdata compare`` against live yfinance +
the free-tier (IEX) Alpaca data API — which changes NOTHING live: yfinance
remains authoritative for every consumer until a deliberate cutover phase.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from trading_bot import marketdata_compare as mc
from trading_bot.alpaca_market_data import (
    AlpacaMarketDataClient,
    BarsResult,
    MarketBar,
)

_NOW = datetime(2026, 7, 15, 23, 0, tzinfo=UTC)   # 19:00 ET — today still forming


def _yf_df(closes: dict[str, float]) -> pd.DataFrame:
    index = pd.DatetimeIndex([pd.Timestamp(d) for d in closes])
    values = list(closes.values())
    return pd.DataFrame({
        "Open": values, "High": values, "Low": values, "Close": values,
        "Volume": [1_000] * len(values),
    }, index=index)


def _bar(day: str, close: float, symbol: str) -> MarketBar:
    start = datetime.fromisoformat(f"{day}T04:00:00+00:00")
    return MarketBar(symbol=symbol, start=start, open=close, high=close,
                     low=close, close=close, volume=1_000.0)


class _FixtureClient(AlpacaMarketDataClient):
    """Serves fabricated Alpaca bars — the same shapes the real client returns."""

    def __init__(self) -> None:
        self._stock = {
            # Clean match: IEX closes a few basis points off Yahoo's.
            "META": [
                _bar("2026-07-13", 480.10, "META"),
                _bar("2026-07-14", 484.95, "META"),
                _bar("2026-07-15", 999.0, "META"),      # today: still forming
            ],
            # Flagged: the 14th diverges well beyond the 0.5% tolerance.
            "GS": [
                _bar("2026-07-13", 700.00, "GS"),
                _bar("2026-07-14", 712.60, "GS"),       # ~1.2% off Yahoo
            ],
        }

    def get_stock_bars(self, symbols: list[str], **_kw: object) -> BarsResult:
        return BarsResult(ok=True, bars={
            s: self._stock.get(s, []) for s in symbols if s in self._stock
        })

    def get_crypto_bars(self, symbols: list[str], **_kw: object) -> BarsResult:
        return BarsResult(ok=True, bars={})


def main() -> None:
    frames = {
        "META": _yf_df({
            "2026-07-13": 480.00, "2026-07-14": 485.00,
            "2026-07-15": 998.0,                        # today's partial row
        }),
        "GS": _yf_df({"2026-07-13": 700.05, "2026-07-14": 704.00}),
    }

    report = mc.compare_universe(
        ["META", "GS"], now=_NOW,
        client=_FixtureClient(),
        yf_fetch=lambda t, _w: frames.get(t),
    )

    print("=== yfinance vs Alpaca (IEX) — clean match and flagged divergence ===")
    for r in report.results:
        print(
            f"  {r.ticker:<6} {r.status:<10} bars={r.bars_compared} "
            f"max_diff={r.max_close_diff_pct:.3f}% "
            f"latest_closed yf={r.latest_closed_yf} "
            f"alpaca={r.latest_closed_alpaca}"
            + (f"  <- {r.note}" if r.note else "")
        )
    print(f"\n  Summary: {report.matched} matched, {report.divergent} divergent "
          f"(tolerance {report.tolerance_pct}%)")
    print("  Note: both sources included TODAY's still-forming bar; each "
          "side's closure rule ignored it (latest closed = the 14th).")
    print("  Diagnostic only - yfinance stays authoritative for every consumer.")


if __name__ == "__main__":
    main()
