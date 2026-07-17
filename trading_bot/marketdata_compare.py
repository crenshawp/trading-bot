"""yfinance-vs-Alpaca bar comparison — Phase 25. DIAGNOSTIC OUTPUT ONLY.

Fetches matching-window daily bars for the bot's universe from BOTH sources
and reports where they agree or diverge: close prices (tolerance-based, never
exact-float), bar-date alignment, and whether the two sources disagree about
the most recent CLOSED bar. No consumer's behavior changes based on this
report — yfinance remains authoritative everywhere until a separate,
deliberate cutover phase gated on this report being clean.

Sources are injected (callables / client) so tests and the demo run fully
offline; the CLI wires the real yfinance + free-tier Alpaca client.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from trading_bot import config, db
from trading_bot.alpaca_market_data import (
    AlpacaMarketDataClient,
    MarketBar,
    bar_is_closed,
    latest_closed_bar,
)
from trading_bot.discovery_universe import SHADOW_UNIVERSE

_ET = ZoneInfo("America/New_York")

STATUS_MATCH = "match"
STATUS_DIVERGENT = "divergent"
STATUS_MISSING_ALPACA = "missing_alpaca"
STATUS_MISSING_YFINANCE = "missing_yfinance"

# yfinance daily fetch per ticker — injected into the comparison so tests and
# the demo never touch the network.
YfFetch = Callable[[str, int], "pd.DataFrame | None"]


@dataclass(frozen=True)
class TickerComparison:
    """One ticker's agreement/discrepancy summary."""

    ticker: str
    status: str                      # match | divergent | missing_*
    bars_compared: int = 0
    max_close_diff_pct: float | None = None
    mean_close_diff_pct: float | None = None
    latest_closed_agrees: bool | None = None
    latest_closed_yf: str | None = None       # ISO date of yf's latest closed bar
    latest_closed_alpaca: str | None = None   # ISO date of Alpaca's
    note: str = ""


@dataclass(frozen=True)
class ComparisonReport:
    """The full diagnostic run: per-ticker results + summary counts."""

    results: list[TickerComparison] = field(default_factory=list)
    tolerance_pct: float = config.MD_COMPARE_TOLERANCE_PCT

    @property
    def matched(self) -> int:
        return sum(1 for r in self.results if r.status == STATUS_MATCH)

    @property
    def divergent(self) -> int:
        return sum(1 for r in self.results if r.status == STATUS_DIVERGENT)

    @property
    def missing(self) -> int:
        return sum(1 for r in self.results if r.status.startswith("missing"))


def default_universe() -> list[str]:
    """The bot's full comparison universe: active watchlist ∪ shadow universe
    + the crypto pairs, deduped preserving order. Fail-soft to the shadow
    universe alone if the watchlist table is unreadable."""
    tickers: list[str] = []
    try:
        tickers.extend(db.get_active_watchlist())
    except Exception as exc:  # noqa: BLE001 - a missing table must not sink the report
        print(f"  compare: watchlist unavailable ({exc})", file=sys.stderr)
    tickers.extend(SHADOW_UNIVERSE)
    tickers.extend(config.LONGTERM_CRYPTO_UNIVERSE)
    seen: set[str] = set()
    unique: list[str] = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


def fetch_yf_daily(ticker: str, window_days: int) -> pd.DataFrame | None:
    """Daily yfinance candles for the comparison window. Fail-soft → None."""
    try:
        df = yf.download(
            ticker, period=f"{window_days}d", interval="1d", progress=False,
            auto_adjust=False,
        )
    except Exception as exc:  # noqa: BLE001 - a fetch must never sink the report
        print(f"  compare: yfinance error for {ticker} ({exc})", file=sys.stderr)
        return None
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    return df


def _yf_closes_by_date(df: pd.DataFrame) -> dict[date, float]:
    """yfinance daily rows → {ET calendar date: close}."""
    closes: dict[date, float] = {}
    for raw_ts, row in df.iterrows():
        ts = raw_ts.to_pydatetime() if hasattr(raw_ts, "to_pydatetime") else raw_ts
        bar_date = ts.date() if isinstance(ts, datetime) else ts
        try:
            closes[bar_date] = float(row["Close"])
        except (KeyError, TypeError, ValueError):
            continue
    return closes


def _alpaca_closes_by_date(bars: list[MarketBar]) -> dict[date, float]:
    """Alpaca daily bars → {ET calendar date of the bar START: close}."""
    return {b.start.astimezone(_ET).date(): b.close for b in bars}


def _yf_latest_closed_date(
    closes: dict[date, float], now: datetime,
) -> date | None:
    """yfinance's most recent CLOSED daily bar: the newest row dated BEFORE
    today (ET) — the same conservatism the scanner's iloc[-2] applies (today's
    row may be the still-forming session)."""
    today_et = now.astimezone(_ET).date()
    closed = [d for d in closes if d < today_et]
    return max(closed) if closed else None


def compare_ticker(
    ticker: str,
    yf_df: pd.DataFrame | None,
    alpaca_bars: list[MarketBar],
    *,
    now: datetime,
    tolerance_pct: float = config.MD_COMPARE_TOLERANCE_PCT,
    window_bars: int = config.MD_COMPARE_WINDOW_BARS,
) -> TickerComparison:
    """Compare one ticker's daily bars from both sources. Pure — no I/O."""
    yf_closes = _yf_closes_by_date(yf_df) if yf_df is not None else {}
    alpaca_closes = _alpaca_closes_by_date(alpaca_bars)

    if not yf_closes and not alpaca_closes:
        return TickerComparison(
            ticker, STATUS_MISSING_ALPACA, note="no data from either source",
        )
    if not alpaca_closes:
        return TickerComparison(
            ticker, STATUS_MISSING_ALPACA, note="no Alpaca bars",
        )
    if not yf_closes:
        return TickerComparison(
            ticker, STATUS_MISSING_YFINANCE, note="no yfinance bars",
        )

    # Most-recent-CLOSED-bar agreement, each source judged by its own rules.
    yf_latest = _yf_latest_closed_date(yf_closes, now)
    alpaca_latest_bar = latest_closed_bar(alpaca_bars, "1Day", now)
    alpaca_latest = (
        alpaca_latest_bar.start.astimezone(_ET).date()
        if alpaca_latest_bar is not None else None
    )
    latest_agrees = (
        yf_latest == alpaca_latest
        if yf_latest is not None and alpaca_latest is not None else None
    )

    # Close-price agreement uses CLOSED rows only. In particular, today's
    # partial rows must not affect either tolerance metrics or bar counts.
    today_et = now.astimezone(_ET).date()
    yf_closed = {d: close for d, close in yf_closes.items() if d < today_et}
    alpaca_closed = _alpaca_closes_by_date([
        bar for bar in alpaca_bars if bar_is_closed(bar, "1Day", now)
    ])
    shared_dates = sorted(set(yf_closed) & set(alpaca_closed))
    shared_dates = shared_dates[-max(1, window_bars):]
    diffs_pct: list[float] = []
    for d in shared_dates:
        yf_close = yf_closed[d]
        if yf_close == 0.0:
            continue
        diffs_pct.append(abs(alpaca_closed[d] - yf_close) / yf_close * 100.0)

    if not shared_dates:
        return TickerComparison(
            ticker, STATUS_DIVERGENT, bars_compared=0,
            latest_closed_agrees=latest_agrees,
            latest_closed_yf=yf_latest.isoformat() if yf_latest else None,
            latest_closed_alpaca=alpaca_latest.isoformat() if alpaca_latest else None,
            note="no overlapping closed bar dates",
        )

    max_diff = max(diffs_pct) if diffs_pct else 0.0
    mean_diff = sum(diffs_pct) / len(diffs_pct) if diffs_pct else 0.0
    diverged = max_diff > tolerance_pct or latest_agrees is False
    note = ""
    if max_diff > tolerance_pct:
        note = f"close diff {max_diff:.3f}% exceeds {tolerance_pct}% tolerance"
    elif latest_agrees is False:
        note = "sources disagree on the most recent closed bar"

    return TickerComparison(
        ticker,
        STATUS_DIVERGENT if diverged else STATUS_MATCH,
        bars_compared=len(shared_dates),
        max_close_diff_pct=max_diff,
        mean_close_diff_pct=mean_diff,
        latest_closed_agrees=latest_agrees,
        latest_closed_yf=yf_latest.isoformat() if yf_latest else None,
        latest_closed_alpaca=alpaca_latest.isoformat() if alpaca_latest else None,
        note=note,
    )


def compare_universe(
    tickers: list[str],
    *,
    window_bars: int = config.MD_COMPARE_WINDOW_BARS,
    now: datetime | None = None,
    tolerance_pct: float = config.MD_COMPARE_TOLERANCE_PCT,
    client: AlpacaMarketDataClient | None = None,
    yf_fetch: YfFetch = fetch_yf_daily,
) -> ComparisonReport:
    """Run the comparison across ``tickers``. DIAGNOSTIC ONLY — changes nothing.

    Alpaca fetches are batched (stocks multi-symbol, crypto separately) and
    throttled by the client, respecting the free tier's 200 rpm cap. yfinance
    is fetched per ticker, fail-soft. ``window_bars`` daily bars are compared;
    the fetch window is padded for weekends/holidays.
    """
    moment = now if now is not None else datetime.now(UTC)
    md = client if client is not None else AlpacaMarketDataClient()
    window_days = window_bars * 2 + 5              # pad for non-trading days
    start = moment - timedelta(days=window_days)

    stocks = [t for t in tickers if not t.endswith("-USD")]
    crypto = [t for t in tickers if t.endswith("-USD")]

    alpaca_bars: dict[str, list[MarketBar]] = {}
    if stocks:
        stock_result = md.get_stock_bars(stocks, timeframe="1Day", start=start)
        if not stock_result.ok:
            print(
                f"  compare: Alpaca stock bars unavailable ({stock_result.reason})",
                file=sys.stderr,
            )
        alpaca_bars.update(stock_result.bars)
    if crypto:
        crypto_result = md.get_crypto_bars(crypto, timeframe="1Day", start=start)
        if not crypto_result.ok:
            print(
                f"  compare: Alpaca crypto bars unavailable ({crypto_result.reason})",
                file=sys.stderr,
            )
        alpaca_bars.update(crypto_result.bars)

    results: list[TickerComparison] = []
    for ticker in tickers:
        try:
            yf_df = yf_fetch(ticker, window_days)
            results.append(compare_ticker(
                ticker, yf_df, alpaca_bars.get(ticker, []),
                now=moment, tolerance_pct=tolerance_pct,
                window_bars=window_bars,
            ))
        except Exception as exc:  # noqa: BLE001 - one ticker must never sink the report
            print(f"  compare: error on {ticker} ({exc})", file=sys.stderr)
            results.append(TickerComparison(
                ticker, STATUS_MISSING_YFINANCE, note=f"error: {exc}",
            ))
    return ComparisonReport(results=results, tolerance_pct=tolerance_pct)
