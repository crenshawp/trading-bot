"""Phase 3.1 discovery engine — backtest runner.

Discovery scans the effective universe (:mod:`trading_bot.discovery_universe`),
runs the EXISTING EMA21 Pullback backtest against each ticker, and (in later
sections) scores + auto-promotes qualifiers into the DB-driven watchlist.

This module deliberately reuses the legacy :func:`backtest.run_backtest`
verbatim: same EMA21 Pullback CALL/PUT thresholds the live scanner is
validated against, and the same ``get_historical_data`` 3-year yfinance fetch
with MultiIndex flattening. ``backtest.py`` is exempt from strict mypy, so the
untyped function is wrapped in a typed ``cast`` adapter to keep this (strict)
module clean.

NO SILENT SKIPS: every fetch failure, empty frame, or backtest exception is
caught, logged with ticker + reason, and recorded as a failure so the run
summary is honest.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import cast

import pandas as pd

import backtest
from trading_bot.discovery_universe import effective_universe

# Small sleep between yfinance-backed backtests to avoid rate limiting.
_THROTTLE_SECONDS = 1.0

# backtest.run_backtest is untyped (backtest.py is exempt from strict mypy).
# Cast to a typed callable so this strict module can call it without a
# no-untyped-call error. Behaviour is unchanged — we reuse it verbatim.
_run_ema21_backtest = cast(
    "Callable[[str, str], pd.DataFrame | None]",
    backtest.run_backtest,
)


@dataclass(frozen=True)
class DiscoveryFailure:
    """A ticker that could not be scored.

    Recorded (never silently skipped) for the run summary: no data, an empty
    backtest result, or an exception thrown by the backtest.
    """

    ticker: str
    reason: str


@dataclass
class BacktestRunResult:
    """Raw output of one discovery sweep, before scoring.

    ``successes`` maps ticker -> the backtest result DataFrame (columns
    ``direction, price, take_profit, stop_loss, outcome``). ``failures``
    lists every ticker that could not be backtested, with its reason.
    """

    successes: dict[str, pd.DataFrame]
    failures: list[DiscoveryFailure]

    @property
    def scanned(self) -> int:
        return len(self.successes) + len(self.failures)

    @property
    def succeeded(self) -> int:
        return len(self.successes)

    @property
    def failed(self) -> int:
        return len(self.failures)


def _backtest_ticker(ticker: str) -> tuple[pd.DataFrame | None, str | None]:
    """Run the EMA21 Pullback backtest for one ticker.

    Returns ``(dataframe, None)`` on success or ``(None, reason)`` on any
    failure — fetch error, no data, empty frame, or an exception thrown by the
    backtest. Never raises; the caller records the reason.
    """
    try:
        df = _run_ema21_backtest(ticker, "stock")
    except Exception as exc:  # noqa: BLE001 - any failure becomes a recorded reason
        return None, f"backtest error: {exc}"
    if df is None:
        return None, "no data (fetch returned None or empty frame)"
    if df.empty:
        return None, "no signals fired in backtest window"
    return df, None


def run_backtests(
    *,
    tickers: Sequence[str] | None = None,
    throttle_seconds: float = _THROTTLE_SECONDS,
) -> BacktestRunResult:
    """Backtest every ticker in the effective universe (or an explicit list).

    Catches and LOGS every fetch failure, empty frame, or backtest exception
    with the ticker + reason and records it as a failure — no silent skips.
    Throttles between calls to avoid yfinance rate limiting (skipped before
    the first call and when ``throttle_seconds <= 0``).
    """
    universe = list(tickers) if tickers is not None else effective_universe()
    successes: dict[str, pd.DataFrame] = {}
    failures: list[DiscoveryFailure] = []

    for idx, ticker in enumerate(universe):
        if idx > 0 and throttle_seconds > 0:
            time.sleep(throttle_seconds)
        df, reason = _backtest_ticker(ticker)
        if df is None:
            # reason is always set when df is None.
            detail = reason or "unknown error"
            print(
                f"  discovery: {ticker} FAILED — {detail}",
                file=sys.stderr,
            )
            failures.append(DiscoveryFailure(ticker=ticker, reason=detail))
            continue
        successes[ticker] = df

    return BacktestRunResult(successes=successes, failures=failures)
