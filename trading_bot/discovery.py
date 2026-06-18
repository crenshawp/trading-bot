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
from datetime import UTC, datetime
from typing import cast

import pandas as pd

import backtest
from trading_bot import db
from trading_bot.discovery_universe import effective_universe

# Small sleep between yfinance-backed backtests to avoid rate limiting.
_THROTTLE_SECONDS = 1.0

# Permissive promotion gate (Phase 3.1). A ticker qualifies if it has at least
# this many decided backtest trades AND positive expectancy. No win-rate floor:
# the watchlist is intentionally permissive at this stage — this filters only
# money-losers and tiny-sample noise. Phase 3.3 (kickout) prunes marginal names
# later on live data.
_MIN_TRADE_COUNT = 10

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


# ──────────────────────────────────────────────────────────────────────────
# Scoring + permissive promotion gate (Section 4)
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TickerScore:
    """Per-ticker backtest score.

    ``win_rate`` is 0-100. ``avg_return_pct`` and ``expectancy`` are in
    percentage points per trade. All three are ``None`` when a ticker had no
    *decided* trades (every signal expired without hitting TP/SL). ``qualified``
    is the permissive gate: ``trade_count >= 10 AND expectancy > 0``.
    """

    ticker: str
    win_rate: float | None
    trade_count: int
    avg_return_pct: float | None
    expectancy: float | None
    qualified: bool


def _per_trade_return(
    direction: str,
    price: float,
    take_profit: float,
    stop_loss: float,
    outcome: str,
) -> float | None:
    """Signed % return for one decided trade; ``None`` for UNKNOWN (no exit).

    Reconstructs the exit from the backtest's TP/SL levels: a WIN exits at
    ``take_profit``, a LOSS at ``stop_loss``. Positive for wins, negative for
    losses, for both CALL and PUT.
    """
    if outcome == "WIN":
        exit_price = take_profit
    elif outcome == "LOSS":
        exit_price = stop_loss
    else:
        return None
    if price == 0:
        return None
    if direction.upper() == "CALL":
        return float((exit_price - price) / price * 100.0)
    return float((price - exit_price) / price * 100.0)


def score_backtest(ticker: str, df: pd.DataFrame) -> TickerScore:
    """Compute win rate, trade count, avg return, and expectancy for a ticker.

    Only *decided* trades (outcome WIN/LOSS) count — UNKNOWN (no TP/SL hit in
    the forward window) are excluded, matching the backtest's own summary.
    expectancy = (win_frac * avg_win) - (loss_frac * avg_loss), where avg_win
    and avg_loss are average magnitudes in percentage points.
    """
    win_returns: list[float] = []
    loss_returns: list[float] = []
    for row in df.itertuples(index=False):
        ret = _per_trade_return(
            str(row.direction),
            float(row.price),
            float(row.take_profit),
            float(row.stop_loss),
            str(row.outcome),
        )
        if ret is None:
            continue
        if str(row.outcome) == "WIN":
            win_returns.append(ret)
        else:
            loss_returns.append(ret)

    wins = len(win_returns)
    losses = len(loss_returns)
    trade_count = wins + losses
    if trade_count == 0:
        return TickerScore(
            ticker=ticker,
            win_rate=None,
            trade_count=0,
            avg_return_pct=None,
            expectancy=None,
            qualified=False,
        )

    win_frac = wins / trade_count
    loss_frac = losses / trade_count
    avg_win = sum(win_returns) / wins if wins else 0.0
    avg_loss = sum(-r for r in loss_returns) / losses if losses else 0.0
    expectancy = win_frac * avg_win - loss_frac * avg_loss
    avg_return_pct = sum(win_returns + loss_returns) / trade_count
    win_rate = win_frac * 100.0
    qualified = trade_count >= _MIN_TRADE_COUNT and expectancy > 0

    return TickerScore(
        ticker=ticker,
        win_rate=win_rate,
        trade_count=trade_count,
        avg_return_pct=avg_return_pct,
        expectancy=expectancy,
        qualified=qualified,
    )


def score_run(result: BacktestRunResult) -> list[TickerScore]:
    """Score every successfully-backtested ticker from a sweep."""
    return [score_backtest(ticker, df) for ticker, df in result.successes.items()]


def rank_qualifiers(scores: Sequence[TickerScore]) -> list[TickerScore]:
    """Return the qualifying scores ranked by expectancy, descending."""
    quals = [s for s in scores if s.qualified]
    return sorted(quals, key=lambda s: s.expectancy or 0.0, reverse=True)


# ──────────────────────────────────────────────────────────────────────────
# Informational ranking + persistence (Section 5)
# ──────────────────────────────────────────────────────────────────────────
#
# As of Phase 3.1-LIVE the backtest scan is INFORMATIONAL ONLY — it ranks and
# persists candidates but promotes NOTHING. Live-shadow (trading_bot.
# shadow_discovery) is the sole path into the active watchlist, driven by real
# resolved shadow outcomes rather than a backtest.


@dataclass
class DiscoveryRun:
    """The full result of one on-demand backtest discovery sweep.

    ``scores`` is every successfully-backtested ticker; ``failures`` every one
    that could not be scored. This sweep does not promote anything — see
    :mod:`trading_bot.shadow_discovery` for the live-shadow promotion path.
    """

    run_timestamp: datetime
    scores: list[TickerScore]
    failures: list[DiscoveryFailure]

    @property
    def scanned(self) -> int:
        return len(self.scores) + len(self.failures)

    @property
    def succeeded(self) -> int:
        return len(self.scores)

    @property
    def failed(self) -> int:
        return len(self.failures)

    @property
    def qualifiers(self) -> list[TickerScore]:
        return rank_qualifiers(self.scores)


def _score_to_row(score: TickerScore) -> dict[str, object]:
    """Flatten a TickerScore into the dict shape db.insert_discovery_results
    expects. Kept here so db.py stays decoupled from the discovery dataclass."""
    return {
        "ticker": score.ticker,
        "win_rate": score.win_rate,
        "trade_count": score.trade_count,
        "avg_return_pct": score.avg_return_pct,
        "expectancy": score.expectancy,
        "qualified": score.qualified,
    }


def run_discovery(
    *,
    tickers: Sequence[str] | None = None,
    throttle_seconds: float = _THROTTLE_SECONDS,
) -> DiscoveryRun:
    """Informational backtest sweep: backtest -> score -> persist. Promotes nothing.

    Persists every scored ticker to ``discovery_results`` (full audit trail)
    and returns the ranked run for display. Promotion into the active watchlist
    is owned exclusively by live-shadow (:mod:`trading_bot.shadow_discovery`);
    this sweep never touches ``active_watchlist``.
    """
    run_ts = datetime.now(UTC)
    result = run_backtests(tickers=tickers, throttle_seconds=throttle_seconds)
    scores = score_run(result)

    db.insert_discovery_results(
        run_ts.isoformat(), [_score_to_row(s) for s in scores]
    )

    return DiscoveryRun(
        run_timestamp=run_ts,
        scores=scores,
        failures=result.failures,
    )
