"""Long-term buy-and-hold trading system — Phase 14 (stocks + crypto, PAPER).

Entry and protective-exit logic for the LONG_TERM (buy-hold stocks) and CRYPTO
(long-term BTC/ETH/BNB) pools reserved by the Phase 12 allocator. Buy-and-hold:
fractional shares, NO options, NO tight stops.

Entry candidates are generated in the daily scan and feed the SAME
operator-inspected Phase 12 allocation plan as swing/options signals — there is
NO new auto-execution of entries. The one thing that runs and acts automatically
is the PROTECTIVE EXIT watcher (closing, never opening), matching the Phase 13
options-watcher / kill-switch precedent.

Entry gates (applied by the candidate generator):
  1. fundamental red-flag screen (stocks only, FAIL-OPEN — a loose filter, not a
     ranker; blocks only on a genuine red flag);
  2. long-horizon technical confirmation (stocks + crypto): price above the
     TREND_PERIOD trend AND ADX >= a moderate floor AND RSI <= overbought —
     reusing the Phase 6 indicator functions, not reimplementing them;
  3. earnings SOFT wait-window (stocks only) — distinct from the Phase 5 blackout.

Protective exits (this phase, no profit-taking): close on EITHER a trend
breakdown (N consecutive closes below the trend) OR a drawdown-from-entry stop.

The crypto SWING signals (Oversold Reversal, Momentum Breakout) remain data-only
FOREVER — they never route to execution (enforced in allocation.build_plan via
config.DATA_ONLY_SIGNAL_TYPES). Only this module's long-term crypto entry signal
may route to the CRYPTO pool. Still PAPER ONLY.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
import yfinance as yf

from trading_bot import allocation, config, earnings, indicators


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class LongTermCandidate:
    """A buy-and-hold entry candidate to feed the Phase 12 allocation plan.

    ``signal_type`` is one of ``config.LONGTERM_STOCK_SIGNAL`` /
    ``LONGTERM_CRYPTO_SIGNAL`` — the ONLY long-term types that route to
    execution. ``entry_rationale`` records which gates it cleared (audit)."""

    ticker: str
    asset_class: str               # 'stock' | 'crypto'
    signal_type: str               # long_term_stock | long_term_crypto
    entry_price: float
    entry_rationale: str = ""


@dataclass(frozen=True)
class LongTermPosition:
    """An open/closed buy-and-hold position (no tight stop; protective exit only)."""

    ticker: str
    asset_class: str
    entry_price: float
    entry_date: datetime
    qty: float
    status: str = "open"           # 'open' | 'closed'
    exit_price: float | None = None
    exit_date: datetime | None = None
    exit_reason: str | None = None  # trend_breakdown | drawdown_stop
    id: int | None = None


# ── fundamental red-flag screen (stocks only, FAIL-OPEN) ─────────────────────


def _fetch_fundamentals(ticker: str) -> tuple[float | None, float | None]:
    """Fetch ``(earnings_growth, debt_to_equity)`` via yfinance. Never raises.

    Isolated so tests monkeypatch it without touching the network. Any failure —
    or a field yfinance simply does not carry — yields ``None`` for that datum,
    which the screen treats as fail-open.
    """
    try:
        info = yf.Ticker(ticker).info
        if not isinstance(info, dict):
            return None, None
        growth = info.get("earningsGrowth")
        if growth is None:
            growth = info.get("earningsQuarterlyGrowth")
        return _to_float(growth), _to_float(info.get("debtToEquity"))
    except Exception as exc:  # noqa: BLE001 - fundamentals lookup must never raise
        print(f"  longterm: fundamentals fetch error for {ticker}: {exc}",
              file=sys.stderr)
        return None, None


def screen_fundamentals(
    earnings_growth: float | None, debt_to_equity: float | None,
) -> tuple[bool, str]:
    """Pure red-flag screen. Returns ``(blocked, reason)``.

    A LOOSE filter, NOT a quality ranker: a candidate is blocked ONLY on a
    genuine red flag — materially negative earnings growth (below
    ``FUND_EARNINGS_GROWTH_FLOOR``) AND excessive debt/equity (above
    ``FUND_DEBT_EQUITY_CEILING``), BOTH required. If EITHER datum is missing the
    screen FAILS OPEN (``blocked=False``) — unknown fundamentals never block.
    """
    if earnings_growth is None or debt_to_equity is None:
        return False, "fundamentals incomplete - fail-open (no block)"
    if (
        earnings_growth < config.FUND_EARNINGS_GROWTH_FLOOR
        and debt_to_equity > config.FUND_DEBT_EQUITY_CEILING
    ):
        return True, (
            f"red flag: earnings growth {earnings_growth:.2f} < "
            f"{config.FUND_EARNINGS_GROWTH_FLOOR} AND debt/equity "
            f"{debt_to_equity:.0f} > {config.FUND_DEBT_EQUITY_CEILING:.0f}"
        )
    return False, "no red flag"


def fundamental_red_flag(ticker: str) -> tuple[bool, str]:
    """Fetch fundamentals and apply the red-flag screen. Fail-open, never raises."""
    growth, debt_equity = _fetch_fundamentals(ticker)
    blocked, reason = screen_fundamentals(growth, debt_equity)
    if blocked:
        print(f"  longterm: {ticker} blocked by fundamentals ({reason})",
              file=sys.stderr)
    return blocked, reason


# ── long-horizon technical entry confirmation (stocks + crypto) ──────────────


def _val(series: pd.Series, at: int = -1) -> float | None:
    """Scalar at position ``at``; None on out-of-range or NaN."""
    try:
        value = series.iloc[at]
    except (IndexError, KeyError):
        return None
    return None if pd.isna(value) else float(value)


def _confirm_conditions(
    price: float | None, trend: float | None, adx: float | None,
    rsi: float | None, *, adx_min: float, rsi_max: float,
) -> tuple[bool, str]:
    """Pure entry-confirmation logic: ALL THREE conditions required.

    Price above the long-horizon trend AND ADX at/above the moderate-trend floor
    AND RSI at/below the overbought ceiling. Any missing datum -> not confirmed
    (insufficient data). Returns ``(confirmed, reason)``.
    """
    if price is None or trend is None or adx is None or rsi is None:
        return False, "insufficient data"
    above = price > trend
    adx_ok = adx >= adx_min
    rsi_ok = rsi <= rsi_max
    if above and adx_ok and rsi_ok:
        return True, (
            f"price {price:.2f} > trend {trend:.2f}, ADX {adx:.1f} >= {adx_min}, "
            f"RSI {rsi:.1f} <= {rsi_max}"
        )
    fails: list[str] = []
    if not above:
        fails.append(f"price {price:.2f} <= trend {trend:.2f}")
    if not adx_ok:
        fails.append(f"ADX {adx:.1f} < {adx_min}")
    if not rsi_ok:
        fails.append(f"RSI {rsi:.1f} > {rsi_max}")
    return False, "; ".join(fails)


def confirm_technical_entry(
    df: pd.DataFrame,
    *,
    trend_period: int = config.TREND_PERIOD,
    adx_min: float = config.ADX_MIN_TREND,
    rsi_max: float = config.RSI_OVERBOUGHT_MAX,
    at: int = -1,
) -> tuple[bool, str]:
    """Confirm a long-term entry from a candle frame, reusing the Phase 6
    indicators (``indicators.sma`` / ``adx`` / ``rsi``) at the long horizon.
    Pure — no I/O. Fail-soft: a short frame yields ``(False, 'insufficient
    data')`` rather than raising."""
    try:
        close = df["Close"]
        price = _val(close, at)
        trend = _val(indicators.sma(close, trend_period), at)
        adx = _val(indicators.adx(df), at)
        rsi = _val(indicators.rsi(close), at)
    except Exception as exc:  # noqa: BLE001 - confirmation must never raise into a scan
        print(f"  longterm: technical confirm error: {exc}", file=sys.stderr)
        return False, f"error: {exc}"
    return _confirm_conditions(price, trend, adx, rsi, adx_min=adx_min, rsi_max=rsi_max)


# ── earnings SOFT wait-window (stocks only; distinct from the Phase 5 blackout) ─


def in_earnings_wait_window(
    ticker: str, *, wait_days: int = config.EARNINGS_WAIT_DAYS,
    now: datetime | None = None,
) -> tuple[bool, str]:
    """Return ``(in_window, reason)`` for the long-term earnings SOFT window.

    DISTINCT from the Phase 5 hold-window blackout (different threshold, different
    purpose): if a KNOWN earnings date falls within ``[today, today+wait_days]``
    we SKIP this cycle for a clean entry price and simply retry a later cycle — it
    is not a permanent suppression. An UNKNOWN date proceeds (never a reason to
    skip). Reuses ``earnings.next_earnings_date``.
    """
    moment = now if now is not None else datetime.now(UTC)
    info = earnings.next_earnings_date(ticker)
    if info.earnings_date is None:
        return False, "earnings unknown - proceed"
    today = moment.date()
    window_end = (moment + timedelta(days=wait_days)).date()
    earnings_day = info.earnings_date.date()
    if today <= earnings_day <= window_end:
        return True, (
            f"earnings {earnings_day.isoformat()} within {wait_days}d wait "
            "window - skip, retry later"
        )
    return False, f"earnings {earnings_day.isoformat()} outside {wait_days}d wait window"


# ── candidate generation (feeds the SAME Phase 12 allocation plan) ────────────


def to_allocation_candidate(candidate: LongTermCandidate) -> allocation.Candidate:
    """Convert a long-term candidate into a Phase 12 allocation Candidate.

    Long-term entries are buy-and-hold longs with NO ATR/stop sizing (``atr=None``
    — the LONG_TERM/CRYPTO pools size by diversification weight, not Phase 7); the
    swing gate flags default to permissive since the long-term gates were already
    applied by the generator."""
    return allocation.Candidate(
        ticker=candidate.ticker, signal_type=candidate.signal_type,
        direction="long", asset_class=candidate.asset_class,
        entry=candidate.entry_price, atr=None,
    )


def _safe_fetch(
    price_fetch: Callable[[str], pd.DataFrame | None], ticker: str,
) -> pd.DataFrame | None:
    try:
        return price_fetch(ticker)
    except Exception as exc:  # noqa: BLE001 - one fetch failure must not sink the run
        print(f"  longterm: price fetch error for {ticker}: {exc}", file=sys.stderr)
        return None


def generate_candidates(
    price_fetch: Callable[[str], pd.DataFrame | None],
    *,
    now: datetime | None = None,
    stock_universe: tuple[str, ...] = config.LONGTERM_STOCK_UNIVERSE,
    crypto_universe: tuple[str, ...] = config.LONGTERM_CRYPTO_UNIVERSE,
) -> list[LongTermCandidate]:
    """Generate long-term entry candidates for the daily scan.

    STOCKS: fundamental red-flag screen (fail-open) → technical confirmation →
    earnings wait-window. CRYPTO: technical confirmation ONLY (no fundamental
    screen exists for crypto). Emits a :class:`LongTermCandidate` for anything
    that clears all applicable gates; each emitted candidate carries ONLY a
    long-term signal type, so the crypto SWING signals can never appear here.
    Populates candidates for the next allocate-plan inspection — submits NOTHING.
    ``price_fetch`` is injected (candles per ticker), keeping this testable.
    """
    moment = now if now is not None else datetime.now(UTC)
    out: list[LongTermCandidate] = []

    for ticker in stock_universe:
        df = _safe_fetch(price_fetch, ticker)
        if df is None:
            continue
        blocked, _reason = fundamental_red_flag(ticker)
        if blocked:
            continue
        confirmed, tech_reason = confirm_technical_entry(df)
        if not confirmed:
            continue
        in_wait, _wreason = in_earnings_wait_window(ticker, now=moment)
        if in_wait:
            continue
        price = _val(df["Close"], -1)
        if price is None:
            continue
        out.append(LongTermCandidate(
            ticker=ticker, asset_class="stock",
            signal_type=config.LONGTERM_STOCK_SIGNAL, entry_price=price,
            entry_rationale=f"stock long-term: {tech_reason}",
        ))

    for ticker in crypto_universe:
        df = _safe_fetch(price_fetch, ticker)
        if df is None:
            continue
        confirmed, tech_reason = confirm_technical_entry(df)
        if not confirmed:
            continue
        price = _val(df["Close"], -1)
        if price is None:
            continue
        out.append(LongTermCandidate(
            ticker=ticker, asset_class="crypto",
            signal_type=config.LONGTERM_CRYPTO_SIGNAL, entry_price=price,
            entry_rationale=f"crypto long-term: {tech_reason}",
        ))

    return out
