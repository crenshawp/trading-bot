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

import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
import yfinance as yf

from trading_bot import allocation, config, db, earnings, indicators, risk_of_ruin
from trading_bot.broker.base import ORDER_TYPE_LIMIT, TIF_DAY, Broker, OrderResult
from trading_bot.models import LongTermCandidate, LongTermPosition, Signal

__all__ = ["LongTermCandidate", "LongTermPosition"]


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


def fetch_daily_candles(
    ticker: str, *, period_days: int | None = None,
) -> pd.DataFrame | None:
    """Fetch daily candles for the long-horizon trend. Fail-soft → None.

    Pulls enough history to compute the TREND_PERIOD SMA. Isolated so the CLI and
    scan can use it while tests inject their own fetch."""
    days = period_days if period_days is not None else config.TREND_PERIOD + 60
    try:
        frame = yf.download(
            ticker, period=f"{days}d", interval="1d", progress=False,
            auto_adjust=False,
        )
    except Exception as exc:  # noqa: BLE001 - a data fetch must never raise into a scan
        print(f"  longterm: candle fetch error for {ticker}: {exc}", file=sys.stderr)
        return None
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return None
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.droplevel(1)
    return frame


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


# ── candidate persistence (Phase 17 — long-term entries land in signals) ─────


def persist_candidates(
    candidates: Sequence[LongTermCandidate], *, now: datetime | None = None,
) -> list[int]:
    """Log generated long-term candidates into the ``signals`` table (Phase 17).

    Long-term entries historically lived only in memory (generated on demand,
    never persisted) — so the live candidate source could never see them. This
    records each one as a fired signal (direction ``'long'``, the generator's
    rationale kept in ``raw_indicators_json``) WITHOUT opening a trades row:
    long-term outcomes are tracked in ``long_term_positions`` by the protective
    exit watcher, not by the TP/SL trade resolver. One shared timestamp per
    batch lets the dedupe index collapse an accidental double-insert within a
    run. Returns the signal ids. Fail-soft per candidate: one bad row is logged
    and never blocks the rest.
    """
    moment = now if now is not None else datetime.now(UTC)
    ids: list[int] = []
    for c in candidates:
        try:
            ids.append(db.insert_signal(Signal(
                timestamp=moment,
                ticker=c.ticker,
                asset_class=c.asset_class,
                signal_type=c.signal_type,
                direction="long",
                entry_price=c.entry_price,
                raw_indicators_json=json.dumps(
                    {"entry_rationale": c.entry_rationale}
                ),
            )))
        except Exception as exc:  # noqa: BLE001 - one candidate must never block the batch
            print(
                f"  longterm: candidate persist error for {c.ticker}: {exc}",
                file=sys.stderr,
            )
    return ids


# ── entry submission (Phase 16 — operator-triggered via allocate execute) ────


def submit_long_term_entry(
    broker: Broker,
    *,
    ticker: str,
    asset_class: str,
    qty: float,
    entry_price: float,
    now: datetime,
) -> tuple[OrderResult, int | None]:
    """THE long-term ENTRY path (Phase 16) — a fractional-share BUY LIMIT via
    the Phase 11 equity order path, at the diversification-weighted qty/cost
    the Phase 12 allocator already computed. NO new sizing logic — this is only
    the submission call that was missing between "plan produced" and "order
    submitted".

    The mirror of :func:`close_long_term_position`: on a broker-accepted order
    the position row is inserted OPEN so the EXISTING protective exit watcher
    manages it from the next cycle. A rejected/errored order records NOTHING —
    no position is ever recorded that was not submitted. Never called
    automatically; the operator-gated ``allocate execute --confirm`` is the
    only caller. Returns ``(order, position_id | None)``.
    """
    order = broker.submit_order(
        ticker, qty, "buy", order_type=ORDER_TYPE_LIMIT,
        limit_price=entry_price, time_in_force=TIF_DAY,
    )
    risk_of_ruin.record_broker_result(order.ok)   # Phase 15 detector
    position_id: int | None = None
    if order.ok:
        position_id = db.insert_long_term_position(LongTermPosition(
            ticker=ticker, asset_class=asset_class, entry_price=entry_price,
            entry_date=now, qty=qty, status="open",
        ))
    return order, position_id


# ── protective exit watcher (AUTO-EXECUTING; closing is safe to automate) ─────
#
# Mirrors the Phase 13 options watcher: closing (never opening) is the one action
# safe to automate. No profit-taking this phase — protective exits ONLY.

EXIT_TREND_BREAKDOWN = "trend_breakdown"
EXIT_DRAWDOWN_STOP = "drawdown_stop"
EXIT_HOLDING = "holding"


@dataclass(frozen=True)
class ProtectiveExitDecision:
    """Whether to protectively close a long-term position now, and why."""

    action: str      # 'close' | 'hold'
    reason: str      # trend_breakdown | drawdown_stop | holding


@dataclass(frozen=True)
class ExitAction:
    """What the watcher did for one position this cycle."""

    position: LongTermPosition
    action: str                      # 'close' | 'hold' | 'error'
    reason: str
    order: OrderResult | None = None


def evaluate_protective_exit(
    entry_price: float,
    current_price: float | None,
    consecutive_closes_below_trend: int,
    *,
    breakdown_days: int = config.TREND_BREAKDOWN_DAYS,
    max_drawdown_pct: float = config.MAX_DRAWDOWN_STOP_PCT,
) -> ProtectiveExitDecision:
    """Pure protective-exit decision. Close on EITHER a drawdown-from-entry stop
    OR a trend breakdown (N consecutive closes below the long-horizon trend);
    otherwise hold. There is NO profit-taking exit this phase. Never raises.

    The drawdown stop is checked first (capital protection); with neither
    condition met the position is held.
    """
    if entry_price > 0.0 and current_price is not None:
        drawdown_pct = (entry_price - current_price) / entry_price * 100.0
        if drawdown_pct >= max_drawdown_pct:
            return ProtectiveExitDecision("close", EXIT_DRAWDOWN_STOP)
    if consecutive_closes_below_trend >= breakdown_days:
        return ProtectiveExitDecision("close", EXIT_TREND_BREAKDOWN)
    return ProtectiveExitDecision("hold", EXIT_HOLDING)


def close_long_term_position(
    broker: Broker,
    position: LongTermPosition,
    *,
    exit_price: float | None,
    now: datetime,
    reason: str,
) -> OrderResult:
    """THE long-term closing path — a SELL LIMIT via the Phase 11 equity path.
    Used by the protective exit watcher AND the Phase 15 emergency shutdown, so
    there is exactly one closer.

    The position row is marked closed ONLY when the broker accepted the order —
    a rejected close (e.g. market closed) leaves it open so the next cycle
    retries; nothing is recorded closed without a real order.
    """
    limit_price = exit_price if exit_price is not None else position.entry_price
    order = broker.submit_order(
        position.ticker, position.qty, "sell", order_type=ORDER_TYPE_LIMIT,
        limit_price=limit_price, time_in_force=TIF_DAY,
    )
    risk_of_ruin.record_broker_result(order.ok)   # Phase 15 detector
    if order.ok and position.id is not None:
        db.update_long_term_position(
            position.id, status="closed", exit_price=exit_price,
            exit_date=now, exit_reason=reason,
        )
    return order


def _consecutive_closes_below_trend(
    df: pd.DataFrame, trend_period: int = config.TREND_PERIOD,
) -> int:
    """Count the trailing run of closes STRICTLY below the ``trend_period`` SMA.

    Walks backward from the latest candle; stops at the first close at/above the
    trend or the first NaN trend value (too little data). Never raises → 0."""
    try:
        close = df["Close"]
        trend = indicators.sma(close, trend_period)
        count = 0
        for i in range(len(close) - 1, -1, -1):
            trend_val = trend.iloc[i]
            if pd.isna(trend_val):
                break
            if float(close.iloc[i]) < float(trend_val):
                count += 1
            else:
                break
        return count
    except Exception as exc:  # noqa: BLE001 - the watcher must never raise here
        print(f"  longterm: trend-breakdown count error: {exc}", file=sys.stderr)
        return 0


def watch_long_term_positions(
    broker: Broker,
    *,
    price_fetch: Callable[[str], pd.DataFrame | None],
    now: datetime,
    trend_period: int = config.TREND_PERIOD,
    positions: Sequence[LongTermPosition] | None = None,
) -> list[ExitAction]:
    """Check each open long-term position and AUTO-CLOSE the ones that hit a
    protective-exit condition (trend breakdown or drawdown stop).

    For a closing position it submits a SELL LIMIT via the Phase 11 equity path
    (marketable at the latest close), records the ``exit_reason``, and marks the
    row closed. FAIL-SOFT: an error on one position is logged and NEVER blocks the
    others. ``price_fetch`` is injected (candles per ticker), keeping this
    testable; ``positions`` defaults to the open book from the DB.
    """
    book = positions if positions is not None else db.get_open_long_term_positions()
    actions: list[ExitAction] = []
    for pos in book:
        try:
            df = price_fetch(pos.ticker)
            if df is None:
                actions.append(ExitAction(pos, "hold", "no price data"))
                continue
            current_price = _val(df["Close"], -1)
            consecutive = _consecutive_closes_below_trend(df, trend_period)
            decision = evaluate_protective_exit(
                pos.entry_price, current_price, consecutive,
            )
            if decision.action == "hold":
                actions.append(ExitAction(pos, "hold", decision.reason))
                continue

            order = close_long_term_position(
                broker, pos, exit_price=current_price, now=now,
                reason=decision.reason,
            )
            if not order.ok:
                # Close not accepted (e.g. market closed) — row stays open so the
                # next cycle retries; never recorded closed without a real order.
                actions.append(ExitAction(
                    pos, "error", f"close order failed: {order.reason}", order=order,
                ))
                continue
            actions.append(ExitAction(pos, "close", decision.reason, order=order))
        except Exception as exc:  # noqa: BLE001 - one position must never block others
            print(f"  longterm watcher error for {pos.ticker}: {exc}",
                  file=sys.stderr)
            actions.append(ExitAction(pos, "error", str(exc)))
    return actions
