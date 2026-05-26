"""Trade outcome resolver.

Pulls historical price data from yfinance for each open trade and decides
whether the take_profit or stop_loss was hit first within the hold window.

Conservative rule for same-candle ambiguity: if both TP and SL are touched
inside the same candle, we mark the trade as ``loss``. We can't tell from
OHLC which level was hit first inside a candle; assuming SL wins keeps the
win rate honest rather than optimistic.

PnL is percentage-only in Phase 1.3. ``pnl_dollars`` stays ``None`` until
Phase 7 introduces position sizing.
"""

from __future__ import annotations

import dataclasses
import sys
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
import yfinance as yf

from trading_bot import db
from trading_bot.models import Signal, Trade

# Default hold windows when the signal didn't capture an estimate.
_DEFAULT_HOLD_DAYS_STOCK = 30
_DEFAULT_HOLD_DAYS_CRYPTO = 14

# yfinance interval per asset class.
_INTERVAL_STOCK = "1d"
_INTERVAL_CRYPTO = "1h"


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────


def _now_utc() -> datetime:
    return datetime.now(UTC)


def _to_utc(value: Any) -> datetime:
    """Coerce ``value`` (a pandas Timestamp, naive datetime, or aware datetime)
    to a tz-aware UTC ``datetime``.

    Naive timestamps are assumed UTC. The 20 historical CSV-migrated signals
    are naive ET, but the resolver windows are days-wide so the TZ slop is
    smaller than the hold window.
    """
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    if not isinstance(value, datetime):
        raise TypeError(f"Cannot coerce {type(value).__name__} to datetime")
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def calculate_pnl_pct(direction: str, entry: float, exit_price: float) -> float:
    """Return percentage PnL for a closed trade.

    Long-side (``call``/``long``) profits when exit > entry; short-side
    (``put``/``short``) profits when exit < entry. Raises ``ValueError`` for
    any other direction so a typo doesn't silently produce a nonsense number.
    """
    if entry == 0:
        raise ValueError("entry price must be non-zero")
    if direction in ("call", "long"):
        return (exit_price - entry) / entry * 100.0
    if direction in ("put", "short"):
        return (entry - exit_price) / entry * 100.0
    raise ValueError(f"Unknown direction: {direction!r}")


def _check_hit(signal: Signal, high: float, low: float) -> tuple[bool, bool]:
    """Return ``(tp_hit, sl_hit)`` for this candle, given the signal's direction."""
    tp = signal.take_profit
    sl = signal.stop_loss
    if tp is None or sl is None:
        return (False, False)
    if signal.direction in ("call", "long"):
        return (high >= tp, low <= sl)
    if signal.direction in ("put", "short"):
        return (low <= tp, high >= sl)
    return (False, False)


def _hold_window(signal: Signal) -> timedelta:
    if signal.hold_estimate_days is not None and signal.hold_estimate_days > 0:
        return timedelta(days=signal.hold_estimate_days)
    if signal.asset_class == "crypto":
        return timedelta(days=_DEFAULT_HOLD_DAYS_CRYPTO)
    return timedelta(days=_DEFAULT_HOLD_DAYS_STOCK)


def _interval(signal: Signal) -> str:
    return _INTERVAL_CRYPTO if signal.asset_class == "crypto" else _INTERVAL_STOCK


# ────────────────────────────────────────────────────────────────────────────
# yfinance fetch (with per-resolver-run cache)
# ────────────────────────────────────────────────────────────────────────────

# Cache key: (ticker, interval, start_iso, end_iso). Cleared at the start of
# every ``resolve_all_open_trades`` call so stale data doesn't leak across runs.
_CandleCache = dict[tuple[str, str, str, str], pd.DataFrame | None]


def _fetch_candles(
    ticker: str,
    interval: str,
    start: datetime,
    end: datetime,
    cache: _CandleCache | None = None,
) -> pd.DataFrame | None:
    """Pull OHLCV candles from yfinance. Returns ``None`` on empty/failure."""
    key = (ticker, interval, start.isoformat(), end.isoformat())
    if cache is not None and key in cache:
        return cache[key]

    try:
        df = yf.download(
            ticker,
            start=start,
            end=end + timedelta(days=1),  # yfinance end is exclusive on daily
            interval=interval,
            progress=False,
            auto_adjust=False,
        )
    except Exception as exc:
        print(f"  yfinance error for {ticker}: {exc}", file=sys.stderr)
        if cache is not None:
            cache[key] = None
        return None

    if df is None or df.empty:
        if cache is not None:
            cache[key] = None
        return None

    # yfinance sometimes returns MultiIndex columns for single tickers — flatten.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    if cache is not None:
        cache[key] = df
    return df


# ────────────────────────────────────────────────────────────────────────────
# Core resolver
# ────────────────────────────────────────────────────────────────────────────


def _close_trade(
    trade: Trade,
    signal: Signal,
    *,
    outcome: str,
    exit_price: float,
    closed_at: datetime,
) -> Trade:
    pnl_pct = calculate_pnl_pct(signal.direction, signal.entry_price, exit_price)
    return dataclasses.replace(
        trade,
        outcome=outcome,
        exit_price=exit_price,
        closed_at=closed_at,
        pnl_pct=pnl_pct,
        pnl_dollars=None,
    )


def resolve_trade(
    trade: Trade,
    signal: Signal,
    *,
    now: datetime | None = None,
    cache: _CandleCache | None = None,
) -> Trade:
    """Resolve a single trade against historical price data.

    Returns a new ``Trade`` with updated outcome / closed_at / exit_price /
    pnl_pct, or the trade unchanged with ``outcome='open'`` if it can't yet
    be resolved (hold window still active, no candles available, etc.).
    """
    if trade.outcome not in (None, "open"):
        return trade
    if signal.take_profit is None or signal.stop_loss is None:
        # Without TP/SL we can't resolve. Leave as open so a future run can
        # try again if the signal is fixed up.
        return dataclasses.replace(trade, outcome="open")

    opened_at = _to_utc(trade.opened_at)
    deadline  = opened_at + _hold_window(signal)
    now_utc   = now if now is not None else _now_utc()
    end       = min(now_utc, deadline)

    if end <= opened_at:
        # Brand-new trade, no price action to evaluate yet.
        return dataclasses.replace(trade, outcome="open")

    df = _fetch_candles(signal.ticker, _interval(signal), opened_at, end, cache)
    if df is None or df.empty:
        # No data. If we're already past the deadline, give up and mark expired
        # at the deadline timestamp with no exit price.
        if now_utc >= deadline:
            return dataclasses.replace(
                trade, outcome="expired", closed_at=deadline, exit_price=None, pnl_pct=None
            )
        return dataclasses.replace(trade, outcome="open")

    last_close: float | None = None
    for raw_ts, row in df.iterrows():
        candle_time = _to_utc(raw_ts)
        if candle_time < opened_at:
            continue

        high = float(row["High"])
        low  = float(row["Low"])
        last_close = float(row["Close"])

        tp_hit, sl_hit = _check_hit(signal, high, low)

        # Same-candle ambiguity → conservatively mark as loss.
        # OHLC can't tell us which side was touched first; assuming SL keeps
        # the win rate honest rather than optimistic.
        if tp_hit and sl_hit:
            return _close_trade(
                trade, signal,
                outcome="loss",
                exit_price=float(signal.stop_loss),
                closed_at=candle_time,
            )
        if sl_hit:
            return _close_trade(
                trade, signal,
                outcome="loss",
                exit_price=float(signal.stop_loss),
                closed_at=candle_time,
            )
        if tp_hit:
            return _close_trade(
                trade, signal,
                outcome="win",
                exit_price=float(signal.take_profit),
                closed_at=candle_time,
            )

    # No TP/SL hit in any candle.
    if now_utc >= deadline:
        # Hold window elapsed — close at deadline using last available close.
        return _close_trade(
            trade, signal,
            outcome="expired",
            exit_price=last_close if last_close is not None else signal.entry_price,
            closed_at=deadline,
        )
    return dataclasses.replace(trade, outcome="open")


# ────────────────────────────────────────────────────────────────────────────
# Bulk operations
# ────────────────────────────────────────────────────────────────────────────


def resolve_all_open_trades(*, now: datetime | None = None) -> dict[str, int]:
    """Resolve every open trade against historical price data.

    Persists updates via :func:`db.update_trade`. Returns a summary dict:
    ``{'wins': N, 'losses': N, 'expired': N, 'still_open': N}``.
    """
    cache: _CandleCache = {}
    summary = {"wins": 0, "losses": 0, "expired": 0, "still_open": 0}

    for trade in db.get_open_trades():
        if trade.id is None:
            continue
        signal = db.get_signal_by_id(trade.signal_id)
        if signal is None:
            # Orphan trade with no signal — shouldn't happen given FK constraints,
            # but defensively skip rather than crash.
            continue
        try:
            resolved = resolve_trade(trade, signal, now=now, cache=cache)
        except Exception as exc:
            print(
                f"  resolver error for trade #{trade.id} ({signal.ticker}): {exc}",
                file=sys.stderr,
            )
            summary["still_open"] += 1
            continue

        if resolved.outcome == "win":
            summary["wins"] += 1
        elif resolved.outcome == "loss":
            summary["losses"] += 1
        elif resolved.outcome == "expired":
            summary["expired"] += 1
        else:
            summary["still_open"] += 1
            continue  # nothing to persist

        db.update_trade(
            trade.id,
            outcome=resolved.outcome,
            closed_at=resolved.closed_at,
            exit_price=resolved.exit_price,
            pnl_pct=resolved.pnl_pct,
        )

    return summary


def backfill_signals_without_trades() -> int:
    """Open a Trade for every signal that doesn't yet have one, then resolve.

    Returns the count of Trade rows created. Idempotent — running twice in a
    row creates zero trades the second time. Intended as a one-shot for the
    20 historical CSV-imported signals from Phase 1.1c.
    """
    orphans = db.get_signals_without_trades()
    created = 0
    for sig in orphans:
        if sig.id is None:
            continue
        db.insert_trade(
            Trade(signal_id=sig.id, opened_at=sig.timestamp, outcome="open")
        )
        created += 1

    # Resolve everything we just opened (plus any pre-existing open trades).
    resolve_all_open_trades()
    return created


# ────────────────────────────────────────────────────────────────────────────
# Reporting
# ────────────────────────────────────────────────────────────────────────────


def summary() -> dict[str, Any]:
    """Aggregate closed-trade outcomes by signal_type, plus open/expired counts."""
    by_type_sql = (
        "SELECT "
        "  signals.signal_type AS signal_type, "
        "  SUM(CASE WHEN trades.outcome = 'win'  THEN 1 ELSE 0 END) AS wins, "
        "  SUM(CASE WHEN trades.outcome = 'loss' THEN 1 ELSE 0 END) AS losses, "
        "  AVG(CASE WHEN trades.outcome IN ('win','loss') THEN trades.pnl_pct END) AS avg_pnl "
        "FROM signals "
        "JOIN trades ON trades.signal_id = signals.id "
        "WHERE trades.outcome IN ('win', 'loss') "
        "GROUP BY signals.signal_type "
        "ORDER BY signals.signal_type"
    )
    open_sql = (
        "SELECT COUNT(*) AS c FROM trades WHERE outcome IS NULL OR outcome = 'open'"
    )
    expired_sql = "SELECT COUNT(*) AS c FROM trades WHERE outcome = 'expired'"

    conn = db.get_connection()
    try:
        by_type_rows = conn.execute(by_type_sql).fetchall()
        open_row    = conn.execute(open_sql).fetchone()
        expired_row = conn.execute(expired_sql).fetchone()
    finally:
        conn.close()

    by_signal_type = []
    for row in by_type_rows:
        wins = int(row["wins"])
        losses = int(row["losses"])
        total = wins + losses
        win_rate = (wins / total * 100.0) if total else 0.0
        avg_pnl = float(row["avg_pnl"]) if row["avg_pnl"] is not None else 0.0
        by_signal_type.append(
            {
                "signal_type": str(row["signal_type"]),
                "wins": wins,
                "losses": losses,
                "win_rate": win_rate,
                "avg_pnl": avg_pnl,
            }
        )

    return {
        "by_signal_type": by_signal_type,
        "open":    int(open_row["c"])    if open_row    is not None else 0,
        "expired": int(expired_row["c"]) if expired_row is not None else 0,
    }
