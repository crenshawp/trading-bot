"""Macro regime detection — bull / bear / sideways from SPY 50/200 EMA dynamics.

Phase 2.1 (Market Intelligence). The classifier is intentionally pure and
exposed via :func:`classify` so the bull/bear/sideways rule set can be unit
tested without yfinance involvement. Production code calls
:func:`get_current_regime`, which handles fetching, caching, and error
propagation.

Caching:

* In-process cache lives for the lifetime of the interpreter.
* Disk cache lives at ``.regime_cache.json`` next to the project root.
* TTL is 24 hours.
* ``force_refresh=True`` bypasses both layers.

yfinance failures (network, rate limit, malformed response) raise
:class:`RegimeFetchError`. The scanner catches this and tags the trade with
``market_regime='unknown'`` rather than crashing — signal capture is more
important than regime tagging.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

from trading_bot import config, db

# ────────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────────

_CACHE_PATH: Path = config.ROOT_DIR / ".regime_cache.json"
_CACHE_TTL: timedelta = timedelta(hours=24)
_SPY_TICKER: str = "SPY"
_EMA_SLOPE_WINDOW: int = 5  # 5-day slope of the 50 EMA
_MIN_ROWS_FOR_200_EMA: int = 200
_MIN_ROWS: int = _MIN_ROWS_FOR_200_EMA + _EMA_SLOPE_WINDOW


def _calendar_days_for_bars(bars: int) -> int:
    """Calendar-day span that reliably contains ``bars`` daily TRADING bars.

    yfinance's ``period`` is a CALENDAR range, but ``_MIN_ROWS`` is a BAR count.
    Equities trade ~5 of every 7 calendar days, and fewer still after market
    holidays, so a calendar span sized as if it were a bar count comes up short:
    the previous ``250d`` returned only ~171 NYSE sessions against a 205-row
    requirement, so ``_fetch_spy`` raised ``RegimeFetchError`` on EVERY call and
    the regime was permanently ``unknown``.

    Scale by 7/5 and keep a 60-day cushion to absorb holidays. This mirrors
    ``long_term._calendar_days_for_bars``, which fixed the identical bug for the
    long-horizon trend SMA; regime.py was simply never given the same treatment.
    """
    return -(-bars * 7 // 5) + 60


# 347 days -> ~238 NYSE sessions, a 33-session margin over the 205 required.
_FETCH_PERIOD_DAYS: int = _calendar_days_for_bars(_MIN_ROWS)

VALID_REGIMES: frozenset[str] = frozenset({"bull", "bear", "sideways", "unknown"})


class RegimeFetchError(RuntimeError):
    """Raised when SPY data can't be fetched or is malformed."""


@dataclass(frozen=True)
class RegimeSnapshot:
    date: str            # YYYY-MM-DD
    regime: str          # 'bull' | 'bear' | 'sideways'
    spy_close: float
    ema50: float
    ema200: float
    ema50_slope: float   # 5-day slope of the 50 EMA


# ────────────────────────────────────────────────────────────────────────────
# Pure classification
# ────────────────────────────────────────────────────────────────────────────


def classify(
    spy_close: float, ema50: float, ema200: float, ema50_slope: float
) -> str:
    """Classify the macro regime from current SPY indicator values.

    Rules:

    * **bull** — ``spy_close > ema200`` AND ``ema50 > ema200`` AND ``slope > 0``
    * **bear** — ``spy_close < ema200`` AND ``ema50 < ema200`` AND ``slope < 0``
    * **sideways** — anything else (mixed signals, boundary cases)

    Boundaries (e.g. ``slope == 0`` or ``ema50 == ema200``) classify as
    ``sideways`` because they fail the strict inequalities for both bull and
    bear.
    """
    if spy_close > ema200 and ema50 > ema200 and ema50_slope > 0:
        return "bull"
    if spy_close < ema200 and ema50 < ema200 and ema50_slope < 0:
        return "bear"
    return "sideways"


# ────────────────────────────────────────────────────────────────────────────
# Cache
# ────────────────────────────────────────────────────────────────────────────

# (cached_at_utc, snapshot). Cleared on force_refresh and reset by tests.
_memory_cache: tuple[datetime, RegimeSnapshot] | None = None


def _reset_cache_for_tests() -> None:
    """Test helper — clears the in-process cache. Disk cache untouched."""
    global _memory_cache
    _memory_cache = None


def _load_disk_cache() -> tuple[datetime, RegimeSnapshot] | None:
    """Read ``.regime_cache.json`` if it exists and parses cleanly."""
    if not _CACHE_PATH.exists():
        return None
    try:
        raw = _CACHE_PATH.read_text(encoding="utf-8")
        data: Any = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    cached_at_iso = data.get("cached_at")
    snap_dict = data.get("snapshot")
    if not isinstance(cached_at_iso, str) or not isinstance(snap_dict, dict):
        return None
    try:
        cached_at = datetime.fromisoformat(cached_at_iso)
        if cached_at.tzinfo is None:
            cached_at = cached_at.replace(tzinfo=UTC)
        snap = RegimeSnapshot(**snap_dict)
    except (TypeError, ValueError):
        return None
    return (cached_at, snap)


def _write_disk_cache(cached_at: datetime, snap: RegimeSnapshot) -> None:
    """Persist the latest snapshot. Failures are logged, never raised —
    a corrupt cache file shouldn't fail a live scan."""
    payload = {
        "cached_at": cached_at.isoformat(),
        "snapshot": asdict(snap),
    }
    try:
        _CACHE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as exc:
        print(f"  regime cache write error: {exc}", file=sys.stderr)


def last_cached_at() -> datetime | None:
    """Return the timestamp of the in-memory cached snapshot, or None.

    Used by the CLI to display "Snapshot age: X hours". Disk cache is also
    consulted so that a fresh process can still report age.
    """
    if _memory_cache is not None:
        return _memory_cache[0]
    disk = _load_disk_cache()
    return disk[0] if disk is not None else None


# ────────────────────────────────────────────────────────────────────────────
# Fetch
# ────────────────────────────────────────────────────────────────────────────


def _fetch_spy(period_days: int = _FETCH_PERIOD_DAYS) -> pd.DataFrame:
    """Pull SPY daily candles from yfinance. Raises :class:`RegimeFetchError`
    on any failure — network, rate limit, empty, malformed, or insufficient.
    """
    try:
        df = yf.download(
            _SPY_TICKER,
            period=f"{period_days}d",
            interval="1d",
            progress=False,
            auto_adjust=False,
        )
    except Exception as exc:  # noqa: BLE001 - yfinance can raise anything
        raise RegimeFetchError(f"yfinance error fetching SPY: {exc}") from exc

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise RegimeFetchError("yfinance returned empty SPY data")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    if "Close" not in df.columns:
        raise RegimeFetchError("SPY data missing 'Close' column")
    df = df.dropna(subset=["Close"])
    if len(df) < _MIN_ROWS:
        raise RegimeFetchError(
            f"insufficient SPY data: got {len(df)} rows, need >= {_MIN_ROWS}"
        )
    return df


def _snapshot_from_df(df: pd.DataFrame, as_of_index: int = -1) -> RegimeSnapshot:
    """Compute a :class:`RegimeSnapshot` from SPY OHLC data at the given row.

    ``as_of_index`` defaults to ``-1`` (most recent row). Negative indices are
    normalized to positive. Raises :class:`RegimeFetchError` if the index
    doesn't have enough prior data for the slope window.
    """
    idx = as_of_index if as_of_index >= 0 else len(df) + as_of_index
    if idx < _EMA_SLOPE_WINDOW or idx >= len(df):
        raise RegimeFetchError(
            f"insufficient data at index {idx}; need >= {_EMA_SLOPE_WINDOW} prior rows"
        )

    ema50_series = df["Close"].ewm(span=50, adjust=False).mean()
    ema200_series = df["Close"].ewm(span=200, adjust=False).mean()

    spy_close = float(df["Close"].iloc[idx])
    ema50_now = float(ema50_series.iloc[idx])
    ema200_now = float(ema200_series.iloc[idx])
    ema50_back = float(ema50_series.iloc[idx - _EMA_SLOPE_WINDOW])
    ema50_slope = (ema50_now - ema50_back) / _EMA_SLOPE_WINDOW

    raw_index = df.index[idx]
    date_obj = (
        raw_index.to_pydatetime()
        if hasattr(raw_index, "to_pydatetime")
        else raw_index
    )
    date_str = date_obj.strftime("%Y-%m-%d")

    return RegimeSnapshot(
        date=date_str,
        regime=classify(spy_close, ema50_now, ema200_now, ema50_slope),
        spy_close=spy_close,
        ema50=ema50_now,
        ema200=ema200_now,
        ema50_slope=ema50_slope,
    )


# ────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────


def get_current_regime(force_refresh: bool = False) -> RegimeSnapshot:
    """Return the current macro regime, cached for 24h unless ``force_refresh``.

    Raises :class:`RegimeFetchError` if SPY data is unavailable. Never returns
    None and never returns a stale snapshot beyond the TTL — callers (e.g. the
    scanner) catch the error and decide whether to proceed with
    ``regime='unknown'``.
    """
    global _memory_cache
    now = datetime.now(UTC)

    if not force_refresh:
        if _memory_cache is not None:
            cached_at, snap = _memory_cache
            if now - cached_at < _CACHE_TTL:
                return snap
        disk = _load_disk_cache()
        if disk is not None:
            cached_at, snap = disk
            if now - cached_at < _CACHE_TTL:
                _memory_cache = (cached_at, snap)
                return snap

    df = _fetch_spy()
    snap = _snapshot_from_df(df)
    _memory_cache = (now, snap)
    _write_disk_cache(now, snap)
    return snap


def snapshot_regime_for_date(target_date: str) -> RegimeSnapshot | None:
    """Historical regime for ``target_date`` (YYYY-MM-DD).

    Returns ``None`` when SPY data isn't available for that date (weekend,
    holiday, future, or pre-200-EMA period). Raises :class:`RegimeFetchError`
    only on full fetch failures — a missing date inside an otherwise-good
    response is not an error.
    """
    # Fetch enough history to cover the date plus the 200 EMA + slope window.
    # The historical backfill targets the 20 trades from April 2026; SPY's
    # period= parameter looks back from "now", so we ask for a wide enough
    # window to comfortably cover the requested date.
    df = _fetch_spy(period_days=_FETCH_PERIOD_DAYS + 365)

    try:
        target_ts = pd.Timestamp(target_date)
    except (TypeError, ValueError) as exc:
        raise RegimeFetchError(f"invalid target_date {target_date!r}: {exc}") from exc

    normalized = pd.to_datetime(df.index).normalize()
    # Strip any tz so naive target_ts compares cleanly.
    if normalized.tz is not None:
        normalized = normalized.tz_localize(None)
    matches = normalized == target_ts.normalize()
    if not bool(matches.any()):
        # Weekend, holiday, future date, or simply outside our fetch window.
        return None

    idx = int(matches.argmax())
    if idx < _MIN_ROWS_FOR_200_EMA or idx < _EMA_SLOPE_WINDOW:
        # Pre-IPO / insufficient prior data for a reliable 200 EMA.
        return None

    return _snapshot_from_df(df, as_of_index=idx)


# ────────────────────────────────────────────────────────────────────────────
# Backfill (Phase 2.1 one-shot)
# ────────────────────────────────────────────────────────────────────────────


def backfill_trade_regimes() -> dict[str, int]:
    """Backfill ``market_regime`` for closed trades that don't have one.

    For each candidate trade, look up the historical regime as of
    ``opened_at`` via :func:`snapshot_regime_for_date` and update the row.
    Any new dates touched are also persisted to ``regime_snapshots`` so
    subsequent ``regime history`` queries see them.

    Idempotent — a second run encounters zero candidate trades. Returns a
    summary dict for the CLI to display.
    """
    candidates = db.get_closed_trades_missing_regime()
    summary = {
        "trades_updated": 0,
        "dates_snapshotted": 0,
        "errors": 0,
        "skipped_no_data": 0,
    }

    # Cache per-date snapshots so 10 trades on the same day share one yfinance
    # slice and one DB write. (snapshot_regime_for_date fetches the full SPY
    # history each call — would be wasteful to repeat.)
    snapshot_by_date: dict[str, RegimeSnapshot] = {}
    captured_at = datetime.now(UTC)

    for trade in candidates:
        if trade.id is None:
            continue
        opened_date = trade.opened_at.strftime("%Y-%m-%d")

        snap: RegimeSnapshot | None
        if opened_date in snapshot_by_date:
            snap = snapshot_by_date[opened_date]
        else:
            try:
                snap = snapshot_regime_for_date(opened_date)
            except RegimeFetchError as exc:
                print(
                    f"  regime backfill error for {opened_date}: {exc}",
                    file=sys.stderr,
                )
                summary["errors"] += 1
                continue
            if snap is not None:
                snapshot_by_date[opened_date] = snap

        if snap is None:
            summary["skipped_no_data"] += 1
            continue

        # Persist the snapshot row (one per unique date) if not already there.
        if db.get_regime_snapshot(snap.date) is None:
            db.upsert_regime_snapshot(
                snapshot_date=snap.date,
                regime=snap.regime,
                spy_close=snap.spy_close,
                ema50=snap.ema50,
                ema200=snap.ema200,
                ema50_slope=snap.ema50_slope,
                captured_at=captured_at,
            )
            summary["dates_snapshotted"] += 1

        db.update_trade(trade.id, market_regime=snap.regime)
        summary["trades_updated"] += 1

    return summary
