"""VIX context — volatility bands layered on top of macro regime.

Phase 2.2 (Market Intelligence). Mirrors :mod:`trading_bot.regime` deliberately:
isolated pure classifier, fetch + cache via yfinance, scheduled daily
snapshot, and a backfill subcommand. Regime tells us direction; VIX tells
us how violently the market is moving. A bull at VIX 12 and a bull at VIX
38 are entirely different trading environments.

Phase 2.3 (regime-aware consolidation) will eventually merge regime + VIX
into a unified market-context score. The modules stay separate here so
each axis can evolve independently.

Caching:

* In-process cache for the lifetime of the interpreter.
* Disk cache at ``.vix_cache.json`` next to the project root.
* TTL is **1 hour** — VIX moves faster than the 50/200 EMA, and crypto
  scans run hourly so each scan grabs fresh data.

yfinance failures (network, rate limit, malformed response) raise
:class:`VixFetchError`. The scanner catches and tags the trade with
``vix_band='unknown'`` / ``vix_level=None`` — signal capture trumps
context tagging.
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

_CACHE_PATH: Path = config.ROOT_DIR / ".vix_cache.json"
_CACHE_TTL: timedelta = timedelta(hours=1)
_VIX_TICKER: str = "^VIX"
_FETCH_PERIOD_DAYS: int = 30  # plenty for "today's close + recent history"

# Band thresholds (upper-bound exclusive except for 'extreme'). Boundaries go
# UP a band — 20.0 is elevated, not low; 30.0 is high; 40.0 is extreme.
_LOW_UPPER = 20.0
_ELEVATED_UPPER = 30.0
_HIGH_UPPER = 40.0

VALID_VIX_BANDS: frozenset[str] = frozenset(
    {"low", "elevated", "high", "extreme", "unknown"}
)


class VixFetchError(RuntimeError):
    """Raised when VIX data can't be fetched or is malformed."""


@dataclass(frozen=True)
class VixSnapshot:
    date: str            # YYYY-MM-DD
    vix_level: float
    vix_band: str        # 'low' | 'elevated' | 'high' | 'extreme'
    captured_at: str     # ISO timestamp


# ────────────────────────────────────────────────────────────────────────────
# Pure classification
# ────────────────────────────────────────────────────────────────────────────


def classify(vix_level: float) -> str:
    """Classify a VIX value into a volatility band.

    Bands:

    * **low** — ``< 20``
    * **elevated** — ``20 <= x < 30``
    * **high** — ``30 <= x < 40``
    * **extreme** — ``>= 40``

    Boundary values go to the higher band (matches standard market
    convention — VIX 20.0 is "starting to spike," not "still quiet").
    Negative input raises ``ValueError`` as a sanity guard.
    """
    if vix_level < 0:
        raise ValueError(f"vix_level must be >= 0, got {vix_level}")
    if vix_level < _LOW_UPPER:
        return "low"
    if vix_level < _ELEVATED_UPPER:
        return "elevated"
    if vix_level < _HIGH_UPPER:
        return "high"
    return "extreme"


# ────────────────────────────────────────────────────────────────────────────
# Cache
# ────────────────────────────────────────────────────────────────────────────

_memory_cache: tuple[datetime, VixSnapshot] | None = None


def _reset_cache_for_tests() -> None:
    """Test helper — clears the in-process cache. Disk cache untouched."""
    global _memory_cache
    _memory_cache = None


def _load_disk_cache() -> tuple[datetime, VixSnapshot] | None:
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
        snap = VixSnapshot(**snap_dict)
    except (TypeError, ValueError):
        return None
    return (cached_at, snap)


def _write_disk_cache(cached_at: datetime, snap: VixSnapshot) -> None:
    payload = {
        "cached_at": cached_at.isoformat(),
        "snapshot": asdict(snap),
    }
    try:
        _CACHE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as exc:
        print(f"  vix cache write error: {exc}", file=sys.stderr)


def last_cached_at() -> datetime | None:
    """Return the timestamp of the cached snapshot, or None. CLI uses this
    to show 'Snapshot age: X minutes'."""
    if _memory_cache is not None:
        return _memory_cache[0]
    disk = _load_disk_cache()
    return disk[0] if disk is not None else None


# ────────────────────────────────────────────────────────────────────────────
# Fetch
# ────────────────────────────────────────────────────────────────────────────


def _fetch_vix(period_days: int = _FETCH_PERIOD_DAYS) -> pd.DataFrame:
    """Pull ^VIX daily candles. Raises :class:`VixFetchError` on any failure."""
    try:
        df = yf.download(
            _VIX_TICKER,
            period=f"{period_days}d",
            interval="1d",
            progress=False,
            auto_adjust=False,
        )
    except Exception as exc:  # noqa: BLE001 - yfinance can raise anything
        raise VixFetchError(f"yfinance error fetching VIX: {exc}") from exc

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise VixFetchError("yfinance returned empty VIX data")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    if "Close" not in df.columns:
        raise VixFetchError("VIX data missing 'Close' column")
    df = df.dropna(subset=["Close"])
    if df.empty:
        raise VixFetchError("VIX data had no usable Close rows after cleanup")
    return df


def _snapshot_from_df(
    df: pd.DataFrame, as_of_index: int = -1, captured_at: datetime | None = None,
) -> VixSnapshot:
    """Build a :class:`VixSnapshot` from the row at ``as_of_index`` (-1 = latest)."""
    idx = as_of_index if as_of_index >= 0 else len(df) + as_of_index
    if idx < 0 or idx >= len(df):
        raise VixFetchError(f"as_of_index {as_of_index} out of range")

    vix_level = float(df["Close"].iloc[idx])
    raw_index = df.index[idx]
    date_obj = (
        raw_index.to_pydatetime()
        if hasattr(raw_index, "to_pydatetime")
        else raw_index
    )
    date_str = date_obj.strftime("%Y-%m-%d")
    captured = captured_at if captured_at is not None else datetime.now(UTC)

    return VixSnapshot(
        date=date_str,
        vix_level=vix_level,
        vix_band=classify(vix_level),
        captured_at=captured.isoformat(),
    )


# ────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────


def get_current_vix(force_refresh: bool = False) -> VixSnapshot:
    """Return the current VIX level + band, cached 1h unless ``force_refresh``.

    Raises :class:`VixFetchError` on yfinance failure. Never returns None and
    never returns a stale snapshot beyond TTL — callers (scanner) decide
    whether to fall back to ``vix_band='unknown'``.
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

    df = _fetch_vix()
    snap = _snapshot_from_df(df, captured_at=now)
    _memory_cache = (now, snap)
    _write_disk_cache(now, snap)
    return snap


def snapshot_vix_for_date(target_date: str) -> VixSnapshot | None:
    """Historical VIX for ``target_date`` (YYYY-MM-DD).

    Returns ``None`` for weekends, holidays, or any date outside the fetch
    window where VIX simply has no row. Raises :class:`VixFetchError` only on
    full fetch failures — a missing date inside an otherwise-good response is
    not an error.
    """
    # Pull more history for backfill — the 20 closed trades reach back to
    # April 2026. 250 calendar days covers that comfortably.
    df = _fetch_vix(period_days=250)

    try:
        target_ts = pd.Timestamp(target_date)
    except (TypeError, ValueError) as exc:
        raise VixFetchError(f"invalid target_date {target_date!r}: {exc}") from exc

    normalized = pd.to_datetime(df.index).normalize()
    if normalized.tz is not None:
        normalized = normalized.tz_localize(None)
    matches = normalized == target_ts.normalize()
    if not bool(matches.any()):
        return None

    idx = int(matches.argmax())
    return _snapshot_from_df(df, as_of_index=idx)


# ────────────────────────────────────────────────────────────────────────────
# Backfill (Phase 2.2 one-shot)
# ────────────────────────────────────────────────────────────────────────────


def backfill_trade_vix() -> dict[str, int]:
    """Backfill ``vix_level`` + ``vix_band`` for closed trades missing them.

    Independent of Phase 2.1's regime backfill — running one does not affect
    the other. Idempotent. Returns a summary dict for the CLI to display.
    """
    candidates = db.get_closed_trades_missing_vix()
    summary = {
        "trades_updated": 0,
        "dates_snapshotted": 0,
        "errors": 0,
        "skipped_no_data": 0,
    }

    snapshot_by_date: dict[str, VixSnapshot] = {}
    captured_at = datetime.now(UTC)

    for trade in candidates:
        if trade.id is None:
            continue
        opened_date = trade.opened_at.strftime("%Y-%m-%d")

        snap: VixSnapshot | None
        if opened_date in snapshot_by_date:
            snap = snapshot_by_date[opened_date]
        else:
            try:
                snap = snapshot_vix_for_date(opened_date)
            except VixFetchError as exc:
                print(
                    f"  vix backfill error for {opened_date}: {exc}",
                    file=sys.stderr,
                )
                summary["errors"] += 1
                continue
            if snap is not None:
                snapshot_by_date[opened_date] = snap

        if snap is None:
            summary["skipped_no_data"] += 1
            continue

        if db.get_vix_snapshot(snap.date) is None:
            db.upsert_vix_snapshot(
                snapshot_date=snap.date,
                vix_level=snap.vix_level,
                vix_band=snap.vix_band,
                captured_at=captured_at,
            )
            summary["dates_snapshotted"] += 1

        db.update_trade(
            trade.id, vix_level=snap.vix_level, vix_band=snap.vix_band,
        )
        summary["trades_updated"] += 1

    return summary
