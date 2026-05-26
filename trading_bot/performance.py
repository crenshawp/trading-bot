"""Performance attribution — pure SQL aggregations over signals + trades.

Aggregation rules (uniformly applied across every ``stats_*`` function):

* **"Closed" means** ``outcome IN ('win', 'loss', 'expired')``. Trades with
  ``outcome IS NULL`` or ``outcome = 'open'`` are excluded from every stat
  in this module.
* **``win_rate = wins / (wins + losses)``.** Expired trades do NOT count
  toward the win-rate denominator. If a slice has zero wins+losses (e.g.
  only expired trades), ``win_rate`` is ``None``, not 0.
* **``avg_pnl_pct`` / ``best_pnl_pct`` / ``worst_pnl_pct``** are computed
  across *all* closed trades in the slice, including expired ones. An
  expired trade with no available exit price (``pnl_pct IS NULL``) is
  silently ignored by AVG/MAX/MIN — it still counts toward ``total`` and
  the ``expired`` bucket.
* **``total``** is the count of closed trades in the slice. Open/null
  trades are not counted.
* **Empty slices** return ``None`` for every numeric stat — never 0.
  Returning 0 would be silently misleading ("0% win rate" reads
  identically to "no data yet").

All queries use parameter binding. The only dynamic clause shapes are
asset-class and date filters, threaded through with ``?`` placeholders.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

from trading_bot import db
from trading_bot.models import DailyPerf

_CLOSED_OUTCOMES = ("win", "loss", "expired")


@dataclass(frozen=True)
class PerfStats:
    """Aggregated performance for some slice of trades.

    ``label`` identifies the slice (e.g. ``"ema21_pullback"``, ``"META"``,
    ``"stock"``, ``"overall"``, ``"ema21_pullback / META"``).
    Numeric fields are ``None`` when ``total == 0`` — see module docstring.
    """
    label: str
    total: int
    wins: int
    losses: int
    expired: int
    win_rate: float | None
    avg_pnl_pct: float | None
    best_pnl_pct: float | None
    worst_pnl_pct: float | None


# ────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ────────────────────────────────────────────────────────────────────────────

# Each grouped stats query uses these aggregates so the row→PerfStats mapping
# stays uniform. ``total`` is wins+losses+expired (already filtered by the
# WHERE clause); kept as an explicit COUNT(*) so it survives any future
# rule changes.
_AGGREGATE_COLS = (
    "COUNT(*) AS total, "
    "SUM(CASE WHEN trades.outcome = 'win'     THEN 1 ELSE 0 END) AS wins, "
    "SUM(CASE WHEN trades.outcome = 'loss'    THEN 1 ELSE 0 END) AS losses, "
    "SUM(CASE WHEN trades.outcome = 'expired' THEN 1 ELSE 0 END) AS expired, "
    "AVG(trades.pnl_pct) AS avg_pnl, "
    "MAX(trades.pnl_pct) AS best_pnl, "
    "MIN(trades.pnl_pct) AS worst_pnl"
)


def _row_to_stats(label: str, row: sqlite3.Row | None) -> PerfStats:
    """Convert an aggregate row to a ``PerfStats``. Handles all-None edge cases."""
    if row is None or row["total"] in (None, 0):
        return PerfStats(
            label=label,
            total=0,
            wins=0,
            losses=0,
            expired=0,
            win_rate=None,
            avg_pnl_pct=None,
            best_pnl_pct=None,
            worst_pnl_pct=None,
        )
    wins   = int(row["wins"])
    losses = int(row["losses"])
    decided = wins + losses
    return PerfStats(
        label=label,
        total=int(row["total"]),
        wins=wins,
        losses=losses,
        expired=int(row["expired"]),
        win_rate=(wins / decided * 100.0) if decided > 0 else None,
        avg_pnl_pct=float(row["avg_pnl"]) if row["avg_pnl"] is not None else None,
        best_pnl_pct=float(row["best_pnl"]) if row["best_pnl"] is not None else None,
        worst_pnl_pct=float(row["worst_pnl"]) if row["worst_pnl"] is not None else None,
    )


# ────────────────────────────────────────────────────────────────────────────
# Public API — slice queries
# ────────────────────────────────────────────────────────────────────────────


def stats_overall(since: datetime | None = None) -> PerfStats:
    """Single aggregated ``PerfStats`` across all closed trades.

    If ``since`` is given, only trades with ``closed_at >= since`` count.
    """
    where = "WHERE trades.outcome IN ('win','loss','expired')"
    params: list[object] = []
    if since is not None:
        where += " AND trades.closed_at >= ?"
        params.append(since.isoformat())

    sql = f"SELECT {_AGGREGATE_COLS} FROM trades {where}"  # noqa: S608 - whitelisted fragments
    conn = db.get_connection()
    try:
        row = conn.execute(sql, params).fetchone()
    finally:
        conn.close()
    return _row_to_stats("overall", row)


def stats_recent(days: int = 30) -> PerfStats:
    """Convenience: ``stats_overall(since=now-days)``."""
    since = datetime.now(UTC) - timedelta(days=days)
    stats = stats_overall(since=since)
    # Rename the label so reports can distinguish recent-vs-overall.
    return PerfStats(
        label=f"recent ({days}d)",
        total=stats.total,
        wins=stats.wins,
        losses=stats.losses,
        expired=stats.expired,
        win_rate=stats.win_rate,
        avg_pnl_pct=stats.avg_pnl_pct,
        best_pnl_pct=stats.best_pnl_pct,
        worst_pnl_pct=stats.worst_pnl_pct,
    )


def stats_by_signal_type() -> list[PerfStats]:
    """One ``PerfStats`` per distinct signal_type, sorted by win_rate desc.

    Slices with all-expired (no wins, no losses) sort to the end because
    their ``win_rate`` is ``None``.
    """
    sql = (
        f"SELECT signals.signal_type AS slice, {_AGGREGATE_COLS} "
        "FROM trades "
        "JOIN signals ON signals.id = trades.signal_id "
        "WHERE trades.outcome IN ('win','loss','expired') "
        "GROUP BY signals.signal_type"
    )
    conn = db.get_connection()
    try:
        rows = conn.execute(sql).fetchall()
    finally:
        conn.close()
    result = [_row_to_stats(str(r["slice"]), r) for r in rows]
    # None win_rate sorts to the back; otherwise descending.
    result.sort(key=lambda s: (s.win_rate is None, -(s.win_rate or 0.0)))
    return result


def stats_by_ticker(asset_class: str | None = None) -> list[PerfStats]:
    """One ``PerfStats`` per distinct ticker, sorted by avg_pnl_pct desc.

    If ``asset_class`` is provided ("stock" or "crypto"), only that class
    is included.
    """
    where = "WHERE trades.outcome IN ('win','loss','expired')"
    params: list[object] = []
    if asset_class is not None:
        if asset_class not in ("stock", "crypto"):
            raise ValueError(f"asset_class must be 'stock' or 'crypto', got {asset_class!r}")
        where += " AND signals.asset_class = ?"
        params.append(asset_class)

    sql = (
        f"SELECT signals.ticker AS slice, {_AGGREGATE_COLS} "
        "FROM trades "
        "JOIN signals ON signals.id = trades.signal_id "
        f"{where} "
        "GROUP BY signals.ticker"
    )  # noqa: S608 - whitelisted fragments
    conn = db.get_connection()
    try:
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()
    result = [_row_to_stats(str(r["slice"]), r) for r in rows]
    result.sort(key=lambda s: (s.avg_pnl_pct is None, -(s.avg_pnl_pct or 0.0)))
    return result


def stats_by_asset_class() -> list[PerfStats]:
    """One ``PerfStats`` per asset class. Up to two rows: 'stock', 'crypto'."""
    sql = (
        f"SELECT signals.asset_class AS slice, {_AGGREGATE_COLS} "
        "FROM trades "
        "JOIN signals ON signals.id = trades.signal_id "
        "WHERE trades.outcome IN ('win','loss','expired') "
        "GROUP BY signals.asset_class "
        "ORDER BY signals.asset_class"
    )
    conn = db.get_connection()
    try:
        rows = conn.execute(sql).fetchall()
    finally:
        conn.close()
    return [_row_to_stats(str(r["slice"]), r) for r in rows]


def stats_by_signal_type_and_ticker(min_trades: int = 3) -> list[PerfStats]:
    """Cross-section ``signal_type × ticker``, sorted by avg_pnl_pct desc.

    Slices with fewer than ``min_trades`` closed trades are excluded — too
    few samples for the average to be informative. ``min_trades`` defaults
    to 3 per the spec; lowered values are useful for ad-hoc exploration.
    """
    sql = (
        f"SELECT signals.signal_type AS sig_type, signals.ticker AS ticker, {_AGGREGATE_COLS} "
        "FROM trades "
        "JOIN signals ON signals.id = trades.signal_id "
        "WHERE trades.outcome IN ('win','loss','expired') "
        "GROUP BY signals.signal_type, signals.ticker "
        "HAVING COUNT(*) >= ?"
    )
    conn = db.get_connection()
    try:
        rows = conn.execute(sql, (min_trades,)).fetchall()
    finally:
        conn.close()
    result = [
        _row_to_stats(f"{r['sig_type']} / {r['ticker']}", r) for r in rows
    ]
    result.sort(key=lambda s: (s.avg_pnl_pct is None, -(s.avg_pnl_pct or 0.0)))
    return result


# ────────────────────────────────────────────────────────────────────────────
# Daily performance materialization
# ────────────────────────────────────────────────────────────────────────────


def update_daily_performance(target_date: date | None = None) -> DailyPerf:
    """Compute and upsert a ``DailyPerf`` row for ``target_date``.

    Defaults to yesterday in UTC. Counts are produced via SQLite's
    ``date()`` function on the stored ISO timestamps — for the 20
    historical CSV-migrated signals (naive ET) this effectively groups by
    ET-local date; for new signals (tz-aware UTC) it groups by UTC date.
    Close enough for daily reporting.
    """
    target = target_date if target_date is not None else (datetime.now(UTC).date() - timedelta(days=1))
    target_iso = target.isoformat()

    conn = db.get_connection()
    try:
        signals_fired = conn.execute(
            "SELECT COUNT(*) AS c FROM signals WHERE date(timestamp) = ?",
            (target_iso,),
        ).fetchone()["c"]

        trades_opened = conn.execute(
            "SELECT COUNT(*) AS c FROM trades WHERE date(opened_at) = ?",
            (target_iso,),
        ).fetchone()["c"]

        closed_row = conn.execute(
            "SELECT "
            "  COUNT(*) AS closed, "
            "  SUM(CASE WHEN outcome = 'win'  THEN 1 ELSE 0 END) AS wins, "
            "  SUM(CASE WHEN outcome = 'loss' THEN 1 ELSE 0 END) AS losses, "
            "  SUM(pnl_pct) AS total_pnl "
            "FROM trades "
            "WHERE date(closed_at) = ? AND outcome IN ('win','loss','expired')",
            (target_iso,),
        ).fetchone()
    finally:
        conn.close()

    trades_closed = int(closed_row["closed"]) if closed_row["closed"] is not None else 0
    wins   = int(closed_row["wins"])   if closed_row["wins"]   is not None else 0
    losses = int(closed_row["losses"]) if closed_row["losses"] is not None else 0
    decided = wins + losses
    win_rate = (wins / decided * 100.0) if decided > 0 else None
    total_pnl_pct = (
        float(closed_row["total_pnl"]) if closed_row["total_pnl"] is not None else None
    )

    perf = DailyPerf(
        date=target_iso,
        signals_fired=int(signals_fired),
        trades_opened=int(trades_opened),
        trades_closed=trades_closed,
        wins=wins,
        losses=losses,
        win_rate=win_rate,
        total_pnl_pct=total_pnl_pct,
    )
    db.upsert_daily_performance(perf)
    return perf


def _distinct_activity_dates() -> list[date]:
    """Every distinct YYYY-MM-DD that appears as a signal/trade timestamp."""
    sql = (
        "SELECT DISTINCT day FROM ("
        "  SELECT date(timestamp) AS day FROM signals "
        "  UNION "
        "  SELECT date(opened_at) AS day FROM trades "
        "  UNION "
        "  SELECT date(closed_at) AS day FROM trades WHERE closed_at IS NOT NULL"
        ") WHERE day IS NOT NULL "
        "ORDER BY day ASC"
    )
    conn = db.get_connection()
    try:
        rows = conn.execute(sql).fetchall()
    finally:
        conn.close()
    return [date.fromisoformat(str(r["day"])) for r in rows]


def backfill_daily_performance() -> int:
    """Compute ``DailyPerf`` for every distinct activity date.

    Idempotent — re-running just overwrites the same upsert rows. Intended
    as a one-shot to seed the table; afterward the scheduled job at 00:30
    UTC keeps it current.
    """
    dates = _distinct_activity_dates()
    for d in dates:
        update_daily_performance(d)
    return len(dates)
