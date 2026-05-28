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
from zoneinfo import ZoneInfo

from trading_bot import db
from trading_bot.models import DailyPerf

_CLOSED_OUTCOMES = ("win", "loss", "expired")

# Display ordering for regime slices: bull → sideways → bear → unknown.
# bull/sideways/bear is the natural macro continuum; unknown trails because
# it's a fallback bucket, not a regime in its own right.
_REGIME_ORDER = ("bull", "sideways", "bear", "unknown")

# Display ordering for VIX bands: low → elevated → high → extreme → unknown.
# Same convention — quietest first, fallback bucket last.
_VIX_ORDER = ("low", "elevated", "high", "extreme", "unknown")


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


@dataclass(frozen=True)
class RegimeBreakdown:
    """A signal_type or ticker sliced across macro regimes.

    ``by_regime`` maps regime ('bull' / 'sideways' / 'bear' / 'unknown') to
    that slice's :class:`PerfStats`. Regimes with no closed trades for this
    key are simply absent from the dict — callers render those as ``-``.
    ``overall`` is the aggregate across all regimes for the same key.
    """
    label: str
    by_regime: dict[str, PerfStats]
    overall: PerfStats


@dataclass(frozen=True)
class VixBreakdown:
    """A signal_type or ticker sliced across VIX bands (Phase 2.2).

    Same shape as :class:`RegimeBreakdown` but on the volatility axis.
    ``by_vix`` keys are 'low' / 'elevated' / 'high' / 'extreme' / 'unknown'.
    Bands with no closed trades are absent from the dict.
    """
    label: str
    by_vix: dict[str, PerfStats]
    overall: PerfStats


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


def _min_context_filter(
    min_context: int, table_alias: str = "trades",
) -> tuple[str, tuple[object, ...]]:
    """Build the SQL fragment that filters by ``context_score >= N``.

    Returns ``("", ())`` for ``min_context <= 0`` so callers can splat the
    result without conditional logic. Strictly greater-or-equal — rows
    where ``context_score IS NULL`` are EXCLUDED when the filter is active
    (a NULL score is not "at least 1").
    """
    if min_context <= 0:
        return ("", ())
    return (f"AND {table_alias}.context_score >= ?", (min_context,))


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


def stats_by_signal_type(min_context: int = 0) -> list[PerfStats]:
    """One ``PerfStats`` per distinct signal_type, sorted by win_rate desc.

    Slices with all-expired (no wins, no losses) sort to the end because
    their ``win_rate`` is ``None``. ``min_context`` (Phase 2.3) filters
    out trades whose context_score is below ``N``.
    """
    ctx_clause, ctx_params = _min_context_filter(min_context)
    sql = (
        f"SELECT signals.signal_type AS slice, {_AGGREGATE_COLS} "
        "FROM trades "
        "JOIN signals ON signals.id = trades.signal_id "
        f"WHERE trades.outcome IN ('win','loss','expired') {ctx_clause} "
        "GROUP BY signals.signal_type"
    )
    conn = db.get_connection()
    try:
        rows = conn.execute(sql, ctx_params).fetchall()
    finally:
        conn.close()
    result = [_row_to_stats(str(r["slice"]), r) for r in rows]
    result.sort(key=lambda s: (s.win_rate is None, -(s.win_rate or 0.0)))
    return result


def stats_by_ticker(
    asset_class: str | None = None, min_context: int = 0,
) -> list[PerfStats]:
    """One ``PerfStats`` per distinct ticker, sorted by avg_pnl_pct desc.

    If ``asset_class`` is provided ("stock" or "crypto"), only that class
    is included. ``min_context`` (Phase 2.3) excludes trades whose
    context_score is below ``N``.
    """
    where = "WHERE trades.outcome IN ('win','loss','expired')"
    params: list[object] = []
    if asset_class is not None:
        if asset_class not in ("stock", "crypto"):
            raise ValueError(f"asset_class must be 'stock' or 'crypto', got {asset_class!r}")
        where += " AND signals.asset_class = ?"
        params.append(asset_class)
    ctx_clause, ctx_params = _min_context_filter(min_context)
    if ctx_clause:
        where += f" {ctx_clause}"
        params.extend(ctx_params)

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
# Regime-aware slices (Phase 2.1)
# ────────────────────────────────────────────────────────────────────────────


def stats_by_regime(min_context: int = 0) -> list[PerfStats]:
    """One :class:`PerfStats` per macro regime present in closed trades.

    Ordered bull → sideways → bear → unknown. NULL ``market_regime`` is
    folded into the ``unknown`` bucket. ``min_context`` excludes trades
    whose context_score is below ``N``.
    """
    ctx_clause, ctx_params = _min_context_filter(min_context)
    sql = (
        f"SELECT COALESCE(trades.market_regime, 'unknown') AS slice, "
        f"{_AGGREGATE_COLS} "
        "FROM trades "
        "JOIN signals ON signals.id = trades.signal_id "
        f"WHERE trades.outcome IN ('win','loss','expired') {ctx_clause} "
        "GROUP BY COALESCE(trades.market_regime, 'unknown')"
    )
    conn = db.get_connection()
    try:
        rows = conn.execute(sql, ctx_params).fetchall()
    finally:
        conn.close()
    result = [_row_to_stats(str(r["slice"]), r) for r in rows]
    order_map = {r: i for i, r in enumerate(_REGIME_ORDER)}
    result.sort(key=lambda s: order_map.get(s.label, len(_REGIME_ORDER)))
    return result


def _stats_grouped_by_regime(
    group_col: str,
    *,
    extra_where: str = "",
    extra_params: tuple[object, ...] = (),
) -> list[RegimeBreakdown]:
    """Build a regime-cross-tab for an arbitrary signals column.

    Two queries: one grouped by (group_col, regime) and one grouped by
    group_col alone for the ``overall`` column. Results are merged in
    Python so an empty regime slice doesn't pollute the overall stat.
    ``group_col`` MUST be a whitelisted column name on ``signals``.
    """
    if group_col not in {"signal_type", "ticker"}:
        raise ValueError(
            f"group_col must be 'signal_type' or 'ticker', got {group_col!r}"
        )

    where = "WHERE trades.outcome IN ('win','loss','expired')"
    if extra_where:
        where += f" AND {extra_where}"

    by_regime_sql = (  # noqa: S608 - group_col and where fragments are whitelisted
        f"SELECT signals.{group_col} AS slice, "
        f"COALESCE(trades.market_regime, 'unknown') AS regime, "
        f"{_AGGREGATE_COLS} "
        "FROM trades "
        "JOIN signals ON signals.id = trades.signal_id "
        f"{where} "
        f"GROUP BY signals.{group_col}, COALESCE(trades.market_regime, 'unknown')"
    )
    overall_sql = (  # noqa: S608 - group_col and where fragments are whitelisted
        f"SELECT signals.{group_col} AS slice, {_AGGREGATE_COLS} "
        "FROM trades "
        "JOIN signals ON signals.id = trades.signal_id "
        f"{where} "
        f"GROUP BY signals.{group_col}"
    )

    conn = db.get_connection()
    try:
        regime_rows = conn.execute(by_regime_sql, extra_params).fetchall()
        overall_rows = conn.execute(overall_sql, extra_params).fetchall()
    finally:
        conn.close()

    by_label: dict[str, dict[str, PerfStats]] = {}
    for row in regime_rows:
        label = str(row["slice"])
        regime_key = str(row["regime"])
        by_label.setdefault(label, {})[regime_key] = _row_to_stats(
            f"{label} / {regime_key}", row
        )

    overall_by_label: dict[str, PerfStats] = {
        str(r["slice"]): _row_to_stats(str(r["slice"]), r) for r in overall_rows
    }

    breakdowns: list[RegimeBreakdown] = []
    for label, overall in overall_by_label.items():
        breakdowns.append(
            RegimeBreakdown(
                label=label,
                by_regime=by_label.get(label, {}),
                overall=overall,
            )
        )
    # Sort by overall avg_pnl_pct desc, matching stats_by_ticker / by_signal.
    breakdowns.sort(
        key=lambda b: (b.overall.avg_pnl_pct is None, -(b.overall.avg_pnl_pct or 0.0))
    )
    return breakdowns


def stats_by_signal_type_with_regime(min_context: int = 0) -> list[RegimeBreakdown]:
    """``stats_by_signal_type`` cross-tabbed by macro regime."""
    ctx_clause, ctx_params = _min_context_filter(min_context)
    return _stats_grouped_by_regime(
        "signal_type", extra_where=ctx_clause.removeprefix("AND ").strip(),
        extra_params=ctx_params,
    ) if ctx_clause else _stats_grouped_by_regime("signal_type")


def stats_by_ticker_with_regime(
    asset_class: str | None = None, min_context: int = 0,
) -> list[RegimeBreakdown]:
    """``stats_by_ticker`` cross-tabbed by macro regime."""
    if asset_class is not None and asset_class not in ("stock", "crypto"):
        raise ValueError(
            f"asset_class must be 'stock' or 'crypto', got {asset_class!r}"
        )
    where_parts: list[str] = []
    params: list[object] = []
    if asset_class is not None:
        where_parts.append("signals.asset_class = ?")
        params.append(asset_class)
    ctx_clause, ctx_params = _min_context_filter(min_context)
    if ctx_clause:
        where_parts.append(ctx_clause.removeprefix("AND ").strip())
        params.extend(ctx_params)
    extra_where = " AND ".join(where_parts)
    return _stats_grouped_by_regime(
        "ticker", extra_where=extra_where, extra_params=tuple(params),
    )


# ────────────────────────────────────────────────────────────────────────────
# VIX-aware slices (Phase 2.2)
# ────────────────────────────────────────────────────────────────────────────


def stats_by_vix_band(min_context: int = 0) -> list[PerfStats]:
    """One :class:`PerfStats` per VIX band present in closed trades.

    Ordered low → elevated → high → extreme → unknown. ``min_context``
    excludes trades whose context_score is below ``N``.
    """
    ctx_clause, ctx_params = _min_context_filter(min_context)
    sql = (
        f"SELECT COALESCE(trades.vix_band, 'unknown') AS slice, "
        f"{_AGGREGATE_COLS} "
        "FROM trades "
        "JOIN signals ON signals.id = trades.signal_id "
        f"WHERE trades.outcome IN ('win','loss','expired') {ctx_clause} "
        "GROUP BY COALESCE(trades.vix_band, 'unknown')"
    )
    conn = db.get_connection()
    try:
        rows = conn.execute(sql, ctx_params).fetchall()
    finally:
        conn.close()
    result = [_row_to_stats(str(r["slice"]), r) for r in rows]
    order_map = {b: i for i, b in enumerate(_VIX_ORDER)}
    result.sort(key=lambda s: order_map.get(s.label, len(_VIX_ORDER)))
    return result


def _stats_grouped_by_vix(
    group_col: str,
    *,
    extra_where: str = "",
    extra_params: tuple[object, ...] = (),
) -> list[VixBreakdown]:
    """Parallel of ``_stats_grouped_by_regime`` for the VIX axis."""
    if group_col not in {"signal_type", "ticker"}:
        raise ValueError(
            f"group_col must be 'signal_type' or 'ticker', got {group_col!r}"
        )

    where = "WHERE trades.outcome IN ('win','loss','expired')"
    if extra_where:
        where += f" AND {extra_where}"

    by_vix_sql = (  # noqa: S608 - group_col and where fragments are whitelisted
        f"SELECT signals.{group_col} AS slice, "
        f"COALESCE(trades.vix_band, 'unknown') AS vix_band, "
        f"{_AGGREGATE_COLS} "
        "FROM trades "
        "JOIN signals ON signals.id = trades.signal_id "
        f"{where} "
        f"GROUP BY signals.{group_col}, COALESCE(trades.vix_band, 'unknown')"
    )
    overall_sql = (  # noqa: S608 - group_col and where fragments are whitelisted
        f"SELECT signals.{group_col} AS slice, {_AGGREGATE_COLS} "
        "FROM trades "
        "JOIN signals ON signals.id = trades.signal_id "
        f"{where} "
        f"GROUP BY signals.{group_col}"
    )

    conn = db.get_connection()
    try:
        axis_rows = conn.execute(by_vix_sql, extra_params).fetchall()
        overall_rows = conn.execute(overall_sql, extra_params).fetchall()
    finally:
        conn.close()

    by_label: dict[str, dict[str, PerfStats]] = {}
    for row in axis_rows:
        label = str(row["slice"])
        vix_key = str(row["vix_band"])
        by_label.setdefault(label, {})[vix_key] = _row_to_stats(
            f"{label} / {vix_key}", row
        )

    overall_by_label: dict[str, PerfStats] = {
        str(r["slice"]): _row_to_stats(str(r["slice"]), r) for r in overall_rows
    }

    breakdowns: list[VixBreakdown] = []
    for label, overall in overall_by_label.items():
        breakdowns.append(
            VixBreakdown(
                label=label,
                by_vix=by_label.get(label, {}),
                overall=overall,
            )
        )
    breakdowns.sort(
        key=lambda b: (b.overall.avg_pnl_pct is None, -(b.overall.avg_pnl_pct or 0.0))
    )
    return breakdowns


def stats_by_signal_type_with_vix(min_context: int = 0) -> list[VixBreakdown]:
    """``stats_by_signal_type`` cross-tabbed by VIX band."""
    ctx_clause, ctx_params = _min_context_filter(min_context)
    extra_where = ctx_clause.removeprefix("AND ").strip() if ctx_clause else ""
    return _stats_grouped_by_vix(
        "signal_type", extra_where=extra_where, extra_params=ctx_params,
    )


def stats_by_ticker_with_vix(
    asset_class: str | None = None, min_context: int = 0,
) -> list[VixBreakdown]:
    """``stats_by_ticker`` cross-tabbed by VIX band."""
    if asset_class is not None and asset_class not in ("stock", "crypto"):
        raise ValueError(
            f"asset_class must be 'stock' or 'crypto', got {asset_class!r}"
        )
    where_parts: list[str] = []
    params: list[object] = []
    if asset_class is not None:
        where_parts.append("signals.asset_class = ?")
        params.append(asset_class)
    ctx_clause, ctx_params = _min_context_filter(min_context)
    if ctx_clause:
        where_parts.append(ctx_clause.removeprefix("AND ").strip())
        params.extend(ctx_params)
    return _stats_grouped_by_vix(
        "ticker", extra_where=" AND ".join(where_parts),
        extra_params=tuple(params),
    )


# ────────────────────────────────────────────────────────────────────────────
# Prediction aggregations (Phase 2.2b)
# ────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PredictionStats:
    """Aggregated prediction performance. Pushes are excluded from the
    accuracy denominator (same rule as expired trades for win_rate)."""
    label: str
    total: int
    correct: int
    incorrect: int
    push: int
    unresolved: int
    accuracy: float | None  # None when correct+incorrect == 0


def _pred_stats_from_row(label: str, row: sqlite3.Row | None) -> PredictionStats:
    if row is None or row["total"] in (None, 0):
        return PredictionStats(
            label=label, total=0,
            correct=0, incorrect=0, push=0, unresolved=0,
            accuracy=None,
        )
    correct = int(row["correct"] or 0)
    incorrect = int(row["incorrect"] or 0)
    push = int(row["push"] or 0)
    unresolved = int(row["unresolved"] or 0)
    decided = correct + incorrect
    accuracy = (correct / decided * 100.0) if decided > 0 else None
    return PredictionStats(
        label=label, total=int(row["total"]),
        correct=correct, incorrect=incorrect, push=push, unresolved=unresolved,
        accuracy=accuracy,
    )


_PRED_AGG = (
    "COUNT(*) AS total, "
    "SUM(CASE WHEN outcome = 'correct'   THEN 1 ELSE 0 END) AS correct, "
    "SUM(CASE WHEN outcome = 'incorrect' THEN 1 ELSE 0 END) AS incorrect, "
    "SUM(CASE WHEN outcome = 'push'      THEN 1 ELSE 0 END) AS push, "
    "SUM(CASE WHEN outcome IS NULL       THEN 1 ELSE 0 END) AS unresolved"
)


def _pred_where(min_context: int) -> tuple[str, tuple[object, ...]]:
    """Build the prediction WHERE/AND fragment for the min_context filter."""
    if min_context <= 0:
        return ("", ())
    return ("WHERE context_score >= ?", (min_context,))


def stats_predictions_overall(min_context: int = 0) -> PredictionStats:
    where, params = _pred_where(min_context)
    sql = f"SELECT {_PRED_AGG} FROM predictions {where}"  # noqa: S608 - whitelist
    conn = db.get_connection()
    try:
        row = conn.execute(sql, params).fetchone()
    finally:
        conn.close()
    return _pred_stats_from_row("TOTAL", row)


def stats_predictions_by_ticker(min_context: int = 0) -> list[PredictionStats]:
    where, params = _pred_where(min_context)
    sql = (
        f"SELECT ticker AS slice, {_PRED_AGG} "
        f"FROM predictions {where} GROUP BY ticker ORDER BY ticker"
    )
    conn = db.get_connection()
    try:
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()
    return [_pred_stats_from_row(str(r["slice"]), r) for r in rows]


def stats_predictions_by_regime(min_context: int = 0) -> list[PredictionStats]:
    """One row per regime present in predictions. Ordered bull/sideways/bear/unknown."""
    where, params = _pred_where(min_context)
    sql = (
        f"SELECT COALESCE(market_regime, 'unknown') AS slice, {_PRED_AGG} "
        f"FROM predictions {where} GROUP BY COALESCE(market_regime, 'unknown')"
    )
    conn = db.get_connection()
    try:
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()
    result = [_pred_stats_from_row(str(r["slice"]), r) for r in rows]
    order = {r: i for i, r in enumerate(_REGIME_ORDER)}
    result.sort(key=lambda s: order.get(s.label, len(_REGIME_ORDER)))
    return result


def stats_predictions_by_vix(min_context: int = 0) -> list[PredictionStats]:
    """One row per VIX band present in predictions. Ordered low→unknown."""
    where, params = _pred_where(min_context)
    sql = (
        f"SELECT COALESCE(vix_band, 'unknown') AS slice, {_PRED_AGG} "
        f"FROM predictions {where} GROUP BY COALESCE(vix_band, 'unknown')"
    )
    conn = db.get_connection()
    try:
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()
    result = [_pred_stats_from_row(str(r["slice"]), r) for r in rows]
    order = {b: i for i, b in enumerate(_VIX_ORDER)}
    result.sort(key=lambda s: order.get(s.label, len(_VIX_ORDER)))
    return result


def stats_predictions_by_hour(min_context: int = 0) -> list[PredictionStats]:
    """One row per hour-of-day (in ET) that has at least one prediction.

    The hour bucket is derived from ``created_at`` converted to ET. We do the
    conversion in Python (SQLite TZ functions are limited) — pull rows,
    group in-memory, sort by hour ascending. ``min_context`` filters in SQL.
    """
    where, params = _pred_where(min_context)
    sql = f"SELECT * FROM predictions {where}"  # noqa: S608 - whitelisted fragment
    conn = db.get_connection()
    try:
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()

    et = ZoneInfo("America/New_York")
    buckets: dict[int, dict[str, int]] = {}
    for row in rows:
        try:
            created = datetime.fromisoformat(str(row["created_at"]))
        except ValueError:
            # created_at is always written as ISO by us, so this is defensive
            # against corrupt data — but skipping a row from the histogram
            # silently would understate counts. Log which row was dropped.
            import sys
            print(
                f"  stats_predictions_by_hour: skipping prediction "
                f"id={row['id']} with unparseable created_at "
                f"{row['created_at']!r}",
                file=sys.stderr,
            )
            continue
        if created.tzinfo is None:
            created = created.replace(tzinfo=UTC)
        hour = created.astimezone(et).hour
        b = buckets.setdefault(
            hour,
            {"total": 0, "correct": 0, "incorrect": 0, "push": 0, "unresolved": 0},
        )
        b["total"] += 1
        outcome = row["outcome"]
        if outcome == "correct":
            b["correct"] += 1
        elif outcome == "incorrect":
            b["incorrect"] += 1
        elif outcome == "push":
            b["push"] += 1
        else:
            b["unresolved"] += 1

    out: list[PredictionStats] = []
    for hour in sorted(buckets):
        d = buckets[hour]
        decided = d["correct"] + d["incorrect"]
        accuracy = (d["correct"] / decided * 100.0) if decided > 0 else None
        out.append(PredictionStats(
            label=f"{hour:02d}",
            total=d["total"],
            correct=d["correct"],
            incorrect=d["incorrect"],
            push=d["push"],
            unresolved=d["unresolved"],
            accuracy=accuracy,
        ))
    return out


def stats_by_regime_x_vix(min_context: int = 0) -> list[PerfStats]:
    """The full regime x VIX cross-tab — one row per non-empty bucket.

    Sorted by trade count descending so the most-populated buckets surface
    first. Empty buckets are omitted entirely, never returned as 0/0 rows.
    ``min_context`` excludes trades below the given score.
    """
    ctx_clause, ctx_params = _min_context_filter(min_context)
    sql = (
        f"SELECT COALESCE(trades.market_regime, 'unknown') AS regime, "
        f"COALESCE(trades.vix_band, 'unknown') AS vix_band, "
        f"{_AGGREGATE_COLS} "
        "FROM trades "
        "JOIN signals ON signals.id = trades.signal_id "
        f"WHERE trades.outcome IN ('win','loss','expired') {ctx_clause} "
        "GROUP BY COALESCE(trades.market_regime, 'unknown'), "
        "COALESCE(trades.vix_band, 'unknown')"
    )
    conn = db.get_connection()
    try:
        rows = conn.execute(sql, ctx_params).fetchall()
    finally:
        conn.close()
    result = [
        _row_to_stats(f"{r['regime']} / {r['vix_band']}", r) for r in rows
    ]
    result.sort(key=lambda s: (-s.total, s.label))
    return result


# ────────────────────────────────────────────────────────────────────────────
# Context score slices (Phase 2.3)
# ────────────────────────────────────────────────────────────────────────────


def stats_by_context() -> list[PerfStats]:
    """One :class:`PerfStats` per context score (0-5), descending.

    Always emits all six rows (5 → 0). Buckets with no trades show
    ``total=0`` and ``win_rate=None`` — caller is responsible for
    rendering ``-`` if desired. The all-six emit is intentional so the
    report header always lines up the same way.
    """
    sql = (
        f"SELECT COALESCE(trades.context_score, 0) AS slice, "
        f"{_AGGREGATE_COLS} "
        "FROM trades "
        "JOIN signals ON signals.id = trades.signal_id "
        "WHERE trades.outcome IN ('win','loss','expired') "
        "GROUP BY COALESCE(trades.context_score, 0)"
    )
    conn = db.get_connection()
    try:
        rows = conn.execute(sql).fetchall()
    finally:
        conn.close()
    bucket: dict[int, PerfStats] = {}
    for r in rows:
        score = int(r["slice"])
        bucket[score] = _row_to_stats(str(score), r)
    # Emit all six 5 → 0; missing scores get zero-filled placeholders.
    out: list[PerfStats] = []
    for score in (5, 4, 3, 2, 1, 0):
        out.append(bucket.get(
            score,
            PerfStats(
                label=str(score), total=0, wins=0, losses=0, expired=0,
                win_rate=None, avg_pnl_pct=None,
                best_pnl_pct=None, worst_pnl_pct=None,
            ),
        ))
    return out


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
