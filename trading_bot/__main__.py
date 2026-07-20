"""CLI entry point: python -m trading_bot <command>"""

import argparse
import getpass
import json
import sys
from datetime import date

from trading_bot import (
    allocation,
    broker,
    candidate_source,
    config,
    context,
    db,
    discovery,
    discovery_universe,
    long_term,
    marketdata_compare,
    outcomes,
    performance,
    plan_execution,
    predictions,
    readiness,
    regime,
    risk_of_ruin,
    secrets,
    self_optimization,
    settings,
    shadow_discovery,
    signal_pairs,
    vix,
    watchlist_state,
)
from trading_bot.migrate_csv import migrate_csv
from trading_bot.performance import (
    PerfStats,
    PredictionStats,
    RegimeBreakdown,
    VixBreakdown,
)

# ---- secrets ----


def cmd_secrets_list() -> None:
    set_secrets = secrets.list_secrets()
    for name in sorted(secrets.KNOWN_SECRETS):
        status = "SET" if name in set_secrets else "NOT SET"
        print(f"  {name:<30} {status}")


def cmd_secrets_set(name: str) -> None:
    value = getpass.getpass(f"Value for {name}: ")
    secrets.set_secret(name, value.strip())
    print(f"Secret '{name}' stored.")


def cmd_secrets_get(name: str) -> None:
    from trading_bot.config import RUNTIME
    if RUNTIME == "railway":
        print("Cannot read secrets to stdout on Railway.", file=sys.stderr)
        sys.exit(1)
    value = secrets.get_secret(name)
    if value is None:
        print(f"Secret '{name}' is not set.", file=sys.stderr)
        sys.exit(1)
    print(value)


def cmd_secrets_delete(name: str) -> None:
    secrets.delete_secret(name)
    print(f"Secret '{name}' deleted.")


# ---- db ----


def cmd_db_init() -> None:
    db.init_db()
    print(f"Schema version: {db.schema_version()}")


def cmd_db_status() -> None:
    version = db.schema_version()
    counts = db.get_table_counts()
    print(f"Schema version: {version}")
    width = max(len(name) for name in counts) + 2
    for name, count in counts.items():
        print(f"  {name:<{width}} {count}")


# ---- migrate ----


def cmd_migrate_csv() -> None:
    imported, skipped = migrate_csv()
    print(f"Imported {imported} signals, skipped {skipped} malformed rows")


# ---- outcomes ----


def cmd_outcomes_resolve() -> None:
    result = outcomes.resolve_all_open_trades()
    print(
        f"Resolved: wins={result['wins']} losses={result['losses']} "
        f"expired={result['expired']} still_open={result['still_open']}"
    )


def cmd_outcomes_backfill() -> None:
    created = outcomes.backfill_signals_without_trades()
    print(f"Backfilled {created} signals")


def cmd_outcomes_status(track_mode: str | None = "active") -> None:
    report = outcomes.summary(track_mode=track_mode)
    scope = "shadow" if track_mode == "shadow" else "active"
    print(f"Outcomes by signal type ({scope} closed trades only):")
    if not report["by_signal_type"]:
        print("  (no closed trades yet)")
    for entry in report["by_signal_type"]:
        sign = "+" if entry["avg_pnl"] >= 0 else ""
        print(
            f"  {entry['signal_type']:<22}"
            f" wins: {entry['wins']:>2}"
            f"  losses: {entry['losses']:>2}"
            f"  win_rate: {entry['win_rate']:>5.1f}%"
            f"  avg_pnl: {sign}{entry['avg_pnl']:.2f}%"
        )
    print()
    print(f"Open trades:    {report['open']}")
    print(f"Expired trades: {report['expired']}")


# ---- report (Phase 1.4) ----


def _fmt_pct(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "   -"
    if signed:
        sign = "+" if value >= 0 else ""
        return f"{sign}{value:.2f}%"
    return f"{value:.1f}%"


def _format_perf_summary(stats: PerfStats, title: str) -> str:
    """Tall single-block layout for one PerfStats (used by overall / recent / daily)."""
    if stats.total == 0:
        return f"{title}\n  (no closed trades)"
    return (
        f"{title}\n"
        f"  Total trades:   {stats.total:>4}\n"
        f"  Wins:           {stats.wins:>4}\n"
        f"  Losses:         {stats.losses:>4}\n"
        f"  Expired:        {stats.expired:>4}\n"
        f"  Win rate:       {_fmt_pct(stats.win_rate):>6}\n"
        f"  Avg PnL:        {_fmt_pct(stats.avg_pnl_pct, signed=True):>7}\n"
        f"  Best trade:     {_fmt_pct(stats.best_pnl_pct, signed=True):>7}\n"
        f"  Worst trade:    {_fmt_pct(stats.worst_pnl_pct, signed=True):>7}"
    )


def _format_perf_table(rows: list[PerfStats], title: str, label_header: str = "Slice") -> str:
    """Tabular layout for grouped stats (by-signal, by-ticker, by-asset, cross)."""
    if not rows:
        return f"{title}\n  (no closed trades)"

    # Dynamic label column width — fit the longest label, minimum 18 chars.
    label_w = max(18, max(len(r.label) for r in rows))
    header = (
        f"  {label_header:<{label_w}}  Total  Wins  Losses  Expired  Win rate"
        f"   Avg PnL    Best    Worst"
    )
    lines = [title, header]
    for r in rows:
        if r.total == 0:
            lines.append(f"  {r.label:<{label_w}}  (no closed trades)")
            continue
        lines.append(
            f"  {r.label:<{label_w}}"
            f"  {r.total:>5}"
            f"  {r.wins:>4}"
            f"  {r.losses:>6}"
            f"  {r.expired:>7}"
            f"  {_fmt_pct(r.win_rate):>8}"
            f"  {_fmt_pct(r.avg_pnl_pct, signed=True):>8}"
            f"  {_fmt_pct(r.best_pnl_pct, signed=True):>7}"
            f"  {_fmt_pct(r.worst_pnl_pct, signed=True):>7}"
        )
    return "\n".join(lines)


def cmd_report_overall(track_mode: str | None = "active") -> None:
    title = "OVERALL PERFORMANCE" + (
        " (SHADOW)" if track_mode == "shadow" else ""
    )
    print(_format_perf_summary(
        performance.stats_overall(track_mode=track_mode), title,
    ))


def cmd_report_recent(days: int) -> None:
    stats = performance.stats_recent(days=days)
    print(_format_perf_summary(stats, f"RECENT PERFORMANCE — last {days} days"))


def _min_ctx_suffix(min_context: int) -> str:
    """Title suffix shown when a context filter is active. Empty otherwise."""
    return f"  [min context {min_context}]" if min_context > 0 else ""


def cmd_report_by_signal(min_context: int = 0) -> None:
    print(_format_perf_table(
        performance.stats_by_signal_type(min_context=min_context),
        f"BY SIGNAL TYPE{_min_ctx_suffix(min_context)}",
        label_header="Signal type",
    ))


def cmd_report_by_ticker(asset_class: str | None, min_context: int = 0) -> None:
    title = "BY TICKER"
    if asset_class is not None:
        title = f"BY TICKER ({asset_class})"
    title += _min_ctx_suffix(min_context)
    print(_format_perf_table(
        performance.stats_by_ticker(
            asset_class=asset_class, min_context=min_context,
        ),
        title,
        label_header="Ticker",
    ))


def cmd_report_by_asset() -> None:
    print(_format_perf_table(
        performance.stats_by_asset_class(),
        "BY ASSET CLASS",
        label_header="Asset class",
    ))


def cmd_report_cross() -> None:
    # ASCII-only title: Windows' default cp1252 console can't encode '≥'
    # (U+2265) and would crash print() with UnicodeEncodeError.
    print(_format_perf_table(
        performance.stats_by_signal_type_and_ticker(),
        "BY SIGNAL TYPE x TICKER  (slices with >=3 trades)",
        label_header="Signal / Ticker",
    ))


def cmd_report_pairs() -> None:
    """Per-(ticker, signal_type) performance (Phase 4).

    Windowed resolved stats across ALL track_modes — so a muted pair (which
    fires as shadow) still shows its real standing — plus its current gate
    status. Headline active reports are unaffected: muted pairs fire as shadow
    and are already excluded from those.
    """
    stats = signal_pairs.pair_stats()
    print(
        f"PER-PAIR PERFORMANCE  "
        f"(window {config.SP_WINDOW_DAYS}d, all track modes)"
    )
    print("-" * 72)
    if not stats:
        print("  (no traded pairs yet)")
        return
    # Worst expectancy first so problem pairs surface; None (no data) last.
    rows = sorted(
        stats,
        key=lambda s: (s.expectancy is None, s.expectancy if s.expectancy is not None else 0.0),
    )
    print(
        f"  {'Ticker':<8} {'Signal type':<22} {'Closed':>6}  "
        f"{'Win rate':>8}  {'Expectancy':>10}  {'Status':<8}"
    )
    for s in rows:
        print(
            f"  {s.ticker:<8} {s.signal_type:<22} {s.closed_count:>6}  "
            f"{_fmt_win_rate(s.win_rate):>8}  {_fmt_signed(s.expectancy, 3):>10}  "
            f"{s.status:<8}"
        )


def cmd_report_daily_backfill() -> None:
    count = performance.backfill_daily_performance()
    print(f"Backfilled daily_performance for {count} dates")


def _format_cache_age(cached_at: object) -> str:
    """Return a human-readable age string like '4 hours' or '23 minutes'.

    Accepts a datetime or None. The CLI prints 'unknown' for missing cache
    (e.g. after force_refresh where no prior cache existed).
    """
    from datetime import UTC as _UTC
    from datetime import datetime as _dt

    if not isinstance(cached_at, _dt):
        return "unknown"
    if cached_at.tzinfo is None:
        cached_at = cached_at.replace(tzinfo=_UTC)
    delta = _dt.now(_UTC) - cached_at
    seconds = int(delta.total_seconds())
    if seconds < 60:
        return f"{seconds} seconds"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} minute{'s' if minutes != 1 else ''}"
    hours = minutes // 60
    if hours < 24:
        return f"{hours} hour{'s' if hours != 1 else ''}"
    days = hours // 24
    return f"{days} day{'s' if days != 1 else ''}"


def cmd_regime_current() -> None:
    try:
        snap = regime.get_current_regime()
    except regime.RegimeFetchError as exc:
        print(f"Regime fetch error: {exc}", file=sys.stderr)
        sys.exit(1)
    age = _format_cache_age(regime.last_cached_at())
    ema50_marker = "above 200 EMA" if snap.ema50 > snap.ema200 else (
        "below 200 EMA" if snap.ema50 < snap.ema200 else "equal to 200 EMA"
    )
    slope_label = (
        "positive" if snap.ema50_slope > 0
        else "negative" if snap.ema50_slope < 0 else "flat"
    )
    slope_sign = "+" if snap.ema50_slope >= 0 else ""

    print(f"Market Regime: {snap.regime.upper()}")
    print("-" * 38)
    print(f"SPY Close:    ${snap.spy_close:.2f}")
    print(f"50 EMA:       ${snap.ema50:.2f}  ({ema50_marker})")
    print(f"200 EMA:      ${snap.ema200:.2f}")
    print(f"50 EMA Slope: {slope_sign}{snap.ema50_slope:.2f}/day ({slope_label})")
    print(f"Snapshot age: {age}")


def cmd_regime_history(days: int) -> None:
    rows = db.get_regime_snapshots(limit=days)
    if not rows:
        print("Regime history")
        print("  (no snapshots recorded)")
        return
    print("Regime history")
    print(
        f"  {'Date':<12} {'Regime':<10} {'SPY Close':>10}  "
        f"{'50 EMA':>10}  {'200 EMA':>10}"
    )
    for row in rows:
        print(
            f"  {row['date']:<12} {row['regime']:<10} "
            f"${row['spy_close']:>9.2f}  "
            f"${row['ema50']:>9.2f}  "
            f"${row['ema200']:>9.2f}"
        )


def cmd_regime_backfill() -> None:
    summary = regime.backfill_trade_regimes()
    print(
        f"Backfilled {summary['trades_updated']} trades across "
        f"{summary['dates_snapshotted']} unique dates. "
        f"{summary['errors']} errors."
    )
    if summary["skipped_no_data"] > 0:
        print(
            f"  ({summary['skipped_no_data']} trades skipped — "
            f"opened_at fell on a non-trading date or pre-history)"
        )


# ---- regime-aware report rendering ----


def _fmt_cell(stats: PerfStats | None) -> str:
    """Format one cell of a regime cross-tab: '8/12 66%' or '-' for empty."""
    if stats is None or stats.total == 0:
        return "-"
    decided = stats.wins + stats.losses
    if decided == 0:
        # All-expired slice: surface counts without a misleading 0% win rate.
        return f"0/{stats.expired} -"
    return f"{stats.wins}/{decided} {stats.win_rate:.0f}%"


def _format_axis_matrix(
    rows: list[object],
    title: str,
    label_header: str,
    axis_keys: tuple[str, ...],
    axis_headers: tuple[str, ...],
    cell_getter: object,
) -> str:
    """Render a label × axis cross-tab. ``cell_getter`` extracts the per-axis
    dict from a breakdown row (e.g. ``lambda r: r.by_regime`` or
    ``lambda r: r.by_vix``). ``axis_keys`` is the lookup order on that dict,
    ``axis_headers`` the display labels. Both are positionally aligned."""
    if not rows:
        return f"{title}\n  (no closed trades)"

    # rows are RegimeBreakdown | VixBreakdown — both expose .label and .overall.
    label_w = max(18, max(len(getattr(r, "label")) for r in rows))  # noqa: B009
    cell_w = 12
    header_cells = "".join(h.ljust(cell_w) for h in axis_headers)
    header = f"  {label_header:<{label_w}}  {header_cells}{'Overall':<{cell_w}}"
    lines = [title, header]

    # Local typed cast — cell_getter is callable but kept loosely typed at the
    # signature boundary so it can accept either RegimeBreakdown or VixBreakdown.
    getter = cell_getter
    for r in rows:
        bucket = getter(r)  # type: ignore[operator]
        cells = "".join(_fmt_cell(bucket.get(k)).ljust(cell_w) for k in axis_keys)
        overall_cell = _fmt_cell(getattr(r, "overall"))  # noqa: B009
        lines.append(
            f"  {getattr(r, 'label'):<{label_w}}  {cells}{overall_cell:<{cell_w}}"  # noqa: B009
        )
    return "\n".join(lines)


def _format_regime_matrix(
    rows: list[RegimeBreakdown], title: str, label_header: str
) -> str:
    return _format_axis_matrix(
        rows=list(rows),
        title=title,
        label_header=label_header,
        axis_keys=("bull", "sideways", "bear", "unknown"),
        axis_headers=("Bull", "Sideways", "Bear", "Unknown"),
        cell_getter=lambda r: r.by_regime,
    )


def _format_vix_matrix(
    rows: list[VixBreakdown], title: str, label_header: str
) -> str:
    return _format_axis_matrix(
        rows=list(rows),
        title=title,
        label_header=label_header,
        axis_keys=("low", "elevated", "high", "extreme", "unknown"),
        axis_headers=("Low", "Elevated", "High", "Extreme", "Unknown"),
        cell_getter=lambda r: r.by_vix,
    )


def cmd_report_by_regime(min_context: int = 0) -> None:
    stats = performance.stats_by_regime(min_context=min_context)
    if not stats:
        print(f"BY MARKET REGIME{_min_ctx_suffix(min_context)}\n  (no closed trades)")
        return
    print(f"BY MARKET REGIME{_min_ctx_suffix(min_context)}")
    print(
        f"  {'Regime':<12} {'Trades':>6}  {'Wins':>4}  {'Losses':>6}  "
        f"{'Expired':>7}  {'Win rate':>8}  {'Avg PnL':>8}"
    )
    for s in stats:
        wr = "    -" if s.win_rate is None else f"{s.win_rate:.1f}%"
        ap = (
            "    -" if s.avg_pnl_pct is None
            else f"{'+' if s.avg_pnl_pct >= 0 else ''}{s.avg_pnl_pct:.2f}%"
        )
        print(
            f"  {s.label:<12} {s.total:>6}  {s.wins:>4}  {s.losses:>6}  "
            f"{s.expired:>7}  {wr:>8}  {ap:>8}"
        )


def cmd_report_by_signal_regime(min_context: int = 0) -> None:
    rows = performance.stats_by_signal_type_with_regime(min_context=min_context)
    print(_format_regime_matrix(
        rows,
        title=f"BY SIGNAL TYPE x REGIME{_min_ctx_suffix(min_context)}",
        label_header="Signal",
    ))


def cmd_report_by_ticker_regime(
    asset_class: str | None, min_context: int = 0,
) -> None:
    rows = performance.stats_by_ticker_with_regime(
        asset_class=asset_class, min_context=min_context,
    )
    title = "BY TICKER x REGIME"
    if asset_class is not None:
        title = f"BY TICKER x REGIME ({asset_class})"
    title += _min_ctx_suffix(min_context)
    print(_format_regime_matrix(rows, title=title, label_header="Ticker"))


# ---- VIX CLI (Phase 2.2) ----


def cmd_vix_current() -> None:
    try:
        snap = vix.get_current_vix()
    except vix.VixFetchError as exc:
        print(f"VIX fetch error: {exc}", file=sys.stderr)
        sys.exit(1)
    age = _format_cache_age(vix.last_cached_at())
    thresholds = {
        "low":      "< 20",
        "elevated": "20 <= x < 30",
        "high":     "30 <= x < 40",
        "extreme":  ">= 40",
        "unknown":  "fetch failed",
    }
    thr = thresholds.get(snap.vix_band, "")
    print(f"VIX: {snap.vix_level:.1f} ({snap.vix_band})")
    print("-" * 38)
    print(f"Level:         {snap.vix_level:.1f}")
    print(f"Band:          {snap.vix_band:<8} (threshold: {thr})")
    print(f"Snapshot age:  {age}")


def cmd_vix_history(days: int) -> None:
    rows = db.get_vix_snapshots(limit=days)
    if not rows:
        print("VIX history")
        print("  (no snapshots recorded)")
        return
    print("VIX history")
    print(f"  {'Date':<12} {'Level':>7}   {'Band':<10}")
    for row in rows:
        print(
            f"  {row['date']:<12} {row['vix_level']:>7.2f}   {row['vix_band']:<10}"
        )


def cmd_vix_backfill() -> None:
    summary = vix.backfill_trade_vix()
    print(
        f"Backfilled {summary['trades_updated']} trades across "
        f"{summary['dates_snapshotted']} unique dates. "
        f"{summary['errors']} errors."
    )
    if summary["skipped_no_data"] > 0:
        print(
            f"  ({summary['skipped_no_data']} trades skipped — "
            f"opened_at fell on a non-trading date or pre-history)"
        )


def cmd_report_by_vix(min_context: int = 0) -> None:
    stats = performance.stats_by_vix_band(min_context=min_context)
    if not stats:
        print(f"BY VIX BAND{_min_ctx_suffix(min_context)}\n  (no closed trades)")
        return
    print(f"BY VIX BAND{_min_ctx_suffix(min_context)}")
    print(
        f"  {'Band':<12} {'Trades':>6}  {'Wins':>4}  {'Losses':>6}  "
        f"{'Expired':>7}  {'Win rate':>8}  {'Avg PnL':>8}"
    )
    for s in stats:
        wr = "    -" if s.win_rate is None else f"{s.win_rate:.1f}%"
        ap = (
            "    -" if s.avg_pnl_pct is None
            else f"{'+' if s.avg_pnl_pct >= 0 else ''}{s.avg_pnl_pct:.2f}%"
        )
        print(
            f"  {s.label:<12} {s.total:>6}  {s.wins:>4}  {s.losses:>6}  "
            f"{s.expired:>7}  {wr:>8}  {ap:>8}"
        )


def cmd_report_by_signal_vix(min_context: int = 0) -> None:
    rows = performance.stats_by_signal_type_with_vix(min_context=min_context)
    print(_format_vix_matrix(
        rows,
        title=f"BY SIGNAL TYPE x VIX{_min_ctx_suffix(min_context)}",
        label_header="Signal",
    ))


def cmd_report_by_ticker_vix(
    asset_class: str | None, min_context: int = 0,
) -> None:
    rows = performance.stats_by_ticker_with_vix(
        asset_class=asset_class, min_context=min_context,
    )
    title = "BY TICKER x VIX"
    if asset_class is not None:
        title = f"BY TICKER x VIX ({asset_class})"
    title += _min_ctx_suffix(min_context)
    print(_format_vix_matrix(rows, title=title, label_header="Ticker"))


def cmd_report_by_regime_vix(min_context: int = 0) -> None:
    """The headline Phase 2.2 view: every regime x VIX bucket with data,
    sorted by trade count desc. See ``stats_by_regime_x_vix`` for the
    aggregation rules."""
    stats = performance.stats_by_regime_x_vix(min_context=min_context)
    if not stats:
        print(f"BY REGIME x VIX{_min_ctx_suffix(min_context)}\n  (no closed trades)")
        return
    print(f"BY REGIME x VIX{_min_ctx_suffix(min_context)}")
    print(
        f"  {'Regime':<10} {'VIX Band':<10} {'Trades':>6}  {'Wins':>4}  "
        f"{'Losses':>6}  {'Expired':>7}  {'Win rate':>8}  {'Avg PnL':>8}"
    )
    for s in stats:
        # label looks like "bull / low" — split for display.
        if " / " in s.label:
            regime_part, vix_part = s.label.split(" / ", 1)
        else:  # pragma: no cover - defensive; aggregator always uses ' / '
            regime_part, vix_part = s.label, "-"
        wr = "    -" if s.win_rate is None else f"{s.win_rate:.1f}%"
        ap = (
            "    -" if s.avg_pnl_pct is None
            else f"{'+' if s.avg_pnl_pct >= 0 else ''}{s.avg_pnl_pct:.2f}%"
        )
        print(
            f"  {regime_part:<10} {vix_part:<10} {s.total:>6}  {s.wins:>4}  "
            f"{s.losses:>6}  {s.expired:>7}  {wr:>8}  {ap:>8}"
        )


# ---- context CLI (Phase 2.3) ----


def cmd_context_current() -> None:
    snap = context.get_current_context()
    label_upper = snap.label.upper()
    vix_str = (
        f"{snap.vix_level:.1f}" if snap.vix_level is not None else "unknown"
    )
    # Per-axis cache age strings, for the bottom line.
    reg_age = _format_cache_age(regime.last_cached_at())
    vix_age = _format_cache_age(vix.last_cached_at())

    print(f"Market Context: {label_upper} ({snap.score}/5)")
    print("-" * 38)
    print(f"Regime:      {snap.regime}")
    print(f"VIX:         {vix_str:<10} {snap.vix_band}")
    print(f"Score:       {snap.score} / 5    {snap.label}")
    print(f"Caches:      regime {reg_age}, VIX {vix_age}")


def cmd_context_history(days: int) -> None:
    rows = db.get_combined_context_history(limit=days)
    if not rows:
        print("Context history")
        print("  (no overlapping snapshots — backfill regime + vix first)")
        return
    print("Context history")
    print(
        f"  {'Date':<12} {'Regime':<10} {'VIX Band':<10} "
        f"{'Score':>5}   {'Label':<12}"
    )
    # rows come newest-first from db; flip for chronological display.
    for row in reversed(rows):
        s = context.score(row["regime"], row["vix_band"])
        lab = context.label(s)
        print(
            f"  {row['date']:<12} {row['regime']:<10} {row['vix_band']:<10} "
            f"{s:>5}   {lab:<12}"
        )


def cmd_context_backfill() -> None:
    summary = context.backfill_context_scores()
    print(
        f"Backfilled context_score on {summary['trades_updated']} trades "
        f"and {summary['predictions_updated']} predictions."
    )


# ---- by-context + context-matrix reports ----


def cmd_report_by_context() -> None:
    rows = performance.stats_by_context()
    # rows always emits all 6 (5→0); show them even when empty so the user
    # sees the full ladder.
    print("BY CONTEXT SCORE")
    print(
        f"  {'Score':<5} {'Label':<12} {'Trades':>6}  {'Wins':>4}  "
        f"{'Losses':>6}  {'Win rate':>8}  {'Avg PnL':>8}"
    )
    for s in rows:
        score_val = int(s.label)
        lab = context.label(score_val)
        wr = "    -" if s.win_rate is None else f"{s.win_rate:.1f}%"
        ap = (
            "    -" if s.avg_pnl_pct is None
            else f"{'+' if s.avg_pnl_pct >= 0 else ''}{s.avg_pnl_pct:.2f}%"
        )
        print(
            f"  {score_val:<5} {lab:<12} {s.total:>6}  {s.wins:>4}  "
            f"{s.losses:>6}  {wr:>8}  {ap:>8}"
        )


def cmd_report_context_matrix() -> None:
    """The 3x4 regime x VIX matrix view with per-cell win rates.

    Same data as ``report by-regime-vix`` but rendered as a matrix with
    fixed row/col order so the user can read it geometrically. Sparse
    cells show ``-``.
    """
    raw = performance.stats_by_regime_x_vix()
    by_cell: dict[tuple[str, str], PerfStats] = {}
    for s in raw:
        if " / " not in s.label:
            continue
        reg, vix_part = s.label.split(" / ", 1)
        by_cell[(reg, vix_part)] = s

    regime_order = ("bull", "sideways", "bear")
    vix_order = ("low", "elevated", "high", "extreme")

    if not raw:
        print("Win rates: regime x VIX\n  (no closed trades)")
        return

    print("Win rates: regime x VIX")
    cell_w = 11
    header = f"  {'Regime':<10}" + "".join(
        h.capitalize().ljust(cell_w) for h in vix_order
    )
    print(header)
    for reg in regime_order:
        cells = ""
        for v in vix_order:
            cell_stats = by_cell.get((reg, v))
            cells += _fmt_cell(cell_stats).ljust(cell_w)
        print(f"  {reg.capitalize():<10}{cells}")


def _maybe_min_context(args: object) -> int:
    """Pull the --min-context value off argparse Namespace; 0 = no filter."""
    return int(getattr(args, "min_context", 0) or 0)


# ---- discovery CLI (Phase 3.1) ----


def _fmt_signed(value: float | None, places: int) -> str:
    """Signed float like '+1.23', or '-' for None. Used by discovery tables."""
    if value is None:
        return "-"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.{places}f}"


def cmd_discovery_scan(*, throttle_seconds: float) -> None:
    run = discovery.run_discovery(throttle_seconds=throttle_seconds)

    print("DISCOVERY SCAN  (informational only — promotes nothing)")
    print("-" * 54)
    print(f"Scanned:   {run.scanned}")
    print(f"Succeeded: {run.succeeded}")
    print(f"Failed:    {run.failed}")
    print()

    quals = run.qualifiers
    print(f"Ranked candidates ({len(quals)}) — trade_count >= 10 AND expectancy > 0:")
    if not quals:
        print("  (none)")
    else:
        print(
            f"  {'Ticker':<8} {'Trades':>6}  {'Win rate':>8}  "
            f"{'Avg ret':>8}  {'Expectancy':>10}"
        )
        for s in quals:
            wr = "    -" if s.win_rate is None else f"{s.win_rate:.1f}%"
            print(
                f"  {s.ticker:<8} {s.trade_count:>6}  {wr:>8}  "
                f"{_fmt_signed(s.avg_return_pct, 2):>8}  "
                f"{_fmt_signed(s.expectancy, 3):>10}"
            )
    print()
    print(
        "Promotion is owned by live-shadow — run `shadow evaluate` to promote "
        "names whose RESOLVED shadow signals qualify."
    )
    print()

    print(f"Failures ({run.failed}):")
    if not run.failures:
        print("  (none)")
    else:
        for f in run.failures:
            print(f"  {f.ticker:<8} {f.reason}")


# ---- shadow CLI (Phase 3.1-LIVE) ----


def _fmt_win_rate(value: float | None) -> str:
    return "    -" if value is None else f"{value:.1f}%"


def _shadow_bar_line() -> str:
    return (
        f"Promotion bar: closed_count >= {shadow_discovery.MIN_SHADOW_SIGNALS} "
        f"AND expectancy >= {shadow_discovery.PROMOTE_EXPECTANCY} "
        f"(resolved within {shadow_discovery.RECENT_WINDOW_DAYS}d)"
    )


def _shadow_candidates() -> list[str]:
    """Shadow-universe names not already on the active watchlist."""
    try:
        active = set(db.get_active_watchlist())
    except Exception as exc:  # noqa: BLE001 - degrade gracefully for a read-only view
        print(f"  (active watchlist unreadable: {exc})", file=sys.stderr)
        active = set()
    return [t for t in discovery_universe.SHADOW_UNIVERSE if t not in active]


def cmd_shadow_status() -> None:
    candidates = _shadow_candidates()
    universe = len(discovery_universe.SHADOW_UNIVERSE)

    print("SHADOW STATUS")
    print("-" * 60)
    print(
        f"Universe: {universe}   on active watchlist (skipped): "
        f"{universe - len(candidates)}   candidates: {len(candidates)}"
    )
    print(_shadow_bar_line())
    print()

    evals = [shadow_discovery.evaluate_ticker(t) for t in candidates]
    with_data = [e for e in evals if e.closed_count > 0]
    # Eligible first, then by expectancy desc, then by sample size desc.
    with_data.sort(
        key=lambda e: (not e.eligible, -(e.expectancy or -1e9), -e.closed_count)
    )

    print(f"  {'Ticker':<8} {'Resolved':>8}  {'Win rate':>8}  "
          f"{'Expectancy':>10}  {'Eligible':>8}")
    if not with_data:
        print("  (no candidate has resolved shadow signals yet)")
    for e in with_data:
        print(
            f"  {e.ticker:<8} {e.closed_count:>8}  {_fmt_win_rate(e.win_rate):>8}  "
            f"{_fmt_signed(e.expectancy, 3):>10}  "
            f"{'yes' if e.eligible else 'no':>8}"
        )

    no_data = len(candidates) - len(with_data)
    print()
    print(f"  ({no_data} candidates have no resolved shadow signals yet)")


def _withhold_reason(ev: shadow_discovery.ShadowEvaluation) -> str:
    if ev.closed_count < shadow_discovery.MIN_SHADOW_SIGNALS:
        return (
            f"only {ev.closed_count} resolved "
            f"(need {shadow_discovery.MIN_SHADOW_SIGNALS})"
        )
    exp = "n/a" if ev.expectancy is None else f"{ev.expectancy:.3f}"
    return f"expectancy {exp} < {shadow_discovery.PROMOTE_EXPECTANCY}"


def cmd_shadow_evaluate(*, dry_run: bool) -> None:
    evals = shadow_discovery.evaluate_shadow_universe(dry_run=dry_run)
    eligible = [e for e in evals if e.eligible]

    header = "SHADOW EVALUATE"
    if dry_run:
        header += "  (DRY RUN - no promotion, no persistence)"
    print(header)
    print("-" * 60)
    print(f"Candidates evaluated: {len(evals)}   eligible: {len(eligible)}")
    print(_shadow_bar_line())
    print()

    # In a real run, `promoted` reflects what was actually added; in dry-run,
    # show the eligible names that WOULD be promoted.
    shown = eligible if dry_run else [e for e in evals if e.promoted]
    verb = "Would promote" if dry_run else "Promoted"
    print(f"{verb} ({len(shown)}):")
    if not shown:
        print("  (none)")
    for e in sorted(shown, key=lambda e: -(e.expectancy or 0.0)):
        print(
            f"  {e.ticker:<8} resolved={e.closed_count}  "
            f"win_rate={_fmt_win_rate(e.win_rate)}  "
            f"expectancy={_fmt_signed(e.expectancy, 3)}"
        )
    print()

    withheld = [e for e in evals if not e.eligible and e.closed_count > 0]
    print(f"Withheld with data ({len(withheld)}):")
    if not withheld:
        print("  (none)")
    for e in sorted(withheld, key=lambda e: -e.closed_count):
        print(f"  {e.ticker:<8} {_withhold_reason(e)}")

    no_data = sum(1 for e in evals if e.closed_count == 0)
    print()
    print(f"  ({no_data} candidates have no resolved shadow signals yet)")


# ---- watchlist state-machine CLI (Phase 3.3) ----


def _sm_bar_line() -> str:
    return (
        f"Bars: demote expectancy <= {config.SM_DEMOTE_EXPECTANCY}, "
        f"recover >= {config.SM_PROMOTE_EXPECTANCY}, "
        f"min sample {config.SM_MIN_CLOSED_SIGNALS}, "
        f"window {config.SM_WINDOW_DAYS}d, active floor {config.SM_MIN_ACTIVE}"
    )


def cmd_watchlist_evaluate(*, dry_run: bool) -> None:
    run = watchlist_state.evaluate_watchlist(dry_run=dry_run)

    header = "WATCHLIST EVALUATE"
    if dry_run:
        header += "  (DRY RUN - no changes)"
    print(header)
    print("-" * 60)
    print(_sm_bar_line())
    print()

    def _verdict_line(ev: watchlist_state.TickerEvaluation) -> str:
        return (
            f"  {ev.ticker:<8} {ev.status} -> {ev.new_status}  "
            f"closed={ev.closed_count} expectancy={_fmt_signed(ev.expectancy, 3)}"
            f"  {ev.reason}"
        )

    print(f"Demotions ({len(run.demotions)}):")
    if not run.demotions:
        print("  (none)")
    for ev in run.demotions:
        print(_verdict_line(ev))
    print()

    print(f"Recoveries ({len(run.recoveries)}):")
    if not run.recoveries:
        print("  (none)")
    for ev in run.recoveries:
        print(_verdict_line(ev))
    print()

    # Floor holds — demotions withheld to honour SM_MIN_ACTIVE — shown apart.
    floor_holds = [e for e in run.holds if "active floor" in e.reason]
    print(f"Floor holds ({len(floor_holds)}):")
    if not floor_holds:
        print("  (none)")
    for ev in floor_holds:
        print(
            f"  {ev.ticker:<8} expectancy={_fmt_signed(ev.expectancy, 3)}  "
            f"{ev.reason}"
        )
    print()

    other_holds = [e for e in run.holds if "active floor" not in e.reason]
    print(f"Other holds: {len(other_holds)} "
          f"(insufficient sample / dead band)")
    print()

    print(f"Active  ({len(run.active_after)}): "
          f"{', '.join(run.active_after) or '(none)'}")
    print(f"Benched ({len(run.benched_after)}): "
          f"{', '.join(run.benched_after) or '(none)'}")


def _print_watchlist_rows(entries: list[dict[str, object]]) -> None:
    if not entries:
        print("  (none)")
        return
    print(
        f"  {'Ticker':<8} {'Closed':>6}  {'Win rate':>8}  "
        f"{'Expectancy':>10}  {'Source':<10}"
    )
    for entry in entries:
        ticker = str(entry["ticker"])
        try:
            closed, win_rate, expectancy = watchlist_state.windowed_stats_for(ticker)
        except Exception as exc:  # noqa: BLE001 - a read-only view must not crash
            print(f"  {ticker:<8} (stats error: {exc})", file=sys.stderr)
            continue
        print(
            f"  {ticker:<8} {closed:>6}  {_fmt_win_rate(win_rate):>8}  "
            f"{_fmt_signed(expectancy, 3):>10}  {str(entry['source']):<10}"
        )


def cmd_watchlist_status() -> None:
    try:
        entries = db.get_watchlist_entries()
    except Exception as exc:  # noqa: BLE001 - degrade gracefully for a read-only view
        print(f"  watchlist unreadable: {exc}", file=sys.stderr)
        entries = []

    active = [e for e in entries if e["status"] == "active"]
    benched = [e for e in entries if e["status"] == "benched"]

    print("WATCHLIST STATUS")
    print("-" * 60)
    print(f"Active ({len(active)}):")
    _print_watchlist_rows(active)
    print()
    print(f"Benched ({len(benched)}):")
    _print_watchlist_rows(benched)


# ---- per-pair gate CLI (Phase 4) ----


def _sp_bar_line() -> str:
    return (
        f"Bars: mute expectancy <= {config.SP_MUTE_EXPECTANCY}, "
        f"enable >= {config.SP_ENABLE_EXPECTANCY}, "
        f"min sample {config.SP_MIN_CLOSED_SIGNALS}, "
        f"window {config.SP_WINDOW_DAYS}d"
    )


def cmd_pairs_evaluate(*, dry_run: bool) -> None:
    run = signal_pairs.evaluate_signal_pairs(dry_run=dry_run)

    header = "PAIRS EVALUATE"
    if dry_run:
        header += "  (DRY RUN - no changes)"
    print(header)
    print("-" * 60)
    print(_sp_bar_line())
    print()

    def _line(ev: signal_pairs.PairEvaluation) -> str:
        return (
            f"  {ev.ticker}/{ev.signal_type:<22} {ev.status} -> "
            f"{ev.new_status:<8} closed={ev.closed_count} "
            f"expectancy={_fmt_signed(ev.expectancy, 3)}  {ev.reason}"
        )

    print(f"Muted ({len(run.mutes)}):")
    if not run.mutes:
        print("  (none)")
    for ev in run.mutes:
        print(_line(ev))
    print()

    print(f"Enabled ({len(run.enables)}):")
    if not run.enables:
        print("  (none)")
    for ev in run.enables:
        print(_line(ev))
    print()

    print(f"Holds: {len(run.holds)} (dead band / insufficient sample)")
    print()

    muted_now = sorted(
        f"{e.ticker}/{e.signal_type}"
        for e in run.evaluations if e.new_status == "muted"
    )
    enabled_now = sum(1 for e in run.evaluations if e.new_status == "enabled")
    print(f"Muted now ({len(muted_now)}): {', '.join(muted_now) or '(none)'}")
    print(f"Enabled now: {enabled_now}")


def _print_pair_rows(rows: list[signal_pairs.PairStat]) -> None:
    if not rows:
        print("  (none)")
        return
    print(
        f"  {'Ticker':<8} {'Signal type':<22} {'Closed':>6}  "
        f"{'Win rate':>8}  {'Expectancy':>10}"
    )
    for s in rows:
        print(
            f"  {s.ticker:<8} {s.signal_type:<22} {s.closed_count:>6}  "
            f"{_fmt_win_rate(s.win_rate):>8}  {_fmt_signed(s.expectancy, 3):>10}"
        )


def cmd_pairs_status() -> None:
    stats = signal_pairs.pair_stats()
    enabled = [s for s in stats if s.status == "enabled"]
    muted = [s for s in stats if s.status == "muted"]

    print("PAIRS STATUS")
    print("-" * 60)
    print(f"Enabled ({len(enabled)}):")
    _print_pair_rows(enabled)
    print()
    print(f"Muted ({len(muted)}):")
    _print_pair_rows(muted)


# ---- sentiment CLI (Phase 5) ----


def cmd_sentiment_status(limit: int = 20) -> None:
    """Recent fired signals with their advisory sentiment context."""
    rows = db.get_recent_trade_sentiment(limit=limit)
    print("SENTIMENT STATUS  (recent scored signals)")
    print("-" * 86)
    if not rows:
        print("  (no sentiment-scored signals yet)")
        return
    print(
        f"  {'Opened':<20} {'Ticker':<8} {'Signal':<20} {'Score':>6}  "
        f"{'Label':<8} {'News':>4}  {'Heavy':<5} {'Outcome':<8}"
    )
    for r in rows:
        score = r["sentiment_score"]
        score_txt = (
            f"{score:+.2f}" if isinstance(score, (int, float)) else "   -"
        )
        opened = r["opened_at"][:19].replace("T", " ")
        headlines = r["headline_count"] if r["headline_count"] is not None else "-"
        print(
            f"  {opened:<20} {r['ticker']:<8} {r['signal_type']:<20} "
            f"{score_txt:>6}  {r['sentiment_label']:<8} {str(headlines):>4}  "
            f"{('yes' if r['heavy_news'] else 'no'):<5} "
            f"{str(r['outcome'] or 'open'):<8}"
        )


# ---- indicators CLI (Phase 6) ----


def cmd_indicators_status(limit: int = 20) -> None:
    """Recent fired signals with their advisory indicator-family context.

    The per-signal Phase 6 report view: volatility regime, RSI, ADX, OBV
    (volume flow) and the cross-asset correlation/concentration, next to the
    eventual outcome. Advisory only — these never gated the signal.
    """
    def _n(value: object, spec: str) -> str:
        return format(value, spec) if isinstance(value, (int, float)) else "-"

    rows = db.get_recent_trade_indicators(limit=limit)
    print("INDICATOR STATUS  (recent signals with indicator-family context)")
    print("-" * 104)
    if not rows:
        print("  (no indicator-scored signals yet)")
        return
    print(
        f"  {'Opened':<20} {'Ticker':<7} {'Signal':<20} {'Vol':<7} "
        f"{'RSI':>5} {'ADX':>5} {'OBV':>13} {'Corr':>6} "
        f"{'Concentration':<13} {'Outcome':<8}"
    )
    for r in rows:
        opened = r["opened_at"][:19].replace("T", " ")
        print(
            f"  {opened:<20} {r['ticker']:<7} {r['signal_type']:<20} "
            f"{r['ind_vol_regime']:<7} "
            f"{_n(r['ind_rsi'], '.1f'):>5} {_n(r['ind_adx'], '.1f'):>5} "
            f"{_n(r['ind_obv'], ',.0f'):>13} "
            f"{_n(r['ind_correlation'], '+.2f'):>6} "
            f"{(r['ind_concentration'] or '-'):<13} "
            f"{str(r['outcome'] or 'open'):<8}"
        )


# ---- risk CLI (Phase 7) ----


def cmd_risk_status(limit: int = 20) -> None:
    """Recent fired signals with their advisory risk recommendation.

    Shows the recommended size (a trailing ``*`` marks a position capped at the
    per-position limit), the risk %, and the three portfolio verdicts next to
    the eventual outcome. Advisory only — nothing here was enforced.
    """
    def _n(value: object, spec: str) -> str:
        return format(value, spec) if isinstance(value, (int, float)) else "-"

    rows = db.get_recent_trade_risk(limit=limit)
    print("RISK STATUS  (recent signals with advisory sizing + portfolio verdicts)")
    print("-" * 112)
    if not rows:
        print("  (no risk-assessed signals yet)")
        return
    print(
        f"  {'Opened':<20} {'Ticker':<7} {'Signal':<18} {'Size':>9} {'Risk%':>6}  "
        f"{'Portfolio':<22} {'Position':<22} {'Cluster':<20} {'Outcome':<8}"
    )
    for r in rows:
        size = r["risk_recommended_size"]
        size_txt = (
            f"{size:,.2f}" + ("*" if r["risk_capped"] else "")
            if isinstance(size, (int, float)) else "-"
        )
        print(
            f"  {r['opened_at'][:19].replace('T', ' '):<20} {r['ticker']:<7} "
            f"{r['signal_type']:<18} {size_txt:>9} {_n(r['risk_pct'], '.2f'):>6}  "
            f"{r['risk_portfolio_verdict']:<22} "
            f"{(r['risk_position_verdict'] or '-'):<22} "
            f"{(r['risk_cluster_verdict'] or '-'):<20} "
            f"{str(r['outcome'] or 'open'):<8}"
        )
    print("\n  * size capped at the per-position limit")


def cmd_signal_risk_grades(limit: int = 20) -> None:
    """Show persisted fire-time earnings/news grades without recomputation."""
    rows = db.get_recent_signal_risk_grades(limit=limit)
    print("SIGNAL RISK GRADES  (persisted at signal fire; no live lookup)")
    print("-" * 78)
    if not rows:
        print("  (no fired signals yet)")
        return
    for row in rows:
        fired = row["timestamp"][:19].replace("T", " ")
        print(f"  {fired}  {row['ticker']}  {row['signal_type']}")
        print(f"    earnings: {row['earnings_risk']}")
        print(f"    news:     {row['news_risk']}")


def cmd_risk_exposure() -> None:
    """Current open-trade exposure vs the configured advisory limits.

    Sums the recorded risk across currently-OPEN active trades (pre-Phase-7
    trades carry no risk and are skipped) and compares it to the notional
    portfolio / cluster / position limits. Advisory only — nothing is enforced.
    """
    open_active = [t for t in db.get_open_trades() if t.track_mode == "active"]
    n_sized = sum(1 for t in open_active if t.risk_pct is not None)
    total_risk = sum(t.risk_pct for t in open_active if t.risk_pct is not None)
    cluster_risk = sum(
        t.risk_pct for t in open_active
        if t.risk_pct is not None and t.ind_concentration == "concentrated"
    )
    largest_pos = max(
        (t.risk_position_pct for t in open_active if t.risk_position_pct is not None),
        default=0.0,
    )

    def _verdict(value: float, limit: float) -> str:
        return "OVER" if value > limit else "ok"

    print("RISK EXPOSURE  (open ACTIVE trades vs advisory limits)")
    print("-" * 64)
    print(f"  Notional account:      ${config.NOTIONAL_ACCOUNT:,.2f}")
    print(f"  Open active trades:     {len(open_active)}  ({n_sized} risk-sized)")
    print(
        f"  Total open risk:        {total_risk:.2f}%  / "
        f"{config.MAX_PORTFOLIO_RISK_PCT:.1f}% limit  "
        f"[{_verdict(total_risk, config.MAX_PORTFOLIO_RISK_PCT)}]"
    )
    print(
        f"  Concentrated cluster:   {cluster_risk:.2f}%  / "
        f"{config.MAX_CORRELATED_CLUSTER_PCT:.1f}% limit  "
        f"[{_verdict(cluster_risk, config.MAX_CORRELATED_CLUSTER_PCT)}]"
    )
    print(
        f"  Largest position:       {largest_pos:.2f}%  / "
        f"{config.MAX_POSITION_PCT:.1f}% limit  "
        f"[{_verdict(largest_pos, config.MAX_POSITION_PCT)}]"
    )
    print("\n  Advisory only - nothing is enforced; no capital is at risk.")


# ---- self-optimization CLI (Phase 9) ----


def cmd_optimize_run(
    degrade_window: int | None = None, baseline_window: int | None = None,
) -> None:
    """Run degradation detection + feature evaluation, persist, and print.

    Operator-triggered. Computes fresh findings over the resolved active book,
    writes the run to history, and prints the full report. Flags only — nothing
    here changes any threshold or pair/ticker status.
    """
    dwd = degrade_window if degrade_window is not None else config.SO_DEGRADE_WINDOW_DAYS
    bwd = baseline_window if baseline_window is not None else config.SO_BASELINE_WINDOW_DAYS
    if bwd <= dwd:
        print(
            f"baseline window ({bwd}d) must exceed the degrade window ({dwd}d) "
            "so the baseline is an older, disjoint period.",
            file=sys.stderr,
        )
        sys.exit(1)

    payload = self_optimization.run_optimization(
        degrade_window_days=dwd, baseline_window_days=bwd,
    )
    self_optimization.persist_run(payload)
    print(self_optimization.render_report(payload))


def cmd_optimize_report() -> None:
    """Print the most recent persisted self-optimization run.

    Read-only history view — it replays the last ``optimize run`` verbatim from
    the optimization_runs table. Flags only; nothing here changes behavior.
    """
    run = db.get_latest_optimization_run()
    if run is None:
        print("No self-optimization runs yet. Run `optimize run` first.")
        return
    payload = json.loads(run["findings_json"])
    print(self_optimization.render_report(payload))


# ---- readiness CLI (Phase 10) ----


def cmd_readiness_status() -> None:
    """Every capability: kind, n vs threshold, status, announced, and — for the
    deterministic ones — whether it is currently ACTIVE (ready == active). ML
    capabilities show '-' for active: crossing summons a build, never auto-acts.
    """
    print("READINESS STATUS")
    print("-" * 96)
    print(
        f"  {'Capability':<24} {'Kind':<13} {'n / threshold':>14} "
        f"{'Status':<9} {'Announced':<10} {'Active':<7}"
    )
    for cap in readiness.REGISTRY:
        count = readiness.resolved_count(cap.name)
        ready = readiness.is_ready(cap.name)
        state = db.get_readiness_state(cap.name)
        announced = bool(state["announced"]) if state else False
        status = "ready" if ready else "warming"
        active = ("yes" if ready else "no") if cap.kind == "deterministic" else "-"
        print(
            f"  {cap.name:<24} {cap.kind:<13} "
            f"{f'{count} / {cap.threshold}':>14} {status:<9} "
            f"{('yes' if announced else 'no'):<10} {active:<7}"
        )
    print("\n  Deterministic: ready == active (auto-activates). "
          "ML: ready == ready-to-build (summons a human, never self-trains).")


def cmd_readiness_check() -> None:
    """Run one readiness evaluation pass on demand (same path as the scan loop),
    then print the status table. Fires any first-crossing notifications."""
    results = readiness.evaluate_readiness()
    newly = [r.capability for r in results if r.newly_announced]
    if newly:
        print(f"Announced (first crossing): {', '.join(newly)}\n")
    else:
        print("No new crossings this pass.\n")
    cmd_readiness_status()


# ---- broker CLI (Phase 11) ----


def _fmt_money(value: float | None) -> str:
    return f"${value:,.2f}" if isinstance(value, (int, float)) else "-"


def cmd_broker_account() -> None:
    """Inspect the live Alpaca PAPER account (buying power, cash, equity)."""
    acct = broker.AlpacaBroker().get_account()
    print("BROKER ACCOUNT  (Alpaca paper)")
    print("-" * 48)
    if not acct.ok:
        print(f"  unavailable: {acct.reason}")
        return
    print(f"  Account:       {acct.account_number}")
    print(f"  Status:        {acct.status}")
    print(f"  Buying power:  {_fmt_money(acct.buying_power)}")
    print(f"  Cash:          {_fmt_money(acct.cash)}")
    print(f"  Equity:        {_fmt_money(acct.equity)}")
    print(f"  Currency:      {acct.currency}")


def cmd_broker_positions() -> None:
    """List current open positions on the Alpaca PAPER account."""
    res = broker.AlpacaBroker().get_positions()
    print("BROKER POSITIONS  (Alpaca paper)")
    print("-" * 64)
    if not res.ok:
        print(f"  unavailable: {res.reason}")
        return
    if not res.positions:
        print("  (no open positions)")
        return
    print(
        f"  {'Symbol':<8} {'Qty':>10}  {'Side':<6} {'Avg entry':>10}  "
        f"{'Mkt value':>12}"
    )
    for p in res.positions:
        print(
            f"  {p.symbol:<8} {p.qty:>10.4f}  {p.side:<6} "
            f"{_fmt_money(p.avg_entry_price):>10}  {_fmt_money(p.market_value):>12}"
        )


def cmd_broker_reconcile() -> None:
    """Reconcile internal open trades against the Alpaca PAPER account.

    Broker is authoritative. Divergences are REPORTED only — no trade record is
    mutated (that wiring is a later phase).
    """
    report = broker.reconcile(broker.AlpacaBroker())
    print("BROKER RECONCILIATION  (broker is authoritative)")
    print("-" * 64)
    if not report.ok:
        print(f"  cannot reconcile: {report.note or 'broker unavailable'}")
        return
    print(
        f"  Internal open (active): {len(report.internal_symbols)}  "
        f"{report.internal_symbols or '[]'}"
    )
    print(
        f"  Broker positions:       {len(report.broker_symbols)}  "
        f"{report.broker_symbols or '[]'}"
    )
    if report.broker_open_orders is not None:
        print(f"  Broker open orders:     {report.broker_open_orders}")
    print()
    if not report.divergences:
        print("  No divergences. Internal records match the broker.")
        return
    print(
        f"  Divergences ({len(report.divergences)}) "
        "— reported only, never auto-resolved:"
    )
    for d in report.divergences:
        print(f"    [{d.kind}] {d.symbol or '-'}: {d.detail}")


# ---- allocate CLI (Phase 12 plan / Phase 16 execute) ----


def _build_live_plan(
    *, sample: bool = False, mark_considered: bool = False,
) -> tuple[
    broker.AccountInfo, list[allocation.Candidate],
    allocation.AllocationResult, str,
]:
    """The ONE plan-building path both ``allocate plan`` and ``allocate execute``
    use: live PAPER account state + candidates through the four-stage allocator,
    with the Phase 15 entry gate applied.

    Phase 17: candidates default to the bot's LIVE fired signals
    (``candidate_source.live_candidates``); ``sample=True`` keeps the labelled
    illustrative fixture available for demos. ``mark_considered`` is passed
    through to the live source — True only on the execute-bound pull (the
    ``--confirm`` run), so previews never consume signals.
    """
    account = broker.AlpacaBroker().get_account()
    if sample:
        candidates = allocation.sample_candidates()
        source_label = "illustrative SAMPLE set (--sample)"
    else:
        candidates = candidate_source.live_candidates(
            mark_considered=mark_considered,
        )
        source_label = (
            f"live fired signals (last {config.CANDIDATE_RECENCY_HOURS}h, "
            "unconsidered)"
        )
    # Phase 15: the risk-of-ruin gate — a revoked new_position_entry capability
    # yields an entries-empty plan (existing positions' watchers run regardless).
    result = allocation.build_plan(
        candidates, account,
        entry_authorized=risk_of_ruin.is_entry_authorized(),
    )
    return account, candidates, result, source_label


def cmd_allocate_plan(*, sample: bool = False) -> None:
    """Run the four-stage allocator against live PAPER account state and print
    the inspectable execution plan. EXECUTES NOTHING — no order is submitted,
    and no signal is marked considered (this is the review step)."""
    _account, candidates, result, source_label = _build_live_plan(sample=sample)
    _print_allocation_result(candidates, result, source_label)


def _print_allocation_result(
    candidates: list[allocation.Candidate],
    result: allocation.AllocationResult,
    source_label: str,
) -> None:
    print("ALLOCATION PLAN  (Phase 12 - PLAN ONLY, executes nothing)")
    print("-" * 84)
    print(f"  Candidates: {len(candidates)} ({source_label})")
    if result.note:
        print(f"  {result.note}")

    print()
    print("  Pools:")
    for p in result.pools:
        print(
            f"    {p.pool:<10} capital={_fmt_money(p.capital):>12} "
            f"cash={_fmt_money(p.cash):>12} deployed={_fmt_money(p.deployed):>12} "
            f"orders={p.orders}"
        )

    print()
    print(
        f"  Plan: {len(result.plan.orders)} orders, "
        f"est cost {_fmt_money(result.plan.total_est_cost)}, "
        f"$risk {_fmt_money(result.plan.total_dollar_risk)}"
    )
    if not result.plan.orders:
        print("    (no orders)")
    else:
        print(
            f"    {'#':>2} {'Pool':<10} {'Tier':<6} {'Ticker':<8} {'Signal':<18} "
            f"{'Side':<4} {'Qty':>11} {'Est cost':>12} {'$risk':>8} {'Score':>6}"
        )
        for o in result.plan.orders:
            print(
                f"    {o.rank:>2} {o.pool:<10} {o.tier:<6} {o.ticker:<8} "
                f"{o.signal_type:<18} {o.side:<4} {o.qty:>11.4f} "
                f"{_fmt_money(o.est_cost):>12} {_fmt_money(o.dollar_risk):>8} "
                f"{o.score:>6.3f}"
            )

    print()
    print(f"  Skipped ({len(result.skipped)}):")
    if not result.skipped:
        print("    (none)")
    else:
        for s in result.skipped:
            print(f"    {s.ticker:<8} {s.signal_type:<18} [{s.stage}] {s.reason}")

    print()
    print("  NOTE: this is a PLAN only - no orders were submitted.")


def cmd_allocate_execute(*, confirm: bool, sample: bool = False) -> None:
    """Phase 16: submit an approved plan's orders via their pool paths.

    WITHOUT ``--confirm`` this REFUSES to submit anything and prints the plan
    for review instead (the same output as ``allocate plan``) — the explicit
    confirmation is mandatory, not a suggestion. With ``--confirm`` it builds a
    FRESH plan (same path as ``allocate plan``), checks the Phase 15
    authorization independently, then routes every order through
    ``plan_execution.execute_plan`` and prints what was submitted / rejected /
    skipped.

    Phase 17: the plan is built from LIVE fired signals, and the ``--confirm``
    pull is the one that marks them considered — a signal enters a submitting
    plan exactly once, whatever that plan later does with it.
    """
    account, candidates, result, source_label = _build_live_plan(
        sample=sample, mark_considered=confirm,
    )

    if not confirm:
        _print_allocation_result(candidates, result, source_label)
        print()
        print("  REFUSED: --confirm is required to submit orders.")
        print("  Review the plan above, then re-run:")
        print("    python -m trading_bot allocate execute --confirm")
        return

    print("PLAN EXECUTION  (Phase 16 - operator-confirmed, Alpaca PAPER)")
    print("-" * 84)
    if not result.ok:
        print(f"  Nothing to execute: {result.note}")
        return

    # Options are available only when the paper account has an options trading
    # level; the chain client is the existing Phase 13 read-only data client.
    options_available = (account.options_trading_level or 0) > 0
    chain_fetch = broker.AlpacaOptionsClient().get_option_chain

    run = plan_execution.execute_plan(
        broker.AlpacaBroker(), result.plan,
        option_chain_fetch=chain_fetch,
        options_available=options_available,
    )

    if not run.ok:
        print(f"  EXECUTION REFUSED - {run.note}")
        print("  Nothing was submitted.")
        return

    submitted = [e for e in run.executions if e.status == "submitted"]
    rejected = [e for e in run.executions if e.status == "rejected"]
    errored = [e for e in run.executions if e.status == "error"]
    skipped = [e for e in run.executions if e.status == "skipped"]

    print(f"  Run: {run.plan_id}   planned orders: {len(run.executions)}")

    print()
    print(f"  Submitted ({len(submitted)}):")
    for e in submitted:
        print(
            f"    {e.order.ticker:<8} {e.order.pool:<10} {e.vehicle or '-':<18} "
            f"qty {e.order.qty:>11.4f}   ref {e.order_ref or '-'}"
        )
    if not submitted:
        print("    (none)")

    print()
    print(f"  Rejected ({len(rejected) + len(errored)}):")
    for e in [*rejected, *errored]:
        print(f"    {e.order.ticker:<8} {e.order.pool:<10} [{e.status}] {e.reason}")
    if not rejected and not errored:
        print("    (none)")

    print()
    print(f"  Skipped ({len(skipped)}):")
    for e in skipped:
        print(f"    {e.order.ticker:<8} {e.order.pool:<10} {e.reason}")
    if not skipped:
        print("    (none)")

    if result.skipped:
        print()
        print(f"  Filtered by the plan ({len(result.skipped)}, never routed):")
        for s in result.skipped:
            print(f"    {s.ticker:<8} {s.signal_type:<18} [{s.stage}] {s.reason}")

    print()
    print("  NOTE: Alpaca PAPER account only - no real money.")


# ---- options CLI (Phase 13) ----


def _pass(flag: bool) -> str:
    return "ok " if flag else "-- "


def cmd_options_chain(ticker: str, min_dte: int = 0) -> None:
    """Show the current option chain with per-contract delta / liquidity / DTE
    floor pass-fail marks (the same gates the selector applies)."""
    chain = broker.AlpacaOptionsClient().get_option_chain(ticker)
    print(f"OPTIONS CHAIN  ({ticker}, Alpaca paper)")
    print("-" * 104)
    if not chain.ok:
        print(f"  unavailable: {chain.reason}")
        return
    today = date.today()
    print(
        f"  {'Symbol':<22} {'Type':<4} {'Strike':>8} {'Expiry':<11} {'DTE':>4} "
        f"{'Delta':>6} {'OI':>7} {'Spr%':>6}  {'dlt':<3} {'dte':<3} {'liq':<3}"
    )
    for c in chain.contracts:
        try:
            dte: int | None = (date.fromisoformat(c.expiry) - today).days
        except ValueError:
            dte = None
        if min_dte > 0 and (dte is None or dte < min_dte):
            continue
        delta_ok = (
            c.delta is not None
            and config.TARGET_DELTA_LOW <= abs(c.delta) <= config.TARGET_DELTA_HIGH
        )
        dte_ok = dte is not None and dte >= config.MIN_DTE
        liq_ok = (
            c.open_interest is not None
            and c.open_interest >= config.MIN_OPEN_INTEREST
            and c.spread_pct is not None
            and c.spread_pct <= config.MAX_SPREAD_PCT
        )
        print(
            f"  {c.symbol:<22} {c.option_type:<4} {c.strike:>8.2f} {c.expiry:<11} "
            f"{('-' if dte is None else dte):>4} "
            f"{('-' if c.delta is None else f'{c.delta:+.2f}'):>6} "
            f"{('-' if c.open_interest is None else c.open_interest):>7} "
            f"{('-' if c.spread_pct is None else f'{c.spread_pct:.1f}'):>6}  "
            f"{_pass(delta_ok)} {_pass(dte_ok)} {_pass(liq_ok)}"
        )


def _option_price_map(positions: list[object]) -> dict[str, float]:
    """Best-effort current option mids by symbol (fail-soft, one chain/underlying)."""
    price_map: dict[str, float] = {}
    client = broker.AlpacaOptionsClient()
    underlyings = sorted({str(getattr(p, "underlying")) for p in positions})  # noqa: B009
    for underlying in underlyings:
        chain = client.get_option_chain(underlying)
        if not chain.ok:
            continue
        for c in chain.contracts:
            if c.mid is not None:
                price_map[c.symbol] = c.mid
    return price_map


def cmd_options_positions() -> None:
    """Open option positions with current (best-effort) P&L, cost basis, and
    TP/SL/deadline status. Contract-aware dollars — distinct from equity."""
    positions = db.get_open_option_positions()
    print("OPEN OPTION POSITIONS")
    print("-" * 104)
    if not positions:
        print("  (none)")
        return
    prices = _option_price_map(list(positions))
    today = date.today()
    print(
        f"  {'Symbol':<22} {'Und':<6} {'Ctr':>4} {'Entry':>7} {'Cur':>7} "
        f"{'Cost':>10} {'P&L$':>10} {'TP':>7} {'SL':>7} {'DtDl':>5}"
    )
    total_cost = 0.0
    total_pnl = 0.0
    for p in positions:
        entry = p.premium_entry
        cost = (entry or 0.0) * p.multiplier * p.contracts
        total_cost += cost
        cur = prices.get(p.symbol)
        pnl = (
            (cur - entry) * p.multiplier * p.contracts
            if cur is not None and entry is not None else None
        )
        if pnl is not None:
            total_pnl += pnl
        try:
            dtd: int | None = (date.fromisoformat(p.expiry) - today).days
        except ValueError:
            dtd = None
        print(
            f"  {p.symbol:<22} {p.underlying:<6} {p.contracts:>4.0f} "
            f"{_fmt_money(entry):>7} {_fmt_money(cur):>7} "
            f"{_fmt_money(cost):>10} {_fmt_money(pnl):>10} "
            f"{_fmt_money(p.tp):>7} {_fmt_money(p.sl):>7} "
            f"{('-' if dtd is None else dtd):>5}"
        )
    print("-" * 104)
    print(
        f"  Open: {len(positions)}   cost basis {_fmt_money(total_cost)}   "
        f"unrealized P&L {_fmt_money(total_pnl)}"
    )


# ---- long-term CLI (Phase 14) ----


def cmd_longterm_candidates() -> None:
    """Show current long-term entry candidates (pending allocation). Generated
    live; Phase 17 also LOGS each one into the signals table so the live
    candidate source can pull it into `allocate plan` / `allocate execute` —
    nothing here submits an order."""
    candidates = long_term.generate_candidates(long_term.fetch_daily_candles)
    print("LONG-TERM ENTRY CANDIDATES  (pending allocation)")
    print("-" * 84)
    if not candidates:
        print("  (none)")
        return
    print(f"  {'Ticker':<10} {'Asset':<7} {'Signal':<16} {'Entry':>10}  Rationale")
    for c in candidates:
        print(
            f"  {c.ticker:<10} {c.asset_class:<7} {c.signal_type:<16} "
            f"{_fmt_money(c.entry_price):>10}  {c.entry_rationale}"
        )
    ids = long_term.persist_candidates(candidates)
    print(
        f"\n  Logged {len(ids)} candidate(s) to signals - they feed "
        "`allocate plan` / `allocate execute`. Nothing is submitted here."
    )


def cmd_longterm_positions() -> None:
    """Open lifecycle-book positions with per-source exit status: genuine
    long-term rows show drawdown + trend-breakdown vs the protective-exit
    thresholds; swing-fallback rows (Phase 19) show their OWN swing TP/SL —
    the levels the watcher actually applies to them."""
    positions = db.get_open_long_term_positions()
    print("OPEN LONG-TERM POSITIONS")
    print("-" * 104)
    if not positions:
        print("  (none)")
        return
    print(
        f"  {'Ticker':<10} {'Asset':<7} {'Source':<14} {'Entry':>9} "
        f"{'Current':>9} "
        f"{'Draw%':>7}/{int(config.MAX_DRAWDOWN_STOP_PCT)} "
        f"{'Below':>5}/{config.TREND_BREAKDOWN_DAYS}  Swing exits"
    )
    for p in positions:
        df = long_term.fetch_daily_candles(p.ticker)
        current = long_term._val(df["Close"], -1) if df is not None else None
        drawdown = (
            (p.entry_price - current) / p.entry_price * 100.0
            if current is not None and p.entry_price > 0 else None
        )
        below = (
            long_term._consecutive_closes_below_trend(df) if df is not None else None
        )
        if p.source == "swing_fallback":
            # Long-term thresholds do not apply to a swing-fallback position.
            draw_txt, below_txt = "n/a", "n/a"
            swing_txt = (
                f"tp={_fmt_money(p.tp)} sl={_fmt_money(p.sl)} "
                f"deadline={p.deadline.date().isoformat() if p.deadline else '-'}"
                + (" (short)" if p.direction == "short" else "")
            )
        else:
            draw_txt = "-" if drawdown is None else f"{drawdown:.1f}"
            below_txt = "-" if below is None else str(below)
            swing_txt = "-"
        print(
            f"  {p.ticker:<10} {p.asset_class:<7} {p.source:<14} "
            f"{_fmt_money(p.entry_price):>9} "
            f"{_fmt_money(current):>9} {draw_txt:>7}   {below_txt:>5}  {swing_txt}"
        )


# ---- market data comparison CLI (Phase 25) ----


def _positive_marketdata_window(value: str) -> int:
    """Argparse converter for a strictly positive daily-bar count."""
    try:
        window = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "--window must be a positive integer",
        ) from None
    if window <= 0:
        raise argparse.ArgumentTypeError("--window must be a positive integer")
    return window


def cmd_marketdata_compare(
    *, tickers: str | None = None, window: int | None = None,
) -> None:
    """Run the yfinance-vs-Alpaca bar comparison and print the diagnostic
    report. DIAGNOSTIC ONLY — no consumer's behavior changes; yfinance stays
    authoritative everywhere until a separate, deliberate cutover phase."""
    universe = (
        [t.strip() for t in tickers.split(",") if t.strip()]
        if tickers else marketdata_compare.default_universe()
    )
    window_bars = window if window is not None else config.MD_COMPARE_WINDOW_BARS

    print("MARKET DATA COMPARISON  (yfinance vs Alpaca IEX free tier)")
    print("-" * 104)
    print(
        f"  Tickers: {len(universe)}   window: {window_bars} daily bars   "
        f"tolerance: {config.MD_COMPARE_TOLERANCE_PCT}%"
    )
    report = marketdata_compare.compare_universe(universe, window_bars=window_bars)

    print()
    print(
        f"  {'Ticker':<10} {'Status':<18} {'Bars':>4} {'MaxDiff%':>9} "
        f"{'MeanDiff%':>10} {'LatestAgree':<11} Note"
    )
    for r in report.results:
        max_txt = "-" if r.max_close_diff_pct is None else f"{r.max_close_diff_pct:.3f}"
        mean_txt = (
            "-" if r.mean_close_diff_pct is None else f"{r.mean_close_diff_pct:.3f}"
        )
        agree_txt = "-" if r.latest_closed_agrees is None else str(r.latest_closed_agrees)
        print(
            f"  {r.ticker:<10} {r.status:<18} {r.bars_compared:>4} {max_txt:>9} "
            f"{mean_txt:>10} {agree_txt:<11} {r.note}"
        )

    print("-" * 104)
    print(
        f"  Summary: {report.matched} matched, {report.divergent} divergent, "
        f"{report.missing} missing (of {len(report.results)})"
    )
    print(
        "  NOTE: diagnostic only - yfinance remains authoritative for every "
        "consumer. No cutover happens here."
    )


# ---- risk-of-ruin CLI (Phase 15) ----


def cmd_risk_ror_status() -> None:
    """Current tier state, drawdown, streaks, and entry authorization."""
    drawdown = risk_of_ruin.current_drawdown_pct()
    print("RISK-OF-RUIN STATUS  (Phase 15 safety layer)")
    print("-" * 64)
    print(f"  State:                 {risk_of_ruin.get_state()}")
    print(
        f"  Entry authorized:      "
        f"{'yes' if risk_of_ruin.is_entry_authorized() else 'NO'}"
    )
    reason = risk_of_ruin.revoke_reason(config.ENTRY_CAPABILITY)
    if reason:
        print(f"  Revoke reason:         {reason}")
    print(
        f"  Consecutive losses:    {risk_of_ruin.consecutive_losses()} "
        f"/ {config.MAX_CONSECUTIVE_LOSSES} limit"
    )
    dd_txt = "-" if drawdown is None else f"{drawdown:.1f}%"
    print(f"  Equity drawdown:       {dd_txt} / {config.MAX_DRAWDOWN_PCT:.0f}% limit")
    print(
        f"  Broker-error streak:   {risk_of_ruin.broker_error_streak()} "
        f"/ {config.MAX_CONSECUTIVE_BROKER_ERRORS} limit"
    )
    print(
        f"  Reconcile divergences: {risk_of_ruin.reconcile_divergence_streak()} "
        f"/ {config.MAX_CONSECUTIVE_RECONCILE_DIVERGENCES} limit"
    )
    catastrophic = risk_of_ruin.check_catastrophic()
    if catastrophic:
        print(f"  CATASTROPHIC TRIGGER:  {catastrophic}")


def cmd_risk_reauthorize(token: str) -> None:
    """Operator re-authorization after a Tier-1 pause or Tier-2 halt."""
    ok, message = risk_of_ruin.reauthorize(token)
    print(message)
    if not ok:
        sys.exit(1)


def cmd_risk_killswitch(token: str) -> None:
    """Human-invoked Tier-2 emergency shutdown — the IDENTICAL orchestrator an
    auto-detected catastrophic trigger runs."""
    result = risk_of_ruin.kill_switch(token, broker.AlpacaBroker())
    if result is None:
        print(
            "kill switch REFUSED: invalid confirmation token "
            f"(expected {config.ROR_KILLSWITCH_TOKEN!r})"
        )
        sys.exit(1)
    print(f"KILL SWITCH: {result.status}  (trigger: {result.trigger})")
    for line in result.closed:
        print(f"  closed:  {line}")
    for line in result.pending:
        print(f"  pending: {line}")
    if result.note:
        print(f"  note: {result.note}")


# ---- predictions CLI (Phase 2.2b) ----

# Seeded defaults — duplicated from scanner._PRED_DEFAULTS so the CLI can
# query the canonical view of "what should this setting be?" without
# importing scanner (scanner imports half the universe at module load).
_PRED_DEFAULT_VALUES: dict[str, str] = {
    "predictions.enabled":           "false",
    "predictions.window_start":      "08:00",
    "predictions.window_end":        "22:00",
    "predictions.tickers":           "BTC-USD,ETH-USD",
    "predictions.notify_resolution": "false",
}


def _validate_hhmm(s: str) -> tuple[int, int]:
    parts = s.strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"expected HH:MM, got {s!r}")
    try:
        hh, mm = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise ValueError(f"invalid time {s!r}: {exc}") from exc
    if not (0 <= hh <= 23 and 0 <= mm <= 59):
        raise ValueError(f"out-of-range time {s!r}")
    return (hh, mm)


def cmd_predictions_enable() -> None:
    settings.set_bool("predictions.enabled", True)
    print("Predictions: ENABLED")


def cmd_predictions_disable() -> None:
    settings.set_bool("predictions.enabled", False)
    print("Predictions: DISABLED")


def cmd_predictions_status() -> None:
    enabled = settings.get_bool("predictions.enabled", default=False)
    win_start = settings.get(
        "predictions.window_start", _PRED_DEFAULT_VALUES["predictions.window_start"],
    )
    win_end = settings.get(
        "predictions.window_end", _PRED_DEFAULT_VALUES["predictions.window_end"],
    )
    tickers = settings.get(
        "predictions.tickers", _PRED_DEFAULT_VALUES["predictions.tickers"],
    )
    notify_res = settings.get_bool(
        "predictions.notify_resolution", default=False,
    )
    pause_until_raw = settings.get("predictions.pause_until")

    label = "ENABLED" if enabled else "DISABLED"
    print(f"Predictions: {label}")
    print("-" * 38)
    print(f"Active window:    {win_start} - {win_end} ET")
    print(f"Tickers:          {tickers}")
    print(f"Resolve notify:   {'on' if notify_res else 'off'}")
    if pause_until_raw:
        print(f"Paused until:     {pause_until_raw}")
    print()

    # Recent activity (last 24h)
    from datetime import UTC as _UTC
    from datetime import datetime as _dt
    from datetime import timedelta as _td

    since = _dt.now(_UTC) - _td(hours=24)
    recent = db.get_predictions(since=since, limit=10_000)
    made = len(recent)
    resolved = sum(1 for p in recent if p.outcome is not None)
    correct = sum(1 for p in recent if p.outcome == "correct")
    incorrect = sum(1 for p in recent if p.outcome == "incorrect")
    decided = correct + incorrect
    acc = f"{correct / decided * 100:.1f}%" if decided > 0 else "N/A"

    print("Recent activity (last 24h):")
    print(f"  Predictions made:     {made}")
    print(f"  Resolved:             {resolved}")
    print(f"  Accuracy:             {acc}")

    if recent:
        last = max(recent, key=lambda p: p.created_at)
        print(f"  Most recent:          {last.created_at.isoformat()}  "
              f"{last.ticker} {last.direction}")


def cmd_predictions_window(start: str, end: str) -> None:
    try:
        s = _validate_hhmm(start)
        e = _validate_hhmm(end)
    except ValueError as exc:
        print(f"Invalid window: {exc}", file=sys.stderr)
        sys.exit(1)
    if e <= s:
        print(
            f"Invalid window: end {end} must be after start {start}",
            file=sys.stderr,
        )
        sys.exit(1)
    settings.set("predictions.window_start", start)
    settings.set("predictions.window_end", end)
    print(f"Active window updated: {start} - {end} ET")


def cmd_predictions_tickers(raw: str) -> None:
    """Validate each ticker resolves on yfinance before persisting."""
    tickers = [t.strip() for t in raw.split(",") if t.strip()]
    if not tickers:
        print("Error: at least one ticker required", file=sys.stderr)
        sys.exit(1)
    bad: list[str] = []
    for t in tickers:
        try:
            df = predictions._fetch_15m_candles(t)  # noqa: SLF001 - intentional reuse
        except Exception as exc:  # noqa: BLE001 - yfinance can raise anything
            print(f"  {t}: error checking — {exc}", file=sys.stderr)
            bad.append(t)
            continue
        if df is None:
            bad.append(t)
    if bad:
        print(
            f"Error: tickers failed yfinance validation: {bad}",
            file=sys.stderr,
        )
        sys.exit(1)
    settings.set("predictions.tickers", ",".join(tickers))
    print(f"Tickers updated: {', '.join(tickers)}")


def cmd_predictions_pause(minutes: int) -> None:
    if minutes <= 0:
        print("Error: --minutes must be positive", file=sys.stderr)
        sys.exit(1)
    from datetime import UTC as _UTC
    from datetime import datetime as _dt
    from datetime import timedelta as _td

    until = _dt.now(_UTC) + _td(minutes=minutes)
    settings.set("predictions.pause_until", until.isoformat())
    print(f"Predictions paused until {until.isoformat()}")


# ---- prediction reports ----


def _fmt_accuracy(stats: PredictionStats) -> str:
    return "    -" if stats.accuracy is None else f"{stats.accuracy:.1f}%"


def cmd_report_predictions(min_context: int = 0) -> None:
    by_ticker = performance.stats_predictions_by_ticker(min_context=min_context)
    overall = performance.stats_predictions_overall(min_context=min_context)
    if overall.total == 0:
        print(f"Prediction accuracy{_min_ctx_suffix(min_context)}\n  (no predictions yet)")
        return
    label_w = max(8, max((len(s.label) for s in by_ticker), default=8))
    print(f"Prediction accuracy{_min_ctx_suffix(min_context)}")
    print(
        f"  {'Ticker':<{label_w}}  {'Total':>5}  {'Correct':>7}  "
        f"{'Incorrect':>9}  {'Push':>4}  {'Accuracy':>8}"
    )
    for s in by_ticker:
        print(
            f"  {s.label:<{label_w}}  {s.total:>5}  {s.correct:>7}  "
            f"{s.incorrect:>9}  {s.push:>4}  {_fmt_accuracy(s):>8}"
        )
    print(
        f"  {'TOTAL':<{label_w}}  {overall.total:>5}  {overall.correct:>7}  "
        f"{overall.incorrect:>9}  {overall.push:>4}  "
        f"{_fmt_accuracy(overall):>8}"
    )


def _print_pred_axis_table(
    rows: list[PredictionStats], title: str, label_header: str,
) -> None:
    if not rows or all(r.total == 0 for r in rows):
        print(f"{title}\n  (no predictions yet)")
        return
    label_w = max(10, max(len(r.label) for r in rows))
    print(title)
    print(
        f"  {label_header:<{label_w}}  {'Total':>5}  {'Correct':>7}  "
        f"{'Incorrect':>9}  {'Push':>4}  {'Accuracy':>8}"
    )
    for s in rows:
        print(
            f"  {s.label:<{label_w}}  {s.total:>5}  {s.correct:>7}  "
            f"{s.incorrect:>9}  {s.push:>4}  {_fmt_accuracy(s):>8}"
        )


def cmd_report_predictions_by_regime(min_context: int = 0) -> None:
    _print_pred_axis_table(
        performance.stats_predictions_by_regime(min_context=min_context),
        title=f"Prediction accuracy by regime{_min_ctx_suffix(min_context)}",
        label_header="Regime",
    )


def cmd_report_predictions_by_vix(min_context: int = 0) -> None:
    _print_pred_axis_table(
        performance.stats_predictions_by_vix(min_context=min_context),
        title=f"Prediction accuracy by VIX band{_min_ctx_suffix(min_context)}",
        label_header="VIX Band",
    )


def cmd_report_predictions_by_time(min_context: int = 0) -> None:
    _print_pred_axis_table(
        performance.stats_predictions_by_hour(min_context=min_context),
        title=f"Prediction accuracy by hour (ET){_min_ctx_suffix(min_context)}",
        label_header="Hour (ET)",
    )


def cmd_report_daily(target_date: date | None) -> None:
    perf = performance.update_daily_performance(target_date)
    print(f"DAILY PERFORMANCE — {perf.date}")
    print(f"  Signals fired:  {perf.signals_fired:>4}")
    print(f"  Trades opened:  {perf.trades_opened:>4}")
    print(f"  Trades closed:  {perf.trades_closed:>4}")
    print(f"  Wins:           {perf.wins:>4}")
    print(f"  Losses:         {perf.losses:>4}")
    print(f"  Win rate:       {_fmt_pct(perf.win_rate):>6}")
    print(f"  Total PnL:      {_fmt_pct(perf.total_pnl_pct, signed=True):>7}")


# ---- dispatcher ----


def main() -> None:
    db.init_db()
    parser = argparse.ArgumentParser(prog="trading_bot")
    sub = parser.add_subparsers(dest="command", required=True)

    # secrets
    secrets_parser = sub.add_parser("secrets")
    secrets_sub = secrets_parser.add_subparsers(dest="secrets_cmd", required=True)
    secrets_sub.add_parser("list")
    set_p = secrets_sub.add_parser("set")
    set_p.add_argument("name")
    get_p = secrets_sub.add_parser("get")
    get_p.add_argument("name")
    del_p = secrets_sub.add_parser("delete")
    del_p.add_argument("name")

    # db
    db_parser = sub.add_parser("db")
    db_sub = db_parser.add_subparsers(dest="db_cmd", required=True)
    db_sub.add_parser("init")
    db_sub.add_parser("status")

    # migrate
    migrate_parser = sub.add_parser("migrate")
    migrate_sub = migrate_parser.add_subparsers(dest="migrate_cmd", required=True)
    migrate_sub.add_parser("csv")

    # outcomes
    outcomes_parser = sub.add_parser("outcomes")
    outcomes_sub = outcomes_parser.add_subparsers(dest="outcomes_cmd", required=True)
    outcomes_sub.add_parser("resolve")
    outcomes_sub.add_parser("backfill")
    outcomes_status_p = outcomes_sub.add_parser("status")
    outcomes_status_p.add_argument(
        "--shadow", action="store_true",
        help="Show shadow-tracked trades instead of active (Phase 3.1-LIVE)",
    )

    # report (Phase 1.4 + Phase 2.1 --by-regime extensions)
    report_parser = sub.add_parser("report")
    report_sub = report_parser.add_subparsers(dest="report_cmd", required=True)
    overall_p = report_sub.add_parser("overall")
    overall_p.add_argument(
        "--shadow", action="store_true",
        help="Show shadow-tracked trades instead of active (Phase 3.1-LIVE)",
    )
    recent_p = report_sub.add_parser("recent")
    recent_p.add_argument("--days", type=int, default=30)
    def _add_min_context(parser_: argparse.ArgumentParser) -> None:
        """Attach --min-context N to a subparser. Phase 2.3."""
        parser_.add_argument(
            "--min-context", type=int, default=0,
            help="Filter trades to context_score >= N (0 = no filter)",
        )

    bs_p = report_sub.add_parser("by-signal")
    bs_axis = bs_p.add_mutually_exclusive_group()
    bs_axis.add_argument("--by-regime", action="store_true",
                         help="Break out win rates by macro regime (Phase 2.1)")
    bs_axis.add_argument("--by-vix", action="store_true",
                         help="Break out win rates by VIX band (Phase 2.2)")
    _add_min_context(bs_p)
    bt_p = report_sub.add_parser("by-ticker")
    bt_group = bt_p.add_mutually_exclusive_group()
    bt_group.add_argument("--stocks", action="store_true",
                          help="Restrict to asset_class='stock'")
    bt_group.add_argument("--crypto", action="store_true",
                          help="Restrict to asset_class='crypto'")
    bt_axis = bt_p.add_mutually_exclusive_group()
    bt_axis.add_argument("--by-regime", action="store_true",
                         help="Break out win rates by macro regime (Phase 2.1)")
    bt_axis.add_argument("--by-vix", action="store_true",
                         help="Break out win rates by VIX band (Phase 2.2)")
    _add_min_context(bt_p)
    report_sub.add_parser("by-asset")
    report_sub.add_parser("cross")
    by_regime_p = report_sub.add_parser("by-regime")
    _add_min_context(by_regime_p)
    by_vix_p = report_sub.add_parser("by-vix")
    _add_min_context(by_vix_p)
    by_rv_p = report_sub.add_parser("by-regime-vix")
    _add_min_context(by_rv_p)
    report_sub.add_parser("by-context")          # Phase 2.3
    report_sub.add_parser("context-matrix")      # Phase 2.3
    report_sub.add_parser("pairs")               # Phase 4
    pred_report_p = report_sub.add_parser("predictions")
    pred_report_axis = pred_report_p.add_mutually_exclusive_group()
    pred_report_axis.add_argument("--by-regime", action="store_true",
                                  help="Break accuracy out by macro regime")
    pred_report_axis.add_argument("--by-vix", action="store_true",
                                  help="Break accuracy out by VIX band")
    pred_report_axis.add_argument("--by-time", action="store_true",
                                  help="Break accuracy out by hour-of-day ET")
    _add_min_context(pred_report_p)
    report_sub.add_parser("daily-backfill")
    daily_p = report_sub.add_parser("daily")
    daily_p.add_argument("--date", type=str, default=None,
                         help="YYYY-MM-DD; defaults to yesterday (UTC)")

    # regime (Phase 2.1)
    regime_parser = sub.add_parser("regime")
    regime_sub = regime_parser.add_subparsers(dest="regime_cmd", required=True)
    regime_sub.add_parser("current")
    hist_p = regime_sub.add_parser("history")
    hist_p.add_argument("--days", type=int, default=30,
                        help="How many recent snapshots to show (default 30)")
    regime_sub.add_parser("backfill")

    # vix (Phase 2.2)
    vix_parser = sub.add_parser("vix")
    vix_sub = vix_parser.add_subparsers(dest="vix_cmd", required=True)
    vix_sub.add_parser("current")
    vhist_p = vix_sub.add_parser("history")
    vhist_p.add_argument("--days", type=int, default=30,
                         help="How many recent snapshots to show (default 30)")
    vix_sub.add_parser("backfill")

    # context (Phase 2.3)
    context_parser = sub.add_parser("context")
    context_sub = context_parser.add_subparsers(dest="context_cmd", required=True)
    context_sub.add_parser("current")
    chist_p = context_sub.add_parser("history")
    chist_p.add_argument("--days", type=int, default=30,
                         help="How many recent overlapping snapshots (default 30)")
    context_sub.add_parser("backfill")

    # discovery (Phase 3.1)
    discovery_parser = sub.add_parser("discovery")
    discovery_sub = discovery_parser.add_subparsers(
        dest="discovery_cmd", required=True
    )
    scan_p = discovery_sub.add_parser("scan")
    scan_p.add_argument(
        "--throttle", type=float, default=discovery._THROTTLE_SECONDS,
        help="Seconds to sleep between yfinance calls (default 1.0)",
    )

    # shadow (Phase 3.1-LIVE)
    shadow_parser = sub.add_parser("shadow")
    shadow_sub = shadow_parser.add_subparsers(dest="shadow_cmd", required=True)
    shadow_sub.add_parser("status")
    shadow_eval_p = shadow_sub.add_parser("evaluate")
    shadow_eval_p.add_argument(
        "--dry-run", action="store_true",
        help="Evaluate without promoting or persisting",
    )

    # watchlist state machine (Phase 3.3)
    watchlist_parser = sub.add_parser("watchlist")
    watchlist_sub = watchlist_parser.add_subparsers(
        dest="watchlist_cmd", required=True
    )
    watchlist_sub.add_parser("status")
    watchlist_eval_p = watchlist_sub.add_parser("evaluate")
    watchlist_eval_p.add_argument(
        "--dry-run", action="store_true",
        help="Evaluate without changing any statuses or recording transitions",
    )

    # per-pair gate (Phase 4)
    pairs_parser = sub.add_parser("pairs")
    pairs_sub = pairs_parser.add_subparsers(dest="pairs_cmd", required=True)
    pairs_sub.add_parser("status")
    pairs_eval_p = pairs_sub.add_parser("evaluate")
    pairs_eval_p.add_argument(
        "--dry-run", action="store_true",
        help="Evaluate without changing any statuses or recording transitions",
    )

    # sentiment (Phase 5)
    sentiment_parser = sub.add_parser("sentiment")
    sentiment_sub = sentiment_parser.add_subparsers(
        dest="sentiment_cmd", required=True
    )
    sent_status_p = sentiment_sub.add_parser("status")
    sent_status_p.add_argument(
        "--limit", type=int, default=20,
        help="How many recent scored signals to show (default 20)",
    )

    # indicators (Phase 6)
    indicators_parser = sub.add_parser("indicators")
    indicators_sub = indicators_parser.add_subparsers(
        dest="indicators_cmd", required=True
    )
    ind_status_p = indicators_sub.add_parser("status")
    ind_status_p.add_argument(
        "--limit", type=int, default=20,
        help="How many recent indicator-scored signals to show (default 20)",
    )

    # risk (Phase 7)
    risk_parser = sub.add_parser("risk")
    risk_sub = risk_parser.add_subparsers(dest="risk_cmd", required=True)
    risk_status_p = risk_sub.add_parser("status")
    risk_status_p.add_argument(
        "--limit", type=int, default=20,
        help="How many recent risk-assessed signals to show (default 20)",
    )
    risk_grades_p = risk_sub.add_parser("grades")
    risk_grades_p.add_argument(
        "--limit", type=int, default=20,
        help="How many recent fired signals to show (default 20)",
    )
    risk_sub.add_parser("exposure")
    # Phase 15 — risk-of-ruin operator controls.
    risk_reauth_p = risk_sub.add_parser("reauthorize")
    risk_reauth_p.add_argument("token", help="Confirmation token (operator-only)")
    risk_kill_p = risk_sub.add_parser("killswitch")
    risk_kill_p.add_argument("token", help="Confirmation token (operator-only)")

    # optimize (Phase 9)
    optimize_parser = sub.add_parser("optimize")
    optimize_sub = optimize_parser.add_subparsers(
        dest="optimize_cmd", required=True
    )
    optimize_run_p = optimize_sub.add_parser("run")
    optimize_run_p.add_argument(
        "--degrade-window", type=int, default=None,
        help=f"Recent window in days (default {config.SO_DEGRADE_WINDOW_DAYS})",
    )
    optimize_run_p.add_argument(
        "--baseline-window", type=int, default=None,
        help=f"Baseline window in days (default {config.SO_BASELINE_WINDOW_DAYS})",
    )
    optimize_sub.add_parser("report")

    # readiness (Phase 10)
    readiness_parser = sub.add_parser("readiness")
    readiness_sub = readiness_parser.add_subparsers(
        dest="readiness_cmd", required=True
    )
    readiness_sub.add_parser("status")
    readiness_sub.add_parser("check")

    # broker (Phase 11) — Alpaca PAPER only
    broker_parser = sub.add_parser("broker")
    broker_sub = broker_parser.add_subparsers(dest="broker_cmd", required=True)
    broker_sub.add_parser("account")
    broker_sub.add_parser("positions")
    broker_sub.add_parser("reconcile")

    # allocate (Phase 12 plan / Phase 16 execute / Phase 17 live candidates)
    allocate_parser = sub.add_parser("allocate")
    allocate_sub = allocate_parser.add_subparsers(dest="allocate_cmd", required=True)
    allocate_plan_p = allocate_sub.add_parser("plan")
    allocate_plan_p.add_argument(
        "--sample", action="store_true",
        help="Use the illustrative sample candidate set instead of live fired "
             "signals (demo/inspection only).",
    )
    allocate_execute_p = allocate_sub.add_parser("execute")
    allocate_execute_p.add_argument(
        "--confirm", action="store_true",
        help="Actually submit the plan's orders (PAPER). Without this flag the "
             "plan is printed for review and NOTHING is submitted.",
    )
    allocate_execute_p.add_argument(
        "--sample", action="store_true",
        help="Use the illustrative sample candidate set instead of live fired "
             "signals (demo/inspection only).",
    )

    # options (Phase 13) — single-leg, PAPER
    options_parser = sub.add_parser("options")
    options_sub = options_parser.add_subparsers(dest="options_cmd", required=True)
    opt_chain_p = options_sub.add_parser("chain")
    opt_chain_p.add_argument("ticker", help="Underlying ticker, e.g. AAPL")
    opt_chain_p.add_argument(
        "--min-dte",
        type=int,
        default=0,
        help="Exclude contracts with fewer days to expiry than this (e.g. 1 to skip 0DTE).",
    )
    options_sub.add_parser("positions")

    # long-term (Phase 14) — buy-and-hold, stocks + crypto
    longterm_parser = sub.add_parser("longterm")
    longterm_sub = longterm_parser.add_subparsers(dest="longterm_cmd", required=True)
    longterm_sub.add_parser("candidates")
    longterm_sub.add_parser("positions")

    # marketdata (Phase 25) — yfinance-vs-Alpaca comparison, diagnostic only
    md_parser = sub.add_parser("marketdata")
    md_sub = md_parser.add_subparsers(dest="marketdata_cmd", required=True)
    md_compare_p = md_sub.add_parser("compare")
    md_compare_p.add_argument(
        "--tickers", type=str, default=None,
        help="Comma-separated subset (default: watchlist + shadow universe "
             "+ crypto)",
    )
    md_compare_p.add_argument(
        "--window", type=_positive_marketdata_window, default=None,
        help=f"Daily bars to compare (default {config.MD_COMPARE_WINDOW_BARS})",
    )

    # predictions (Phase 2.2b)
    pred_parser = sub.add_parser("predictions")
    pred_sub = pred_parser.add_subparsers(dest="predictions_cmd", required=True)
    pred_sub.add_parser("enable")
    pred_sub.add_parser("disable")
    pred_sub.add_parser("status")
    win_p = pred_sub.add_parser("window")
    win_p.add_argument("--start", type=str, required=True,
                       help="HH:MM in 24h ET (e.g. 08:00)")
    win_p.add_argument("--end", type=str, required=True,
                       help="HH:MM in 24h ET (e.g. 22:00)")
    tick_p = pred_sub.add_parser("tickers")
    tick_p.add_argument("--set", dest="ticker_list", type=str, required=True,
                        help="Comma-separated, e.g. BTC-USD,ETH-USD")
    pause_p = pred_sub.add_parser("pause")
    pause_p.add_argument("--minutes", type=int, required=True,
                         help="How long to pause predictions for")

    args = parser.parse_args()

    if args.command == "secrets":
        if args.secrets_cmd == "list":
            cmd_secrets_list()
        elif args.secrets_cmd == "set":
            cmd_secrets_set(args.name)
        elif args.secrets_cmd == "get":
            cmd_secrets_get(args.name)
        elif args.secrets_cmd == "delete":
            cmd_secrets_delete(args.name)
    elif args.command == "db":
        if args.db_cmd == "init":
            cmd_db_init()
        elif args.db_cmd == "status":
            cmd_db_status()
    elif args.command == "migrate" and args.migrate_cmd == "csv":
        cmd_migrate_csv()
    elif args.command == "outcomes":
        if args.outcomes_cmd == "resolve":
            cmd_outcomes_resolve()
        elif args.outcomes_cmd == "backfill":
            cmd_outcomes_backfill()
        elif args.outcomes_cmd == "status":
            cmd_outcomes_status(
                track_mode="shadow" if args.shadow else "active"
            )
    elif args.command == "report":
        if args.report_cmd == "overall":
            cmd_report_overall(
                track_mode="shadow" if args.shadow else "active"
            )
        elif args.report_cmd == "recent":
            cmd_report_recent(args.days)
        elif args.report_cmd == "by-signal":
            mc = _maybe_min_context(args)
            if getattr(args, "by_regime", False):
                cmd_report_by_signal_regime(min_context=mc)
            elif getattr(args, "by_vix", False):
                cmd_report_by_signal_vix(min_context=mc)
            else:
                cmd_report_by_signal(min_context=mc)
        elif args.report_cmd == "by-ticker":
            asset = "stock" if args.stocks else "crypto" if args.crypto else None
            mc = _maybe_min_context(args)
            if getattr(args, "by_regime", False):
                cmd_report_by_ticker_regime(asset, min_context=mc)
            elif getattr(args, "by_vix", False):
                cmd_report_by_ticker_vix(asset, min_context=mc)
            else:
                cmd_report_by_ticker(asset, min_context=mc)
        elif args.report_cmd == "by-asset":
            cmd_report_by_asset()
        elif args.report_cmd == "cross":
            cmd_report_cross()
        elif args.report_cmd == "by-regime":
            cmd_report_by_regime(min_context=_maybe_min_context(args))
        elif args.report_cmd == "by-vix":
            cmd_report_by_vix(min_context=_maybe_min_context(args))
        elif args.report_cmd == "by-regime-vix":
            cmd_report_by_regime_vix(min_context=_maybe_min_context(args))
        elif args.report_cmd == "by-context":
            cmd_report_by_context()
        elif args.report_cmd == "context-matrix":
            cmd_report_context_matrix()
        elif args.report_cmd == "pairs":
            cmd_report_pairs()
        elif args.report_cmd == "predictions":
            mc = _maybe_min_context(args)
            if getattr(args, "by_regime", False):
                cmd_report_predictions_by_regime(min_context=mc)
            elif getattr(args, "by_vix", False):
                cmd_report_predictions_by_vix(min_context=mc)
            elif getattr(args, "by_time", False):
                cmd_report_predictions_by_time(min_context=mc)
            else:
                cmd_report_predictions(min_context=mc)
        elif args.report_cmd == "daily-backfill":
            cmd_report_daily_backfill()
        elif args.report_cmd == "daily":
            target = date.fromisoformat(args.date) if args.date else None
            cmd_report_daily(target)
    elif args.command == "regime":
        if args.regime_cmd == "current":
            cmd_regime_current()
        elif args.regime_cmd == "history":
            cmd_regime_history(args.days)
        elif args.regime_cmd == "backfill":
            cmd_regime_backfill()
    elif args.command == "vix":
        if args.vix_cmd == "current":
            cmd_vix_current()
        elif args.vix_cmd == "history":
            cmd_vix_history(args.days)
        elif args.vix_cmd == "backfill":
            cmd_vix_backfill()
    elif args.command == "context":
        if args.context_cmd == "current":
            cmd_context_current()
        elif args.context_cmd == "history":
            cmd_context_history(args.days)
        elif args.context_cmd == "backfill":
            cmd_context_backfill()
    elif args.command == "discovery":
        if args.discovery_cmd == "scan":
            cmd_discovery_scan(throttle_seconds=args.throttle)
    elif args.command == "shadow":
        if args.shadow_cmd == "status":
            cmd_shadow_status()
        elif args.shadow_cmd == "evaluate":
            cmd_shadow_evaluate(dry_run=args.dry_run)
    elif args.command == "watchlist":
        if args.watchlist_cmd == "status":
            cmd_watchlist_status()
        elif args.watchlist_cmd == "evaluate":
            cmd_watchlist_evaluate(dry_run=args.dry_run)
    elif args.command == "pairs":
        if args.pairs_cmd == "status":
            cmd_pairs_status()
        elif args.pairs_cmd == "evaluate":
            cmd_pairs_evaluate(dry_run=args.dry_run)
    elif args.command == "sentiment":
        if args.sentiment_cmd == "status":
            cmd_sentiment_status(limit=args.limit)
    elif args.command == "indicators":
        if args.indicators_cmd == "status":
            cmd_indicators_status(limit=args.limit)
    elif args.command == "risk":
        if args.risk_cmd == "status":
            # Phase 15 tier state first, then the Phase 7 advisory signal view.
            cmd_risk_ror_status()
            print()
            cmd_risk_status(limit=args.limit)
        elif args.risk_cmd == "grades":
            cmd_signal_risk_grades(limit=args.limit)
        elif args.risk_cmd == "exposure":
            cmd_risk_exposure()
        elif args.risk_cmd == "reauthorize":
            cmd_risk_reauthorize(args.token)
        elif args.risk_cmd == "killswitch":
            cmd_risk_killswitch(args.token)
    elif args.command == "optimize":
        if args.optimize_cmd == "run":
            cmd_optimize_run(
                degrade_window=args.degrade_window,
                baseline_window=args.baseline_window,
            )
        elif args.optimize_cmd == "report":
            cmd_optimize_report()
    elif args.command == "readiness":
        if args.readiness_cmd == "status":
            cmd_readiness_status()
        elif args.readiness_cmd == "check":
            cmd_readiness_check()
    elif args.command == "broker":
        if args.broker_cmd == "account":
            cmd_broker_account()
        elif args.broker_cmd == "positions":
            cmd_broker_positions()
        elif args.broker_cmd == "reconcile":
            cmd_broker_reconcile()
    elif args.command == "allocate":
        if args.allocate_cmd == "plan":
            cmd_allocate_plan(sample=args.sample)
        elif args.allocate_cmd == "execute":
            cmd_allocate_execute(confirm=args.confirm, sample=args.sample)
    elif args.command == "options":
        if args.options_cmd == "chain":
            cmd_options_chain(args.ticker, min_dte=args.min_dte)
        elif args.options_cmd == "positions":
            cmd_options_positions()
    elif args.command == "longterm":
        if args.longterm_cmd == "candidates":
            cmd_longterm_candidates()
        elif args.longterm_cmd == "positions":
            cmd_longterm_positions()
    elif args.command == "marketdata":
        if args.marketdata_cmd == "compare":
            cmd_marketdata_compare(tickers=args.tickers, window=args.window)
    elif args.command == "predictions":
        if args.predictions_cmd == "enable":
            cmd_predictions_enable()
        elif args.predictions_cmd == "disable":
            cmd_predictions_disable()
        elif args.predictions_cmd == "status":
            cmd_predictions_status()
        elif args.predictions_cmd == "window":
            cmd_predictions_window(args.start, args.end)
        elif args.predictions_cmd == "tickers":
            cmd_predictions_tickers(args.ticker_list)
        elif args.predictions_cmd == "pause":
            cmd_predictions_pause(args.minutes)


if __name__ == "__main__":
    main()
