"""CLI entry point: python -m trading_bot <command>"""

import argparse
import getpass
import sys
from datetime import date

from trading_bot import (
    db,
    outcomes,
    performance,
    predictions,
    regime,
    secrets,
    settings,
    vix,
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


def cmd_outcomes_status() -> None:
    report = outcomes.summary()
    print("Outcomes by signal type (closed trades only):")
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


def cmd_report_overall() -> None:
    print(_format_perf_summary(performance.stats_overall(), "OVERALL PERFORMANCE"))


def cmd_report_recent(days: int) -> None:
    stats = performance.stats_recent(days=days)
    print(_format_perf_summary(stats, f"RECENT PERFORMANCE — last {days} days"))


def cmd_report_by_signal() -> None:
    print(_format_perf_table(
        performance.stats_by_signal_type(),
        "BY SIGNAL TYPE",
        label_header="Signal type",
    ))


def cmd_report_by_ticker(asset_class: str | None) -> None:
    title = "BY TICKER"
    if asset_class is not None:
        title = f"BY TICKER ({asset_class})"
    print(_format_perf_table(
        performance.stats_by_ticker(asset_class=asset_class),
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


def cmd_report_by_regime() -> None:
    stats = performance.stats_by_regime()
    if not stats:
        print("BY MARKET REGIME\n  (no closed trades)")
        return
    print("BY MARKET REGIME")
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


def cmd_report_by_signal_regime() -> None:
    rows = performance.stats_by_signal_type_with_regime()
    print(_format_regime_matrix(
        rows,
        title="BY SIGNAL TYPE x REGIME",
        label_header="Signal",
    ))


def cmd_report_by_ticker_regime(asset_class: str | None) -> None:
    rows = performance.stats_by_ticker_with_regime(asset_class=asset_class)
    title = "BY TICKER x REGIME"
    if asset_class is not None:
        title = f"BY TICKER x REGIME ({asset_class})"
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


def cmd_report_by_vix() -> None:
    stats = performance.stats_by_vix_band()
    if not stats:
        print("BY VIX BAND\n  (no closed trades)")
        return
    print("BY VIX BAND")
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


def cmd_report_by_signal_vix() -> None:
    rows = performance.stats_by_signal_type_with_vix()
    print(_format_vix_matrix(
        rows, title="BY SIGNAL TYPE x VIX", label_header="Signal",
    ))


def cmd_report_by_ticker_vix(asset_class: str | None) -> None:
    rows = performance.stats_by_ticker_with_vix(asset_class=asset_class)
    title = "BY TICKER x VIX"
    if asset_class is not None:
        title = f"BY TICKER x VIX ({asset_class})"
    print(_format_vix_matrix(rows, title=title, label_header="Ticker"))


def cmd_report_by_regime_vix() -> None:
    """The headline Phase 2.2 view: every regime x VIX bucket with data,
    sorted by trade count desc. See ``stats_by_regime_x_vix`` for the
    aggregation rules."""
    stats = performance.stats_by_regime_x_vix()
    if not stats:
        print("BY REGIME x VIX\n  (no closed trades)")
        return
    print("BY REGIME x VIX")
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


def cmd_report_predictions() -> None:
    by_ticker = performance.stats_predictions_by_ticker()
    overall = performance.stats_predictions_overall()
    if overall.total == 0:
        print("Prediction accuracy\n  (no predictions yet)")
        return
    label_w = max(8, max((len(s.label) for s in by_ticker), default=8))
    print("Prediction accuracy")
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


def cmd_report_predictions_by_regime() -> None:
    _print_pred_axis_table(
        performance.stats_predictions_by_regime(),
        title="Prediction accuracy by regime", label_header="Regime",
    )


def cmd_report_predictions_by_vix() -> None:
    _print_pred_axis_table(
        performance.stats_predictions_by_vix(),
        title="Prediction accuracy by VIX band", label_header="VIX Band",
    )


def cmd_report_predictions_by_time() -> None:
    _print_pred_axis_table(
        performance.stats_predictions_by_hour(),
        title="Prediction accuracy by hour (ET)", label_header="Hour (ET)",
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
    outcomes_sub.add_parser("status")

    # report (Phase 1.4 + Phase 2.1 --by-regime extensions)
    report_parser = sub.add_parser("report")
    report_sub = report_parser.add_subparsers(dest="report_cmd", required=True)
    report_sub.add_parser("overall")
    recent_p = report_sub.add_parser("recent")
    recent_p.add_argument("--days", type=int, default=30)
    bs_p = report_sub.add_parser("by-signal")
    bs_axis = bs_p.add_mutually_exclusive_group()
    bs_axis.add_argument("--by-regime", action="store_true",
                         help="Break out win rates by macro regime (Phase 2.1)")
    bs_axis.add_argument("--by-vix", action="store_true",
                         help="Break out win rates by VIX band (Phase 2.2)")
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
    report_sub.add_parser("by-asset")
    report_sub.add_parser("cross")
    report_sub.add_parser("by-regime")
    report_sub.add_parser("by-vix")
    report_sub.add_parser("by-regime-vix")
    pred_report_p = report_sub.add_parser("predictions")
    pred_report_axis = pred_report_p.add_mutually_exclusive_group()
    pred_report_axis.add_argument("--by-regime", action="store_true",
                                  help="Break accuracy out by macro regime")
    pred_report_axis.add_argument("--by-vix", action="store_true",
                                  help="Break accuracy out by VIX band")
    pred_report_axis.add_argument("--by-time", action="store_true",
                                  help="Break accuracy out by hour-of-day ET")
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
            cmd_outcomes_status()
    elif args.command == "report":
        if args.report_cmd == "overall":
            cmd_report_overall()
        elif args.report_cmd == "recent":
            cmd_report_recent(args.days)
        elif args.report_cmd == "by-signal":
            if getattr(args, "by_regime", False):
                cmd_report_by_signal_regime()
            elif getattr(args, "by_vix", False):
                cmd_report_by_signal_vix()
            else:
                cmd_report_by_signal()
        elif args.report_cmd == "by-ticker":
            asset = "stock" if args.stocks else "crypto" if args.crypto else None
            if getattr(args, "by_regime", False):
                cmd_report_by_ticker_regime(asset)
            elif getattr(args, "by_vix", False):
                cmd_report_by_ticker_vix(asset)
            else:
                cmd_report_by_ticker(asset)
        elif args.report_cmd == "by-asset":
            cmd_report_by_asset()
        elif args.report_cmd == "cross":
            cmd_report_cross()
        elif args.report_cmd == "by-regime":
            cmd_report_by_regime()
        elif args.report_cmd == "by-vix":
            cmd_report_by_vix()
        elif args.report_cmd == "by-regime-vix":
            cmd_report_by_regime_vix()
        elif args.report_cmd == "predictions":
            if getattr(args, "by_regime", False):
                cmd_report_predictions_by_regime()
            elif getattr(args, "by_vix", False):
                cmd_report_predictions_by_vix()
            elif getattr(args, "by_time", False):
                cmd_report_predictions_by_time()
            else:
                cmd_report_predictions()
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
