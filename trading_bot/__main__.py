"""CLI entry point: python -m trading_bot <command>"""

import argparse
import getpass
import sys
from datetime import date

from trading_bot import db, outcomes, performance, regime, secrets
from trading_bot.migrate_csv import migrate_csv
from trading_bot.performance import PerfStats, RegimeBreakdown

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


def _format_regime_age(cached_at: object) -> str:
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
    age = _format_regime_age(regime.last_cached_at())
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


def _format_regime_matrix(
    rows: list[RegimeBreakdown], title: str, label_header: str
) -> str:
    if not rows:
        return f"{title}\n  (no closed trades)"
    label_w = max(18, max(len(r.label) for r in rows))
    cell_w = 12
    header = (
        f"  {label_header:<{label_w}}"
        f"  {'Bull':<{cell_w}}{'Sideways':<{cell_w}}{'Bear':<{cell_w}}"
        f"{'Unknown':<{cell_w}}{'Overall':<{cell_w}}"
    )
    lines = [title, header]
    for r in rows:
        bull   = _fmt_cell(r.by_regime.get("bull"))
        side   = _fmt_cell(r.by_regime.get("sideways"))
        bear   = _fmt_cell(r.by_regime.get("bear"))
        unkn   = _fmt_cell(r.by_regime.get("unknown"))
        ovr    = _fmt_cell(r.overall)
        lines.append(
            f"  {r.label:<{label_w}}"
            f"  {bull:<{cell_w}}{side:<{cell_w}}{bear:<{cell_w}}"
            f"{unkn:<{cell_w}}{ovr:<{cell_w}}"
        )
    return "\n".join(lines)


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
    bs_p.add_argument("--by-regime", action="store_true",
                      help="Break out win rates by macro regime (Phase 2.1)")
    bt_p = report_sub.add_parser("by-ticker")
    bt_group = bt_p.add_mutually_exclusive_group()
    bt_group.add_argument("--stocks", action="store_true",
                          help="Restrict to asset_class='stock'")
    bt_group.add_argument("--crypto", action="store_true",
                          help="Restrict to asset_class='crypto'")
    bt_p.add_argument("--by-regime", action="store_true",
                      help="Break out win rates by macro regime (Phase 2.1)")
    report_sub.add_parser("by-asset")
    report_sub.add_parser("cross")
    report_sub.add_parser("by-regime")
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
            else:
                cmd_report_by_signal()
        elif args.report_cmd == "by-ticker":
            asset = "stock" if args.stocks else "crypto" if args.crypto else None
            if getattr(args, "by_regime", False):
                cmd_report_by_ticker_regime(asset)
            else:
                cmd_report_by_ticker(asset)
        elif args.report_cmd == "by-asset":
            cmd_report_by_asset()
        elif args.report_cmd == "cross":
            cmd_report_cross()
        elif args.report_cmd == "by-regime":
            cmd_report_by_regime()
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


if __name__ == "__main__":
    main()
