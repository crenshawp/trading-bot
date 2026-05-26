"""CLI entry point: python -m trading_bot <command>"""

import argparse
import getpass
import sys
from datetime import date

from trading_bot import db, outcomes, performance, secrets
from trading_bot.migrate_csv import migrate_csv
from trading_bot.performance import PerfStats

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

    # report (Phase 1.4)
    report_parser = sub.add_parser("report")
    report_sub = report_parser.add_subparsers(dest="report_cmd", required=True)
    report_sub.add_parser("overall")
    recent_p = report_sub.add_parser("recent")
    recent_p.add_argument("--days", type=int, default=30)
    report_sub.add_parser("by-signal")
    bt_p = report_sub.add_parser("by-ticker")
    bt_group = bt_p.add_mutually_exclusive_group()
    bt_group.add_argument("--stocks", action="store_true",
                          help="Restrict to asset_class='stock'")
    bt_group.add_argument("--crypto", action="store_true",
                          help="Restrict to asset_class='crypto'")
    report_sub.add_parser("by-asset")
    report_sub.add_parser("cross")
    report_sub.add_parser("daily-backfill")
    daily_p = report_sub.add_parser("daily")
    daily_p.add_argument("--date", type=str, default=None,
                         help="YYYY-MM-DD; defaults to yesterday (UTC)")

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
            cmd_report_by_signal()
        elif args.report_cmd == "by-ticker":
            asset = "stock" if args.stocks else "crypto" if args.crypto else None
            cmd_report_by_ticker(asset)
        elif args.report_cmd == "by-asset":
            cmd_report_by_asset()
        elif args.report_cmd == "cross":
            cmd_report_cross()
        elif args.report_cmd == "daily-backfill":
            cmd_report_daily_backfill()
        elif args.report_cmd == "daily":
            target = date.fromisoformat(args.date) if args.date else None
            cmd_report_daily(target)


if __name__ == "__main__":
    main()
