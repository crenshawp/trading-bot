"""Smoke tests for the ``python -m trading_bot`` CLI dispatcher.

These exercise main() with patched sys.argv so the argparse plumbing and
each subcommand are exercised end-to-end. Interactive paths (``secrets set``,
``secrets get``) are not covered here — they require getpass interaction.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from trading_bot.__main__ import main


def _run(argv: list[str]) -> None:
    with patch.object(sys, "argv", ["trading_bot", *argv]):
        main()


def test_cli_db_init_prints_version(tmp_db: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _run(["db", "init"])
    out = capsys.readouterr().out
    assert "Schema version: 2" in out


def test_cli_db_status_shows_counts(tmp_db: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _run(["db", "status"])
    out = capsys.readouterr().out
    assert "Schema version: 2" in out
    assert "signals" in out
    assert "trades" in out
    assert "daily_performance" in out


def test_cli_migrate_csv_no_csv(
    tmp_db: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("trading_bot.config.CSV_LEGACY_PATH", tmp_path / "missing.csv")
    _run(["migrate", "csv"])
    out = capsys.readouterr().out
    assert "Imported 0 signals" in out
    assert "skipped 0" in out


def test_cli_secrets_list(
    mock_keyring: dict[str, str],
    local_env: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _run(["secrets", "list"])
    out = capsys.readouterr().out
    assert "DISCORD_WEBHOOK_URL" in out
    assert "NOT SET" in out


# ───────────────────── Phase 1.3 outcomes subcommands ─────────────────────


def test_outcomes_resolve_subcommand(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[bool] = []

    def fake_resolve(**_kwargs: object) -> dict[str, int]:
        calls.append(True)
        return {"wins": 2, "losses": 1, "expired": 0, "still_open": 4}

    monkeypatch.setattr("trading_bot.outcomes.resolve_all_open_trades", fake_resolve)
    _run(["outcomes", "resolve"])
    out = capsys.readouterr().out
    assert calls, "outcomes resolve should call resolve_all_open_trades"
    assert "wins=2" in out
    assert "losses=1" in out
    assert "still_open=4" in out


def test_outcomes_backfill_subcommand(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "trading_bot.outcomes.backfill_signals_without_trades", lambda: 17
    )
    _run(["outcomes", "backfill"])
    assert "Backfilled 17 signals" in capsys.readouterr().out


def test_outcomes_status_subcommand(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "trading_bot.outcomes.summary",
        lambda: {
            "by_signal_type": [
                {
                    "signal_type": "ema21_pullback",
                    "wins": 8, "losses": 4,
                    "win_rate": 66.7, "avg_pnl": 2.31,
                },
                {
                    "signal_type": "oversold_reversal",
                    "wins": 3, "losses": 2,
                    "win_rate": 60.0, "avg_pnl": -1.5,
                },
            ],
            "open": 5,
            "expired": 2,
        },
    )
    _run(["outcomes", "status"])
    out = capsys.readouterr().out
    assert "ema21_pullback" in out
    assert "66.7%" in out
    assert "+2.31%" in out
    assert "-1.50%" in out
    assert "Open trades:    5" in out
    assert "Expired trades: 2" in out


def test_outcomes_status_empty_database(
    tmp_db: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _run(["outcomes", "status"])
    out = capsys.readouterr().out
    assert "no closed trades yet" in out


# ───────────────────── Phase 1.4 report subcommands ─────────────────────


def test_report_overall_subcommand(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_bot.performance import PerfStats
    monkeypatch.setattr(
        "trading_bot.performance.stats_overall",
        lambda **_kwargs: PerfStats(
            label="overall", total=47, wins=29, losses=14, expired=4,
            win_rate=67.4, avg_pnl_pct=1.83, best_pnl_pct=8.21, worst_pnl_pct=-4.05,
        ),
    )
    _run(["report", "overall"])
    out = capsys.readouterr().out
    assert "OVERALL PERFORMANCE" in out
    assert "Total trades:" in out
    assert "47" in out
    assert "67.4%" in out
    assert "+1.83%" in out


def test_report_by_signal_subcommand(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_bot.performance import PerfStats
    monkeypatch.setattr(
        "trading_bot.performance.stats_by_signal_type",
        lambda: [
            PerfStats(label="ema21_pullback", total=24, wins=16, losses=7, expired=1,
                      win_rate=69.6, avg_pnl_pct=2.41, best_pnl_pct=8.21, worst_pnl_pct=-3.18),
            PerfStats(label="oversold_reversal", total=15, wins=9, losses=5, expired=1,
                      win_rate=64.3, avg_pnl_pct=1.22, best_pnl_pct=5.04, worst_pnl_pct=-4.05),
        ],
    )
    _run(["report", "by-signal"])
    out = capsys.readouterr().out
    assert "BY SIGNAL TYPE" in out
    assert "ema21_pullback" in out
    assert "oversold_reversal" in out
    assert "69.6%" in out
    assert "+2.41%" in out


def test_report_by_ticker_with_stocks_filter(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    received_kwargs: dict[str, object] = {}

    def spy(*, asset_class: str | None = None) -> list:  # type: ignore[type-arg]
        received_kwargs["asset_class"] = asset_class
        return []

    monkeypatch.setattr("trading_bot.performance.stats_by_ticker", spy)
    _run(["report", "by-ticker", "--stocks"])
    assert received_kwargs["asset_class"] == "stock"
    assert "BY TICKER (stock)" in capsys.readouterr().out


def test_report_by_ticker_with_crypto_filter(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    received_kwargs: dict[str, object] = {}

    def spy(*, asset_class: str | None = None) -> list:  # type: ignore[type-arg]
        received_kwargs["asset_class"] = asset_class
        return []

    monkeypatch.setattr("trading_bot.performance.stats_by_ticker", spy)
    _run(["report", "by-ticker", "--crypto"])
    assert received_kwargs["asset_class"] == "crypto"


def test_report_daily_with_explicit_date(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from datetime import date as _date

    from trading_bot.models import DailyPerf

    received: dict[str, object] = {}

    def fake_update(target_date: _date | None = None) -> DailyPerf:
        received["target"] = target_date
        return DailyPerf(
            date="2026-04-15", signals_fired=3, trades_opened=3, trades_closed=2,
            wins=1, losses=1, win_rate=50.0, total_pnl_pct=2.5,
        )

    monkeypatch.setattr("trading_bot.performance.update_daily_performance", fake_update)
    _run(["report", "daily", "--date", "2026-04-15"])
    assert received["target"] == _date(2026, 4, 15)
    out = capsys.readouterr().out
    assert "DAILY PERFORMANCE — 2026-04-15" in out
    assert "Wins:" in out
    assert "50.0%" in out


def test_report_overall_empty_database_message(
    tmp_db: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _run(["report", "overall"])
    assert "(no closed trades)" in capsys.readouterr().out


def test_report_daily_backfill_subcommand(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "trading_bot.performance.backfill_daily_performance", lambda: 17
    )
    _run(["report", "daily-backfill"])
    assert "Backfilled daily_performance for 17 dates" in capsys.readouterr().out


# ───────────────────── Phase 2.1 regime + by-regime subcommands ─────────────────────


def test_cli_regime_current_happy_path(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from datetime import UTC as _UTC
    from datetime import datetime as _dt

    from trading_bot import regime as regime_mod

    fake = regime_mod.RegimeSnapshot(
        date="2026-05-26", regime="bull",
        spy_close=612.45, ema50=598.21, ema200=571.88, ema50_slope=1.23,
    )
    monkeypatch.setattr("trading_bot.regime.get_current_regime", lambda **_: fake)
    monkeypatch.setattr(
        "trading_bot.regime.last_cached_at", lambda: _dt.now(_UTC),
    )
    _run(["regime", "current"])
    out = capsys.readouterr().out
    assert "Market Regime: BULL" in out
    assert "SPY Close:    $612.45" in out
    assert "50 EMA:       $598.21" in out
    assert "above 200 EMA" in out
    assert "200 EMA:      $571.88" in out
    assert "+1.23/day" in out
    assert "positive" in out
    assert "Snapshot age:" in out


def test_cli_regime_current_uses_plain_ascii_only(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No Unicode like checkmarks / >= — Windows cp1252 can't encode them."""
    from trading_bot import regime as regime_mod

    fake = regime_mod.RegimeSnapshot(
        date="2026-05-26", regime="sideways",
        spy_close=540.0, ema50=545.0, ema200=550.0, ema50_slope=-0.5,
    )
    monkeypatch.setattr("trading_bot.regime.get_current_regime", lambda **_: fake)
    monkeypatch.setattr("trading_bot.regime.last_cached_at", lambda: None)
    _run(["regime", "current"])
    out = capsys.readouterr().out
    # cp1252-encodable characters only
    out.encode("cp1252")


def test_cli_regime_current_fetch_error_exits_nonzero(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_bot import regime as regime_mod

    def boom(**_: object) -> regime_mod.RegimeSnapshot:
        raise regime_mod.RegimeFetchError("network down")

    monkeypatch.setattr("trading_bot.regime.get_current_regime", boom)
    with pytest.raises(SystemExit) as exc:
        _run(["regime", "current"])
    assert exc.value.code == 1
    assert "network down" in capsys.readouterr().err


def test_cli_regime_history_renders_table(
    tmp_db: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from datetime import UTC as _UTC
    from datetime import datetime as _dt

    from trading_bot import db

    db.upsert_regime_snapshot(
        snapshot_date="2026-05-26", regime="bull", spy_close=612.45,
        ema50=598.21, ema200=571.88, ema50_slope=1.23,
        captured_at=_dt.now(_UTC),
    )
    db.upsert_regime_snapshot(
        snapshot_date="2026-05-25", regime="sideways", spy_close=608.91,
        ema50=597.15, ema200=571.01, ema50_slope=0.4,
        captured_at=_dt.now(_UTC),
    )
    _run(["regime", "history", "--days", "10"])
    out = capsys.readouterr().out
    assert "2026-05-26" in out
    assert "bull" in out
    assert "2026-05-25" in out
    assert "sideways" in out


def test_cli_regime_history_empty_message(
    tmp_db: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _run(["regime", "history"])
    assert "no snapshots recorded" in capsys.readouterr().out


def test_cli_regime_backfill_happy_path(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "trading_bot.regime.backfill_trade_regimes",
        lambda: {"trades_updated": 20, "dates_snapshotted": 14,
                 "errors": 0, "skipped_no_data": 0},
    )
    _run(["regime", "backfill"])
    out = capsys.readouterr().out
    assert "Backfilled 20 trades across 14 unique dates. 0 errors." in out


def test_cli_regime_backfill_already_done_no_op(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "trading_bot.regime.backfill_trade_regimes",
        lambda: {"trades_updated": 0, "dates_snapshotted": 0,
                 "errors": 0, "skipped_no_data": 0},
    )
    _run(["regime", "backfill"])
    out = capsys.readouterr().out
    assert "Backfilled 0 trades" in out


def test_cli_report_by_regime(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_bot.performance import PerfStats

    monkeypatch.setattr(
        "trading_bot.performance.stats_by_regime",
        lambda: [
            PerfStats(label="bull", total=12, wins=8, losses=4, expired=0,
                      win_rate=66.7, avg_pnl_pct=2.10, best_pnl_pct=8.0, worst_pnl_pct=-3.0),
            PerfStats(label="sideways", total=6, wins=3, losses=3, expired=0,
                      win_rate=50.0, avg_pnl_pct=0.5, best_pnl_pct=5.0, worst_pnl_pct=-4.0),
            PerfStats(label="bear", total=2, wins=0, losses=2, expired=0,
                      win_rate=0.0, avg_pnl_pct=-2.5, best_pnl_pct=-1.0, worst_pnl_pct=-4.0),
        ],
    )
    _run(["report", "by-regime"])
    out = capsys.readouterr().out
    assert "BY MARKET REGIME" in out
    assert "bull" in out
    assert "66.7%" in out
    assert "sideways" in out
    assert "bear" in out


def test_cli_report_by_regime_empty(
    tmp_db: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _run(["report", "by-regime"])
    assert "(no closed trades)" in capsys.readouterr().out


def test_cli_report_by_signal_with_regime(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_bot.performance import PerfStats, RegimeBreakdown

    bull_stats = PerfStats(label="ema21_pullback / bull", total=10, wins=8, losses=2,
                           expired=0, win_rate=80.0, avg_pnl_pct=3.0,
                           best_pnl_pct=8.0, worst_pnl_pct=-1.0)
    side_stats = PerfStats(label="ema21_pullback / sideways", total=2, wins=1, losses=1,
                           expired=0, win_rate=50.0, avg_pnl_pct=0.5,
                           best_pnl_pct=2.0, worst_pnl_pct=-1.0)
    overall = PerfStats(label="ema21_pullback", total=12, wins=9, losses=3,
                        expired=0, win_rate=75.0, avg_pnl_pct=2.6,
                        best_pnl_pct=8.0, worst_pnl_pct=-1.0)
    monkeypatch.setattr(
        "trading_bot.performance.stats_by_signal_type_with_regime",
        lambda: [RegimeBreakdown(
            label="ema21_pullback",
            by_regime={"bull": bull_stats, "sideways": side_stats},
            overall=overall,
        )],
    )
    _run(["report", "by-signal", "--by-regime"])
    out = capsys.readouterr().out
    assert "BY SIGNAL TYPE x REGIME" in out
    assert "ema21_pullback" in out
    assert "Bull" in out
    assert "Sideways" in out
    assert "Bear" in out
    assert "8/10 80%" in out


def test_cli_report_by_ticker_with_regime_and_stocks_filter(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_bot.performance import RegimeBreakdown

    received: dict[str, object] = {}

    def spy(*, asset_class: str | None = None) -> list[RegimeBreakdown]:
        received["asset_class"] = asset_class
        return []

    monkeypatch.setattr(
        "trading_bot.performance.stats_by_ticker_with_regime", spy
    )
    _run(["report", "by-ticker", "--by-regime", "--stocks"])
    assert received["asset_class"] == "stock"
    assert "BY TICKER x REGIME (stock)" in capsys.readouterr().out


def test_cli_report_by_signal_without_regime_flag_uses_old_view(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Without --by-regime, the Phase 1.4 by-signal output is unchanged.
    monkeypatch.setattr(
        "trading_bot.performance.stats_by_signal_type", lambda: [],
    )
    _run(["report", "by-signal"])
    assert "BY SIGNAL TYPE" in capsys.readouterr().out


def test_report_recent_with_custom_days(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_bot.performance import PerfStats

    received: dict[str, object] = {}

    def fake_recent(*, days: int = 30) -> PerfStats:
        received["days"] = days
        return PerfStats(label=f"recent ({days}d)", total=0, wins=0, losses=0, expired=0,
                         win_rate=None, avg_pnl_pct=None, best_pnl_pct=None, worst_pnl_pct=None)

    monkeypatch.setattr("trading_bot.performance.stats_recent", fake_recent)
    _run(["report", "recent", "--days", "90"])
    assert received["days"] == 90
    assert "last 90 days" in capsys.readouterr().out
