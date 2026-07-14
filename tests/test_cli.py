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
    assert "Schema version: 22" in out


def test_cli_db_status_shows_counts(tmp_db: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _run(["db", "status"])
    out = capsys.readouterr().out
    assert "Schema version: 22" in out
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
        lambda **_kwargs: {
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
        lambda **_: [
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

    def spy(*, asset_class: str | None = None, **_: object) -> list:  # type: ignore[type-arg]
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

    def spy(*, asset_class: str | None = None, **_: object) -> list:  # type: ignore[type-arg]
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


# ───────────────────── Phase 3.1 discovery subcommand ─────────────────────


def test_cli_discovery_scan_reports_results(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from datetime import UTC, datetime

    from trading_bot import discovery

    run = discovery.DiscoveryRun(
        run_timestamp=datetime.now(UTC),
        scores=[
            discovery.TickerScore("AAA", 70.0, 10, 1.9, 1.9, True),
            discovery.TickerScore("ZZZ", 40.0, 12, -0.5, -0.5, False),
        ],
        failures=[discovery.DiscoveryFailure("CCC", "no data")],
    )

    def fake_run_discovery(**_kwargs: object) -> discovery.DiscoveryRun:
        return run

    monkeypatch.setattr("trading_bot.discovery.run_discovery", fake_run_discovery)
    _run(["discovery", "scan"])
    out = capsys.readouterr().out
    assert "DISCOVERY SCAN" in out
    assert "informational only" in out
    assert "Ranked candidates (1)" in out
    assert "AAA" in out
    # Informational only — the scan promotes nothing and points to live-shadow.
    assert "shadow evaluate" in out
    assert "Failures (1)" in out
    assert "CCC" in out


# ───────────────────── Phase 3.1-LIVE shadow subcommands ─────────────────────


def test_cli_shadow_status(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_bot import shadow_discovery

    monkeypatch.setattr(
        "trading_bot.discovery_universe.SHADOW_UNIVERSE", ["AAA", "BBB"],
    )

    def fake_eval(ticker: str, **_kw: object) -> shadow_discovery.ShadowEvaluation:
        if ticker == "AAA":
            return shadow_discovery.ShadowEvaluation("AAA", 12, 75.0, 0.80, True, False)
        return shadow_discovery.ShadowEvaluation("BBB", 4, 50.0, -0.10, False, False)

    monkeypatch.setattr("trading_bot.shadow_discovery.evaluate_ticker", fake_eval)
    _run(["shadow", "status"])
    out = capsys.readouterr().out
    assert "SHADOW STATUS" in out
    assert "AAA" in out
    assert "yes" in out          # AAA is eligible
    assert "BBB" in out


def test_cli_shadow_evaluate_promotes(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_bot import shadow_discovery

    evals = [
        shadow_discovery.ShadowEvaluation("AAA", 12, 75.0, 0.80, True, True),
        shadow_discovery.ShadowEvaluation("BBB", 4, 50.0, -0.10, False, False),
    ]
    captured: dict[str, object] = {}

    def fake_universe(**kwargs: object) -> list[shadow_discovery.ShadowEvaluation]:
        captured.update(kwargs)
        return evals

    monkeypatch.setattr(
        "trading_bot.shadow_discovery.evaluate_shadow_universe", fake_universe,
    )
    _run(["shadow", "evaluate"])
    out = capsys.readouterr().out
    assert "SHADOW EVALUATE" in out
    assert "Promoted (1)" in out
    assert "AAA" in out
    assert "Withheld with data (1)" in out
    assert "BBB" in out
    assert captured["dry_run"] is False


def test_cli_shadow_evaluate_dry_run(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_bot import shadow_discovery

    # Dry-run: eligible but not promoted.
    evals = [shadow_discovery.ShadowEvaluation("AAA", 12, 75.0, 0.80, True, False)]
    captured: dict[str, object] = {}

    def fake_universe(**kwargs: object) -> list[shadow_discovery.ShadowEvaluation]:
        captured.update(kwargs)
        return evals

    monkeypatch.setattr(
        "trading_bot.shadow_discovery.evaluate_shadow_universe", fake_universe,
    )
    _run(["shadow", "evaluate", "--dry-run"])
    out = capsys.readouterr().out
    assert "DRY RUN" in out
    assert "Would promote (1)" in out
    assert "AAA" in out
    assert captured["dry_run"] is True


# ───────────────────── Phase 3.3 watchlist subcommands ─────────────────────


def test_cli_watchlist_evaluate(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from datetime import UTC, datetime

    from trading_bot import watchlist_state

    evals = [
        watchlist_state.TickerEvaluation(
            "BAD", "active", 10, 40.0, -0.4, "demote", "expectancy -0.400 <= 0.0"),
        watchlist_state.TickerEvaluation(
            "REC", "benched", 10, 70.0, 1.4, "recover", "expectancy 1.400 >= 0.05"),
        watchlist_state.TickerEvaluation(
            "HELD", "active", 10, 45.0, -0.1, "hold",
            "held: active floor (active would drop below 5)"),
        watchlist_state.TickerEvaluation(
            "OK", "active", 10, 80.0, 1.4, "hold", "active held: ..."),
    ]
    run = watchlist_state.StateRun(
        evaluated_at=datetime.now(UTC), evaluations=evals,
    )
    captured: dict[str, object] = {}

    def fake_eval(**kwargs: object) -> watchlist_state.StateRun:
        captured.update(kwargs)
        return run

    monkeypatch.setattr(
        "trading_bot.watchlist_state.evaluate_watchlist", fake_eval,
    )
    _run(["watchlist", "evaluate"])
    out = capsys.readouterr().out
    assert "WATCHLIST EVALUATE" in out
    assert "Demotions (1)" in out and "BAD" in out
    assert "Recoveries (1)" in out and "REC" in out
    assert "Floor holds (1)" in out and "HELD" in out
    assert "Active  (3)" in out          # HELD, OK, REC end active
    assert "Benched (1)" in out          # BAD ends benched
    assert captured["dry_run"] is False


def test_cli_watchlist_evaluate_dry_run_passes_flag(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from datetime import UTC, datetime

    from trading_bot import watchlist_state

    run = watchlist_state.StateRun(
        evaluated_at=datetime.now(UTC), evaluations=[],
    )
    captured: dict[str, object] = {}

    def fake_eval(**kwargs: object) -> watchlist_state.StateRun:
        captured.update(kwargs)
        return run

    monkeypatch.setattr(
        "trading_bot.watchlist_state.evaluate_watchlist", fake_eval,
    )
    _run(["watchlist", "evaluate", "--dry-run"])
    out = capsys.readouterr().out
    assert "DRY RUN" in out
    assert captured["dry_run"] is True


def test_cli_watchlist_status(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_bot import db

    db.add_to_active_watchlist("ACT", "seed")
    db.add_to_active_watchlist("BEN", "shadow")
    db.set_watchlist_status("BEN", "benched")
    monkeypatch.setattr(
        "trading_bot.watchlist_state.windowed_stats_for",
        lambda t, **_kw: (12, 75.0, 0.80) if t == "ACT" else (11, 40.0, -0.30),
    )
    _run(["watchlist", "status"])
    out = capsys.readouterr().out
    assert "WATCHLIST STATUS" in out
    assert "Active (1)" in out and "ACT" in out
    assert "Benched (1)" in out and "BEN" in out


# ───────────────────── Phase 4 pairs subcommands ─────────────────────


def test_cli_pairs_evaluate(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from datetime import UTC, datetime

    from trading_bot import signal_pairs

    evals = [
        signal_pairs.PairEvaluation(
            "GOOGL", "ema21_pullback", "enabled", 10, 40.0, -0.4, "mute", "bad"),
        signal_pairs.PairEvaluation(
            "META", "overbought_reversal", "muted", 10, 70.0, 1.4, "enable", "good"),
        signal_pairs.PairEvaluation(
            "AMZN", "trend_continuation", "enabled", 10, 80.0, 1.4, "hold", "ok"),
    ]
    run = signal_pairs.PairRun(evaluated_at=datetime.now(UTC), evaluations=evals)
    captured: dict[str, object] = {}

    def fake(**kwargs: object) -> signal_pairs.PairRun:
        captured.update(kwargs)
        return run

    monkeypatch.setattr("trading_bot.signal_pairs.evaluate_signal_pairs", fake)
    _run(["pairs", "evaluate"])
    out = capsys.readouterr().out
    assert "PAIRS EVALUATE" in out
    assert "Muted (1)" in out and "GOOGL/ema21_pullback" in out
    assert "Enabled (1)" in out and "META/overbought_reversal" in out
    assert "Muted now (1)" in out          # only GOOGL ends muted
    assert captured["dry_run"] is False


def test_cli_pairs_evaluate_dry_run_passes_flag(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from datetime import UTC, datetime

    from trading_bot import signal_pairs

    run = signal_pairs.PairRun(evaluated_at=datetime.now(UTC), evaluations=[])
    captured: dict[str, object] = {}

    def fake(**kwargs: object) -> signal_pairs.PairRun:
        captured.update(kwargs)
        return run

    monkeypatch.setattr("trading_bot.signal_pairs.evaluate_signal_pairs", fake)
    _run(["pairs", "evaluate", "--dry-run"])
    assert "DRY RUN" in capsys.readouterr().out
    assert captured["dry_run"] is True


def test_cli_pairs_status(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_bot import signal_pairs

    stats = [
        signal_pairs.PairStat("GOOGL", "ema21_pullback", "muted", 10, 40.0, -0.4),
        signal_pairs.PairStat("AMZN", "trend_continuation", "enabled", 12, 80.0, 1.4),
    ]
    monkeypatch.setattr("trading_bot.signal_pairs.pair_stats", lambda **_kw: stats)
    _run(["pairs", "status"])
    out = capsys.readouterr().out
    assert "PAIRS STATUS" in out
    assert "Enabled (1)" in out and "AMZN" in out
    assert "Muted (1)" in out and "GOOGL" in out


def test_cli_report_pairs(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_bot import signal_pairs

    stats = [
        signal_pairs.PairStat("GOOGL", "ema21_pullback", "muted", 10, 40.0, -0.4),
    ]
    monkeypatch.setattr("trading_bot.signal_pairs.pair_stats", lambda **_kw: stats)
    _run(["report", "pairs"])
    out = capsys.readouterr().out
    assert "PER-PAIR PERFORMANCE" in out
    assert "GOOGL" in out and "muted" in out


# ───────────────────── Phase 5 sentiment subcommand ─────────────────────


def test_cli_sentiment_status(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rows = [{
        "ticker": "GOOGL", "signal_type": "ema21_pullback", "direction": "call",
        "opened_at": "2026-06-01T10:00:00", "outcome": "win", "track_mode": "active",
        "sentiment_score": 0.6, "sentiment_label": "bullish",
        "heavy_news": True, "headline_count": 9,
    }]
    monkeypatch.setattr(
        "trading_bot.db.get_recent_trade_sentiment", lambda **_kw: rows,
    )
    _run(["sentiment", "status"])
    out = capsys.readouterr().out
    assert "SENTIMENT STATUS" in out
    assert "GOOGL" in out
    assert "bullish" in out
    assert "+0.60" in out


def test_cli_sentiment_status_empty(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "trading_bot.db.get_recent_trade_sentiment", lambda **_kw: [],
    )
    _run(["sentiment", "status"])
    assert "no sentiment-scored signals yet" in capsys.readouterr().out


# ───────────────────── Phase 6 indicators subcommand ─────────────────────


def test_cli_indicators_status(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rows = [{
        "ticker": "GOOGL", "signal_type": "ema21_pullback", "direction": "call",
        "opened_at": "2026-06-01T10:00:00", "outcome": "win", "track_mode": "active",
        "ind_atr": 2.5, "ind_realized_vol": 0.018, "ind_vol_regime": "normal",
        "ind_rsi": 54.3, "ind_adx": 27.1, "ind_obv": 1234567.0,
        "ind_correlation": 0.42, "ind_concentration": "moderate",
    }]
    monkeypatch.setattr(
        "trading_bot.db.get_recent_trade_indicators", lambda **_kw: rows,
    )
    _run(["indicators", "status"])
    out = capsys.readouterr().out
    assert "INDICATOR STATUS" in out
    assert "GOOGL" in out
    assert "normal" in out      # vol regime
    assert "54.3" in out        # RSI
    assert "27.1" in out        # ADX
    assert "+0.42" in out       # correlation
    assert "moderate" in out    # concentration
    assert "1,234,567" in out   # OBV (volume flow)


def test_cli_indicators_status_handles_missing_values(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # a fail-soft bundle: per-ticker math None, only the regime/concentration set
    rows = [{
        "ticker": "TSLA", "signal_type": "trend_continuation", "direction": "call",
        "opened_at": "2026-06-02T09:31:00", "outcome": None, "track_mode": "active",
        "ind_atr": None, "ind_realized_vol": None, "ind_vol_regime": "unknown",
        "ind_rsi": None, "ind_adx": None, "ind_obv": None,
        "ind_correlation": None, "ind_concentration": "unknown",
    }]
    monkeypatch.setattr(
        "trading_bot.db.get_recent_trade_indicators", lambda **_kw: rows,
    )
    _run(["indicators", "status"])
    out = capsys.readouterr().out
    assert "TSLA" in out
    assert "open" in out     # NULL outcome rendered as 'open'
    assert "-" in out        # None numeric fields rendered as '-'


def test_cli_indicators_status_empty(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "trading_bot.db.get_recent_trade_indicators", lambda **_kw: [],
    )
    _run(["indicators", "status"])
    assert "no indicator-scored signals yet" in capsys.readouterr().out


# ───────────────────── Phase 7 risk subcommand ─────────────────────


def test_cli_risk_status(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rows = [{
        "ticker": "GOOGL", "signal_type": "ema21_pullback", "direction": "call",
        "opened_at": "2026-06-01T10:00:00", "outcome": "win", "track_mode": "active",
        "risk_recommended_size": 20.0, "risk_pct": 0.6, "risk_position_pct": 20.0,
        "risk_capped": True, "risk_total_pct": 4.5,
        "risk_portfolio_verdict": "ok",
        "risk_position_verdict": "would-exceed-position",
        "risk_cluster_pct": 3.0, "risk_cluster_verdict": "ok",
    }]
    monkeypatch.setattr("trading_bot.db.get_recent_trade_risk", lambda **_kw: rows)
    _run(["risk", "status"])
    out = capsys.readouterr().out
    assert "RISK STATUS" in out
    assert "GOOGL" in out
    assert "20.00*" in out                      # capped marker
    assert "0.60" in out                        # risk %
    assert "would-exceed-position" in out


def test_cli_risk_status_empty(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("trading_bot.db.get_recent_trade_risk", lambda **_kw: [])
    _run(["risk", "status"])
    assert "no risk-assessed signals yet" in capsys.readouterr().out


def test_cli_risk_exposure(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from types import SimpleNamespace
    trades = [
        SimpleNamespace(track_mode="active", risk_pct=2.0,
                        ind_concentration="concentrated", risk_position_pct=20.0),
        SimpleNamespace(track_mode="active", risk_pct=3.0,
                        ind_concentration="diversified", risk_position_pct=15.0),
        SimpleNamespace(track_mode="shadow", risk_pct=99.0,        # excluded (shadow)
                        ind_concentration="concentrated", risk_position_pct=99.0),
        SimpleNamespace(track_mode="active", risk_pct=None,        # pre-Phase-7, skipped
                        ind_concentration="unknown", risk_position_pct=None),
    ]
    monkeypatch.setattr("trading_bot.db.get_open_trades", lambda: trades)
    _run(["risk", "exposure"])
    out = capsys.readouterr().out
    assert "RISK EXPOSURE" in out
    assert "3  (2 risk-sized)" in out           # 3 active (shadow excluded), 2 sized
    assert "5.00%" in out                       # total: 2 + 3 (None skipped)
    assert "Concentrated cluster" in out
    assert "2.00%" in out                       # only the concentrated active position
    assert "20.00%" in out                      # largest position


# ───────────────────── Phase 9 optimize subcommand ─────────────────────


def test_cli_optimize_report_empty(
    tmp_db: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _run(["optimize", "report"])
    assert "No self-optimization runs yet" in capsys.readouterr().out


def test_cli_optimize_report_populated(
    tmp_db: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import json

    from trading_bot import db
    payload = {
        "run_timestamp": "2026-07-01T00:00:00",
        "degrade_window_days": 30, "baseline_window_days": 90,
        "degradations": [{
            "scope": "overall", "verdict": "stable", "delta": 0.0,
            "baseline_expectancy": 1.0, "baseline_n": 30,
            "recent_expectancy": 1.0, "recent_n": 30, "note": "within delta",
        }],
        "features": [{
            "feature": "sentiment", "actionable": False, "note": "n<floor",
            "buckets": [{"label": "bullish", "n": 5, "win_rate": 60.0,
                         "expectancy": 1.2, "actionable": False}],
        }],
    }
    db.insert_optimization_run("2026-07-01T00:00:00", 30, 90, json.dumps(payload))
    _run(["optimize", "report"])
    out = capsys.readouterr().out
    assert "SELF-OPTIMIZATION REPORT" in out
    assert "overall" in out and "stable" in out
    assert "sentiment" in out and "bullish" in out


def test_cli_optimize_run_computes_persists_and_prints(
    tmp_db: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_bot import db
    _run(["optimize", "run", "--degrade-window", "30", "--baseline-window", "90"])
    out = capsys.readouterr().out
    assert "SELF-OPTIMIZATION REPORT" in out
    assert "DEGRADATION" in out
    assert "FEATURE EVALUATION" in out
    assert db.get_latest_optimization_run() is not None     # the run was persisted


def test_cli_optimize_run_rejects_inverted_window(
    tmp_db: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        _run(["optimize", "run", "--degrade-window", "60", "--baseline-window", "30"])
    assert "must exceed" in capsys.readouterr().err


# ───────────────────── Phase 10 readiness subcommand ─────────────────────


def _seed_resolved_active(n: int) -> None:
    from datetime import datetime, timedelta

    from trading_bot import db
    from trading_bot.models import Signal, Trade
    for i in range(n):
        ts = datetime(2026, 1, 1, 9, 0) + timedelta(minutes=i)
        sid = db.insert_signal(Signal(
            timestamp=ts, ticker=f"C{i}", asset_class="stock",
            signal_type="ema21_pullback", direction="call", entry_price=100.0,
        ))
        db.insert_trade(Trade(
            signal_id=sid, opened_at=ts, closed_at=ts + timedelta(days=1),
            outcome="win", pnl_pct=2.0, track_mode="active",
        ))


def test_cli_readiness_status_lists_every_capability(
    tmp_db: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _run(["readiness", "status"])
    out = capsys.readouterr().out
    assert "READINESS STATUS" in out
    for name in ("watchlist_rotation", "pair_gating", "shadow_promotion",
                 "self_optimization", "ml_pattern_recognition",
                 "ml_predictive_sizing"):
        assert name in out
    assert "warming" in out          # nothing seeded -> all warming
    assert "ml" in out               # ML kind shown


def test_cli_readiness_check_evaluates_persists_and_prints(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _seed_resolved_active(10)         # crosses watchlist_rotation + pair_gating (10)
    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "trading_bot.readiness._pushover_notify",
        lambda t, m: bool(sent.append((t, m))) or True,
    )
    _run(["readiness", "check"])
    out = capsys.readouterr().out
    assert "Announced (first crossing): watchlist_rotation" in out
    assert "READINESS STATUS" in out
    assert "ready" in out
    # the one-time notification fired through the (mocked) Pushover path
    assert any("watchlist_rotation" in m for _, m in sent)


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
        lambda **_: [
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
        lambda **_: [RegimeBreakdown(
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

    def spy(*, asset_class: str | None = None, **_: object) -> list[RegimeBreakdown]:
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
        "trading_bot.performance.stats_by_signal_type", lambda **_: [],
    )
    _run(["report", "by-signal"])
    assert "BY SIGNAL TYPE" in capsys.readouterr().out


# ───────────────────── Phase 2.2 vix + by-vix + by-regime-vix ─────────────────────


def test_cli_vix_current_happy_path(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from datetime import UTC as _UTC
    from datetime import datetime as _dt

    from trading_bot import vix as vix_mod

    fake = vix_mod.VixSnapshot(
        date="2026-05-26", vix_level=18.4, vix_band="low",
        captured_at=_dt.now(_UTC).isoformat(),
    )
    monkeypatch.setattr("trading_bot.vix.get_current_vix", lambda **_: fake)
    monkeypatch.setattr(
        "trading_bot.vix.last_cached_at", lambda: _dt.now(_UTC),
    )
    _run(["vix", "current"])
    out = capsys.readouterr().out
    assert "VIX: 18.4 (low)" in out
    assert "Level:         18.4" in out
    assert "Band:          low" in out
    assert "< 20" in out
    assert "Snapshot age:" in out


def test_cli_vix_current_uses_plain_ascii(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from datetime import UTC as _UTC
    from datetime import datetime as _dt

    from trading_bot import vix as vix_mod

    fake = vix_mod.VixSnapshot(
        date="2026-05-26", vix_level=35.0, vix_band="high",
        captured_at=_dt.now(_UTC).isoformat(),
    )
    monkeypatch.setattr("trading_bot.vix.get_current_vix", lambda **_: fake)
    monkeypatch.setattr("trading_bot.vix.last_cached_at", lambda: None)
    _run(["vix", "current"])
    out = capsys.readouterr().out
    out.encode("cp1252")  # raises if any Unicode that cp1252 can't encode


def test_cli_vix_current_fetch_error_exits_nonzero(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_bot import vix as vix_mod

    def boom(**_: object) -> vix_mod.VixSnapshot:
        raise vix_mod.VixFetchError("net down")

    monkeypatch.setattr("trading_bot.vix.get_current_vix", boom)
    with pytest.raises(SystemExit) as exc:
        _run(["vix", "current"])
    assert exc.value.code == 1
    assert "net down" in capsys.readouterr().err


def test_cli_vix_history_renders_table(
    tmp_db: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from datetime import UTC as _UTC
    from datetime import datetime as _dt

    from trading_bot import db

    db.upsert_vix_snapshot(
        snapshot_date="2026-05-26", vix_level=18.4, vix_band="low",
        captured_at=_dt.now(_UTC),
    )
    db.upsert_vix_snapshot(
        snapshot_date="2026-05-25", vix_level=21.7, vix_band="elevated",
        captured_at=_dt.now(_UTC),
    )
    _run(["vix", "history", "--days", "10"])
    out = capsys.readouterr().out
    assert "2026-05-26" in out
    assert "low" in out
    assert "elevated" in out


def test_cli_vix_history_empty(
    tmp_db: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    _run(["vix", "history"])
    assert "no snapshots recorded" in capsys.readouterr().out


def test_cli_vix_backfill_happy_path(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "trading_bot.vix.backfill_trade_vix",
        lambda: {"trades_updated": 20, "dates_snapshotted": 14,
                 "errors": 0, "skipped_no_data": 0},
    )
    _run(["vix", "backfill"])
    assert "Backfilled 20 trades across 14 unique dates. 0 errors." \
        in capsys.readouterr().out


def test_cli_vix_backfill_no_op(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "trading_bot.vix.backfill_trade_vix",
        lambda: {"trades_updated": 0, "dates_snapshotted": 0,
                 "errors": 0, "skipped_no_data": 0},
    )
    _run(["vix", "backfill"])
    assert "Backfilled 0 trades" in capsys.readouterr().out


def test_cli_report_by_vix(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_bot.performance import PerfStats

    monkeypatch.setattr(
        "trading_bot.performance.stats_by_vix_band",
        lambda **_: [
            PerfStats(label="low", total=14, wins=9, losses=5, expired=0,
                      win_rate=64.3, avg_pnl_pct=1.5, best_pnl_pct=7.0, worst_pnl_pct=-3.0),
            PerfStats(label="elevated", total=5, wins=2, losses=3, expired=0,
                      win_rate=40.0, avg_pnl_pct=-0.5, best_pnl_pct=4.0, worst_pnl_pct=-5.0),
        ],
    )
    _run(["report", "by-vix"])
    out = capsys.readouterr().out
    assert "BY VIX BAND" in out
    assert "low" in out
    assert "64.3%" in out
    assert "elevated" in out
    assert "40.0%" in out


def test_cli_report_by_vix_empty(
    tmp_db: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    _run(["report", "by-vix"])
    assert "(no closed trades)" in capsys.readouterr().out


def test_cli_report_by_signal_with_vix(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_bot.performance import PerfStats, VixBreakdown

    low = PerfStats(label="ema21_pullback / low", total=8, wins=6, losses=2,
                    expired=0, win_rate=75.0, avg_pnl_pct=2.5,
                    best_pnl_pct=6.0, worst_pnl_pct=-1.0)
    elev = PerfStats(label="ema21_pullback / elevated", total=3, wins=2, losses=1,
                     expired=0, win_rate=66.7, avg_pnl_pct=1.2,
                     best_pnl_pct=4.0, worst_pnl_pct=-2.0)
    overall = PerfStats(label="ema21_pullback", total=12, wins=9, losses=3,
                        expired=0, win_rate=75.0, avg_pnl_pct=2.3,
                        best_pnl_pct=6.0, worst_pnl_pct=-2.0)
    monkeypatch.setattr(
        "trading_bot.performance.stats_by_signal_type_with_vix",
        lambda **_: [VixBreakdown(
            label="ema21_pullback",
            by_vix={"low": low, "elevated": elev},
            overall=overall,
        )],
    )
    _run(["report", "by-signal", "--by-vix"])
    out = capsys.readouterr().out
    assert "BY SIGNAL TYPE x VIX" in out
    assert "ema21_pullback" in out
    assert "Low" in out
    assert "Elevated" in out
    assert "Extreme" in out
    assert "6/8 75%" in out


def test_cli_report_by_ticker_with_vix(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_bot.performance import PerfStats, VixBreakdown

    received: dict[str, object] = {}

    def spy(*, asset_class: str | None = None, **_: object) -> list[VixBreakdown]:
        received["asset_class"] = asset_class
        return []

    monkeypatch.setattr(
        "trading_bot.performance.stats_by_ticker_with_vix", spy
    )
    _run(["report", "by-ticker", "--by-vix", "--crypto"])
    assert received["asset_class"] == "crypto"
    assert "BY TICKER x VIX (crypto)" in capsys.readouterr().out
    _ = PerfStats  # silence unused-import lint


def test_cli_report_by_regime_vix_happy(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_bot.performance import PerfStats

    monkeypatch.setattr(
        "trading_bot.performance.stats_by_regime_x_vix",
        lambda **_: [
            PerfStats(label="bull / low", total=9, wins=7, losses=2, expired=0,
                      win_rate=77.8, avg_pnl_pct=2.5, best_pnl_pct=7.0, worst_pnl_pct=-2.0),
            PerfStats(label="sideways / low", total=4, wins=2, losses=2, expired=0,
                      win_rate=50.0, avg_pnl_pct=0.0, best_pnl_pct=3.0, worst_pnl_pct=-3.0),
            PerfStats(label="bull / elevated", total=3, wins=1, losses=2, expired=0,
                      win_rate=33.3, avg_pnl_pct=-1.0, best_pnl_pct=4.0, worst_pnl_pct=-4.0),
        ],
    )
    _run(["report", "by-regime-vix"])
    out = capsys.readouterr().out
    assert "BY REGIME x VIX" in out
    assert "bull" in out
    assert "low" in out
    assert "77.8%" in out
    assert "sideways" in out
    assert "elevated" in out


def test_cli_report_by_regime_vix_empty(
    tmp_db: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    _run(["report", "by-regime-vix"])
    assert "(no closed trades)" in capsys.readouterr().out


def test_cli_report_by_regime_vix_sparse_one_bucket(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_bot.performance import PerfStats

    monkeypatch.setattr(
        "trading_bot.performance.stats_by_regime_x_vix",
        lambda **_: [PerfStats(
            label="bull / low", total=1, wins=1, losses=0, expired=0,
            win_rate=100.0, avg_pnl_pct=5.0, best_pnl_pct=5.0, worst_pnl_pct=5.0,
        )],
    )
    _run(["report", "by-regime-vix"])
    out = capsys.readouterr().out
    assert "bull" in out
    assert "low" in out
    assert "100.0%" in out


# ───────────────────── Phase 2.2b predictions CLI ─────────────────────


def test_cli_predictions_enable_then_disable(
    tmp_db: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_bot import settings

    _run(["predictions", "enable"])
    out = capsys.readouterr().out
    assert "Predictions: ENABLED" in out
    assert settings.get_bool("predictions.enabled") is True

    _run(["predictions", "disable"])
    out = capsys.readouterr().out
    assert "Predictions: DISABLED" in out
    assert settings.get_bool("predictions.enabled") is False


def test_cli_predictions_status_default_disabled(
    tmp_db: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _run(["predictions", "status"])
    out = capsys.readouterr().out
    assert "Predictions: DISABLED" in out
    assert "BTC-USD,ETH-USD" in out
    assert "Accuracy:             N/A" in out


def test_cli_predictions_status_enabled_with_prediction(
    tmp_db: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from datetime import UTC as _UTC
    from datetime import datetime as _dt
    from datetime import timedelta as _td

    from trading_bot import db, settings
    from trading_bot.models import Prediction

    settings.set_bool("predictions.enabled", True)
    db.insert_prediction(Prediction(
        ticker="BTC-USD", direction="HIGHER", confidence=72.0,
        entry_price=94000.0,
        target_window_end=_dt.now(_UTC) - _td(minutes=10),
        signals_used="{}",
        created_at=_dt.now(_UTC) - _td(minutes=25),
        market_regime="bull", vix_band="low", vix_level=18.0,
        outcome="correct", exit_price=94050.0,
        resolved_at=_dt.now(_UTC) - _td(minutes=8),
    ))
    _run(["predictions", "status"])
    out = capsys.readouterr().out
    assert "ENABLED" in out
    assert "Predictions made:     1" in out
    assert "Most recent:" in out


def test_cli_predictions_window_happy(
    tmp_db: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_bot import settings

    _run(["predictions", "window", "--start", "09:00", "--end", "17:00"])
    out = capsys.readouterr().out
    assert "09:00 - 17:00 ET" in out
    assert settings.get("predictions.window_start") == "09:00"
    assert settings.get("predictions.window_end") == "17:00"


def test_cli_predictions_window_rejects_bad_format(
    tmp_db: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc:
        _run(["predictions", "window", "--start", "9am", "--end", "5pm"])
    assert exc.value.code == 1
    assert "Invalid window" in capsys.readouterr().err


def test_cli_predictions_window_rejects_end_before_start(
    tmp_db: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc:
        _run(["predictions", "window", "--start", "22:00", "--end", "08:00"])
    assert exc.value.code == 1
    assert "end" in capsys.readouterr().err


def test_cli_predictions_tickers_happy(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import pandas as pd

    from trading_bot import settings

    # Stub _fetch_15m_candles to return a non-None DataFrame for any ticker.
    monkeypatch.setattr(
        "trading_bot.predictions._fetch_15m_candles",
        lambda _t: pd.DataFrame({"x": [1]}),
    )
    _run(["predictions", "tickers", "--set", "BTC-USD,ETH-USD"])
    assert "BTC-USD, ETH-USD" in capsys.readouterr().out
    assert settings.get("predictions.tickers") == "BTC-USD,ETH-USD"


def test_cli_predictions_tickers_rejects_unresolvable(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "trading_bot.predictions._fetch_15m_candles", lambda _t: None,
    )
    with pytest.raises(SystemExit) as exc:
        _run(["predictions", "tickers", "--set", "BOGUS-USD"])
    assert exc.value.code == 1
    assert "failed yfinance validation" in capsys.readouterr().err


def test_cli_predictions_pause_sets_until(
    tmp_db: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_bot import settings

    _run(["predictions", "pause", "--minutes", "30"])
    out = capsys.readouterr().out
    assert "paused until" in out.lower()
    until_raw = settings.get("predictions.pause_until")
    assert until_raw is not None


def test_cli_predictions_pause_rejects_zero(
    tmp_db: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc:
        _run(["predictions", "pause", "--minutes", "0"])
    assert exc.value.code == 1
    assert "must be positive" in capsys.readouterr().err


def test_cli_report_predictions_empty(
    tmp_db: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    _run(["report", "predictions"])
    assert "(no predictions yet)" in capsys.readouterr().out


def test_cli_report_predictions_happy(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_bot.performance import PredictionStats

    monkeypatch.setattr(
        "trading_bot.performance.stats_predictions_by_ticker",
        lambda **_: [
            PredictionStats("BTC-USD", total=48, correct=31, incorrect=16,
                            push=1, unresolved=0, accuracy=66.0),
            PredictionStats("ETH-USD", total=52, correct=29, incorrect=22,
                            push=1, unresolved=0, accuracy=56.9),
        ],
    )
    monkeypatch.setattr(
        "trading_bot.performance.stats_predictions_overall",
        lambda **_: PredictionStats(
            "TOTAL", total=100, correct=60, incorrect=38, push=2,
            unresolved=0, accuracy=61.2,
        ),
    )
    _run(["report", "predictions"])
    out = capsys.readouterr().out
    assert "Prediction accuracy" in out
    assert "BTC-USD" in out
    assert "ETH-USD" in out
    assert "TOTAL" in out
    assert "61.2%" in out


def test_cli_report_predictions_by_regime(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_bot.performance import PredictionStats

    monkeypatch.setattr(
        "trading_bot.performance.stats_predictions_by_regime",
        lambda **_: [PredictionStats(
            "bull", total=20, correct=15, incorrect=4, push=1,
            unresolved=0, accuracy=78.9,
        )],
    )
    _run(["report", "predictions", "--by-regime"])
    out = capsys.readouterr().out
    assert "by regime" in out
    assert "bull" in out
    assert "78.9%" in out


def test_cli_report_predictions_by_vix(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_bot.performance import PredictionStats

    monkeypatch.setattr(
        "trading_bot.performance.stats_predictions_by_vix",
        lambda **_: [PredictionStats(
            "low", total=12, correct=8, incorrect=4, push=0,
            unresolved=0, accuracy=66.7,
        )],
    )
    _run(["report", "predictions", "--by-vix"])
    out = capsys.readouterr().out
    assert "by VIX band" in out
    assert "low" in out


def test_cli_report_predictions_by_time(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_bot.performance import PredictionStats

    monkeypatch.setattr(
        "trading_bot.performance.stats_predictions_by_hour",
        lambda **_: [
            PredictionStats("09", total=12, correct=9, incorrect=3, push=0,
                            unresolved=0, accuracy=75.0),
            PredictionStats("14", total=18, correct=10, incorrect=8, push=0,
                            unresolved=0, accuracy=55.5),
        ],
    )
    _run(["report", "predictions", "--by-time"])
    out = capsys.readouterr().out
    assert "by hour" in out
    assert "75.0%" in out
    assert "55.5%" in out


# ───────────────────── Phase 2.3 context CLI + --min-context + matrix ─────────────────────


def test_cli_context_current_happy_path(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_bot import context as ctx_mod

    monkeypatch.setattr(
        "trading_bot.__main__.context.get_current_context",
        lambda: ctx_mod.ContextSnapshot(
            regime="bull", vix_band="low", vix_level=18.4,
            score=5, label="ideal", captured_at="2026-05-26T14:30:00+00:00",
        ),
    )
    monkeypatch.setattr("trading_bot.regime.last_cached_at", lambda: None)
    monkeypatch.setattr("trading_bot.vix.last_cached_at", lambda: None)
    _run(["context", "current"])
    out = capsys.readouterr().out
    assert "Market Context: IDEAL (5/5)" in out
    assert "Regime:      bull" in out
    assert "18.4" in out
    assert "low" in out
    assert "Score:       5 / 5" in out


def test_cli_context_current_unknown_state(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_bot import context as ctx_mod

    monkeypatch.setattr(
        "trading_bot.__main__.context.get_current_context",
        lambda: ctx_mod.ContextSnapshot(
            regime="unknown", vix_band="unknown", vix_level=None,
            score=0, label="unknown", captured_at="2026-05-26T14:30:00+00:00",
        ),
    )
    monkeypatch.setattr("trading_bot.regime.last_cached_at", lambda: None)
    monkeypatch.setattr("trading_bot.vix.last_cached_at", lambda: None)
    _run(["context", "current"])
    out = capsys.readouterr().out
    assert "UNKNOWN (0/5)" in out
    assert "unknown" in out


def test_cli_context_history_happy(
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
    db.upsert_vix_snapshot(
        snapshot_date="2026-05-26", vix_level=18.4, vix_band="low",
        captured_at=_dt.now(_UTC),
    )
    db.upsert_regime_snapshot(
        snapshot_date="2026-05-25", regime="sideways", spy_close=608.0,
        ema50=597.0, ema200=571.0, ema50_slope=0.5,
        captured_at=_dt.now(_UTC),
    )
    db.upsert_vix_snapshot(
        snapshot_date="2026-05-25", vix_level=22.0, vix_band="elevated",
        captured_at=_dt.now(_UTC),
    )
    _run(["context", "history", "--days", "10"])
    out = capsys.readouterr().out
    assert "2026-05-25" in out
    assert "2026-05-26" in out
    assert "bull" in out
    assert "sideways" in out
    assert "ideal" in out
    assert "neutral" in out


def test_cli_context_history_empty(
    tmp_db: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    _run(["context", "history"])
    out = capsys.readouterr().out
    assert "no overlapping snapshots" in out


def test_cli_context_history_sparse_one_axis_missing(
    tmp_db: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """A date with only a regime snapshot but no VIX snapshot is skipped."""
    from datetime import UTC as _UTC
    from datetime import datetime as _dt

    from trading_bot import db

    db.upsert_regime_snapshot(
        snapshot_date="2026-05-26", regime="bull", spy_close=612.0,
        ema50=598.0, ema200=571.0, ema50_slope=1.0,
        captured_at=_dt.now(_UTC),
    )
    _run(["context", "history"])
    out = capsys.readouterr().out
    assert "no overlapping snapshots" in out


def test_cli_context_backfill_happy(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "trading_bot.context.backfill_context_scores",
        lambda: {"trades_updated": 20, "predictions_updated": 5},
    )
    _run(["context", "backfill"])
    out = capsys.readouterr().out
    assert "Backfilled context_score on 20 trades and 5 predictions." in out


def test_cli_context_backfill_no_op(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "trading_bot.context.backfill_context_scores",
        lambda: {"trades_updated": 0, "predictions_updated": 0},
    )
    _run(["context", "backfill"])
    assert "0 trades and 0 predictions" in capsys.readouterr().out


def test_cli_report_by_context_happy(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_bot.performance import PerfStats

    monkeypatch.setattr(
        "trading_bot.performance.stats_by_context",
        lambda: [
            PerfStats(label="5", total=8, wins=7, losses=1, expired=0,
                      win_rate=87.5, avg_pnl_pct=3.0, best_pnl_pct=6.0, worst_pnl_pct=-1.0),
            PerfStats(label="4", total=6, wins=4, losses=2, expired=0,
                      win_rate=66.7, avg_pnl_pct=1.5, best_pnl_pct=5.0, worst_pnl_pct=-2.0),
            PerfStats(label="3", total=3, wins=1, losses=2, expired=0,
                      win_rate=33.3, avg_pnl_pct=-0.5, best_pnl_pct=2.0, worst_pnl_pct=-3.0),
            PerfStats(label="2", total=0, wins=0, losses=0, expired=0,
                      win_rate=None, avg_pnl_pct=None, best_pnl_pct=None, worst_pnl_pct=None),
            PerfStats(label="1", total=0, wins=0, losses=0, expired=0,
                      win_rate=None, avg_pnl_pct=None, best_pnl_pct=None, worst_pnl_pct=None),
            PerfStats(label="0", total=0, wins=0, losses=0, expired=0,
                      win_rate=None, avg_pnl_pct=None, best_pnl_pct=None, worst_pnl_pct=None),
        ],
    )
    _run(["report", "by-context"])
    out = capsys.readouterr().out
    assert "BY CONTEXT SCORE" in out
    assert "ideal" in out
    assert "favorable" in out
    assert "87.5%" in out


def test_cli_report_context_matrix_happy(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_bot.performance import PerfStats

    monkeypatch.setattr(
        "trading_bot.performance.stats_by_regime_x_vix",
        lambda **_: [
            PerfStats(label="bull / low", total=6, wins=5, losses=1, expired=0,
                      win_rate=83.3, avg_pnl_pct=2.0, best_pnl_pct=5.0, worst_pnl_pct=-1.0),
            PerfStats(label="bull / elevated", total=4, wins=3, losses=1, expired=0,
                      win_rate=75.0, avg_pnl_pct=1.5, best_pnl_pct=4.0, worst_pnl_pct=-1.0),
            PerfStats(label="sideways / low", total=3, wins=2, losses=1, expired=0,
                      win_rate=66.7, avg_pnl_pct=1.0, best_pnl_pct=3.0, worst_pnl_pct=-1.0),
            PerfStats(label="bear / elevated", total=1, wins=0, losses=1, expired=0,
                      win_rate=0.0, avg_pnl_pct=-2.0, best_pnl_pct=-2.0, worst_pnl_pct=-2.0),
        ],
    )
    _run(["report", "context-matrix"])
    out = capsys.readouterr().out
    assert "Win rates: regime x VIX" in out
    assert "Bull" in out
    assert "Sideways" in out
    assert "Bear" in out
    assert "Low" in out
    assert "Elevated" in out
    assert "5/6 83%" in out


def test_cli_report_context_matrix_empty(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "trading_bot.performance.stats_by_regime_x_vix", lambda **_: [],
    )
    _run(["report", "context-matrix"])
    assert "(no closed trades)" in capsys.readouterr().out


def test_cli_report_context_matrix_sparse(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_bot.performance import PerfStats

    monkeypatch.setattr(
        "trading_bot.performance.stats_by_regime_x_vix",
        lambda **_: [PerfStats(
            label="bull / low", total=1, wins=1, losses=0, expired=0,
            win_rate=100.0, avg_pnl_pct=5.0, best_pnl_pct=5.0, worst_pnl_pct=5.0,
        )],
    )
    _run(["report", "context-matrix"])
    out = capsys.readouterr().out
    assert "Bull" in out
    # Missing cells should render '-'
    assert "-" in out


def test_min_context_flag_threads_through_by_signal(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    received: dict[str, object] = {}

    def spy(**kwargs: object) -> list:  # type: ignore[type-arg]
        received.update(kwargs)
        return []

    monkeypatch.setattr("trading_bot.performance.stats_by_signal_type", spy)
    _run(["report", "by-signal", "--min-context", "4"])
    assert received["min_context"] == 4
    assert "min context 4" in capsys.readouterr().out


def test_min_context_zero_is_no_op(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    received: dict[str, object] = {}

    def spy(**kwargs: object) -> list:  # type: ignore[type-arg]
        received.update(kwargs)
        return []

    monkeypatch.setattr("trading_bot.performance.stats_by_signal_type", spy)
    _run(["report", "by-signal", "--min-context", "0"])
    assert received["min_context"] == 0
    out = capsys.readouterr().out
    assert "min context" not in out  # no suffix shown


def test_min_context_threads_through_by_regime(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    received: dict[str, object] = {}

    def spy(**kwargs: object) -> list:  # type: ignore[type-arg]
        received.update(kwargs)
        return []

    monkeypatch.setattr("trading_bot.performance.stats_by_regime", spy)
    _run(["report", "by-regime", "--min-context", "3"])
    assert received["min_context"] == 3


def test_min_context_threads_through_by_vix(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: dict[str, object] = {}

    def spy(**kwargs: object) -> list:  # type: ignore[type-arg]
        received.update(kwargs)
        return []

    monkeypatch.setattr("trading_bot.performance.stats_by_vix_band", spy)
    _run(["report", "by-vix", "--min-context", "2"])
    assert received["min_context"] == 2


def test_min_context_threads_through_by_regime_vix(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: dict[str, object] = {}

    def spy(**kwargs: object) -> list:  # type: ignore[type-arg]
        received.update(kwargs)
        return []

    monkeypatch.setattr("trading_bot.performance.stats_by_regime_x_vix", spy)
    _run(["report", "by-regime-vix", "--min-context", "5"])
    assert received["min_context"] == 5


def test_min_context_threads_through_predictions(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trading_bot.performance import PredictionStats

    received: dict[str, object] = {}

    def overall_spy(**kwargs: object) -> PredictionStats:
        received["overall_min"] = kwargs.get("min_context")
        return PredictionStats(
            "TOTAL", total=0, correct=0, incorrect=0, push=0,
            unresolved=0, accuracy=None,
        )

    def ticker_spy(**kwargs: object) -> list[PredictionStats]:
        received["ticker_min"] = kwargs.get("min_context")
        return []

    monkeypatch.setattr(
        "trading_bot.performance.stats_predictions_overall", overall_spy,
    )
    monkeypatch.setattr(
        "trading_bot.performance.stats_predictions_by_ticker", ticker_spy,
    )
    _run(["report", "predictions", "--min-context", "4"])
    assert received["overall_min"] == 4
    assert received["ticker_min"] == 4


def test_min_context_with_by_regime_matrix(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: dict[str, object] = {}

    def spy(**kwargs: object) -> list:  # type: ignore[type-arg]
        received.update(kwargs)
        return []

    monkeypatch.setattr(
        "trading_bot.performance.stats_by_signal_type_with_regime", spy,
    )
    _run(["report", "by-signal", "--by-regime", "--min-context", "4"])
    assert received["min_context"] == 4


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
