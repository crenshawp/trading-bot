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
    assert "Schema version: 1" in out


def test_cli_db_status_shows_counts(tmp_db: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _run(["db", "status"])
    out = capsys.readouterr().out
    assert "Schema version: 1" in out
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
