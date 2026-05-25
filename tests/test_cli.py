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
