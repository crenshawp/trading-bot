"""Tests for trading_bot.settings — generic key/value SQLite store."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from trading_bot import settings

# ────────────────────── round-trip ──────────────────────


def test_set_and_get_round_trip(tmp_db: Path) -> None:
    settings.set("predictions.enabled", "true")
    assert settings.get("predictions.enabled") == "true"


def test_get_returns_default_for_missing_key(tmp_db: Path) -> None:
    assert settings.get("nonexistent") is None
    assert settings.get("nonexistent", "fallback") == "fallback"


def test_set_overwrites_existing(tmp_db: Path) -> None:
    settings.set("k", "v1")
    settings.set("k", "v2")
    assert settings.get("k") == "v2"


def test_delete_removes_key(tmp_db: Path) -> None:
    settings.set("k", "v")
    settings.delete("k")
    assert settings.get("k") is None


def test_delete_missing_key_is_noop(tmp_db: Path) -> None:
    settings.delete("never-set")  # must not raise


# ────────────────────── bool helpers ──────────────────────


@pytest.mark.parametrize("raw", ["true", "True", "TRUE", "1", "yes", "on", "  true  "])
def test_get_bool_truthy_variants(tmp_db: Path, raw: str) -> None:
    settings.set("k", raw)
    assert settings.get_bool("k") is True


@pytest.mark.parametrize("raw", ["false", "False", "FALSE", "0", "no", "off"])
def test_get_bool_falsy_variants(tmp_db: Path, raw: str) -> None:
    settings.set("k", raw)
    assert settings.get_bool("k") is False


def test_get_bool_missing_uses_default(tmp_db: Path) -> None:
    assert settings.get_bool("missing", default=True) is True
    assert settings.get_bool("missing", default=False) is False


def test_get_bool_unknown_string_uses_default(tmp_db: Path) -> None:
    settings.set("k", "maybe")
    assert settings.get_bool("k", default=True) is True
    assert settings.get_bool("k", default=False) is False


def test_set_bool_round_trip(tmp_db: Path) -> None:
    settings.set_bool("k", True)
    assert settings.get_bool("k") is True
    settings.set_bool("k", False)
    assert settings.get_bool("k") is False
    # Verify stored form is the canonical lowercase.
    assert settings.get("k") == "false"


# ────────────────────── updated_at ──────────────────────


def test_get_updated_at_returns_none_for_missing(tmp_db: Path) -> None:
    assert settings.get_updated_at("missing") is None


def test_updated_at_advances_on_overwrite(tmp_db: Path) -> None:
    settings.set("k", "v1")
    first = settings.get_updated_at("k")
    assert first is not None

    # SQLite ISO timestamps have microsecond resolution; pause briefly to
    # ensure the second write is observably later.
    time.sleep(0.01)

    settings.set("k", "v2")
    second = settings.get_updated_at("k")
    assert second is not None
    assert second >= first
