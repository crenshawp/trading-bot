"""Tests for trading_bot.discovery_universe — candidate pool + exclusions."""

import pytest

from trading_bot import discovery_universe as du


def test_effective_universe_excludes_dropped_names() -> None:
    effective = du.effective_universe()
    for excluded in du.EXCLUDED_TICKERS:
        assert excluded not in effective


def test_effective_universe_has_no_duplicates() -> None:
    effective = du.effective_universe()
    assert len(effective) == len(set(effective))


def test_effective_universe_keeps_required_seeds() -> None:
    effective = set(du.effective_universe())
    for seed in ("MSFT", "NVDA", "PLTR", "AMD", "COST"):
        assert seed in effective


def test_effective_universe_is_subset_of_full_universe() -> None:
    assert set(du.effective_universe()).issubset(set(du.DISCOVERY_UNIVERSE))


def test_filter_universe_removes_excluded_and_dedupes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Order-preserving subtraction + dedup, exercised with overlap."""
    monkeypatch.setattr(
        du, "DISCOVERY_UNIVERSE", ["MSFT", "AAPL", "NVDA", "MSFT", "JPM"]
    )
    monkeypatch.setattr(du, "EXCLUDED_TICKERS", frozenset({"AAPL", "JPM"}))
    assert du.effective_universe() == ["MSFT", "NVDA"]
