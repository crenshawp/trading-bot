"""Tests for trading_bot.discovery — runner, scoring, gate, promotion.

All backtests are mocked: ``discovery._run_ema21_backtest`` is monkeypatched
to return synthetic DataFrames. No live yfinance, no network.
"""

from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd
import pytest

from trading_bot import db, discovery

# A row tuple is (direction, price, take_profit, stop_loss, outcome).
_Row = tuple[str, float, float, float, str]


def _bt_df(rows: Sequence[_Row]) -> pd.DataFrame:
    """Build a backtest-shaped result DataFrame (the columns score_backtest reads)."""
    return pd.DataFrame(
        list(rows),
        columns=["direction", "price", "take_profit", "stop_loss", "outcome"],
    )


def _call_rows(wins: int, losses: int, unknown: int = 0) -> list[_Row]:
    """CALL rows: WIN -> +4.0%, LOSS -> -3.0% (tp=104, sl=97 from price 100)."""
    rows: list[_Row] = []
    rows += [("CALL", 100.0, 104.0, 97.0, "WIN")] * wins
    rows += [("CALL", 100.0, 104.0, 97.0, "LOSS")] * losses
    rows += [("CALL", 100.0, 104.0, 97.0, "UNKNOWN")] * unknown
    return rows


# ───────────────────────── _per_trade_return ─────────────────────────


def test_per_trade_return_call_win_is_positive() -> None:
    assert discovery._per_trade_return("CALL", 100.0, 104.0, 97.0, "WIN") == pytest.approx(4.0)


def test_per_trade_return_call_loss_is_negative() -> None:
    assert discovery._per_trade_return("CALL", 100.0, 104.0, 97.0, "LOSS") == pytest.approx(-3.0)


def test_per_trade_return_put_win_is_positive() -> None:
    # PUT win exits at take_profit BELOW entry -> positive return.
    assert discovery._per_trade_return("PUT", 100.0, 96.0, 103.0, "WIN") == pytest.approx(4.0)


def test_per_trade_return_put_loss_is_negative() -> None:
    assert discovery._per_trade_return("PUT", 100.0, 96.0, 103.0, "LOSS") == pytest.approx(-3.0)


def test_per_trade_return_unknown_is_none() -> None:
    assert discovery._per_trade_return("CALL", 100.0, 104.0, 97.0, "UNKNOWN") is None


def test_per_trade_return_zero_price_is_none() -> None:
    assert discovery._per_trade_return("CALL", 0.0, 104.0, 97.0, "WIN") is None


# ───────────────────────── score_backtest ─────────────────────────


def test_score_backtest_basic_math() -> None:
    score = discovery.score_backtest("AAA", _bt_df(_call_rows(wins=7, losses=3)))
    assert score.trade_count == 10
    assert score.win_rate == pytest.approx(70.0)
    # expectancy == mean per-trade return: (7*4 - 3*3)/10 = 1.9
    assert score.expectancy == pytest.approx(1.9)
    assert score.avg_return_pct == pytest.approx(1.9)
    assert score.qualified is True


def test_score_backtest_excludes_unknown_from_trade_count() -> None:
    score = discovery.score_backtest("AAA", _bt_df(_call_rows(wins=6, losses=4, unknown=5)))
    assert score.trade_count == 10  # the 5 UNKNOWN do not count


def test_score_backtest_all_unknown_yields_none_metrics() -> None:
    score = discovery.score_backtest("AAA", _bt_df(_call_rows(wins=0, losses=0, unknown=8)))
    assert score.trade_count == 0
    assert score.win_rate is None
    assert score.avg_return_pct is None
    assert score.expectancy is None
    assert score.qualified is False


def test_gate_rejects_below_trade_count_floor() -> None:
    # 7 wins + 2 losses = 9 decided trades, positive expectancy, still rejected.
    score = discovery.score_backtest("AAA", _bt_df(_call_rows(wins=7, losses=2)))
    assert score.trade_count == 9
    assert score.expectancy is not None and score.expectancy > 0
    assert score.qualified is False


def test_gate_rejects_negative_expectancy() -> None:
    # 3 wins + 7 losses = 10 trades, expectancy (12-21)/10 = -0.9.
    score = discovery.score_backtest("AAA", _bt_df(_call_rows(wins=3, losses=7)))
    assert score.trade_count == 10
    assert score.expectancy == pytest.approx(-0.9)
    assert score.qualified is False


def test_gate_rejects_zero_expectancy() -> None:
    # Symmetric +/-3 with equal wins/losses -> expectancy exactly 0 (gate is strict >).
    rows: list[_Row] = (
        [("CALL", 100.0, 103.0, 97.0, "WIN")] * 5
        + [("CALL", 100.0, 103.0, 97.0, "LOSS")] * 5
    )
    score = discovery.score_backtest("AAA", _bt_df(rows))
    assert score.expectancy == pytest.approx(0.0)
    assert score.qualified is False


def test_rank_qualifiers_orders_by_expectancy_desc() -> None:
    low = discovery.score_backtest("LOW", _bt_df(_call_rows(wins=6, losses=4)))
    high = discovery.score_backtest("HIGH", _bt_df(_call_rows(wins=9, losses=1)))
    ranked = discovery.rank_qualifiers([low, high])
    assert [s.ticker for s in ranked] == ["HIGH", "LOW"]


# ───────────────────────── run_backtests (failure path) ─────────────────────────


def _patch_backtest(
    monkeypatch: pytest.MonkeyPatch,
    fn: Callable[[str, str], pd.DataFrame | None],
) -> None:
    monkeypatch.setattr(discovery, "_run_ema21_backtest", fn)


def test_run_backtests_records_exception_as_failure_not_silent(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def boom(ticker: str, _asset: str) -> pd.DataFrame | None:
        raise RuntimeError("yfinance exploded")

    _patch_backtest(monkeypatch, boom)
    result = discovery.run_backtests(tickers=["AAA"], throttle_seconds=0)

    assert result.succeeded == 0
    assert result.failed == 1
    assert result.failures[0].ticker == "AAA"
    assert "backtest error" in result.failures[0].reason
    assert "yfinance exploded" in result.failures[0].reason
    # NOT a silent skip — the failure is logged to stderr.
    assert "AAA FAILED" in capsys.readouterr().err


def test_run_backtests_records_none_as_no_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_backtest(monkeypatch, lambda t, a: None)
    result = discovery.run_backtests(tickers=["AAA"], throttle_seconds=0)
    assert result.failed == 1
    assert "no data" in result.failures[0].reason


def test_run_backtests_records_empty_frame_as_no_signals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_backtest(monkeypatch, lambda t, a: _bt_df([]))
    result = discovery.run_backtests(tickers=["AAA"], throttle_seconds=0)
    assert result.failed == 1
    assert "no signals" in result.failures[0].reason


def test_run_backtests_mixed_success_and_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake(ticker: str, _asset: str) -> pd.DataFrame | None:
        if ticker == "GOOD":
            return _bt_df(_call_rows(wins=7, losses=3))
        return None

    _patch_backtest(monkeypatch, fake)
    result = discovery.run_backtests(tickers=["GOOD", "BAD"], throttle_seconds=0)
    assert result.scanned == 2
    assert set(result.successes) == {"GOOD"}
    assert [f.ticker for f in result.failures] == ["BAD"]


def test_run_backtests_throttles_between_tickers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(discovery.time, "sleep", lambda s: sleeps.append(s))
    _patch_backtest(monkeypatch, lambda t, a: _bt_df(_call_rows(wins=5, losses=5)))
    discovery.run_backtests(tickers=["AAA", "BBB", "CCC"], throttle_seconds=0.01)
    # Sleeps before the 2nd and 3rd tickers only — never before the first.
    assert sleeps == [0.01, 0.01]


# ───────────────────────── run_discovery (promotion + persistence) ─────────────────────────


def test_run_discovery_promotes_qualifiers_without_duplicates(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # BBB is already on the watchlist; AAA + BBB both qualify; CCC fails.
    db.add_to_active_watchlist("BBB", "manual")

    def fake(ticker: str, _asset: str) -> pd.DataFrame | None:
        if ticker in {"AAA", "BBB"}:
            return _bt_df(_call_rows(wins=7, losses=3))
        return None

    _patch_backtest(monkeypatch, fake)
    run = discovery.run_discovery(
        tickers=["AAA", "BBB", "CCC"], promote=True, throttle_seconds=0
    )

    assert run.scanned == 3
    assert run.succeeded == 2
    assert run.failed == 1
    assert {s.ticker for s in run.qualifiers} == {"AAA", "BBB"}
    # Only AAA is newly promoted — BBB was already present, no duplicate.
    assert run.promoted == ["AAA"]
    watchlist = db.get_active_watchlist()
    assert watchlist.count("BBB") == 1
    assert "AAA" in watchlist


def test_run_discovery_persists_every_scored_ticker(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake(ticker: str, _asset: str) -> pd.DataFrame | None:
        if ticker == "AAA":
            return _bt_df(_call_rows(wins=7, losses=3))   # qualifies
        if ticker == "ZZZ":
            return _bt_df(_call_rows(wins=3, losses=7))   # scored but fails gate
        return None  # CCC fails to backtest

    _patch_backtest(monkeypatch, fake)
    discovery.run_discovery(
        tickers=["AAA", "ZZZ", "CCC"], promote=True, throttle_seconds=0
    )

    persisted = db.get_discovery_results()
    by_ticker = {r["ticker"]: r for r in persisted}
    # Both scored tickers persisted; the failed one is not.
    assert set(by_ticker) == {"AAA", "ZZZ"}
    assert by_ticker["AAA"]["qualified"] is True
    assert by_ticker["ZZZ"]["qualified"] is False
    assert by_ticker["AAA"]["trade_count"] == 10


def test_run_discovery_no_promote_leaves_watchlist_untouched(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_backtest(monkeypatch, lambda t, a: _bt_df(_call_rows(wins=8, losses=2)))
    run = discovery.run_discovery(
        tickers=["AAA"], promote=False, throttle_seconds=0
    )
    assert run.promoted == []
    assert db.get_active_watchlist() == []
    # Persistence still happens even when promotion is off.
    assert len(db.get_discovery_results()) == 1


# ───────────────────────── db.add_to_active_watchlist ─────────────────────────


def test_add_to_active_watchlist_dedupes(tmp_db: Path) -> None:
    assert db.add_to_active_watchlist("AAA", "discovery") is True
    assert db.add_to_active_watchlist("AAA", "discovery") is False
    assert db.get_active_watchlist().count("AAA") == 1
