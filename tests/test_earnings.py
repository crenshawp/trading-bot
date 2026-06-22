"""Tests for trading_bot.earnings — the earnings-in-hold-window blackout gate.

yfinance is mocked; no network.
"""

from datetime import UTC, datetime, timedelta

import pytest

from trading_bot import earnings

NOW = datetime(2026, 6, 1, tzinfo=UTC)


def test_blackout_when_earnings_in_window(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        earnings, "_fetch_earnings_date", lambda _t: NOW + timedelta(days=3)
    )
    blackout, reason = earnings.is_in_blackout("GOOGL", 5, now=NOW)
    assert blackout is True
    assert "within" in reason


def test_no_blackout_when_earnings_outside_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        earnings, "_fetch_earnings_date", lambda _t: NOW + timedelta(days=20)
    )
    blackout, reason = earnings.is_in_blackout("GOOGL", 5, now=NOW)
    assert blackout is False
    assert "outside" in reason


def test_fail_open_when_earnings_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(earnings, "_fetch_earnings_date", lambda _t: None)
    blackout, reason = earnings.is_in_blackout("GOOGL", 5, now=NOW)
    assert blackout is False
    assert "unknown" in reason


def test_earnings_today_blacks_out_regardless_of_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    midnight = NOW.replace(hour=0, minute=0)
    monkeypatch.setattr(earnings, "_fetch_earnings_date", lambda _t: midnight)
    blackout, _ = earnings.is_in_blackout("GOOGL", 5, now=NOW.replace(hour=14))
    assert blackout is True


def test_past_earnings_not_a_blackout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        earnings, "_fetch_earnings_date", lambda _t: NOW - timedelta(days=2)
    )
    blackout, _ = earnings.is_in_blackout("GOOGL", 5, now=NOW)
    assert blackout is False


# ── yfinance fetch isolation ──


class _FakeTicker:
    def __init__(self, calendar: object) -> None:
        self._calendar = calendar

    @property
    def calendar(self) -> object:
        if isinstance(self._calendar, Exception):
            raise self._calendar
        return self._calendar


def test_fetch_earnings_date_parses_calendar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cal = {"Earnings Date": [datetime(2026, 7, 1).date()]}
    monkeypatch.setattr(
        "trading_bot.earnings.yf.Ticker", lambda _t: _FakeTicker(cal)
    )
    info = earnings.next_earnings_date("GOOGL")
    assert info.known is True
    assert info.earnings_date is not None
    assert info.earnings_date.date().isoformat() == "2026-07-01"


def test_fetch_earnings_date_unknown_on_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "trading_bot.earnings.yf.Ticker",
        lambda _t: _FakeTicker(RuntimeError("yf down")),
    )
    assert earnings._fetch_earnings_date("GOOGL") is None
    assert "earnings fetch error" in capsys.readouterr().err


def test_fetch_earnings_date_none_when_no_date(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "trading_bot.earnings.yf.Ticker", lambda _t: _FakeTicker({})
    )
    assert earnings._fetch_earnings_date("GOOGL") is None
