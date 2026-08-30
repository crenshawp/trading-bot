"""The NYSE closes at 13:00 ET three times a year, and the order path knew nothing about it.

`is_market_open` (scanner) and `is_option_market_open` (options_execution) both
established that the day was a *session* and then assumed a 16:00 bell. On a
half-day that leaves a three-hour window in which both guards answer "open"
against a shut market — and both guards sit directly in front of live order
submission:

  * `_run_allocation_execution_cycle` submits opening orders and then durably
    consumes the signals behind them via `mark_final_candidates`;
  * `watch_open_option_positions` submits DAY limit exits, which a broker
    cancels at end of day — leaving a position the watcher recorded as exited.

The dates are DERIVED, in the same style as the holiday set they sit beside, so
that the table cannot silently expire the way the scanner's old hardcoded 2026
holiday list did.
"""

from __future__ import annotations

from datetime import date, datetime, time
from unittest.mock import patch
from zoneinfo import ZoneInfo

from trading_bot import outcomes, scanner
from trading_bot.options_execution import is_option_market_open

_ET = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# the derived calendar
# ---------------------------------------------------------------------------


def test_the_published_2026_half_days_are_derived() -> None:
    """2026: Independence Day is a Saturday, so July 3 is a full closure.

    That leaves the Friday after Thanksgiving and Christmas Eve — and July 3
    must NOT appear, because it is the observed holiday rather than a half-day.
    """
    assert outcomes.is_early_close(date(2026, 11, 27)) is True  # after Thanksgiving
    assert outcomes.is_early_close(date(2026, 12, 24)) is True  # Christmas Eve
    assert outcomes.is_early_close(date(2026, 7, 3)) is False   # observed holiday
    assert outcomes.is_stock_session(date(2026, 7, 3)) is False


def test_july_3_is_a_half_day_only_when_the_fourth_is_a_weekday() -> None:
    # 2024: July 4 Thursday -> July 3 Wednesday is a half-day.
    assert outcomes.is_early_close(date(2024, 7, 3)) is True
    # 2025: July 4 Friday -> July 3 Thursday is a half-day.
    assert outcomes.is_early_close(date(2025, 7, 3)) is True
    # 2027: July 4 Sunday -> July 3 is a Saturday, no session to shorten.
    assert outcomes.is_early_close(date(2027, 7, 3)) is False


def test_christmas_eve_is_a_half_day_only_when_the_25th_is_a_weekday() -> None:
    # 2024: Christmas Wednesday -> Dec 24 Tuesday is a half-day.
    assert outcomes.is_early_close(date(2024, 12, 24)) is True
    # 2025: Christmas Thursday -> Dec 24 Wednesday is a half-day.
    assert outcomes.is_early_close(date(2025, 12, 24)) is True
    # 2027: Christmas Saturday, observed Friday Dec 24 — a full closure, and so
    # not a half-day. This is the case a hardcoded "Dec 24" table gets wrong.
    assert outcomes.is_early_close(date(2027, 12, 24)) is False
    assert outcomes.is_stock_session(date(2027, 12, 24)) is False


def test_the_friday_after_thanksgiving_is_always_a_half_day() -> None:
    for year, day in ((2024, 29), (2025, 28), (2026, 27), (2027, 26)):
        assert outcomes.is_early_close(date(year, 11, day)) is True, year


def test_an_ordinary_session_is_not_a_half_day() -> None:
    assert outcomes.is_early_close(date(2026, 8, 21)) is False
    assert outcomes.stock_session_close(date(2026, 8, 21)) == time(16, 0)


def test_the_close_time_follows_the_calendar() -> None:
    assert outcomes.stock_session_close(date(2026, 11, 27)) == time(13, 0)
    assert outcomes.stock_session_close(date(2026, 12, 24)) == time(13, 0)


def test_the_resolver_session_close_also_follows_the_calendar() -> None:
    """The resolver's private helper must agree with the public one.

    Regression: `stock_session_close` was taught about half-days, but the
    private `_stock_session_close` used by
    `_last_closed_stock_session_in_window` and `_covers_deadline` still
    hardcoded 16:00. A half-day session was therefore treated as open for
    three hours after the bell, so the resolver rejected a session that had
    already closed and deferred settlement to the prior session or a later
    cycle.
    """
    for half_day in (date(2026, 11, 27), date(2026, 12, 24)):
        closed_at = outcomes._stock_session_close(half_day)
        assert closed_at.astimezone(_ET).timetz().replace(tzinfo=None) == time(13, 0), (
            f"{half_day} resolver close disagrees with the calendar"
        )
    # An ordinary session is unchanged.
    ordinary = outcomes._stock_session_close(date(2026, 8, 21))
    assert ordinary.astimezone(_ET).timetz().replace(tzinfo=None) == time(16, 0)


def test_a_derived_half_day_never_lands_on_a_non_session() -> None:
    """Every derived date must be a real session; an early close on a holiday
    would be a contradiction that silently widened the guard."""
    for year in range(2022, 2036):
        for day in outcomes._stock_market_early_closes(year):
            assert outcomes.is_stock_session(day), f"{day} is not a session"


# ---------------------------------------------------------------------------
# the guards that submit orders
# ---------------------------------------------------------------------------


def _at(moment: datetime) -> bool:
    with patch("trading_bot.scanner.datetime") as clock:
        clock.now.return_value = moment
        return scanner.is_market_open()


def test_scanner_guard_shuts_at_13_00_on_a_half_day() -> None:
    half_day = datetime(2026, 11, 27, 14, 0, tzinfo=_ET)
    assert _at(half_day) is False, (
        "the execution cycle would submit opening orders and burn signals three "
        "hours after the closing bell"
    )


def test_scanner_guard_still_open_before_13_00_on_a_half_day() -> None:
    assert _at(datetime(2026, 11, 27, 12, 59, tzinfo=_ET)) is True


def test_scanner_guard_unchanged_on_an_ordinary_session() -> None:
    assert _at(datetime(2026, 8, 21, 14, 0, tzinfo=_ET)) is True
    assert _at(datetime(2026, 8, 21, 16, 30, tzinfo=_ET)) is False
    assert _at(datetime(2026, 8, 21, 9, 0, tzinfo=_ET)) is False


def test_option_guard_shuts_at_13_00_on_a_half_day() -> None:
    assert is_option_market_open(datetime(2026, 12, 24, 14, 0, tzinfo=_ET)) is False, (
        "the exit watcher would submit a DAY limit the broker cancels at end of "
        "day, against a position it has already recorded as exiting"
    )


def test_option_guard_still_open_before_13_00_on_a_half_day() -> None:
    assert is_option_market_open(datetime(2026, 12, 24, 12, 59, tzinfo=_ET)) is True


def test_option_guard_unchanged_on_an_ordinary_session() -> None:
    assert is_option_market_open(datetime(2026, 8, 21, 15, 59, tzinfo=_ET)) is True
    assert is_option_market_open(datetime(2026, 8, 21, 16, 0, tzinfo=_ET)) is False
    assert is_option_market_open(datetime(2026, 8, 22, 12, 0, tzinfo=_ET)) is False
