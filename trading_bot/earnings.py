"""Earnings-date lookup + hold-window blackout — Phase 5 (the one hard gate).

The single hard gate of this phase: if a KNOWN earnings report falls inside a
trade's hold window, suppress the alert and do NOT open the trade (a known
binary gap risk, not a model opinion). If the earnings date is UNKNOWN we
FAIL-OPEN — unknown is never a reason to suppress.

The earnings date comes from yfinance's calendar (the same source the legacy
``check_earnings_risk`` used). Any fetch/parse failure returns ``None``
(unknown) and is logged; this module never raises.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import yfinance as yf


@dataclass(frozen=True)
class EarningsInfo:
    """The next earnings datetime for a ticker, or ``None`` when unknown."""

    ticker: str
    earnings_date: datetime | None

    @property
    def known(self) -> bool:
        return self.earnings_date is not None


def _coerce_dt(value: Any) -> datetime | None:
    """Coerce a yfinance calendar value (date / Timestamp / str) to UTC datetime."""
    try:
        import pandas as pd

        ts = pd.Timestamp(value)
        dt: datetime = ts.to_pydatetime()
    except Exception:  # noqa: BLE001 - any parse failure is just "unknown"
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _fetch_earnings_date(ticker: str) -> datetime | None:
    """Next earnings datetime via yfinance, or ``None``. Never raises.

    Isolated so tests can monkeypatch it without touching the network.
    """
    try:
        calendar = yf.Ticker(ticker).calendar
        if not isinstance(calendar, dict):
            return None
        dates = calendar.get("Earnings Date")
        if not dates:
            return None
        first = dates[0] if isinstance(dates, list | tuple) else dates
        return _coerce_dt(first)
    except Exception as exc:  # noqa: BLE001 - earnings lookup must never raise
        print(f"  earnings fetch error for {ticker}: {exc}", file=sys.stderr)
        return None


def next_earnings_date(ticker: str) -> EarningsInfo:
    """Return the ticker's next earnings date, or unknown (fail-soft)."""
    dt = _fetch_earnings_date(ticker)
    if dt is None:
        print(
            f"  earnings: unknown date for {ticker} (fail-open, no blackout)",
            file=sys.stderr,
        )
    return EarningsInfo(ticker=ticker, earnings_date=dt)


def is_in_blackout(
    ticker: str, hold_window_days: int, *, now: datetime | None = None
) -> tuple[bool, str]:
    """Return ``(blackout, reason)`` for the earnings-in-hold-window gate.

    ``blackout`` is True ONLY when a KNOWN earnings date falls on or within
    ``[today, today + hold_window_days]``. An unknown date fails OPEN
    (``blackout=False``). Comparison is by calendar date so an earnings report
    landing today still blacks out regardless of the intraday clock.
    """
    moment = now if now is not None else datetime.now(UTC)
    info = next_earnings_date(ticker)
    if info.earnings_date is None:
        return (False, "earnings date unknown — fail-open (no blackout)")

    today = moment.date()
    window_end = (moment + timedelta(days=hold_window_days)).date()
    earnings_day = info.earnings_date.date()
    if today <= earnings_day <= window_end:
        return (
            True,
            f"earnings {earnings_day.isoformat()} within {hold_window_days}d "
            f"hold window",
        )
    return (
        False,
        f"earnings {earnings_day.isoformat()} outside {hold_window_days}d "
        f"hold window",
    )
