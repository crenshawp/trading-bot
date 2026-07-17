"""Alpaca market data client — Phase 25. FREE TIER ONLY, parallel to yfinance.

A read-only bars/quotes client for Alpaca's data API used by the yfinance-vs-Alpaca
COMPARISON tooling. It replaces nothing: every existing yfinance call site
(scanner detection, resolver settlement, indicators, discovery, earnings,
long-term confirmation, predictions, regime/VIX) is untouched and remains
authoritative until a separate, deliberate cutover phase gated on the
comparison report.

FREE-TIER GUARD. Stocks are fetched from the free IEX feed ONLY. A request
that would need a paid feed (``sip`` consolidated stocks, ``opra`` options) is
REFUSED before any network I/O with a clear "requires Algo Trader Plus (paid
data plan)" reason — never silently attempted, never upgraded, never nagged.
Crypto bars have no feed tiers and are free.

BAR-CLOSURE SEMANTICS (the critical correctness check, re-derived for Alpaca
rather than ported from the yfinance ``iloc[-2]`` fix): an Alpaca bar's
timestamp ``t`` marks the START of its interval, and the historical bars
endpoint CAN include the currently-forming bar for the in-progress interval —
the same still-forming-candle bug class the Phase 2 fix addressed. Whether the
partial bar appears depends on query timing, so a positional rule (second-to-
last) is NOT safe here. Minute/hour closure uses elapsed duration. Daily bars
are New York calendar days and close at the next local midnight, including
23/25-hour DST days. See :func:`bar_is_closed`.

Fail-soft everywhere, matching every other external client in this codebase:
auth/rate-limit/transport failures return ``ok=False`` results with a logged
reason and never raise into a caller.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import requests

from trading_bot import config, secrets
from trading_bot.broker.alpaca import to_alpaca_symbol

# The free-tier data host. The paid SIP/OPRA feeds live behind the same host —
# the guard below is what keeps this client from ever requesting them.
ALPACA_DATA_BASE_URL = "https://data.alpaca.markets"

FEED_IEX = "iex"
# Feeds that exist on Alpaca but REQUIRE the paid Algo Trader Plus plan. Any
# request naming one is refused before network I/O.
PAID_FEEDS: frozenset[str] = frozenset({"sip", "opra"})

_HTTP_TIMEOUT_SECONDS = 10

# Fixed-duration timeframes used for bar closure. Daily is calendar-based below.
TIMEFRAME_DURATIONS: dict[str, timedelta] = {
    "1Min": timedelta(minutes=1),
    "1Hour": timedelta(hours=1),
}

_ET = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class MarketBar:
    """One OHLCV bar in neutral form. ``start`` is the bar interval's START
    (Alpaca's ``t``), tz-aware UTC."""

    symbol: str
    start: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class BarsResult:
    """A bars fetch outcome. ``ok=False`` means nothing usable arrived (guard
    refusal, credentials unset, rate limit, transport failure) — ``reason``
    says why. ``bars`` is keyed by the CALLER'S symbol (internal yfinance-style
    form for crypto, e.g. ``BTC-USD``), each list sorted oldest-first."""

    ok: bool = False
    reason: str = ""
    bars: dict[str, list[MarketBar]] = field(default_factory=dict)


@dataclass(frozen=True)
class MarketQuote:
    """One latest bid/ask quote in provider-neutral form.

    ``symbol`` uses the caller's symbol form (for example ``BTC-USD``), and
    ``timestamp`` is always timezone-aware UTC.
    """

    symbol: str
    timestamp: datetime
    bid_price: float
    bid_size: float
    ask_price: float
    ask_size: float


@dataclass(frozen=True)
class QuotesResult:
    """A fail-soft latest-quotes outcome keyed by the caller's symbols."""

    ok: bool = False
    reason: str = ""
    quotes: dict[str, MarketQuote] = field(default_factory=dict)


# ── bar closure (the re-derived check — NOT a port of iloc[-2]) ──────────────


def bar_is_closed(
    bar: MarketBar, timeframe: str, now: datetime | None = None,
) -> bool:
    """True iff the bar's complete interval has elapsed.

    Alpaca timestamps mark the interval START, and the API may or may not
    include the currently-forming bar depending on when the query lands —
    so a positional second-to-last rule (the yfinance fix) is unsafe in both
    directions. Minute/hour bars use elapsed durations. Alpaca daily bars are
    New York calendar days, so they close at the next local midnight rather
    than after a fixed 24 hours; this is correct across both DST transitions.
    An unknown timeframe returns False (never treat an unknown interval as
    closed).
    """
    moment = now if now is not None else datetime.now(UTC)
    if timeframe == "1Day":
        # Alpaca labels a daily bar at New York midnight. The next local
        # midnight can be 23, 24, or 25 elapsed hours later across DST.
        local_start = bar.start.astimezone(_ET)
        next_local_date = local_start.date() + timedelta(days=1)
        next_local_midnight = datetime.combine(
            next_local_date, datetime.min.time(), tzinfo=_ET,
        )
        return next_local_midnight.astimezone(UTC) <= moment
    duration = TIMEFRAME_DURATIONS.get(timeframe)
    if duration is None:
        return False
    return bar.start + duration <= moment


def latest_closed_bar(
    bars: list[MarketBar], timeframe: str, now: datetime | None = None,
) -> MarketBar | None:
    """The most recent bar whose interval has FULLY elapsed, or None.

    This is the value every consumer comparison should reason on — never the
    raw last element, which may be the still-forming bar."""
    closed = [b for b in bars if bar_is_closed(b, timeframe, now)]
    return max(closed, key=lambda b: b.start) if closed else None


# ── the client ───────────────────────────────────────────────────────────────


def _parse_bar(symbol: str, raw: dict[str, Any]) -> MarketBar | None:
    """One Alpaca bar dict → :class:`MarketBar`; None when malformed."""
    try:
        start = datetime.fromisoformat(str(raw["t"]))
        if start.tzinfo is None:
            start = start.replace(tzinfo=UTC)
        return MarketBar(
            symbol=symbol, start=start.astimezone(UTC),
            open=float(raw["o"]), high=float(raw["h"]), low=float(raw["l"]),
            close=float(raw["c"]), volume=float(raw.get("v", 0.0)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        print(f"  marketdata: malformed bar for {symbol} ({exc})", file=sys.stderr)
        return None


def _parse_quote(symbol: str, raw: dict[str, Any]) -> MarketQuote | None:
    """One Alpaca quote dict to :class:`MarketQuote`; None when malformed."""
    try:
        timestamp = datetime.fromisoformat(str(raw["t"]))
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)
        return MarketQuote(
            symbol=symbol,
            timestamp=timestamp.astimezone(UTC),
            bid_price=float(raw["bp"]),
            bid_size=float(raw["bs"]),
            ask_price=float(raw["ap"]),
            ask_size=float(raw["as"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        print(f"  marketdata: malformed quote for {symbol} ({exc})", file=sys.stderr)
        return None


class AlpacaMarketDataClient:
    """Free-tier bars/quotes client. Batched, throttled, guarded, fail-soft."""

    def __init__(
        self,
        *,
        base_url: str = ALPACA_DATA_BASE_URL,
        min_request_interval_s: float = config.MARKETDATA_MIN_REQUEST_INTERVAL_S,
        batch_size: int = config.MARKETDATA_BATCH_SIZE,
    ) -> None:
        self._base_url = base_url
        self._min_interval = min_request_interval_s
        self._batch_size = max(1, batch_size)
        self._key = secrets.get_secret("ALPACA_API_KEY")
        self._secret = secrets.get_secret("ALPACA_SECRET_KEY")
        self._last_request_at: float | None = None

    # ── free-tier guard ──────────────────────────────────────────────────────

    @staticmethod
    def _feed_refusal(feed: str) -> str | None:
        """A refusal reason when ``feed`` needs the paid plan, else None.

        The refusal happens BEFORE any network I/O — a paid feed is never
        silently attempted, and nothing here suggests upgrading."""
        if feed in PAID_FEEDS:
            return (
                f"feed '{feed}' requires Algo Trader Plus (paid data plan) - "
                "refused: this bot runs on the free tier (IEX) only"
            )
        if feed != FEED_IEX:
            return f"unknown feed '{feed}' - only the free '{FEED_IEX}' feed is used"
        return None

    # ── plumbing ─────────────────────────────────────────────────────────────

    def _throttle(self) -> None:
        """Space requests at least ``min_request_interval_s`` apart, keeping
        even a degenerate one-symbol-per-request loop under the 200 rpm cap."""
        if self._last_request_at is not None:
            elapsed = time.monotonic() - self._last_request_at
            wait = self._min_interval - elapsed
            if wait > 0:
                time.sleep(wait)
        self._last_request_at = time.monotonic()

    def _get(
        self, path: str, params: dict[str, Any],
    ) -> tuple[bool, int | None, Any, str]:
        """Fail-soft throttled GET → ``(ok, status, body, error)``."""
        if not self._key or not self._secret:
            print(
                "  marketdata: ALPACA credentials unset - request skipped",
                file=sys.stderr,
            )
            return False, None, None, "ALPACA credentials unset"
        self._throttle()
        try:
            resp = requests.get(
                f"{self._base_url}{path}",
                headers={
                    "APCA-API-KEY-ID": self._key,
                    "APCA-API-SECRET-KEY": self._secret,
                },
                params=params,
                timeout=_HTTP_TIMEOUT_SECONDS,
            )
        except Exception as exc:  # noqa: BLE001 - market data I/O must never raise
            print(f"  marketdata: GET {path} error ({exc})", file=sys.stderr)
            return False, None, None, str(exc)
        try:
            body = resp.json() if resp.content else None
        except Exception:  # noqa: BLE001 - a non-JSON body is not fatal
            body = None
        return True, resp.status_code, body, ""

    def _paged_bars(
        self, path: str, params: dict[str, Any], symbols_out: dict[str, str],
    ) -> tuple[dict[str, list[MarketBar]], str | None]:
        """Fetch one symbol batch, following ``next_page_token``. Returns the
        per-caller-symbol bars and an error reason (None on success).

        ``symbols_out`` maps the WIRE symbol (e.g. ``BTC/USD``) back to the
        caller's symbol (e.g. ``BTC-USD``)."""
        collected: dict[str, list[MarketBar]] = {}
        page_token: str | None = None
        for _page in range(20):                     # hard stop against loops
            page_params = dict(params)
            if page_token is not None:
                page_params["page_token"] = page_token
            ok, status, body, error = self._get(path, page_params)
            if not ok:
                return collected, error
            if status == 429:
                return collected, "rate limited (HTTP 429) - free tier is 200 rpm"
            if status is None or not (200 <= status < 300):
                message = ""
                if isinstance(body, dict):
                    message = str(body.get("message", ""))
                return collected, message or f"HTTP {status}"
            if not isinstance(body, dict):
                return collected, "malformed bars response: expected an object"
            raw_bars = body.get("bars")
            if not isinstance(raw_bars, dict):
                return collected, "malformed bars response: 'bars' must be an object"
            malformed: list[str] = []
            for wire_symbol, rows in raw_bars.items():
                caller_symbol = symbols_out.get(str(wire_symbol), str(wire_symbol))
                if not isinstance(rows, list):
                    malformed.append(f"{caller_symbol} bars must be a list")
                    continue
                for row in rows:
                    if not isinstance(row, dict):
                        print(
                            f"  marketdata: malformed bar for {caller_symbol}",
                            file=sys.stderr,
                        )
                        malformed.append(f"malformed bar for {caller_symbol}")
                        continue
                    bar = _parse_bar(caller_symbol, row)
                    if bar is not None:
                        collected.setdefault(caller_symbol, []).append(bar)
                    else:
                        malformed.append(f"malformed bar for {caller_symbol}")
            if malformed:
                reasons = list(dict.fromkeys(malformed))
                return collected, "malformed bars response: " + "; ".join(reasons[:3])
            page_token = body.get("next_page_token")
            if not page_token:
                break
        for bars in collected.values():
            bars.sort(key=lambda b: b.start)
        return collected, None

    def _fetch_batched(
        self,
        path: str,
        symbols: list[str],
        base_params: dict[str, Any],
        *,
        translate: bool,
    ) -> BarsResult:
        """Fetch bars for ``symbols`` in batches. Partial results are returned
        with ``ok=True`` as long as at least one batch succeeded."""
        all_bars: dict[str, list[MarketBar]] = {}
        errors: list[str] = []
        for i in range(0, len(symbols), self._batch_size):
            batch = symbols[i:i + self._batch_size]
            wire_map = {
                (to_alpaca_symbol(s) if translate else s): s for s in batch
            }
            params = dict(base_params)
            params["symbols"] = ",".join(wire_map.keys())
            bars, error = self._paged_bars(path, params, wire_map)
            all_bars.update(bars)
            if error is not None:
                errors.append(error)
                print(f"  marketdata: batch failed ({error})", file=sys.stderr)
        if not all_bars and errors:
            return BarsResult(ok=False, reason="; ".join(errors[:3]))
        return BarsResult(
            ok=True, reason="; ".join(errors[:3]), bars=all_bars,
        )

    def _fetch_latest_quotes(
        self,
        path: str,
        symbols: list[str],
        base_params: dict[str, Any],
        *,
        translate: bool,
    ) -> QuotesResult:
        """Fetch latest quotes in batches and translate wire symbols back.

        Each endpoint returns one top-level ``quotes`` mapping and does not
        paginate. Successful batches remain usable if another batch fails.
        """
        all_quotes: dict[str, MarketQuote] = {}
        errors: list[str] = []

        def record_error(reason: str) -> None:
            if reason:
                errors.append(reason)
                print(f"  marketdata: quote batch failed ({reason})", file=sys.stderr)

        for i in range(0, len(symbols), self._batch_size):
            batch = symbols[i:i + self._batch_size]
            wire_map = {
                (to_alpaca_symbol(s) if translate else s): s for s in batch
            }
            params = dict(base_params)
            params["symbols"] = ",".join(wire_map.keys())
            ok, status, body, error = self._get(path, params)
            if not ok:
                record_error(error)
                continue
            if status == 429:
                record_error("rate limited (HTTP 429) - free tier is 200 rpm")
                continue
            if status is None or not (200 <= status < 300):
                message = str(body.get("message", "")) if isinstance(body, dict) else ""
                record_error(message or f"HTTP {status}")
                continue
            raw_quotes = body.get("quotes") if isinstance(body, dict) else None
            if not isinstance(raw_quotes, dict):
                record_error("malformed quote response")
                continue
            parsed_in_batch = 0
            malformed_symbols: list[str] = []
            for wire_symbol, raw in raw_quotes.items():
                caller_symbol = wire_map.get(str(wire_symbol), str(wire_symbol))
                if not isinstance(raw, dict):
                    print(
                        f"  marketdata: malformed quote for {caller_symbol}",
                        file=sys.stderr,
                    )
                    malformed_symbols.append(caller_symbol)
                    continue
                quote = _parse_quote(caller_symbol, raw)
                if quote is not None:
                    all_quotes[caller_symbol] = quote
                    parsed_in_batch += 1
                else:
                    malformed_symbols.append(caller_symbol)
            for symbol in dict.fromkeys(malformed_symbols):
                record_error(f"malformed quote for {symbol}")
            if raw_quotes and parsed_in_batch == 0:
                record_error("no valid quotes in response")
        if not all_quotes and errors:
            return QuotesResult(ok=False, reason="; ".join(errors[:3]))
        return QuotesResult(
            ok=True, reason="; ".join(errors[:3]), quotes=all_quotes,
        )

    # ── public fetches ───────────────────────────────────────────────────────

    def get_stock_bars(
        self,
        symbols: list[str],
        *,
        timeframe: str = "1Day",
        start: datetime,
        end: datetime | None = None,
        feed: str = FEED_IEX,
    ) -> BarsResult:
        """Historical stock bars from the FREE IEX feed.

        A ``sip`` (or any paid) feed request is refused before network I/O —
        see :meth:`_feed_refusal`. Multi-symbol batched; fail-soft.
        """
        refusal = self._feed_refusal(feed)
        if refusal is not None:
            print(f"  marketdata: {refusal}", file=sys.stderr)
            return BarsResult(ok=False, reason=refusal)
        params: dict[str, Any] = {
            "timeframe": timeframe,
            "start": start.astimezone(UTC).isoformat(),
            "feed": FEED_IEX,                 # explicit — never left to default
            "adjustment": "raw",
            "limit": 10_000,
        }
        if end is not None:
            params["end"] = end.astimezone(UTC).isoformat()
        return self._fetch_batched(
            "/v2/stocks/bars", symbols, params, translate=False,
        )

    def get_crypto_bars(
        self,
        symbols: list[str],
        *,
        timeframe: str = "1Day",
        start: datetime,
        end: datetime | None = None,
    ) -> BarsResult:
        """Historical crypto bars (free — crypto data has no feed tiers).

        Callers pass internal yfinance-style pairs (``BTC-USD``); the wire
        symbol (``BTC/USD``) is translated at this boundary and results are
        keyed back by the caller's form.
        """
        params: dict[str, Any] = {
            "timeframe": timeframe,
            "start": start.astimezone(UTC).isoformat(),
            "limit": 10_000,
        }
        if end is not None:
            params["end"] = end.astimezone(UTC).isoformat()
        return self._fetch_batched(
            "/v1beta3/crypto/us/bars", symbols, params, translate=True,
        )

    def get_stock_latest_quotes(
        self,
        symbols: list[str],
        *,
        feed: str = FEED_IEX,
    ) -> QuotesResult:
        """Latest stock quotes explicitly pinned to the free IEX feed.

        Paid or unknown feeds are refused before credentials or network I/O.
        """
        refusal = self._feed_refusal(feed)
        if refusal is not None:
            print(f"  marketdata: {refusal}", file=sys.stderr)
            return QuotesResult(ok=False, reason=refusal)
        return self._fetch_latest_quotes(
            "/v2/stocks/quotes/latest",
            symbols,
            {"feed": FEED_IEX},
            translate=False,
        )

    def get_crypto_latest_quotes(self, symbols: list[str]) -> QuotesResult:
        """Latest free crypto quotes, translating ``BTC-USD`` to ``BTC/USD``."""
        return self._fetch_latest_quotes(
            "/v1beta3/crypto/us/latest/quotes",
            symbols,
            {},
            translate=True,
        )
