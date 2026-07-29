"""Fail-soft NewsAPI client — Phase 5.

Fetches recent headlines for a ticker, called ONLY when a signal fires (never
across the shadow universe) to bound cost/rate. Every failure path — missing
key, HTTP error, rate-limit, timeout, malformed body, or simply no articles —
returns a neutral ``NewsResult`` with ``ok=False`` and is logged. It NEVER
raises and NEVER blocks a signal: news being down can never stop an alert.

Results are cached per ticker for a short TTL so multiple fired signals on the
same ticker within one scan cycle don't refetch.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import requests

from trading_bot import secrets

# Short per-ticker cache so duplicate fired signals within one scan cycle reuse
# the same fetch. Cleared explicitly at the top of a scan via clear_cache().
_NEWS_CACHE_TTL_SECONDS = 600
_HTTP_TIMEOUT_SECONDS = 10
_DEFAULT_MAX_ARTICLES = 10

_REDACTED = "***REDACTED***"


def redact(text: str, secret: str | None) -> str:
    """Mask ``secret`` wherever it appears in ``text``.

    Defence in depth for the fail-soft logging below. The key is sent as a
    header rather than a query parameter precisely so it cannot reach a log,
    but a `requests` exception renders the full request URL and there is no
    guarantee a future redirect, proxy error, or library change won't surface
    it another way. A credential must never be one library detail away from
    the deploy log, so we scrub on the way out as well.
    """
    if not secret:
        return text
    return text.replace(secret, _REDACTED)


@dataclass(frozen=True)
class NewsResult:
    """Recent headlines for a ticker. ``ok=False`` means the fetch fell back to
    a neutral no-data result (error, missing key, or empty) — callers treat it
    as 'no news', never as a reason to suppress a signal."""

    ticker: str
    headlines: list[str] = field(default_factory=list)
    ok: bool = False

    @property
    def count(self) -> int:
        return len(self.headlines)


_cache: dict[str, tuple[NewsResult, datetime]] = {}


def clear_cache() -> None:
    """Drop the per-ticker cache. Called at the start of each scan cycle."""
    _cache.clear()


def _cached(ticker: str, now: datetime) -> NewsResult | None:
    entry = _cache.get(ticker)
    if entry is None:
        return None
    result, fetched_at = entry
    if now - fetched_at <= timedelta(seconds=_NEWS_CACHE_TTL_SECONDS):
        return result
    return None


def fetch_headlines(
    ticker: str,
    *,
    max_articles: int = _DEFAULT_MAX_ARTICLES,
    now: datetime | None = None,
) -> NewsResult:
    """Return recent headlines for ``ticker``. Never raises (fail-soft).

    A neutral ``NewsResult(ok=False)`` is returned and logged on any failure
    or when there are no articles / no API key.
    """
    moment = now if now is not None else datetime.now(UTC)
    hit = _cached(ticker, moment)
    if hit is not None:
        return hit

    result = _fetch_uncached(ticker, max_articles)
    _cache[ticker] = (result, moment)
    return result


def _fetch_uncached(ticker: str, max_articles: int) -> NewsResult:
    api_key = secrets.get_secret("NEWSAPI_KEY")
    if not api_key:
        print(
            f"  news: NEWSAPI_KEY not set — neutral no-data for {ticker}",
            file=sys.stderr,
        )
        return NewsResult(ticker=ticker, ok=False)

    yesterday = (datetime.now(UTC) - timedelta(days=1)).strftime("%Y-%m-%d")
    # The key goes in the X-Api-Key header, NOT the query string: `requests`
    # renders the full URL into its connection/proxy/retry exception messages,
    # and the `except` below prints that message to stderr — so an `apiKey=`
    # query parameter put the live NEWSAPI_KEY into the Railway deploy log on
    # every DNS blip, once per ticker. NewsAPI accepts either form.
    url = (
        "https://newsapi.org/v2/everything"
        f"?q={ticker} stock"
        f"&from={yesterday}"
        "&sortBy=publishedAt"
        "&language=en"
    )
    try:
        response = requests.get(
            url,
            timeout=_HTTP_TIMEOUT_SECONDS,
            headers={"X-Api-Key": api_key},
        )
        if response.status_code != 200:
            print(
                f"  news: HTTP {response.status_code} for {ticker} "
                f"— neutral no-data",
                file=sys.stderr,
            )
            return NewsResult(ticker=ticker, ok=False)
        body: Any = response.json()
        articles = body.get("articles", []) if isinstance(body, dict) else []
        headlines = [
            str(a["title"])
            for a in articles[:max_articles]
            if isinstance(a, dict) and a.get("title")
        ]
    except Exception as exc:  # noqa: BLE001 - news must never raise into a scan
        print(
            f"  news fetch error for {ticker}: {redact(str(exc), api_key)}",
            file=sys.stderr,
        )
        return NewsResult(ticker=ticker, ok=False)

    if not headlines:
        print(f"  news: no articles for {ticker} — neutral no-data", file=sys.stderr)
        return NewsResult(ticker=ticker, ok=False)

    return NewsResult(ticker=ticker, headlines=headlines, ok=True)
