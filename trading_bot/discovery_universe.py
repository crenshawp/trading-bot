"""Phase 3.1 discovery universe — the candidate pool and exclusion list.

Discovery scans a broad, liquid, large-cap universe (S&P 100 quality) and
backtests the existing EMA21 Pullback signal against each name. The
*effective* universe is :data:`DISCOVERY_UNIVERSE` minus
:data:`EXCLUDED_TICKERS` — names we previously dropped and do not want to
re-test.

These are intentionally plain module constants (not DB rows): the candidate
pool is a code-reviewed decision, whereas the *live* watchlist that the
running bot scans is database-driven (see :mod:`trading_bot.db`
``active_watchlist`` and the scanner refactor in Phase 3.1 Section 2).
"""

from __future__ import annotations

from collections.abc import Collection, Sequence

# ~40 large-cap, liquid, EMA-respecting S&P 100 quality names. Seeded with the
# five required tickers (MSFT, NVDA, PLTR, AMD, COST) and expanded across
# mega-cap tech, healthcare, staples, energy, industrials, and financials.
# Order is preserved by effective_universe() so the scan is deterministic.
DISCOVERY_UNIVERSE: list[str] = [
    "MSFT", "NVDA", "PLTR", "AMD", "COST",   # required seeds
    "GOOGL", "META", "AMZN", "AVGO", "LLY",
    "TSLA", "BLK", "GS", "NOW", "JNJ",
    "WMT", "PG", "XOM", "CVX", "KO",
    "PEP", "MRK", "ABBV", "TMO", "ACN",
    "MCD", "CSCO", "ADBE", "QCOM", "TXN",
    "AMAT", "LRCX", "INTU", "UNH", "CAT",
    "HON", "AXP", "LIN", "ISRG", "BKNG",
]

# Previously-dropped names. Discovery must never re-test these even if one is
# accidentally added to DISCOVERY_UNIVERSE above. frozenset for O(1) lookup.
EXCLUDED_TICKERS: frozenset[str] = frozenset(
    {"AAPL", "NFLX", "V", "MA", "CRM", "HD", "ORCL", "QQQ", "JPM"}
)


def _filter_universe(
    universe: Sequence[str], excluded: Collection[str]
) -> list[str]:
    """Return ``universe`` minus ``excluded``, order-preserving and deduped.

    Pure helper so the subtraction + dedup is testable in isolation without
    reaching into module globals.
    """
    seen: set[str] = set()
    result: list[str] = []
    for ticker in universe:
        if ticker in excluded or ticker in seen:
            continue
        seen.add(ticker)
        result.append(ticker)
    return result


def effective_universe() -> list[str]:
    """The candidate tickers discovery will actually backtest.

    :data:`DISCOVERY_UNIVERSE` minus :data:`EXCLUDED_TICKERS`, with order
    preserved and duplicates removed.
    """
    return _filter_universe(DISCOVERY_UNIVERSE, EXCLUDED_TICKERS)
