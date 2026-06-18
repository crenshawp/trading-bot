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

# ──────────────────────────────────────────────────────────────────────────
# Live-shadow universe (Phase 3.1-LIVE)
# ──────────────────────────────────────────────────────────────────────────
#
# 100 of the most liquid, highest-profile large/mega-cap and notable
# high-momentum names. The scanner shadow-tracks every one of these that is
# NOT already on the active watchlist: it runs the IDENTICAL signal pipeline,
# opens trades tagged track_mode='shadow', and resolves them via the existing
# resolver — but suppresses alerts. Promotion to the active watchlist is then
# driven by RESOLVED shadow outcomes (see trading_bot.shadow_discovery).
#
# This is a STATIC, hand-editable seed list. A later phase will refresh it
# automatically; for now, edit it here. EXCLUDED_TICKERS does NOT gate this
# universe — under live-shadow, real resolved data is the judge, so a name's
# historical backtest standing is irrelevant (note AAPL/NFLX/V/MA/CRM/HD/ORCL
# appear here despite being excluded from the backtest universe).
SHADOW_UNIVERSE: list[str] = [
    # Mega-cap tech & semis
    "AAPL", "MSFT", "NVDA", "GOOGL", "GOOG", "AMZN", "META", "AVGO", "TSLA", "ORCL",
    "AMD", "NFLX", "ADBE", "CRM", "CSCO", "QCOM", "TXN", "INTC", "IBM", "NOW",
    "INTU", "AMAT", "LRCX", "KLAC", "MU", "PLTR", "PANW", "SNPS", "CDNS", "ANET",
    "MRVL", "ARM", "SMCI", "DELL", "CRWD", "FTNT", "ADI",
    # High-growth / momentum software & platforms
    "APP", "UBER", "ABNB", "SHOP", "PYPL", "COIN", "SNOW", "DDOG", "NET", "MDB",
    "ZS", "TEAM", "WDAY",
    # Financials
    "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "BLK", "SCHW", "AXP",
    "C", "SPGI", "ADP",
    # Healthcare
    "LLY", "UNH", "JNJ", "ABBV", "MRK", "PFE", "TMO", "ABT", "DHR", "AMGN",
    "ISRG", "VRTX", "REGN",
    # Consumer & staples
    "WMT", "COST", "PG", "KO", "PEP", "MCD", "HD", "NKE", "SBUX", "LOW",
    "TGT", "BKNG",
    # Energy & industrials
    "XOM", "CVX", "COP", "CAT", "BA", "GE", "HON", "UPS", "LIN", "DE",
    "LMT", "RTX",
]


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
