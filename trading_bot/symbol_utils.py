"""Symbol form translation at provider boundaries.

The bot stores tickers in their Alpaca-facing form (``BRK.B``) everywhere —
watchlists, signals, plans, orders. Yahoo Finance uses a hyphen for class
shares (``BRK-B``), so a raw ``BRK.B`` fetch returns nothing at all. This
module translates at the yfinance CALL BOUNDARY only; internal storage,
display, and the Alpaca side keep the dotted form.

The mirror of ``broker.alpaca.to_alpaca_symbol`` (which converts the internal
yfinance-style crypto pair ``BTC-USD`` to Alpaca's ``BTC/USD``) — each
provider gets its own form applied at the edge, and neither leaks inward.
"""

from __future__ import annotations

# Explicit overrides only — never a blanket ``.`` → ``-`` rewrite, which would
# corrupt symbols that legitimately carry a dot. Add entries as class-share
# tickers enter the universe.
_YFINANCE_SYMBOL_OVERRIDES: dict[str, str] = {"BRK.B": "BRK-B"}


def to_yfinance_symbol(ticker: str) -> str:
    """Translate an internal ticker to its yfinance form.

    ``BRK.B`` → ``BRK-B``; every other symbol passes through unchanged.
    """
    return _YFINANCE_SYMBOL_OVERRIDES.get(ticker, ticker)
