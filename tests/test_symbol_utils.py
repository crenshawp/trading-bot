"""Tests for trading_bot.symbol_utils — yfinance symbol form at the boundary."""

from __future__ import annotations

from trading_bot.symbol_utils import to_yfinance_symbol


def test_class_share_ticker_is_hyphenated() -> None:
    # Confirmed broken before this: a raw BRK.B fetch returns nothing from
    # yfinance (marketdata compare reported it as missing_yfinance).
    assert to_yfinance_symbol("BRK.B") == "BRK-B"


def test_ordinary_tickers_pass_through_unchanged() -> None:
    for ticker in ("AAPL", "META", "GS", "BTC-USD", "SPY", "^VIX"):
        assert to_yfinance_symbol(ticker) == ticker


def test_translation_is_override_only_not_a_blanket_dot_rewrite() -> None:
    # A dotted symbol with no override must NOT be rewritten — only the
    # explicitly mapped class shares translate.
    assert to_yfinance_symbol("ABC.D") == "ABC.D"
