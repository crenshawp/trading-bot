"""The long-term exit watcher must not queue stale share limits out of hours.

`_run_exit_watcher_cycle` is registered `schedule.every(1).hours`, so it fires
overnight, at weekends, and on holidays. The OPTION half was given a
market-hours guard whose docstring states the reason verbatim — "a
weekend/overnight watcher cannot queue a stale limit for the next session" —
but the long-term / swing-fallback half submits through the same durable path
with `time_in_force=TIF_DAY`, priced off DAILY candles, and had no equivalent
guard.

The consequence is not a missed check but a suspended stop: the out-of-hours
limit is queued for the next open, `_submit_long_term_watcher_exit` refuses any
further attempt while it is nonterminal, and on a gap-down the resting limit
sits above the market until it expires at that session's close.

Crypto must stay watched 24/7, which is why the guard splits the book by asset
class rather than deferring it wholesale.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from trading_bot import scanner


def _position(ticker: str, asset_class: str) -> Any:
    """A minimal open long-term row — only `asset_class` is read by the guard."""
    class _P:
        pass

    p = _P()
    p.ticker = ticker
    p.asset_class = asset_class
    return p


def _run_cycle(*, market_open: bool, book: list[Any]) -> list[list[Any] | None]:
    """Run one exit-watcher cycle.

    Returns one entry per call to `watch_long_term_positions`, each holding the
    `positions` kwarg it received. The watcher is always called (that it runs in
    production at all is a pinned regression), so what distinguishes new code
    from old is the kwarg: the old code passed none at all — `None`, meaning
    "read the whole open book from the DB yourself".
    """
    seen: list[list[Any] | None] = []

    def _watch(_broker: Any, **kwargs: Any) -> list[Any]:
        seen.append(kwargs.get("positions"))
        return []

    with (
        patch.object(scanner, "is_market_open", return_value=market_open),
        patch("trading_bot.broker.AlpacaBroker"),
        patch("trading_bot.db.get_open_option_positions", return_value=[]),
        patch("trading_bot.db.get_open_long_term_positions", return_value=book),
        patch("trading_bot.long_term.watch_long_term_positions", side_effect=_watch),
    ):
        scanner._run_exit_watcher_cycle()

    return seen


def test_closed_market_defers_share_positions() -> None:
    """An all-equity book out of hours reaches the watcher as an empty book."""
    book = [_position("META", "stock"), _position("GOOGL", "stock")]
    calls = _run_cycle(market_open=False, book=book)
    assert len(calls) == 1, "the watcher must stay wired in production"
    assert calls[0] == [], (
        "share positions were watched with the equity market closed — the exit "
        "limit is priced off the last daily close and queued for the next open"
    )


def test_closed_market_still_watches_crypto() -> None:
    """Crypto trades 24/7; deferring it would suspend its stop every night."""
    book = [_position("META", "stock"), _position("BTC-USD", "crypto")]
    calls = _run_cycle(market_open=False, book=book)
    assert len(calls) == 1
    watched = calls[0]
    assert watched is not None, "the watcher was handed the whole DB book again"
    assert [p.ticker for p in watched] == ["BTC-USD"]


def test_open_market_watches_the_whole_book() -> None:
    """Positive control: the guard must not disable the feature in hours."""
    book = [_position("META", "stock"), _position("BTC-USD", "crypto")]
    calls = _run_cycle(market_open=True, book=book)
    assert len(calls) == 1
    watched = calls[0]
    assert watched is not None, "the watcher was handed the whole DB book again"
    assert [p.ticker for p in watched] == ["META", "BTC-USD"]


def test_empty_book_passes_an_empty_book_not_a_none() -> None:
    """`None` would make the watcher re-read the whole DB book, undoing the filter."""
    calls = _run_cycle(market_open=True, book=[])
    assert calls == [[]]
