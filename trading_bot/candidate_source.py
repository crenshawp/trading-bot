"""Live fired-signal candidate sourcing — Phase 17.

Replaces ``allocation.sample_candidates()`` as the production source feeding
``build_plan``: a real query over the bot's ACTUAL fired signals instead of an
illustrative fixture. Three boundaries are enforced by the query itself
(``db.get_actionable_signals``), never left to downstream checks:

1. RECENCY — only signals fired within ``config.CANDIDATE_RECENCY_HOURS``.
2. CONSIDERED-ONCE — only signals with ``considered_at IS NULL``. When
   ``mark_considered=True`` (the execute-bound pull), every matched signal is
   stamped AT PULL TIME — before authorization, allocation, or submission — so
   a signal is considered exactly once regardless of what the plan later did
   with it. Phase 16's (ticker, pool) idempotency guard prevents double
   EXECUTION; this flag prevents endless re-CONSIDERATION.
3. TRACK — shadow-tracked signals and signals whose paper trade already
   resolved are excluded; long-term entries (no trade row — they are tracked in
   ``long_term_positions``) flow through the LEFT JOIN.

The crypto SWING signals (oversold_reversal / momentum_breakout) ARE returned
here — deliberately. There is exactly ONE sourcing path, and the EXISTING
Phase 14 data-only filter inside ``allocation.build_plan`` drops them before
routing, unchanged. Sourcing must never grow a second path that could bypass
that filter.

Every candidate is emitted in the exact :class:`allocation.Candidate` shape, so
build_plan / allocate / execute need no changes downstream of sourcing.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from trading_bot import config, db, signal_pairs
from trading_bot.allocation import Candidate


def _watchlist_status_map() -> dict[str, str]:
    """ticker -> 'active' | 'benched' from the DB watchlist. Tickers absent
    from the table (crypto, long-term names — the watchlist state machine
    governs the stock swing list only) default to active downstream."""
    return {
        str(e["ticker"]): str(e["status"]) for e in db.get_watchlist_entries()
    }


def _to_candidate(
    row: dict[str, Any], watchlist: dict[str, str], now: datetime,
) -> Candidate:
    """Map one joined signal row into the exact Candidate shape.

    Gate inputs mirror what the scanner consulted at fire time: watchlist
    status (absent = active, permissive-by-default), pair gate (no row =
    enabled), and the signal's own earnings_risk flag. Scoring context comes
    from the trade's fire-time advisory columns; a signal with no trade row
    (long-term entries) gets the same permissive defaults Phase 14's
    ``to_allocation_candidate`` used — its own gates already ran at generation.
    """
    ticker = str(row["ticker"])
    signal_type = str(row["signal_type"])
    _closed, _win_rate, expectancy = signal_pairs.windowed_stats_for_pair(
        ticker, signal_type, now=now,
    )
    rsi = row["trade_ind_rsi"] if row["trade_ind_rsi"] is not None else row["rsi"]
    return Candidate(
        ticker=ticker,
        signal_type=signal_type,
        direction=str(row["direction"]),
        asset_class=str(row["asset_class"]),
        entry=float(row["entry_price"]),
        atr=row["atr"],
        ticker_active=watchlist.get(ticker, "active") == "active",
        pair_enabled=db.get_signal_pair_status(ticker, signal_type) == "enabled",
        earnings_blackout=bool(row["earnings_risk"]),
        expectancy=expectancy,
        rsi=rsi,
        adx=row["trade_ind_adx"],
        obv=row["trade_ind_obv"],
        vol_regime=row["trade_ind_vol_regime"] or "unknown",
        sentiment_score=row["trade_sentiment_score"],
        concentration=row["trade_ind_concentration"] or "unknown",
    )


def live_candidates(
    *,
    now: datetime | None = None,
    window_hours: int | None = None,
    mark_considered: bool = False,
) -> list[Candidate]:
    """The production candidate source: live fired signals → Candidates.

    Queries the recency/considered/track boundaries (see module docstring),
    dedupes to the NEWEST signal per (ticker, signal_type) — a re-fired setup
    supersedes its older sibling — and maps each row into the exact
    :class:`allocation.Candidate` shape.

    ``mark_considered=True`` stamps EVERY matched signal (including older
    duplicates superseded by the dedupe — they were pulled too) the moment they
    are pulled, per the locked Phase 17 design: planned-but-not-executed,
    filtered, skipped, or unauthorized signals are all consumed exactly once
    and never re-planned indefinitely. Preview paths (``allocate plan``,
    ``execute`` without ``--confirm``) pass False so reviewing a plan does not
    consume the signals the confirmed run needs.
    """
    moment = now if now is not None else datetime.now(UTC)
    hours = window_hours if window_hours is not None else config.CANDIDATE_RECENCY_HOURS
    since = moment - timedelta(hours=hours)

    rows = db.get_actionable_signals(since)
    if mark_considered and rows:
        db.mark_signals_considered([int(r["id"]) for r in rows], moment)

    watchlist = _watchlist_status_map()
    candidates: list[Candidate] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:                       # newest first — first wins the dedupe
        key = (str(row["ticker"]), str(row["signal_type"]))
        if key in seen:
            continue
        seen.add(key)
        candidates.append(_to_candidate(row, watchlist, moment))
    return candidates
