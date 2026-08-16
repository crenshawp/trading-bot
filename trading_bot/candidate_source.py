"""Live fired-signal candidate sourcing — Phase 17.

Replaces ``allocation.sample_candidates()`` as the production source feeding
``build_plan``: a real query over the bot's ACTUAL fired signals instead of an
illustrative fixture. Three boundaries are enforced by the query itself
(``db.get_actionable_signals``), never left to downstream checks:

1. RECENCY — only signals fired within ``config.CANDIDATE_RECENCY_HOURS``.
2. CONSIDERED-ONCE — only signals with ``considered_at IS NULL``. Production
   pulls are non-destructive; the execution caller stamps a candidate only
   after a permanent filter decision or accepted/skipped execution. Capacity-
   blocked and transiently failed candidates remain retryable until recency
   expires.
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

import dataclasses
from datetime import UTC, datetime, timedelta
from typing import Any

from trading_bot import config, db, outcomes, signal_pairs
from trading_bot.allocation import AllocationResult, Candidate, route_pool
from trading_bot.models import is_hard_risk


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
    strategy_closed, _strategy_win_rate, strategy_expectancy = (
        signal_pairs.windowed_stats_for_signal_type(signal_type, now=now)
    )
    strategy_enabled = not (
        strategy_closed >= config.GLOBAL_SIGNAL_MIN_CLOSED
        and strategy_expectancy is not None
        and strategy_expectancy <= config.GLOBAL_SIGNAL_MIN_EXPECTANCY
    )
    rsi = row["trade_ind_rsi"] if row["trade_ind_rsi"] is not None else row["rsi"]
    candidate = Candidate(
        ticker=ticker,
        signal_type=signal_type,
        direction=str(row["direction"]),
        asset_class=str(row["asset_class"]),
        entry=float(row["entry_price"]),
        atr=row["atr"],
        ticker_active=watchlist.get(ticker, "active") == "active",
        strategy_enabled=strategy_enabled,
        pair_enabled=db.get_signal_pair_status(ticker, signal_type) == "enabled",
        # Only the existing hard-equivalent HIGH grade blocks. MEDIUM/LOW/
        # UNKNOWN are persisted advisory context and remain non-blocking.
        earnings_blackout=is_hard_risk(row["earnings_risk"]),
        expectancy=expectancy,
        rsi=rsi,
        adx=row["trade_ind_adx"],
        obv=row["trade_ind_obv"],
        vol_regime=row["trade_ind_vol_regime"] or "unknown",
        sentiment_score=row["trade_sentiment_score"],
        concentration=row["trade_ind_concentration"] or "unknown",
        signal_id=int(row["id"]),
    )
    # Phase 20: SWING-pool candidates carry the signal's EXISTING resolver
    # settlement deadline (fire timestamp + the Phase 1 hold window) so a
    # shares-fallback position inherits the original swing time stop.
    # LONG_TERM/CRYPTO pools have no such window by design — left None.
    if route_pool(candidate) == config.POOL_SWING:
        window = outcomes.hold_window_for(
            row["hold_estimate_days"], candidate.asset_class,
        )
        candidate = dataclasses.replace(
            candidate,
            hold_deadline=datetime.fromisoformat(str(row["timestamp"])) + window,
        )
    return candidate


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

    ``mark_considered=True`` remains available for explicit administrative
    consumption. The production executor passes False and consumes
    ``Candidate.source_signal_ids`` only after it knows the handling outcome.
    """
    moment = now if now is not None else datetime.now(UTC)
    hours = window_hours if window_hours is not None else config.CANDIDATE_RECENCY_HOURS
    since = moment - timedelta(hours=hours)

    rows = db.get_actionable_signals(since)
    if mark_considered and rows:
        db.mark_signals_considered([int(r["id"]) for r in rows], moment)

    watchlist = _watchlist_status_map()
    candidates: list[Candidate] = []
    source_ids: dict[tuple[str, str], list[int]] = {}
    for row in rows:
        key = (str(row["ticker"]), str(row["signal_type"]))
        source_ids.setdefault(key, []).append(int(row["id"]))
    seen: set[tuple[str, str]] = set()
    for row in rows:                       # newest first — first wins the dedupe
        key = (str(row["ticker"]), str(row["signal_type"]))
        if key in seen:
            continue
        seen.add(key)
        candidates.append(dataclasses.replace(
            _to_candidate(row, watchlist, moment),
            source_signal_ids=tuple(source_ids[key]),
        ))
    return candidates


def mark_final_candidates(
    candidates: list[Candidate],
    result: AllocationResult,
    execution_run: Any | None = None,
    *,
    now: datetime | None = None,
) -> int:
    """Durably consume only permanent filters and accepted/skipped orders.

    Allocation-capacity skips, authorization pauses, rejections, and transport
    errors remain retryable. Returns the number of newly stamped signal rows.
    """
    candidate_by_key = {(c.ticker, c.signal_type): c for c in candidates}
    candidate_by_id = {
        c.signal_id: c for c in candidates if c.signal_id is not None
    }
    final_candidates = {
        candidate_by_key[(skip.ticker, skip.signal_type)]
        for skip in result.skipped
        if skip.stage == "filter"
        and (skip.ticker, skip.signal_type) in candidate_by_key
    }
    if execution_run is not None:
        for execution in execution_run.executions:
            if execution.status not in {"submitted", "skipped"}:
                continue
            candidate = candidate_by_id.get(execution.order.signal_id)
            if candidate is not None:
                final_candidates.add(candidate)

    signal_ids = sorted({
        signal_id
        for candidate in final_candidates
        for signal_id in (
            candidate.source_signal_ids
            or (() if candidate.signal_id is None else (candidate.signal_id,))
        )
    })
    if not signal_ids:
        return 0
    return db.mark_signals_considered(signal_ids, now or datetime.now(UTC))
