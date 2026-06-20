"""Watchlist state machine — active<->benched management (Phase 3.3).

Closes the self-managing loop. Shadow promotion (trading_bot.shadow_discovery)
owns ENTRY into the active watchlist; this evaluator owns MANAGEMENT of names
already on it: demote an underperforming 'active' ticker to 'benched', recover
a 'benched' ticker that turns around back to 'active'. The two never overlap —
shadow only touches tickers NOT in the watchlist, this only touches tickers IN
it.

Stats are windowed resolved-trade outcomes over ``SM_WINDOW_DAYS`` across ALL
track_modes: a benched ticker keeps firing trades tagged 'shadow' (alerts
suppressed) and those are still its real outcomes, so recovery can be detected.

Hysteresis: demote at ``expectancy <= SM_DEMOTE_EXPECTANCY``, recover at
``expectancy >= SM_PROMOTE_EXPECTANCY``; the gap between them is a dead band
that prevents flapping. A MIN_ACTIVE floor caps demotions so the live set
never shrinks below ``SM_MIN_ACTIVE`` (recovery is never floor-gated). Tickers
are NEVER removed — benched is indefinite monitoring by design.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from trading_bot import config, db


@dataclass(frozen=True)
class TickerEvaluation:
    """One ticker's verdict this evaluation run.

    ``status`` is the status BEFORE the run; ``decision`` is one of
    ``'demote' | 'recover' | 'hold'``; ``new_status`` is the status after the
    decision is applied.
    """

    ticker: str
    status: str
    closed_count: int
    win_rate: float | None
    expectancy: float | None
    decision: str
    reason: str

    @property
    def new_status(self) -> str:
        if self.decision == "demote":
            return "benched"
        if self.decision == "recover":
            return "active"
        return self.status


@dataclass
class StateRun:
    """The full result of one evaluation pass."""

    evaluated_at: datetime
    evaluations: list[TickerEvaluation]

    @property
    def demotions(self) -> list[TickerEvaluation]:
        return [e for e in self.evaluations if e.decision == "demote"]

    @property
    def recoveries(self) -> list[TickerEvaluation]:
        return [e for e in self.evaluations if e.decision == "recover"]

    @property
    def holds(self) -> list[TickerEvaluation]:
        return [e for e in self.evaluations if e.decision == "hold"]

    @property
    def active_after(self) -> list[str]:
        return sorted(e.ticker for e in self.evaluations if e.new_status == "active")

    @property
    def benched_after(self) -> list[str]:
        return sorted(e.ticker for e in self.evaluations if e.new_status == "benched")


@dataclass
class _Proposal:
    """Mutable per-ticker working state; the floor pass may downgrade a
    proposed demotion to a hold before the frozen evaluation is built."""

    ticker: str
    status: str
    closed_count: int
    win_rate: float | None
    expectancy: float | None
    decision: str
    reason: str


def _windowed_stats(
    rows: Sequence[tuple[str, float | None]],
) -> tuple[int, float | None, float | None]:
    """``(closed_count, win_rate, expectancy)`` from ``(outcome, pnl)`` rows.

    ``expectancy`` is the mean resolved pnl % across decided trades (wins
    positive, losses negative) — i.e. the per-trade expected return.
    """
    wins = sum(1 for outcome, _ in rows if outcome == "win")
    losses = sum(1 for outcome, _ in rows if outcome == "loss")
    closed = wins + losses
    pnls = [pnl for _, pnl in rows if pnl is not None]
    win_rate = (wins / closed * 100.0) if closed else None
    expectancy = (sum(pnls) / len(pnls)) if pnls else None
    return closed, win_rate, expectancy


def windowed_stats_for(
    ticker: str, *, now: datetime | None = None
) -> tuple[int, float | None, float | None]:
    """Windowed resolved-trade stats for one ticker, across ALL track_modes."""
    run_ts = now if now is not None else datetime.now(UTC)
    cutoff = run_ts - timedelta(days=config.SM_WINDOW_DAYS)
    rows = db.get_resolved_outcomes(ticker, cutoff)  # track_mode=None -> all
    return _windowed_stats(rows)


def _propose(status: str, closed: int, expectancy: float | None) -> tuple[str, str]:
    """Per-ticker proposal BEFORE the active-floor pass."""
    if closed < config.SM_MIN_CLOSED_SIGNALS:
        return "hold", (
            f"insufficient sample ({closed} < {config.SM_MIN_CLOSED_SIGNALS})"
        )
    if expectancy is None:
        return "hold", "insufficient data (no resolved pnl)"
    if status == "active":
        if expectancy <= config.SM_DEMOTE_EXPECTANCY:
            return "demote", (
                f"expectancy {expectancy:.3f} <= demote bar "
                f"{config.SM_DEMOTE_EXPECTANCY}"
            )
        return "hold", (
            f"active held: expectancy {expectancy:.3f} > demote bar "
            f"{config.SM_DEMOTE_EXPECTANCY}"
        )
    # benched
    if expectancy >= config.SM_PROMOTE_EXPECTANCY:
        return "recover", (
            f"expectancy {expectancy:.3f} >= recover bar "
            f"{config.SM_PROMOTE_EXPECTANCY}"
        )
    return "hold", (
        f"benched held: expectancy {expectancy:.3f} < recover bar "
        f"{config.SM_PROMOTE_EXPECTANCY}"
    )


def _apply_active_floor(proposals: list[_Proposal]) -> None:
    """Cap demotions so the post-run active count never drops below
    ``SM_MIN_ACTIVE``. Recoveries (which only ADD to the active set) make room
    and are never floor-gated. Worst-expectancy demotions go first; the rest
    are downgraded to a hold in place."""
    active_now = sum(1 for p in proposals if p.status == "active")
    recoveries = sum(1 for p in proposals if p.decision == "recover")
    demotions = [p for p in proposals if p.decision == "demote"]
    max_demotions = max(0, active_now + recoveries - config.SM_MIN_ACTIVE)
    # Worst (most negative) expectancy demoted first.
    demotions.sort(key=lambda p: p.expectancy if p.expectancy is not None else 0.0)
    for held in demotions[max_demotions:]:
        held.decision = "hold"
        held.reason = (
            f"held: active floor (active would drop below "
            f"{config.SM_MIN_ACTIVE})"
        )


def evaluate_watchlist(
    *, dry_run: bool = False, now: datetime | None = None
) -> StateRun:
    """Evaluate every watchlist ticker and (unless dry_run) apply transitions.

    Computes each ticker's windowed expectancy, proposes demote/recover/hold
    with hysteresis, applies the MIN_ACTIVE floor to demotions, then — when not
    a dry run — flips statuses and records each transition to
    ``watchlist_transitions``. Per-ticker errors yield a hold (no transition)
    and are logged; nothing is ever removed.
    """
    run_ts = now if now is not None else datetime.now(UTC)

    try:
        entries = db.get_watchlist_entries()
    except Exception as exc:  # noqa: BLE001 - degrade gracefully, never crash
        print(f"  state eval: watchlist unreadable: {exc}", file=sys.stderr)
        entries = []

    proposals: list[_Proposal] = []
    for entry in entries:
        ticker = str(entry["ticker"])
        status = str(entry["status"])
        try:
            closed, win_rate, expectancy = windowed_stats_for(ticker, now=run_ts)
        except Exception as exc:  # noqa: BLE001 - one ticker must not kill the run
            print(f"  state eval error for {ticker}: {exc}", file=sys.stderr)
            proposals.append(
                _Proposal(ticker, status, 0, None, None, "hold", f"error: {exc}")
            )
            continue
        decision, reason = _propose(status, closed, expectancy)
        proposals.append(
            _Proposal(ticker, status, closed, win_rate, expectancy, decision, reason)
        )

    _apply_active_floor(proposals)

    evaluations: list[TickerEvaluation] = []
    for p in proposals:
        ev = TickerEvaluation(
            ticker=p.ticker,
            status=p.status,
            closed_count=p.closed_count,
            win_rate=p.win_rate,
            expectancy=p.expectancy,
            decision=p.decision,
            reason=p.reason,
        )
        evaluations.append(ev)
        if ev.decision in ("demote", "recover") and not dry_run:
            try:
                db.set_watchlist_status(ev.ticker, ev.new_status, changed_at=run_ts)
                db.insert_watchlist_transition(
                    ticker=ev.ticker,
                    from_status=ev.status,
                    to_status=ev.new_status,
                    evaluated_at=run_ts,
                    win_rate=ev.win_rate,
                    closed_count=ev.closed_count,
                    expectancy=ev.expectancy,
                    reason=ev.reason,
                )
            except Exception as exc:  # noqa: BLE001 - apply failure isn't fatal
                print(
                    f"  state apply error for {ev.ticker}: {exc}", file=sys.stderr
                )

    return StateRun(evaluated_at=run_ts, evaluations=evaluations)
