"""Per-(ticker, signal_type) gate evaluator — Phase 4 strategy evolution.

One grain finer than the Phase 3.3 watchlist state machine: instead of
managing whole tickers (active/benched), this manages individual
``(ticker, signal_type)`` PAIRS (enabled/muted). A pair that consistently
loses is MUTED — it stops alerting but keeps firing as a shadow trade
(``track_mode='shadow'``, no alert) so it keeps collecting resolved outcomes
and re-enables when it recovers. Same window, min-sample, and hysteresis as
the watchlist machine; NO min-floor (muting one setup on a ticker can never
empty the watchlist, so the floor is deliberately omitted).

Pairs are default-enabled (a pair with no status row is treated as enabled),
evaluated regardless of the ticker's active/benched status — so a benched
ticker's pair statuses are already correct when it re-activates — and never
removed.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from trading_bot import config, db, readiness


@dataclass(frozen=True)
class PairEvaluation:
    """One (ticker, signal_type) pair's verdict this run.

    ``status`` is the status BEFORE the run; ``decision`` is one of
    ``'mute' | 'enable' | 'hold'``; ``new_status`` is the status after applying.
    """

    ticker: str
    signal_type: str
    status: str
    closed_count: int
    win_rate: float | None
    expectancy: float | None
    decision: str
    reason: str

    @property
    def new_status(self) -> str:
        if self.decision == "mute":
            return "muted"
        if self.decision == "enable":
            return "enabled"
        return self.status


@dataclass(frozen=True)
class PairStat:
    """A pair's current windowed standing — used by reporting + the status CLI."""

    ticker: str
    signal_type: str
    status: str
    closed_count: int
    win_rate: float | None
    expectancy: float | None


@dataclass
class PairRun:
    """The full result of one per-pair evaluation pass."""

    evaluated_at: datetime
    evaluations: list[PairEvaluation]

    @property
    def mutes(self) -> list[PairEvaluation]:
        return [e for e in self.evaluations if e.decision == "mute"]

    @property
    def enables(self) -> list[PairEvaluation]:
        return [e for e in self.evaluations if e.decision == "enable"]

    @property
    def holds(self) -> list[PairEvaluation]:
        return [e for e in self.evaluations if e.decision == "hold"]


def _windowed_stats(
    rows: Sequence[tuple[str, float | None]],
) -> tuple[int, float | None, float | None]:
    """``(closed_count, win_rate, expectancy)`` from ``(outcome, pnl)`` rows.

    ``expectancy`` is the mean resolved pnl % across decided trades — the
    per-trade expected return.
    """
    wins = sum(1 for outcome, _ in rows if outcome == "win")
    losses = sum(1 for outcome, _ in rows if outcome == "loss")
    closed = wins + losses
    pnls = [pnl for _, pnl in rows if pnl is not None]
    win_rate = (wins / closed * 100.0) if closed else None
    expectancy = (sum(pnls) / len(pnls)) if pnls else None
    return closed, win_rate, expectancy


def windowed_stats_for_pair(
    ticker: str, signal_type: str, *, now: datetime | None = None
) -> tuple[int, float | None, float | None]:
    """Windowed resolved stats for one pair, across ALL track_modes."""
    run_ts = now if now is not None else datetime.now(UTC)
    cutoff = run_ts - timedelta(days=config.SP_WINDOW_DAYS)
    rows = db.get_resolved_outcomes(ticker, cutoff, signal_type=signal_type)
    return _windowed_stats(rows)


def windowed_stats_for_signal_type(
    signal_type: str,
    *,
    now: datetime | None = None,
) -> tuple[int, float | None, float | None]:
    """Windowed ACTIVE-book stats for a setup across every ticker."""
    run_ts = now if now is not None else datetime.now(UTC)
    cutoff = run_ts - timedelta(days=config.SP_WINDOW_DAYS)
    return _windowed_stats(
        db.get_resolved_signal_type_outcomes(signal_type, cutoff)
    )


def _propose(status: str, closed: int, expectancy: float | None) -> tuple[str, str]:
    """Per-pair proposal with hysteresis."""
    if closed < config.SP_MIN_CLOSED_SIGNALS:
        return "hold", (
            f"insufficient sample ({closed} < {config.SP_MIN_CLOSED_SIGNALS})"
        )
    if expectancy is None:
        return "hold", "insufficient data (no resolved pnl)"
    if status == "enabled":
        if expectancy <= config.SP_MUTE_EXPECTANCY:
            return "mute", (
                f"expectancy {expectancy:.3f} <= mute bar "
                f"{config.SP_MUTE_EXPECTANCY}"
            )
        return "hold", (
            f"enabled held: expectancy {expectancy:.3f} > mute bar "
            f"{config.SP_MUTE_EXPECTANCY}"
        )
    # muted
    if expectancy >= config.SP_ENABLE_EXPECTANCY:
        return "enable", (
            f"expectancy {expectancy:.3f} >= enable bar "
            f"{config.SP_ENABLE_EXPECTANCY}"
        )
    return "hold", (
        f"muted held: expectancy {expectancy:.3f} < enable bar "
        f"{config.SP_ENABLE_EXPECTANCY}"
    )


def pair_stats(*, now: datetime | None = None) -> list[PairStat]:
    """Current windowed standing for every pair present in trades."""
    run_ts = now if now is not None else datetime.now(UTC)
    out: list[PairStat] = []
    for ticker, signal_type in db.get_traded_signal_pairs():
        status = db.get_signal_pair_status(ticker, signal_type)
        closed, win_rate, expectancy = windowed_stats_for_pair(
            ticker, signal_type, now=run_ts
        )
        out.append(
            PairStat(ticker, signal_type, status, closed, win_rate, expectancy)
        )
    return out


def evaluate_signal_pairs(
    *, dry_run: bool = False, now: datetime | None = None
) -> PairRun:
    """Evaluate every traded pair and (unless dry_run) apply enable/mute flips.

    Computes each pair's windowed expectancy across all track_modes, proposes
    mute/enable/hold with hysteresis, then — when not a dry run AND the
    capability is ready — upserts the new status and records each transition to
    ``signal_pair_transitions``. Pairs are evaluated regardless of their
    ticker's active/benched status. Per-pair errors yield a hold (no transition)
    and are logged; nothing is ever removed; there is no min-floor.

    Phase 10: the capability is ACTIVE iff ``readiness.is_ready`` — below the
    centralized threshold it is dormant (evaluations still computed, no
    transition applied). Behavior preserved: the per-pair sample gate already
    holds everything below threshold.
    """
    run_ts = now if now is not None else datetime.now(UTC)
    ready = readiness.is_ready("pair_gating")
    effective_dry_run = dry_run or not ready
    if not dry_run and not ready:
        print(
            "  pair_gating dormant: below readiness threshold "
            "(no transitions applied)",
            file=sys.stderr,
        )

    try:
        pairs = db.get_traded_signal_pairs()
    except Exception as exc:  # noqa: BLE001 - degrade gracefully, never crash
        print(f"  pair eval: traded pairs unreadable: {exc}", file=sys.stderr)
        pairs = []

    evaluations: list[PairEvaluation] = []
    for ticker, signal_type in pairs:
        try:
            status = db.get_signal_pair_status(ticker, signal_type)
            closed, win_rate, expectancy = windowed_stats_for_pair(
                ticker, signal_type, now=run_ts
            )
        except Exception as exc:  # noqa: BLE001 - one pair must not kill the run
            print(
                f"  pair eval error for {ticker}/{signal_type}: {exc}",
                file=sys.stderr,
            )
            evaluations.append(
                PairEvaluation(
                    ticker, signal_type, "enabled", 0, None, None,
                    "hold", f"error: {exc}",
                )
            )
            continue

        decision, reason = _propose(status, closed, expectancy)
        ev = PairEvaluation(
            ticker=ticker,
            signal_type=signal_type,
            status=status,
            closed_count=closed,
            win_rate=win_rate,
            expectancy=expectancy,
            decision=decision,
            reason=reason,
        )
        evaluations.append(ev)

        if ev.decision in ("mute", "enable") and not effective_dry_run:
            try:
                db.set_signal_pair_status(
                    ev.ticker, ev.signal_type, ev.new_status, changed_at=run_ts
                )
                db.insert_signal_pair_transition(
                    ticker=ev.ticker,
                    signal_type=ev.signal_type,
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
                    f"  pair apply error for {ev.ticker}/{ev.signal_type}: {exc}",
                    file=sys.stderr,
                )

    return PairRun(evaluated_at=run_ts, evaluations=evaluations)
