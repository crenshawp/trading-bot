"""Live-shadow promotion evaluator (Phase 3.1-LIVE).

The sole promotion path into the active watchlist. For each shadow-universe
ticker that is NOT already on the active watchlist, it reads that ticker's
RESOLVED shadow trades (``track_mode='shadow'`` with a win/loss outcome)
within a recent window, computes win rate + expectancy from REAL resolved
outcomes, and promotes the name (``source='shadow'``) once it clears a small
sample + expectancy bar.

Everything reads the trades table — the same resolved outcomes the headline
reports use — so promotion is driven by the EMA21 Pullback actually working
on that name recently, not by a backtest. Backtest discovery no longer
promotes anything (see :mod:`trading_bot.discovery`, Section 5).

The operator triggers evaluation via the ``shadow evaluate`` CLI; shadow
SCANNING (data accumulation) runs automatically in the scanner loop.
"""

from __future__ import annotations

import dataclasses
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from trading_bot import db
from trading_bot.discovery_universe import SHADOW_UNIVERSE

# How far back resolved shadow outcomes count toward a promotion decision.
RECENT_WINDOW_DAYS = 60
# Minimum resolved (win/loss) shadow signals before a name can be promoted —
# filters tiny-sample noise.
MIN_SHADOW_SIGNALS = 10
# Minimum per-trade expectancy (mean resolved pnl %, points) to promote. A
# small positive bar: live-shadow is permissive, kickout prunes later.
PROMOTE_EXPECTANCY = 0.05


@dataclass(frozen=True)
class ShadowEvaluation:
    """One ticker's live-shadow standing at evaluation time.

    ``win_rate`` (0-100) and ``expectancy`` (mean resolved pnl %, per trade)
    are ``None`` when there are no resolved shadow trades in the window.
    ``eligible`` is the promotion bar; ``promoted`` is whether this run
    actually added it to the watchlist (False on dry-run or if already there).
    """

    ticker: str
    closed_count: int
    win_rate: float | None
    expectancy: float | None
    eligible: bool
    promoted: bool


def _compute_stats(
    rows: Sequence[tuple[str, float | None]],
) -> tuple[int, float | None, float | None]:
    """From ``(outcome, pnl_pct)`` rows return ``(closed_count, win_rate, expectancy)``.

    ``closed_count`` is the number of decided (win/loss) trades. ``expectancy``
    is the mean pnl_pct across them — which, with wins positive and losses
    negative, equals ``win_frac*avg_win - loss_frac*avg_loss``.
    """
    wins = sum(1 for outcome, _ in rows if outcome == "win")
    losses = sum(1 for outcome, _ in rows if outcome == "loss")
    closed = wins + losses
    pnls = [pnl for _, pnl in rows if pnl is not None]
    win_rate = (wins / closed * 100.0) if closed else None
    expectancy = (sum(pnls) / len(pnls)) if pnls else None
    return closed, win_rate, expectancy


def evaluate_ticker(
    ticker: str, *, now: datetime | None = None
) -> ShadowEvaluation:
    """Evaluate one ticker against its recent resolved shadow outcomes.

    Pure read + compute — no promotion, no persistence. ``promoted`` is always
    ``False`` here; the universe-level evaluator fills it in when it acts.
    """
    run_ts = now if now is not None else datetime.now(UTC)
    cutoff = run_ts - timedelta(days=RECENT_WINDOW_DAYS)
    rows = db.get_resolved_shadow_outcomes(ticker, cutoff)
    closed, win_rate, expectancy = _compute_stats(rows)
    eligible = (
        closed >= MIN_SHADOW_SIGNALS
        and expectancy is not None
        and expectancy >= PROMOTE_EXPECTANCY
    )
    return ShadowEvaluation(
        ticker=ticker,
        closed_count=closed,
        win_rate=win_rate,
        expectancy=expectancy,
        eligible=eligible,
        promoted=False,
    )


def _eval_to_row(ev: ShadowEvaluation) -> dict[str, object]:
    return {
        "ticker": ev.ticker,
        "closed_count": ev.closed_count,
        "win_rate": ev.win_rate,
        "expectancy": ev.expectancy,
        "eligible": ev.eligible,
        "promoted": ev.promoted,
    }


def evaluate_shadow_universe(
    *, dry_run: bool = False, now: datetime | None = None
) -> list[ShadowEvaluation]:
    """Evaluate every shadow-universe ticker not on the active watchlist.

    For each candidate, compute its recent resolved-shadow standing and, unless
    ``dry_run``, promote the eligible ones (``source='shadow'``) and persist the
    full run to ``shadow_evaluations``. Per-ticker errors are logged and that
    ticker is skipped with NO promotion — never a silent failure. Returns the
    evaluations (ordered as the shadow universe, candidates only).
    """
    run_ts = now if now is not None else datetime.now(UTC)

    try:
        active = set(db.get_active_watchlist())
    except Exception as exc:  # noqa: BLE001 - degrade gracefully, never crash
        print(
            f"  shadow eval: active watchlist unreadable, "
            f"evaluating full universe: {exc}",
            file=sys.stderr,
        )
        active = set()

    candidates = [t for t in SHADOW_UNIVERSE if t not in active]
    evaluations: list[ShadowEvaluation] = []

    for ticker in candidates:
        try:
            ev = evaluate_ticker(ticker, now=run_ts)
        except Exception as exc:  # noqa: BLE001 - one ticker must not kill the run
            print(f"  shadow eval error for {ticker}: {exc}", file=sys.stderr)
            continue

        promoted = False
        if ev.eligible and not dry_run:
            try:
                promoted = db.add_to_active_watchlist(ticker, "shadow")
            except Exception as exc:  # noqa: BLE001 - promotion failure isn't fatal
                print(
                    f"  shadow promote error for {ticker}: {exc}",
                    file=sys.stderr,
                )
                promoted = False
        evaluations.append(dataclasses.replace(ev, promoted=promoted))

    if not dry_run:
        try:
            db.insert_shadow_evaluations(
                run_ts.isoformat(), [_eval_to_row(e) for e in evaluations]
            )
        except Exception as exc:  # noqa: BLE001 - persistence failure shouldn't crash
            print(f"  shadow eval persist error: {exc}", file=sys.stderr)

    return evaluations
