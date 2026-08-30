"""The hourly autonomous execution cycle must only run on a live trading day.

`schedule.every(1).hours` fires around the clock, and
`_run_allocation_execution_cycle` is the only scheduled job that submits
OPENING orders. Two failure modes without a market-hours guard:

  1. orders submitted outside regular trading hours — no submission path
     anywhere checks market hours;
  2. `live_candidates(mark_considered=True)` consumes every actionable signal
     the moment it is pulled, so an off-hours pass burns signals the next
     in-hours pass would have acted on.

(2) is why the guard must precede the candidate pull, which is what these
tests pin down.
"""

from __future__ import annotations

from unittest.mock import patch

from trading_bot import scanner


def _run_with_market(*, open_: bool) -> tuple[bool, bool]:
    """Run one cycle with market open/closed.

    Returns (broker_constructed, candidates_pulled).
    """
    with (
        patch.object(scanner, "is_market_open", return_value=open_),
        patch("trading_bot.broker.AlpacaBroker") as broker,
        patch("trading_bot.candidate_source.live_candidates") as candidates,
    ):
        candidates.return_value = []
        scanner._run_allocation_execution_cycle()
        return broker.called, candidates.called


def test_closed_market_submits_nothing_and_consumes_no_signals() -> None:
    broker_constructed, candidates_pulled = _run_with_market(open_=False)
    assert broker_constructed is False
    assert candidates_pulled is False, (
        "candidates were pulled with the market closed — live_candidates marks "
        "them considered, so those signals are burned"
    )


def test_open_market_still_runs_the_cycle() -> None:
    """Positive control: the guard must not disable the feature outright."""
    broker_constructed, _ = _run_with_market(open_=True)
    assert broker_constructed is True


def test_guard_runs_before_the_candidate_pull() -> None:
    """The guard is checked first, so a closed market short-circuits early."""
    calls: list[str] = []

    def _closed() -> bool:
        calls.append("market_check")
        return False

    with (
        patch.object(scanner, "is_market_open", side_effect=_closed),
        patch("trading_bot.candidate_source.live_candidates") as candidates,
    ):
        candidates.side_effect = lambda **_kw: calls.append("candidates") or []
        scanner._run_allocation_execution_cycle()

    assert calls == ["market_check"]
