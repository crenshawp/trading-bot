"""Plan execution — the Phase 12 plan → broker bridge (Phase 16, PAPER).

Closes the gap between "plan produced" and "order actually submitted": an
OPERATOR-TRIGGERED path (``allocate execute --confirm``) that takes an approved
:class:`~trading_bot.allocation.ExecutionPlan` and routes each PlannedOrder to
the correct EXISTING submission path by pool. Nothing here is scheduled —
opening new exposure stays operator-gated, exactly the discipline every prior
phase held (closing/protecting positions remains the watchers' job, untouched).

This module owns the execution AUDIT TRAIL and IDEMPOTENCY GUARD: every planned
order the execute command acts on is recorded in ``plan_executions``
(submitted / rejected / error / skipped, with the broker order ref), and a
(ticker, pool) that already has a SUBMITTED row within the current cycle is
skipped as already-executed — running execute twice on the same plan never
double-submits. Only 'submitted' blocks; a rejected or errored attempt may be
retried.

Still PAPER ONLY. No real money.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from trading_bot import config, db, risk_of_ruin

_ET = ZoneInfo("America/New_York")

# Plan-execution row statuses (mirrored by models.VALID_PLAN_EXECUTION_STATUSES).
STATUS_SUBMITTED = "submitted"
STATUS_REJECTED = "rejected"
STATUS_ERROR = "error"
STATUS_SKIPPED = "skipped"

SKIP_ALREADY_EXECUTED = "already-executed"


def make_plan_id(now: datetime) -> str:
    """A run identifier derived from the execution timestamp (UTC).

    Every row written by one execute run carries the same plan_id, so the audit
    trail groups naturally by run."""
    return "plan-" + now.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


def cycle_start(now: datetime) -> datetime:
    """Start of the current execution CYCLE, in UTC.

    A cycle is one ET trading date — the plan cadence is daily (the stock scan
    fires 09:31 ET), so "the same planned order" means the same (ticker, pool)
    on the same ET calendar day. Midnight ET converted to UTC, matching the
    UTC ``executed_at`` timestamps the audit rows store.
    """
    et_day = now.astimezone(_ET).date()
    midnight_et = datetime(et_day.year, et_day.month, et_day.day, tzinfo=_ET)
    return midnight_et.astimezone(UTC)


def already_executed(ticker: str, pool: str, now: datetime) -> bool:
    """The idempotency guard: True when a SUBMITTED plan-execution row already
    exists for (ticker, pool) in the current cycle.

    Running execute twice on the same plan therefore never double-submits.
    Rejected / error / skipped rows do NOT block — a failed attempt may be
    retried within the same cycle.
    """
    return db.has_submitted_plan_execution(ticker, pool, cycle_start(now))


# ── authorization gate (Phase 15 — checked INDEPENDENTLY, never assumed) ─────


@dataclass(frozen=True)
class AuthorizationCheck:
    """Whether the execute run may open new positions, and why not if not."""

    authorized: bool
    reason: str


def check_authorization() -> AuthorizationCheck:
    """Consult the Phase 15 ledger before executing ANYTHING.

    ``build_plan`` already goes entries-empty when ``new_position_entry`` is
    revoked, but execute NEVER assumes the plan it holds was filtered — it asks
    the ledger itself. Revoked (Tier 1 pause, Tier 2 shutdown, or any manual
    revocation) refuses the ENTIRE run: nothing is submitted, and the recorded
    revocation reason plus the risk state are surfaced so the operator knows
    exactly why.
    """
    if risk_of_ruin.is_entry_authorized():
        return AuthorizationCheck(True, "new_position_entry authorized")
    revoke_reason = risk_of_ruin.revoke_reason(config.ENTRY_CAPABILITY)
    state = risk_of_ruin.get_state()
    return AuthorizationCheck(
        False,
        f"new_position_entry REVOKED ({revoke_reason or 'no reason recorded'}; "
        f"risk state: {state})",
    )
