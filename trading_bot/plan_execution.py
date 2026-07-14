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

import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from trading_bot import config, db, long_term, risk_of_ruin
from trading_bot import options_execution as oe
from trading_bot.allocation import ExecutionPlan, PlannedOrder
from trading_bot.broker import base as broker_base
from trading_bot.broker.base import Broker, OrderResult
from trading_bot.broker.options import OptionChainResult, OptionContract
from trading_bot.models import PlanExecution

_ET = ZoneInfo("America/New_York")

# A chain fetch returns the option chain for one underlying (the production
# caller passes AlpacaOptionsClient().get_option_chain) — injected so tests and
# the demo never touch the network.
ChainFetch = Callable[[str], OptionChainResult]

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


# ── routing: each PlannedOrder → its EXISTING pool submission path ───────────

# The scanner's take-profit multiple (tp = price ± 2×ATR, the same formula every
# fired swing signal carries). Together with config.RISK_ATR_STOP_MULTIPLE it
# lets the exit levels be recovered from the order's own fields — see
# _swing_exit_levels. Not a new sizing rule; the existing convention, reused.
_TP_ATR_MULTIPLE = 2.0


@dataclass(frozen=True)
class OrderExecution:
    """What the execute run did with ONE planned order."""

    order: PlannedOrder
    status: str                     # submitted | rejected | error | skipped
    reason: str
    vehicle: str | None = None      # option_full | option_undersized | shares
    order_ref: str | None = None    # broker order id when submitted


@dataclass(frozen=True)
class ExecutionRunResult:
    """The outcome of one execute run.

    ``ok=False`` means the ENTIRE run was refused (Phase 15 authorization) and
    nothing was submitted; ``note`` says why. Otherwise ``executions`` carries
    one entry per planned order — submitted, rejected, errored, or skipped.
    """

    ok: bool
    plan_id: str
    executions: list[OrderExecution] = field(default_factory=list)
    note: str = ""


def _swing_exit_levels(order: PlannedOrder) -> tuple[float | None, float | None]:
    """Recover the signal's TP/SL levels from the order's own fields.

    Phase 7 sizing guarantees ``dollar_risk = qty × (ATR × RISK_ATR_STOP_MULTIPLE)``,
    so the stop distance is exactly ``dollar_risk / qty`` and the scanner's
    take-profit distance is that × (2 / 1.5). These feed the EXISTING Phase 13
    exit watcher (which needs its TP/SL levels to manage the position) — no new
    exit logic, just the fired signal's own levels reconstructed. Returns
    ``(None, None)`` when the order cannot supply them (never raises).
    """
    if order.qty <= 0.0 or order.dollar_risk <= 0.0:
        return None, None
    stop_distance = order.dollar_risk / order.qty
    tp_distance = stop_distance * (_TP_ATR_MULTIPLE / config.RISK_ATR_STOP_MULTIPLE)
    if order.side == "buy":
        return order.entry + tp_distance, order.entry - stop_distance
    return order.entry - tp_distance, order.entry + stop_distance


def _status_from_order_result(result: OrderResult) -> str:
    """Map a broker OrderResult onto the audit-row status: an accepted order is
    'submitted', a structured broker refusal is 'rejected', anything else
    (transport failure, unknown state) is 'error'."""
    if result.ok:
        return STATUS_SUBMITTED
    if result.status == broker_base.STATUS_REJECTED:
        return STATUS_REJECTED
    return STATUS_ERROR


def _execute_swing(
    broker: Broker,
    order: PlannedOrder,
    *,
    now: datetime,
    option_chain_fetch: ChainFetch | None,
    options_available: bool,
) -> OrderExecution:
    """Route one SWING order through the Phase 13 hierarchy (full option →
    undersized option → fractional shares), submitting via ``execute_decision``.

    The chain fetch is fail-soft: unavailable/erroring chains simply mean
    options are unavailable and the hierarchy falls through to shares — exactly
    Phase 13's documented behaviour. The exit watcher's TP/SL levels are
    recovered from the order (``_swing_exit_levels``); the hold deadline is the
    order's CARRIED ``hold_deadline`` (Phase 20 — the original swing signal's
    resolver settlement deadline), applied to the SHARES vehicle only: option
    positions keep their Phase 13 behaviour untouched. A missing value on a
    SWING order degrades to no time stop (today's behaviour), logged — never a
    crash in the fire path.
    """
    contracts: list[OptionContract] = []
    chain_ok = False
    if options_available and option_chain_fetch is not None:
        try:
            chain = option_chain_fetch(order.ticker)
            chain_ok = chain.ok
            contracts = list(chain.contracts)
        except Exception as exc:  # noqa: BLE001 - a chain fetch must never sink the batch
            print(
                f"  execute: chain fetch error for {order.ticker}: {exc} "
                "-> shares fallback",
                file=sys.stderr,
            )

    direction = "long" if order.side == "buy" else "short"
    decision = oe.choose_execution(
        direction, order.est_cost, order.ticker, order.entry, contracts,
        ref_date=now.astimezone(_ET).date(),
        options_available=options_available and chain_ok,
    )
    if decision.vehicle == oe.VEHICLE_NONE:
        return OrderExecution(
            order, STATUS_REJECTED, decision.reason, vehicle=oe.VEHICLE_NONE,
        )

    tp, sl = _swing_exit_levels(order)
    # Phase 20: only the shares fallback takes the carried hold window —
    # option-position deadlines are Phase 13's domain and stay untouched.
    deadline = None
    if decision.vehicle == oe.VEHICLE_SHARES:
        deadline = order.hold_deadline
        if deadline is None:
            print(
                f"  execute: no hold window carried for {order.ticker} "
                "shares fallback - deadline unset (TP/SL only)",
                file=sys.stderr,
            )
    result, _position_id = oe.execute_decision(
        broker, decision, opened_at=now, tp=tp, sl=sl, deadline=deadline,
    )
    return OrderExecution(
        order, _status_from_order_result(result),
        result.reason or "ok", vehicle=decision.vehicle,
        order_ref=result.order_id,
    )


def _execute_long_term(
    broker: Broker, order: PlannedOrder, *, now: datetime,
) -> OrderExecution:
    """Route one LONG_TERM / CRYPTO order through the Phase 14 entry submission
    path — a fractional-share BUY at the allocator's computed qty/cost."""
    asset_class = "crypto" if order.pool == config.POOL_CRYPTO else "stock"
    result, _position_id = long_term.submit_long_term_entry(
        broker, ticker=order.ticker, asset_class=asset_class,
        qty=order.qty, entry_price=order.entry, now=now,
    )
    return OrderExecution(
        order, _status_from_order_result(result),
        result.reason or "ok", vehicle=oe.VEHICLE_SHARES,
        order_ref=result.order_id,
    )


def _record(execution: OrderExecution, plan_id: str, now: datetime) -> None:
    """Persist one audit row. Fail-soft: a persistence error is logged and the
    batch continues — the audit trail must never block the remaining orders."""
    try:
        db.insert_plan_execution(PlanExecution(
            plan_id=plan_id, executed_at=now, ticker=execution.order.ticker,
            pool=execution.order.pool, status=execution.status,
            signal_type=execution.order.signal_type, side=execution.order.side,
            qty=execution.order.qty, vehicle=execution.vehicle,
            order_ref=execution.order_ref, reason=execution.reason,
        ))
    except Exception as exc:  # noqa: BLE001 - the audit row must never block the batch
        print(
            f"  execute: audit-row insert failed for {execution.order.ticker} "
            f"({exc})",
            file=sys.stderr,
        )


def execute_plan(
    broker: Broker,
    plan: ExecutionPlan,
    *,
    now: datetime | None = None,
    option_chain_fetch: ChainFetch | None = None,
    options_available: bool = True,
) -> ExecutionRunResult:
    """Submit every order in an approved plan via its pool's EXISTING path.

    OPERATOR-TRIGGERED only (the ``allocate execute --confirm`` CLI); nothing
    schedules this. The run:

    1. checks the Phase 15 ``new_position_entry`` authorization INDEPENDENTLY —
       revoked refuses the whole run, submits nothing;
    2. skips any (ticker, pool) already SUBMITTED this cycle (idempotency);
    3. routes each remaining order by pool — SWING through the Phase 13
       full-option → undersized-option → shares hierarchy, LONG_TERM / CRYPTO
       through the Phase 14 fractional-share entry;
    4. records every outcome (submitted / rejected / error / skipped) in
       ``plan_executions``.

    FAIL-SOFT batch: a rejection or error on one order never blocks the others.
    """
    moment = now if now is not None else datetime.now(UTC)
    plan_id = make_plan_id(moment)

    auth = check_authorization()
    if not auth.authorized:
        print(f"  execute: REFUSED - {auth.reason}", file=sys.stderr)
        return ExecutionRunResult(
            ok=False, plan_id=plan_id, executions=[], note=auth.reason,
        )

    executions: list[OrderExecution] = []
    for order in plan.orders:
        try:
            if already_executed(order.ticker, order.pool, moment):
                execution = OrderExecution(
                    order, STATUS_SKIPPED, SKIP_ALREADY_EXECUTED,
                )
            elif order.pool in (config.POOL_LONG_TERM, config.POOL_CRYPTO):
                execution = _execute_long_term(broker, order, now=moment)
            elif order.pool == config.POOL_SWING:
                execution = _execute_swing(
                    broker, order, now=moment,
                    option_chain_fetch=option_chain_fetch,
                    options_available=options_available,
                )
            else:
                # Defensive: a pool this router does not know is never submitted.
                execution = OrderExecution(
                    order, STATUS_ERROR, f"unknown pool {order.pool!r}",
                )
        except Exception as exc:  # noqa: BLE001 - one order must never block the batch
            print(
                f"  execute: error on {order.ticker}/{order.pool}: {exc}",
                file=sys.stderr,
            )
            execution = OrderExecution(order, STATUS_ERROR, str(exc))
        _record(execution, plan_id, moment)
        executions.append(execution)

    return ExecutionRunResult(ok=True, plan_id=plan_id, executions=executions)
