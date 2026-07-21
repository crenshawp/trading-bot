"""Options execution — selection, cost model, hierarchy, exit watcher (Phase 13).

Consumes a Phase 12 allocated SWING candidate (side + allocated capital +
underlying) and turns it into an actual single-leg option order (buy call / buy
put), with a fractional-share fallback. Still PAPER ONLY.

Execution hierarchy per allocated swing signal (Section 4):
  1. a FULL-sized option at target delta (0.65-0.75), or
  2. an UNDERSIZED option (delta band widened down to 0.50, still liquidity-
     gated, still a WHOLE contract) that fits the allocated capital, or
  3. fall back to FRACTIONAL SHARES via the existing Phase 11 equity order path.

Greeks (theta/vega/gamma) are stored for audit but are NOT hard gates this phase.
No spreads / multi-leg. No naive market orders (LIMIT default). Alpaca has no
bracket/OTO for options, so exits are managed explicitly (Section 6).
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime

from trading_bot import config, db, order_lifecycle, risk_of_ruin
from trading_bot.broker.base import ORDER_TYPE_LIMIT, TIF_DAY, Broker, OrderResult
from trading_bot.broker.options import (
    OPTION_TYPE_CALL,
    OPTION_TYPE_PUT,
    OptionContract,
)
from trading_bot.models import LongTermPosition, OptionPosition

VEHICLE_OPTION_FULL = "option_full"
VEHICLE_OPTION_UNDERSIZED = "option_undersized"
VEHICLE_SHARES = "shares"
VEHICLE_NONE = "none"


def _dte(expiry: str, ref_date: date) -> int | None:
    """Days-to-expiration from ``ref_date`` to the ``'YYYY-MM-DD'`` expiry, or
    None if the date is malformed (never raises)."""
    try:
        exp = date.fromisoformat(expiry)
    except ValueError:
        return None
    return (exp - ref_date).days


def _passes_liquidity(
    contract: OptionContract, min_open_interest: int, max_spread_pct: float,
) -> bool:
    """Liquidity floor: enough open interest AND a tight enough bid-ask spread.
    A contract missing either datum fails (we do not trade what we cannot price)."""
    if contract.open_interest is None or contract.open_interest < min_open_interest:
        return False
    spread = contract.spread_pct
    return spread is not None and spread <= max_spread_pct


def select_contract(
    side: str,
    contracts: Sequence[OptionContract],
    *,
    ref_date: date,
    delta_low: float = config.TARGET_DELTA_LOW,
    delta_high: float = config.TARGET_DELTA_HIGH,
    min_dte: int = config.MIN_DTE,
    min_open_interest: int = config.MIN_OPEN_INTEREST,
    max_spread_pct: float = config.MAX_SPREAD_PCT,
) -> OptionContract | None:
    """Pick the best contract in the delta band that clears DTE + liquidity.

    ``side`` maps call/long → a CALL and put/short → a PUT. A contract qualifies
    when: its type matches the side, ``|delta|`` is within ``[delta_low,
    delta_high]``, its DTE ≥ ``min_dte``, and it clears the liquidity floor.
    Among qualifiers, the one whose ``|delta|`` is CLOSEST to the band centre is
    chosen. Returns None if nothing qualifies (the caller widens the band, then
    falls back to shares). Pure — no I/O, ``ref_date`` is injected.
    """
    want_type = OPTION_TYPE_CALL if side in ("call", "long") else OPTION_TYPE_PUT
    centre = (delta_low + delta_high) / 2.0

    qualifying: list[tuple[float, OptionContract]] = []
    for c in contracts:
        if c.option_type != want_type:
            continue
        d = c.delta
        if d is None or not (delta_low <= abs(d) <= delta_high):
            continue
        dte = _dte(c.expiry, ref_date)
        if dte is None or dte < min_dte:
            continue
        if not _passes_liquidity(c, min_open_interest, max_spread_pct):
            continue
        qualifying.append((abs(d), c))

    if not qualifying:
        return None
    return min(qualifying, key=lambda t: abs(t[0] - centre))[1]


# ── cost model (every calc routes through OPTION_MULTIPLIER) ──────────────────


def _premium(contract: OptionContract) -> float | None:
    """The per-share premium to size against: prefer mid, then ask, then bid."""
    for price in (contract.mid, contract.ask, contract.bid):
        if price is not None and price > 0.0:
            return price
    return None


def cost_for_contract(contract: OptionContract, qty: float) -> float | None:
    """Total premium outlay = ``premium × OPTION_MULTIPLIER × qty``.

    EVERY options cost calculation routes through this so the 100-share
    multiplier is applied in exactly one place — the single most common options
    bug. Returns None when the contract has no usable premium.
    """
    premium = _premium(contract)
    if premium is None:
        return None
    return premium * config.OPTION_MULTIPLIER * qty


# ── execution hierarchy: full option → undersized option → shares ────────────


@dataclass(frozen=True)
class ExecutionDecision:
    """The chosen execution vehicle for one allocated swing candidate.

    ``vehicle`` is one of ``option_full`` / ``option_undersized`` / ``shares`` /
    ``none``. ``side`` is the ORDER side (options are always bought — buy call /
    buy put; the share fallback buys for a bullish signal, sells for a bearish
    one). ``qty`` is whole contracts for options (never fractional) or fractional
    shares for the fallback. ``dollar_risk`` is the premium at risk (== est_cost
    for a long option; the share cost for the fallback)."""

    vehicle: str
    side: str
    symbol: str
    qty: float
    est_cost: float
    dollar_risk: float
    contract: OptionContract | None
    reason: str


def _bullish(direction: str) -> bool:
    return direction in ("call", "long")


def _size_option(
    contract: OptionContract | None,
    allocated_capital: float,
    vehicle: str,
    min_dollar_risk: float,
) -> ExecutionDecision | None:
    """Size a selected contract to WHOLE contracts within the allocated capital.

    Returns None (so the caller tries the next tier) when there is no contract,
    no usable premium, not even one whole contract fits the capital, or the
    resulting outlay is below the dollar-risk floor. NEVER a fractional contract.
    """
    if contract is None:
        return None
    premium = _premium(contract)
    if premium is None:
        return None
    per_contract = premium * config.OPTION_MULTIPLIER
    qty = int(allocated_capital // per_contract)   # floor to whole contracts
    if qty < 1:
        return None
    est_cost = per_contract * qty
    if est_cost < min_dollar_risk:
        return None
    return ExecutionDecision(
        vehicle=vehicle, side="buy", symbol=contract.symbol, qty=float(qty),
        est_cost=est_cost, dollar_risk=est_cost, contract=contract, reason="ok",
    )


def _shares_decision(
    direction: str, underlying: str, underlying_price: float | None,
    allocated_capital: float,
) -> ExecutionDecision:
    """Fractional-share fallback via the existing Phase 11 equity order path."""
    side = "buy" if _bullish(direction) else "sell"
    if underlying_price is None or underlying_price <= 0.0:
        return ExecutionDecision(
            vehicle=VEHICLE_NONE, side=side, symbol=underlying, qty=0.0,
            est_cost=0.0, dollar_risk=0.0, contract=None,
            reason="shares fallback unsizeable: no underlying price",
        )
    qty = allocated_capital / underlying_price      # fractional shares are fine
    est_cost = qty * underlying_price
    return ExecutionDecision(
        vehicle=VEHICLE_SHARES, side=side, symbol=underlying, qty=qty,
        est_cost=est_cost, dollar_risk=est_cost, contract=None, reason="ok",
    )


def choose_execution(
    direction: str,
    allocated_capital: float,
    underlying: str,
    underlying_price: float | None,
    contracts: Sequence[OptionContract],
    *,
    ref_date: date,
    options_available: bool = True,
    min_dollar_risk: float = config.MIN_DOLLAR_RISK,
) -> ExecutionDecision:
    """Pick the execution vehicle for an allocated swing candidate.

    Hierarchy: (1) a FULL-band option sized to whole contracts that fits the
    capital; else (2) an UNDERSIZED option (delta band widened to the floor);
    else (3) fractional shares. Options are skipped entirely (straight to shares)
    when they are unavailable (not enabled, or an empty chain). Every fallback
    step is logged with its reason.
    """
    if options_available and contracts:
        full = select_contract(direction, contracts, ref_date=ref_date)
        decision = _size_option(
            full, allocated_capital, VEHICLE_OPTION_FULL, min_dollar_risk,
        )
        if decision is not None:
            return decision
        print(
            f"  options: no full-band fit for {underlying} -> trying undersized",
            file=sys.stderr,
        )
        under = select_contract(
            direction, contracts, ref_date=ref_date,
            delta_low=config.UNDERSIZED_DELTA_FLOOR,
        )
        decision = _size_option(
            under, allocated_capital, VEHICLE_OPTION_UNDERSIZED, min_dollar_risk,
        )
        if decision is not None:
            return decision
        print(
            f"  options: no undersized fit for {underlying} -> shares fallback",
            file=sys.stderr,
        )
    else:
        print(
            f"  options: unavailable for {underlying} -> shares fallback",
            file=sys.stderr,
        )
    return _shares_decision(
        direction, underlying, underlying_price, allocated_capital,
    )


# ── order submission (reuses the Phase 11 order path; NO bracket attributes) ──


def _limit_price_for(decision: ExecutionDecision) -> float | None:
    """The LIMIT buy/sell price: an option's ask (else mid), or the share price."""
    if decision.contract is not None:
        contract = decision.contract
        if contract.ask is not None and contract.ask > 0.0:
            return contract.ask
        return contract.mid
    if decision.qty:
        return decision.est_cost / decision.qty
    return None


def submit_execution_order(
    broker: Broker, decision: ExecutionDecision, *, time_in_force: str = TIF_DAY,
) -> OrderResult:
    """Submit the chosen vehicle via the EXISTING order path — a single-leg LIMIT
    order with NO bracket/OTO attributes (Alpaca has none for options). Options
    and the share fallback both go through ``Broker.submit_order``; the Phase 11
    structured-rejection handling is reused verbatim."""
    limit_price = _limit_price_for(decision)
    order = broker.submit_order(
        decision.symbol, decision.qty, decision.side,
        order_type=ORDER_TYPE_LIMIT, limit_price=limit_price,
        time_in_force=time_in_force,
    )
    # Phase 15: feed the consecutive broker-error detector (resets on success).
    risk_of_ruin.record_broker_result(order.ok)
    return order


def record_option_position(
    decision: ExecutionDecision,
    order: OrderResult,
    *,
    opened_at: datetime,
    signal_id: int | None = None,
    tp: float | None = None,
    sl: float | None = None,
    deadline: datetime | None = None,
) -> int | None:
    """Persist an option position from a filled/accepted decision. Returns the
    row id, or None for a share fallback (no contract to record)."""
    contract = decision.contract
    if contract is None:
        return None
    pos = OptionPosition(
        symbol=contract.symbol, underlying=contract.underlying,
        option_type=contract.option_type, strike=contract.strike,
        expiry=contract.expiry, contracts=decision.qty, opened_at=opened_at,
        multiplier=config.OPTION_MULTIPLIER, signal_id=signal_id,
        order_id=order.order_id, premium_entry=_premium(contract),
        delta_entry=contract.delta, theta=contract.theta, vega=contract.vega,
        gamma=contract.gamma, tp=tp, sl=sl, deadline=deadline, outcome="open",
        vehicle=decision.vehicle,
    )
    return db.insert_option_position(pos)


def record_shares_position(
    decision: ExecutionDecision,
    *,
    opened_at: datetime,
    tp: float | None = None,
    sl: float | None = None,
    deadline: datetime | None = None,
) -> int | None:
    """Persist a SHARES-FALLBACK position into the long-term lifecycle book
    (Phase 19). Returns the row id, or None for a non-shares decision.

    The third rung of the execution hierarchy submits a real order but had no
    lifecycle tracking (Phase 18's reconciliation fix surfaced it as an
    untracked ``broker_only`` position). It is tagged into
    ``long_term_positions`` with ``source='swing_fallback'`` and its ORIGINAL
    swing ``tp``/``sl``/``deadline`` — the position carries SWING intent, and
    the Phase 14 watcher applies THESE levels to it, never the long-term
    trend/drawdown rules. ``direction='short'`` records a put-signal fallback
    (a SELL entry, whose close must BUY).
    """
    if decision.vehicle != VEHICLE_SHARES or decision.qty <= 0.0:
        return None
    return db.insert_long_term_position(LongTermPosition(
        ticker=decision.symbol,
        asset_class="crypto" if decision.symbol.endswith("-USD") else "stock",
        entry_price=decision.est_cost / decision.qty,
        entry_date=opened_at,
        qty=decision.qty,
        status="open",
        source="swing_fallback",
        direction="short" if decision.side == "sell" else "long",
        tp=tp,
        sl=sl,
        deadline=deadline,
    ))


def _submit_execution_with_intent(
    broker: Broker,
    decision: ExecutionDecision,
    *,
    submitted_at: datetime,
    signal_id: int | None,
    tp: float | None,
    sl: float | None,
    deadline: datetime | None,
    time_in_force: str,
) -> OrderResult:
    """Prepare the already-selected intent, submit once, and bind its reply."""
    contract = decision.contract
    if contract is not None:
        ticker = contract.underlying
        asset_class = "stock"
        target_position_kind = "option"
        intent_payload: dict[str, object] = {
            "intent_kind": "option",
            "option_type": contract.option_type,
            "strike": contract.strike,
            "expiry": contract.expiry,
            "multiplier": config.OPTION_MULTIPLIER,
            "delta_entry": contract.delta,
            "theta": contract.theta,
            "vega": contract.vega,
            "gamma": contract.gamma,
            "tp": tp,
            "sl": sl,
            "deadline": deadline.isoformat() if deadline is not None else None,
        }
    elif decision.vehicle == VEHICLE_SHARES:
        ticker = decision.symbol
        asset_class = "crypto" if decision.symbol.endswith("-USD") else "stock"
        target_position_kind = "long_term"
        intent_payload = {
            "intent_kind": "shares_fallback",
            "source": "swing_fallback",
            "direction": "short" if decision.side == "sell" else "long",
            "tp": tp,
            "sl": sl,
            "deadline": deadline.isoformat() if deadline is not None else None,
        }
    else:
        raise order_lifecycle.OrderIntentCaptureError(
            f"accepted order has unsupported execution vehicle {decision.vehicle!r}; "
            "materialization intent was not durably captured"
        )

    return order_lifecycle.submit_prepared_order(
        broker,
        ticker=ticker,
        broker_symbol=decision.symbol,
        asset_class=asset_class,
        vehicle=decision.vehicle,
        target_position_kind=target_position_kind,
        side=decision.side,
        requested_qty=decision.qty,
        requested_limit_price=_limit_price_for(decision),
        submitted_at=submitted_at,
        signal_id=signal_id,
        intent_payload=intent_payload,
        order_type=ORDER_TYPE_LIMIT,
        time_in_force=time_in_force,
    )


def execute_decision(
    broker: Broker,
    decision: ExecutionDecision,
    *,
    opened_at: datetime,
    signal_id: int | None = None,
    tp: float | None = None,
    sl: float | None = None,
    deadline: datetime | None = None,
    time_in_force: str = TIF_DAY,
) -> tuple[OrderResult, int | None]:
    """Submit the decision and, on a broker-accepted order, persist the
    position: an OPTION records into ``option_positions``; a SHARES FALLBACK
    records into the long-term lifecycle book with its original swing exit
    data (Phase 19 — previously untracked). Returns ``(order_result,
    position_row_id | None)`` — the id belongs to whichever table matches the
    vehicle. A rejected/errored order records nothing anywhere."""
    order = _submit_execution_with_intent(
        broker,
        decision,
        submitted_at=opened_at,
        signal_id=signal_id,
        tp=tp,
        sl=sl,
        deadline=deadline,
        time_in_force=time_in_force,
    )
    risk_of_ruin.record_broker_result(order.ok)
    position_id: int | None = None
    if order.ok:
        if decision.contract is not None:
            position_id = record_option_position(
                decision, order, opened_at=opened_at, signal_id=signal_id,
                tp=tp, sl=sl, deadline=deadline,
            )
        elif decision.vehicle == VEHICLE_SHARES:
            position_id = record_shares_position(
                decision, opened_at=opened_at, tp=tp, sl=sl, deadline=deadline,
            )
    return order, position_id


# ── manual exit management (Alpaca has no bracket/OTO for options) ────────────
#
# Alpaca supports NO bracket/OTO orders for options, so the bot must manage exits
# itself: on each scan cycle it checks every open option position's UNDERLYING
# price against the signal's TP/SL levels (derived exactly as the equity
# resolver's are) and the hold-window deadline, and submits a closing SELL when
# any exit condition trips. This is the options analogue of the equity resolver.

EXIT_TAKE_PROFIT = "take_profit"
EXIT_STOP_LOSS = "stop_loss"
EXIT_HOLD_DEADLINE = "hold_deadline"
EXIT_HOLDING = "holding"


@dataclass(frozen=True)
class ExitDecision:
    """Whether to close an option position now, and why."""

    action: str      # 'close' | 'hold'
    reason: str      # take_profit | stop_loss | hold_deadline | holding


@dataclass(frozen=True)
class ExitAction:
    """What the watcher did for one position on this cycle."""

    position: OptionPosition
    action: str                      # 'close' | 'hold' | 'error'
    reason: str
    order: OrderResult | None = None
    pnl_dollars: float | None = None


def evaluate_option_exit(
    option_type: str,
    underlying_price: float | None,
    tp: float | None,
    sl: float | None,
    now: datetime,
    deadline: datetime | None,
) -> ExitDecision:
    """Pure exit decision from the UNDERLYING price vs the TP/SL levels + deadline.

    A ``call`` is bullish (take-profit above, stop below); a ``put`` is bearish
    (take-profit below, stop above) — the same orientation the equity resolver
    uses. The stop is checked FIRST (conservative). With no TP/SL hit, the
    hold-window deadline forces a close. Otherwise: hold. Never raises.
    """
    bullish = option_type == OPTION_TYPE_CALL
    if underlying_price is not None and tp is not None and sl is not None:
        sl_hit = underlying_price <= sl if bullish else underlying_price >= sl
        tp_hit = underlying_price >= tp if bullish else underlying_price <= tp
        if sl_hit:
            return ExitDecision("close", EXIT_STOP_LOSS)
        if tp_hit:
            return ExitDecision("close", EXIT_TAKE_PROFIT)
    if deadline is not None and now >= deadline:
        return ExitDecision("close", EXIT_HOLD_DEADLINE)
    return ExitDecision("hold", EXIT_HOLDING)


def _option_pnl_dollars(
    position: OptionPosition, exit_price: float | None,
) -> float | None:
    """Contract-aware realized PnL: ``(exit − entry) × multiplier × contracts``."""
    if exit_price is None or position.premium_entry is None:
        return None
    return (
        (exit_price - position.premium_entry)
        * position.multiplier * position.contracts
    )


def close_option_position(
    broker: Broker,
    position: OptionPosition,
    *,
    exit_price: float | None,
    now: datetime,
    outcome: str,
) -> tuple[OrderResult, float | None]:
    """THE option-closing path — a single-leg SELL LIMIT via the Phase 11 order
    path (no bracket attributes exist for options). Used by the exit watcher AND
    the Phase 15 emergency shutdown, so there is exactly one closer.

    The position row is marked closed ONLY when the broker accepted the order —
    a rejected close (e.g. market closed) leaves the row open so the next cycle
    retries; nothing is ever recorded closed that was not submitted.
    Returns ``(order, realized_pnl_dollars)``.
    """
    limit_price = exit_price if exit_price is not None else position.premium_entry
    order = broker.submit_order(
        position.symbol, position.contracts, "sell",
        order_type=ORDER_TYPE_LIMIT, limit_price=limit_price,
        time_in_force=TIF_DAY,
    )
    risk_of_ruin.record_broker_result(order.ok)   # Phase 15 detector
    pnl = _option_pnl_dollars(position, exit_price)
    if order.ok and position.id is not None:
        db.update_option_position(
            position.id, order_id=order.order_id, outcome=outcome,
            closed_at=now, exit_price=exit_price, pnl_dollars=pnl,
        )
    return order, pnl


def watch_open_option_positions(
    broker: Broker,
    *,
    underlying_price_fetch: Callable[[str], float | None],
    option_price_fetch: Callable[[str], float | None],
    now: datetime,
    positions: Sequence[OptionPosition] | None = None,
) -> list[ExitAction]:
    """Check each open option position and close the ones that hit TP/SL/deadline.

    For a closing position it submits a single-leg SELL LIMIT (marketable at the
    current option price; falling back to the entry premium when the current
    price is unavailable — the documented deadline fallback) with NO bracket
    attributes, then records the outcome + contract-aware PnL. FAIL-SOFT: an
    error on one position is logged and NEVER blocks the others.
    """
    book = positions if positions is not None else db.get_open_option_positions()
    actions: list[ExitAction] = []
    for pos in book:
        try:
            underlying_price = underlying_price_fetch(pos.underlying)
            decision = evaluate_option_exit(
                pos.option_type, underlying_price, pos.tp, pos.sl, now,
                pos.deadline,
            )
            if decision.action == "hold":
                actions.append(ExitAction(pos, "hold", decision.reason))
                continue

            exit_price = option_price_fetch(pos.symbol)
            outcome = (
                "win" if decision.reason == EXIT_TAKE_PROFIT
                else "loss" if decision.reason == EXIT_STOP_LOSS
                else "expired"
            )
            order, pnl = close_option_position(
                broker, pos, exit_price=exit_price, now=now, outcome=outcome,
            )
            if not order.ok:
                # Close not accepted (e.g. market closed) — row stays open so the
                # next cycle retries; never recorded closed without a real order.
                actions.append(ExitAction(
                    pos, "error", f"close order failed: {order.reason}", order=order,
                ))
                continue
            actions.append(
                ExitAction(pos, "close", decision.reason, order=order, pnl_dollars=pnl)
            )
        except Exception as exc:  # noqa: BLE001 - one position must never block others
            print(
                f"  options watcher error for {pos.symbol}: {exc}",
                file=sys.stderr,
            )
            actions.append(ExitAction(pos, "error", str(exc)))
    return actions
