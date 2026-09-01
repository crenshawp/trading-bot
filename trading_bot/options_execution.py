"""Options execution — selection, cost model, hierarchy, exit watcher (Phase 13).

Consumes a Phase 12 allocated SWING candidate (side + allocated capital +
underlying) and turns it into an actual single-leg option order (buy call / buy
put), with a share fallback. Still PAPER ONLY.

Execution hierarchy per allocated swing signal (Section 4):
  1. a FULL-sized option at target delta (0.65-0.75), or
  2. an UNDERSIZED option (delta band widened down to 0.50, still liquidity-
     gated, still a WHOLE contract) that fits the allocated capital, or
  3. fall back to SHARES via the existing Phase 11 equity order path. Long
     fallbacks may be fractional; Alpaca short fallbacks must be whole shares.

Greeks (theta/vega/gamma) are stored for audit but are NOT hard gates this phase.
No spreads / multi-leg. No naive market orders (LIMIT default). Alpaca has no
bracket/OTO for options, so exits are managed explicitly (Section 6).
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

from trading_bot import config, db, order_lifecycle, risk_of_ruin
from trading_bot.broker.base import ORDER_TYPE_LIMIT, TIF_DAY, Broker, OrderResult
from trading_bot.broker.options import (
    OPTION_TYPE_CALL,
    OPTION_TYPE_PUT,
    OptionContract,
)
from trading_bot.models import OptionPosition

VEHICLE_OPTION_FULL = "option_full"
VEHICLE_OPTION_UNDERSIZED = "option_undersized"
VEHICLE_SHARES = "shares"
VEHICLE_NONE = "none"

_ET = ZoneInfo("America/New_York")


def is_option_market_open(now: datetime) -> bool:
    """Whether regular US options trading is open at ``now``.

    Alpaca does not support extended-hours option execution.  The scanner uses
    this guard before requesting an exit quote or submitting a DAY order, so a
    weekend/overnight watcher cannot queue a stale limit for the next session.

    That promise needs the session's REAL closing bell, which is 13:00 ET on the
    three NYSE half-days rather than 16:00.  Between those two times the guard
    used to answer "open", and the watcher submitted a DAY limit into a shut
    market — an order the broker cancels at end of day, leaving a position the
    watcher had already recorded as being exited.
    """
    from trading_bot.outcomes import is_stock_session, stock_session_close

    local = now.astimezone(_ET)
    return (
        is_stock_session(local.date())
        and time(9, 30) <= local.time().replace(tzinfo=None)
        < stock_session_close(local.date())
    )


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
    max_dte: int = config.MAX_DTE,
    target_dte: int = config.TARGET_DTE,
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
        if dte is None or not (min_dte <= dte <= max_dte):
            continue
        if not _passes_liquidity(c, min_open_interest, max_spread_pct):
            continue
        qualifying.append((abs(d), c))

    if not qualifying:
        return None
    return min(
        qualifying,
        key=lambda item: (
            abs(item[0] - centre),
            abs((_dte(item[1].expiry, ref_date) or target_dte) - target_dte),
        ),
    )[1]


# ── cost model (every calc routes through OPTION_MULTIPLIER) ──────────────────


def _premium(contract: OptionContract) -> float | None:
    """Executable buy premium used for sizing: prefer ask, then mid, then bid.

    Sizing at midpoint while submitting at the ask can exceed the risk budget
    before the order even leaves the process. The ask is therefore the first
    usable price for a long-option entry.
    """
    for price in (contract.ask, contract.mid, contract.bid):
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
    return round(premium * config.OPTION_MULTIPLIER * qty, 2)


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
    per_contract = round(premium * config.OPTION_MULTIPLIER, 2)
    qty = int(allocated_capital // per_contract)   # floor to whole contracts
    if qty < 1:
        return None
    est_cost = round(per_contract * qty, 2)
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
    """Share fallback via the existing Phase 11 equity order path.

    Alpaca accepts fractional-share BUY orders but rejects fractional short
    sales. Bearish fallbacks therefore floor to a whole-share quantity while
    bullish fallbacks retain the allocator's exact fractional sizing. Refuse a
    short fallback when its capital cannot cover one share instead of sending a
    broker request that is guaranteed to fail.
    """
    side = "buy" if _bullish(direction) else "sell"
    if underlying_price is None or underlying_price <= 0.0:
        return ExecutionDecision(
            vehicle=VEHICLE_NONE, side=side, symbol=underlying, qty=0.0,
            est_cost=0.0, dollar_risk=0.0, contract=None,
            reason="shares fallback unsizeable: no underlying price",
        )
    raw_qty = allocated_capital / underlying_price
    qty = raw_qty if side == "buy" else float(int(raw_qty))
    if qty <= 0.0:
        return ExecutionDecision(
            vehicle=VEHICLE_NONE, side=side, symbol=underlying, qty=0.0,
            est_cost=0.0, dollar_risk=0.0, contract=None,
            reason="shares fallback unsizeable: short requires one whole share",
        )
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
    shares_capital: float | None = None,
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
        direction,
        underlying,
        underlying_price,
        allocated_capital if shares_capital is None else shares_capital,
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
    risk_of_ruin.record_broker_result(order.ok, order.status)
    return order


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
    """Submit a durably prepared decision and persist only broker fill truth.

    Accepted-unfilled orders remain solely in ``pending_orders`` for scanner
    refresh. A real partial/full fill records an OPTION position or SHARES
    FALLBACK in the long-term lifecycle book; requested quantity and quote
    values are never inserted as a holding.

    The legacy return contract remains ``(order_result, position_row_id |
    None)``; the position id is present only when a usable fill materialized.
    """
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
    risk_of_ruin.record_broker_result(order.ok, order.status)
    materialized = order_lifecycle.materialize_submitted_order_fill(
        order,
        observed_at=opened_at,
    )
    position_id = materialized.position_id if materialized is not None else None
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
# The underlying could not be quoted, so TP/SL were never evaluated. A HOLD,
# because nothing was closed -- but never the same thing as "nothing triggered".
EXIT_NO_PRICE_DATA = "no_price_data"


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
    # Kept as an inline conjunction rather than a `has_levels` flag: mypy does
    # not narrow tp/sl through a boolean variable, and the audit gate is strict.
    if underlying_price is not None and tp is not None and sl is not None:
        sl_hit = underlying_price <= sl if bullish else underlying_price >= sl
        tp_hit = underlying_price >= tp if bullish else underlying_price <= tp
        if sl_hit:
            return ExitDecision("close", EXIT_STOP_LOSS)
        if tp_hit:
            return ExitDecision("close", EXIT_TAKE_PROFIT)
    # The deadline is checked even with no price: a hold window that has run out
    # forces a close regardless of whether the underlying could be quoted.
    if deadline is not None and now >= deadline:
        return ExitDecision("close", EXIT_HOLD_DEADLINE)
    if underlying_price is None and tp is not None and sl is not None:
        # DISTINCT from EXIT_HOLDING. A missing underlying quote skipped the
        # whole TP/SL block above and then returned a plain "holding" -- byte
        # identical to "I checked the price and nothing triggered". The stop
        # loss was never evaluated, and nothing anywhere said so: the cycle
        # printed "N open position(s), 0 closed, 0 error(s)".
        #
        # The fetcher (scanner._build_latest_price_fetch) returns prices.get,
        # i.e. None for any symbol absent from the batch or whose IEX bid/ask
        # came back 0 -- partial free-tier coverage makes that ordinary, not
        # exceptional. long_term.watch_long_term_positions already distinguishes
        # this case ("no price data"); the option watcher did not.
        return ExitDecision("hold", EXIT_NO_PRICE_DATA)
    return ExitDecision("hold", EXIT_HOLDING)


def close_option_position(
    broker: Broker,
    position: OptionPosition,
    *,
    exit_price: float | None,
    now: datetime,
    reason: str,
) -> tuple[OrderResult | None, float | None]:
    """Restart-safe emergency option close from durable execution truth.

    A still-growing entry is canceled and materialized first. Acceptance and
    partial fills leave the typed position open; only aggregate terminal actual
    fills produce P/L and a close summary.
    """
    if position.id is None:
        raise ValueError("lifecycle exit refused: option position has no durable id")
    frozen = order_lifecycle.freeze_position_entry_intent(
        broker,
        position_kind="option",
        position_id=position.id,
        observed_at=now,
    )
    if not frozen.frozen:
        raise ValueError(
            "lifecycle exit refused: entry quantity is not frozen"
            + (f" ({frozen.reason})" if frozen.reason else "")
        )
    current = db.get_option_position(position.id)
    if current is None or current.outcome not in (None, "open"):
        raise ValueError("lifecycle exit refused: option position is no longer open")
    order, fill = _submit_option_watcher_exit(
        broker,
        current,
        exit_price=exit_price,
        now=now,
        reason=reason,
    )
    return order, fill.pnl_dollars


def _submit_option_watcher_exit(
    broker: Broker,
    position: OptionPosition,
    *,
    exit_price: float | None,
    now: datetime,
    reason: str,
) -> tuple[OrderResult | None, order_lifecycle.ExitFillMaterializationResult]:
    """Submit remaining broker-linked quantity and apply immediate actual fills.

    The legacy direct closer above remains the Phase 15 emergency path until
    17a5. Watchers use this lifecycle path so acceptance is never a close.
    """
    if position.id is None:
        raise ValueError("lifecycle exit refused: option position has no durable id")
    state = order_lifecycle.materialize_position_exit_fills("option", position.id)
    if not state.integrity_ok or state.remaining_qty is None:
        raise ValueError(
            "lifecycle exit refused: option fill integrity is unknown"
            + (f" ({state.reason})" if state.reason else "")
        )
    attempts = db.get_pending_exit_orders_for_position("option", position.id)
    if any(attempt.terminal_at is None for attempt in attempts):
        return None, state
    if state.remaining_qty <= 0:
        return None, state

    if exit_price is None:
        # NEVER fall back to the entry premium. A close priced at entry is
        # above the market for exactly the positions an exit exists to close
        # (a stop-loss fires because the position is DOWN), so the order rests
        # unfillable while the book records a close as submitted. Refusing is
        # honest: the position stays visibly open and the next pass retries.
        raise ValueError(
            "lifecycle exit refused: no current option price is available "
            "(refusing to price the close at the entry premium)"
        )
    limit_price = exit_price
    order = order_lifecycle.submit_position_exit(
        broker,
        position_kind="option",
        position_id=position.id,
        requested_qty=state.remaining_qty,
        requested_limit_price=limit_price,
        exit_reason=reason,
        submitted_at=now,
        broker_result_observer=risk_of_ruin.record_broker_result,
    )
    if not order.ok:
        return order, state
    return order, order_lifecycle.materialize_position_exit_fills(
        "option", position.id
    )


def watch_open_option_positions(
    broker: Broker,
    *,
    underlying_price_fetch: Callable[[str], float | None],
    option_price_fetch: Callable[[str], float | None],
    now: datetime,
    positions: Sequence[OptionPosition] | None = None,
) -> list[ExitAction]:
    """Check each open option position and close the ones that hit TP/SL/deadline.

    For a triggered position, durable exit intent is submitted for only the
    actual remaining quantity. Accepted/unfilled and partial attempts remain
    open; a terminal full fill is summarized from actual broker price and time.
    FAIL-SOFT: an error on one position is logged and NEVER blocks the others.
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
                if decision.reason == EXIT_NO_PRICE_DATA:
                    # Say it out loud. An unevaluated stop is the one "hold"
                    # the operator needs to see, and the cycle summary counts
                    # it among the quiet ones.
                    print(
                        f"  options watcher: no underlying quote for "
                        f"{pos.underlying} ({pos.symbol}) - TP/SL NOT "
                        f"evaluated this cycle",
                        file=sys.stderr,
                    )
                actions.append(ExitAction(pos, "hold", decision.reason))
                continue

            # A triggered exit must terminalize a still-working entry FIRST,
            # exactly as the emergency closer above already does. The exit
            # submitter opens with materialize_position_exit_fills, whose
            # _load_entry_truth refuses outright while the linked entry order is
            # non-terminal ("linked entry order is not terminal") — so without
            # this, a partially-filled entry (a real, ordinary state: the typed
            # position exists and is tradeable the moment the first contract
            # fills) has NO reachable TP, SL or hold-deadline exit. The watcher
            # just re-errors every hour while the stop it was built to honour
            # goes unfilled. freeze_position_entry_intent is idempotent — an
            # already-terminal entry short-circuits to "already_terminal"
            # without cancelling anything — so this is a no-op on the ordinary
            # fully-filled path.
            if pos.id is not None:
                frozen = order_lifecycle.freeze_position_entry_intent(
                    broker,
                    position_kind="option",
                    position_id=pos.id,
                    observed_at=now,
                )
                if not frozen.frozen:
                    detail = frozen.reason or "entry quantity is not frozen"
                    print(
                        f"  options watcher lifecycle exit refused for "
                        f"{pos.symbol}: entry not frozen ({detail})",
                        file=sys.stderr,
                    )
                    actions.append(ExitAction(
                        pos, "error", f"entry quantity is not frozen ({detail})",
                    ))
                    continue

            exit_price = option_price_fetch(pos.symbol)
            order, fill = _submit_option_watcher_exit(
                broker, pos, exit_price=exit_price, now=now,
                reason=decision.reason,
            )
            if not fill.integrity_ok:
                detail = fill.reason or "exit fill integrity is unknown"
                print(
                    f"  options watcher lifecycle refusal for {pos.symbol}: "
                    f"{detail}",
                    file=sys.stderr,
                )
                actions.append(ExitAction(pos, "error", detail, order=order))
                continue
            if order is None:
                actions.append(ExitAction(pos, "hold", decision.reason))
                continue
            if not order.ok:
                # The row stays open; durable lifecycle state decides whether a
                # later watcher may retry or must await lookup-only recovery.
                actions.append(ExitAction(
                    pos, "error", f"close order failed: {order.reason}", order=order,
                ))
                continue
            actions.append(ExitAction(
                pos, "close", decision.reason, order=order,
                pnl_dollars=fill.pnl_dollars,
            ))
        except Exception as exc:  # noqa: BLE001 - one position must never block others
            print(
                f"  options watcher error for {pos.symbol}: {exc}",
                file=sys.stderr,
            )
            actions.append(ExitAction(pos, "error", str(exc)))
    return actions
