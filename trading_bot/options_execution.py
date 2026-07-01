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
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from trading_bot import config
from trading_bot.broker.options import (
    OPTION_TYPE_CALL,
    OPTION_TYPE_PUT,
    OptionContract,
)

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
