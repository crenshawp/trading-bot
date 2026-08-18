"""Broker-backed exposure snapshot for new-entry allocation.

The allocator is deliberately pure, so this module translates the live broker
position book plus durable working entry orders into its ``ExistingExposure``
input.  If the broker book is unavailable or a holding cannot be valued, the
snapshot fails closed: the caller skips new entries but the scanner and exit
watchers keep running.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

from trading_bot import allocation, config
from trading_bot.broker.base import AccountInfo, Position, PositionsResult
from trading_bot.models import LongTermPosition, OptionPosition, PendingOrder


@dataclass(frozen=True)
class ExposureResult:
    """Result of translating broker truth into allocator exposure."""

    ok: bool
    exposure: allocation.ExistingExposure | None = None
    reason: str = ""


def _position_value(position: Position) -> float | None:
    """Absolute current value, falling back to broker entry cost if needed."""
    if position.market_value is not None:
        return abs(float(position.market_value))
    if position.avg_entry_price is not None:
        return abs(float(position.avg_entry_price) * float(position.qty))
    return None


def _pool_for_long_term(position: LongTermPosition) -> str:
    if position.source == "swing_fallback":
        return config.POOL_SWING
    if position.asset_class == "crypto":
        return config.POOL_CRYPTO
    return config.POOL_LONG_TERM


def _pending_source(order: PendingOrder) -> str | None:
    """The intent payload's ``source``, or ``None`` if it cannot be read.

    Fail-soft by contract: an unreadable payload must not break the exposure
    snapshot, so the caller falls back to the structural routing below.
    """
    try:
        payload = json.loads(order.intent_payload_json)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    source = payload.get("source")
    return source if isinstance(source, str) else None


def _pending_pool(order: PendingOrder) -> str:
    if order.target_position_kind == "option":
        return config.POOL_SWING
    # A swing shares-fallback entry (the options hierarchy's third tier) carries
    # target_position_kind="long_term" because it materializes into the long-term
    # book — but it is SWING capital, and _pool_for_long_term already routes the
    # filled position to POOL_SWING on exactly this predicate. Routing on
    # target_position_kind alone made the same dollars count against LONG_TERM
    # while working and SWING once filled, so the pool they consumed flipped at
    # fill time: the swing pool under-reported its deployment (letting the tier
    # deploy-cap over-commit) while genuine long-term candidates were skipped
    # capital-exhausted against exposure that was never theirs.
    if _pending_source(order) == "swing_fallback":
        return config.POOL_SWING
    if order.asset_class == "crypto":
        return config.POOL_CRYPTO
    return config.POOL_LONG_TERM


def build_existing_exposure(
    account: AccountInfo,
    broker_positions: PositionsResult,
    option_positions: Sequence[OptionPosition],
    long_term_positions: Sequence[LongTermPosition],
    pending_orders: Sequence[PendingOrder],
) -> ExposureResult:
    """Build exposure from broker positions and unfilled entry-order remainder.

    Internal books identify the pool and, for option contracts, the underlying
    ticker. Unknown broker holdings are conservatively assigned to SWING using
    their broker symbol, so they still consume gross and concentration room.
    """
    bases = allocation.available_capital(account)
    if bases is None:
        return ExposureResult(False, reason=account.reason or "account unavailable")
    if not broker_positions.ok:
        return ExposureResult(
            False, reason=broker_positions.reason or "broker positions unavailable",
        )

    portfolio_capital, _cash = bases
    option_by_symbol = {p.symbol: p for p in option_positions}
    long_term_by_symbol = {p.ticker: p for p in long_term_positions}
    pending_by_symbol = {
        p.broker_symbol: p
        for p in pending_orders
        if p.order_role == "entry" and p.terminal_at is None
    }
    pool_values: dict[str, float] = {}
    ticker_values: dict[str, float] = {}
    gross_value = 0.0

    def add(pool: str, ticker: str, value: float) -> None:
        nonlocal gross_value
        gross_value += value
        pool_values[pool] = pool_values.get(pool, 0.0) + value
        ticker_values[ticker] = ticker_values.get(ticker, 0.0) + value

    for position in broker_positions.positions:
        value = _position_value(position)
        if value is None:
            return ExposureResult(
                False, reason=f"cannot value broker position {position.symbol}",
            )
        option = option_by_symbol.get(position.symbol)
        if option is not None:
            add(config.POOL_SWING, option.underlying, value)
            continue
        long_term = long_term_by_symbol.get(position.symbol)
        if long_term is not None:
            add(_pool_for_long_term(long_term), long_term.ticker, value)
            continue
        pending = pending_by_symbol.get(position.symbol)
        if pending is not None:
            add(_pending_pool(pending), pending.ticker, value)
            continue
        add(config.POOL_SWING, position.symbol, value)

    for order in pending_orders:
        if order.order_role != "entry" or order.terminal_at is not None:
            continue
        remaining = max(0.0, order.requested_qty - order.filled_qty)
        if remaining <= 0.0:
            continue
        if order.requested_limit_price is None or order.requested_limit_price <= 0.0:
            return ExposureResult(
                False, reason=f"cannot value working order {order.client_order_id}",
            )
        multiplier = 100.0 if order.target_position_kind == "option" else 1.0
        add(
            _pending_pool(order),
            order.ticker,
            remaining * order.requested_limit_price * multiplier,
        )

    return ExposureResult(
        True,
        allocation.ExistingExposure(
            portfolio_capital=portfolio_capital,
            gross_value=gross_value,
            pool_values=pool_values,
            ticker_values=ticker_values,
        ),
    )
