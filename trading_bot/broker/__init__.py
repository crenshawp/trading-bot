"""Broker execution interface (Phase 11).

The abstract :class:`Broker` and its neutral result types are the only shapes
higher layers see. Concrete adapters live alongside this module.
"""

from __future__ import annotations

from trading_bot.broker.base import (
    NEUTRAL_STATUSES,
    ORDER_TYPE_LIMIT,
    ORDER_TYPE_MARKET,
    SIDE_BUY,
    SIDE_SELL,
    STATUS_CANCELED,
    STATUS_ERROR,
    STATUS_FILLED,
    STATUS_NEW,
    STATUS_PARTIALLY_FILLED,
    STATUS_REJECTED,
    STATUS_UNKNOWN,
    TERMINAL_STATUSES,
    TIF_DAY,
    TIF_GTC,
    VALID_ORDER_TYPES,
    VALID_SIDES,
    VALID_TIF,
    AccountInfo,
    Broker,
    OrderResult,
    OrdersResult,
    Position,
    PositionsResult,
)

__all__ = [
    "NEUTRAL_STATUSES",
    "ORDER_TYPE_LIMIT",
    "ORDER_TYPE_MARKET",
    "SIDE_BUY",
    "SIDE_SELL",
    "STATUS_CANCELED",
    "STATUS_ERROR",
    "STATUS_FILLED",
    "STATUS_NEW",
    "STATUS_PARTIALLY_FILLED",
    "STATUS_REJECTED",
    "STATUS_UNKNOWN",
    "TERMINAL_STATUSES",
    "TIF_DAY",
    "TIF_GTC",
    "VALID_ORDER_TYPES",
    "VALID_SIDES",
    "VALID_TIF",
    "AccountInfo",
    "Broker",
    "OrderResult",
    "OrdersResult",
    "Position",
    "PositionsResult",
]
