"""Abstract broker interface and neutral result types — Phase 11.

Every higher layer (future allocation / risk / execution wiring) talks to this
abstract :class:`Broker`, NEVER to a concrete broker SDK. Concrete adapters
(``AlpacaBroker`` is the first) translate the broker's wire shapes into the
neutral dataclasses defined here, so callers never see broker-specific JSON.
Adding or swapping a broker later is one new adapter against this interface.

Fail-soft is part of the contract: NO method on a ``Broker`` may raise on a
network / API failure. Every result type carries an ``ok`` flag and a ``reason``
string; a failed read returns ``ok=False`` with the reason logged, and an order
that the broker rejects returns a structured ``OrderResult`` with
``status='rejected'`` — never an exception into the scan loop.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

# ── order sides ──────────────────────────────────────────────────────────────
SIDE_BUY = "buy"
SIDE_SELL = "sell"
VALID_SIDES: frozenset[str] = frozenset({SIDE_BUY, SIDE_SELL})

# ── order types ──────────────────────────────────────────────────────────────
# LIMIT is the swing-trade default (see Broker.submit_order). MARKET exists for
# completeness but the design deliberately avoids naive market orders.
ORDER_TYPE_LIMIT = "limit"
ORDER_TYPE_MARKET = "market"
VALID_ORDER_TYPES: frozenset[str] = frozenset({ORDER_TYPE_LIMIT, ORDER_TYPE_MARKET})

# ── time in force ────────────────────────────────────────────────────────────
TIF_DAY = "day"
TIF_GTC = "gtc"
VALID_TIF: frozenset[str] = frozenset({TIF_DAY, TIF_GTC})

# ── neutral order statuses ───────────────────────────────────────────────────
# Broker-specific lifecycle states are mapped onto this small neutral set by
# each adapter. 'error' and 'unknown' are fail-soft sentinels, never real
# broker states: 'error' = the call could not reach a usable response;
# 'unknown' = the broker returned a state the adapter does not recognise.
STATUS_NEW = "new"                          # accepted / working, no fill yet
STATUS_PARTIALLY_FILLED = "partially_filled"
STATUS_FILLED = "filled"
STATUS_CANCELED = "canceled"
STATUS_REJECTED = "rejected"                # broker refused the order
STATUS_ERROR = "error"                      # transport / unavailable (fail-soft)
STATUS_UNKNOWN = "unknown"                  # unmapped broker status
NEUTRAL_STATUSES: frozenset[str] = frozenset({
    STATUS_NEW, STATUS_PARTIALLY_FILLED, STATUS_FILLED,
    STATUS_CANCELED, STATUS_REJECTED, STATUS_ERROR, STATUS_UNKNOWN,
})
# Statuses that mean the order is no longer live and never filled.
TERMINAL_STATUSES: frozenset[str] = frozenset({
    STATUS_CANCELED, STATUS_REJECTED,
})


@dataclass(frozen=True)
class AccountInfo:
    """Neutral account snapshot. ``ok=False`` means the fetch was unavailable
    (network / API error / missing credentials) — callers treat it as 'unknown',
    never as a zero-balance account."""

    ok: bool = False
    reason: str = ""
    account_number: str | None = None
    buying_power: float | None = None
    cash: float | None = None
    equity: float | None = None
    currency: str = "USD"
    status: str | None = None       # broker account status, e.g. 'ACTIVE'
    # Phase 13: Alpaca options approval level (0 = options not enabled). None when
    # the field was absent; the options layer fails soft to shares on 0/None.
    options_trading_level: int | None = None


@dataclass(frozen=True)
class Position:
    """Neutral open-position record. ``qty`` is signed by ``side`` semantics:
    a short position reports a positive ``qty`` with ``side='short'``."""

    symbol: str
    qty: float
    side: str = "long"              # 'long' | 'short'
    avg_entry_price: float | None = None
    market_value: float | None = None
    unrealized_pl: float | None = None


@dataclass(frozen=True)
class PositionsResult:
    """Result of a positions read. The ``ok`` flag distinguishes 'the broker
    has no positions' (``ok=True``, empty list) from 'the broker was
    unreachable' (``ok=False``) — a distinction reconciliation depends on."""

    ok: bool = False
    reason: str = ""
    positions: list[Position] = field(default_factory=list)


@dataclass(frozen=True)
class OrderResult:
    """Neutral order record / submission outcome.

    ``ok`` is True only when the broker accepted the order and returned a usable
    record. A broker rejection returns ``ok=False, status='rejected'`` with the
    reason; a transport failure returns ``ok=False, status='error'``. Either
    way the caller gets a structured value, never an exception.
    """

    ok: bool = False
    status: str = STATUS_UNKNOWN    # one of NEUTRAL_STATUSES
    reason: str = ""
    order_id: str | None = None
    client_order_id: str | None = None
    symbol: str | None = None
    qty: float | None = None
    filled_qty: float = 0.0
    filled_avg_price: float | None = None
    side: str | None = None
    order_type: str | None = None
    time_in_force: str | None = None
    limit_price: float | None = None
    submitted_at: str | None = None
    raw_status: str | None = None   # the broker's own status string, for audit


@dataclass(frozen=True)
class OrdersResult:
    """Result of an orders list read. Same fail-soft ``ok`` semantics as
    :class:`PositionsResult`."""

    ok: bool = False
    reason: str = ""
    orders: list[OrderResult] = field(default_factory=list)


class Broker(ABC):
    """The abstract two-way broker interface every adapter implements.

    Implementations MUST be fail-soft: no method raises on network / API
    failure. Reads return ``ok=False`` results; writes return structured
    rejected / error ``OrderResult`` values.
    """

    @abstractmethod
    def get_account(self) -> AccountInfo:
        """Buying power, cash, equity. Fail-soft → ``AccountInfo(ok=False)``."""

    @abstractmethod
    def get_positions(self) -> PositionsResult:
        """Current open positions. Fail-soft → ``PositionsResult(ok=False)``."""

    @abstractmethod
    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        *,
        order_type: str = ORDER_TYPE_LIMIT,
        limit_price: float | None = None,
        time_in_force: str = TIF_DAY,
        client_order_id: str | None = None,
    ) -> OrderResult:
        """Submit an order. Defaults to a LIMIT order (no naive market orders).

        A broker rejection (insufficient buying power, market closed, invalid
        symbol, …) returns a structured ``OrderResult(ok=False,
        status='rejected')`` — it never raises.
        """

    @abstractmethod
    def get_order(self, order_id: str) -> OrderResult:
        """Fetch one order's current state. Fail-soft → ``ok=False``."""

    @abstractmethod
    def cancel_order(self, order_id: str) -> OrderResult:
        """Cancel an order. Fail-soft → ``ok=False``."""

    @abstractmethod
    def list_orders(self, status: str = "open") -> OrdersResult:
        """List orders filtered by broker status. Fail-soft → ``ok=False``."""
