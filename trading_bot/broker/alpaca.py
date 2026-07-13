"""Alpaca PAPER-trading adapter — Phase 11.

A concrete :class:`Broker` implemented with direct REST calls (``requests``),
mirroring the project's existing fail-soft HTTP clients (``news_client``,
``readiness._pushover_notify``). It targets Alpaca's PAPER endpoint ONLY:

* :data:`ALPACA_PAPER_BASE_URL` is the sole base URL defined in this module —
  the live trading URL is never written down here.
* ``AlpacaBroker.__init__`` asserts the base URL IS the paper URL and refuses to
  construct otherwise, so this phase's code cannot reach the live endpoint even
  by mistake. Going live is a deliberate future switch behind the Phase 15
  safety layer, not a parameter you can pass here.

No method raises on a network / API failure: every read returns an ``ok=False``
neutral result and every write returns a structured rejected / error
``OrderResult``. Credentials are read from the secrets layer; when unset, calls
fail soft (``reason='ALPACA credentials unset'``) and nothing is sent.
"""

from __future__ import annotations

import dataclasses
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import requests

from trading_bot import secrets
from trading_bot.broker.base import (
    ORDER_TYPE_LIMIT,
    STATUS_CANCELED,
    STATUS_ERROR,
    STATUS_FILLED,
    STATUS_NEW,
    STATUS_PARTIALLY_FILLED,
    STATUS_REJECTED,
    STATUS_UNKNOWN,
    TIF_DAY,
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

# The ONLY base URL this module knows. Paper trading, no real capital. The live
# endpoint is intentionally absent — see the module docstring.
ALPACA_PAPER_BASE_URL = "https://paper-api.alpaca.markets"

_HTTP_TIMEOUT_SECONDS = 10

# The bot stores crypto tickers in yfinance form (BTC-USD); Alpaca's trading
# API expects the pair with a slash (BTC/USD). The suffix match is deliberately
# tight so only crypto pairs translate — stock tickers and OCC option symbols
# contain no '-USD' suffix and pass through untouched.
_CRYPTO_TICKER_RE = re.compile(r"^(?P<base>[A-Z0-9]+)-USD$")


def to_alpaca_symbol(symbol: str) -> str:
    """Translate an internal symbol to Alpaca's wire format (Phase 17).

    ``BTC-USD`` → ``BTC/USD``; anything that is not a yfinance-style crypto
    pair (stock tickers, OCC option symbols) is returned unchanged. Applied at
    THE broker boundary — ``AlpacaBroker.submit_order`` — so every submission
    path (long-term crypto entry, protective-exit close, emergency shutdown)
    gets the translation and internal storage/display keeps the yfinance form.
    """
    match = _CRYPTO_TICKER_RE.match(symbol)
    if match is None:
        return symbol
    return f"{match.group('base')}/USD"

# Alpaca order lifecycle states → the neutral status set. Anything unmapped
# becomes STATUS_UNKNOWN (a new Alpaca state we have not classified), never an
# error — the call still succeeded.
_NEUTRAL_STATUS_MAP: dict[str, str] = {
    "new": STATUS_NEW,
    "accepted": STATUS_NEW,
    "pending_new": STATUS_NEW,
    "accepted_for_bidding": STATUS_NEW,
    "held": STATUS_NEW,
    "calculated": STATUS_NEW,
    "partially_filled": STATUS_PARTIALLY_FILLED,
    "filled": STATUS_FILLED,
    "done_for_day": STATUS_CANCELED,
    "canceled": STATUS_CANCELED,
    "expired": STATUS_CANCELED,
    "replaced": STATUS_CANCELED,
    "pending_cancel": STATUS_CANCELED,
    "pending_replace": STATUS_CANCELED,
    "stopped": STATUS_CANCELED,
    "suspended": STATUS_CANCELED,
    "rejected": STATUS_REJECTED,
}


def map_status(raw: str | None) -> str:
    """Map an Alpaca order status onto the neutral set (STATUS_UNKNOWN if new)."""
    if raw is None:
        return STATUS_UNKNOWN
    return _NEUTRAL_STATUS_MAP.get(raw.lower(), STATUS_UNKNOWN)


def _to_float(value: Any) -> float | None:
    """Coerce Alpaca's string-encoded numbers to float; None on absent/garbage."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _str_or_none(value: Any) -> str | None:
    return str(value) if value is not None else None


def _to_int(value: Any) -> int | None:
    """Coerce a value to int; None on absent/garbage (options level, etc.)."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_position(d: Mapping[str, Any]) -> Position:
    return Position(
        symbol=str(d.get("symbol", "")),
        qty=_to_float(d.get("qty")) or 0.0,
        side=str(d.get("side", "long")),
        avg_entry_price=_to_float(d.get("avg_entry_price")),
        market_value=_to_float(d.get("market_value")),
        unrealized_pl=_to_float(d.get("unrealized_pl")),
    )


def _parse_order(d: Mapping[str, Any]) -> OrderResult:
    """Map an Alpaca order object onto the neutral ``OrderResult`` (``ok=True``
    — the record was read successfully; its lifecycle is carried by ``status``)."""
    raw_status = _str_or_none(d.get("status"))
    return OrderResult(
        ok=True,
        status=map_status(raw_status),
        order_id=_str_or_none(d.get("id")),
        client_order_id=_str_or_none(d.get("client_order_id")),
        symbol=_str_or_none(d.get("symbol")),
        qty=_to_float(d.get("qty")),
        filled_qty=_to_float(d.get("filled_qty")) or 0.0,
        filled_avg_price=_to_float(d.get("filled_avg_price")),
        side=_str_or_none(d.get("side")),
        order_type=_str_or_none(d.get("type")),
        time_in_force=_str_or_none(d.get("time_in_force")),
        limit_price=_to_float(d.get("limit_price")),
        submitted_at=_str_or_none(d.get("submitted_at")),
        raw_status=raw_status,
    )


def _validate_order(
    symbol: str, qty: float, side: str, order_type: str,
    limit_price: float | None, time_in_force: str,
) -> OrderResult | None:
    """Local pre-flight validation. Returns a structured rejection (so nothing
    is sent) or None when the order is well-formed. Catches the obvious
    mistakes — bad side/type/tif, non-positive qty, a LIMIT order with no price
    (the design forbids naive market orders, so LIMIT needs a price)."""

    def reject(reason: str) -> OrderResult:
        return OrderResult(
            ok=False, status=STATUS_REJECTED, symbol=symbol, side=side,
            order_type=order_type, qty=qty, limit_price=limit_price,
            time_in_force=time_in_force, reason=reason,
        )

    if side not in VALID_SIDES:
        return reject(f"invalid side {side!r}")
    if order_type not in VALID_ORDER_TYPES:
        return reject(f"invalid order_type {order_type!r}")
    if time_in_force not in VALID_TIF:
        return reject(f"invalid time_in_force {time_in_force!r}")
    if qty <= 0:
        return reject("qty must be positive")
    if order_type == ORDER_TYPE_LIMIT and limit_price is None:
        return reject("limit order requires a limit_price")
    return None


@dataclass(frozen=True)
class _Response:
    """Outcome of a single HTTP attempt. ``ok`` means a response was received
    (any status code); a transport failure or missing creds gives ``ok=False``
    with ``error`` populated. ``status_code`` is None when no response arrived.
    """

    ok: bool
    status_code: int | None
    body: Any
    error: str


class AlpacaBroker(Broker):
    """Alpaca paper-trading adapter. Construction is guarded to the paper URL."""

    def __init__(self, *, base_url: str = ALPACA_PAPER_BASE_URL) -> None:
        if base_url != ALPACA_PAPER_BASE_URL:
            # Structural guard: this phase is paper-only. Refuse anything else.
            raise ValueError(
                "AlpacaBroker is PAPER-ONLY: base_url must be "
                f"{ALPACA_PAPER_BASE_URL!r}, got {base_url!r}. Live trading is a "
                "deliberate future switch behind the Phase 15 safety layer."
            )
        self._base_url = base_url
        self._key = secrets.get_secret("ALPACA_API_KEY")
        self._secret = secrets.get_secret("ALPACA_SECRET_KEY")

    # ── HTTP chokepoint (single fail-soft network path) ──────────────────────

    def _headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self._key or "",
            "APCA-API-SECRET-KEY": self._secret or "",
        }

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> _Response:
        """Perform one paper-API request. Never raises.

        Missing credentials short-circuit to a clear ``ok=False`` before any
        network call. A transport error is caught and logged. A received
        response — even a 4xx/5xx — is returned ``ok=True`` so callers can
        distinguish a broker rejection (has a status code) from an outage.
        """
        if not self._key or not self._secret:
            print(
                "  broker: ALPACA credentials unset - request skipped",
                file=sys.stderr,
            )
            return _Response(
                ok=False, status_code=None, body=None,
                error="ALPACA credentials unset",
            )
        url = f"{self._base_url}{path}"
        try:
            resp = requests.request(
                method, url, headers=self._headers(), params=params,
                json=json_body, timeout=_HTTP_TIMEOUT_SECONDS,
            )
        except Exception as exc:  # noqa: BLE001 - broker I/O must never raise into the loop
            print(f"  broker: {method} {path} error ({exc})", file=sys.stderr)
            return _Response(ok=False, status_code=None, body=None, error=str(exc))
        try:
            body = resp.json() if resp.content else None
        except Exception:  # noqa: BLE001 - a non-JSON body is not fatal
            body = None
        return _Response(ok=True, status_code=resp.status_code, body=body, error="")

    @staticmethod
    def _succeeded(resp: _Response) -> bool:
        """True iff a response arrived with a 2xx status code."""
        return (
            resp.ok and resp.status_code is not None
            and 200 <= resp.status_code < 300
        )

    @staticmethod
    def _error_reason(resp: _Response) -> str:
        """Human-readable reason from a failed response, for logs / results."""
        if resp.status_code is None:
            return resp.error or "unavailable"
        message = ""
        if isinstance(resp.body, dict):
            message = str(resp.body.get("message", ""))
        return message or f"HTTP {resp.status_code}"

    def _log_unavailable(self, what: str, resp: _Response) -> str:
        """Log a non-transport failure (transport ones are logged in _request)
        and return the reason string."""
        reason = self._error_reason(resp)
        if resp.status_code is not None:  # transport/creds already logged once
            print(f"  broker: {what} unavailable ({reason})", file=sys.stderr)
        return reason

    # ── read path (Section 3) ────────────────────────────────────────────────

    def get_account(self) -> AccountInfo:
        resp = self._request("GET", "/v2/account")
        if not self._succeeded(resp):
            return AccountInfo(ok=False, reason=self._log_unavailable("account", resp))
        body: Mapping[str, Any] = resp.body if isinstance(resp.body, dict) else {}
        return AccountInfo(
            ok=True,
            account_number=_str_or_none(body.get("account_number")),
            buying_power=_to_float(body.get("buying_power")),
            cash=_to_float(body.get("cash")),
            equity=_to_float(body.get("equity")),
            currency=str(body.get("currency", "USD")),
            status=_str_or_none(body.get("status")),
            options_trading_level=_to_int(body.get("options_trading_level")),
        )

    def get_positions(self) -> PositionsResult:
        resp = self._request("GET", "/v2/positions")
        if not self._succeeded(resp):
            return PositionsResult(
                ok=False, reason=self._log_unavailable("positions", resp),
            )
        rows = resp.body if isinstance(resp.body, list) else []
        positions = [_parse_position(r) for r in rows if isinstance(r, dict)]
        return PositionsResult(ok=True, positions=positions)

    # ── write path (Section 4) ───────────────────────────────────────────────

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
        # Log intent BEFORE anything is sent, so there is always a record of
        # what was attempted even if the send fails.
        print(
            f"  broker: submit intent side={side} qty={qty} {symbol} "
            f"type={order_type} limit={limit_price} tif={time_in_force}",
            file=sys.stderr,
        )
        rejection = _validate_order(
            symbol, qty, side, order_type, limit_price, time_in_force,
        )
        if rejection is not None:
            print(
                f"  broker: submit rejected locally ({rejection.reason})",
                file=sys.stderr,
            )
            return rejection

        # Phase 17: crypto pairs are stored internally in yfinance form
        # (BTC-USD) but Alpaca's API wants BTC/USD — translate at THE boundary.
        body: dict[str, Any] = {
            "symbol": to_alpaca_symbol(symbol), "qty": str(qty), "side": side,
            "type": order_type, "time_in_force": time_in_force,
        }
        if limit_price is not None:
            body["limit_price"] = str(limit_price)
        if client_order_id is not None:
            body["client_order_id"] = client_order_id

        resp = self._request("POST", "/v2/orders", json_body=body)
        result = self._submit_result(resp, symbol)
        print(
            f"  broker: submit result status={result.status} "
            f"order_id={result.order_id} reason={result.reason!r}",
            file=sys.stderr,
        )
        return result

    def _submit_result(self, resp: _Response, symbol: str) -> OrderResult:
        """Map a submission response to an OrderResult. A 4xx is a broker
        REJECTION (insufficient buying power, market closed, invalid symbol);
        a 5xx / transport failure is an ERROR. Never raises."""
        if not resp.ok:
            return OrderResult(
                ok=False, status=STATUS_ERROR, symbol=symbol,
                reason=resp.error or "unavailable",
            )
        if self._succeeded(resp):
            body = resp.body if isinstance(resp.body, dict) else {}
            parsed = _parse_order(body)
            if parsed.status == STATUS_REJECTED:
                # A rejected order can arrive inside a 2xx body; surface its own
                # message rather than the (misleading) "HTTP 200".
                message = str(body.get("message", "")) if body else ""
                return dataclasses.replace(
                    parsed, ok=False, reason=message or "rejected",
                )
            return parsed
        reason = self._error_reason(resp)
        code = resp.status_code or 0
        status = STATUS_REJECTED if 400 <= code < 500 else STATUS_ERROR
        return OrderResult(
            ok=False, status=status, symbol=symbol, reason=reason,
        )

    def get_order(self, order_id: str) -> OrderResult:
        resp = self._request("GET", f"/v2/orders/{order_id}")
        if not self._succeeded(resp):
            reason = self._log_unavailable("order", resp)
            return OrderResult(
                ok=False, status=STATUS_ERROR, order_id=order_id, reason=reason,
            )
        body = resp.body if isinstance(resp.body, dict) else {}
        parsed = _parse_order(body)
        if parsed.order_id is None:
            parsed = dataclasses.replace(parsed, order_id=order_id)
        return parsed

    def cancel_order(self, order_id: str) -> OrderResult:
        # Alpaca returns 204 No Content on a successful cancel.
        resp = self._request("DELETE", f"/v2/orders/{order_id}")
        if not self._succeeded(resp):
            reason = self._log_unavailable("cancel", resp)
            return OrderResult(
                ok=False, status=STATUS_ERROR, order_id=order_id, reason=reason,
            )
        return OrderResult(
            ok=True, status=STATUS_CANCELED, order_id=order_id,
            raw_status="canceled",
        )

    def list_orders(self, status: str = "open") -> OrdersResult:
        resp = self._request("GET", "/v2/orders", params={"status": status})
        if not self._succeeded(resp):
            return OrdersResult(ok=False, reason=self._log_unavailable("orders", resp))
        rows = resp.body if isinstance(resp.body, list) else []
        orders = [_parse_order(r) for r in rows if isinstance(r, dict)]
        return OrdersResult(ok=True, orders=orders)
