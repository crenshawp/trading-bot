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

import sys
from dataclasses import dataclass
from typing import Any

import requests

from trading_bot import secrets
from trading_bot.broker.base import (
    ORDER_TYPE_LIMIT,
    STATUS_ERROR,
    TIF_DAY,
    AccountInfo,
    Broker,
    OrderResult,
    OrdersResult,
    PositionsResult,
)

# The ONLY base URL this module knows. Paper trading, no real capital. The live
# endpoint is intentionally absent — see the module docstring.
ALPACA_PAPER_BASE_URL = "https://paper-api.alpaca.markets"

_HTTP_TIMEOUT_SECONDS = 10


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
    def _error_reason(resp: _Response) -> str:
        """Human-readable reason from a failed response, for logs / results."""
        if resp.status_code is None:
            return resp.error or "unavailable"
        message = ""
        if isinstance(resp.body, dict):
            message = str(resp.body.get("message", ""))
        return message or f"HTTP {resp.status_code}"

    # ── read path (Section 3) ────────────────────────────────────────────────

    def get_account(self) -> AccountInfo:
        return AccountInfo(ok=False, reason="not implemented")

    def get_positions(self) -> PositionsResult:
        return PositionsResult(ok=False, reason="not implemented")

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
        return OrderResult(ok=False, status=STATUS_ERROR, reason="not implemented")

    def get_order(self, order_id: str) -> OrderResult:
        return OrderResult(ok=False, status=STATUS_ERROR, reason="not implemented")

    def cancel_order(self, order_id: str) -> OrderResult:
        return OrderResult(ok=False, status=STATUS_ERROR, reason="not implemented")

    def list_orders(self, status: str = "open") -> OrdersResult:
        return OrdersResult(ok=False, reason="not implemented")
