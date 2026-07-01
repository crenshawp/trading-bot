"""Options broker layer — Phase 13. Neutral contract type + OCC encoding.

The neutral :class:`OptionContract` is the only options shape higher layers see;
the companion ``AlpacaOptionsClient`` (added in Section 2) translates Alpaca's
raw contract + snapshot JSON into it. Single-leg only (buy call / buy put) — no
spreads or multi-leg this phase.

OCC symbol format (Alpaca form, no space-padding of the root):

    {ROOT}{YYMMDD}{C|P}{strike × 1000, zero-padded to 8 digits}

e.g. AAPL 2026-01-16 call $150.00 → ``AAPL260116C00150000``. Alpaca returns this
as each contract's ``symbol``; the existing Phase 11 order path submits it
verbatim (an option order is just an order whose ``symbol`` is an OCC string).
"""

from __future__ import annotations

import dataclasses
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import requests

from trading_bot import secrets
from trading_bot.broker.alpaca import ALPACA_PAPER_BASE_URL

# Alpaca options market data lives on the DATA host (read-only quotes + greeks).
# This is NOT a trading endpoint — the client never places orders — so it opens
# no live-trading path. Option orders reuse AlpacaBroker.submit_order (paper).
ALPACA_OPTIONS_DATA_BASE_URL = "https://data.alpaca.markets"

_HTTP_TIMEOUT_SECONDS = 10

OPTION_TYPE_CALL = "call"
OPTION_TYPE_PUT = "put"
VALID_OPTION_TYPES: frozenset[str] = frozenset({OPTION_TYPE_CALL, OPTION_TYPE_PUT})


@dataclass(frozen=True)
class OptionContract:
    """A single option contract in neutral form. Greeks/quotes are ``None`` when
    the chain snapshot did not supply them (such a contract fails the liquidity /
    delta gates and is excluded from selection)."""

    symbol: str                    # OCC symbol, directly submittable
    underlying: str
    option_type: str               # 'call' | 'put'
    strike: float
    expiry: str                    # 'YYYY-MM-DD'
    delta: float | None = None
    theta: float | None = None
    vega: float | None = None
    gamma: float | None = None
    open_interest: int | None = None
    bid: float | None = None
    ask: float | None = None
    mid: float | None = None

    @property
    def spread_pct(self) -> float | None:
        """Bid-ask spread as a percentage of mid, or None if not computable."""
        if (
            self.bid is None or self.ask is None
            or self.mid is None or self.mid <= 0.0
        ):
            return None
        return (self.ask - self.bid) / self.mid * 100.0


def occ_symbol(underlying: str, expiry: str, option_type: str, strike: float) -> str:
    """Build the Alpaca OCC option symbol. ``expiry`` is ``'YYYY-MM-DD'``.

    The strike is encoded as ``strike × 1000`` zero-padded to 8 digits, so a
    $150.00 strike → ``00150000`` and a $7.50 strike → ``00007500``.
    """
    if option_type not in VALID_OPTION_TYPES:
        raise ValueError(f"invalid option_type {option_type!r}")
    yymmdd = f"{expiry[2:4]}{expiry[5:7]}{expiry[8:10]}"
    cp = "C" if option_type == OPTION_TYPE_CALL else "P"
    strike_thousandths = int(round(strike * 1000))
    return f"{underlying.upper()}{yymmdd}{cp}{strike_thousandths:08d}"


def parse_occ_symbol(symbol: str) -> tuple[str, str, str, float] | None:
    """Reverse :func:`occ_symbol` → ``(root, 'YYYY-MM-DD', option_type, strike)``.

    Returns None on a malformed symbol (too short, bad C/P marker, or a
    non-numeric date/strike) — parsing must never raise on garbage input.
    """
    if len(symbol) < 16:   # need root(>=1) + 6 date + 1 cp + 8 strike
        return None
    tail = symbol[-15:]
    root = symbol[:-15]
    cp = tail[6]
    if not root or cp not in ("C", "P"):
        return None
    date_part, strike_part = tail[:6], tail[7:]
    if not (date_part.isdigit() and strike_part.isdigit()):
        return None
    expiry = f"20{date_part[0:2]}-{date_part[2:4]}-{date_part[4:6]}"
    option_type = OPTION_TYPE_CALL if cp == "C" else OPTION_TYPE_PUT
    strike = int(strike_part) / 1000.0
    return root, expiry, option_type, strike


# ── contract discovery + chain/greeks fetch (companion options client) ───────


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_contract(d: Mapping[str, Any]) -> OptionContract:
    """Neutral OptionContract from an Alpaca /v2/options/contracts row (metadata
    only — greeks/quotes come from the snapshot merge)."""
    return OptionContract(
        symbol=str(d.get("symbol", "")),
        underlying=str(d.get("underlying_symbol", "")),
        option_type=str(d.get("type", "")),
        strike=_as_float(d.get("strike_price")) or 0.0,
        expiry=str(d.get("expiration_date", "")),
        open_interest=_as_int(d.get("open_interest")),
    )


def _apply_snapshot(
    contract: OptionContract, snap: Mapping[str, Any] | None,
) -> OptionContract:
    """Populate greeks + bid/ask/mid on a contract from its data-API snapshot."""
    if snap is None:
        return contract
    raw_greeks = snap.get("greeks")
    greeks = raw_greeks if isinstance(raw_greeks, dict) else {}
    raw_quote = snap.get("latestQuote")
    quote = raw_quote if isinstance(raw_quote, dict) else {}
    bid = _as_float(quote.get("bp"))
    ask = _as_float(quote.get("ap"))
    mid = (bid + ask) / 2.0 if bid is not None and ask is not None else None
    return dataclasses.replace(
        contract,
        delta=_as_float(greeks.get("delta")),
        theta=_as_float(greeks.get("theta")),
        vega=_as_float(greeks.get("vega")),
        gamma=_as_float(greeks.get("gamma")),
        bid=bid, ask=ask, mid=mid,
    )


@dataclass(frozen=True)
class OptionChainResult:
    """A fetched option chain. ``ok=False`` means the chain was unavailable
    (options not enabled, API error, or none listed) — the caller then falls
    through to the fractional-share path."""

    ok: bool = False
    reason: str = ""
    contracts: list[OptionContract] = field(default_factory=list)


class AlpacaOptionsClient:
    """Read-only options market-data client (Phase 13). Fail-soft, PAPER-guarded.

    Contract metadata comes from the paper trading API
    (``/v2/options/contracts``); greeks + quotes come from the Alpaca options
    DATA API. Every method is fail-soft: an API error, missing credentials, or
    options-not-enabled (403) returns an empty result with a logged reason and
    NEVER raises. It places no orders — option orders reuse
    ``AlpacaBroker.submit_order`` with the OCC symbol.
    """

    def __init__(
        self,
        *,
        trading_base_url: str = ALPACA_PAPER_BASE_URL,
        data_base_url: str = ALPACA_OPTIONS_DATA_BASE_URL,
    ) -> None:
        if trading_base_url != ALPACA_PAPER_BASE_URL:
            raise ValueError(
                "AlpacaOptionsClient is PAPER-ONLY: trading_base_url must be "
                f"{ALPACA_PAPER_BASE_URL!r}, got {trading_base_url!r}."
            )
        self._trading_base = trading_base_url
        self._data_base = data_base_url
        self._key = secrets.get_secret("ALPACA_API_KEY")
        self._secret = secrets.get_secret("ALPACA_SECRET_KEY")

    def _headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self._key or "",
            "APCA-API-SECRET-KEY": self._secret or "",
        }

    def _get(
        self, base: str, path: str, *, params: dict[str, Any] | None = None,
    ) -> tuple[bool, int | None, Any, str]:
        """Fail-soft GET → ``(ok, status_code, body, error)``. Never raises."""
        if not self._key or not self._secret:
            print(
                "  options: ALPACA credentials unset - request skipped",
                file=sys.stderr,
            )
            return False, None, None, "ALPACA credentials unset"
        try:
            resp = requests.get(
                f"{base}{path}", headers=self._headers(), params=params,
                timeout=_HTTP_TIMEOUT_SECONDS,
            )
        except Exception as exc:  # noqa: BLE001 - options I/O must never raise
            print(f"  options: GET {path} error ({exc})", file=sys.stderr)
            return False, None, None, str(exc)
        try:
            body = resp.json() if resp.content else None
        except Exception:  # noqa: BLE001 - a non-JSON body is not fatal
            body = None
        return True, resp.status_code, body, ""

    @staticmethod
    def _ok2xx(status: int | None) -> bool:
        return status is not None and 200 <= status < 300

    def list_option_contracts(
        self,
        underlying: str,
        *,
        expiration_gte: str | None = None,
        expiration_lte: str | None = None,
        option_type: str | None = None,
        limit: int = 100,
    ) -> list[OptionContract]:
        """List option contracts (metadata) for ``underlying``. Fail-soft → []."""
        params: dict[str, Any] = {"underlying_symbols": underlying, "limit": limit}
        if expiration_gte is not None:
            params["expiration_date_gte"] = expiration_gte
        if expiration_lte is not None:
            params["expiration_date_lte"] = expiration_lte
        if option_type is not None:
            params["type"] = option_type
        ok, status, body, error = self._get(
            self._trading_base, "/v2/options/contracts", params=params,
        )
        if not ok or not self._ok2xx(status):
            print(
                f"  options: contracts unavailable for {underlying} "
                f"({error or f'HTTP {status}'})",
                file=sys.stderr,
            )
            return []
        rows = body.get("option_contracts", []) if isinstance(body, dict) else []
        return [_parse_contract(r) for r in rows if isinstance(r, dict)]

    def _fetch_snapshots(self, underlying: str) -> dict[str, Mapping[str, Any]]:
        """Fetch greeks + quotes keyed by OCC symbol. Fail-soft → {}."""
        ok, status, body, error = self._get(
            self._data_base, f"/v1beta1/options/snapshots/{underlying}",
            params={"limit": 1000},
        )
        if not ok or not self._ok2xx(status):
            print(
                f"  options: snapshots unavailable for {underlying} "
                f"({error or f'HTTP {status}'})",
                file=sys.stderr,
            )
            return {}
        snaps = body.get("snapshots", {}) if isinstance(body, dict) else {}
        return {str(k): v for k, v in snaps.items() if isinstance(v, dict)}

    def get_option_chain(self, underlying: str) -> OptionChainResult:
        """The full chain with greeks populated. Fail-soft → ``ok=False``.

        Contract metadata (strike/expiry/type/open_interest) is merged with the
        data-API snapshot (delta/theta/vega/gamma + bid/ask/mid). If contracts
        cannot be listed the result is ``ok=False``; if only the snapshots fail,
        contracts are returned with ``None`` greeks (they then fail the delta /
        liquidity gates and are excluded from selection).
        """
        contracts = self.list_option_contracts(underlying)
        if not contracts:
            return OptionChainResult(
                ok=False,
                reason="no contracts (options unavailable or none listed)",
            )
        snapshots = self._fetch_snapshots(underlying)
        merged = [_apply_snapshot(c, snapshots.get(c.symbol)) for c in contracts]
        return OptionChainResult(ok=True, contracts=merged)
