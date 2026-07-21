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
_MAX_CONTRACT_PAGES = 100
_MAX_SNAPSHOT_PAGES = 100

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


# A malformed 2xx body is an upstream failure, not a valid empty collection.
# These endpoint-specific reasons ride on OptionChainResult.reason and are also
# logged to stderr so a shares fallback cannot hide the provider failure.
_MALFORMED_CONTRACTS = "malformed contracts response"
_MALFORMED_SNAPSHOTS = "malformed snapshots response"


def _parse_contracts_body(
    body: Any,
) -> tuple[list[OptionContract], str | None, str]:
    """Parse a 2xx contracts body into contracts, next token, and error.

    An empty ``option_contracts`` list is valid. Missing or wrongly typed
    top-level data is not. A missing, null, or empty ``next_page_token`` marks
    the final page; any other token must be a string.
    """
    if body is None:
        return [], None, f"{_MALFORMED_CONTRACTS}: empty body or JSON null"
    if not isinstance(body, Mapping):
        return (
            [],
            None,
            f"{_MALFORMED_CONTRACTS}: top-level body must be an object",
        )
    rows = body.get("option_contracts")
    if not isinstance(rows, list):
        return (
            [],
            None,
            f"{_MALFORMED_CONTRACTS}: missing 'option_contracts' list",
        )
    raw_page_token = body.get("next_page_token")
    if raw_page_token is not None and not isinstance(raw_page_token, str):
        return (
            [],
            None,
            f"{_MALFORMED_CONTRACTS}: 'next_page_token' must be a string or null",
        )
    page_token = raw_page_token or None
    contracts = [_parse_contract(r) for r in rows if isinstance(r, Mapping)]
    return contracts, page_token, ""


def _parse_snapshots_body(
    body: Any,
) -> tuple[dict[str, Mapping[str, Any]], str | None, str]:
    """Parse a 2xx snapshots body into snapshots, next token, and error.

    An empty ``snapshots`` object is valid. Missing or wrongly typed top-level
    data is not. A missing, null, or empty ``next_page_token`` marks the final
    page; any other token must be a string.
    """
    if body is None:
        return {}, None, f"{_MALFORMED_SNAPSHOTS}: empty body or JSON null"
    if not isinstance(body, Mapping):
        return (
            {},
            None,
            f"{_MALFORMED_SNAPSHOTS}: top-level body must be an object",
        )
    raw_snapshots = body.get("snapshots")
    if not isinstance(raw_snapshots, Mapping):
        return (
            {},
            None,
            f"{_MALFORMED_SNAPSHOTS}: missing 'snapshots' object",
        )
    raw_page_token = body.get("next_page_token")
    if raw_page_token is not None and not isinstance(raw_page_token, str):
        return (
            {},
            None,
            f"{_MALFORMED_SNAPSHOTS}: 'next_page_token' must be a string or null",
        )
    page_token = raw_page_token or None
    snapshots = {
        str(symbol): snapshot
        for symbol, snapshot in raw_snapshots.items()
        if isinstance(snapshot, Mapping)
    }
    return snapshots, page_token, ""


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
        except Exception:  # noqa: BLE001 - malformed upstream data must fail soft
            # Preserve non-JSON as a distinct error only for a successful HTTP
            # response. A non-2xx free-tier/policy response keeps its existing
            # HTTP status semantics.
            body_error = (
                "non-JSON response body"
                if 200 <= resp.status_code < 300
                else ""
            )
            return True, resp.status_code, None, body_error
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
        contracts, _reason = self._list_option_contracts_result(
            underlying,
            expiration_gte=expiration_gte,
            expiration_lte=expiration_lte,
            option_type=option_type,
            limit=limit,
        )
        return contracts

    def _list_option_contracts_result(
        self,
        underlying: str,
        *,
        expiration_gte: str | None = None,
        expiration_lte: str | None = None,
        option_type: str | None = None,
        limit: int = 100,
    ) -> tuple[list[OptionContract], str]:
        """List contracts while retaining an upstream-failure reason."""
        params: dict[str, Any] = {"underlying_symbols": underlying, "limit": limit}
        if expiration_gte is not None:
            params["expiration_date_gte"] = expiration_gte
        if expiration_lte is not None:
            params["expiration_date_lte"] = expiration_lte
        if option_type is not None:
            params["type"] = option_type
        contracts_by_symbol: dict[str, OptionContract] = {}
        page_token: str | None = None
        seen_page_tokens: set[str] = set()

        for page_number in range(1, _MAX_CONTRACT_PAGES + 1):
            page_params = dict(params)
            if page_token is not None:
                page_params["page_token"] = page_token
            ok, status, body, error = self._get(
                self._trading_base,
                "/v2/options/contracts",
                params=page_params,
            )
            if not ok or not self._ok2xx(status):
                if page_number == 1:
                    reason = (
                        f"contracts unavailable for {underlying} "
                        f"({error or f'HTTP {status}'})"
                    )
                else:
                    reason = (
                        f"contracts pagination incomplete for {underlying} on "
                        f"page {page_number} after {len(contracts_by_symbol)} "
                        f"unique contracts ({error or f'HTTP {status}'})"
                    )
                print(f"  options: {reason}", file=sys.stderr)
                return [], reason

            page_contracts, next_page_token, malformed_reason = (
                _parse_contracts_body(body)
            )
            if error:
                malformed_reason = f"{_MALFORMED_CONTRACTS}: {error}"
            if malformed_reason:
                if page_number == 1:
                    reason = (
                        f"{malformed_reason} for {underlying} (HTTP {status})"
                    )
                else:
                    reason = (
                        f"{malformed_reason} for {underlying} on page "
                        f"{page_number} after {len(contracts_by_symbol)} unique "
                        f"contracts (HTTP {status})"
                    )
                print(f"  options: {reason}", file=sys.stderr)
                return [], reason

            for contract in page_contracts:
                contracts_by_symbol.setdefault(contract.symbol, contract)
            if next_page_token is None:
                return list(contracts_by_symbol.values()), ""
            if next_page_token in seen_page_tokens:
                reason = (
                    f"contracts pagination incomplete for {underlying}: repeated "
                    f"page token after page {page_number}; "
                    f"{len(contracts_by_symbol)} unique contracts discarded"
                )
                print(f"  options: {reason}", file=sys.stderr)
                return [], reason
            seen_page_tokens.add(next_page_token)
            page_token = next_page_token

        reason = (
            f"contracts pagination incomplete for {underlying}: exceeded maximum "
            f"of {_MAX_CONTRACT_PAGES} pages; {len(contracts_by_symbol)} unique "
            "contracts discarded"
        )
        print(f"  options: {reason}", file=sys.stderr)
        return [], reason

    def _fetch_snapshots(self, underlying: str) -> dict[str, Mapping[str, Any]]:
        """Fetch greeks + quotes keyed by OCC symbol. Fail-soft → {}."""
        snapshots, _reason = self._fetch_snapshots_result(underlying)
        return snapshots

    def _fetch_snapshots_result(
        self, underlying: str,
    ) -> tuple[dict[str, Mapping[str, Any]], str]:
        """Fetch every snapshot page while retaining upstream-failure reasons.

        A first-page HTTP/transport/access failure keeps the established
        free-tier behavior: return an empty mapping so listed contracts remain
        available without greeks. Once a continuation token has been followed,
        any incomplete traversal returns an explicit reason that invalidates the
        chain rather than exposing partial greeks as authoritative.
        """
        params: dict[str, Any] = {"limit": 1000}
        snapshots_by_symbol: dict[str, Mapping[str, Any]] = {}
        page_token: str | None = None
        seen_page_tokens: set[str] = set()

        for page_number in range(1, _MAX_SNAPSHOT_PAGES + 1):
            page_params = dict(params)
            if page_token is not None:
                page_params["page_token"] = page_token
            ok, status, body, error = self._get(
                self._data_base,
                f"/v1beta1/options/snapshots/{underlying}",
                params=page_params,
            )
            if not ok or not self._ok2xx(status):
                if page_number == 1:
                    print(
                        f"  options: snapshots unavailable for {underlying} "
                        f"({error or f'HTTP {status}'})",
                        file=sys.stderr,
                    )
                    return {}, ""
                reason = (
                    f"snapshots pagination incomplete for {underlying} on page "
                    f"{page_number} after {len(snapshots_by_symbol)} unique "
                    "snapshots; partial snapshots discarded "
                    f"({error or f'HTTP {status}'})"
                )
                print(f"  options: {reason}", file=sys.stderr)
                return {}, reason

            page_snapshots, next_page_token, malformed_reason = (
                _parse_snapshots_body(body)
            )
            if error:
                malformed_reason = f"{_MALFORMED_SNAPSHOTS}: {error}"
            if malformed_reason:
                if page_number == 1:
                    reason = (
                        f"{malformed_reason} for {underlying} (HTTP {status})"
                    )
                else:
                    reason = (
                        f"{malformed_reason} for {underlying} on page "
                        f"{page_number} after {len(snapshots_by_symbol)} unique "
                        f"snapshots; partial snapshots discarded (HTTP {status})"
                    )
                print(f"  options: {reason}", file=sys.stderr)
                return {}, reason

            for symbol, snapshot in page_snapshots.items():
                snapshots_by_symbol.setdefault(symbol, snapshot)
            if next_page_token is None:
                return snapshots_by_symbol, ""
            if next_page_token in seen_page_tokens:
                reason = (
                    f"snapshots pagination incomplete for {underlying}: repeated "
                    f"page token after page {page_number}; "
                    f"{len(snapshots_by_symbol)} unique snapshots discarded"
                )
                print(f"  options: {reason}", file=sys.stderr)
                return {}, reason
            seen_page_tokens.add(next_page_token)
            page_token = next_page_token

        reason = (
            f"snapshots pagination incomplete for {underlying}: exceeded maximum "
            f"of {_MAX_SNAPSHOT_PAGES} pages; {len(snapshots_by_symbol)} unique "
            "snapshots discarded"
        )
        print(f"  options: {reason}", file=sys.stderr)
        return {}, reason

    def get_option_chain(self, underlying: str) -> OptionChainResult:
        """The full chain with greeks populated. Fail-soft → ``ok=False``.

        Contract metadata (strike/expiry/type/open_interest) is merged with the
        data-API snapshot (delta/theta/vega/gamma + bid/ask/mid). If contracts
        cannot be listed the result is ``ok=False``. A first-page snapshot
        transport, HTTP, or access-policy failure retains the established
        degraded behavior: contracts are returned with ``None`` greeks. An
        incomplete paginated response or malformed successful response is
        instead an explicit ``ok=False`` upstream failure.
        """
        contracts, contracts_reason = self._list_option_contracts_result(underlying)
        if contracts_reason:
            return OptionChainResult(ok=False, reason=contracts_reason)
        if not contracts:
            return OptionChainResult(
                ok=False,
                reason=f"no contracts listed for {underlying}",
            )
        snapshots, snapshots_reason = self._fetch_snapshots_result(underlying)
        if snapshots_reason:
            return OptionChainResult(ok=False, reason=snapshots_reason)
        merged = [_apply_snapshot(c, snapshots.get(c.symbol)) for c in contracts]
        return OptionChainResult(ok=True, contracts=merged)
