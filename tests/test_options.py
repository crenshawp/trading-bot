"""Tests for the options execution layer (Phase 13, single-leg, PAPER).

All Alpaca network calls are mocked — no live or paper network is touched. The
multiplier, strike selection, execution hierarchy, and exit watcher are the
core logic; the fetch client is tested against monkeypatched responses.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from trading_bot import config, db
from trading_bot.broker import options
from trading_bot.broker.options import OptionChainResult, OptionContract
from trading_bot.models import OptionPosition

# ───────────────────────── config sanity ────────────────────────────────────


def test_option_multiplier_is_100() -> None:
    assert config.OPTION_MULTIPLIER == 100


def test_delta_band_config() -> None:
    assert config.TARGET_DELTA_LOW == 0.65
    assert config.TARGET_DELTA_HIGH == 0.75
    assert config.UNDERSIZED_DELTA_FLOOR == 0.50
    assert config.MIN_DTE == 14


# ───────────────────────── OptionContract ───────────────────────────────────


def test_option_contract_spread_pct() -> None:
    c = OptionContract(
        symbol="AAPL260116C00150000", underlying="AAPL", option_type="call",
        strike=150.0, expiry="2026-01-16", bid=4.9, ask=5.1, mid=5.0,
    )
    assert c.spread_pct == pytest.approx(4.0)     # (5.1 - 4.9) / 5.0 * 100


def test_option_contract_spread_pct_none_when_missing_quotes() -> None:
    c = OptionContract(
        symbol="X", underlying="AAPL", option_type="call", strike=150.0,
        expiry="2026-01-16",
    )
    assert c.spread_pct is None


# ───────────────────────── OCC symbol encoding ──────────────────────────────


def test_occ_symbol_encoding() -> None:
    assert options.occ_symbol("AAPL", "2026-01-16", "call", 150.0) == "AAPL260116C00150000"
    assert options.occ_symbol("SPY", "2026-03-20", "put", 7.5) == "SPY260320P00007500"


def test_occ_symbol_rejects_bad_type() -> None:
    with pytest.raises(ValueError, match="invalid option_type"):
        options.occ_symbol("AAPL", "2026-01-16", "straddle", 150.0)


def test_parse_occ_symbol_round_trip() -> None:
    sym = options.occ_symbol("AAPL", "2026-01-16", "call", 150.0)
    assert options.parse_occ_symbol(sym) == ("AAPL", "2026-01-16", "call", 150.0)
    sym2 = options.occ_symbol("SPY", "2026-03-20", "put", 7.5)
    assert options.parse_occ_symbol(sym2) == ("SPY", "2026-03-20", "put", 7.5)


def test_parse_occ_symbol_rejects_garbage() -> None:
    assert options.parse_occ_symbol("too-short") is None
    assert options.parse_occ_symbol("AAPL260116X00150000") is None   # bad C/P
    assert options.parse_occ_symbol("AAPL2601AAC00150000") is None   # non-numeric date


# ───────────────────────── options client (mocked network) ──────────────────


class _FakeGetResp:
    def __init__(self, status_code: int, payload: object) -> None:
        self.status_code = status_code
        self._payload = payload
        self.content = b"x" if payload is not None else b""

    def json(self) -> object:
        return self._payload


class _NonJsonGetResp(_FakeGetResp):
    def __init__(self, status_code: int = 200) -> None:
        super().__init__(status_code, object())

    def json(self) -> object:
        raise ValueError("not JSON")


def _with_opt_creds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "trading_bot.broker.options.secrets.get_secret", lambda _n: "key",
    )


_CONTRACTS_BODY = {
    "option_contracts": [
        {
            "symbol": "AAPL260116C00150000", "underlying_symbol": "AAPL",
            "type": "call", "strike_price": "150", "expiration_date": "2026-01-16",
            "open_interest": "500",
        },
    ],
}
_SNAPSHOTS_BODY = {
    "snapshots": {
        "AAPL260116C00150000": {
            "greeks": {"delta": 0.7, "gamma": 0.01, "theta": -0.05, "vega": 0.1},
            "latestQuote": {"bp": 4.9, "ap": 5.1},
        },
    },
}


def _contract_row(
    index: int, *, open_interest: str = "500",
) -> dict[str, str]:
    return {
        "symbol": f"AAPL260116C{index:08d}",
        "underlying_symbol": "AAPL",
        "type": "call",
        "strike_price": str(index / 1000),
        "expiration_date": "2026-01-16",
        "open_interest": open_interest,
    }


def _snapshot_row(
    *, delta: float = 0.7, bid: float = 4.9, ask: float = 5.1,
) -> dict[str, dict[str, float]]:
    return {
        "greeks": {
            "delta": delta,
            "gamma": 0.01,
            "theta": -0.05,
            "vega": 0.1,
        },
        "latestQuote": {"bp": bid, "ap": ask},
    }


def test_options_client_construction_is_paper_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _with_opt_creds(monkeypatch)
    options.AlpacaOptionsClient()   # default paper base is fine
    with pytest.raises(ValueError, match="PAPER-ONLY"):
        options.AlpacaOptionsClient(trading_base_url="https://api.alpaca.markets")


def test_list_option_contracts_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_opt_creds(monkeypatch)
    monkeypatch.setattr(
        "trading_bot.broker.options.requests.get",
        lambda *a, **k: _FakeGetResp(200, _CONTRACTS_BODY),
    )
    contracts = options.AlpacaOptionsClient().list_option_contracts("AAPL")
    assert len(contracts) == 1
    c = contracts[0]
    assert c.symbol == "AAPL260116C00150000"
    assert c.strike == 150.0 and c.expiry == "2026-01-16"
    assert c.open_interest == 500
    assert c.delta is None            # greeks not populated by the metadata call


def test_list_option_contracts_paginates_more_than_100_and_propagates_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _with_opt_creds(monkeypatch)
    captured_params: list[dict[str, object]] = []
    responses = iter([
        _FakeGetResp(200, {
            "option_contracts": [_contract_row(i) for i in range(100)],
            "next_page_token": "contracts-page-2",
        }),
        _FakeGetResp(200, {
            "option_contracts": [_contract_row(i) for i in range(100, 150)],
            "next_page_token": None,
        }),
    ])

    def route(
        url: str,
        *,
        headers: object,
        params: dict[str, object],
        timeout: object,
    ) -> _FakeGetResp:
        assert "/v2/options/contracts" in url
        captured_params.append(dict(params))
        return next(responses)

    monkeypatch.setattr("trading_bot.broker.options.requests.get", route)
    contracts = options.AlpacaOptionsClient().list_option_contracts(
        "AAPL",
        expiration_gte="2026-01-01",
        expiration_lte="2026-02-01",
        option_type="call",
    )

    assert len(contracts) == 150
    assert contracts[0].symbol == "AAPL260116C00000000"
    assert contracts[-1].symbol == "AAPL260116C00000149"
    assert "page_token" not in captured_params[0]
    assert captured_params[1]["page_token"] == "contracts-page-2"
    assert captured_params[1]["expiration_date_gte"] == "2026-01-01"
    assert captured_params[1]["expiration_date_lte"] == "2026-02-01"
    assert captured_params[1]["type"] == "call"
    assert captured_params[1]["limit"] == 100


def test_list_option_contracts_dedupes_first_seen_symbol_across_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _with_opt_creds(monkeypatch)
    responses = iter([
        _FakeGetResp(200, {
            "option_contracts": [_contract_row(1, open_interest="100")],
            "next_page_token": "next",
        }),
        _FakeGetResp(200, {
            "option_contracts": [
                _contract_row(1, open_interest="999"),
                _contract_row(2),
            ],
        }),
    ])
    monkeypatch.setattr(
        "trading_bot.broker.options.requests.get",
        lambda *args, **kwargs: next(responses),
    )

    contracts = options.AlpacaOptionsClient().list_option_contracts("AAPL")

    assert [contract.symbol for contract in contracts] == [
        "AAPL260116C00000001",
        "AAPL260116C00000002",
    ]
    assert contracts[0].open_interest == 100


def test_list_option_contracts_accepts_empty_final_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _with_opt_creds(monkeypatch)
    calls = 0
    responses = iter([
        _FakeGetResp(200, {
            "option_contracts": [_contract_row(1)],
            "next_page_token": "empty-final-page",
        }),
        _FakeGetResp(200, {
            "option_contracts": [],
            "next_page_token": None,
        }),
    ])

    def route(*args: object, **kwargs: object) -> _FakeGetResp:
        nonlocal calls
        calls += 1
        return next(responses)

    monkeypatch.setattr("trading_bot.broker.options.requests.get", route)

    contracts = options.AlpacaOptionsClient().list_option_contracts("AAPL")

    assert calls == 2
    assert [contract.symbol for contract in contracts] == [
        "AAPL260116C00000001",
    ]


def test_get_option_chain_rejects_repeated_contract_page_token(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    _with_opt_creds(monkeypatch)
    calls = 0
    responses = iter([
        _FakeGetResp(200, {
            "option_contracts": [_contract_row(1)],
            "next_page_token": "repeated",
        }),
        _FakeGetResp(200, {
            "option_contracts": [_contract_row(2)],
            "next_page_token": "repeated",
        }),
    ])

    def route(*args: object, **kwargs: object) -> _FakeGetResp:
        nonlocal calls
        calls += 1
        return next(responses)

    monkeypatch.setattr("trading_bot.broker.options.requests.get", route)

    chain = options.AlpacaOptionsClient().get_option_chain("AAPL")

    assert calls == 2
    assert chain.ok is False
    assert chain.contracts == []
    assert "repeated page token" in chain.reason
    assert "2 unique contracts discarded" in chain.reason
    assert chain.reason in capsys.readouterr().err


def test_get_option_chain_rejects_contract_page_guard_exhaustion(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    _with_opt_creds(monkeypatch)
    monkeypatch.setattr(options, "_MAX_CONTRACT_PAGES", 2)
    calls = 0

    def route(*args: object, **kwargs: object) -> _FakeGetResp:
        nonlocal calls
        calls += 1
        return _FakeGetResp(200, {
            "option_contracts": [_contract_row(calls)],
            "next_page_token": f"page-{calls + 1}",
        })

    monkeypatch.setattr("trading_bot.broker.options.requests.get", route)

    chain = options.AlpacaOptionsClient().get_option_chain("AAPL")

    assert calls == 2
    assert chain.ok is False
    assert chain.contracts == []
    assert "exceeded maximum of 2 pages" in chain.reason
    assert "2 unique contracts discarded" in chain.reason
    assert chain.reason in capsys.readouterr().err


def test_get_option_chain_rejects_later_contract_page_http_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    _with_opt_creds(monkeypatch)
    responses = iter([
        _FakeGetResp(200, {
            "option_contracts": [_contract_row(1)],
            "next_page_token": "page-2",
        }),
        _FakeGetResp(503, {"message": "unavailable"}),
    ])
    monkeypatch.setattr(
        "trading_bot.broker.options.requests.get",
        lambda *args, **kwargs: next(responses),
    )

    chain = options.AlpacaOptionsClient().get_option_chain("AAPL")

    assert chain.ok is False
    assert chain.contracts == []
    assert "pagination incomplete" in chain.reason
    assert "page 2 after 1 unique contracts" in chain.reason
    assert "HTTP 503" in chain.reason
    assert chain.reason in capsys.readouterr().err


def test_get_option_chain_rejects_later_contract_page_malformed_2xx(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    _with_opt_creds(monkeypatch)
    responses = iter([
        _FakeGetResp(200, {
            "option_contracts": [_contract_row(1)],
            "next_page_token": "page-2",
        }),
        _FakeGetResp(200, {"next_page_token": None}),
    ])
    monkeypatch.setattr(
        "trading_bot.broker.options.requests.get",
        lambda *args, **kwargs: next(responses),
    )

    chain = options.AlpacaOptionsClient().get_option_chain("AAPL")

    assert chain.ok is False
    assert chain.contracts == []
    assert "malformed contracts response" in chain.reason
    assert "missing 'option_contracts' list" in chain.reason
    assert "page 2 after 1 unique contracts" in chain.reason
    assert chain.reason in capsys.readouterr().err


def test_list_option_contracts_failsoft(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    _with_opt_creds(monkeypatch)
    monkeypatch.setattr(
        "trading_bot.broker.options.requests.get",
        lambda *a, **k: _FakeGetResp(403, {"message": "options not enabled"}),
    )
    assert options.AlpacaOptionsClient().list_option_contracts("AAPL") == []
    assert "contracts unavailable" in capsys.readouterr().err


def test_list_option_contracts_no_creds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "trading_bot.broker.options.secrets.get_secret", lambda _n: None,
    )
    assert options.AlpacaOptionsClient().list_option_contracts("AAPL") == []


def _route(url: str, headers: object = None, params: object = None,
           timeout: object = None) -> _FakeGetResp:
    if "/v2/options/contracts" in url:
        return _FakeGetResp(200, _CONTRACTS_BODY)
    if "/snapshots/" in url:
        return _FakeGetResp(200, _SNAPSHOTS_BODY)
    return _FakeGetResp(404, {})


def test_get_option_chain_merges_greeks(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_opt_creds(monkeypatch)
    monkeypatch.setattr("trading_bot.broker.options.requests.get", _route)
    chain = options.AlpacaOptionsClient().get_option_chain("AAPL")
    assert chain.ok is True
    c = chain.contracts[0]
    assert c.delta == 0.7
    assert c.bid == 4.9 and c.ask == 5.1 and c.mid == pytest.approx(5.0)
    assert c.open_interest == 500     # metadata preserved through the merge


def test_get_option_chain_paginates_more_than_1000_snapshots_and_merges_page_two(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _with_opt_creds(monkeypatch)
    captured_params: list[dict[str, object]] = []
    snapshot_responses = iter([
        _FakeGetResp(200, {
            "snapshots": {
                f"AAPL260116C{i:08d}": _snapshot_row(delta=0.61)
                for i in range(1000)
            },
            "next_page_token": "snapshots-page-2",
        }),
        _FakeGetResp(200, {
            "snapshots": {
                "AAPL260116C00001000": _snapshot_row(
                    delta=0.72, bid=7.8, ask=8.2,
                ),
            },
            "next_page_token": None,
        }),
    ])

    def route(
        url: str,
        *,
        headers: object,
        params: dict[str, object],
        timeout: object,
    ) -> _FakeGetResp:
        if "/v2/options/contracts" in url:
            return _FakeGetResp(200, {
                "option_contracts": [_contract_row(1), _contract_row(1000)],
            })
        captured_params.append(dict(params))
        return next(snapshot_responses)

    monkeypatch.setattr("trading_bot.broker.options.requests.get", route)

    chain = options.AlpacaOptionsClient().get_option_chain("AAPL")

    assert chain.ok is True
    assert len(chain.contracts) == 2
    by_symbol = {contract.symbol: contract for contract in chain.contracts}
    assert by_symbol["AAPL260116C00000001"].delta == 0.61
    page_two = by_symbol["AAPL260116C00001000"]
    assert page_two.delta == 0.72
    assert page_two.bid == 7.8 and page_two.ask == 8.2
    assert page_two.mid == pytest.approx(8.0)
    assert "page_token" not in captured_params[0]
    assert captured_params[1]["page_token"] == "snapshots-page-2"
    assert captured_params[1]["limit"] == 1000


def test_fetch_snapshots_dedupes_first_seen_symbol_across_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _with_opt_creds(monkeypatch)
    responses = iter([
        _FakeGetResp(200, {
            "snapshots": {
                "AAPL260116C00000001": _snapshot_row(delta=0.61),
            },
            "next_page_token": "next",
        }),
        _FakeGetResp(200, {
            "snapshots": {
                "AAPL260116C00000001": _snapshot_row(delta=0.99),
                "AAPL260116C00000002": _snapshot_row(delta=0.72),
            },
        }),
    ])
    monkeypatch.setattr(
        "trading_bot.broker.options.requests.get",
        lambda *args, **kwargs: next(responses),
    )

    snapshots, reason = (
        options.AlpacaOptionsClient()._fetch_snapshots_result("AAPL")
    )

    assert reason == ""
    assert list(snapshots) == [
        "AAPL260116C00000001",
        "AAPL260116C00000002",
    ]
    assert snapshots["AAPL260116C00000001"]["greeks"]["delta"] == 0.61


def test_fetch_snapshots_accepts_empty_final_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _with_opt_creds(monkeypatch)
    calls = 0
    responses = iter([
        _FakeGetResp(200, {
            "snapshots": {
                "AAPL260116C00000001": _snapshot_row(),
            },
            "next_page_token": "empty-final-page",
        }),
        _FakeGetResp(200, {"snapshots": {}, "next_page_token": None}),
    ])

    def route(*args: object, **kwargs: object) -> _FakeGetResp:
        nonlocal calls
        calls += 1
        return next(responses)

    monkeypatch.setattr("trading_bot.broker.options.requests.get", route)

    snapshots, reason = (
        options.AlpacaOptionsClient()._fetch_snapshots_result("AAPL")
    )

    assert calls == 2
    assert reason == ""
    assert list(snapshots) == ["AAPL260116C00000001"]


def test_get_option_chain_rejects_repeated_snapshot_page_token(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    _with_opt_creds(monkeypatch)
    snapshot_calls = 0
    snapshot_responses = iter([
        _FakeGetResp(200, {
            "snapshots": {
                "AAPL260116C00000001": _snapshot_row(),
            },
            "next_page_token": "repeated",
        }),
        _FakeGetResp(200, {
            "snapshots": {
                "AAPL260116C00000002": _snapshot_row(),
            },
            "next_page_token": "repeated",
        }),
    ])

    def route(url: str, **kwargs: object) -> _FakeGetResp:
        nonlocal snapshot_calls
        if "/v2/options/contracts" in url:
            return _FakeGetResp(200, _CONTRACTS_BODY)
        snapshot_calls += 1
        return next(snapshot_responses)

    monkeypatch.setattr("trading_bot.broker.options.requests.get", route)

    chain = options.AlpacaOptionsClient().get_option_chain("AAPL")

    assert snapshot_calls == 2
    assert chain.ok is False
    assert chain.contracts == []
    assert "repeated page token" in chain.reason
    assert "2 unique snapshots discarded" in chain.reason
    assert chain.reason in capsys.readouterr().err


def test_get_option_chain_rejects_snapshot_page_guard_exhaustion(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    _with_opt_creds(monkeypatch)
    monkeypatch.setattr(options, "_MAX_SNAPSHOT_PAGES", 2)
    snapshot_calls = 0

    def route(url: str, **kwargs: object) -> _FakeGetResp:
        nonlocal snapshot_calls
        if "/v2/options/contracts" in url:
            return _FakeGetResp(200, _CONTRACTS_BODY)
        snapshot_calls += 1
        return _FakeGetResp(200, {
            "snapshots": {
                f"AAPL260116C{snapshot_calls:08d}": _snapshot_row(),
            },
            "next_page_token": f"page-{snapshot_calls + 1}",
        })

    monkeypatch.setattr("trading_bot.broker.options.requests.get", route)

    chain = options.AlpacaOptionsClient().get_option_chain("AAPL")

    assert snapshot_calls == 2
    assert chain.ok is False
    assert chain.contracts == []
    assert "exceeded maximum of 2 pages" in chain.reason
    assert "2 unique snapshots discarded" in chain.reason
    assert chain.reason in capsys.readouterr().err


def test_get_option_chain_rejects_later_snapshot_page_http_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    _with_opt_creds(monkeypatch)
    snapshot_responses = iter([
        _FakeGetResp(200, {
            "snapshots": {
                "AAPL260116C00150000": _snapshot_row(),
            },
            "next_page_token": "page-2",
        }),
        _FakeGetResp(503, {"message": "unavailable"}),
    ])

    def route(url: str, **kwargs: object) -> _FakeGetResp:
        if "/v2/options/contracts" in url:
            return _FakeGetResp(200, _CONTRACTS_BODY)
        return next(snapshot_responses)

    monkeypatch.setattr("trading_bot.broker.options.requests.get", route)

    chain = options.AlpacaOptionsClient().get_option_chain("AAPL")

    assert chain.ok is False
    assert chain.contracts == []
    assert "snapshots pagination incomplete" in chain.reason
    assert "page 2 after 1 unique snapshots" in chain.reason
    assert "HTTP 503" in chain.reason
    assert chain.reason in capsys.readouterr().err


def test_get_option_chain_rejects_later_snapshot_page_malformed_2xx(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    _with_opt_creds(monkeypatch)
    snapshot_responses = iter([
        _FakeGetResp(200, {
            "snapshots": {
                "AAPL260116C00150000": _snapshot_row(),
            },
            "next_page_token": "page-2",
        }),
        _FakeGetResp(200, {"next_page_token": None}),
    ])

    def route(url: str, **kwargs: object) -> _FakeGetResp:
        if "/v2/options/contracts" in url:
            return _FakeGetResp(200, _CONTRACTS_BODY)
        return next(snapshot_responses)

    monkeypatch.setattr("trading_bot.broker.options.requests.get", route)

    chain = options.AlpacaOptionsClient().get_option_chain("AAPL")

    assert chain.ok is False
    assert chain.contracts == []
    assert "malformed snapshots response" in chain.reason
    assert "missing 'snapshots' object" in chain.reason
    assert "page 2 after 1 unique snapshots" in chain.reason
    assert chain.reason in capsys.readouterr().err


def test_get_option_chain_no_contracts_is_not_ok(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _with_opt_creds(monkeypatch)
    monkeypatch.setattr(
        "trading_bot.broker.options.requests.get",
        lambda *a, **k: _FakeGetResp(403, {}),
    )
    chain = options.AlpacaOptionsClient().get_option_chain("AAPL")
    assert chain.ok is False
    assert chain.contracts == []


@pytest.mark.parametrize(
    ("response", "detail"),
    [
        (_NonJsonGetResp(), "non-JSON response body"),
        (_FakeGetResp(200, []), "top-level body must be an object"),
        (
            _FakeGetResp(200, {"option_contracts": {}}),
            "missing 'option_contracts' list",
        ),
        (
            _FakeGetResp(200, {
                "option_contracts": [], "next_page_token": 123,
            }),
            "'next_page_token' must be a string or null",
        ),
    ],
    ids=(
        "non-json", "non-object", "wrong-collection-shape", "wrong-token-shape",
    ),
)
def test_get_option_chain_rejects_malformed_contracts_2xx(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    response: _FakeGetResp,
    detail: str,
) -> None:
    _with_opt_creds(monkeypatch)
    monkeypatch.setattr(
        "trading_bot.broker.options.requests.get",
        lambda *a, **k: response,
    )

    chain = options.AlpacaOptionsClient().get_option_chain("AAPL")

    assert chain.ok is False
    assert chain.contracts == []
    assert "malformed contracts response" in chain.reason
    assert detail in chain.reason
    assert chain.reason in capsys.readouterr().err


@pytest.mark.parametrize(
    ("row", "detail"),
    [
        (None, "item 0 must be an object"),
        ({**_CONTRACTS_BODY["option_contracts"][0], "symbol": ""},
         "'symbol' must be a non-empty string"),
        ({**_CONTRACTS_BODY["option_contracts"][0], "underlying_symbol": None},
         "'underlying_symbol' must be a non-empty string"),
        ({**_CONTRACTS_BODY["option_contracts"][0], "type": ["call"]},
         "'type' must be 'call' or 'put'"),
        ({**_CONTRACTS_BODY["option_contracts"][0], "strike_price": "bad"},
         "'strike_price' must be a finite number"),
        ({**_CONTRACTS_BODY["option_contracts"][0], "expiration_date": "2026-99-99"},
         "'expiration_date' must be an ISO date string"),
    ],
    ids=(
        "non-object", "missing-identity", "missing-underlying", "invalid-type",
        "invalid-strike", "invalid-expiry",
    ),
)
def test_get_option_chain_rejects_malformed_contract_items(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    row: object,
    detail: str,
) -> None:
    _with_opt_creds(monkeypatch)
    monkeypatch.setattr(
        "trading_bot.broker.options.requests.get",
        lambda *a, **k: _FakeGetResp(200, {"option_contracts": [row]}),
    )

    chain = options.AlpacaOptionsClient().get_option_chain("AAPL")

    assert chain.ok is False
    assert chain.contracts == []
    assert "malformed contracts response" in chain.reason
    assert detail in chain.reason
    assert chain.reason in capsys.readouterr().err


def test_get_option_chain_preserves_valid_empty_contracts_response(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _with_opt_creds(monkeypatch)
    monkeypatch.setattr(
        "trading_bot.broker.options.requests.get",
        lambda *a, **k: _FakeGetResp(200, {"option_contracts": []}),
    )

    chain = options.AlpacaOptionsClient().get_option_chain("AAPL")

    assert chain.ok is False
    assert chain.reason == "no contracts listed for AAPL"
    assert "malformed" not in capsys.readouterr().err


@pytest.mark.parametrize(
    ("response", "detail"),
    [
        (_NonJsonGetResp(), "non-JSON response body"),
        (_FakeGetResp(200, []), "top-level body must be an object"),
        (
            _FakeGetResp(200, {"snapshots": []}),
            "missing 'snapshots' object",
        ),
        (
            _FakeGetResp(200, {"snapshots": {}, "next_page_token": 123}),
            "'next_page_token' must be a string or null",
        ),
    ],
    ids=(
        "non-json", "non-object", "wrong-collection-shape", "wrong-token-shape",
    ),
)
def test_get_option_chain_rejects_malformed_snapshots_2xx(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    response: _FakeGetResp,
    detail: str,
) -> None:
    _with_opt_creds(monkeypatch)

    def route(url: str, **kwargs: object) -> _FakeGetResp:
        if "/v2/options/contracts" in url:
            return _FakeGetResp(200, _CONTRACTS_BODY)
        return response

    monkeypatch.setattr("trading_bot.broker.options.requests.get", route)

    chain = options.AlpacaOptionsClient().get_option_chain("AAPL")

    assert chain.ok is False
    assert chain.contracts == []
    assert "malformed snapshots response" in chain.reason
    assert detail in chain.reason
    assert chain.reason in capsys.readouterr().err


@pytest.mark.parametrize("snapshot", [None, [], "not-an-object"])
def test_get_option_chain_rejects_non_object_snapshot_values(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    snapshot: object,
) -> None:
    _with_opt_creds(monkeypatch)

    def route(url: str, **kwargs: object) -> _FakeGetResp:
        if "/v2/options/contracts" in url:
            return _FakeGetResp(200, _CONTRACTS_BODY)
        return _FakeGetResp(200, {
            "snapshots": {"AAPL260116C00150000": snapshot},
        })

    monkeypatch.setattr("trading_bot.broker.options.requests.get", route)

    chain = options.AlpacaOptionsClient().get_option_chain("AAPL")

    assert chain.ok is False
    assert chain.contracts == []
    assert "malformed snapshots response" in chain.reason
    assert "must be an object" in chain.reason
    assert chain.reason in capsys.readouterr().err


def test_get_option_chain_accepts_snapshot_without_optional_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _with_opt_creds(monkeypatch)

    def route(url: str, **kwargs: object) -> _FakeGetResp:
        if "/v2/options/contracts" in url:
            return _FakeGetResp(200, _CONTRACTS_BODY)
        return _FakeGetResp(200, {
            "snapshots": {"AAPL260116C00150000": {}},
        })

    monkeypatch.setattr("trading_bot.broker.options.requests.get", route)

    chain = options.AlpacaOptionsClient().get_option_chain("AAPL")

    assert chain.ok is True
    assert len(chain.contracts) == 1
    assert chain.contracts[0].delta is None
    assert chain.contracts[0].bid is None


def test_get_option_chain_preserves_valid_empty_snapshots_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _with_opt_creds(monkeypatch)

    def route(url: str, **kwargs: object) -> _FakeGetResp:
        if "/v2/options/contracts" in url:
            return _FakeGetResp(200, _CONTRACTS_BODY)
        return _FakeGetResp(200, {"snapshots": {}})

    monkeypatch.setattr("trading_bot.broker.options.requests.get", route)

    chain = options.AlpacaOptionsClient().get_option_chain("AAPL")

    assert chain.ok is True
    assert len(chain.contracts) == 1
    assert chain.contracts[0].delta is None


def test_get_option_chain_snapshot_failure_keeps_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _with_opt_creds(monkeypatch)

    def route(url: str, **k: object) -> _FakeGetResp:
        if "/v2/options/contracts" in url:
            return _FakeGetResp(200, _CONTRACTS_BODY)
        return _FakeGetResp(500, {})      # snapshots down

    monkeypatch.setattr("trading_bot.broker.options.requests.get", route)
    chain = options.AlpacaOptionsClient().get_option_chain("AAPL")
    assert chain.ok is True
    assert chain.contracts[0].delta is None   # greeks absent, contract still listed


# ───────────────────────── account options level (broker) ───────────────────


def test_account_parses_options_trading_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "trading_bot.broker.alpaca.secrets.get_secret", lambda _n: "key",
    )

    class _R:
        status_code = 200
        content = b"x"

        def json(self) -> object:
            return {
                "account_number": "PA123", "currency": "USD",
                "options_trading_level": 2, "cash": "1000", "equity": "1000",
                "buying_power": "1000", "status": "ACTIVE",
            }

    monkeypatch.setattr(
        "trading_bot.broker.alpaca.requests.request", lambda *a, **k: _R(),
    )
    from trading_bot.broker.alpaca import AlpacaBroker
    assert AlpacaBroker().get_account().options_trading_level == 2


# ───────────────────────── options CLI (chain / positions) ──────────────────


class _FakeChainClient:
    def __init__(self, result: OptionChainResult) -> None:
        self._result = result

    def get_option_chain(self, _underlying: str) -> OptionChainResult:
        return self._result


def test_cli_options_chain(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_bot import __main__ as m
    result = OptionChainResult(ok=True, contracts=[OptionContract(
        symbol="AAPL270115C00150000", underlying="AAPL", option_type="call",
        strike=150.0, expiry="2027-01-15", delta=0.70, open_interest=500,
        bid=4.9, ask=5.1, mid=5.0,
    )])
    monkeypatch.setattr(m.broker, "AlpacaOptionsClient", lambda: _FakeChainClient(result))
    m.cmd_options_chain("AAPL")
    out = capsys.readouterr().out
    assert "OPTIONS CHAIN" in out
    assert "AAPL270115C00150000" in out


def test_cli_options_chain_unavailable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_bot import __main__ as m
    monkeypatch.setattr(
        m.broker, "AlpacaOptionsClient",
        lambda: _FakeChainClient(OptionChainResult(ok=False, reason="not enabled")),
    )
    m.cmd_options_chain("AAPL")
    assert "unavailable" in capsys.readouterr().out


def test_cli_options_chain_propagates_malformed_upstream_reason(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_bot import __main__ as m
    reason = "malformed snapshots response: non-JSON response body"
    monkeypatch.setattr(
        m.broker,
        "AlpacaOptionsClient",
        lambda: _FakeChainClient(OptionChainResult(ok=False, reason=reason)),
    )

    m.cmd_options_chain("AAPL")

    assert f"unavailable: {reason}" in capsys.readouterr().out


def test_cli_options_positions_empty(
    tmp_db: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_bot import __main__ as m
    m.cmd_options_positions()
    assert "(none)" in capsys.readouterr().out


def test_cli_options_positions_with_pnl(
    monkeypatch: pytest.MonkeyPatch, tmp_db: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_bot import __main__ as m
    db.insert_option_position(OptionPosition(
        symbol="AAPL270115C00150000", underlying="AAPL", option_type="call",
        strike=150.0, expiry="2027-01-15", contracts=2.0,
        opened_at=datetime(2026, 1, 1, tzinfo=UTC), premium_entry=5.0,
        tp=160.0, sl=140.0, multiplier=100, outcome="open",
    ))
    result = OptionChainResult(ok=True, contracts=[OptionContract(
        symbol="AAPL270115C00150000", underlying="AAPL", option_type="call",
        strike=150.0, expiry="2027-01-15", mid=8.0,
    )])
    monkeypatch.setattr(m.broker, "AlpacaOptionsClient", lambda: _FakeChainClient(result))
    m.cmd_options_positions()
    out = capsys.readouterr().out
    assert "AAPL270115C00150000" in out
    assert "unrealized P&L" in out
    assert "600" in out              # (8 - 5) * 100 * 2 current P&L
