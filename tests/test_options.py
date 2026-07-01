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
