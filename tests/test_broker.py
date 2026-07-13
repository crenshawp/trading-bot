"""Tests for the broker execution interface (Phase 11).

All Alpaca network calls are mocked — no live or paper network is touched by the
suite. The fake in-memory broker exercises the abstract interface; the Alpaca
adapter is tested against monkeypatched ``requests`` responses.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from trading_bot import broker, db
from trading_bot.broker import alpaca, base
from trading_bot.broker.fake import FakeBroker
from trading_bot.models import LongTermPosition, Signal, Trade


def _seed_open_active(ticker: str, *, track_mode: str = "active") -> None:
    """Seed one OPEN signal-tracking trade for ``ticker`` (yfinance-resolved;
    never routed to a broker — OUT of reconciliation's scope since Phase 18)."""
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    sid = db.insert_signal(Signal(
        timestamp=ts, ticker=ticker, asset_class="stock",
        signal_type="ema21_pullback", direction="call", entry_price=100.0,
    ))
    db.insert_trade(Trade(
        signal_id=sid, opened_at=ts, outcome="open", track_mode=track_mode,
    ))


def _seed_broker_tracked(ticker: str, qty: float = 10.0) -> None:
    """Seed one OPEN broker-tracked position (a Phase 16 long-term entry —
    rows in this table exist only after a broker-accepted order)."""
    db.insert_long_term_position(LongTermPosition(
        ticker=ticker, asset_class="stock", entry_price=100.0,
        entry_date=datetime(2026, 1, 1, tzinfo=UTC), qty=qty, status="open",
    ))


def _with_creds(
    monkeypatch: pytest.MonkeyPatch, *, key: str = "k", secret: str = "s",
) -> None:
    """Patch the secrets layer so AlpacaBroker loads paper creds (no keyring)."""
    creds = {"ALPACA_API_KEY": key, "ALPACA_SECRET_KEY": secret}
    monkeypatch.setattr(
        "trading_bot.broker.alpaca.secrets.get_secret", lambda name: creds.get(name),
    )


class _FakeResp:
    """A minimal stand-in for a ``requests`` Response (mocks the network)."""

    def __init__(
        self, status_code: int, payload: object, *, raise_json: bool = False,
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.content = b"x" if payload is not None else b""
        self._raise_json = raise_json

    def json(self) -> object:
        if self._raise_json:
            raise ValueError("bad json")
        return self._payload


def _patch_request(
    monkeypatch: pytest.MonkeyPatch,
    resp: _FakeResp | None = None,
    *,
    raises: Exception | None = None,
    capture: list[dict[str, object]] | None = None,
) -> None:
    """Patch ``requests.request`` in the Alpaca adapter. No network is touched."""

    def fake(method: str, url: str, **kw: object) -> _FakeResp:
        if capture is not None:
            capture.append({"method": method, "url": url, **kw})
        if raises is not None:
            raise raises
        assert resp is not None
        return resp

    monkeypatch.setattr("trading_bot.broker.alpaca.requests.request", fake)

# ───────────────────────── neutral types + interface ────────────────────────


def test_fake_broker_is_a_broker() -> None:
    assert isinstance(FakeBroker(), base.Broker)


def test_neutral_result_defaults_are_failsoft() -> None:
    # Constructed with no args, every result type is a safe 'unavailable'.
    assert base.AccountInfo().ok is False
    assert base.PositionsResult().ok is False and base.PositionsResult().positions == []
    assert base.OrdersResult().ok is False and base.OrdersResult().orders == []
    o = base.OrderResult()
    assert o.ok is False and o.status == base.STATUS_UNKNOWN and o.filled_qty == 0.0


def test_status_sets_are_coherent() -> None:
    assert base.STATUS_REJECTED in base.TERMINAL_STATUSES
    assert base.STATUS_CANCELED in base.TERMINAL_STATUSES
    assert base.STATUS_FILLED not in base.TERMINAL_STATUSES
    assert base.TERMINAL_STATUSES <= base.NEUTRAL_STATUSES


# ───────────────────────── fake account / positions ─────────────────────────


def test_fake_account_ok() -> None:
    acct = FakeBroker(buying_power=42_000.0).get_account()
    assert acct.ok is True
    assert acct.buying_power == 42_000.0
    assert acct.status == "ACTIVE"


def test_fake_account_failsoft() -> None:
    acct = FakeBroker(fail=True).get_account()
    assert acct.ok is False
    assert "outage" in acct.reason


def test_fake_positions_ok_and_failsoft() -> None:
    b = FakeBroker()
    b.set_position("AAPL", 10, avg_entry_price=190.0)
    res = b.get_positions()
    assert res.ok is True
    assert res.positions[0].symbol == "AAPL"
    assert res.positions[0].qty == 10

    down = FakeBroker(fail=True).get_positions()
    assert down.ok is False
    assert down.positions == []


# ───────────────────────── fake order lifecycle ─────────────────────────────


def test_fake_submit_defaults_to_limit_and_rests_new() -> None:
    b = FakeBroker()
    res = b.submit_order("MSFT", 5, broker.SIDE_BUY, limit_price=400.0)
    assert res.ok is True
    assert res.order_type == broker.ORDER_TYPE_LIMIT   # LIMIT is the default
    assert res.status == broker.STATUS_NEW
    assert res.filled_qty == 0.0
    assert res.order_id is not None


def test_fake_submit_auto_fill_marks_filled_and_creates_position() -> None:
    b = FakeBroker(auto_fill=True)
    res = b.submit_order("MSFT", 5, broker.SIDE_BUY, limit_price=400.0)
    assert res.status == broker.STATUS_FILLED
    assert res.filled_qty == 5
    assert b.get_positions().positions[0].symbol == "MSFT"


def test_fake_submit_rejections_are_structured() -> None:
    b = FakeBroker()
    bad_side = b.submit_order("MSFT", 5, "hodl", limit_price=400.0)
    assert bad_side.ok is False and bad_side.status == broker.STATUS_REJECTED
    assert "invalid side" in bad_side.reason

    bad_type = b.submit_order("MSFT", 5, broker.SIDE_BUY, order_type="stop")
    assert bad_type.status == broker.STATUS_REJECTED

    bad_tif = b.submit_order("MSFT", 5, broker.SIDE_BUY, time_in_force="fok")
    assert bad_tif.status == broker.STATUS_REJECTED


def test_fake_submit_business_rejection_reason() -> None:
    b = FakeBroker(reject_reason="insufficient buying power")
    res = b.submit_order("MSFT", 5, broker.SIDE_BUY, limit_price=400.0)
    assert res.ok is False
    assert res.status == broker.STATUS_REJECTED
    assert res.reason == "insufficient buying power"


def test_fake_submit_failsoft_is_error_not_rejected() -> None:
    res = FakeBroker(fail=True).submit_order("MSFT", 5, broker.SIDE_BUY)
    assert res.ok is False
    assert res.status == broker.STATUS_ERROR   # transport error, not a rejection


def test_fake_get_and_cancel_order() -> None:
    b = FakeBroker()
    submitted = b.submit_order("MSFT", 5, broker.SIDE_BUY, limit_price=400.0)
    assert submitted.order_id is not None
    fetched = b.get_order(submitted.order_id)
    assert fetched.ok is True and fetched.order_id == submitted.order_id

    canceled = b.cancel_order(submitted.order_id)
    assert canceled.ok is True and canceled.status == broker.STATUS_CANCELED
    # After cancel the order is no longer 'open'.
    assert b.list_orders("open").orders == []


def test_fake_get_order_not_found_and_failsoft() -> None:
    b = FakeBroker()
    assert b.get_order("nope").ok is False
    assert FakeBroker(fail=True).get_order("x").status == broker.STATUS_ERROR
    assert FakeBroker(fail=True).cancel_order("x").status == broker.STATUS_ERROR
    assert b.cancel_order("nope").ok is False


def test_fake_list_orders_filters() -> None:
    b = FakeBroker()
    b.submit_order("MSFT", 5, broker.SIDE_BUY, limit_price=400.0)
    b.submit_order("AAPL", 3, broker.SIDE_BUY, limit_price=190.0)
    assert len(b.list_orders("open").orders) == 2
    assert len(b.list_orders("all").orders) == 2
    assert b.list_orders("closed").orders == []
    assert FakeBroker(fail=True).list_orders().ok is False


# ───────────────────────── Alpaca paper construction + guard ────────────────


def test_alpaca_constructs_against_paper_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _with_creds(monkeypatch)
    b = alpaca.AlpacaBroker()
    assert isinstance(b, base.Broker)
    assert b._base_url == alpaca.ALPACA_PAPER_BASE_URL
    assert "paper-api" in alpaca.ALPACA_PAPER_BASE_URL


def test_alpaca_refuses_non_paper_url(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_creds(monkeypatch)
    # The live endpoint (and any other URL) must be structurally unreachable.
    with pytest.raises(ValueError, match="PAPER-ONLY"):
        alpaca.AlpacaBroker(base_url="https://api.alpaca.markets")


def test_alpaca_loads_credentials_into_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _with_creds(monkeypatch, key="abc", secret="xyz")
    headers = alpaca.AlpacaBroker()._headers()
    assert headers["APCA-API-KEY-ID"] == "abc"
    assert headers["APCA-API-SECRET-KEY"] == "xyz"


def test_alpaca_request_failsoft_without_credentials(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "trading_bot.broker.alpaca.secrets.get_secret", lambda _name: None,
    )
    resp = alpaca.AlpacaBroker()._request("GET", "/v2/account")
    assert resp.ok is False
    assert "credentials unset" in resp.error
    assert "credentials unset" in capsys.readouterr().err


# ───────────────────────── Alpaca account / positions read ──────────────────


def test_parse_helpers_are_lenient() -> None:
    assert alpaca._to_float("3.5") == 3.5
    assert alpaca._to_float(None) is None
    assert alpaca._to_float("garbage") is None
    p = alpaca._parse_position({"symbol": "X"})
    assert p.qty == 0.0 and p.avg_entry_price is None


def test_alpaca_get_account_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_creds(monkeypatch)
    _patch_request(monkeypatch, _FakeResp(200, {
        "account_number": "PA123", "buying_power": "50000.5", "cash": "10000",
        "equity": "60000", "currency": "USD", "status": "ACTIVE",
    }))
    acct = alpaca.AlpacaBroker().get_account()
    assert acct.ok is True
    assert acct.buying_power == 50000.5
    assert acct.cash == 10000.0
    assert acct.account_number == "PA123"
    assert acct.status == "ACTIVE"


def test_alpaca_get_account_non_2xx_is_failsoft(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    _with_creds(monkeypatch)
    _patch_request(monkeypatch, _FakeResp(401, {"message": "forbidden"}))
    acct = alpaca.AlpacaBroker().get_account()
    assert acct.ok is False
    assert acct.reason == "forbidden"
    assert "account unavailable" in capsys.readouterr().err


def test_alpaca_get_account_transport_error_is_failsoft(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    _with_creds(monkeypatch)
    _patch_request(monkeypatch, raises=RuntimeError("net down"))
    acct = alpaca.AlpacaBroker().get_account()
    assert acct.ok is False
    assert "net down" in acct.reason
    assert "error" in capsys.readouterr().err


def test_alpaca_get_positions_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_creds(monkeypatch)
    _patch_request(monkeypatch, _FakeResp(200, [
        {"symbol": "AAPL", "qty": "10", "side": "long",
         "avg_entry_price": "190.0", "market_value": "2000",
         "unrealized_pl": "50"},
        {"symbol": "TSLA", "qty": "-3", "side": "short",
         "avg_entry_price": "240.0"},
    ]))
    res = alpaca.AlpacaBroker().get_positions()
    assert res.ok is True
    assert [p.symbol for p in res.positions] == ["AAPL", "TSLA"]
    assert res.positions[0].qty == 10.0
    assert res.positions[1].side == "short"


def test_alpaca_get_positions_empty_is_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_creds(monkeypatch)
    _patch_request(monkeypatch, _FakeResp(200, []))
    res = alpaca.AlpacaBroker().get_positions()
    assert res.ok is True and res.positions == []


def test_alpaca_get_positions_server_error_is_failsoft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _with_creds(monkeypatch)
    _patch_request(monkeypatch, _FakeResp(500, {"message": "server error"}))
    res = alpaca.AlpacaBroker().get_positions()
    assert res.ok is False
    assert res.positions == []


# ───────────────────────── broker CLI (account / positions) ─────────────────


def test_cli_broker_account_ok(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_bot import __main__ as m
    monkeypatch.setattr(
        m.broker, "AlpacaBroker", lambda: FakeBroker(buying_power=5_000.0),
    )
    m.cmd_broker_account()
    out = capsys.readouterr().out
    assert "Buying power:  $5,000.00" in out


def test_cli_broker_account_unavailable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_bot import __main__ as m
    monkeypatch.setattr(m.broker, "AlpacaBroker", lambda: FakeBroker(fail=True))
    m.cmd_broker_account()
    assert "unavailable" in capsys.readouterr().out


def test_cli_broker_positions(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_bot import __main__ as m
    b = FakeBroker()
    b.set_position("AAPL", 10, avg_entry_price=190.0)
    monkeypatch.setattr(m.broker, "AlpacaBroker", lambda: b)
    m.cmd_broker_positions()
    assert "AAPL" in capsys.readouterr().out


def test_cli_broker_positions_empty_and_unavailable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_bot import __main__ as m
    monkeypatch.setattr(m.broker, "AlpacaBroker", lambda: FakeBroker())
    m.cmd_broker_positions()
    assert "no open positions" in capsys.readouterr().out
    monkeypatch.setattr(m.broker, "AlpacaBroker", lambda: FakeBroker(fail=True))
    m.cmd_broker_positions()
    assert "unavailable" in capsys.readouterr().out


# ───────────────────────── Alpaca order lifecycle mapping ────────────────────


def test_map_status_covers_lifecycle() -> None:
    assert alpaca.map_status("new") == broker.STATUS_NEW
    assert alpaca.map_status("accepted") == broker.STATUS_NEW
    assert alpaca.map_status("partially_filled") == broker.STATUS_PARTIALLY_FILLED
    assert alpaca.map_status("filled") == broker.STATUS_FILLED
    assert alpaca.map_status("canceled") == broker.STATUS_CANCELED
    assert alpaca.map_status("expired") == broker.STATUS_CANCELED
    assert alpaca.map_status("rejected") == broker.STATUS_REJECTED
    # An Alpaca state we have not classified maps to 'unknown', never 'error'.
    assert alpaca.map_status("some_future_state") == broker.STATUS_UNKNOWN
    assert alpaca.map_status(None) == broker.STATUS_UNKNOWN


# ───────────────────────── Alpaca submit_order (write path) ──────────────────


def test_alpaca_submit_order_success_defaults_to_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _with_creds(monkeypatch)
    cap: list[dict[str, object]] = []
    _patch_request(monkeypatch, _FakeResp(200, {
        "id": "o1", "client_order_id": "c1", "symbol": "AAPL", "qty": "10",
        "filled_qty": "0", "side": "buy", "type": "limit",
        "time_in_force": "day", "limit_price": "190.0", "status": "accepted",
        "submitted_at": "2026-01-01T00:00:00Z",
    }), capture=cap)
    res = alpaca.AlpacaBroker().submit_order("AAPL", 10, broker.SIDE_BUY, limit_price=190.0)
    assert res.ok is True
    assert res.status == broker.STATUS_NEW          # 'accepted' -> new
    assert res.order_id == "o1"
    body = cap[0]["json"]
    assert isinstance(body, dict)
    assert body["type"] == "limit"                  # LIMIT is the default
    assert body["limit_price"] == "190.0"
    assert body["time_in_force"] == "day"


def test_alpaca_submit_logs_intent_and_result(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    _with_creds(monkeypatch)
    _patch_request(monkeypatch, _FakeResp(200, {
        "id": "o1", "status": "new", "symbol": "AAPL",
    }))
    alpaca.AlpacaBroker().submit_order("AAPL", 1, broker.SIDE_BUY, limit_price=10.0)
    err = capsys.readouterr().err
    assert "submit intent" in err
    assert "submit result" in err


def test_alpaca_submit_local_validation_rejections_send_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _with_creds(monkeypatch)
    # No _patch_request: validation must short-circuit before any network call.
    b = alpaca.AlpacaBroker()
    no_price = b.submit_order("AAPL", 10, broker.SIDE_BUY, limit_price=None)
    assert no_price.status == broker.STATUS_REJECTED
    assert "limit_price" in no_price.reason
    bad_qty = b.submit_order("AAPL", 0, broker.SIDE_BUY, limit_price=10.0)
    assert bad_qty.status == broker.STATUS_REJECTED and "positive" in bad_qty.reason
    bad_side = b.submit_order("AAPL", 5, "hodl", limit_price=10.0)
    assert bad_side.status == broker.STATUS_REJECTED


def test_alpaca_submit_broker_rejection_4xx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _with_creds(monkeypatch)
    _patch_request(monkeypatch, _FakeResp(403, {"message": "insufficient buying power"}))
    res = alpaca.AlpacaBroker().submit_order("AAPL", 1000, broker.SIDE_BUY, limit_price=190.0)
    assert res.ok is False
    assert res.status == broker.STATUS_REJECTED
    assert res.reason == "insufficient buying power"


def test_alpaca_submit_rejected_in_2xx_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _with_creds(monkeypatch)
    _patch_request(monkeypatch, _FakeResp(200, {
        "id": "o2", "status": "rejected", "symbol": "AAPL",
    }))
    res = alpaca.AlpacaBroker().submit_order("AAPL", 1, broker.SIDE_BUY, limit_price=10.0)
    assert res.ok is False
    assert res.status == broker.STATUS_REJECTED
    assert res.reason == "rejected"


def test_alpaca_submit_server_error_is_error_not_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _with_creds(monkeypatch)
    _patch_request(monkeypatch, _FakeResp(500, {"message": "server"}))
    res = alpaca.AlpacaBroker().submit_order("AAPL", 1, broker.SIDE_BUY, limit_price=10.0)
    assert res.ok is False
    assert res.status == broker.STATUS_ERROR        # 5xx is transport, not a rejection


def test_alpaca_submit_transport_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_creds(monkeypatch)
    _patch_request(monkeypatch, raises=RuntimeError("net down"))
    res = alpaca.AlpacaBroker().submit_order("AAPL", 1, broker.SIDE_BUY, limit_price=10.0)
    assert res.ok is False
    assert res.status == broker.STATUS_ERROR


# ───────────────────────── Alpaca get / cancel / list orders ─────────────────


def test_alpaca_get_order_success(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_creds(monkeypatch)
    _patch_request(monkeypatch, _FakeResp(200, {
        "id": "o1", "status": "filled", "symbol": "AAPL",
        "filled_qty": "10", "filled_avg_price": "191.0",
    }))
    res = alpaca.AlpacaBroker().get_order("o1")
    assert res.ok is True
    assert res.status == broker.STATUS_FILLED
    assert res.filled_qty == 10.0


def test_alpaca_get_order_fills_in_missing_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _with_creds(monkeypatch)
    _patch_request(monkeypatch, _FakeResp(200, {"status": "new", "symbol": "AAPL"}))
    res = alpaca.AlpacaBroker().get_order("o9")
    assert res.order_id == "o9"


def test_alpaca_get_order_not_found_is_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    _with_creds(monkeypatch)
    _patch_request(monkeypatch, _FakeResp(404, {"message": "order not found"}))
    res = alpaca.AlpacaBroker().get_order("missing")
    assert res.ok is False
    assert res.status == broker.STATUS_ERROR        # a 404 read is not a 'rejection'
    assert res.order_id == "missing"
    assert "order unavailable" in capsys.readouterr().err


def test_alpaca_get_order_transport_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _with_creds(monkeypatch)
    _patch_request(monkeypatch, raises=RuntimeError("boom"))
    res = alpaca.AlpacaBroker().get_order("o1")
    assert res.status == broker.STATUS_ERROR
    assert res.order_id == "o1"


def test_alpaca_cancel_success(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_creds(monkeypatch)
    _patch_request(monkeypatch, _FakeResp(204, None))   # Alpaca cancel → 204
    res = alpaca.AlpacaBroker().cancel_order("o1")
    assert res.ok is True
    assert res.status == broker.STATUS_CANCELED
    assert res.order_id == "o1"


def test_alpaca_cancel_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    _with_creds(monkeypatch)
    _patch_request(monkeypatch, _FakeResp(422, {"message": "cannot cancel"}))
    res = alpaca.AlpacaBroker().cancel_order("o1")
    assert res.ok is False
    assert res.status == broker.STATUS_ERROR
    assert "cancel unavailable" in capsys.readouterr().err


def test_alpaca_list_orders_success(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_creds(monkeypatch)
    _patch_request(monkeypatch, _FakeResp(200, [
        {"id": "o1", "status": "new", "symbol": "AAPL"},
        {"id": "o2", "status": "partially_filled", "symbol": "TSLA"},
    ]))
    res = alpaca.AlpacaBroker().list_orders("open")
    assert res.ok is True
    assert [o.status for o in res.orders] == [
        broker.STATUS_NEW, broker.STATUS_PARTIALLY_FILLED,
    ]


def test_alpaca_list_orders_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_creds(monkeypatch)
    _patch_request(monkeypatch, _FakeResp(500, {"message": "down"}))
    res = alpaca.AlpacaBroker().list_orders()
    assert res.ok is False
    assert res.orders == []


# ───────────────────────── reconciliation (pure comparison) ─────────────────


def test_compare_positions_detects_each_divergence_kind() -> None:
    internal: dict[str, float | None] = {
        "AAPL": None,    # present in both -> no divergence (qty unknown)
        "GOOG": None,    # internal only
        "TSLA": 5.0,     # qty mismatch (internal 5 vs broker 8)
    }
    broker_map = {"AAPL": 10.0, "TSLA": 8.0, "MSFT": 3.0}
    kinds = {d.symbol: d.kind for d in broker.compare_positions(internal, broker_map)}
    assert kinds == {
        "GOOG": "internal_only",
        "TSLA": "qty_mismatch",
        "MSFT": "broker_only",
    }


def test_compare_positions_clean_when_aligned() -> None:
    assert broker.compare_positions({"AAPL": None}, {"AAPL": 10.0}) == []
    assert broker.compare_positions({"AAPL": 10.0}, {"AAPL": 10.0}) == []


# ───────────────────────── reconciliation (broker-authoritative) ─────────────


def test_reconcile_internal_only(tmp_db: Path) -> None:
    _seed_broker_tracked("AAPL")              # a real broker-routed position
    report = broker.reconcile(FakeBroker())   # broker holds nothing
    assert report.ok is True
    assert [(d.kind, d.symbol) for d in report.divergences] == [
        ("internal_only", "AAPL"),
    ]
    assert report.internal_symbols == ["AAPL"]
    assert report.broker_symbols == []


def test_reconcile_broker_only(tmp_db: Path) -> None:
    b = FakeBroker()
    b.set_position("TSLA", 4)
    report = broker.reconcile(b)              # bot tracks nothing
    assert report.ok is True
    assert [(d.kind, d.symbol) for d in report.divergences] == [
        ("broker_only", "TSLA"),
    ]


def test_reconcile_clean_when_aligned(tmp_db: Path) -> None:
    _seed_broker_tracked("AAPL", qty=10.0)
    b = FakeBroker()
    b.set_position("AAPL", 10)
    report = broker.reconcile(b)
    assert report.ok is True
    assert report.divergences == []          # presence AND quantity match


def test_reconcile_excludes_shadow_trades(tmp_db: Path) -> None:
    _seed_open_active("SHDW", track_mode="shadow")
    report = broker.reconcile(FakeBroker())
    # A shadow trade is signal tracking, not a broker position — never flagged.
    # (Since Phase 18 the whole trades table is out of scope; this guards the
    # original shadow guarantee within that.)
    assert report.internal_symbols == []
    assert report.divergences == []


def test_reconcile_broker_unavailable_flags_nothing(tmp_db: Path) -> None:
    _seed_broker_tracked("AAPL")
    report = broker.reconcile(FakeBroker(fail=True))
    assert report.ok is False
    # An outage must NOT flag the tracked position as 'internal_only'.
    assert [d.kind for d in report.divergences] == ["broker_unavailable"]
    assert all(d.kind != "internal_only" for d in report.divergences)


def test_reconcile_surfaces_open_order_count(tmp_db: Path) -> None:
    b = FakeBroker()
    b.submit_order("NVDA", 2, broker.SIDE_BUY, limit_price=120.0)  # rests as 'new'
    report = broker.reconcile(b)
    assert report.broker_open_orders == 1


# ───────────────────────── broker reconcile CLI ─────────────────────────────


def test_cli_broker_reconcile_with_divergences(
    monkeypatch: pytest.MonkeyPatch, tmp_db: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_bot import __main__ as m
    b = FakeBroker()
    b.set_position("TSLA", 4)
    monkeypatch.setattr(m.broker, "AlpacaBroker", lambda: b)
    m.cmd_broker_reconcile()
    out = capsys.readouterr().out
    assert "broker_only" in out
    assert "never auto-resolved" in out


def test_cli_broker_reconcile_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_db: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_bot import __main__ as m
    monkeypatch.setattr(m.broker, "AlpacaBroker", lambda: FakeBroker(fail=True))
    m.cmd_broker_reconcile()
    assert "cannot reconcile" in capsys.readouterr().out


def test_cli_broker_reconcile_clean(
    monkeypatch: pytest.MonkeyPatch, tmp_db: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_bot import __main__ as m
    monkeypatch.setattr(m.broker, "AlpacaBroker", lambda: FakeBroker())
    m.cmd_broker_reconcile()
    assert "No divergences" in capsys.readouterr().out
