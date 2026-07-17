"""Tests for trading_bot.alpaca_market_data (Phase 25 — free tier, parallel).

Mocked network only. The two properties that matter most: the free-tier guard
refuses paid feeds BEFORE any network I/O, and bar closure is decided by
timestamp arithmetic (start + duration <= now) — immune to the still-forming-
candle bug class the yfinance iloc[-2] fix addressed positionally.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from trading_bot import alpaca_market_data as amd

_NOW = datetime(2026, 7, 15, 18, 0, tzinfo=UTC)


def _bar(start: datetime, close: float = 100.0, symbol: str = "AAPL") -> amd.MarketBar:
    return amd.MarketBar(
        symbol=symbol, start=start, open=close, high=close + 1.0,
        low=close - 1.0, close=close, volume=1_000.0,
    )


def _with_creds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "trading_bot.alpaca_market_data.secrets.get_secret", lambda _n: "k",
    )


class _FakeResp:
    def __init__(self, payload: object, status: int = 200) -> None:
        self.status_code = status
        self.content = b"x"
        self._payload = payload

    def json(self) -> object:
        return self._payload


def _patch_get(
    monkeypatch: pytest.MonkeyPatch,
    responses: list[_FakeResp],
    capture: list[dict[str, Any]] | None = None,
) -> None:
    calls = iter(responses)

    def fake(url: str, **kw: Any) -> _FakeResp:
        if capture is not None:
            capture.append({"url": url, **kw})
        return next(calls)

    monkeypatch.setattr("trading_bot.alpaca_market_data.requests.get", fake)


def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    sleeps: list[float] = []
    monkeypatch.setattr(
        "trading_bot.alpaca_market_data.time.sleep",
        lambda s: sleeps.append(float(s)),
    )
    return sleeps


# ───────────────────────── bar closure (the critical check) ──────────────────


def test_daily_bar_today_is_still_forming() -> None:
    # Today's daily bar started 04:00Z; at 18:00Z the day has NOT elapsed.
    today = _bar(datetime(2026, 7, 15, 4, 0, tzinfo=UTC))
    assert amd.bar_is_closed(today, "1Day", _NOW) is False


def test_daily_bar_yesterday_is_closed() -> None:
    yesterday = _bar(datetime(2026, 7, 14, 4, 0, tzinfo=UTC))
    assert amd.bar_is_closed(yesterday, "1Day", _NOW) is True


def test_daily_bar_spring_dst_day_closes_after_23_elapsed_hours() -> None:
    # 2026-03-08 starts at midnight EST (05:00Z); the next New York midnight
    # is in EDT (04:00Z), only 23 elapsed hours later.
    bar = _bar(datetime(2026, 3, 8, 5, 0, tzinfo=UTC))
    just_before = datetime(2026, 3, 9, 3, 59, 59, tzinfo=UTC)
    at_next_midnight = datetime(2026, 3, 9, 4, 0, tzinfo=UTC)

    assert amd.bar_is_closed(bar, "1Day", just_before) is False
    assert amd.bar_is_closed(bar, "1Day", at_next_midnight) is True


def test_daily_bar_fall_dst_day_closes_after_25_elapsed_hours() -> None:
    # 2026-11-01 starts at midnight EDT (04:00Z); the next New York midnight
    # is in EST (05:00Z), 25 elapsed hours later. Fixed +24h would close early.
    bar = _bar(datetime(2026, 11, 1, 4, 0, tzinfo=UTC))
    after_24_hours = datetime(2026, 11, 2, 4, 30, tzinfo=UTC)
    at_next_midnight = datetime(2026, 11, 2, 5, 0, tzinfo=UTC)

    assert amd.bar_is_closed(bar, "1Day", after_24_hours) is False
    assert amd.bar_is_closed(bar, "1Day", at_next_midnight) is True


def test_hour_bar_boundaries() -> None:
    forming = _bar(datetime(2026, 7, 15, 17, 30, tzinfo=UTC))
    exactly_done = _bar(datetime(2026, 7, 15, 17, 0, tzinfo=UTC))
    assert amd.bar_is_closed(forming, "1Hour", _NOW) is False
    assert amd.bar_is_closed(exactly_done, "1Hour", _NOW) is True   # start+1h == now


def test_unknown_timeframe_is_never_closed() -> None:
    assert amd.bar_is_closed(_bar(_NOW - timedelta(days=9)), "3Day", _NOW) is False


def test_latest_closed_bar_skips_the_forming_last_element() -> None:
    """THE iloc[-2]-class regression, in Alpaca's shape: the response includes
    the currently-forming bar as its LAST element; the latest CLOSED bar must
    be the one before it — selected by timestamp arithmetic, not position."""
    closed_1 = _bar(datetime(2026, 7, 13, 4, 0, tzinfo=UTC), close=101.0)
    closed_2 = _bar(datetime(2026, 7, 14, 4, 0, tzinfo=UTC), close=102.0)
    forming = _bar(datetime(2026, 7, 15, 4, 0, tzinfo=UTC), close=999.0)

    latest = amd.latest_closed_bar([closed_1, closed_2, forming], "1Day", _NOW)
    assert latest is not None
    assert latest.close == 102.0                  # never the still-forming 999

    # And when the API happens NOT to include the forming bar, the SAME rule
    # still picks the same bar — a positional rule would have broken here.
    latest2 = amd.latest_closed_bar([closed_1, closed_2], "1Day", _NOW)
    assert latest2 is not None
    assert latest2.close == 102.0


def test_latest_closed_bar_none_when_all_forming() -> None:
    forming = _bar(datetime(2026, 7, 15, 4, 0, tzinfo=UTC))
    assert amd.latest_closed_bar([forming], "1Day", _NOW) is None


# ───────────────────────── free-tier guard ───────────────────────────────────


def test_sip_feed_refused_before_any_network_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _with_creds(monkeypatch)
    attempts: list[dict[str, Any]] = []
    _patch_get(monkeypatch, [], capture=attempts)   # any request would raise

    result = amd.AlpacaMarketDataClient().get_stock_bars(
        ["AAPL"], start=_NOW - timedelta(days=5), feed="sip",
    )
    assert result.ok is False
    assert "requires Algo Trader Plus" in result.reason
    assert attempts == []                           # NEVER attempted


def test_opra_feed_refused_before_any_network_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _with_creds(monkeypatch)
    attempts: list[dict[str, Any]] = []
    _patch_get(monkeypatch, [], capture=attempts)

    result = amd.AlpacaMarketDataClient().get_stock_bars(
        ["AAPL"], start=_NOW - timedelta(days=5), feed="opra",
    )
    assert result.ok is False
    assert "requires Algo Trader Plus" in result.reason
    assert attempts == []


def test_unknown_feed_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_creds(monkeypatch)
    attempts: list[dict[str, Any]] = []
    _patch_get(monkeypatch, [], capture=attempts)
    result = amd.AlpacaMarketDataClient().get_stock_bars(
        ["AAPL"], start=_NOW - timedelta(days=5), feed="delayed_sip",
    )
    assert result.ok is False and attempts == []


def test_stock_request_always_pins_iex_feed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _with_creds(monkeypatch)
    _no_sleep(monkeypatch)
    captured: list[dict[str, Any]] = []
    _patch_get(monkeypatch, [_FakeResp({"bars": {}})], capture=captured)

    amd.AlpacaMarketDataClient().get_stock_bars(
        ["AAPL"], start=_NOW - timedelta(days=5),
    )
    assert captured[0]["params"]["feed"] == "iex"   # explicit, never defaulted


@pytest.mark.parametrize("feed", ["sip", "opra", "delayed_sip"])
def test_stock_latest_quote_paid_or_unknown_feed_refused_before_io(
    monkeypatch: pytest.MonkeyPatch,
    feed: str,
) -> None:
    _with_creds(monkeypatch)
    attempts: list[dict[str, Any]] = []
    _patch_get(monkeypatch, [], capture=attempts)

    result = amd.AlpacaMarketDataClient().get_stock_latest_quotes(
        ["AAPL"], feed=feed,
    )

    assert result.ok is False
    assert "paid data plan" in result.reason or "unknown feed" in result.reason
    assert attempts == []


# ───────────────────────── fetch plumbing ────────────────────────────────────


def test_stock_bars_parse_and_sort(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_creds(monkeypatch)
    _no_sleep(monkeypatch)
    payload = {
        "bars": {
            "AAPL": [
                {"t": "2026-07-14T04:00:00Z", "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 10},
                {"t": "2026-07-13T04:00:00Z", "o": 1, "h": 2, "l": 0.5, "c": 1.2, "v": 11},
            ],
        },
        "next_page_token": None,
    }
    _patch_get(monkeypatch, [_FakeResp(payload)])
    result = amd.AlpacaMarketDataClient().get_stock_bars(
        ["AAPL"], start=_NOW - timedelta(days=5),
    )
    assert result.ok is True
    bars = result.bars["AAPL"]
    assert [b.close for b in bars] == [1.2, 1.5]        # sorted oldest-first
    assert bars[0].start.tzinfo is not None


@pytest.mark.parametrize(
    "payload",
    [[], {}, {"bars": []}, {"bars": {"AAPL": {}}}],
)
def test_malformed_http_200_bar_payload_fails_soft(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    payload: object,
) -> None:
    _with_creds(monkeypatch)
    _no_sleep(monkeypatch)
    _patch_get(monkeypatch, [_FakeResp(payload)])

    result = amd.AlpacaMarketDataClient().get_stock_bars(
        ["AAPL"], start=_NOW - timedelta(days=5),
    )

    assert result.ok is False
    assert result.bars == {}
    assert "malformed" in result.reason
    assert result.reason in capsys.readouterr().err


def test_malformed_bar_row_preserves_valid_partial_result(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _with_creds(monkeypatch)
    _no_sleep(monkeypatch)
    payload = {"bars": {"AAPL": [
        {"t": "2026-07-14T04:00:00Z", "o": 1, "h": 2, "l": 0.5,
         "c": 1.5, "v": 10},
        {"t": "bad"},
    ]}}
    _patch_get(monkeypatch, [_FakeResp(payload)])

    result = amd.AlpacaMarketDataClient().get_stock_bars(
        ["AAPL"], start=_NOW - timedelta(days=5),
    )

    assert result.ok is True
    assert [bar.close for bar in result.bars["AAPL"]] == [1.5]
    assert "malformed bar for AAPL" in result.reason
    assert result.reason in capsys.readouterr().err


def test_crypto_bars_translate_and_key_by_internal_symbol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _with_creds(monkeypatch)
    _no_sleep(monkeypatch)
    captured: list[dict[str, Any]] = []
    payload = {"bars": {"BTC/USD": [
        {"t": "2026-07-14T05:00:00Z", "o": 1, "h": 2, "l": 0.5, "c": 64000.0, "v": 3},
    ]}}
    _patch_get(monkeypatch, [_FakeResp(payload)], capture=captured)

    result = amd.AlpacaMarketDataClient().get_crypto_bars(
        ["BTC-USD"], start=_NOW - timedelta(days=5),
    )
    assert captured[0]["params"]["symbols"] == "BTC/USD"   # wire form
    assert result.ok is True
    assert list(result.bars.keys()) == ["BTC-USD"]         # caller's form


def test_stock_latest_quotes_parse_and_pin_iex(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _with_creds(monkeypatch)
    _no_sleep(monkeypatch)
    captured: list[dict[str, Any]] = []
    payload = {"quotes": {"AAPL": {
        "t": "2026-07-15T17:59:59Z",
        "bp": 211.10,
        "bs": 12,
        "ap": 211.14,
        "as": 9,
    }}}
    _patch_get(monkeypatch, [_FakeResp(payload)], capture=captured)

    result = amd.AlpacaMarketDataClient().get_stock_latest_quotes(["AAPL"])

    assert result.ok is True
    assert captured[0]["url"].endswith("/v2/stocks/quotes/latest")
    assert captured[0]["params"] == {"feed": "iex", "symbols": "AAPL"}
    quote = result.quotes["AAPL"]
    assert quote.symbol == "AAPL"
    assert quote.bid_price == 211.10
    assert quote.ask_price == 211.14
    assert quote.timestamp.tzinfo is not None


def test_crypto_latest_quotes_translate_and_key_by_internal_symbol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _with_creds(monkeypatch)
    _no_sleep(monkeypatch)
    captured: list[dict[str, Any]] = []
    payload = {"quotes": {"BTC/USD": {
        "t": "2026-07-15T17:59:59Z",
        "bp": 64_000,
        "bs": 0.1,
        "ap": 64_010,
        "as": 0.2,
    }}}
    _patch_get(monkeypatch, [_FakeResp(payload)], capture=captured)

    result = amd.AlpacaMarketDataClient().get_crypto_latest_quotes(["BTC-USD"])

    assert result.ok is True
    assert captured[0]["url"].endswith("/v1beta3/crypto/us/latest/quotes")
    assert captured[0]["params"]["symbols"] == "BTC/USD"
    assert result.quotes["BTC-USD"].symbol == "BTC-USD"


def test_latest_quotes_malformed_payload_fails_soft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _with_creds(monkeypatch)
    _no_sleep(monkeypatch)
    payload = {"quotes": {"AAPL": {
        "t": "2026-07-15T17:59:59Z", "bp": 211.10, "bs": 12,
    }}}
    _patch_get(monkeypatch, [_FakeResp(payload)])

    result = amd.AlpacaMarketDataClient().get_stock_latest_quotes(["AAPL"])

    assert result.ok is False
    assert result.quotes == {}
    assert "no valid quotes" in result.reason


def test_latest_quotes_transport_failure_fails_soft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _with_creds(monkeypatch)
    _no_sleep(monkeypatch)

    def fail(*_args: object, **_kwargs: object) -> None:
        raise OSError("offline")

    monkeypatch.setattr("trading_bot.alpaca_market_data.requests.get", fail)
    result = amd.AlpacaMarketDataClient().get_stock_latest_quotes(["AAPL"])

    assert result.ok is False
    assert "offline" in result.reason


def test_batching_splits_large_symbol_lists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _with_creds(monkeypatch)
    _no_sleep(monkeypatch)
    captured: list[dict[str, Any]] = []
    _patch_get(
        monkeypatch,
        [_FakeResp({"bars": {}}), _FakeResp({"bars": {}})],
        capture=captured,
    )
    client = amd.AlpacaMarketDataClient(batch_size=2)
    client.get_stock_bars(["A", "B", "C"], start=_NOW - timedelta(days=5))
    assert len(captured) == 2
    assert captured[0]["params"]["symbols"] == "A,B"
    assert captured[1]["params"]["symbols"] == "C"


def test_pagination_follows_next_page_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _with_creds(monkeypatch)
    _no_sleep(monkeypatch)
    captured: list[dict[str, Any]] = []
    page1 = {
        "bars": {"AAPL": [
            {"t": "2026-07-13T04:00:00Z", "o": 1, "h": 2, "l": 0.5, "c": 1.0, "v": 1},
        ]},
        "next_page_token": "tok",
    }
    page2 = {
        "bars": {"AAPL": [
            {"t": "2026-07-14T04:00:00Z", "o": 1, "h": 2, "l": 0.5, "c": 2.0, "v": 1},
        ]},
    }
    _patch_get(monkeypatch, [_FakeResp(page1), _FakeResp(page2)], capture=captured)
    result = amd.AlpacaMarketDataClient().get_stock_bars(
        ["AAPL"], start=_NOW - timedelta(days=5),
    )
    assert len(captured) == 2
    assert captured[1]["params"]["page_token"] == "tok"
    assert [b.close for b in result.bars["AAPL"]] == [1.0, 2.0]


def test_later_bar_page_failure_preserves_and_logs_partial_result(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _with_creds(monkeypatch)
    _no_sleep(monkeypatch)
    page1 = {
        "bars": {"AAPL": [
            {"t": "2026-07-13T04:00:00Z", "o": 1, "h": 2, "l": 0.5,
             "c": 1.0, "v": 1},
        ]},
        "next_page_token": "tok",
    }
    page2 = {"message": "page unavailable"}
    _patch_get(monkeypatch, [_FakeResp(page1), _FakeResp(page2, status=503)])

    result = amd.AlpacaMarketDataClient().get_stock_bars(
        ["AAPL"], start=_NOW - timedelta(days=5),
    )

    assert result.ok is True
    assert [bar.close for bar in result.bars["AAPL"]] == [1.0]
    assert result.reason == "page unavailable"
    assert "batch failed (page unavailable)" in capsys.readouterr().err


def test_quote_non_2xx_reason_is_logged(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _with_creds(monkeypatch)
    _no_sleep(monkeypatch)
    _patch_get(monkeypatch, [_FakeResp({"message": "forbidden"}, status=403)])

    result = amd.AlpacaMarketDataClient().get_stock_latest_quotes(["AAPL"])

    assert result.ok is False
    assert result.reason == "forbidden"
    assert "quote batch failed (forbidden)" in capsys.readouterr().err


def test_quote_batch_failure_preserves_and_logs_partial_result(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _with_creds(monkeypatch)
    _no_sleep(monkeypatch)
    good = {"quotes": {"AAPL": {
        "t": "2026-07-15T17:59:59Z", "bp": 211.10, "bs": 12,
        "ap": 211.14, "as": 9,
    }}}
    failed = {"message": "quote page unavailable"}
    _patch_get(monkeypatch, [_FakeResp(good), _FakeResp(failed, status=503)])

    result = amd.AlpacaMarketDataClient(
        batch_size=1,
    ).get_stock_latest_quotes(["AAPL", "MSFT"])

    assert result.ok is True
    assert list(result.quotes) == ["AAPL"]
    assert result.reason == "quote page unavailable"
    assert "quote batch failed (quote page unavailable)" in capsys.readouterr().err


def test_rate_limit_fails_soft(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_creds(monkeypatch)
    _no_sleep(monkeypatch)
    _patch_get(monkeypatch, [_FakeResp({"message": "too many requests"}, status=429)])
    result = amd.AlpacaMarketDataClient().get_stock_bars(
        ["AAPL"], start=_NOW - timedelta(days=5),
    )
    assert result.ok is False
    assert "rate limited" in result.reason


def test_missing_credentials_fail_soft(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "trading_bot.alpaca_market_data.secrets.get_secret", lambda _n: None,
    )
    result = amd.AlpacaMarketDataClient().get_stock_bars(
        ["AAPL"], start=_NOW - timedelta(days=5),
    )
    assert result.ok is False
    assert "credentials unset" in result.reason


def test_throttle_spaces_consecutive_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two back-to-back requests are spaced by the minimum interval — the belt
    that keeps even a degenerate loop under the 200 rpm free-tier cap."""
    _with_creds(monkeypatch)
    sleeps = _no_sleep(monkeypatch)
    clock = iter([100.0, 100.0, 100.01, 100.01, 100.5, 100.5, 101.0, 101.0])
    monkeypatch.setattr(
        "trading_bot.alpaca_market_data.time.monotonic", lambda: next(clock),
    )
    _patch_get(monkeypatch, [_FakeResp({"bars": {}}), _FakeResp({"bars": {}})])

    client = amd.AlpacaMarketDataClient(batch_size=1, min_request_interval_s=0.35)
    client.get_stock_bars(["A", "B"], start=_NOW - timedelta(days=5))
    assert sleeps                                   # the second request waited
    assert sleeps[0] == pytest.approx(0.34, abs=0.01)


def test_default_throttle_is_safely_below_200_requests_per_minute() -> None:
    requests_per_minute = 60.0 / amd.config.MARKETDATA_MIN_REQUEST_INTERVAL_S
    assert requests_per_minute < 200.0
