"""Tests for trading_bot.marketdata_compare (Phase 25 — diagnostic only).

Mocked/injected data everywhere — no live network. The report flags
tolerance-exceeding close differences and most-recent-closed-bar
disagreements; it never changes any consumer's behavior.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from trading_bot import config
from trading_bot import marketdata_compare as mc
from trading_bot.alpaca_market_data import BarsResult, MarketBar

# A Wednesday evening ET (23:00 UTC = 19:00 ET) — the 15th's session is over
# but its daily bar has not fully elapsed under the conservative closure rule,
# so both sources should agree the latest CLOSED bar is the 14th.
_NOW = datetime(2026, 7, 15, 23, 0, tzinfo=UTC)


def _yf_df(closes: dict[str, float]) -> pd.DataFrame:
    index = pd.DatetimeIndex([pd.Timestamp(d) for d in closes])
    return pd.DataFrame({
        "Open": list(closes.values()), "High": list(closes.values()),
        "Low": list(closes.values()), "Close": list(closes.values()),
        "Volume": [1_000] * len(closes),
    }, index=index)


def _alpaca_bar(day: str, close: float, symbol: str = "AAPL") -> MarketBar:
    # Daily bars start at 04:00Z (midnight ET during EDT).
    start = datetime.fromisoformat(f"{day}T04:00:00+00:00")
    return MarketBar(
        symbol=symbol, start=start, open=close, high=close, low=close,
        close=close, volume=1_000.0,
    )


# ───────────────────────── per-ticker comparison ─────────────────────────────


def test_matching_ticker_within_tolerance(tmp_db: Path) -> None:
    yf_df = _yf_df({"2026-07-13": 100.00, "2026-07-14": 101.00})
    bars = [
        _alpaca_bar("2026-07-13", 100.02),      # 0.02% off — inside tolerance
        _alpaca_bar("2026-07-14", 100.98),
    ]
    result = mc.compare_ticker("AAPL", yf_df, bars, now=_NOW)
    assert result.status == mc.STATUS_MATCH
    assert result.bars_compared == 2
    assert result.max_close_diff_pct is not None
    assert result.max_close_diff_pct < config.MD_COMPARE_TOLERANCE_PCT
    assert result.latest_closed_agrees is True
    assert result.latest_closed_yf == "2026-07-14"
    assert result.latest_closed_alpaca == "2026-07-14"


def test_divergent_ticker_beyond_tolerance(tmp_db: Path) -> None:
    yf_df = _yf_df({"2026-07-13": 100.00, "2026-07-14": 101.00})
    bars = [
        _alpaca_bar("2026-07-13", 100.00),
        _alpaca_bar("2026-07-14", 101.80),      # 0.79% off — beyond 0.5%
    ]
    result = mc.compare_ticker("AAPL", yf_df, bars, now=_NOW)
    assert result.status == mc.STATUS_DIVERGENT
    assert result.max_close_diff_pct == pytest.approx(0.792, abs=0.01)
    assert "exceeds" in result.note


def test_latest_closed_disagreement_is_flagged(tmp_db: Path) -> None:
    """Closes agree, but Alpaca's newest bar is a day behind yfinance's — a
    materially different 'most recent closed bar' must be flagged."""
    yf_df = _yf_df({"2026-07-13": 100.00, "2026-07-14": 101.00})
    bars = [_alpaca_bar("2026-07-13", 100.00)]    # missing the 14th entirely
    result = mc.compare_ticker("AAPL", yf_df, bars, now=_NOW)
    assert result.status == mc.STATUS_DIVERGENT
    assert result.latest_closed_agrees is False
    assert "most recent closed bar" in result.note


def test_todays_forming_bars_do_not_count_as_latest_closed(tmp_db: Path) -> None:
    """Both sources include TODAY's still-forming daily bar; each side's own
    closure rule must ignore it and land on yesterday — agreement, no flag."""
    yf_df = _yf_df({
        "2026-07-14": 101.00,
        "2026-07-15": 999.0,                     # today's partial session row
    })
    bars = [
        _alpaca_bar("2026-07-14", 101.00),
        _alpaca_bar("2026-07-15", 998.0),        # today's partial Alpaca bar
    ]
    result = mc.compare_ticker("AAPL", yf_df, bars, now=_NOW)
    assert result.latest_closed_yf == "2026-07-14"
    assert result.latest_closed_alpaca == "2026-07-14"
    assert result.latest_closed_agrees is True


def test_missing_alpaca_data_reported(tmp_db: Path) -> None:
    yf_df = _yf_df({"2026-07-14": 101.00})
    result = mc.compare_ticker("AAPL", yf_df, [], now=_NOW)
    assert result.status == mc.STATUS_MISSING_ALPACA


def test_missing_yfinance_data_reported(tmp_db: Path) -> None:
    result = mc.compare_ticker(
        "AAPL", None, [_alpaca_bar("2026-07-14", 101.0)], now=_NOW,
    )
    assert result.status == mc.STATUS_MISSING_YFINANCE


# ───────────────────────── universe run (injected sources) ───────────────────


class _FakeClient:
    """Stands in for AlpacaMarketDataClient — records calls, serves fixtures."""

    def __init__(self, stock_bars: dict[str, list[MarketBar]],
                 crypto_bars: dict[str, list[MarketBar]] | None = None) -> None:
        self._stock = stock_bars
        self._crypto = crypto_bars or {}
        self.stock_calls: list[list[str]] = []
        self.crypto_calls: list[list[str]] = []

    def get_stock_bars(self, symbols: list[str], **_kw: object) -> BarsResult:
        self.stock_calls.append(symbols)
        return BarsResult(ok=True, bars={
            s: self._stock.get(s, []) for s in symbols if s in self._stock
        })

    def get_crypto_bars(self, symbols: list[str], **_kw: object) -> BarsResult:
        self.crypto_calls.append(symbols)
        return BarsResult(ok=True, bars={
            s: self._crypto.get(s, []) for s in symbols if s in self._crypto
        })


def test_compare_universe_summary_counts(tmp_db: Path) -> None:
    client = _FakeClient(
        stock_bars={
            "AAPL": [_alpaca_bar("2026-07-14", 101.00)],
            "MSFT": [_alpaca_bar("2026-07-14", 405.00, symbol="MSFT")],
        },
        crypto_bars={"BTC-USD": [
            _alpaca_bar("2026-07-14", 64_000.0, symbol="BTC-USD"),
        ]},
    )
    frames = {
        "AAPL": _yf_df({"2026-07-14": 101.00}),          # match
        "MSFT": _yf_df({"2026-07-14": 400.00}),          # 1.25% off — divergent
        "BTC-USD": _yf_df({"2026-07-14": 64_010.0}),     # 0.016% — match
        "GS": _yf_df({"2026-07-14": 700.0}),             # no Alpaca — missing
    }

    report = mc.compare_universe(
        ["AAPL", "MSFT", "BTC-USD", "GS"], now=_NOW,
        client=client,  # type: ignore[arg-type]
        yf_fetch=lambda t, _w: frames.get(t),
    )
    assert report.matched == 2
    assert report.divergent == 1
    assert report.missing == 1
    # Stocks went through the stock endpoint, crypto through the crypto one.
    assert client.stock_calls == [["AAPL", "MSFT", "GS"]]
    assert client.crypto_calls == [["BTC-USD"]]


def test_compare_universe_one_bad_ticker_never_sinks_the_report(
    tmp_db: Path,
) -> None:
    client = _FakeClient(stock_bars={
        "AAPL": [_alpaca_bar("2026-07-14", 101.00)],
    })

    def flaky_fetch(ticker: str, _w: int) -> pd.DataFrame | None:
        if ticker == "BAD":
            raise RuntimeError("feed down")
        return _yf_df({"2026-07-14": 101.00})

    report = mc.compare_universe(
        ["BAD", "AAPL"], now=_NOW,
        client=client,  # type: ignore[arg-type]
        yf_fetch=flaky_fetch,
    )
    by_ticker = {r.ticker: r.status for r in report.results}
    assert by_ticker["AAPL"] == mc.STATUS_MATCH      # processed despite BAD
    assert by_ticker["BAD"].startswith("missing")


# ───────────────────────── CLI ───────────────────────────────────────────────


def test_cli_marketdata_compare_prints_report(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_bot import __main__ as m

    fake_report = mc.ComparisonReport(results=[
        mc.TickerComparison(
            "AAPL", mc.STATUS_MATCH, bars_compared=5,
            max_close_diff_pct=0.02, mean_close_diff_pct=0.01,
            latest_closed_agrees=True, latest_closed_yf="2026-07-14",
            latest_closed_alpaca="2026-07-14",
        ),
        mc.TickerComparison(
            "MSFT", mc.STATUS_DIVERGENT, bars_compared=5,
            max_close_diff_pct=1.25, mean_close_diff_pct=0.60,
            latest_closed_agrees=True,
            note="close diff 1.250% exceeds 0.5% tolerance",
        ),
    ])
    captured_args: dict[str, object] = {}

    def fake_compare(tickers: list[str], **kw: object) -> mc.ComparisonReport:
        captured_args["tickers"] = tickers
        captured_args.update(kw)
        return fake_report

    monkeypatch.setattr(m.marketdata_compare, "compare_universe", fake_compare)
    m.cmd_marketdata_compare(tickers="AAPL,MSFT", window=5)
    out = capsys.readouterr().out
    assert "MARKET DATA COMPARISON" in out
    assert "1 matched, 1 divergent, 0 missing" in out
    assert "diagnostic only" in out
    assert captured_args["tickers"] == ["AAPL", "MSFT"]
    assert captured_args["window_bars"] == 5


def test_default_universe_dedupes_and_includes_crypto(tmp_db: Path) -> None:
    universe = mc.default_universe()
    assert len(universe) == len(set(universe))       # deduped
    for pair in config.LONGTERM_CRYPTO_UNIVERSE:
        assert pair in universe
    assert "AAPL" in universe                        # shadow universe present
