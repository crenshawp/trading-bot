"""Tests for trading_bot.indicators — pure quantitative families (Phase 6).

Every value here is hand-computed; no network, no fixtures beyond small
in-line OHLCV frames. The families are advisory context attached to fired
signals, so correctness against known inputs is the whole point.
"""

import pandas as pd
import pytest

from trading_bot import indicators


def _ohlc(
    highs: list[float], lows: list[float], closes: list[float],
) -> pd.DataFrame:
    """Minimal OHLCV frame matching the scanner's candle structure."""
    return pd.DataFrame(
        {
            "Open": closes,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": [1_000_000] * len(closes),
        },
        index=pd.date_range("2026-01-01", periods=len(closes), freq="D"),
    )


# ───────────────────────── volatility: true range / ATR ─────────────────────────


def test_true_range_first_candle_is_high_low() -> None:
    df = _ohlc(highs=[10, 12, 11], lows=[8, 9, 10], closes=[9, 11, 10])
    tr = indicators.true_range(df)
    # TR0 = H-L (no prev close); TR1 = max(3, |12-9|, |9-9|)=3; TR2 = max(1,0,1)=1
    assert list(tr) == [2.0, 3.0, 1.0]


def test_atr_is_rolling_mean_of_true_range() -> None:
    df = _ohlc(highs=[10, 12, 11], lows=[8, 9, 10], closes=[9, 11, 10])
    a = indicators.atr(df, period=2)
    assert pd.isna(a.iloc[0])
    assert a.iloc[1] == pytest.approx(2.5)   # (2+3)/2
    assert a.iloc[2] == pytest.approx(2.0)   # (3+1)/2


# ───────────────────────── volatility: realized vol + bands ─────────────────────


def test_realized_volatility_population_std_of_returns() -> None:
    close = pd.Series([100.0, 110.0, 99.0])
    rv = indicators.realized_volatility(close, period=2)
    # returns = [nan, +0.10, -0.10]; population std of {+0.1,-0.1} = 0.1
    assert pd.isna(rv.iloc[1])
    assert rv.iloc[-1] == pytest.approx(0.1)


def test_realized_vol_bands_envelope_scales_by_returns_vol() -> None:
    close = pd.Series([100.0, 110.0, 99.0])
    upper, mid, lower = indicators.realized_vol_bands(close, period=2, num_std=2.0)
    assert mid.iloc[-1] == pytest.approx(104.5)        # SMA(110, 99)
    assert upper.iloc[-1] == pytest.approx(125.4)      # 104.5 * (1 + 2*0.1)
    assert lower.iloc[-1] == pytest.approx(83.6)       # 104.5 * (1 - 2*0.1)


# ───────────────────────── volatility: regime classifier ────────────────────────


@pytest.mark.parametrize(
    ("atr_now", "baseline", "expected"),
    [
        (1.0, 2.0, "low"),       # ratio 0.5
        (2.0, 2.0, "normal"),    # ratio 1.0
        (3.0, 2.0, "high"),      # ratio 1.5
        (1.6, 2.0, "normal"),    # ratio 0.8 — boundary is NOT low (strict <)
        (2.4, 2.0, "normal"),    # ratio 1.2 — boundary is NOT high (strict >)
    ],
)
def test_classify_vol_regime_thresholds(
    atr_now: float, baseline: float, expected: str
) -> None:
    assert indicators.classify_vol_regime(atr_now, baseline) == expected


@pytest.mark.parametrize(
    ("atr_now", "baseline"),
    [(2.0, None), (2.0, 0.0), (2.0, -1.0), (None, 2.0)],
)
def test_classify_vol_regime_unknown_on_bad_baseline(
    atr_now: float | None, baseline: float | None
) -> None:
    assert indicators.classify_vol_regime(atr_now, baseline) == "unknown"


# ───────────────────────────── momentum: RSI ────────────────────────────────────


def test_rsi_all_gains_is_100() -> None:
    rsi = indicators.rsi(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]), period=2)
    assert rsi.iloc[-1] == pytest.approx(100.0)


def test_rsi_known_intermediate_value() -> None:
    # gains avg = (10+0)/2 = 5; losses avg = (0+5)/2 = 2.5; RS = 2 -> RSI 66.667
    rsi = indicators.rsi(pd.Series([100.0, 110.0, 105.0]), period=2)
    assert rsi.iloc[-1] == pytest.approx(66.66667, abs=1e-4)


def test_rsi_flat_series_is_undefined() -> None:
    rsi = indicators.rsi(pd.Series([5.0, 5.0, 5.0]), period=2)
    assert pd.isna(rsi.iloc[-1])   # 0 gains / 0 losses -> genuinely undefined


# ─────────────────────────── trend strength: ADX ────────────────────────────────


def _UP() -> pd.DataFrame:
    return _ohlc(
        highs=[10, 11, 12, 13, 14, 15],
        lows=[8, 9, 10, 11, 12, 13],
        closes=[9, 10, 11, 12, 13, 14],
    )


def _DOWN() -> pd.DataFrame:
    return _ohlc(
        highs=[15, 14, 13, 12, 11, 10],
        lows=[13, 12, 11, 10, 9, 8],
        closes=[14, 13, 12, 11, 10, 9],
    )


def _CHOP() -> pd.DataFrame:
    return _ohlc(
        highs=[10, 12, 10, 12, 10, 12],
        lows=[8, 10, 8, 10, 8, 10],
        closes=[9, 11, 9, 11, 9, 11],
    )


def test_adx_steady_uptrend_is_max() -> None:
    # +DM dominates entirely, -DM=0 -> DX=100 every candle -> ADX=100.
    assert indicators.adx(_UP(), period=2).iloc[-1] == pytest.approx(100.0)


def test_adx_is_direction_agnostic() -> None:
    # A steady DOWN-move is just as strong a trend -> also maxes ADX.
    assert indicators.adx(_DOWN(), period=2).iloc[-1] == pytest.approx(100.0)


def test_adx_choppy_is_weaker_than_trending() -> None:
    chop = indicators.adx(_CHOP(), period=2).iloc[-1]
    trend = indicators.adx(_UP(), period=2).iloc[-1]
    assert chop < trend


# ───────────────────────────── volume: OBV ──────────────────────────────────────


def test_obv_accumulates_signed_volume() -> None:
    df = _ohlc(
        highs=[10, 11, 10, 10, 12],
        lows=[10, 11, 10, 10, 12],
        closes=[10, 11, 10, 10, 12],
    )
    df["Volume"] = [100, 200, 300, 400, 500]
    obv = indicators.obv(df)
    # dir: [0, +1, -1, 0(flat), +1]; signed vol cumsum: [0, 200, -100, -100, 400]
    assert list(obv) == [0.0, 200.0, -100.0, -100.0, 400.0]


def test_obv_flat_close_carries_prior_total() -> None:
    df = _ohlc(highs=[10, 10, 10], lows=[10, 10, 10], closes=[10, 10, 10])
    df["Volume"] = [100, 200, 300]
    # every close flat -> no volume added -> OBV stays 0 throughout
    assert list(indicators.obv(df)) == [0.0, 0.0, 0.0]


# ─────────────────────── correlation: pure pairwise math ─────────────────────────


@pytest.fixture(autouse=True)
def _clear_corr_cache() -> None:
    """The correlation memo is module-level — reset it between tests."""
    indicators.clear_correlation_cache()


def test_average_pairwise_correlation_identical_is_one() -> None:
    r = pd.DataFrame({"A": [0.1, -0.1, 0.1], "B": [0.1, -0.1, 0.1]})
    assert indicators.average_pairwise_correlation(r, "A") == pytest.approx(1.0)


def test_average_pairwise_correlation_offsets_to_zero() -> None:
    # B identical to A (corr +1), C exact mirror of A (corr -1) -> mean 0.
    r = pd.DataFrame(
        {"A": [0.1, -0.1, 0.1, -0.1],
         "B": [0.1, -0.1, 0.1, -0.1],
         "C": [-0.1, 0.1, -0.1, 0.1]}
    )
    assert indicators.average_pairwise_correlation(r, "A") == pytest.approx(0.0)


def test_average_pairwise_correlation_none_when_absent_or_no_peers() -> None:
    r = pd.DataFrame({"A": [0.1, -0.1], "B": [0.1, -0.1]})
    assert indicators.average_pairwise_correlation(r, "Z") is None      # absent
    solo = pd.DataFrame({"A": [0.1, -0.1]})
    assert indicators.average_pairwise_correlation(solo, "A") is None    # no peers


@pytest.mark.parametrize(
    ("avg_corr", "label"),
    [
        (0.7, "concentrated"),
        (0.6, "concentrated"),   # boundary: >= concentrated_at
        (0.59, "moderate"),
        (0.3, "moderate"),       # boundary: not < diversified_at
        (0.29, "diversified"),
        (-0.8, "diversified"),
        (None, "unknown"),
    ],
)
def test_classify_concentration(avg_corr: float | None, label: str) -> None:
    assert indicators.classify_concentration(avg_corr) == label


# ─────────────────────── correlation: fail-soft wrapper ──────────────────────────


def _fetch_from(data: dict[str, pd.Series]) -> "object":
    def _fetch(name: str) -> pd.Series | None:
        return data.get(name)
    return _fetch


def test_correlation_concentration_concentrated_when_names_move_together() -> None:
    common = pd.Series([100.0, 101.0, 103.0, 106.0, 110.0])
    data = {"AAA": common, "BBB": common, "CCC": common}
    avg, label = indicators.correlation_concentration(
        "AAA", ["AAA", "BBB", "CCC"], _fetch_from(data),
    )
    assert avg == pytest.approx(1.0)
    assert label == "concentrated"


def test_correlation_concentration_diversified_when_anti_correlated() -> None:
    data = {
        "AAA": pd.Series([100.0, 110.0, 100.0, 110.0, 100.0]),
        "BBB": pd.Series([100.0, 90.0, 100.0, 90.0, 100.0]),  # mirror of AAA
    }
    avg, label = indicators.correlation_concentration(
        "AAA", ["AAA", "BBB"], _fetch_from(data),
    )
    assert avg is not None and avg < 0.0
    assert label == "diversified"


def test_correlation_concentration_unknown_with_too_few_names(
    capsys: pytest.CaptureFixture[str],
) -> None:
    avg, label = indicators.correlation_concentration("AAA", [], _fetch_from({}))
    assert (avg, label) == (None, "unknown")
    assert "active name" in capsys.readouterr().err


def test_correlation_concentration_unknown_when_every_fetch_fails(
    capsys: pytest.CaptureFixture[str],
) -> None:
    avg, label = indicators.correlation_concentration(
        "AAA", ["AAA", "BBB"], lambda _n: None,
    )
    assert (avg, label) == (None, "unknown")
    assert "insufficient data" in capsys.readouterr().err


def test_correlation_concentration_fail_soft_on_raising_fetch(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def boom(_name: str) -> pd.Series | None:
        raise RuntimeError("network down")

    avg, label = indicators.correlation_concentration("AAA", ["AAA", "BBB"], boom)
    assert (avg, label) == (None, "unknown")
    err = capsys.readouterr().err
    assert "fetch failed" in err  # per-name failure logged, never raised


def test_correlation_concentration_unknown_on_insufficient_overlap() -> None:
    # one close per name -> no returns to correlate -> undefined -> unknown
    data = {"AAA": pd.Series([100.0]), "BBB": pd.Series([100.0])}
    avg, label = indicators.correlation_concentration(
        "AAA", ["AAA", "BBB"], _fetch_from(data),
    )
    assert (avg, label) == (None, "unknown")


def test_correlation_memo_avoids_refetch_until_cleared() -> None:
    common = pd.Series([100.0, 101.0, 103.0, 106.0, 110.0])
    data = {"AAA": common, "BBB": common, "CCC": common}
    calls: list[str] = []

    def counting_fetch(name: str) -> pd.Series | None:
        calls.append(name)
        return data.get(name)

    universe = ["AAA", "BBB", "CCC"]
    indicators.correlation_concentration("AAA", universe, counting_fetch)
    after_first = len(calls)
    assert after_first == 3  # fetched each name once

    indicators.correlation_concentration("BBB", universe, counting_fetch)
    assert len(calls) == after_first  # same universe -> memo reused, no refetch

    indicators.clear_correlation_cache()
    indicators.correlation_concentration("CCC", universe, counting_fetch)
    assert len(calls) > after_first  # refetched after the cache was cleared
