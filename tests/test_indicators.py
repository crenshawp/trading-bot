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
