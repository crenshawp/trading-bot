"""Advanced quantitative indicator families — Phase 6.

Five GENUINELY INDEPENDENT families, each measuring something the others do
not, computed on the same OHLCV candle structure the scanner already uses
(a pandas ``DataFrame`` with ``Open/High/Low/Close/Volume`` columns and a
``DatetimeIndex``):

1. **Volatility** — ATR, a realized-volatility envelope, and a low/normal/high
   ``vol_regime`` (priority family; feeds Phase 7 position sizing).
2. **Momentum** — RSI, one representative oscillator.
3. **Trend strength** — ADX, distinct from EMA *direction*: it measures whether
   a trend is strong enough to trust, not which way it points.
4. **Volume** — OBV, a cumulative volume-flow measure beyond the scanner's
   existing volume-ratio check.
5. **Correlation** — rolling cross-asset correlation among the currently-active
   watchlist names (a concentration measure), in :mod:`correlation` helpers
   below.

The families are kept independent ON PURPOSE: a pile of correlated oscillators
would corrupt the per-pair statistics with multiple-comparisons false
positives. Everything here is pure (no I/O, no globals) except the correlation
helpers, which take their price data dependency-injected so this module never
imports the scanner. All functions operate on the legacy candle structure so
new indicators reuse it rather than re-fetching.
"""

from __future__ import annotations

import pandas as pd

from trading_bot import config

# ══════════════════════════════════════════════════════════════════════════════
# VOLATILITY FAMILY — ATR, realized-vol bands, vol regime
# ══════════════════════════════════════════════════════════════════════════════


def true_range(df: pd.DataFrame) -> pd.Series:
    """Per-candle True Range = max(H-L, |H-prevC|, |L-prevC|).

    The first row has no previous close, so the two close-based legs are NaN
    and ``max`` (skipna) falls back to the high-low range — matching the
    legacy scanner's ATR convention.
    """
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return pd.Series(tr, index=df.index, name="true_range")


def atr(df: pd.DataFrame, period: int = config.VOL_ATR_PERIOD) -> pd.Series:
    """Average True Range — rolling mean of True Range over ``period`` candles.

    Simple-moving-average smoothing, identical to the scanner's
    ``calculate_atr`` so the advisory ATR matches the value the signal logic
    already reasons about.
    """
    return true_range(df).rolling(period).mean()


def realized_volatility(
    close: pd.Series, period: int = config.VOL_REALIZED_PERIOD
) -> pd.Series:
    """Realized volatility = population std of the last ``period`` simple returns.

    A *fractional* number (e.g. 0.02 == 2% typical move), distinct from ATR
    (an absolute price distance) and from Bollinger Bands (std of price level).
    """
    returns = close.pct_change(fill_method=None)
    return returns.rolling(period).std(ddof=0)


def realized_vol_bands(
    close: pd.Series,
    period: int = config.VOL_REALIZED_PERIOD,
    num_std: float = config.VOL_BAND_NUM_STD,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Expected-move envelope around the SMA, scaled by realized volatility.

    Returns ``(upper, mid, lower)`` where ``mid`` is the ``period`` SMA and the
    band half-width is ``num_std × realized_volatility`` as a fraction of mid.
    Unlike Bollinger Bands (which use the std of the price *level*), this scales
    by the std of *returns*, so it is a genuinely different volatility lens.
    """
    mid = close.rolling(period).mean()
    rv = realized_volatility(close, period)
    upper = mid * (1.0 + num_std * rv)
    lower = mid * (1.0 - num_std * rv)
    return upper, mid, lower


def classify_vol_regime(
    atr_now: float | None,
    atr_baseline: float | None,
    *,
    low_ratio: float = config.VOL_REGIME_LOW_RATIO,
    high_ratio: float = config.VOL_REGIME_HIGH_RATIO,
) -> str:
    """Classify current volatility as ``low`` / ``normal`` / ``high``.

    The regime is the ratio of the current ATR to its own longer baseline:
    below ``low_ratio`` the market is unusually quiet, above ``high_ratio`` it
    is unusually violent. A missing or non-positive baseline (too little data)
    yields ``unknown`` — the caller never raises on it.
    """
    if (
        atr_now is None
        or atr_baseline is None
        or atr_baseline <= 0.0
    ):
        return "unknown"
    ratio = atr_now / atr_baseline
    if ratio < low_ratio:
        return "low"
    if ratio > high_ratio:
        return "high"
    return "normal"
