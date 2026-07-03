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
   watchlist names (a concentration measure); the correlation helpers below
   take their price data dependency-injected.

The families are kept independent ON PURPOSE: a pile of correlated oscillators
would corrupt the per-pair statistics with multiple-comparisons false
positives. Everything here is pure (no I/O, no globals) except the correlation
helpers, which take their price data dependency-injected so this module never
imports the scanner. All functions operate on the legacy candle structure so
new indicators reuse it rather than re-fetching.

MULTIPLE-COMPARISONS CAVEAT — READ BEFORE TRUSTING ANY SINGLE INDICATOR.
These families are ADVISORY context only: they are computed and stored next to
the eventual trade outcome so they can be evaluated LATER, on accumulated
evidence. They do NOT create or block signals in this phase. Five families
tested at once means five chances to find a spurious edge; an indicator that
looks predictive on a small sample is most likely noise. Any apparent edge must
clear the same per-pair minimum-sample and windowed-expectancy bar as every
other gating decision (see trading_bot.signal_pairs / watchlist_state) before it
is believed — that is a job for Phase 9, on real accumulated data, not for a
hunch off a handful of trades. Keeping the families independent is what keeps
that future evaluation statistically honest.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass

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


# ══════════════════════════════════════════════════════════════════════════════
# MOMENTUM FAMILY — RSI (one representative oscillator)
# ══════════════════════════════════════════════════════════════════════════════


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average over ``period`` observations.

    The long-horizon TREND line for the Phase 14 buy-and-hold entry/exit logic
    (``price > sma(close, TREND_PERIOD)``). A plain rolling mean — the trend
    primitive that the momentum/trend/volatility families above did not provide;
    reused rather than reimplemented per caller.
    """
    return series.rolling(period).mean()


def rsi(close: pd.Series, period: int = config.RSI_PERIOD) -> pd.Series:
    """Relative Strength Index over ``period`` candles.

    SMA-smoothed gains/losses, identical to the scanner's ``calculate_rsi`` so
    the advisory momentum reading matches what the signal logic already uses.
    An all-gains window divides by a zero average loss -> RS is +inf -> RSI 100;
    a perfectly flat window is 0/0 -> NaN (genuinely undefined).

    RSI is the SINGLE momentum representative for this phase by design — adding
    Stochastic / Williams %R / CCI would be the same family and corrupt the
    per-pair statistics with correlated duplicates.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


# ══════════════════════════════════════════════════════════════════════════════
# TREND-STRENGTH FAMILY — ADX (distinct from EMA direction)
# ══════════════════════════════════════════════════════════════════════════════


def adx(df: pd.DataFrame, period: int = config.ADX_PERIOD) -> pd.Series:
    """Average Directional Index — trend STRENGTH, not direction.

    ADX answers "is the trend strong enough to trust?" and is deliberately
    direction-agnostic: a steady up-move and a steady down-move both score high.
    That is what makes it independent of the scanner's existing EMA *direction*
    checks rather than a restatement of them.

    Wilder's +DM / -DM / DX construction with SMA smoothing (consistent with
    this module's ATR). ``DX`` is undefined (NaN) on a candle with no directional
    movement; the rolling mean skips those.
    """
    up_move = df["High"].diff()
    down_move = -df["Low"].diff()

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0.0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0.0), 0.0)

    atr_s = true_range(df).rolling(period).mean()
    plus_di = 100.0 * plus_dm.rolling(period).mean() / atr_s
    minus_di = 100.0 * minus_dm.rolling(period).mean() / atr_s

    di_sum = plus_di + minus_di
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum
    return dx.rolling(period).mean()


# ══════════════════════════════════════════════════════════════════════════════
# VOLUME FAMILY — OBV (cumulative volume flow)
# ══════════════════════════════════════════════════════════════════════════════


def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume — running total that adds the candle's volume on an up
    close and subtracts it on a down close (flat closes carry the prior total).

    This measures the DIRECTION of volume flow (is volume confirming the move?),
    which is a different question than the scanner's existing volume-vs-MA
    *ratio* check (is there enough volume?). The series starts at 0 on the first
    candle, so only its slope/changes are meaningful — the absolute level is
    anchored to the start of the window.
    """
    delta = df["Close"].diff()
    direction = (delta > 0.0).astype(float) - (delta < 0.0).astype(float)
    signed_volume = direction * df["Volume"]
    return signed_volume.cumsum()


# ══════════════════════════════════════════════════════════════════════════════
# CORRELATION FAMILY — cross-asset concentration among active watchlist names
# ══════════════════════════════════════════════════════════════════════════════
#
# This family is the odd one out: it is not a per-ticker series but a property
# of the active SET. It answers "are the names we're trading all moving together
# (a hidden single-bet) or genuinely diversified?". The price data is injected
# (a fetch callable + the active universe) so this module never imports the
# scanner; a per-scan memo avoids refetching the universe on every fired signal.

# Memo of the built returns frame, keyed by the sorted active universe. Cleared
# at the top of each scan via clear_correlation_cache() — mirrors the news cache.
_returns_cache: dict[tuple[str, ...], pd.DataFrame] = {}


def clear_correlation_cache() -> None:
    """Drop the per-scan correlation memo. Called at the start of a scan cycle."""
    _returns_cache.clear()


def average_pairwise_correlation(
    returns: pd.DataFrame, ticker: str
) -> float | None:
    """Mean correlation of ``ticker``'s returns against every other column.

    Pure. Returns ``None`` when it cannot be computed — ``ticker`` absent, no
    peers, or every pairwise correlation undefined (constant/too-short series).
    """
    if ticker not in returns.columns:
        return None
    others = [c for c in returns.columns if c != ticker]
    if not others:
        return None
    corr_row = returns.corr().loc[ticker, others].dropna()
    if corr_row.empty:
        return None
    return float(corr_row.mean())


def classify_concentration(
    avg_corr: float | None,
    *,
    concentrated_at: float = config.CORR_CONCENTRATED_AT,
    diversified_at: float = config.CORR_DIVERSIFIED_AT,
) -> str:
    """Bucket an average pairwise correlation into a concentration label.

    High positive correlation == the active names move together == a hidden
    single-bet (``concentrated``). Low/negative == genuine spread
    (``diversified``). ``None`` -> ``unknown``.
    """
    if avg_corr is None:
        return "unknown"
    if avg_corr >= concentrated_at:
        return "concentrated"
    if avg_corr < diversified_at:
        return "diversified"
    return "moderate"


def _build_active_returns(
    universe: Sequence[str],
    fetch_closes: Callable[[str], pd.Series | None],
    window: int,
) -> pd.DataFrame | None:
    """Fetch closes for the universe and build a trailing returns frame.

    Memoized by the sorted universe so a scan that fires several signals only
    pays the fetch once. A per-name fetch failure drops that name (logged) but
    never sinks the whole frame. Returns ``None`` if fewer than two names yield
    data.
    """
    key = tuple(sorted(universe))
    cached = _returns_cache.get(key)
    if cached is not None:
        return cached

    closes: dict[str, pd.Series] = {}
    for name in universe:
        try:
            series = fetch_closes(name)
        except Exception as exc:  # noqa: BLE001 - one name must not sink the frame
            print(f"  correlation: fetch failed for {name}: {exc}", file=sys.stderr)
            series = None
        if series is not None and not series.empty:
            # window+1 closes -> window returns after pct_change.
            closes[name] = series.tail(window + 1)

    if len(closes) < 2:
        return None

    frame = pd.DataFrame(closes).pct_change(fill_method=None)
    _returns_cache[key] = frame
    return frame


def correlation_concentration(
    ticker: str,
    active_tickers: Sequence[str],
    fetch_closes: Callable[[str], pd.Series | None],
    *,
    window: int = config.CORR_RETURN_WINDOW,
    min_names: int = config.CORR_MIN_NAMES,
) -> tuple[float | None, str]:
    """``(avg_corr, concentration_label)`` for ``ticker`` vs the active set.

    FAIL-SOFT: any shortfall — too few active names, every fetch failing, or
    insufficient overlapping data — returns ``(None, "unknown")`` and logs the
    reason. NEVER raises, so the correlation family can never block a signal.
    """
    try:
        universe = list(dict.fromkeys(active_tickers))
        if ticker not in universe:
            universe = [ticker, *universe]
        if len(universe) < min_names:
            print(
                f"  correlation: only {len(universe)} active name(s) — "
                f"unknown for {ticker}",
                file=sys.stderr,
            )
            return None, "unknown"

        returns = _build_active_returns(universe, fetch_closes, window)
        if returns is None or ticker not in returns.columns:
            print(
                f"  correlation: insufficient data for {ticker} — unknown",
                file=sys.stderr,
            )
            return None, "unknown"

        avg = average_pairwise_correlation(returns, ticker)
        if avg is None:
            print(
                f"  correlation: undefined for {ticker} — unknown",
                file=sys.stderr,
            )
        return avg, classify_concentration(avg)
    except Exception as exc:  # noqa: BLE001 - correlation must never block a signal
        print(
            f"  correlation: error for {ticker}, returning unknown: {exc}",
            file=sys.stderr,
        )
        return None, "unknown"


# ══════════════════════════════════════════════════════════════════════════════
# CONTEXT BUNDLE — the per-signal snapshot attached to a fired trade
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class IndicatorContext:
    """The five families snapshotted for one fired signal, advisory only.

    Per-ticker families (volatility/momentum/trend/volume) come from the scan's
    candle frame; the correlation pair is injected (it is a property of the
    active set, not of one ticker). ``ok`` is False whenever the per-ticker math
    fell through to a fail-soft empty bundle — callers persist it either way and
    never gate on it.
    """

    atr: float | None = None
    realized_vol: float | None = None
    vol_regime: str = "unknown"
    rsi: float | None = None
    adx: float | None = None
    obv: float | None = None
    correlation: float | None = None
    concentration: str = "unknown"
    ok: bool = False


def _value_at(series: pd.Series, at: int) -> float | None:
    """Latest valid scalar at position ``at``; None on out-of-range or NaN."""
    try:
        value = series.iloc[at]
    except (IndexError, KeyError):
        return None
    if pd.isna(value):
        return None
    return float(value)


def compute_context(
    df: pd.DataFrame,
    *,
    correlation: float | None = None,
    concentration: str = "unknown",
    at: int = -1,
    atr_period: int = config.VOL_ATR_PERIOD,
    baseline_period: int = config.VOL_ATR_BASELINE_PERIOD,
    realized_period: int = config.VOL_REALIZED_PERIOD,
    rsi_period: int = config.RSI_PERIOD,
    adx_period: int = config.ADX_PERIOD,
) -> IndicatorContext:
    """Snapshot the per-ticker families at candle ``at`` and bundle the injected
    correlation pair. FAIL-SOFT: any error (short frame, missing columns)
    returns a bundle carrying just the correlation pair with ``ok=False``; it
    never raises, so attaching context can never block a signal.

    ``at`` defaults to ``-1`` (latest candle); the stock scanner passes ``-2`` to
    match the last fully-closed candle that ``detect_stock_signals`` reasons on.
    """
    try:
        close = df["Close"]
        atr_series = atr(df, atr_period)
        atr_now = _value_at(atr_series, at)
        atr_baseline = _value_at(atr_series.rolling(baseline_period).mean(), at)
        return IndicatorContext(
            atr=atr_now,
            realized_vol=_value_at(realized_volatility(close, realized_period), at),
            vol_regime=classify_vol_regime(atr_now, atr_baseline),
            rsi=_value_at(rsi(close, rsi_period), at),
            adx=_value_at(adx(df, adx_period), at),
            obv=_value_at(obv(df), at),
            correlation=correlation,
            concentration=concentration,
            ok=True,
        )
    except Exception as exc:  # noqa: BLE001 - attaching context must never block a signal
        print(
            f"  indicators: context computation failed, attaching empty: {exc}",
            file=sys.stderr,
        )
        return IndicatorContext(
            correlation=correlation, concentration=concentration, ok=False,
        )
