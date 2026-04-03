# backtest.py
# Rebuilt around EMA21 Pullback — the only consistently profitable signal
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime


# ══════════════════════════════════════════════════════════════════════════════
# INDICATORS — only what the signal actually needs
# ══════════════════════════════════════════════════════════════════════════════

def calculate_rsi(series, period=14):
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs       = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def calculate_atr(df, period=14):
    high_low   = df["High"] - df["Low"]
    high_close = abs(df["High"] - df["Close"].shift(1))
    low_close  = abs(df["Low"] - df["Close"].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(period).mean()


def calculate_roc(series, period=10):
    return series.diff(period) / series.shift(period) * 100


def calculate_roc_acceleration(series, period=10):
    return calculate_roc(series, period).diff()


def calculate_regression_slope(series, period=20):
    """
    Fits a straight line to recent price data using least squares regression.
    Positive = uptrend, Negative = downtrend. Steeper = stronger trend.
    """
    slopes = []
    for i in range(len(series)):
        if i < period:
            slopes.append(np.nan)
            continue
        y     = series.iloc[i-period:i].values
        x     = np.arange(period)
        slope = np.polyfit(x, y, 1)[0]
        slopes.append(slope)
    return pd.Series(slopes, index=series.index)


def add_indicators(df):
    df["RSI"]       = calculate_rsi(df["Close"])
    df["EMA21"]     = calculate_ema(df["Close"], 21)
    df["EMA50"]     = calculate_ema(df["Close"], 50)
    df["ATR"]       = calculate_atr(df)
    df["ROC_Accel"] = calculate_roc_acceleration(df["Close"])
    df["Slope"]     = calculate_regression_slope(df["Close"])
    df["Vol_MA20"]  = df["Volume"].rolling(20).mean()
    df.dropna(inplace=True)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════

def get_historical_data(ticker):
    df = yf.download(ticker, period="3y", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    df.dropna(inplace=True)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# OUTCOME CHECKER
# Looks forward 10 candles after signal fires
# Checks if take profit or stop loss was hit first
# ══════════════════════════════════════════════════════════════════════════════

def check_outcome(df, signal_index, take_profit, stop_loss, direction):
    future_candles = df.iloc[signal_index + 1 : signal_index + 11]
    for _, candle in future_candles.iterrows():
        high = float(candle["High"])
        low  = float(candle["Low"])
        if direction == "CALL":
            if high >= take_profit: return "WIN"
            if low  <= stop_loss:   return "LOSS"
        elif direction == "PUT":
            if low  <= take_profit: return "WIN"
            if high >= stop_loss:   return "LOSS"
    return "UNKNOWN"


# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST
# asset_type="stock" uses normal ATR multipliers
# asset_type="crypto" uses wider ATR multipliers for higher volatility
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(ticker, asset_type="stock"):
    print(f"\nBacktesting {ticker}...")

    df = get_historical_data(ticker)
    if df is None:
        print(f"{ticker}: No data")
        return None

    df      = add_indicators(df)
    results = []

    for i in range(50, len(df) - 6):
        current  = df.iloc[i]
        previous = df.iloc[i - 1]

        price   = float(current["Close"])
        rsi     = float(current["RSI"])
        atr     = float(current["ATR"])
        volume  = float(current["Volume"])
        vol_ma  = float(current["Vol_MA20"])
        ema21   = float(current["EMA21"])
        ema50   = float(current["EMA50"])
        slope   = float(current["Slope"])
        roc_a   = float(current["ROC_Accel"])

        near_ema21 = abs(price - ema21) / ema21 < 0.015

        # ATR multipliers — wider for crypto, tighter for stocks
        if asset_type == "crypto":
            tp_call = round(price + (atr * 5),   2)
            sl_call = round(price - (atr * 2.5), 2)
            tp_put  = round(price - (atr * 5),   2)
            sl_put  = round(price + (atr * 2.5), 2)
        else:
            tp_call = round(price + (atr * 2),   2)
            sl_call = round(price - (atr * 1.5), 2)
            tp_put  = round(price - (atr * 2),   2)
            sl_put  = round(price + (atr * 1.5), 2)

        date = df.index[i].strftime("%Y-%m-%d")

        # ── CALL — EMA21 Pullback in uptrend ─────────────────────────────────
        if (price > ema50
                and slope > -0.5
                and near_ema21
                and 40 <= rsi <= 58
                and roc_a > 0
                and volume > vol_ma * 0.7):
            outcome = check_outcome(df, i, tp_call, sl_call, "CALL")
            results.append({
                "date":        date,
                "ticker":      ticker,
                "asset_type":  asset_type,
                "setup":       "EMA21 Pullback — CALL",
                "direction":   "CALL",
                "price":       price,
                "take_profit": tp_call,
                "stop_loss":   sl_call,
                "outcome":     outcome,
            })

        # ── PUT — EMA21 Pullback in downtrend ────────────────────────────────
        if (price < ema50
                and slope < 0.5
                and near_ema21
                and 42 <= rsi <= 60
                and roc_a < 0
                and volume > vol_ma * 0.7):
            outcome = check_outcome(df, i, tp_put, sl_put, "PUT")
            results.append({
                "date":        date,
                "ticker":      ticker,
                "asset_type":  asset_type,
                "setup":       "EMA21 Pullback — PUT",
                "direction":   "PUT",
                "price":       price,
                "take_profit": tp_put,
                "stop_loss":   sl_put,
                "outcome":     outcome,
            })

    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(results):
    if results is None or results.empty:
        print("No signals found")
        return

    known    = results[results["outcome"] != "UNKNOWN"]
    total    = len(known)
    wins     = len(known[known["outcome"] == "WIN"])
    losses   = len(known[known["outcome"] == "LOSS"])
    win_rate = (wins / total * 100) if total > 0 else 0

    asset = results["asset_type"].iloc[0].upper()

    print("\n" + "="*50)
    print(f"BACKTEST RESULTS — {results['ticker'].iloc[0]} ({asset})")
    print("="*50)
    print(f"Total Signals:  {total}")
    print(f"Wins:           {wins}")
    print(f"Losses:         {losses}")
    print(f"Win Rate:       {win_rate:.1f}%")
    print("="*50)

    print("\nRESULTS BY SETUP:")
    for setup in known["setup"].unique():
        subset  = known[known["setup"] == setup]
        s_wins  = len(subset[subset["outcome"] == "WIN"])
        s_total = len(subset)
        s_rate  = (s_wins / s_total * 100) if s_total > 0 else 0
        print(f"  {setup}: {s_wins}/{s_total} ({s_rate:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    STOCKS = [
        "GOOGL", "META", "AMZN", "MSFT", "JPM", "SPY", "NVDA",
        "V", "MA", "AVGO", "NOW", "NFLX", "CRM", "GS", "QQQ",
        "COST", "LLY", "UNH", "WDAY", "MS", "BLK", "ORCL",
        "HD", "AXP", "AAPL", "TSLA"
    ]

    CRYPTO = [
        "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "LINK-USD"
    ]

    all_results = []

    print("\n" + "="*50)
    print("RUNNING STOCK BACKTESTS")
    print("="*50)
    for ticker in STOCKS:
        result = run_backtest(ticker, asset_type="stock")
        if result is not None and not result.empty:
            print_summary(result)
            all_results.append(result)

    print("\n" + "="*50)
    print("RUNNING CRYPTO BACKTESTS")
    print("="*50)
    for ticker in CRYPTO:
        result = run_backtest(ticker, asset_type="crypto")
        if result is not None and not result.empty:
            print_summary(result)
            all_results.append(result)

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv("backtest_results.csv", index=False)
        print("\nFull results saved to backtest_results.csv")