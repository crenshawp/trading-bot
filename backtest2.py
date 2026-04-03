# backtest_crypto.py
# Crypto strategy — hourly candles, momentum based signals
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime


# ══════════════════════════════════════════════════════════════════════════════
# INDICATORS
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


def calculate_bbands(series, period=20, std_dev=2):
    middle = series.rolling(period).mean()
    std    = series.rolling(period).std()
    upper  = middle + (std_dev * std)
    lower  = middle - (std_dev * std)
    return upper, middle, lower


def calculate_roc(series, period=10):
    return series.diff(period) / series.shift(period) * 100


def calculate_roc_acceleration(series, period=10):
    return calculate_roc(series, period).diff()


def add_indicators(df):
    df["RSI"]       = calculate_rsi(df["Close"])
    df["EMA21"]     = calculate_ema(df["Close"], 21)
    df["EMA50"]     = calculate_ema(df["Close"], 50)
    df["ATR"]       = calculate_atr(df)
    df["ATR_MA20"]  = df["ATR"].rolling(20).mean()
    df["ROC_Accel"] = calculate_roc_acceleration(df["Close"])
    df["Vol_MA20"]  = df["Volume"].rolling(20).mean()
    df["BB_upper"], df["BB_mid"], df["BB_lower"] = calculate_bbands(df["Close"])
    df["Recent_High"] = df["High"].rolling(20).max()  # 20 hour recent high
    df["Recent_Low"]  = df["Low"].rolling(20).min()   # 20 hour recent low
    df.dropna(inplace=True)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# DATA — hourly candles, 60 days max (yfinance limit for hourly)
# ══════════════════════════════════════════════════════════════════════════════

def get_historical_data(ticker):
    df = yf.download(ticker, period="60d", interval="1h", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    df.dropna(inplace=True)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# OUTCOME CHECKER
# Looks forward 8 hourly candles — roughly 8 hours
# Tighter ATR multipliers for hourly timeframe
# ══════════════════════════════════════════════════════════════════════════════

def check_outcome(df, signal_index, take_profit, stop_loss, direction):
    future_candles = df.iloc[signal_index + 1 : signal_index + 9]
    for _, candle in future_candles.iterrows():
        high = float(candle["High"])
        low  = float(candle["Low"])
        if direction == "LONG":
            if high >= take_profit: return "WIN"
            if low  <= stop_loss:   return "LOSS"
        elif direction == "SHORT":
            if low  <= take_profit: return "WIN"
            if high >= stop_loss:   return "LOSS"
    return "UNKNOWN"


# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(ticker):
    print(f"\nBacktesting {ticker}...")

    df = get_historical_data(ticker)
    if df is None:
        print(f"{ticker}: No data")
        return None

    df      = add_indicators(df)
    results = []

    for i in range(50, len(df) - 9):
        current  = df.iloc[i]
        previous = df.iloc[i - 1]

        price       = float(current["Close"])
        rsi         = float(current["RSI"])
        prev_rsi    = float(previous["RSI"])
        atr         = float(current["ATR"])
        atr_ma      = float(current["ATR_MA20"])
        volume      = float(current["Volume"])
        vol_ma      = float(current["Vol_MA20"])
        ema21       = float(current["EMA21"])
        ema50       = float(current["EMA50"])
        roc_a       = float(current["ROC_Accel"])
        bb_upper    = float(current["BB_upper"])
        bb_lower    = float(current["BB_lower"])
        recent_high = float(current["Recent_High"])
        recent_low  = float(current["Recent_Low"])

        # Hourly ATR multipliers — tighter than daily
        tp_long  = round(price + (atr * 2.5), 2)
        sl_long  = round(price - (atr * 1.5), 2)
        tp_short = round(price - (atr * 2.5), 2)
        sl_short = round(price + (atr * 1.5), 2)

        date = df.index[i].strftime("%Y-%m-%d %H:%M")

        # ── SIGNAL 1 — Oversold Reversal ─────────────────────────────────────
        # RSI was below 25 and is now turning back up
        # Price near or below lower Bollinger Band
        # Volume spike confirms real buyers stepping in
        # Price above recent low — not in freefall
        if (prev_rsi < 25
                and rsi > prev_rsi
                and price <= bb_lower * 1.015
                and volume > vol_ma * 1.5
                and price > recent_low):
            outcome = check_outcome(df, i, tp_long, sl_long, "LONG")
            results.append({
                "date":        date,
                "ticker":      ticker,
                "setup":       "Oversold Reversal",
                "direction":   "LONG",
                "price":       price,
                "take_profit": tp_long,
                "stop_loss":   sl_long,
                "outcome":     outcome,
            })

        # ── SIGNAL 2 — Momentum Breakout ─────────────────────────────────────
        # Price breaks above 20 hour high — new momentum high
        # Volume 2x average — real buying behind the move
        # RSI between 55-72 — strong momentum not exhausted
        # ROC acceleration positive — momentum building
        # ATR was below average before breakout — coiling then expanding
        if (price > recent_high
                and volume > vol_ma * 2
                and 55 <= rsi <= 72
                and roc_a > 0
                and float(previous["ATR"]) < atr_ma):
            outcome = check_outcome(df, i, tp_long, sl_long, "LONG")
            results.append({
                "date":        date,
                "ticker":      ticker,
                "setup":       "Momentum Breakout",
                "direction":   "LONG",
                "price":       price,
                "take_profit": tp_long,
                "stop_loss":   sl_long,
                "outcome":     outcome,
            })

        # ── SIGNAL 3 — Overbought Reversal ───────────────────────────────────
        # RSI was above 75 and is now turning back down
        # Price near or above upper Bollinger Band
        # Volume spike confirms distribution
        # Price below recent high — momentum fading
        if (prev_rsi > 75
                and rsi < prev_rsi
                and price >= bb_upper * 0.985
                and volume > vol_ma * 1.5
                and price < recent_high):
            outcome = check_outcome(df, i, tp_short, sl_short, "SHORT")
            results.append({
                "date":        date,
                "ticker":      ticker,
                "setup":       "Overbought Reversal",
                "direction":   "SHORT",
                "price":       price,
                "take_profit": tp_short,
                "stop_loss":   sl_short,
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

    print("\n" + "="*50)
    print(f"BACKTEST RESULTS — {results['ticker'].iloc[0]}")
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
    CRYPTO = [
        "BTC-USD", "ETH-USD", "SOL-USD",
        "BNB-USD", "LINK-USD", "AVAX-USD",
        "ADA-USD", "XRP-USD"
    ]

    all_results = []
    for ticker in CRYPTO:
        result = run_backtest(ticker)
        if result is not None and not result.empty:
            print_summary(result)
            all_results.append(result)

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv("backtest_crypto_results.csv", index=False)
        print("\nFull results saved to backtest_crypto_results.csv")