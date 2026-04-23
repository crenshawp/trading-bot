# trading_bot.py
# trading_bot.py
import yfinance as yf
import pandas as pd
import numpy as np
import schedule
import time
import requests
from datetime import datetime
from zoneinfo import ZoneInfo
import os


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

# Stock watchlist — backtested, Tier 1 only
STOCK_WATCHLIST = ["BLK", "GOOGL", "META", "GS", "NOW", "AMZN", "LLY", "TSLA"]
TREND_CONT_TICKERS = ["JPM", "GS", "GOOGL", "NOW", "SPY", "BLK", "AMZN"]

# Crypto watchlist — backtested on hourly candles
CRYPTO_WATCHLIST = ["BTC-USD", "BNB-USD", "ETH-USD"]

NOTIFY_METHOD    = "pushover"
DISCORD_WEBHOOK  = "https://discordapp.com/api/webhooks/1489655770734657718/LBrV6158qOKFCacoppOqkrRZroOBeyJOaIbR09DDXJZkyHUjHPQW0it_-kWdTCWWtiDL"
NEWSAPI_KEY      = "uokzEBhMdwXBGl68kurGxZUi1cunwftDIYdZmd8h"
PUSHOVER_TOKEN   = "aay3zwof4v2ubykk5gbusm52e7pmme"
PUSHOVER_USER    = "u7oonukvyknw5131jix4wgip9gb9dw"

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


def calculate_regression_slope(series, period=20):
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


def add_stock_indicators(df):
    df["RSI"]       = calculate_rsi(df["Close"])
    df["EMA21"]     = calculate_ema(df["Close"], 21)
    df["EMA50"]     = calculate_ema(df["Close"], 50)
    df["ATR"]       = calculate_atr(df)
    df["ROC_Accel"] = calculate_roc_acceleration(df["Close"])
    df["Slope"]     = calculate_regression_slope(df["Close"])
    df["Vol_MA20"]  = df["Volume"].rolling(20).mean()
    df["Slope"]       = calculate_regression_slope(df["Close"])
    df["Slope_Accel"] = df["Slope"].diff()
    df["High_20"]     = df["High"].rolling(20).max().shift(1)
    df.dropna(inplace=True)
    return df


def add_crypto_indicators(df):
    df["RSI"]         = calculate_rsi(df["Close"])
    df["EMA50"]       = calculate_ema(df["Close"], 50)
    df["ATR"]         = calculate_atr(df)
    df["ATR_MA20"]    = df["ATR"].rolling(20).mean()
    df["ROC_Accel"]   = calculate_roc_acceleration(df["Close"])
    df["Vol_MA20"]    = df["Volume"].rolling(20).mean()
    df["BB_upper"], df["BB_mid"], df["BB_lower"] = calculate_bbands(df["Close"])
    df["Recent_High"] = df["High"].rolling(20).max()
    df["Recent_Low"]  = df["Low"].rolling(20).min()
    df.dropna(inplace=True)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════

def get_stock_data(ticker):
    df = yf.download(ticker, period="60d", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    df.dropna(inplace=True)
    return df


def get_crypto_data(ticker):
    df = yf.download(ticker, period="60d", interval="1h", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    df.dropna(inplace=True)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# MARKET HOURS
# ══════════════════════════════════════════════════════════════════════════════

def is_market_open():
    now = datetime.now(ZoneInfo("America/New_York"))
    if now.weekday() >= 5:
        return False
    holidays = [
        
        "2026-01-01", "2026-01-19", "2026-02-16",
        "2026-04-03", "2026-05-25", "2026-06-19",
        "2026-07-03", "2026-09-07", "2026-11-26",
        "2026-12-25",
    ]
    if now.strftime("%Y-%m-%d") in holidays:
        return False
    market_open  = now.replace(hour=9,  minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0,  second=0, microsecond=0)
    return market_open <= now <= market_close


# ══════════════════════════════════════════════════════════════════════════════
# RISK CHECKS — stocks only
# ══════════════════════════════════════════════════════════════════════════════

def check_earnings_risk(ticker):
    etfs = ["SPY", "QQQ", "IWM", "DIA", "VXX", "GLD", "SLV", "TLT"]
    if ticker in etfs:
        return "LOW"
    try:
        stock    = yf.Ticker(ticker)
        calendar = stock.calendar
        if calendar is None or not isinstance(calendar, dict):
            return "UNKNOWN"
        if "Earnings Date" not in calendar:
            return "UNKNOWN"
        earnings_date = pd.Timestamp(calendar["Earnings Date"][0]).tz_localize(None)
        days_away     = (earnings_date - pd.Timestamp(datetime.now())).days
        if days_away <= 7:
            return f"HIGH — Earnings in {days_away} days"
        elif days_away <= 14:
            return f"MEDIUM — Earnings in {days_away} days"
        return "LOW"
    except Exception:
        return "UNKNOWN"


def check_news_risk(ticker):
    try:
        yesterday = (datetime.now() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        url = (
            f"https://newsapi.org/v2/everything"
            f"?q={ticker} stock"
            f"&from={yesterday}"
            f"&sortBy=publishedAt"
            f"&language=en"
            f"&apiKey={NEWSAPI_KEY}"
        )
        response = requests.get(url)
        articles = response.json().get("articles", [])
        if not articles:
            return "UNKNOWN"

        HIGH_RISK_WORDS   = ["crash", "lawsuit", "SEC", "fraud", "investigation",
                             "bankruptcy", "recall", "hack", "breach", "downgrade"]
        MEDIUM_RISK_WORDS = ["volatility", "uncertainty", "warning", "concern",
                             "miss", "disappoints", "selloff", "drop"]
        high_hits   = []
        medium_hits = []

        for article in articles:
            headline = article.get("title", "").lower()
            for word in HIGH_RISK_WORDS:
                if word in headline:
                    high_hits.append(word)
            for word in MEDIUM_RISK_WORDS:
                if word in headline:
                    medium_hits.append(word)

        if high_hits:
            return f"HIGH — Risky news detected: {', '.join(set(high_hits))}"
        elif medium_hits:
            return f"MEDIUM — Cautionary news detected: {', '.join(set(medium_hits))}"
        return "LOW"
    except Exception:
        return "UNKNOWN"


def estimate_hold_days(price, atr):
    atr_pct = (atr / price) * 100
    if atr_pct > 3:
        return "1-2 days"
    elif atr_pct > 1.5:
        return "2-3 days"
    return "3-5 days"


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL LOGGING
# ══════════════════════════════════════════════════════════════════════════════

def log_signal(signal):
    log = pd.DataFrame([{
        "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ticker":      signal["ticker"],
        "asset_type":  signal["asset_type"],
        "trade_type":  signal["trade_type"],
        "direction":   signal["direction"],
        "setup":       signal["setup"],
        "price":       signal["price"],
        "take_profit": signal["take_profit"],
        "stop_loss":   signal["stop_loss"],
        "confidence":  signal["confidence"],
    }])
    file_exists = os.path.exists("signal_log.csv")
    log.to_csv("signal_log.csv", mode="a", header=not file_exists, index=False)


# ══════════════════════════════════════════════════════════════════════════════
# STOCK SIGNALS — EMA21 Pullback
# Proven 56-76% win rate across Tier 1 watchlist
# ══════════════════════════════════════════════════════════════════════════════

def detect_stock_signals(ticker, df):
    signals = []

    latest = df.iloc[-1]
    prev   = df.iloc[-2]

    price       = float(latest["Close"])
    rsi         = float(latest["RSI"])
    atr         = float(latest["ATR"])
    volume      = float(latest["Volume"])
    vol_ma      = float(latest["Vol_MA20"])
    ema21       = float(latest["EMA21"])
    ema50       = float(latest["EMA50"])
    slope       = float(latest["Slope"])
    roc_a       = float(latest["ROC_Accel"])
    slope_accel = float(latest["Slope_Accel"])
    high_20     = float(latest["High_20"])

    hold_days  = estimate_hold_days(price, atr)
    near_ema21 = abs(price - ema21) / ema21 < 0.015

    tp_call = round(price + (atr * 2),   2)
    sl_call = round(price - (atr * 1.5), 2)
    tp_put  = round(price - (atr * 2),   2)
    sl_put  = round(price + (atr * 1.5), 2)

    # ── Earnings check ────────────────────────────────────────────────────────
    earnings_risk = check_earnings_risk(ticker)
    if "HIGH" in earnings_risk:
        signals.append({
            "ticker":     ticker,
            "asset_type": "stock",
            "trade_type": "📆 SWING TRADE",
            "direction":  "⚠️ WARNING",
            "setup":      "Earnings Risk",
            "detail":     earnings_risk,
            "price":      price,
            "confidence": "N/A",
            "take_profit": 0.0,
            "stop_loss":   0.0,
            "hold_days":  hold_days,
        })
        return signals

    # ── News check ────────────────────────────────────────────────────────────
    news_risk = check_news_risk(ticker)
    if "HIGH" in news_risk:
        signals.append({
            "ticker":     ticker,
            "asset_type": "stock",
            "trade_type": "📆 SWING TRADE",
            "direction":  "⚠️ WARNING",
            "setup":      "High Risk News",
            "detail":     news_risk,
            "price":      price,
            "confidence": "N/A",
            "take_profit": 0.0,
            "stop_loss":   0.0,
            "hold_days":  hold_days,
        })
        return signals

    # ── CALL — EMA21 Pullback in uptrend ─────────────────────────────────────
    if (price > ema50
            and slope > -0.5
            and near_ema21
            and 40 <= rsi <= 58
            and roc_a > 0
            and volume > vol_ma * 0.7):
        signals.append({
            "ticker":     ticker,
            "asset_type": "stock",
            "trade_type": "📆 SWING TRADE",
            "direction":  "CALL 📈",
            "setup":      "EMA21 Pullback",
            "detail":     f"Price pulled back to EMA21 in uptrend. RSI at {rsi:.1f}, momentum recovering.",
            "price":      price,
            "confidence": "High",
            "take_profit": tp_call,
            "stop_loss":   sl_call,
            "hold_days":  hold_days,
        })

    # ── PUT — EMA21 Pullback in downtrend ────────────────────────────────────
    if (price < ema50
            and slope < 0.5
            and near_ema21
            and 42 <= rsi <= 60
            and roc_a < 0
            and volume > vol_ma * 0.7):
        signals.append({
            "ticker":     ticker,
            "asset_type": "stock",
            "trade_type": "📆 SWING TRADE",
            "direction":  "PUT 📉",
            "setup":      "EMA21 Pullback",
            "detail":     f"Price bounced to EMA21 resistance in downtrend. RSI at {rsi:.1f}, momentum fading.",
            "price":      price,
            "confidence": "High",
            "take_profit": tp_put,
            "stop_loss":   sl_put,
            "hold_days":  hold_days,
        })

    # ── Trend Acceleration ────────────────────────────────────────────────────
    if (price > ema50
            and slope > 2.0
            and slope_accel > 0
            and roc_a > 0
            and volume > vol_ma * 1.3):
        signals.append({
            "ticker":     ticker,
            "asset_type": "stock",
            "trade_type": "📆 SWING TRADE",
            "direction":  "CALL 📈",
            "setup":      "Trend Acceleration",
            "detail":     f"Strong upward momentum detected. Slope accelerating with {volume/vol_ma:.1f}x volume.",
            "price":      price,
            "confidence": "High",
            "take_profit": tp_call,
            "stop_loss":   sl_call,
            "hold_days":  hold_days,
        })

    # ── Higher High Breakout ──────────────────────────────────────────────────
    if (price >= high_20
            and rsi >= 50
            and slope > 1.0
            and volume > vol_ma * 1.5):
        signals.append({
            "ticker":     ticker,
            "asset_type": "stock",
            "trade_type": "📆 SWING TRADE",
            "direction":  "CALL 📈",
            "setup":      "Higher High Breakout",
            "detail":     f"Price breaking 20 day high at ${high_20:.2f} with strong volume.",
            "price":      price,
            "confidence": "High",
            "take_profit": tp_call,
            "stop_loss":   sl_call,
            "hold_days":  hold_days,
        })
    # ── Trend Continuation ─────────────────────────────────────────────────────
# Price above EMA21 (short term trend intact)
# Price above EMA50 (long term trend intact)  
# Slope positive (trend confirmed)
# RSI 50-65 (strong but not overextended)
# Volume average or above (real participation)
    if ticker in TREND_CONT_TICKERS:
        if (price > ema21
                and price > ema50
                and slope > 0
                and 50 <= rsi <= 65
                and volume > vol_ma
                and roc_a > 0):
            signals.append({
             "ticker":     ticker,
             "asset_type": "stock",
             "trade_type": "📆 SWING TRADE",
             "direction":  "CALL 📈",
             "setup":      "Trend Continuation",
             "detail":     f"Established uptrend with healthy momentum. RSI at {rsi:.1f}.",
             "price":      price,
             "confidence": "Medium",
             "take_profit": tp_call,
             "stop_loss":   sl_call,
             "hold_days":  hold_days,
    })

    return signals

# ══════════════════════════════════════════════════════════════════════════════
# CRYPTO SIGNALS — Oversold Reversal + Momentum Breakout + Overbought Reversal
# Proven 56-61% win rate on BTC, BNB, ETH on hourly candles
# ══════════════════════════════════════════════════════════════════════════════

def detect_crypto_signals(ticker, df):
    signals = []

    latest = df.iloc[-1]
    prev   = df.iloc[-2]

    price       = float(latest["Close"])
    rsi         = float(latest["RSI"])
    prev_rsi    = float(prev["RSI"])
    atr         = float(latest["ATR"])
    atr_ma      = float(latest["ATR_MA20"])
    volume      = float(latest["Volume"])
    vol_ma      = float(latest["Vol_MA20"])
    bb_upper    = float(latest["BB_upper"])
    bb_lower    = float(latest["BB_lower"])
    recent_high = float(latest["Recent_High"])
    recent_low  = float(latest["Recent_Low"])
    roc_a       = float(latest["ROC_Accel"])
    prev_atr    = float(prev["ATR"])

    tp_long  = round(price + (atr * 2.5), 2)
    sl_long  = round(price - (atr * 1.5), 2)
    tp_short = round(price - (atr * 2.5), 2)
    sl_short = round(price + (atr * 1.5), 2)

    # ── Oversold Reversal — LONG ──────────────────────────────────────────────
    if (prev_rsi < 25
            and rsi > prev_rsi
            and price <= bb_lower * 1.015
            and volume > vol_ma * 1.5
            and price > recent_low):
        signals.append({
            "ticker":     ticker,
            "asset_type": "crypto",
            "trade_type": "⚡ CRYPTO TRADE",
            "direction":  "LONG 📈",
            "setup":      "Oversold Reversal",
            "detail":     f"RSI recovering from oversold ({prev_rsi:.1f} → {rsi:.1f}). Price near lower BB with volume spike.",
            "price":      price,
            "confidence": "High",
            "take_profit": tp_long,
            "stop_loss":   sl_long,
            "hold_days":  "2-8 hours",
        })

    # ── Momentum Breakout — LONG ──────────────────────────────────────────────
    if (price > recent_high
            and volume > vol_ma * 2
            and 55 <= rsi <= 72
            and roc_a > 0
            and prev_atr < atr_ma):
        signals.append({
            "ticker":     ticker,
            "asset_type": "crypto",
            "trade_type": "⚡ CRYPTO TRADE",
            "direction":  "LONG 📈",
            "setup":      "Momentum Breakout",
            "detail":     f"Price breaking 20hr high with {volume/vol_ma:.1f}x volume. RSI at {rsi:.1f}.",
            "price":      price,
            "confidence": "High",
            "take_profit": tp_long,
            "stop_loss":   sl_long,
            "hold_days":  "2-8 hours",
        })

    # ── Overbought Reversal — SHORT ───────────────────────────────────────────
    if (prev_rsi > 75
            and rsi < prev_rsi
            and price >= bb_upper * 0.985
            and volume > vol_ma * 1.5
            and price < recent_high):
        signals.append({
            "ticker":     ticker,
            "asset_type": "crypto",
            "trade_type": "⚡ CRYPTO TRADE",
            "direction":  "SHORT 📉",
            "setup":      "Overbought Reversal",
            "detail":     f"RSI pulling back from overbought ({prev_rsi:.1f} → {rsi:.1f}). Price near upper BB.",
            "price":      price,
            "confidence": "Medium",
            "take_profit": tp_short,
            "stop_loss":   sl_short,
            "hold_days":  "2-8 hours",
        })

    return signals


# ══════════════════════════════════════════════════════════════════════════════
# NOTIFICATIONS
# ══════════════════════════════════════════════════════════════════════════════

def send_notification(signal):
    try:
        log_signal(signal)
    except Exception as e:
        print(f"  Log error: {e}")

    msg = (
        f"🚨 TRADE ALERT — {signal['ticker']}\n"
        f"Asset:       {signal['asset_type'].upper()}\n"
        f"Type:        {signal['trade_type']}\n"
        f"Direction:   {signal['direction']}\n"
        f"Setup:       {signal['setup']}\n"
        f"Detail:      {signal['detail']}\n"
        f"Price:       ${signal['price']:.2f}\n"
        f"Take Profit: ${signal['take_profit']:.2f}\n"
        f"Stop Loss:   ${signal['stop_loss']:.2f}\n"
        f"Hold Time:   {signal['hold_days']}\n"
        f"Confidence:  {signal['confidence']}\n"
        f"Time:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    )

    try:
        requests.post(DISCORD_WEBHOOK, json={
            "content": "@everyone 🚨 TRADE ALERT",
            "embeds": [{
                "title": f"🚨 TRADE ALERT — {signal['ticker']}",
                "color": 3066993,
                "fields": [
                    {"name": "Asset",       "value": signal['asset_type'].upper(), "inline": True},
                    {"name": "Type",        "value": signal['trade_type'],         "inline": True},
                    {"name": "Direction",   "value": signal['direction'],          "inline": True},
                    {"name": "Setup",       "value": signal['setup'],              "inline": False},
                    {"name": "Detail",      "value": signal['detail'],             "inline": False},
                    {"name": "Price",       "value": f"${signal['price']:.2f}",    "inline": True},
                    {"name": "Take Profit", "value": f"${signal['take_profit']:.2f}", "inline": True},
                    {"name": "Stop Loss",   "value": f"${signal['stop_loss']:.2f}",   "inline": True},
                    {"name": "Hold Time",   "value": signal['hold_days'],          "inline": True},
                    {"name": "Confidence",  "value": signal['confidence'],         "inline": True},
                    {"name": "Time",        "value": datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "inline": True},
                ]
            }]
        })
        print(f"  Discord notification sent")
    except Exception as e:
        print(f"  Discord error: {e}")

    try:
        requests.post("https://api.pushover.net/1/messages.json", data={
            "token":   PUSHOVER_TOKEN,
            "user":    PUSHOVER_USER,
            "title":   f"Alert: {signal['ticker']} {signal['direction']}",
            "message": msg,
        })
        print(f"  Pushover notification sent")
    except Exception as e:
        print(f"  Pushover error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# SCAN FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def scan_stocks():
    """Runs once per day at market open — daily candles, swing trades"""
    if not is_market_open():
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Market closed — skipping stock scan")
        return

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Scanning stocks...")
    for ticker in STOCK_WATCHLIST:
        try:
            df = get_stock_data(ticker)
            if df is None:
                print(f"  {ticker}: No data")
                continue
            df      = add_stock_indicators(df)
            signals = detect_stock_signals(ticker, df)
            for s in signals:
                send_notification(s)
            if not signals:
                print(f"  {ticker}: No signals")
        except Exception as e:
            print(f"  {ticker}: Error — {e}")


def scan_crypto():
    """Runs every hour, 24/7 — hourly candles, short term trades"""
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Scanning crypto...")
    for ticker in CRYPTO_WATCHLIST:
        try:
            df = get_crypto_data(ticker)
            if df is None:
                print(f"  {ticker}: No data")
                continue
            df      = add_crypto_indicators(df)
            signals = detect_crypto_signals(ticker, df)
            for s in signals:
                send_notification(s)
            if not signals:
                print(f"  {ticker}: No signals")
        except Exception as e:
            print(f"  {ticker}: Error — {e}")


# ══════════════════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Trading Bot Started.")
    print(f"Stocks:  {STOCK_WATCHLIST}")
    print(f"Crypto:  {CRYPTO_WATCHLIST}")
    print("Stock scan: once daily at market open")
    print("Crypto scan: every hour 24/7")

    # Run both immediately on start
    scan_stocks()
    scan_crypto()

    # Stock scan — once per day at 9:31am EST
    schedule.every().day.at("09:31").do(scan_stocks)

    # Crypto scan — every hour
    schedule.every(1).hours.do(scan_crypto)

    while True:
        schedule.run_pending()
        time.sleep(1)