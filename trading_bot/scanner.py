"""Live trading bot scanner.

Migrated from the legacy root-level ``tradingbot.py`` in Phase 1.2. The
signal-detection math, indicator functions, watchlists, and notification
formatting are intentionally **unchanged** from that file — only the
storage layer (CSV → SQLite via :mod:`trading_bot.db`) and the credential
reads (broken hardcoded references → :mod:`trading_bot.secrets`) have moved.

Strict mypy is deferred for this module (see ``pyproject.toml`` override and
``CLAUDE.md`` "Deferred cleanup") because annotating ~700 lines of legacy
pandas/yfinance code is its own phase. The new wiring added in 1.2
(``log_signal``, ``_load_secrets``, ``main``, the ``_normalize_*`` helpers)
is annotated regardless.
"""

import argparse
import os
import sys
import time
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import schedule
import yfinance as yf

from trading_bot import (
    config,
    context,
    db,
    earnings,
    evaluator_scheduling,
    indicators,
    news_client,
    order_lifecycle,
    outcomes,
    performance,
    predictions,
    readiness,
    regime,
    risk,
    risk_of_ruin,
    settings,
    vix,
)
from trading_bot import (
    sentiment as sentiment_mod,
)
from trading_bot.discovery_universe import SHADOW_UNIVERSE
from trading_bot.models import Signal, Trade
from trading_bot.secrets import get_required

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

# Stock watchlist — backtested, Tier 1 only
STOCK_WATCHLIST = ["BLK", "GOOGL", "META", "GS", "NOW", "AMZN", "LLY", "TSLA","AAPL", "MSFT", "NVDA", "TJX", "LRCX", "AMAT", "KLAC", "V", "XOM", "NFLX", "MA", "JNJ", "BRK.B", "GOOG", "ROST", "ANET", "MRK", "COST", "LIN", "FTNT", "WMT", "MU", "AMD", "AVGO", "CSCO", "ABBV", "PG", "HD", "PANW", "QCOM", "GE", "CAT", "KO", "WDC", "GEV", "GILD", "ETN", "TXN", "STX", "UTHR", "CRS", "MEDP", "CSL", "HEI", "TMO", "FFIV", "WMB", "MO", "VZ", "TMUS"]
TREND_CONT_TICKERS = ["JPM", "GS", "GOOGL", "NOW", "SPY", "BLK", "AMZN","AAPL", "MSFT", "NVDA", "TJX", "LRCX", "AMAT", "KLAC", "V", "XOM", "NFLX", "MA", "JNJ", "BRK.B", "GOOG", "ROST", "ANET", "MRK", "COST", "LIN", "FTNT", "WMT", "MU", "AMD", "AVGO", "CSCO", "ABBV", "PG", "HD", "PANW", "QCOM", "GE", "CAT", "KO", "WDC", "GEV", "GILD", "ETN", "TXN", "STX", "UTHR", "CRS", "MEDP", "CSL", "HEI", "TMO", "FFIV", "WMB", "MO", "VZ", "TMUS"]
NEWS_WATCHLIST = ["BLK", "GOOGL", "META", "GS", "NOW", "AMZN", "LLY", "TSLA", "PLTR", "NVDA", "AAPL"]

# Crypto watchlist — backtested on hourly candles
CRYPTO_WATCHLIST = ["BTC-USD", "BNB-USD", "ETH-USD"]

# Phase 3.1-LIVE: seconds to sleep between shadow-universe yfinance fetches.
# The shadow scan touches ~100 names; throttling keeps us under rate limits.
_SHADOW_THROTTLE_SECONDS = 0.5

NOTIFY_METHOD    = "pushover"

# One timeout for every legacy Discord/Pushover/NewsAPI call in this module.
# Without it a single hung socket stalls the (single-threaded) schedule loop
# indefinitely — violating the "never block signal capture" contract. The
# newer modules (news_client, readiness, broker.*) already do this.
_HTTP_TIMEOUT_SECONDS = 10

# Credential placeholders — populated by _load_secrets() at startup, not
# at import time, so this module can be imported by tests without keychain
# access. main() always calls _load_secrets() before any scan runs.
DISCORD_WEBHOOK_URL: str = ""
PUSHOVER_USER_KEY:   str = ""
PUSHOVER_APP_TOKEN:  str = ""
NEWSAPI_KEY:         str = ""

# When True, send_notification writes to the DB but suppresses Discord/Pushover
# calls and prints the alert text to stdout instead. Flipped on by --dry-run.
DRY_RUN: bool = False


def _active_stock_watchlist() -> list[str]:
    """Return the live stock watchlist from the database (Phase 3.1).

    The scanner reads its watchlist from the ``active_watchlist`` table so
    discovery can auto-promote tickers into the running bot. CRITICAL
    FALLBACK: if the table is empty, missing, or unreadable, fall back to the
    hardcoded ``STOCK_WATCHLIST`` seed and log a warning — the bot must never
    silently scan an empty watchlist.
    """
    try:
        active = db.get_active_watchlist()
    except Exception as exc:  # noqa: BLE001 - any DB failure must fall back, not crash
        print(
            f"  active_watchlist unreadable, falling back to seed list: {exc}",
            file=sys.stderr,
        )
        return list(STOCK_WATCHLIST)
    if not active:
        print(
            "  active_watchlist empty, falling back to seed list",
            file=sys.stderr,
        )
        return list(STOCK_WATCHLIST)
    return active


def _active_stock_watchlist_entries() -> list[tuple[str, str]]:
    """Return ``(ticker, status)`` for every watchlist row (Phase 3.3).

    Status is 'active' or 'benched'. CRITICAL FALLBACK (same contract as
    ``_active_stock_watchlist``): if the table is empty, missing, or unreadable
    fall back to the hardcoded seed list, all treated as 'active', and log a
    warning — the bot must never silently scan an empty watchlist.
    """
    try:
        entries = db.get_watchlist_entries()
    except Exception as exc:  # noqa: BLE001 - any DB failure must fall back, not crash
        print(
            f"  active_watchlist unreadable, falling back to seed list: {exc}",
            file=sys.stderr,
        )
        return [(t, "active") for t in STOCK_WATCHLIST]
    if not entries:
        print(
            "  active_watchlist empty, falling back to seed list",
            file=sys.stderr,
        )
        return [(t, "active") for t in STOCK_WATCHLIST]
    return [(str(e["ticker"]), str(e["status"])) for e in entries]


def _pair_is_enabled(ticker: str, setup: object) -> bool:
    """Per-pair alert gate (Phase 4).

    Returns True if the ``(ticker, signal_type)`` pair is enabled. BEST-EFFORT
    ISOLATION: if the gate lookup fails for any reason, default to enabled
    (alerting) so a gate error can never silence a live signal — and log it.
    """
    try:
        signal_type = _normalize_signal_type(str(setup))
        return db.get_signal_pair_status(ticker, signal_type) == "enabled"
    except Exception as exc:  # noqa: BLE001 - a gate failure must not break alerting
        print(
            f"  pair gate lookup failed for {ticker}/{setup}, "
            f"defaulting enabled: {exc}",
            file=sys.stderr,
        )
        return True


def _hold_window_days(signal: dict) -> int:
    """Parse the trade's intended hold horizon (days) from the signal's
    ``hold_days`` string, e.g. ``"3-5 days"`` -> 5. Falls back to the config
    default when unparseable. Used as the earnings-blackout window."""
    import re

    nums = re.findall(r"\d+", str(signal.get("hold_days", "")))
    if nums:
        return max(int(n) for n in nums)
    return config.EARNINGS_BLACKOUT_DEFAULT_DAYS


def _opt_float(signal: dict, key: str) -> float | None:
    """Optional float from the legacy signal dict — None when absent or
    malformed (Phase 22). Fail-soft: a bad value must never block signal
    capture; NULL keeps whatever default/fallback the consumer has today."""
    value = signal.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _hold_estimate_from(signal: dict) -> int | None:
    """The DAYS-denominated hold estimate to PERSIST on the signal row
    (Phase 21): ``"3-5 days"`` -> 5, the window's outer bound (matching
    ``_hold_window_days``).

    Distinct from ``_hold_window_days``' earnings-blackout fallback: for
    persistence, absent or non-day strings must stay ``None`` — NULL keeps the
    resolver's asset-class default, unchanged. The crypto signals' ``"2-8
    hours"`` estimate is hours-shaped and does not fit the integer-days
    column, so crypto deliberately persists nothing (14-day default as ever).
    """
    import re

    raw = str(signal.get("hold_days", ""))
    if "day" not in raw.lower():
        return None
    nums = re.findall(r"\d+", raw)
    if not nums:
        return None
    return max(int(n) for n in nums)


def _should_check_earnings(signal: dict) -> bool:
    """Earnings blackout applies only to stock TRADE signals (call/put).
    Crypto has no earnings; warning entries aren't trades."""
    return (
        str(signal.get("asset_type", "")) == "stock"
        and _normalize_direction(str(signal.get("direction", ""))) in {"call", "put"}
    )


def _score_signal_sentiment(signal: dict) -> object:
    """Fetch news + score advisory sentiment for a signal that WILL alert.

    Called only after the ticker-active, pair-enabled, and earnings-blackout
    gates pass — so news/LLM cost is bounded to signals that actually alert.
    Both the news client and the scorer are already fail-soft; this is wrapped
    once more so the alert proceeds (with no sentiment) even on a surprise.
    """
    ticker = str(signal.get("ticker", ""))
    try:
        news = news_client.fetch_headlines(ticker)
        return sentiment_mod.score(ticker, news)
    except Exception as exc:  # noqa: BLE001 - advisory work must never block an alert
        print(
            f"  sentiment pipeline error for {ticker}, alerting without it: {exc}",
            file=sys.stderr,
        )
        return None


def _close_series_for(ticker: str) -> object:
    """Fetch a ticker's Close series for the correlation universe, or None.

    Reuses the scanner's existing daily fetch. Fail-soft: any failure (no data,
    unexpected shape) returns None so one name never sinks the correlation
    frame. ``indicators.correlation_concentration`` is itself fail-soft too.
    """
    try:
        data = get_stock_data(ticker)
        if data is None:
            return None
        return data["Close"]
    except Exception as exc:  # noqa: BLE001 - one name's fetch must not raise
        print(f"  indicators: close fetch failed for {ticker}: {exc}", file=sys.stderr)
        return None


def _compute_signal_indicators(signal: dict, df: object) -> object:
    """Compute the Phase 6 advisory indicator families for a signal that WILL
    alert (called only after the ticker-active / pair-enabled / earnings gates
    pass — so the families are never computed for signals that won't fire).

    Per-ticker families come from ``df`` (the scan's enriched candle frame);
    the correlation pair is computed against the currently-ACTIVE watchlist
    names. Everything here is fail-soft: ``compute_context`` and
    ``correlation_concentration`` both swallow their own errors, and this
    wrapper guards anything systemic so the alert proceeds without context.
    """
    ticker = str(signal.get("ticker", ""))
    try:
        active_tickers = [
            t for t, status in _active_stock_watchlist_entries() if status == "active"
        ]
        corr, concentration = indicators.correlation_concentration(
            ticker, active_tickers, _close_series_for,
        )
        # Stocks reason on the last fully-closed candle (detect uses iloc[-2]).
        return indicators.compute_context(
            df, correlation=corr, concentration=concentration, at=-2,
        )
    except Exception as exc:  # noqa: BLE001 - advisory context must never block an alert
        print(
            f"  indicators pipeline error for {ticker}, alerting without it: {exc}",
            file=sys.stderr,
        )
        return None


def _compute_signal_risk(signal: dict, indicator_ctx: object) -> object:
    """Compute the advisory Phase 7 risk assessment for a signal that WILL alert.

    Consumes the Phase 6 ATR (stop distance) and concentration label (cluster
    exposure) from ``indicator_ctx``, plus the currently-open ACTIVE book for
    portfolio exposure (the candidate is not yet logged, so it is never
    double-counted). Advisory/notional only. Fail-soft: ``risk.assess`` already
    swallows its own errors; this wraps anything systemic (the open-trades read)
    so the alert proceeds without a recommendation.
    """
    ticker = str(signal.get("ticker", ""))
    try:
        entry = float(signal.get("price", 0.0))
        atr = getattr(indicator_ctx, "atr", None)
        concentration = getattr(indicator_ctx, "concentration", "unknown") or "unknown"
        open_positions = [
            risk.OpenPosition(
                risk_pct=t.risk_pct,
                concentration=t.ind_concentration or "unknown",
            )
            for t in db.get_open_trades()
            if t.track_mode == "active"
        ]
        return risk.assess(entry, atr, concentration, open_positions)
    except Exception as exc:  # noqa: BLE001 - advisory risk must never block an alert
        print(
            f"  risk pipeline error for {ticker}, alerting without it: {exc}",
            file=sys.stderr,
        )
        return None


def _emit_active_signal(signal: dict, df: object = None) -> None:
    """Handle a signal that passed ticker-active AND pair-enabled.

    The ONE hard gate runs first: if a known earnings report falls inside the
    trade's hold window, suppress the alert and do NOT open the trade. Otherwise
    score advisory sentiment, compute the Phase 6 indicator families, and derive
    the Phase 7 risk recommendation (only now — after all alert-gating checks
    pass, so news/LLM, indicator, and risk work happen solely for signals that
    will alert) and fire the enriched alert.
    """
    if _should_check_earnings(signal):
        try:
            days = _hold_window_days(signal)
            blackout, reason = earnings.is_in_blackout(signal["ticker"], days)
        except Exception as exc:  # noqa: BLE001 - gate failure must not block alerts
            print(
                f"  earnings blackout check failed for {signal['ticker']}, "
                f"proceeding: {exc}",
                file=sys.stderr,
            )
            blackout, reason = False, "blackout check error — fail-open"
        if blackout:
            print(
                f"  [earnings-blackout] {signal['ticker']} "
                f"{signal.get('setup', '')} — alert suppressed, trade NOT "
                f"opened ({reason})",
                file=sys.stderr,
            )
            return

    sentiment_result = _score_signal_sentiment(signal)
    indicator_ctx = (
        _compute_signal_indicators(signal, df) if df is not None else None
    )
    risk_assessment = _compute_signal_risk(signal, indicator_ctx)
    send_notification(
        signal, sentiment=sentiment_result, indicators=indicator_ctx,
        risk=risk_assessment,
    )


def _load_secrets() -> None:
    """Populate module-level credential constants from the secrets layer.

    Kept out of import side-effects so tests can import this module without
    a populated keychain. Raises KeyError with an actionable message if any
    secret is missing.
    """
    global DISCORD_WEBHOOK_URL, PUSHOVER_USER_KEY, PUSHOVER_APP_TOKEN, NEWSAPI_KEY
    DISCORD_WEBHOOK_URL = get_required("DISCORD_WEBHOOK_URL")
    PUSHOVER_USER_KEY   = get_required("PUSHOVER_USER_KEY")
    PUSHOVER_APP_TOKEN  = get_required("PUSHOVER_APP_TOKEN")
    NEWSAPI_KEY         = get_required("NEWSAPI_KEY")


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
    df["Vol_MA20"]  = df["Volume"].rolling(20).mean()
    # Slope was previously computed twice (identical call both times) — the
    # regression is O(rows x period) so the duplicate doubled the most
    # expensive indicator for no change in output.
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
    # Validate BEFORE touching .columns — the previous order dereferenced
    # df.columns first, so the None/shape guard below it could never actually
    # protect anything (a None return would have raised AttributeError first).
    df = yf.download(ticker, period="60d", interval="1d", progress=False)
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df.dropna(inplace=True)
    return df


def get_crypto_data(ticker):
    df = yf.download(ticker, period="60d", interval="1h", progress=False)
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df.dropna(inplace=True)
    return df


def get_yahoo_news(ticker, max_articles=5):
    """Pull recent news headlines from Yahoo Finance for a ticker"""
    try:
        print(f"  Fetching news for {ticker} from Yahoo Finance...")
        stock    = yf.Ticker(ticker)
        news     = stock.news or []
        articles = []

        for item in news[:max_articles]:
            content = item.get("content", {})
            title   = content.get("title", "No title")
            summary = content.get("summary", "")
            link    = content.get("canonicalUrl", {}).get("url", "")
            articles.append({
                "title":   title,
                "summary": summary,
                "link":    link,
            })

        return articles
    except Exception as e:
        # Network/parse failure on the Yahoo news feed. Returns [] (no news)
        # but log loudly so a persistently-failing feed is visible rather
        # than looking like a quiet news day.
        print(f"  Yahoo news error for {ticker}: {e}", file=sys.stderr)
        return []


def send_morning_report():
    """Send a morning news digest for the watchlist"""
    print("=== MORNING REPORT FUNCTION CALLED ===")
    if not is_market_open():
        # Still send report on market days even before open
        now = datetime.now(ZoneInfo("America/New_York"))
        if now.weekday() >= 5:
            return

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Generating morning news report...")

    report = "📰 **MORNING MARKET REPORT** — " + datetime.now().strftime("%Y-%m-%d") + "\n\n"

    for ticker in NEWS_WATCHLIST:
        articles = get_yahoo_news(ticker)
        report  += f"\n**{ticker}**\n"

        if not articles:
            report += "  No recent news\n"
            continue

        for article in articles[:3]:
            title = article["title"][:100]
            report += f"  • {title}\n"

    # Send to Discord
    try:
        # Discord has a 2000 character limit per message — split if needed.
        if len(report) > 1900:
            chunks = [report[i:i+1900] for i in range(0, len(report), 1900)]
            for chunk in chunks:
                requests.post(
                    DISCORD_WEBHOOK_URL, json={"content": chunk},
                    timeout=_HTTP_TIMEOUT_SECONDS,
                )
        else:
            requests.post(
                DISCORD_WEBHOOK_URL, json={"content": report},
                timeout=_HTTP_TIMEOUT_SECONDS,
            )
        print("  Morning report sent")
    except Exception as e:
        print(f"  Morning report error: {e}")

    # Send to Pushover — split into chunks since limit is 1024 chars.
    try:
        chunk_size = 1000
        chunks     = [report[i:i+chunk_size] for i in range(0, len(report), chunk_size)]

        for idx, chunk in enumerate(chunks, 1):
            requests.post("https://api.pushover.net/1/messages.json", data={
                "token":   PUSHOVER_APP_TOKEN,
                "user":    PUSHOVER_USER_KEY,
                "title":   f"📰 Morning Report ({idx}/{len(chunks)})",
                "message": chunk,
            }, timeout=_HTTP_TIMEOUT_SECONDS)
        print(f"  Pushover report sent ({len(chunks)} parts)")
    except Exception as e:
        print(f"  Pushover report error: {e}")


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
    except Exception as e:
        # yfinance calendar fetch/parse failure. "UNKNOWN" is also a valid
        # non-error result (no calendar / no earnings date), so log the
        # exception path explicitly — otherwise a broken feed is invisible.
        print(f"  earnings risk check error for {ticker}: {e}", file=sys.stderr)
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
        response = requests.get(url, timeout=_HTTP_TIMEOUT_SECONDS)
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
    except Exception as e:
        # NewsAPI request/parse failure (bad key, rate limit, network).
        # "UNKNOWN" is returned either way, but a silently-failing NewsAPI
        # key would otherwise never surface. Log it.
        print(f"  news risk check error for {ticker}: {e}", file=sys.stderr)
        return "UNKNOWN"


def estimate_hold_days(price, atr):
    atr_pct = (atr / price) * 100
    if atr_pct > 3:
        return "1-2 days"
    elif atr_pct > 1.5:
        return "2-3 days"
    return "3-5 days"


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL LOGGING — translates the legacy signal dict to a Signal dataclass and
# persists via SQLite. The CSV writer is gone; the historical CSV stays on
# disk as an artifact (already migrated by Phase 1.1c).
# ══════════════════════════════════════════════════════════════════════════════

_DIRECTION_MAP = {
    "CALL":  "call",
    "PUT":   "put",
    "LONG":  "long",
    "SHORT": "short",
}


def _normalize_direction(raw: str) -> str:
    """``"LONG 📈"`` → ``"long"``. Returns ``""`` for unrecognised input.

    Warning entries (``"⚠️ WARNING"``) produce a value not in the schema's
    allowed set; log_signal uses that to skip them.
    """
    if not raw:
        return ""
    parts = raw.strip().split()
    if not parts:
        return ""
    first = parts[0]
    return _DIRECTION_MAP.get(first.upper(), first.lower())


def _normalize_signal_type(raw: str) -> str:
    """``"Oversold Reversal"`` → ``"oversold_reversal"``."""
    if not raw:
        return ""
    return raw.strip().lower().replace(" ", "_").replace("-", "_")


def log_signal(
    signal: dict, track_mode: str = "active", sentiment: object = None,
    indicators: object = None, risk: object = None,
) -> int | None:
    """Translate a legacy-shape ``signal`` dict into a ``Signal`` row and open
    a corresponding ``Trade`` record (Phase 1.3).

    ``track_mode`` (Phase 3.1-LIVE) tags the opened trade ``'active'`` (default,
    so existing callers are unchanged) or ``'shadow'`` for shadow-universe
    scans whose alerts are suppressed.

    ``sentiment`` (Phase 5) is an optional advisory ``SentimentResult`` captured
    at fire time; its fields are persisted on the trade row next to the eventual
    outcome. ``None`` leaves the sentiment columns empty (shadow/crypto trades).

    ``indicators`` (Phase 6) is an optional advisory ``IndicatorContext`` — the
    five indicator families snapshotted at fire time, persisted the same way.
    ``None`` leaves the ind_* columns empty (shadow/crypto trades).

    ``risk`` (Phase 7) is an optional advisory ``RiskAssessment`` — recommended
    size + portfolio verdicts snapshotted at fire time, persisted the same way.
    ``None`` leaves the risk_* columns empty (shadow/crypto trades).

    Returns the inserted (or deduped) signal id, or ``None`` if the entry is a
    risk-warning alert (``direction = "⚠️ WARNING"``) that the legacy code
    uses for earnings/news risk. Those don't fit the trade-signal schema and
    are silently skipped — the Discord/Pushover alert still fires.

    Trade-opening is idempotent: if a trade already exists for this signal_id
    (e.g. dedupe collision or backfill ran first), no second trade is created.
    """
    direction   = _normalize_direction(str(signal.get("direction", "")))
    asset_class = str(signal.get("asset_type", ""))
    signal_type = _normalize_signal_type(str(signal.get("setup", "")))

    # Skip non-tradable warning entries — schema only models real trade signals.
    if direction not in {"call", "put", "long", "short"}:
        return None
    if asset_class not in {"stock", "crypto"}:
        return None

    stop_loss_raw   = signal.get("stop_loss")
    take_profit_raw = signal.get("take_profit")

    rec = Signal(
        timestamp=datetime.now(UTC),
        ticker=str(signal["ticker"]),
        asset_class=asset_class,
        signal_type=signal_type,
        direction=direction,
        entry_price=float(signal["price"]),
        stop_loss=float(stop_loss_raw) if stop_loss_raw is not None else None,
        take_profit=float(take_profit_raw) if take_profit_raw is not None else None,
        # Phase 21: persist the ATR-based estimate the scanner ALREADY computed
        # (it was dropped here since Phase 1 — every signal resolved on the
        # 30-day default instead of its own 1-5 day intent). Forward-only:
        # existing rows are never touched and NULL still means the default.
        hold_estimate_days=_hold_estimate_from(signal),
        # Phase 22 (signal field audit): persist the fire-time indicator values
        # the detectors ALREADY compute — they were silently dropped, leaving
        # every row NULL (the ATR gap alone made live SWING candidates
        # unsizeable). Forward-only: absent keys stay NULL, old rows untouched.
        atr=_opt_float(signal, "atr"),
        rsi=_opt_float(signal, "rsi"),
        ema21=_opt_float(signal, "ema21"),
        bb_upper=_opt_float(signal, "bb_upper"),
        bb_lower=_opt_float(signal, "bb_lower"),
        earnings_risk=str(signal.get("earnings_risk", "UNKNOWN")),
        news_risk=str(signal.get("news_risk", "UNKNOWN")),
    )
    signal_id = db.insert_signal(rec)

    # Open a Trade if one doesn't already exist for this signal.
    if db.get_trade_by_signal_id(signal_id) is None:
        # Use the persisted signal's timestamp so backfill and live writes
        # converge on the same opened_at after a dedupe collision.
        persisted = db.get_signal_by_id(signal_id)
        opened_at = persisted.timestamp if persisted is not None else rec.timestamp

        # Phase 2.1: tag the trade with the current macro regime. SPY-based
        # regime applies to all asset classes (crypto included) — a bear in
        # equities is relevant context for any trade fired during it. A
        # RegimeFetchError must NOT block signal capture; we tag 'unknown'.
        try:
            market_regime = regime.get_current_regime().regime
        except regime.RegimeFetchError as exc:
            import sys
            print(
                f"  regime fetch failed for {rec.ticker}, tagging trade as 'unknown': {exc}",
                file=sys.stderr,
            )
            market_regime = "unknown"

        # Phase 2.2: tag the trade with the current VIX. Independent of
        # regime — VIX is the volatility axis, regime is the direction axis.
        # Same fail-soft contract: a VixFetchError tags 'unknown' with a
        # NULL level, but the signal still fires.
        vix_level: float | None
        vix_band: str
        try:
            vix_snap = vix.get_current_vix()
            vix_level = vix_snap.vix_level
            vix_band = vix_snap.vix_band
        except vix.VixFetchError as exc:
            import sys
            print(
                f"  vix fetch failed for {rec.ticker}, tagging trade as 'unknown': {exc}",
                file=sys.stderr,
            )
            vix_level = None
            vix_band = "unknown"

        # Phase 2.3: composite (regime x VIX) context score. Pure derivation
        # from the two axes we just tagged — zero new I/O.
        context_score = context.score(market_regime, vix_band)

        db.insert_trade(
            Trade(
                signal_id=signal_id,
                opened_at=opened_at,
                outcome="open",
                market_regime=market_regime,
                vix_level=vix_level,
                vix_band=vix_band,
                context_score=context_score,
                track_mode=track_mode,
                sentiment_score=getattr(sentiment, "score", None),
                sentiment_label=getattr(sentiment, "label", None),
                sentiment_ok=getattr(sentiment, "ok", None),
                sentiment_rationale=getattr(sentiment, "rationale", None),
                heavy_news=bool(getattr(sentiment, "heavy_news", False)),
                headline_count=getattr(sentiment, "headline_count", None),
                ind_ok=getattr(indicators, "ok", None),
                ind_atr=getattr(indicators, "atr", None),
                ind_realized_vol=getattr(indicators, "realized_vol", None),
                ind_vol_regime=getattr(indicators, "vol_regime", None),
                ind_rsi=getattr(indicators, "rsi", None),
                ind_adx=getattr(indicators, "adx", None),
                ind_obv=getattr(indicators, "obv", None),
                ind_correlation=getattr(indicators, "correlation", None),
                ind_concentration=getattr(indicators, "concentration", None),
                risk_ok=getattr(risk, "ok", None),
                risk_reason=getattr(risk, "reason", None),
                risk_recommended_size=getattr(risk, "recommended_size", None),
                risk_stop_distance=getattr(risk, "stop_distance", None),
                risk_dollar_risk=getattr(risk, "dollar_risk", None),
                risk_pct=getattr(risk, "risk_pct", None),
                risk_position_pct=getattr(risk, "position_pct", None),
                risk_capped=bool(getattr(risk, "capped", False)),
                risk_total_pct=getattr(risk, "total_risk_pct", None),
                risk_portfolio_verdict=getattr(risk, "portfolio_verdict", None),
                risk_position_verdict=getattr(risk, "position_verdict", None),
                risk_cluster_pct=getattr(risk, "cluster_risk_pct", None),
                risk_cluster_verdict=getattr(risk, "cluster_verdict", None),
            )
        )

    return signal_id


def _run_outcome_resolver() -> None:
    """Thin wrapper called by the scheduler — never lets an exception escape.

    Yfinance hiccups, rate limits, or schema surprises should NOT kill the
    scheduling loop. They get logged to stderr and the next run tries again.
    """
    try:
        result = outcomes.resolve_all_open_trades()
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] outcome resolver: "
            f"wins={result['wins']} losses={result['losses']} "
            f"expired={result['expired']} still_open={result['still_open']}"
        )
    except Exception as exc:
        import sys
        print(f"  outcome resolver error: {exc}", file=sys.stderr)


def _run_daily_regime_snapshot() -> None:
    """Phase 2.1: capture today's macro regime and persist to regime_snapshots.

    Scheduled at 09:35 EST — four minutes after the stock scan so the most
    recent SPY close is already digested. Force-refresh bypasses the 24h
    cache so this is genuinely today's snapshot. Swallow-all on failure so a
    transient yfinance hiccup doesn't kill the scheduling loop.
    """
    try:
        snap = regime.get_current_regime(force_refresh=True)
        db.upsert_regime_snapshot(
            snapshot_date=snap.date,
            regime=snap.regime,
            spy_close=snap.spy_close,
            ema50=snap.ema50,
            ema200=snap.ema200,
            ema50_slope=snap.ema50_slope,
            captured_at=datetime.now(UTC),
        )
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] regime snapshot: "
            f"{snap.date} {snap.regime} (SPY ${snap.spy_close:.2f})"
        )
    except Exception as exc:
        import sys
        print(f"  regime snapshot error: {exc}", file=sys.stderr)


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION ENGINE (Phase 2.2b)
# ══════════════════════════════════════════════════════════════════════════════

# Default settings for the prediction engine. These are written to the
# settings table on first run only — subsequent runs respect whatever the
# user configured via the CLI.
_PRED_DEFAULTS = {
    "predictions.enabled":           "false",
    "predictions.window_start":      "08:00",
    "predictions.window_end":        "22:00",
    "predictions.tickers":           "BTC-USD,ETH-USD",
    "predictions.notify_resolution": "false",
}


def _seed_prediction_defaults() -> None:
    """Write any missing prediction settings to their defaults. Idempotent.

    After seeding, an optional ``PREDICTIONS_ENABLED`` environment variable
    overrides the master switch. This exists because the Railway DB and the
    local DB are SEPARATE — there is no shell access to flip the setting on
    Railway, so the env var is the only way to control the deployed
    instance. Local development still uses the CLI toggle.

    * ``PREDICTIONS_ENABLED=true``  -> force enabled (regardless of DB state)
    * ``PREDICTIONS_ENABLED=false`` -> force disabled
    * unset / any other value       -> leave whatever the DB already holds
    """
    for key, value in _PRED_DEFAULTS.items():
        if settings.get(key) is None:
            settings.set(key, value)

    env_override = os.environ.get("PREDICTIONS_ENABLED")
    if env_override is not None:
        normalized = env_override.strip().lower()
        if normalized == "true":
            settings.set_bool("predictions.enabled", True)
            print("[predictions] enabled via PREDICTIONS_ENABLED env var")
        elif normalized == "false":
            settings.set_bool("predictions.enabled", False)
            print("[predictions] disabled via PREDICTIONS_ENABLED env var")
        # Any other value: leave the DB value untouched.


def _parse_hhmm(s: str) -> tuple[int, int] | None:
    parts = s.strip().split(":")
    if len(parts) != 2:
        return None
    try:
        hh = int(parts[0])
        mm = int(parts[1])
    except ValueError:
        return None
    if not (0 <= hh <= 23 and 0 <= mm <= 59):
        return None
    return (hh, mm)


def _within_prediction_window(now_et: datetime) -> bool:
    """Inclusive on both ends. start <= now < end (end-exclusive feels
    natural; a 22:00 end means stop sending at 22:00 sharp)."""
    start = _parse_hhmm(settings.get("predictions.window_start", "08:00") or "08:00")
    end = _parse_hhmm(settings.get("predictions.window_end", "22:00") or "22:00")
    if start is None or end is None:
        return False
    now_minutes = now_et.hour * 60 + now_et.minute
    start_minutes = start[0] * 60 + start[1]
    end_minutes = end[0] * 60 + end[1]
    return start_minutes <= now_minutes < end_minutes


def _check_pause_until(now: datetime) -> bool:
    """Return True if predictions are currently paused. Auto-clears an
    expired pause_until so the next sweep doesn't keep checking it."""
    raw = settings.get("predictions.pause_until")
    if not raw:
        return False
    try:
        until = datetime.fromisoformat(raw)
    except ValueError:
        settings.delete("predictions.pause_until")
        return False
    if until.tzinfo is None:
        until = until.replace(tzinfo=UTC)
    if now < until:
        return True
    settings.delete("predictions.pause_until")
    return False


def _warn_if_bad_status(resp: object, channel: str) -> bool:
    """Return delivery success and log an HTTP-status failure.

    ``requests.post`` does NOT raise on 4xx/5xx — only on connection errors.
    A Pushover 429 (rate limit) or 401 (bad token), or a Discord 401/404 on a
    stale webhook, would otherwise look like a successful send and silently
    drop the alert. This surfaces that case to stderr. This is the likely
    root cause of intermittent missing phone alerts.
    """
    status = getattr(resp, "status_code", None)
    if isinstance(status, int) and 200 <= status < 300:
        return True
    status_text = str(status) if status is not None else "unknown"
    print(
        f"  {channel} notification HTTP {status_text} "
        f"(alert may not have been delivered)",
        file=sys.stderr,
    )
    return False


def _send_prediction_notification(pred: object) -> bool:
    """Send via both required channels and return all-or-nothing success.

    Discord and Pushover are one delivery contract. A partial success returns
    False, leaving the prediction retryable. Because delivery state is not
    tracked per channel, a later retry intentionally resends both channels.
    """
    # ``pred`` typed as object to keep this function importable without the
    # Prediction class at module-load time; runtime asserts the shape.
    p = pred  # alias
    ticker = getattr(p, "ticker")  # noqa: B009
    direction = getattr(p, "direction")  # noqa: B009
    confidence = float(getattr(p, "confidence"))  # noqa: B009
    entry = float(getattr(p, "entry_price"))  # noqa: B009
    target_end = getattr(p, "target_window_end")  # noqa: B009
    regime_tag = getattr(p, "market_regime") or "unknown"  # noqa: B009
    vix_tag = getattr(p, "vix_band") or "unknown"  # noqa: B009

    et = ZoneInfo("America/New_York")
    created_at = getattr(p, "created_at")  # noqa: B009
    created_et = created_at.astimezone(et) if created_at.tzinfo else created_at
    target_et = target_end.astimezone(et) if target_end.tzinfo else target_end

    msg = (
        f"[PREDICTION] {ticker} -> {direction} ({confidence:.0f}% confidence)\n"
        f"Entry: ${entry:,.2f} @ {created_et.strftime('%H:%M ET')}\n"
        f"Window closes: {target_et.strftime('%H:%M ET')}\n"
        f"Context: {regime_tag} / {vix_tag} VIX"
    )

    if DRY_RUN:
        print("\n--- DRY-RUN prediction (no Discord/Pushover) ---")
        print(msg)
        return False

    discord_ok = False
    try:
        resp = requests.post(
            DISCORD_WEBHOOK_URL, json={"content": msg},
            timeout=_HTTP_TIMEOUT_SECONDS,
        )
        discord_ok = _warn_if_bad_status(resp, "prediction Discord")
    except Exception as exc:
        print(f"  prediction Discord error: {exc}", file=sys.stderr)
    pushover_ok = False
    try:
        resp = requests.post("https://api.pushover.net/1/messages.json", data={
            "token": PUSHOVER_APP_TOKEN,
            "user":  PUSHOVER_USER_KEY,
            "title": f"Prediction: {ticker} {direction}",
            "message": msg,
        }, timeout=_HTTP_TIMEOUT_SECONDS)
        pushover_ok = _warn_if_bad_status(resp, "prediction Pushover")
    except Exception as exc:
        print(f"  prediction Pushover error: {exc}", file=sys.stderr)
    return discord_ok and pushover_ok


def _record_prediction_delivery(prediction_id: int, pred: object) -> bool:
    """Notify one persisted prediction and durably record only true success."""
    if not _send_prediction_notification(pred):
        return False
    db.update_prediction(prediction_id, notified=True)
    return True


def _send_resolution_notification(
    pred: object, result: object,
) -> None:
    """Sent only when ``predictions.notify_resolution`` is on."""
    ticker = getattr(pred, "ticker")  # noqa: B009
    direction = getattr(pred, "direction")  # noqa: B009
    confidence = float(getattr(pred, "confidence"))  # noqa: B009
    entry = float(getattr(pred, "entry_price"))  # noqa: B009
    exit_price = getattr(result, "exit_price")  # noqa: B009
    outcome = getattr(result, "outcome")  # noqa: B009

    outcome_label = {
        "correct":   "[WIN]",
        "incorrect": "[LOSS]",
        "push":      "[PUSH]",
    }.get(outcome or "", "[?]")

    if exit_price is not None and entry != 0:
        pct = (exit_price - entry) / entry * 100.0
        sign = "+" if pct >= 0 else ""
        change = f"({sign}{pct:.2f}%)"
        exit_str = f"${exit_price:,.2f}"
    else:
        change = ""
        exit_str = "n/a"

    msg = (
        f"[PREDICTION RESOLVED] {ticker} -> {direction} {outcome_label}\n"
        f"Entry: ${entry:,.2f}  Exit: {exit_str}  {change}\n"
        f"{outcome or 'unresolved'} ({confidence:.0f}% confidence)"
    )

    if DRY_RUN:
        print("\n--- DRY-RUN resolution (no Discord/Pushover) ---")
        print(msg)
        return

    try:
        resp = requests.post(
            DISCORD_WEBHOOK_URL, json={"content": msg},
            timeout=_HTTP_TIMEOUT_SECONDS,
        )
        _warn_if_bad_status(resp, "resolution Discord")
    except Exception as exc:
        print(f"  resolution Discord error: {exc}", file=sys.stderr)
    try:
        resp = requests.post("https://api.pushover.net/1/messages.json", data={
            "token": PUSHOVER_APP_TOKEN,
            "user":  PUSHOVER_USER_KEY,
            "title": f"Resolved: {ticker} {outcome_label}",
            "message": msg,
        }, timeout=_HTTP_TIMEOUT_SECONDS)
        _warn_if_bad_status(resp, "resolution Pushover")
    except Exception as exc:
        print(f"  resolution Pushover error: {exc}", file=sys.stderr)


def _gated_skip_reason(now_utc: datetime) -> str | None:
    """Why predictions are skipped this tick, or None if they should run."""
    if not settings.get_bool("predictions.enabled", default=False):
        return "disabled"
    if _check_pause_until(now_utc):
        return "paused"
    et = ZoneInfo("America/New_York")
    now_et = now_utc.astimezone(et)
    if not _within_prediction_window(now_et):
        return "outside_window"
    return None


def _run_prediction_sweep() -> None:
    """15-min cron tick. Gated by settings + active window + pause."""
    now_utc = datetime.now(UTC)
    skip = _gated_skip_reason(now_utc)
    if skip is not None:
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] prediction sweep "
            f"skipped: {skip}"
        )
        return

    raw_tickers = settings.get("predictions.tickers", "BTC-USD,ETH-USD") or ""
    tickers = [t.strip() for t in raw_tickers.split(",") if t.strip()]
    for ticker in tickers:
        try:
            pred = predictions.predict_direction(ticker, now=now_utc)
        except Exception as exc:  # noqa: BLE001 - never let one ticker kill the sweep
            print(f"  prediction error for {ticker}: {exc}", file=sys.stderr)
            continue
        if pred is None:
            print(f"  {ticker}: signals mixed or data unavailable")
            continue
        # Phase 2.3: tag composite context_score from the regime + VIX
        # already populated by predict_direction(). dataclasses.replace
        # keeps Prediction immutable; the score is derived not fetched.
        import dataclasses
        pred = dataclasses.replace(
            pred,
            context_score=context.score(
                pred.market_regime or "unknown", pred.vix_band or "unknown",
            ),
        )
        try:
            pid = db.insert_prediction(pred)
            delivered = _record_prediction_delivery(pid, pred)
            if not delivered:
                print(
                    f"  prediction notification incomplete for {ticker}; "
                    "persisted as unnotified for retry",
                    file=sys.stderr,
                )
            print(
                f"  {ticker}: {pred.direction} {pred.confidence:.0f}% "
                f"(window ends {pred.target_window_end.strftime('%H:%M UTC')})"
            )
        except Exception as exc:  # noqa: BLE001 - per-ticker safety
            print(f"  prediction persist error for {ticker}: {exc}", file=sys.stderr)


def _run_prediction_resolution() -> None:
    """5-min cron tick. Resolves any predictions whose target candle has closed."""
    try:
        results = predictions.resolve_due_predictions()
    except Exception as exc:  # noqa: BLE001 - resolver must never kill the loop
        print(f"  prediction resolver error: {exc}", file=sys.stderr)
        return

    notify_resolution = settings.get_bool(
        "predictions.notify_resolution", default=False,
    )
    resolved = sum(1 for r in results if r.status == "resolved")
    if resolved:
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] resolved {resolved} "
            f"predictions"
        )

    if notify_resolution:
        for result in results:
            if result.status != "resolved":
                continue
            pred = db.get_prediction(result.prediction_id)
            if pred is None:
                continue
            try:
                _send_resolution_notification(pred, result)
            except Exception as exc:  # noqa: BLE001 - notification is best-effort
                print(
                    f"  resolution notify error for {result.prediction_id}: {exc}",
                    file=sys.stderr,
                )


def _run_daily_vix_snapshot() -> None:
    """Phase 2.2: capture today's VIX and persist to vix_snapshots.

    Scheduled at 09:36 EST — one minute after the regime snapshot. VIX
    closes at 16:15 ET, but the latest daily candle is available from
    yfinance shortly after market open of the next day; we want today's
    morning value reflecting last close. Swallow-all on failure so a
    transient yfinance hiccup doesn't kill the scheduling loop.
    """
    try:
        snap = vix.get_current_vix(force_refresh=True)
        db.upsert_vix_snapshot(
            snapshot_date=snap.date,
            vix_level=snap.vix_level,
            vix_band=snap.vix_band,
            captured_at=datetime.now(UTC),
        )
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] vix snapshot: "
            f"{snap.date} {snap.vix_level:.2f} ({snap.vix_band})"
        )
    except Exception as exc:
        import sys
        print(f"  vix snapshot error: {exc}", file=sys.stderr)


def _run_readiness_evaluation() -> None:
    """Phase 10: evaluate every capability's readiness and fire any first-crossing
    notifications. Runs in the scan loop so deterministic capabilities
    auto-activate (and ML capabilities summon a build) automatically as resolved
    data accumulates. Swallow-all so a transient failure never kills the loop."""
    try:
        results = readiness.evaluate_readiness()
        newly = [r.capability for r in results if r.newly_announced]
        if newly:
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] readiness: announced "
                f"{', '.join(newly)}"
            )
    except Exception as exc:
        import sys
        print(f"  readiness evaluation error: {exc}", file=sys.stderr)


def _run_order_lifecycle_cycle() -> None:
    """Refresh and materialize already-submitted paper orders once.

    The reconciliation entry point is lookup/update-only: prepared intents may
    be recovered by client ID, but this scanner hook never submits an opening
    order. Any broker/DB/materialization failure remains durable and retryable
    and cannot block stock or crypto scanning.
    """
    try:
        from trading_bot.broker import AlpacaBroker

        result = order_lifecycle.reconcile_pending_orders(AlpacaBroker())
        changed = sum(
            item.action == order_lifecycle.REFRESH_UPDATED
            for item in result.refreshes
        )
        materialized = sum(
            item.action
            in {
                order_lifecycle.MATERIALIZE_CREATED,
                order_lifecycle.MATERIALIZE_ADOPTED,
                order_lifecycle.MATERIALIZE_UPDATED,
            }
            for item in result.materializations
        )
        exits_closed = sum(
            item.action == order_lifecycle.EXIT_FILL_CLOSED
            for item in result.exit_materializations
        )
        if changed or materialized or exits_closed:
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] order lifecycle: "
                f"{changed} refreshed, {materialized} entries materialized, "
                f"{exits_closed} exits closed"
            )
    except Exception as exc:  # noqa: BLE001 - scanner recovery must fail soft
        print(f"  order lifecycle cycle error: {exc}", file=sys.stderr)


def _record_risk_cycle_reconciliation(broker: object) -> None:
    """Record one conclusive broker-position comparison, or leave state alone.

    Only the three real position divergence kinds count. Broker-unavailable,
    exception, or malformed-report paths are inconclusive and deliberately
    preserve the prior streak.
    """
    from trading_bot.broker import reconcile

    try:
        report = reconcile(broker)
    except Exception as exc:  # noqa: BLE001 - unavailable is not a divergence
        print(
            f"  risk reconcile error: {exc}; divergence streak unchanged",
            file=sys.stderr,
        )
        return

    report_ok = getattr(report, "ok", None)
    divergences = getattr(report, "divergences", None)
    if not isinstance(report_ok, bool) or not isinstance(divergences, list):
        print(
            "  risk reconcile malformed report; divergence streak unchanged",
            file=sys.stderr,
        )
        return

    if not report_ok:
        note = getattr(report, "note", "")
        detail = note or "broker position read unavailable"
        print(
            f"  risk reconcile unavailable: {detail}; divergence streak unchanged",
            file=sys.stderr,
        )
        return

    real_kinds = {"internal_only", "broker_only", "qty_mismatch"}
    kinds = [getattr(item, "kind", None) for item in divergences]
    if any(kind not in real_kinds for kind in kinds):
        print(
            "  risk reconcile malformed report; divergence streak unchanged",
            file=sys.stderr,
        )
        return

    risk_of_ruin.record_reconcile_result(not divergences)


def _run_risk_cycle() -> None:
    """Phase 15: one risk-of-ruin pass per scan cycle.

    Records an equity snapshot (drawdown source), evaluates the Tier-1 circuit
    breaker, and — if a catastrophic detector has tripped or a Tier-2 shutdown
    is mid-flight (holding state, e.g. market was closed) — runs the emergency
    shutdown orchestrator. Swallow-all so a transient failure never kills the
    loop; the orchestrator itself is fail-soft."""
    try:
        from trading_bot.broker import AlpacaBroker

        broker = AlpacaBroker()
        account = broker.get_account()
        risk_of_ruin.record_equity_snapshot(account.equity)
        risk_of_ruin.evaluate_tier1()

        # Reconciliation contributes exactly one conclusive result (clean or
        # divergent) before Tier-2 evaluates the persisted streak. Provider
        # outages and malformed reports preserve the last-known counter.
        _record_risk_cycle_reconciliation(broker)

        reason = risk_of_ruin.check_catastrophic()
        if reason is not None:
            risk_of_ruin.emergency_shutdown(broker, trigger=reason)
        elif risk_of_ruin.get_state() == risk_of_ruin.STATE_HOLDING:
            # A prior shutdown pass left closes pending (market closed) — the
            # bot stays alive and retries automatically until confirmed.
            risk_of_ruin.emergency_shutdown(
                broker, trigger="retrying pending emergency closes",
            )
    except Exception as exc:
        import sys
        print(f"  risk cycle error: {exc}", file=sys.stderr)


def _run_evaluator_cycle() -> None:
    """Phase 15.5: DAILY pairs + watchlist evaluation (real mode), firing ONE
    combined Pushover whenever a run produces any mute/enable/demote/recover
    transition (silent on a no-op day). This is the missing call site behind the
    incident where negative-expectancy pairs ran uncaught for weeks. Swallow-all
    so a transient failure never kills the scheduling loop; the notification is
    itself fail-soft."""
    try:
        result = evaluator_scheduling.run_and_notify()
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] evaluators: "
            f"{result.pair_transitions} pair + {result.watchlist_transitions} "
            f"watchlist transition(s)"
            + (" - notified" if result.notified else "")
        )
    except Exception as exc:
        import sys
        print(f"  evaluator cycle error: {exc}", file=sys.stderr)


def _run_daily_perf_update() -> None:
    """Materialize yesterday's daily_performance row. Swallow-all on failure
    so a transient DB error doesn't kill the scheduling loop."""
    try:
        perf = performance.update_daily_performance()
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] daily perf: {perf.date} "
            f"signals={perf.signals_fired} opened={perf.trades_opened} "
            f"closed={perf.trades_closed} wins={perf.wins} losses={perf.losses}"
        )
    except Exception as exc:
        import sys
        print(f"  daily perf update error: {exc}", file=sys.stderr)


# ══════════════════════════════════════════════════════════════════════════════
# STOCK SIGNALS — EMA21 Pullback
# Proven 56-76% win rate across Tier 1 watchlist
# ══════════════════════════════════════════════════════════════════════════════

def detect_stock_signals(ticker, df):
    signals = []

    latest = df.iloc[-2]

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
            "atr":        atr,
            "rsi":        rsi,
            "ema21":      ema21,
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
            "atr":        atr,
            "rsi":        rsi,
            "ema21":      ema21,
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
            "atr":        atr,
            "rsi":        rsi,
            "ema21":      ema21,
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
            "atr":        atr,
            "rsi":        rsi,
            "ema21":      ema21,
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
            "atr":        atr,
            "rsi":        rsi,
            "ema21":      ema21,
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
            "atr":        atr,
            "rsi":        rsi,
            "ema21":      ema21,
        })

    # ── Trend Continuation ─────────────────────────────────────────────────────
    # Price above EMA21 (short term trend intact)
    # Price above EMA50 (long term trend intact)
    # Slope positive (trend confirmed)
    # RSI 50-65 (strong but not overextended)
    # Volume average or above (real participation)
    if (
        ticker in TREND_CONT_TICKERS
        and price > ema21
        and price > ema50
        and slope > 0
        and 50 <= rsi <= 65
        and volume > vol_ma
        and roc_a > 0
    ):
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
            "atr":        atr,
            "rsi":        rsi,
            "ema21":      ema21,
        })

    # Preserve the detector's already-computed grades on every trade signal.
    # HIGH returned above as a warning and never reaches log_signal; MEDIUM,
    # LOW and UNKNOWN are advisory-only and now survive to the signal row.
    for signal in signals:
        signal["earnings_risk"] = earnings_risk
        signal["news_risk"] = news_risk

    return signals


# ══════════════════════════════════════════════════════════════════════════════
# CRYPTO SIGNALS — Oversold Reversal + Momentum Breakout + Overbought Reversal
# Proven 56-61% win rate on BTC, BNB, ETH on hourly candles
# ══════════════════════════════════════════════════════════════════════════════

def _closed_hourly_crypto_bars(
    df: pd.DataFrame, now: datetime,
) -> pd.DataFrame:
    """Return only hourly crypto bars whose full interval has elapsed.

    yfinance indexes intraday rows by bar-open timestamp, so a row is closed
    only when ``open + 1 hour <= now``. Timezone-aware indexes are normalized
    to UTC before comparison. A naive crypto index is conservatively treated
    as UTC too; it is never reinterpreted in the host machine's local timezone.
    """
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("crypto scan now must be timezone-aware")

    index = pd.DatetimeIndex(pd.to_datetime(df.index))
    opens_utc = (
        index.tz_localize("UTC")
        if index.tz is None
        else index.tz_convert("UTC")
    )
    comparison_now = pd.Timestamp(now).tz_convert("UTC")

    bar_ends = opens_utc + pd.Timedelta(hours=1)
    return df.loc[bar_ends <= comparison_now]


def detect_crypto_signals(
    ticker, df, *, now: datetime | None = None,
):
    comparison_now = now if now is not None else datetime.now(UTC)
    try:
        df = _closed_hourly_crypto_bars(df, comparison_now)
    except (TypeError, ValueError) as exc:
        print(
            f"  crypto candle timestamp error for {ticker}: {exc}",
            file=sys.stderr,
        )
        return []
    if df.shape[0] < 2:
        print(
            f"  crypto insufficient closed candles for {ticker}: "
            f"{df.shape[0]} available, 2 required",
            file=sys.stderr,
        )
        return []

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
            "atr":        atr,
            "rsi":        rsi,
            "bb_upper":   bb_upper,
            "bb_lower":   bb_lower,
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
            "atr":        atr,
            "rsi":        rsi,
            "bb_upper":   bb_upper,
            "bb_lower":   bb_lower,
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
            "atr":        atr,
            "rsi":        rsi,
            "bb_upper":   bb_upper,
            "bb_lower":   bb_lower,
        })

    return signals


# ══════════════════════════════════════════════════════════════════════════════
# NOTIFICATIONS
# ══════════════════════════════════════════════════════════════════════════════

def _risk_alert_lines(risk_obj: object) -> str:
    """Render the advisory risk block for an alert (plain ASCII), or '' when absent.

    Shows the recommended size + risk %, a capped marker, and any
    'would-exceed-*' portfolio advisories. Purely informational — nothing here
    changes whether or what the alert fires.
    """
    if risk_obj is None:
        return ""
    size = getattr(risk_obj, "recommended_size", None)
    if not isinstance(size, (int, float)):
        return f"Risk:        size unavailable ({getattr(risk_obj, 'reason', '')})\n"
    risk_pct = getattr(risk_obj, "risk_pct", None)
    pct_txt = f" ({risk_pct:.2f}% risk)" if isinstance(risk_pct, (int, float)) else ""
    capped = "  [capped to max position]" if getattr(risk_obj, "capped", False) else ""
    lines = f"Risk:        ~{size:.2f} units{pct_txt}{capped}\n"
    advisories = [
        v for v in (
            getattr(risk_obj, "portfolio_verdict", ""),
            getattr(risk_obj, "position_verdict", ""),
            getattr(risk_obj, "cluster_verdict", ""),
        )
        if isinstance(v, str) and v.startswith("would-exceed")
    ]
    if advisories:
        lines += f"Risk advisory: {', '.join(advisories)}\n"
    return lines


def _sentiment_alert_lines(sentiment: object) -> str:
    """Render the advisory sentiment block for an alert, or '' when absent."""
    if sentiment is None:
        return ""
    score = getattr(sentiment, "score", None)
    label = getattr(sentiment, "label", None)
    rationale = getattr(sentiment, "rationale", "")
    score_txt = f"{score:+.2f}" if isinstance(score, (int, float)) else "n/a"
    lines = f"Sentiment:   {label} ({score_txt}) — {rationale}\n"
    if getattr(sentiment, "heavy_news", False):
        lines += "Heavy news:  elevated headline volume — trade with care\n"
    return lines


def send_notification(
    signal, *, track_mode="active", alert=True, sentiment=None, indicators=None,
    risk=None,
):
    """Persist the signal/trade and (unless suppressed) fire the alert.

    Phase 3.1-LIVE: ``track_mode`` tags the opened trade ('active' or
    'shadow') and ``alert=False`` suppresses the Discord/Pushover push for
    shadow-universe signals. The DB write ALWAYS happens — shadow trades must
    accumulate so the resolver can settle them — only the notification is
    gated. Each suppressed alert is logged so the shadow layer is observable.

    Phase 5: ``sentiment`` is an optional advisory ``SentimentResult`` — it is
    persisted on the trade row and (for alerting signals) shown in the alert.
    It NEVER suppresses anything.

    Phase 6: ``indicators`` is an optional advisory ``IndicatorContext`` — the
    five indicator families snapshotted at fire time, persisted on the trade
    row. Like sentiment, it NEVER suppresses anything.

    Phase 7: ``risk`` is an optional advisory ``RiskAssessment`` — the
    recommended size + portfolio verdicts, persisted on the trade row and shown
    in the alert. Advisory/notional only; it NEVER suppresses anything.
    """
    try:
        log_signal(
            signal, track_mode=track_mode, sentiment=sentiment,
            indicators=indicators, risk=risk,
        )
    except Exception as e:
        # DB write failure for the signal/trade row. Route to stderr so a
        # silently-failing log doesn't hide that a fired alert was never
        # persisted (it would vanish from all the reporting downstream).
        print(f"  Log error: {e}", file=sys.stderr)

    if not alert:
        print(
            f"  [shadow] alert suppressed for {signal['ticker']} "
            f"({signal['setup']} {signal['direction']}) — trade opened "
            f"track_mode={track_mode}"
        )
        return

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
        f"{_sentiment_alert_lines(sentiment)}"
        f"{_risk_alert_lines(risk)}"
        f"Time:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    )

    if DRY_RUN:
        print("\n--- DRY-RUN signal (no Discord/Pushover) ---")
        print(msg)
        return

    embed_fields = [
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
    ]
    if sentiment is not None:
        _s = getattr(sentiment, "score", None)
        _s_txt = f"{_s:+.2f}" if isinstance(_s, (int, float)) else "n/a"
        embed_fields.append({
            "name": "Sentiment",
            "value": f"{getattr(sentiment, 'label', 'n/a')} ({_s_txt})"
                     + ("  ⚠️ heavy news" if getattr(sentiment, "heavy_news", False) else ""),
            "inline": True,
        })
    if risk is not None:
        _rsize = getattr(risk, "recommended_size", None)
        if isinstance(_rsize, (int, float)):
            _rpct = getattr(risk, "risk_pct", None)
            _rpct_txt = f" ({_rpct:.2f}%)" if isinstance(_rpct, (int, float)) else ""
            _cap = "  [capped]" if getattr(risk, "capped", False) else ""
            _adv = [
                v for v in (
                    getattr(risk, "portfolio_verdict", ""),
                    getattr(risk, "position_verdict", ""),
                    getattr(risk, "cluster_verdict", ""),
                )
                if isinstance(v, str) and v.startswith("would-exceed")
            ]
            _risk_val = f"~{_rsize:.2f} units{_rpct_txt}{_cap}"
            if _adv:
                _risk_val += "  ⚠️ " + ", ".join(_adv)
        else:
            _risk_val = f"size unavailable ({getattr(risk, 'reason', '')})"
        embed_fields.append({"name": "Risk", "value": _risk_val, "inline": True})
    embed_fields.append(
        {"name": "Time", "value": datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "inline": True}
    )
    try:
        resp = requests.post(DISCORD_WEBHOOK_URL, json={
            "content": "@everyone 🚨 TRADE ALERT",
            "embeds": [{
                "title": f"🚨 TRADE ALERT — {signal['ticker']}",
                "color": 3066993,
                "fields": embed_fields,
            }],
        }, timeout=_HTTP_TIMEOUT_SECONDS)
        _warn_if_bad_status(resp, "Discord")
        print("  Discord notification sent")
    except Exception as e:
        print(f"  Discord error: {e}", file=sys.stderr)

    try:
        resp = requests.post("https://api.pushover.net/1/messages.json", data={
            "token":   PUSHOVER_APP_TOKEN,
            "user":    PUSHOVER_USER_KEY,
            "title":   f"Alert: {signal['ticker']} {signal['direction']}",
            "message": msg,
        }, timeout=_HTTP_TIMEOUT_SECONDS)
        _warn_if_bad_status(resp, "Pushover")
        print("  Pushover notification sent")
    except Exception as e:
        print(f"  Pushover error: {e}", file=sys.stderr)


# ══════════════════════════════════════════════════════════════════════════════
# SCAN FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def scan_stocks():
    """Runs once per day at market open — daily candles, swing trades.

    Gating hierarchy (Phase 4): an alert fires ONLY IF the ticker is 'active'
    AND its (ticker, signal_type) pair is 'enabled'. Ticker status dominates —
    a benched ticker alerts on nothing. Within an active ticker a muted pair is
    silent (fires as track_mode='shadow', alert suppressed, keeps collecting
    data) while that ticker's enabled pairs still alert. Benched tickers and
    muted pairs are scanned here UNCONDITIONALLY so nothing ever goes dark.
    """
    if not is_market_open():
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Market closed — skipping stock scan")
        return

    # Fresh per-cycle news cache so one cycle's fired signals reuse fetches but
    # the next cycle gets current headlines. The Phase 6 correlation memo is
    # cleared the same way — one universe fetch per scan, fresh each cycle.
    news_client.clear_cache()
    indicators.clear_correlation_cache()
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Scanning stocks...")
    for ticker, status in _active_stock_watchlist_entries():
        try:
            df = get_stock_data(ticker)
            if df is None:
                print(f"  {ticker}: No data")
                continue
            df      = add_stock_indicators(df)
            signals = detect_stock_signals(ticker, df)
            is_active = status == "active"
            for s in signals:
                # Alert only when the ticker is active AND the pair is enabled.
                # _pair_is_enabled is short-circuited for benched tickers.
                if is_active and _pair_is_enabled(ticker, s.get("setup", "")):
                    # Earnings-blackout gate runs inside _emit_active_signal,
                    # before any alert/trade (and before Phase 5 sentiment +
                    # Phase 6 indicator work). df is passed so the indicator
                    # families snapshot the same candle frame detect used.
                    _emit_active_signal(s, df)
                else:
                    if is_active:
                        # Active ticker, muted pair: gate the alert but keep
                        # collecting data so the pair can recover.
                        print(
                            f"  [muted pair] {ticker}/{s.get('setup', '')} — "
                            f"alert gated, opening as shadow"
                        )
                    send_notification(s, track_mode="shadow", alert=False)
            if not signals:
                print(f"  {ticker}: No signals")
        except Exception as e:
            print(f"  {ticker}: Error — {e}")


def scan_shadow():
    """Shadow-scan the SHADOW_UNIVERSE names that aren't on the active watchlist.

    Runs the IDENTICAL stock signal pipeline (same indicators, same earnings/
    news/macro/VIX context as the active scan) but opens trades tagged
    ``track_mode='shadow'`` and SUPPRESSES the Discord/Pushover alert. The
    resolver settles these like any other open trade; the live-shadow
    evaluator (Phase 3.1-LIVE) later promotes names whose resolved shadow
    signals show the EMA21 Pullback working recently.

    BEST-EFFORT ISOLATION: a shadow ticker is on a disjoint set from the
    active watchlist, every per-ticker failure is caught/logged/skipped, and
    the whole sweep is gated on market hours just like ``scan_stocks``. This
    function is invoked through ``_run_shadow_scan`` which additionally
    guarantees no exception ever escapes to the active scan or scheduler.
    """
    if not is_market_open():
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Market closed — skipping shadow scan")
        return

    # Active watchlist takes precedence: a name already traded live is never
    # also shadow-tracked. Fall back to scanning the full shadow universe if
    # the watchlist is somehow unreadable (shadow must not silently no-op).
    try:
        active = set(_active_stock_watchlist())
    except Exception as exc:  # noqa: BLE001 - never let this break anything
        print(
            f"  shadow: active watchlist unreadable, shadowing full universe: {exc}",
            file=sys.stderr,
        )
        active = set()

    shadow_tickers = [t for t in SHADOW_UNIVERSE if t not in active]
    print(
        f"\n[{datetime.now().strftime('%H:%M:%S')}] Shadow-scanning "
        f"{len(shadow_tickers)} tickers (alerts suppressed)..."
    )

    scanned = 0
    signals_found = 0
    failed = 0
    for idx, ticker in enumerate(shadow_tickers):
        if idx > 0 and _SHADOW_THROTTLE_SECONDS > 0:
            time.sleep(_SHADOW_THROTTLE_SECONDS)
        try:
            df = get_stock_data(ticker)
            if df is None:
                print(f"  shadow {ticker}: no data — skipped", file=sys.stderr)
                failed += 1
                continue
            df = add_stock_indicators(df)
            signals = detect_stock_signals(ticker, df)
            scanned += 1
            for s in signals:
                signals_found += 1
                send_notification(s, track_mode="shadow", alert=False)
        except Exception as exc:  # noqa: BLE001 - one ticker must never kill the sweep
            print(f"  shadow {ticker}: error — {exc}", file=sys.stderr)
            failed += 1

    print(
        f"  shadow scan summary: scanned={scanned} signals={signals_found} "
        f"failed={failed} (of {len(shadow_tickers)} shadow tickers)"
    )


def _run_shadow_scan() -> None:
    """Scheduler/startup wrapper — never lets a shadow failure escape.

    The active watchlist scan and the resolver are sacred; a broken shadow
    layer must degrade silently (with a logged reason) rather than take them
    down. ``scan_shadow`` already isolates per-ticker errors; this is the
    outer guard for anything systemic (import, watchlist read, scheduler).
    """
    try:
        scan_shadow()
    except Exception as exc:
        import sys
        print(f"  shadow scan error: {exc}", file=sys.stderr)


def scan_crypto(*, now: datetime | None = None):
    """Runs every hour, 24/7 — hourly candles, short term trades"""
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Scanning crypto...")
    for ticker in CRYPTO_WATCHLIST:
        try:
            df = get_crypto_data(ticker)
            if df is None:
                print(f"  {ticker}: No data")
                continue
            df      = add_crypto_indicators(df)
            signals = detect_crypto_signals(ticker, df, now=now)
            for s in signals:
                send_notification(s)
            if not signals:
                print(f"  {ticker}: No signals")
        except Exception as e:
            print(f"  {ticker}: Error — {e}")


# ══════════════════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════════════════

def _never_raise(job_fn):
    """Wrap a scheduled job so an uncaught exception can neither kill the
    ``while True: schedule.run_pending()`` loop nor leave the job permanently
    'due' (the ``schedule`` library only advances ``next_run`` after the job
    returns, so an uncaught error would re-fire the failing job every second
    forever). The error is logged to stderr and the next scheduled run tries
    again — the same never-dark contract the ``_run_*`` wrappers already give
    their jobs, extended to the ones that were registered bare."""
    import functools

    @functools.wraps(job_fn)
    def wrapper():
        try:
            job_fn()
        except Exception as exc:  # noqa: BLE001 - one job must never kill the loop
            print(
                f"  scheduled job {job_fn.__name__} error: {exc}",
                file=sys.stderr,
            )

    return wrapper


def main() -> None:
    parser = argparse.ArgumentParser(prog="tradingbot")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run one scan cycle (stocks if market open, crypto otherwise), "
             "print signals to stdout instead of sending Discord/Pushover, "
             "still write to the database, then exit.",
    )
    args = parser.parse_args()

    global DRY_RUN
    DRY_RUN = bool(args.dry_run)

    db.init_db()         # idempotent — creates schema on first deploy
    # Phase 15: a halted bot NEVER restarts on its own. Explicit operator
    # re-authorization (risk reauthorize <token>) is the only way back.
    if risk_of_ruin.get_state() == risk_of_ruin.STATE_HALTED:
        print(
            "risk-of-ruin: HALTED - refusing to start. Re-authorize via: "
            "python -m trading_bot risk reauthorize <token>",
            file=sys.stderr,
        )
        return
    # Phase 3.1: seed the DB-driven watchlist from the hardcoded seed list on
    # first run. Idempotent — a no-op once the table has any rows, so it never
    # clobbers discovery promotions.
    db.seed_active_watchlist(STOCK_WATCHLIST)
    _load_secrets()      # fail fast if any required credential is missing

    if DRY_RUN:
        print("=== DRY RUN — one scan cycle, no notifications, DB writes only ===")
        if is_market_open():
            scan_stocks()
        else:
            scan_crypto()
        return

    print("Trading Bot Started.")
    print(f"Stocks:  {_active_stock_watchlist()}")
    print(f"Crypto:  {CRYPTO_WATCHLIST}")
    print("Stock scan: once daily at market open")
    print("Crypto scan: every hour 24/7")

    # Run both immediately on start. Guarded: a transient failure during the
    # boot pass (e.g. one flaky yfinance/Discord call) must not crash the
    # process before the schedules are even registered — that would put
    # Railway into a restart loop instead of simply catching the next tick.
    _never_raise(scan_stocks)()
    # Shadow scan AFTER the active scan so active alerting/logging is never
    # delayed or blocked by the 100-name shadow load. Wrapped so it can never
    # take down the active pipeline.
    _run_shadow_scan()
    _never_raise(scan_crypto)()
    _never_raise(send_morning_report)()
    # One recovery/materialization pass at boot. It may query/update accepted
    # orders but has no submission path, preserving the operator-confirmed
    # boundary for every new opening order.
    _run_order_lifecycle_cycle()
    # Evaluate readiness once at boot so a capability already over threshold is
    # announced promptly (idempotent — the one-time flag prevents re-notifying).
    _run_readiness_evaluation()

    # Stock scan — once per day at 9:31am EST
    schedule.every().day.at("09:31").do(_never_raise(scan_stocks))

    # Shadow scan at 09:33 EST — two minutes after the active stock scan so
    # active trades/alerts land first. Phase 3.1-LIVE.
    schedule.every().day.at("09:33").do(_run_shadow_scan)

    # Crypto scan — every hour
    schedule.every(1).hours.do(_never_raise(scan_crypto))

    # News updates — every weekday at 8:00am EST
    schedule.every().day.at("08:00").do(_never_raise(send_morning_report))

    # Resolve open trades every hour (Phase 1.3) — lightweight, only touches
    # trades with outcome IS NULL or 'open' and skips already-closed ones.
    schedule.every(1).hours.do(_run_outcome_resolver)

    # Evaluate capability readiness every hour, right after the resolver so the
    # resolved counts are fresh — auto-activates deterministic capabilities and
    # summons ML builds on first crossing, notifying exactly once (Phase 10).
    schedule.every(1).hours.do(_run_readiness_evaluation)

    # Refresh accepted paper orders and replay all durable positive fills once
    # per hourly scanner cycle. This is registered exactly once; the pass
    # itself isolates failures so scans continue and the next cycle retries.
    schedule.every(1).hours.do(_run_order_lifecycle_cycle)

    # Phase 15: risk-of-ruin pass every hour — equity snapshot (drawdown source),
    # Tier-1 circuit breaker, catastrophic detectors, and holding-state retries.
    _never_raise(_run_risk_cycle)()
    schedule.every(1).hours.do(_never_raise(_run_risk_cycle))

    # Phase 15.5: DAILY pairs + watchlist evaluation at 09:40 EST — after the
    # 09:36 VIX snapshot so all daily context is fresh, and independent of the
    # morning report (08:00) and daily-perf summary (00:30 UTC), which still fire
    # on their own triggers. The missing call site that lets the evaluators
    # (already authorized to auto-act per Phase 10) actually run each day.
    _never_raise(_run_evaluator_cycle)()
    schedule.every().day.at("09:40").do(_never_raise(_run_evaluator_cycle))

    # Materialize yesterday's daily_performance row daily at 00:30 UTC
    # (~20:30 ET, well after market close and the hourly resolver). Phase 1.4.
    schedule.every().day.at("00:30", "UTC").do(_run_daily_perf_update)

    # Daily macro regime snapshot at 09:35 EST — four minutes after the stock
    # scan so SPY's prior close is reflected and signals fired in scan_stocks
    # have already grabbed the cached value. Phase 2.1.
    schedule.every().day.at("09:35").do(_run_daily_regime_snapshot)

    # Daily VIX snapshot at 09:36 EST — one minute after the regime snapshot.
    # VIX moves faster than regime so we record it on its own row. Phase 2.2.
    schedule.every().day.at("09:36").do(_run_daily_vix_snapshot)

    # Phase 2.2b — prediction engine. OFF by default; the sweep self-skips
    # until the user enables it via `python -m trading_bot predictions enable`.
    _seed_prediction_defaults()
    # 15-min prediction sweep at :00, :15, :30, :45. The sweep isolates
    # per-ticker errors itself, but its settings/window reads sit outside that
    # isolation — guard the whole job like everything else.
    for minute in ("00", "15", "30", "45"):
        schedule.every().hour.at(f":{minute}").do(_never_raise(_run_prediction_sweep))
    # Resolution sweep every 5 minutes — closes out predictions whose
    # 15-min target window has passed and whose candle is now available.
    schedule.every(5).minutes.do(_never_raise(_run_prediction_resolution))
    print(f"Schedules registered: {schedule.jobs}")

    # Phase 15: the top-level catastrophic guard. ANY unhandled exception that
    # escapes the per-job wrappers is caught, logged with full context, and
    # routed to the Tier-2 emergency shutdown — the process never crashes
    # silently, and it never keeps trading after a halt.
    def _core_cycle() -> None:
        schedule.run_pending()
        time.sleep(1)

    def _catastrophic_handler(reason: str) -> None:
        from trading_bot.broker import AlpacaBroker

        risk_of_ruin.emergency_shutdown(AlpacaBroker(), trigger=reason)

    while True:
        risk_of_ruin.run_guarded(_core_cycle, on_catastrophic=_catastrophic_handler)
        if risk_of_ruin.get_state() == risk_of_ruin.STATE_HALTED:
            print(
                "risk-of-ruin: HALTED - exiting. Re-authorize via: "
                "python -m trading_bot risk reauthorize <token>",
                file=sys.stderr,
            )
            return


if __name__ == "__main__":
    main()
