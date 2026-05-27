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
    context,
    db,
    outcomes,
    performance,
    predictions,
    regime,
    settings,
    vix,
)
from trading_bot.models import Signal, Trade
from trading_bot.secrets import get_required

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

# Stock watchlist — backtested, Tier 1 only
STOCK_WATCHLIST = ["BLK", "GOOGL", "META", "GS", "NOW", "AMZN", "LLY", "TSLA"]
TREND_CONT_TICKERS = ["JPM", "GS", "GOOGL", "NOW", "SPY", "BLK", "AMZN"]
NEWS_WATCHLIST = ["BLK", "GOOGL", "META", "GS", "NOW", "AMZN", "LLY", "TSLA", "PLTR", "NVDA", "AAPL"]

# Crypto watchlist — backtested on hourly candles
CRYPTO_WATCHLIST = ["BTC-USD", "BNB-USD", "ETH-USD"]

NOTIFY_METHOD    = "pushover"

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
    except Exception:
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
                requests.post(DISCORD_WEBHOOK_URL, json={"content": chunk})
        else:
            requests.post(DISCORD_WEBHOOK_URL, json={"content": report})
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
            })
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


def log_signal(signal: dict) -> int | None:
    """Translate a legacy-shape ``signal`` dict into a ``Signal`` row and open
    a corresponding ``Trade`` record (Phase 1.3).

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
    """Write any missing prediction settings to their defaults. Idempotent."""
    for key, value in _PRED_DEFAULTS.items():
        if settings.get(key) is None:
            settings.set(key, value)


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


def _send_prediction_notification(pred: object) -> None:
    """Send the prediction alert via Discord + Pushover. Plain ASCII."""
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
        return

    try:
        requests.post(DISCORD_WEBHOOK_URL, json={"content": msg})
    except Exception as exc:
        print(f"  prediction Discord error: {exc}", file=sys.stderr)
    try:
        requests.post("https://api.pushover.net/1/messages.json", data={
            "token": PUSHOVER_APP_TOKEN,
            "user":  PUSHOVER_USER_KEY,
            "title": f"Prediction: {ticker} {direction}",
            "message": msg,
        })
    except Exception as exc:
        print(f"  prediction Pushover error: {exc}", file=sys.stderr)


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
        requests.post(DISCORD_WEBHOOK_URL, json={"content": msg})
    except Exception as exc:
        print(f"  resolution Discord error: {exc}", file=sys.stderr)
    try:
        requests.post("https://api.pushover.net/1/messages.json", data={
            "token": PUSHOVER_APP_TOKEN,
            "user":  PUSHOVER_USER_KEY,
            "title": f"Resolved: {ticker} {outcome_label}",
            "message": msg,
        })
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
            _send_prediction_notification(pred)
            db.update_prediction(pid, notified=True)
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

    if DRY_RUN:
        print("\n--- DRY-RUN signal (no Discord/Pushover) ---")
        print(msg)
        return

    try:
        requests.post(DISCORD_WEBHOOK_URL, json={
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
                ],
            }],
        })
        print("  Discord notification sent")
    except Exception as e:
        print(f"  Discord error: {e}")

    try:
        requests.post("https://api.pushover.net/1/messages.json", data={
            "token":   PUSHOVER_APP_TOKEN,
            "user":    PUSHOVER_USER_KEY,
            "title":   f"Alert: {signal['ticker']} {signal['direction']}",
            "message": msg,
        })
        print("  Pushover notification sent")
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
    _load_secrets()      # fail fast if any required credential is missing

    if DRY_RUN:
        print("=== DRY RUN — one scan cycle, no notifications, DB writes only ===")
        if is_market_open():
            scan_stocks()
        else:
            scan_crypto()
        return

    print("Trading Bot Started.")
    print(f"Stocks:  {STOCK_WATCHLIST}")
    print(f"Crypto:  {CRYPTO_WATCHLIST}")
    print("Stock scan: once daily at market open")
    print("Crypto scan: every hour 24/7")

    # Run both immediately on start
    scan_stocks()
    scan_crypto()
    send_morning_report()

    # Stock scan — once per day at 9:31am EST
    schedule.every().day.at("09:31").do(scan_stocks)

    # Crypto scan — every hour
    schedule.every(1).hours.do(scan_crypto)

    # News updates — every weekday at 8:00am EST
    schedule.every().day.at("08:00").do(send_morning_report)

    # Resolve open trades every hour (Phase 1.3) — lightweight, only touches
    # trades with outcome IS NULL or 'open' and skips already-closed ones.
    schedule.every(1).hours.do(_run_outcome_resolver)

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
    # 15-min prediction sweep at :00, :15, :30, :45.
    for minute in ("00", "15", "30", "45"):
        schedule.every().hour.at(f":{minute}").do(_run_prediction_sweep)
    # Resolution sweep every 5 minutes — closes out predictions whose
    # 15-min target window has passed and whose candle is now available.
    schedule.every(5).minutes.do(_run_prediction_resolution)
    print(f"Schedules registered: {schedule.jobs}")

    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    main()
