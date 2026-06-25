"""Central configuration and runtime environment detection."""

import os
from pathlib import Path

# Runtime: "railway" when deployed, "local" otherwise
RUNTIME: str = "railway" if os.getenv("RAILWAY_ENVIRONMENT") else "local"

# Project root (two levels up from this file)
ROOT_DIR: Path = Path(__file__).parent.parent

# Database path (placeholder — used in Phase 1.1c)
DB_PATH: Path = (
    Path("/data/trading_bot.db")
    if RUNTIME == "railway"
    else ROOT_DIR / "trading_bot.db"
)

# Legacy CSV path
CSV_LEGACY_PATH: Path = ROOT_DIR / "signal_log.csv"

# ──────────────────────────────────────────────────────────────────────────
# Watchlist state machine (Phase 3.3)
# ──────────────────────────────────────────────────────────────────────────
# Active<->benched management thresholds. Deliberately mirror the shadow
# PROMOTION thresholds (trading_bot.shadow_discovery) so the bar to ENTER the
# active set (shadow promotion) equals the bar to RECOVER into it. The window
# and minimum-sample also match. The dead band between SM_DEMOTE_EXPECTANCY
# and SM_PROMOTE_EXPECTANCY is the hysteresis that prevents flapping.
SM_MIN_CLOSED_SIGNALS: int = 10   # minimum resolved trades to act (else hold)
SM_WINDOW_DAYS: int = 60          # recency window for the expectancy stats
SM_DEMOTE_EXPECTANCY: float = 0.0   # active -> benched when expectancy <= this
SM_PROMOTE_EXPECTANCY: float = 0.05  # benched -> active when expectancy >= this
SM_MIN_ACTIVE: int = 5            # never demote below this many active tickers

# ──────────────────────────────────────────────────────────────────────────
# Per-pair signal gating (Phase 4)
# ──────────────────────────────────────────────────────────────────────────
# (ticker, signal_type) enable/mute thresholds. Mirror the watchlist state
# machine (SM_*) one grain finer for coherence: same window, same min sample,
# same hysteresis dead band. No min-floor — muting one setup on a ticker can
# never empty the watchlist, so the floor is deliberately omitted.
SP_MIN_CLOSED_SIGNALS: int = 10    # minimum resolved trades to act (else hold)
SP_WINDOW_DAYS: int = 60           # recency window for the expectancy stats
SP_MUTE_EXPECTANCY: float = 0.0     # enabled -> muted when expectancy <= this
SP_ENABLE_EXPECTANCY: float = 0.05  # muted -> enabled when expectancy >= this

# ──────────────────────────────────────────────────────────────────────────
# News & sentiment (Phase 5)
# ──────────────────────────────────────────────────────────────────────────
# Earnings blackout window: suppress a stock alert if a known earnings date
# falls within this many days. We use the trade's INTENDED hold horizon (parsed
# from the signal's hold_days estimate, ~1-5 days), NOT the resolver's 30-day
# expiry — a 30-day window would blanket-suppress, since quarterly earnings hit
# nearly every name. This is the fallback when hold_days can't be parsed.
EARNINGS_BLACKOUT_DEFAULT_DAYS: int = 5

# Sentiment LLM model — Haiku is the cost/latency choice for per-signal scoring.
SENTIMENT_LLM_MODEL: str = "claude-haiku-4-5-20251001"
# "Heavy news day" flag: set when a ticker's fetched headline count is >= this.
SENTIMENT_HEAVY_NEWS_THRESHOLD: int = 8

# ──────────────────────────────────────────────────────────────────────────
# Advanced indicator families (Phase 6)
# ──────────────────────────────────────────────────────────────────────────
# Five deliberately INDEPENDENT families attached to fired signals as advisory
# context (see trading_bot.indicators for the multiple-comparisons caveat).
#
# Volatility regime: classify the current ATR against its own longer baseline.
# ratio < LOW -> 'low' (quiet), ratio > HIGH -> 'high' (violent), else 'normal'.
VOL_REGIME_LOW_RATIO: float = 0.8
VOL_REGIME_HIGH_RATIO: float = 1.2
VOL_ATR_PERIOD: int = 14            # ATR lookback (matches the legacy scanner ATR)
VOL_ATR_BASELINE_PERIOD: int = 20   # baseline ATR the regime ratio is taken against
VOL_REALIZED_PERIOD: int = 20       # realized-volatility (std of returns) lookback
VOL_BAND_NUM_STD: float = 2.0       # width of the realized-vol envelope, in std

# Momentum (RSI) and trend strength (ADX) — one representative each, no
# correlated duplicates (Stochastic/Williams/CCI would be the same family).
RSI_PERIOD: int = 14
ADX_PERIOD: int = 14

# Cross-asset correlation among active-watchlist names: rolling return window +
# the concentration bands the average pairwise correlation is classified into.
CORR_RETURN_WINDOW: int = 20        # trailing daily returns used per ticker
CORR_MIN_NAMES: int = 2             # need at least this many peers to correlate
CORR_CONCENTRATED_AT: float = 0.6   # avg pairwise corr >= this -> 'concentrated'
CORR_DIVERSIFIED_AT: float = 0.3    # avg pairwise corr <  this -> 'diversified'
