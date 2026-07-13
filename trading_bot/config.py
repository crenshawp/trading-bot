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

# ──────────────────────────────────────────────────────────────────────────
# Risk management (Phase 7) — ADVISORY, NOTIONAL ONLY
# ──────────────────────────────────────────────────────────────────────────
# The operator is NOT trading; there is no real capital. Position sizes and
# portfolio-risk verdicts are computed against this NOTIONAL account and
# RECORDED next to the eventual trade outcome so later phases can judge whether
# the rules were sound. Nothing here sizes, blocks, or shrinks a real position —
# every number is a recommendation, never an instruction.
NOTIONAL_ACCOUNT: float = 10_000.0        # notional account size, in dollars
RISK_PER_TRADE_PCT: float = 1.0           # % of notional risked on one trade
MAX_PORTFOLIO_RISK_PCT: float = 6.0       # advisory cap on summed open risk %
MAX_POSITION_PCT: float = 20.0            # advisory per-position notional cap %
MAX_CORRELATED_CLUSTER_PCT: float = 25.0  # advisory cap on a correlated cluster %
# Stop distance for volatility-normalized sizing = ATR * this multiple. 1.5
# matches the scanner's actual stop (sl = price - atr*1.5), so the advisory
# size lines up with the stop the trade is really tracked against.
RISK_ATR_STOP_MULTIPLE: float = 1.5

# ──────────────────────────────────────────────────────────────────────────
# Self-optimization (Phase 9) — DEGRADATION DETECTION + FEATURE EVALUATION
# ──────────────────────────────────────────────────────────────────────────
# This layer FLAGS AND REPORTS ONLY — it never auto-tunes a threshold or changes
# any pair/ticker/watchlist status. Every finding carries its sample size; any
# claim below SO_MIN_SAMPLE is labeled "insufficient sample - not actionable",
# because testing many features against a small outcome set manufactures false
# positives (multiple comparisons). Acting on a finding is a deliberate, later,
# sample-gated step — not this phase.
SO_MIN_SAMPLE: int = 30            # min resolved (win/loss) trades for an actionable claim
SO_DEGRADE_WINDOW_DAYS: int = 30   # the RECENT window degradation is measured over
SO_BASELINE_WINDOW_DAYS: int = 90  # the older BASELINE window end (must exceed degrade)
# Minimum drop in expectancy (mean resolved pnl %, in percentage points) of the
# recent window below baseline required to flag degradation. Paired with the
# sample floor so a normal cold streak (small n) never trips an alarm.
SO_MEANINGFUL_DELTA: float = 0.15

# ──────────────────────────────────────────────────────────────────────────
# Unified readiness gate (Phase 10)
# ──────────────────────────────────────────────────────────────────────────
# The ONE authority over every capability's data-sufficiency. Deterministic
# thresholds are the EXISTING Phase 3/4/9 minimums, centralized here UNCHANGED:
# SM_MIN_CLOSED_SIGNALS, SP_MIN_CLOSED_SIGNALS, MIN_SHADOW_SIGNALS, SO_MIN_SAMPLE.
# MIN_SHADOW_SIGNALS lives here (moved from shadow_discovery, value unchanged) so
# the readiness registry can read it without an import cycle; shadow_discovery
# re-exports it for back-compat.
MIN_SHADOW_SIGNALS: int = 10
# ML build-readiness gates — deliberately LARGE counts befitting model training.
# Crossing one only SUMMONS a human build; no model trains or deploys itself.
ML_PATTERN_MIN_SAMPLE: int = 500    # ml_pattern_recognition: ready-to-build gate
ML_SIZING_MIN_SAMPLE: int = 750     # ml_predictive_sizing: ready-to-build gate

# ──────────────────────────────────────────────────────────────────────────
# Capital allocation engine (Phase 12) — PLAN-ONLY, paper account state
# ──────────────────────────────────────────────────────────────────────────
# The allocator turns simultaneously-firing signals into an ordered execution
# PLAN; it never executes. Deployment caps are on DEPLOYED CAPITAL per pool, by
# confidence tier — HIGH-confidence signals may use up to 50% of a pool's
# capital, all others up to 30%. Position COUNT is an emergent output, not a cap.
POOL_DEPLOY_CAP_HIGH: float = 0.50      # HIGH-tier deployed-capital ceiling / pool
POOL_DEPLOY_CAP_NORMAL: float = 0.30    # NORMAL-tier deployed-capital ceiling / pool
# A position whose volatility-normalized size puts less than this many dollars at
# risk is SKIPPED, never undersized (undersizing breaks the Phase 7 risk math).
MIN_DOLLAR_RISK: float = 10.0

# Three logical pools tracked independently against one paper account's capital.
# LONG_TERM (buy-hold stocks) and CRYPTO (long-term BTC/ETH/BNB) are populated by
# the Phase 14 long-term candidate generator.
POOL_SWING = "SWING"
POOL_LONG_TERM = "LONG_TERM"
POOL_CRYPTO = "CRYPTO"
POOLS: tuple[str, ...] = (POOL_SWING, POOL_LONG_TERM, POOL_CRYPTO)
# Notional split of the account's capital across the pools (sums to 1.0). Tune
# freely.
POOL_CAPITAL_SPLIT: dict[str, float] = {
    POOL_SWING: 0.50, POOL_LONG_TERM: 0.30, POOL_CRYPTO: 0.20,
}
# Route a signal to its pool by signal_type (fallback: asset_class → stock=SWING,
# crypto=CRYPTO). The crypto SWING signals (oversold_reversal / momentum_breakout)
# are DELIBERATELY ABSENT — they are data-only forever (see DATA_ONLY_SIGNAL_TYPES,
# enforced in allocation.build_plan) and must NEVER route to execution. Only the
# Phase 14 long-term signals route into the LONG_TERM / CRYPTO execution pools.
SIGNAL_TYPE_TO_POOL: dict[str, str] = {
    "ema21_pullback": POOL_SWING,
    "long_term_stock": POOL_LONG_TERM,
    "long_term_crypto": POOL_CRYPTO,
}

# Composite confidence weights (sum 1.0; per-pair expectancy is PRIMARY). The
# formula lives in one documented function (allocation.composite_confidence) so
# it stays auditable and tunable later (Phase 9/17), never magic.
CONF_WEIGHT_EXPECTANCY: float = 0.50    # primary: per-pair windowed expectancy
CONF_WEIGHT_AGREEMENT: float = 0.25     # Phase 6 indicator-family agreement (0-3)
CONF_WEIGHT_SENTIMENT: float = 0.15     # Phase 5 advisory sentiment (-1..+1)
CONF_WEIGHT_VOL: float = 0.10           # vol_regime (low cleaner than high)
CONF_EXPECTANCY_REF: float = 1.0        # expectancy % mapping to a full 1.0 term
CONF_ADX_TREND_MIN: float = 20.0        # ADX >= this == a trustworthy trend
CONF_RSI_OVERBOUGHT: float = 70.0       # long not-confirmed above this
CONF_RSI_OVERSOLD: float = 30.0         # short not-confirmed below this
# HIGH tier requires the STRONGEST factors to clear these AND the score gate.
CONF_HIGH_EXPECTANCY: float = 0.50
CONF_HIGH_AGREEMENT: int = 2
CONF_HIGH_SCORE: float = 0.50

# ──────────────────────────────────────────────────────────────────────────
# Options execution layer (Phase 13) — SWING, single-leg, PAPER
# ──────────────────────────────────────────────────────────────────────────
# One contract controls OPTION_MULTIPLIER shares of premium exposure. This is the
# single most error-prone constant in options code: EVERY cost/sizing calculation
# routes through it, and an explicit test guards against a 100x mistake.
OPTION_MULTIPLIER: int = 100
# Strike selection targets a delta band (absolute value; puts use the negatives).
TARGET_DELTA_LOW: float = 0.65
TARGET_DELTA_HIGH: float = 0.75
# An UNDERSIZED option widens the band DOWN to this floor (cheaper, further OTM)
# to fit the allocated capital — still a WHOLE contract, never a fractional one.
UNDERSIZED_DELTA_FLOOR: float = 0.50
# Never enter inside the final fortnight: minimum days-to-expiration at entry.
MIN_DTE: int = 14
# Liquidity floor — a contract failing EITHER gate is excluded regardless of delta.
MIN_OPEN_INTEREST: int = 100
MAX_SPREAD_PCT: float = 10.0        # max bid-ask spread as % of mid price

# ──────────────────────────────────────────────────────────────────────────
# Long-term trading system (Phase 14) — BUY-AND-HOLD, stocks + crypto, PAPER
# ──────────────────────────────────────────────────────────────────────────
# Buy-and-hold: fractional shares, no options, no tight stops. Entry candidates
# feed the SAME operator-inspected Phase 12 allocation plan (no new auto-exec of
# entries); only the PROTECTIVE EXIT watcher acts automatically (closing).
LONGTERM_STOCK_SIGNAL = "long_term_stock"
LONGTERM_CRYPTO_SIGNAL = "long_term_crypto"
# The crypto SWING signals are data-only FOREVER — never routed to execution.
DATA_ONLY_SIGNAL_TYPES: frozenset[str] = frozenset(
    {"oversold_reversal", "momentum_breakout"}
)
LONGTERM_STOCK_UNIVERSE: tuple[str, ...] = (
    "BLK", "GOOGL", "META", "GS", "NOW", "AMZN", "LLY", "TSLA",
)
LONGTERM_CRYPTO_UNIVERSE: tuple[str, ...] = ("BTC-USD", "ETH-USD", "BNB-USD")

# Earnings SOFT window (stocks only) — DISTINCT from the Phase 5 hold-window
# blackout: skip an entry this cycle if earnings fall within this many days
# (clean entry price, not gap-risk avoidance), and simply retry a later cycle.
EARNINGS_WAIT_DAYS: int = 5

# Diversification-weighted sizing (NOT the Phase 7 ATR/stop model): each position
# gets up to this % of its pool's capital. No stop-distance sizing — no tight stop.
MAX_POSITION_WEIGHT_PCT: float = 15.0

# Technical entry confirmation at a long horizon (stocks + crypto): price above
# the TREND_PERIOD trend AND ADX >= a moderate-trend floor AND RSI <= overbought.
TREND_PERIOD: int = 200
ADX_MIN_TREND: float = 20.0
RSI_OVERBOUGHT_MAX: float = 70.0

# Protective exit (no profit-taking this phase): close on EITHER a trend breakdown
# (this many consecutive closes below the TREND_PERIOD trend) OR a drawdown from
# entry exceeding the stop — whichever triggers first.
TREND_BREAKDOWN_DAYS: int = 3
MAX_DRAWDOWN_STOP_PCT: float = 25.0

# Fundamental RED-FLAG screen (stocks only) — a LOOSE filter, not a quality
# ranker. A candidate is blocked ONLY when BOTH hold (a genuine red flag):
# earnings growth below the floor AND debt/equity above the ceiling. Missing
# either datum fails OPEN (does not block). debt/equity from yfinance is a
# PERCENT (e.g. 150.0 == 1.5x), so the ceiling is expressed the same way.
FUND_EARNINGS_GROWTH_FLOOR: float = -0.20     # -20% YoY earnings growth
FUND_DEBT_EQUITY_CEILING: float = 200.0       # 2.0x debt/equity (yfinance %)

# ──────────────────────────────────────────────────────────────────────────
# Live candidate sourcing (Phase 17)
# ──────────────────────────────────────────────────────────────────────────
# Recency window for pulling fired signals into an allocation plan: one daily
# stock-scan cycle (the slowest cadence — crypto fires hourly, long-term
# candidates are daily). A signal older than this is stale: its entry price
# belongs to a market state the next scan has already superseded.
CANDIDATE_RECENCY_HOURS: int = 24

# ──────────────────────────────────────────────────────────────────────────
# Risk of ruin (Phase 15) — CIRCUIT BREAKERS + EMERGENCY SHUTDOWN
# ──────────────────────────────────────────────────────────────────────────
# The mandatory safety layer required before any live-switch decision. Two
# structurally distinct tiers:
#   TIER 1 (pause)  — stop opening NEW positions; existing positions keep being
#                     managed by their own watchers. Cleared only by operator
#                     re-authorization.
#   TIER 2 (emergency shutdown) — close EVERY open position, confirm closure via
#                     reconciliation, then halt. Never claims a closure the
#                     broker did not confirm.
MAX_CONSECUTIVE_LOSSES: int = 7        # Tier 1: trailing resolved-loss streak
MAX_DRAWDOWN_PCT: float = 20.0         # Tier 1: peak-to-trough EQUITY drawdown %
MAX_CONSECUTIVE_BROKER_ERRORS: int = 3     # Tier 2: broker error/rejection streak
MAX_CONSECUTIVE_RECONCILE_DIVERGENCES: int = 3  # Tier 2: unreconcilable persistence
# The capability revoked on a trip. Permissive-by-default (no row == authorized),
# matching the Phase 10 ledger convention; consulted before ANY new entry plans.
ENTRY_CAPABILITY: str = "new_position_entry"
# Operator confirmation tokens (Phase 10 had no token CLI — the discipline is
# DEFINED here): the CLI refuses to act unless the exact token is supplied.
ROR_REAUTHORIZE_TOKEN: str = "CONFIRM-REAUTHORIZE"
ROR_KILLSWITCH_TOKEN: str = "CONFIRM-KILLSWITCH"
