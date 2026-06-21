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
