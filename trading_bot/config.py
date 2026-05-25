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
