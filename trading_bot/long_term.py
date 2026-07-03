"""Long-term buy-and-hold trading system — Phase 14 (stocks + crypto, PAPER).

Entry and protective-exit logic for the LONG_TERM (buy-hold stocks) and CRYPTO
(long-term BTC/ETH/BNB) pools reserved by the Phase 12 allocator. Buy-and-hold:
fractional shares, NO options, NO tight stops.

Entry candidates are generated in the daily scan and feed the SAME
operator-inspected Phase 12 allocation plan as swing/options signals — there is
NO new auto-execution of entries. The one thing that runs and acts automatically
is the PROTECTIVE EXIT watcher (closing, never opening), matching the Phase 13
options-watcher / kill-switch precedent.

Entry gates (applied by the candidate generator):
  1. fundamental red-flag screen (stocks only, FAIL-OPEN — a loose filter, not a
     ranker; blocks only on a genuine red flag);
  2. long-horizon technical confirmation (stocks + crypto): price above the
     TREND_PERIOD trend AND ADX >= a moderate floor AND RSI <= overbought —
     reusing the Phase 6 indicator functions, not reimplementing them;
  3. earnings SOFT wait-window (stocks only) — distinct from the Phase 5 blackout.

Protective exits (this phase, no profit-taking): close on EITHER a trend
breakdown (N consecutive closes below the trend) OR a drawdown-from-entry stop.

The crypto SWING signals (Oversold Reversal, Momentum Breakout) remain data-only
FOREVER — they never route to execution (enforced in allocation.build_plan via
config.DATA_ONLY_SIGNAL_TYPES). Only this module's long-term crypto entry signal
may route to the CRYPTO pool. Still PAPER ONLY.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class LongTermCandidate:
    """A buy-and-hold entry candidate to feed the Phase 12 allocation plan.

    ``signal_type`` is one of ``config.LONGTERM_STOCK_SIGNAL`` /
    ``LONGTERM_CRYPTO_SIGNAL`` — the ONLY long-term types that route to
    execution. ``entry_rationale`` records which gates it cleared (audit)."""

    ticker: str
    asset_class: str               # 'stock' | 'crypto'
    signal_type: str               # long_term_stock | long_term_crypto
    entry_price: float
    entry_rationale: str = ""


@dataclass(frozen=True)
class LongTermPosition:
    """An open/closed buy-and-hold position (no tight stop; protective exit only)."""

    ticker: str
    asset_class: str
    entry_price: float
    entry_date: datetime
    qty: float
    status: str = "open"           # 'open' | 'closed'
    exit_price: float | None = None
    exit_date: datetime | None = None
    exit_reason: str | None = None  # trend_breakdown | drawdown_stop
    id: int | None = None
