# Trading Bot — Project Notes

## Overview

Python algorithmic trading alert bot. Scans Tier 1 stock watchlist
(BLK, GOOGL, META, GS, NOW, AMZN, LLY, TSLA) once daily at 9:31am EST
and crypto watchlist (BTC-USD, BNB-USD, ETH-USD) hourly. Signals fire
via Discord and Pushover. Deployed on Railway.

Primary live signal: EMA21 Pullback (stocks). Crypto: Oversold Reversal
and Momentum Breakout.

## Audit gates

Every commit must pass all three:

```
uv run pytest
uv run ruff check .
uv run mypy trading_bot
```

Do not bypass these with broad `# type: ignore` or `# noqa` directives.
If something is genuinely impossible to satisfy, document why in a
comment on the suppression.

## Package layout

- `trading_bot/` — new code lives here (Phase 1.1+ migration target)
- Root-level `.py` files — legacy bot code, not yet migrated
- `tests/` — pytest suite, mirrors `trading_bot/` structure

## Phase tracker

- [x] Phase 1.1a — Audit harness & tooling
- [x] Phase 1.1b — Secrets layer
- [x] Phase 1.1c — SQLite foundation
- [x] Phase 1.2 — Migrate live scanner to SQLite
- [x] Phase 1.3 — Trade outcome tracking
- [x] Phase 1.4 — Performance attribution
- [x] Phase 2.1 — Macro regime detection (bull/bear/sideways)
- [x] Phase 2.2 — VIX context (low/elevated/high/extreme + regime x vix matrix)
- [x] Phase 2.2b — 15-min direction prediction engine for crypto event markets
- [x] Phase 2.3 — Market context score (composite regime x VIX + --min-context filter)

### Phase 2 is complete. Phase 3 queued:

- [ ] Phase 3 — Watchlist auto-discovery. New tickers should be
      validated against the current context_score before being promoted
      to the live watchlist; a ticker discovered in a hostile context
      (1-2) should be held for re-evaluation in a better window before
      we trade it live.
- [ ] Phase 4 — Strategy evolution: filter signals at fire time using
      the context score (suppress when context_score < threshold,
      possibly per-signal). Phase 2.3 deliberately makes context pure
      metadata; Phase 4 turns it into a gate.

## Outcome resolution

Phase 1.3 added automated win/loss/expired tracking for every fired signal.

- Every call to `log_signal` opens a `Trade` row with `outcome='open'`.
- The resolver (`trading_bot.outcomes.resolve_all_open_trades`) runs **hourly**
  via the scanner's `schedule` loop. It pulls historical candles from yfinance
  for each open trade and decides whether `take_profit` or `stop_loss` was hit
  first within the hold window (default 30 days for stocks, 14 days for crypto).
- **Same-candle TP+SL is conservatively marked as `loss`.** OHLC can't reveal
  intra-candle ordering; we assume SL hits first to keep the win rate honest.
- **PnL is percentage-only** for now. `pnl_dollars` stays `None` until Phase 7
  (Risk Management) introduces position sizing.
- `python -m trading_bot outcomes backfill` is a **one-shot** to seed Trades
  for the 20 CSV-imported historical signals. It IS idempotent (re-running
  creates zero new trades), but pointless to re-run.
- `python -m trading_bot outcomes status` prints per-signal-type win rates
  and average PnL.

## Reporting

Phase 1.4 added a `report` CLI that answers performance questions without
hand-rolling SQL. Every command queries the existing signals+trades tables
and never hits the network.

```
python -m trading_bot report overall              # single-block summary
python -m trading_bot report recent --days 30     # last N days
python -m trading_bot report by-signal            # per signal_type
python -m trading_bot report by-signal --by-regime  # signal x regime matrix (Phase 2.1)
python -m trading_bot report by-ticker [--stocks | --crypto]
python -m trading_bot report by-ticker --by-regime  # ticker x regime matrix (Phase 2.1)
python -m trading_bot report by-asset             # stock vs crypto
python -m trading_bot report by-regime            # per macro regime (Phase 2.1)
python -m trading_bot report by-vix               # per VIX band (Phase 2.2)
python -m trading_bot report by-regime-vix        # regime x vix matrix (Phase 2.2 headline view)
python -m trading_bot report by-context           # per context score 0-5 (Phase 2.3)
python -m trading_bot report context-matrix       # regime x vix as a geometric matrix (Phase 2.3)
# --min-context N filter on any of: by-signal, by-ticker, by-regime, by-vix,
# by-regime-vix, predictions. 0 = no filter; >0 excludes trades below N.
python -m trading_bot report predictions          # prediction accuracy (Phase 2.2b)
python -m trading_bot report predictions --by-regime
python -m trading_bot report predictions --by-vix
python -m trading_bot report predictions --by-time  # accuracy by hour-of-day ET
python -m trading_bot report by-signal --by-vix   # signal x vix matrix (Phase 2.2)
python -m trading_bot report by-ticker --by-vix   # ticker x vix matrix (Phase 2.2)
python -m trading_bot report cross                # signal_type x ticker (>=3 trades)
python -m trading_bot report daily-backfill       # one-shot daily_performance seed
python -m trading_bot report daily [--date YYYY-MM-DD]
```

### Regime CLI (Phase 2.1)

```
python -m trading_bot regime current              # latest bull/bear/sideways + indicators
python -m trading_bot regime history --days 30    # recent regime_snapshots
python -m trading_bot regime backfill             # one-shot: tag closed trades w/ historical regime
```

### VIX CLI (Phase 2.2)

```
python -m trading_bot vix current                 # latest VIX level + band
python -m trading_bot vix history --days 30       # recent vix_snapshots
python -m trading_bot vix backfill                # one-shot: tag closed trades w/ historical VIX
```

### Predictions CLI (Phase 2.2b)

```
python -m trading_bot predictions enable          # flip master switch ON
python -m trading_bot predictions disable         # flip master switch OFF (default)
python -m trading_bot predictions status          # show settings + last-24h activity
python -m trading_bot predictions window --start HH:MM --end HH:MM
python -m trading_bot predictions tickers --set BTC-USD,ETH-USD
python -m trading_bot predictions pause --minutes 60   # temporarily mute
```

### Context CLI (Phase 2.3)

The single unified dashboard. Replaces the two-step "regime current + vix
current" workflow with one composite view. Use this by default; drop down
to the per-axis CLIs only when you need to diagnose one specific layer.

```
python -m trading_bot context current             # composite snapshot + score 0-5
python -m trading_bot context history --days 30   # joined regime+vix history with score per day
python -m trading_bot context backfill            # tag context_score on legacy trades + predictions
```

### Aggregation rules (the only ones that matter)

- **"Closed" means** `outcome IN ('win', 'loss', 'expired')`. Trades with
  `outcome IS NULL` or `outcome = 'open'` are excluded from every stat.
- **Win rate = wins / (wins + losses).** Expired trades do **not** count
  toward the denominator. All-expired slices report `win_rate = None`
  (the CLI prints `-`), not `0%`.
- **Avg / best / worst PnL include expired trades.** An expired trade with
  no available exit (`pnl_pct IS NULL`) is silently ignored by AVG/MAX/MIN
  but still counts toward `total` and the `expired` bucket.
- **Empty slices return numeric `None`**, never 0 — so the CLI can render
  `-` instead of misleading "0% win rate".

### Daily performance

`daily_performance` rows are populated by the scheduler at **00:30 UTC**
each day (covers yesterday's trades). The hourly outcome resolver runs
first, so most decisions are settled by the time daily perf updates. Use
`python -m trading_bot report daily-backfill` once after a long downtime
to fill in any gaps; afterward the scheduler keeps it fresh.

## Macro Regime (Phase 2.1)

Every trade carries a `market_regime` tag at fire time. Three regimes,
classified from SPY:

- **bull** — SPY > 200 EMA AND 50 EMA > 200 EMA AND 5-day slope of 50 EMA > 0
- **bear** — SPY < 200 EMA AND 50 EMA < 200 EMA AND 5-day slope < 0
- **sideways** — anything else (mixed / boundary / transitional)
- **unknown** — fallback when yfinance is unreachable at fire time;
  signal capture is more important than regime tagging, so we log the
  trade with `regime='unknown'` rather than skipping it.

### Key rules

- Regime is tagged at **fire time**, never retroactively. The single
  exception is `python -m trading_bot regime backfill`, a one-shot that
  fills `market_regime` for closed trades that pre-date Phase 2.1.
- SPY drives regime for **all** asset classes — including crypto. A
  bear market in equities is relevant context for any trade fired
  during it.
- `regime_snapshots` table holds one row per date. Populated by:
  1. the scheduled daily job at 09:35 EST (live), and
  2. the backfill subcommand (historical dates).
- Cache: `get_current_regime()` is cached 24h in memory + on disk at
  `.regime_cache.json` (gitignored). Pass `force_refresh=True` to
  bypass — the daily snapshot job does this.

### When yfinance fails

`RegimeFetchError` is raised; the scanner catches it and tags the trade
`'unknown'` plus a stderr log line. The signal still fires to
Discord/Pushover and is still logged to the database — the regime
column is the only thing affected.

## VIX Context (Phase 2.2)

Layered on top of macro regime. Every trade also carries a
`vix_level` (float) and `vix_band` (one of below) at fire time. Regime
tells us direction; VIX tells us how violently the market is moving.

- **low** — `VIX < 20` (quiet, complacent)
- **elevated** — `20 <= VIX < 30` (cautious, choppy)
- **high** — `30 <= VIX < 40` (fear, fast moves)
- **extreme** — `VIX >= 40` (panic, do not trade size)
- **unknown** — fallback when yfinance is unreachable at fire time;
  `vix_level` is NULL in this case, `vix_band` is `'unknown'`.
  Signal still fires.

Boundary values go to the **higher** band — VIX 20.0 is elevated, not
low. Standard market convention.

### Key rules

- VIX is tagged at fire time, never retroactively (except via
  `python -m trading_bot vix backfill`).
- VIX drives context for ALL asset classes including crypto. A VIX
  spike rattles crypto along with equities — that's relevant.
- `vix_snapshots` table holds one row per trading date. Populated by:
  1. the scheduled daily job at 09:36 EST (one minute after the
     regime snapshot), and
  2. the backfill subcommand (historical dates).
- Cache: `get_current_vix()` is cached **1 hour** in memory + on disk
  at `.vix_cache.json` (gitignored). The shorter TTL vs regime's 24h
  reflects how much faster VIX moves; crypto scans (hourly) always
  see fresh data.
- VIX does **not** trade weekends. `vix_snapshots` is sparse on
  Sat/Sun and on market holidays. Backfill skips those dates.

### When yfinance fails

`VixFetchError` is raised; the scanner catches it and tags the trade
`vix_band='unknown'` with `vix_level=None`, plus a stderr log line.
Signal capture is never blocked.

### The `report by-regime-vix` view

This is the single most important Phase 2 analytical surface. It
answers: *which combinations of macro direction and volatility are
actually profitable for us?* Once enough live data accumulates,
Phase 4 (Strategy Evolution) will use this view to gate signals —
suppress combos with negative expectancy, lean into the winners.

Sort order is trade count descending so the most-populated buckets
surface first. Empty buckets are omitted entirely (no `0/0` noise).

## Prediction Engine (Phase 2.2b)

A parallel subsystem alongside the swing trade scanner. Predicts
**HIGHER / LOWER** for a 15-minute window on crypto tickers — designed
to feed signals for Robinhood's event-contract prediction markets
("will BTC be higher in 15 min?").

### What it does NOT do

- **No auto-betting.** Robinhood has no public API for event contracts.
  This is permanent — not a limitation we plan to remove. The bot
  notifies you; you place the bet manually on Robinhood.
- **No strike-price prediction.** Direction only — higher or lower
  than the entry price recorded at prediction time.
- **No swing-trade interference.** Predictions and trades are entirely
  separate: separate `predictions` table, separate scanner sweep,
  separate report (`report predictions`). No commingling.

### The six voting indicators

Each indicator votes +1 (HIGHER), -1 (LOWER), or 0 (NEUTRAL) on the
last 30 closed 15-min candles:

1. **Candle streak** — 3 consecutive green/red candles
2. **RSI(14) slope** — sign of (rsi_now - rsi_3_back)
3. **MACD histogram direction** — sign of (hist_now - hist_prev)
4. **Price vs VWAP** — above or below session VWAP
5. **Volume confirmation** — current vol > 20-MA, sided by candle color
6. **Bollinger position** — upper or lower half of BB(20, 2)

Confidence = winning_side / decided * 100. If confidence is within 5
points of 50% (i.e. < 55%), `predict_direction` returns `None` — the
engine **does not guess** when signals are mixed. Skipping is honest;
guessing poisons the accuracy stats.

**Calibration gates (hardening pass).** A prediction now fires only
when **at least 5 of the 6 indicators agree** AND the winning-side
confidence is **>= 70%** (`MIN_CONFIDENCE`). The 6 indicators are all
momentum-based and correlated, so a simple majority overstates real
edge. 15-minute crypto direction is inherently low-accuracy/noisy —
these gates favor *fewer, higher-conviction* predictions over volume.
The gates tighten WHEN it fires, not HOW it scores; the indicators and
their math are unchanged.

Each fired prediction is tagged with the current regime and VIX at
creation time, so later you can break accuracy down by macro context
(`report predictions --by-regime` / `--by-vix`).

### Off-switch architecture

Three independent gates, all checked at the top of each prediction sweep:

1. **Master switch** — `settings['predictions.enabled']`. Default
   `"false"`. Must be flipped on explicitly via the CLI.
2. **Active window** — `settings['predictions.window_start']` /
   `window_end`. Both `HH:MM` in ET. Default 08:00 - 22:00 ET. You
   don't get pinged at 3am.
3. **Pause** — `settings['predictions.pause_until']`. Convenience for
   "going to dinner, mute for 2 hours" — set via
   `predictions pause --minutes N`. Auto-cleared once expired.

A single skipped tick is silent; the scanner logs the skip reason
(`disabled` / `outside_window` / `paused`) so you can verify the
gate is doing what you think.

**Railway vs local control.** Settings live in SQLite, and the Railway
DB is separate from your local DB — so `predictions enable` on your
laptop does nothing to the deployed instance. Railway is controlled
via the **`PREDICTIONS_ENABLED` env var** (set in the Railway UI):
`true` forces the master switch on at startup, `false` forces it off,
and unset/any-other-value leaves whatever is in the DB. Local stays
CLI-controlled. The override is applied in `_seed_prediction_defaults`
on every boot.

### Resolution

Every 5 minutes, `resolve_due_predictions()` walks unresolved
predictions whose `target_window_end` has passed and fetches the
15-min candle that closed at that timestamp. Outcomes:

- `exit > entry` AND direction HIGHER → `correct`
- `exit < entry` AND direction LOWER → `correct`
- `exit == entry` → `push` (rare on crypto but possible)
- otherwise → `incorrect`

If the candle isn't published yet (yfinance lag, recent prediction),
the row stays `resolved_at IS NULL` and the next sweep retries.

### Notifications

Reuses the existing Pushover + Discord path. Plain ASCII (cp1252
safe). Resolution notifications are off by default — set
`predictions.notify_resolution` to `"true"` if you want them.

## Market Context (Phase 2.3)

Composite score 0-5 derived from regime x VIX. Replaces the awkward
"run regime current then run vix current" workflow with a single
unified dashboard via `context current`. Tagged on every trade and
every prediction at fire time alongside the underlying axes.

### Score matrix

```
              Low VIX    Elevated    High    Extreme
Bull            5           4          3        2
Sideways        4           3          2        1
Bear            2           2          1        1
```

Any `unknown` on either axis collapses the score to `0`. The matrix
is intentionally asymmetric:

- A bear market never scores above 2 regardless of volatility
- Extreme VIX docks every regime by at least one point
- Bull + low is the only `5`

### Label mapping

| Score | Label        |
|-------|--------------|
| 5     | ideal        |
| 4     | favorable    |
| 3     | neutral      |
| 2     | unfavorable  |
| 1     | hostile      |
| 0     | unknown      |

### Long-bias caveat

The score is LONG-BIAS. PUT/SHORT signals fired in a low-score
environment are technically benefiting from that context (a bear is
great for puts), but we do NOT invert the score at this layer. Phase 4
will handle signal-direction × context interaction. For Phase 2.3 the
score is pure metadata — something to filter on, sort by, threshold
against.

### `--min-context N` — the killer feature

Every analytical subcommand accepts `--min-context N`. Pass `4` and
the report only counts trades fired in `favorable` or `ideal`
conditions. This is the single most useful lens in the reporting CLI
and should become your default analytical view once enough data
accumulates. The filter is strict: trades with `context_score IS NULL`
are EXCLUDED when the filter is active (NULL is not "at least 1").

Supported on: `by-signal`, `by-ticker`, `by-regime`, `by-vix`,
`by-regime-vix`, `predictions` — including their matrix variants
(`by-signal --by-regime --min-context 4` etc.).

### Why this is the cleanest accuracy surface in the bot

Swing trades have ambiguous exits — TP, SL, hold-window expiry, gaps.
Predictions have a fixed 15-min window with a deterministic close
price. No ambiguity, no same-candle-TP+SL conservatism. The
`report predictions --by-time` view will eventually answer: *at what
hour of the day is the engine actually reliable?* That's a much
sharper question than the swing-trade win-rate can answer.

## Railway deployment notes

- `requirements.txt` is generated from `pyproject.toml` via uv —
  do not edit it by hand
- Existing env vars (Discord webhook, Pushover keys, NewsAPI key)
  remain set in Railway UI; the secrets layer in 1.1b will read them
  transparently
- TZ=America/New_York is set in Railway for market hours detection
- `PREDICTIONS_ENABLED` (optional) toggles the prediction engine on the
  deployed instance: `true` / `false` overrides the DB setting at boot.
  Unset leaves the DB value. This is the ONLY way to control predictions
  on Railway since its DB is separate from local. See "Prediction
  Engine -> Off-switch architecture".

## Deferred cleanup

- `backtest.py`, `backtest2.py` — legacy backtesters at the project root,
  still excluded from `ruff` and exempt from strict `mypy`. To be migrated
  into `trading_bot/` in a future cleanup phase (likely Phase 1.5+).
- `tradingbot.py` at the project root is now a 3-line shim that forwards
  to `trading_bot.scanner.main()`. The Railway `Procfile`
  (`worker: python tradingbot.py`) keeps working unchanged.
- `trading_bot/scanner.py` (migrated from `tradingbot.py` in Phase 1.2) is
  exempt from strict `mypy` and omitted from coverage. The indicator math,
  signal-detection rules, and live API I/O wrappers are unchanged from the
  legacy file and validated by 30 days of forward testing. The new wiring
  added in 1.2 (`log_signal`, `_load_secrets`, `main`, `_normalize_*`) is
  unit-tested in `tests/test_scanner.py`.
- The `[project.scripts]` entry point generated by `uv init` was removed
  because it referenced a `main` that does not exist yet. A real console
  entry point can be added later.

## Phase 1.2 — post-deploy follow-ups

- **Watchlist re-justification**: `STOCK_WATCHLIST` currently includes
  `TSLA` (explicitly dropped in earlier backtesting as underperforming)
  and `LLY` (never backtested by us). Either re-validate inclusion with
  fresh backtest data or remove. Unrelated to 1.2's scope; address before
  Phase 1.4 (performance attribution) so the metrics aren't skewed.
