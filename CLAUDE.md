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

### Phase 2 sub-phases queued:

- [ ] Phase 2.3 — Regime+VIX consolidation: weighted scoring,
      regime+vix filters, confidence flags emitted at signal fire,
      News-aware weighting layered in
- [ ] Phase 2.4 — Watchlist auto-discovery

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

## Railway deployment notes

- `requirements.txt` is generated from `pyproject.toml` via uv —
  do not edit it by hand
- Existing env vars (Discord webhook, Pushover keys, NewsAPI key)
  remain set in Railway UI; the secrets layer in 1.1b will read them
  transparently
- TZ=America/New_York is set in Railway for market hours detection

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
