# Trading Bot — Project Notes

## Overview

Python algorithmic trading alert bot. Scans Tier 1 stock watchlist
(BLK, GOOGL, META, GS, NOW, AMZN, LLY, TSLA) once daily at 9:31am EST
and crypto watchlist (BTC-USD, BNB-USD, ETH-USD) hourly. Signals fire
via Discord and Pushover. Deployed on Railway.

Primary live signal: EMA21 Pullback (stocks). Crypto: Oversold Reversal
and Momentum Breakout.

---

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
- [x] Phase 3 — Watchlist auto-discovery + live-shadow promotion + state machine
      (3.1 discovery scan, 3.1-LIVE shadow promotion, 3.3 active/benched FSM)
- [x] Phase 4 — Per-(ticker, signal_type) pair gating (mute/enable, permissive-by-default)
- [x] Phase 5 — Advisory LLM sentiment context (stored beside outcomes; never a gate)
- [x] Phase 6 — Advanced math: five independent indicator families (ATR/vol, RSI,
      ADX, OBV, cross-asset correlation) + IndicatorContext stored beside outcomes
- [x] Phase 7 — Advisory risk management: volatility-normalised position sizing +
      portfolio-risk verdicts against a NOTIONAL account; stored beside outcomes
- [ ] Phase 8 — Web dashboard (SKIPPED — see note below)
- [x] Phase 9 — Self-optimisation: degradation detection (recent vs baseline
      expectancy) + feature evaluation (bucket stored context against outcomes).
      FLAGS AND REPORTS ONLY — never auto-tunes, never changes thresholds.
- [x] Phase 10 — Unified readiness gate: ONE authority over every capability's
      data-sufficiency. Deterministic capabilities auto-activate on first threshold
      crossing; ML capabilities flag ready-to-build and summon a human — never
      self-train or self-deploy.

**Phase 8 note:** The web dashboard was intentionally skipped. Robinhood has no
public API for event contracts; the prediction engine notifies you and you bet
manually. Phase 8 was deferred until/unless that changes. ML capabilities
(pattern recognition, predictive sizing) are also deferred — they have NO model
code; they only reach `ready_to_build` and fire a Pushover summons.

---

## Workflow rules for every future Claude Code session

**These must not be changed without explicit operator instruction.**

1. **Work on `main` only.** Never create branches, worktrees, or PRs.
2. **Audit gate before every commit** — in PowerShell on Windows use semicolons,
   NOT `&&` (which is a syntax error in PowerShell 5):
   ```
   uv run pytest ; uv run ruff check . ; uv run mypy trading_bot
   ```
3. **Coverage floor is 91%.** `pyproject.toml` enforces `--cov-fail-under=91`.
   Raise the floor if new code pushes coverage above it; never lower it.
4. **All external calls must fail-soft.** Catch, log the reason to stderr, and
   degrade gracefully. Never let a yfinance / Discord / Pushover / NewsAPI /
   Anthropic failure block signal capture or crash the scan loop.
5. **Never bypass the audit gates.** Do not use broad `# type: ignore` or
   `# noqa` to silence a gate. If a suppression is genuinely unavoidable,
   explain why in a comment on that line.
6. **Single-prompt phases push after the final audit:**
   ```
   git push origin main
   ```
7. **OneDrive dist-info artifact is benign.** `Access is denied … dist-info`
   errors during `uv` rebuilds are a known Windows/OneDrive file-lock artifact.
   They self-recover; retry the command once. Do not investigate further.

---

## Architecture: self-managing loop

```
tradingbot.py  (3-line Railway shim)
   └── trading_bot/scanner.py  main()
         ├── 09:31 EST daily    scan_stocks()     EMA21 Pullback
         ├── hourly             scan_crypto()     Oversold Reversal, Momentum Breakout
         ├── hourly             resolve_all_open_trades()
         ├── hourly             resolve_due_predictions()   (every 5 min inside loop)
         ├── hourly             evaluate_readiness()        (Phase 10)
         ├── 09:35 EST daily    upsert_regime_snapshot()
         ├── 09:36 EST daily    upsert_vix_snapshot()
         └── 00:30 UTC daily    update_daily_performance()
```

**Gating hierarchy (outer → inner):**

1. **Readiness gate** (`trading_bot/readiness.py`) — capability dormant if
   resolved-trade count < threshold. Evaluators check `is_ready(cap)` and use
   `effective_dry_run = dry_run or not is_ready(cap)`.
2. **Watchlist state machine** (`watchlist_state.py`) — tickers active or benched
   based on windowed expectancy (Phase 3.3). Active tickers only get signals.
3. **Per-pair gate** (`signal_pairs.py`) — (ticker, signal_type) pairs muted or
   enabled based on windowed expectancy (Phase 4). Muted pairs fire as shadow.
4. **Context score** (Phase 2.3) — pure metadata tag 0-5. Stored on every trade
   and prediction; never suppresses a signal at fire time (Phase 4 can use it
   later, but that is not yet wired).

**Never-dark guarantee:** regime, VIX, sentiment, indicator, and risk failures
all tag the trade with `'unknown'` / `None` / neutral. The signal fires; it is
logged. Signal capture is never blocked by an external-data failure.

---

## Key design decisions — do not re-litigate

- **`iloc[-2]` not `iloc[-1]`**: The scanner uses the second-to-last candle for
  signal detection. The last candle is still forming (live); its OHLC is
  incomplete. Changing this would compare the signal against a partial bar.

- **Same-candle TP+SL is conservatively marked `loss`**: OHLC cannot reveal
  intra-candle order. We assume SL hit first to keep win rates honest.

- **Permissive-by-default pair gating**: A (ticker, signal_type) pair with no
  row in `signal_pair_status` is treated as `enabled`. A row only appears after
  the evaluator first acts on it. Never require an explicit opt-in to fire.

- **Advisory context stored beside outcomes**: Phase 5/6/7 columns
  (`sentiment_*`, `ind_*`, `risk_*`) are stored at fire time so Phase 9 can
  evaluate them against resolved outcomes later. They never gate signals.

- **ML capabilities are pure flags**: No model code, no training, no inference.
  `ml_pattern_recognition` (threshold 500) and `ml_predictive_sizing` (threshold
  750) only reach `ready_to_build` and send a Pushover build summons. A human
  builds and validates the model. Nothing is automatic.

- **`pnl_dollars` is always `None`**: Position sizing (Phase 7) is ADVISORY
  against a notional account (`NOTIONAL_ACCOUNT = $10 000`). No capital is at
  risk. Dollar PnL tracking is deferred until real money is traded.

- **Earnings blackout uses the hold window, not resolver expiry**: When a ticker
  has an upcoming earnings event, the hold window naturally protects it.
  `earnings_risk` is a flag on the signal; the resolver checks TP/SL within the
  window as normal.

---

## Schema — version 16 (15 tables)

| Table | Phase | Purpose |
|---|---|---|
| `signals` | 1.1c | Every fired alert: ticker, type, direction, price, indicators |
| `trades` | 1.3 | One per signal; carries outcome + all advisory context columns |
| `daily_performance` | 1.4 | Daily aggregate snapshot (seeded at 00:30 UTC) |
| `regime_snapshots` | 2.1 | One row per trading date; bull/bear/sideways from SPY |
| `vix_snapshots` | 2.2 | One row per trading date; level + band (sparse Sat/Sun) |
| `predictions` | 2.2b | 15-min HIGHER/LOWER predictions on crypto tickers |
| `settings` | 2.2b | Key/value pairs (prediction switch, window, tickers, pause) |
| `active_watchlist` | 3.1 | DB-driven stock watchlist; seeded from hardcoded list |
| `discovery_results` | 3.1 | Per-ticker scores from each discovery scan |
| `shadow_evaluations` | 3.1-LIVE | Audit trail of live-shadow promotion evaluations |
| `watchlist_transitions` | 3.3 | Audit trail of active↔benched status changes |
| `signal_pair_status` | 4 | Per-(ticker, signal_type) gate: enabled or muted |
| `signal_pair_transitions` | 4 | Audit trail of enabled↔muted pair status changes |
| `optimization_runs` | 9 | History of self-optimisation runs (full payload JSON) |
| `readiness_state` | 10 | Capability ledger: status, n_at_crossing, announced flag |

The `trades` table carries all advisory context alongside each outcome:

```
-- Phase 2.1:  market_regime
-- Phase 2.2:  vix_level, vix_band
-- Phase 2.3:  context_score
-- Phase 3.1:  track_mode (active | shadow)
-- Phase 5:    sentiment_score, sentiment_label, heavy_news, headline_count
-- Phase 6:    ind_atr, ind_realized_vol, ind_vol_regime, ind_rsi, ind_adx,
--             ind_obv, ind_correlation, ind_concentration
-- Phase 7:    risk_recommended_size, risk_stop_distance, risk_dollar_risk,
--             risk_pct, risk_position_pct, risk_capped, risk_total_pct,
--             risk_portfolio_verdict, risk_position_verdict, risk_cluster_pct,
--             risk_cluster_verdict
```

---

## Capability registry (Phase 10)

Single source of truth in `trading_bot/readiness.py`. Thresholds are the
existing constants, centralised unchanged.

| Capability | Kind | Threshold | Scope | Behaviour at crossing |
|---|---|---|---|---|
| `watchlist_rotation` | deterministic | 10 (`SM_MIN_CLOSED_SIGNALS`) | all | auto-activates |
| `pair_gating` | deterministic | 10 (`SP_MIN_CLOSED_SIGNALS`) | all | auto-activates |
| `shadow_promotion` | deterministic | 10 (`MIN_SHADOW_SIGNALS`) | shadow | auto-activates |
| `self_optimization` | deterministic | 30 (`SO_MIN_SAMPLE`) | active | auto-activates |
| `ml_pattern_recognition` | ml | 500 (`ML_PATTERN_MIN_SAMPLE`) | all | summons a build |
| `ml_predictive_sizing` | ml | 750 (`ML_SIZING_MIN_SAMPLE`) | all | summons a build |

**Scope definitions:**
- `all` — every resolved (win/loss) trade regardless of track_mode
- `active` — resolved active-track trades only
- `shadow` — resolved shadow-track trades only

**One-time edge event:** the `announced` flag in `readiness_state` is set `True`
only on a confirmed Pushover send. A failed send leaves it `False` and the next
evaluation retries. Once `True` it never re-fires.

---

## CLI reference

```
python -m trading_bot <command>
```

### Secrets

```
python -m trading_bot secrets list
python -m trading_bot secrets set <NAME>          # prompts for value
python -m trading_bot secrets get <NAME>          # local only, not on Railway
python -m trading_bot secrets delete <NAME>
```

### Database

```
python -m trading_bot db init                     # create / migrate schema
python -m trading_bot db status                   # schema version + table counts
python -m trading_bot migrate csv                 # one-shot: import historical CSV
```

### Outcomes

```
python -m trading_bot outcomes resolve            # resolve open trades now
python -m trading_bot outcomes backfill           # one-shot: seed trades for CSV signals
python -m trading_bot outcomes status             # win rates by signal type (active)
python -m trading_bot outcomes status --shadow    # same for shadow-tracked trades
```

### Reporting

```
python -m trading_bot report overall              # single-block summary (active)
python -m trading_bot report overall --shadow     # shadow book
python -m trading_bot report recent --days 30     # last N days
python -m trading_bot report by-signal            # per signal_type
python -m trading_bot report by-signal --by-regime
python -m trading_bot report by-signal --by-vix
python -m trading_bot report by-ticker [--stocks | --crypto]
python -m trading_bot report by-ticker --by-regime
python -m trading_bot report by-ticker --by-vix
python -m trading_bot report by-asset             # stock vs crypto
python -m trading_bot report cross                # signal_type x ticker (>=3 trades)
python -m trading_bot report by-regime            # per macro regime
python -m trading_bot report by-vix               # per VIX band
python -m trading_bot report by-regime-vix        # regime x vix (headline Phase 2.2 view)
python -m trading_bot report by-context           # per context score 0-5
python -m trading_bot report context-matrix       # 3x4 regime x vix geometric matrix
python -m trading_bot report pairs                # per-(ticker, signal_type) stats + gate
python -m trading_bot report predictions          # prediction accuracy
python -m trading_bot report predictions --by-regime
python -m trading_bot report predictions --by-vix
python -m trading_bot report predictions --by-time  # by hour-of-day ET
python -m trading_bot report daily-backfill       # one-shot daily_performance seed
python -m trading_bot report daily [--date YYYY-MM-DD]

# --min-context N: filter any of the above to trades with context_score >= N.
# 0 = no filter; >0 excludes NULL context_score trades. Supported on:
# by-signal, by-ticker, by-regime, by-vix, by-regime-vix, predictions (+ matrix variants)
```

### Regime, VIX, Context

```
python -m trading_bot regime current
python -m trading_bot regime history [--days 30]
python -m trading_bot regime backfill             # one-shot: tag historical trades

python -m trading_bot vix current
python -m trading_bot vix history [--days 30]
python -m trading_bot vix backfill                # one-shot: tag historical trades

python -m trading_bot context current             # composite score 0-5 (use this first)
python -m trading_bot context history [--days 30]
python -m trading_bot context backfill            # one-shot: tag context_score on legacy rows
```

### Discovery, Shadow, Watchlist, Pairs

```
python -m trading_bot discovery scan [--throttle 1.0]   # backtests universe, promotes nothing

python -m trading_bot shadow status               # shows candidates vs promotion bar
python -m trading_bot shadow evaluate             # promote eligible shadow candidates
python -m trading_bot shadow evaluate --dry-run   # preview without promoting

python -m trading_bot watchlist status            # active vs benched with windowed stats
python -m trading_bot watchlist evaluate          # run active<->benched state machine
python -m trading_bot watchlist evaluate --dry-run

python -m trading_bot pairs status                # enabled vs muted pairs
python -m trading_bot pairs evaluate              # run enabled<->muted state machine
python -m trading_bot pairs evaluate --dry-run
```

### Sentiment, Indicators, Risk (Phase 5/6/7)

```
python -m trading_bot sentiment status [--limit 20]
python -m trading_bot indicators status [--limit 20]
python -m trading_bot risk status [--limit 20]
python -m trading_bot risk exposure               # open-trade risk vs advisory limits
```

### Self-Optimisation (Phase 9)

```
python -m trading_bot optimize run [--degrade-window 30] [--baseline-window 90]
python -m trading_bot optimize report             # replay the last persisted run
```

### Readiness (Phase 10)

```
python -m trading_bot readiness status            # all capabilities: n / threshold / active
python -m trading_bot readiness check             # run evaluate_readiness() + print status
```

### Predictions (Phase 2.2b)

```
python -m trading_bot predictions enable
python -m trading_bot predictions disable
python -m trading_bot predictions status          # settings + last-24h activity
python -m trading_bot predictions window --start HH:MM --end HH:MM
python -m trading_bot predictions tickers --set BTC-USD,ETH-USD
python -m trading_bot predictions pause --minutes 60
```

---

## Aggregation rules

- **"Closed" means** `outcome IN ('win', 'loss', 'expired')`. Trades with
  `outcome IS NULL` or `outcome = 'open'` are excluded from every stat.
- **Win rate = wins / (wins + losses).** Expired trades do **not** count
  toward the denominator. All-expired slices report `win_rate = None`
  (the CLI prints `-`), not `0%`.
- **Avg / best / worst PnL include expired trades.** An expired trade with
  no available exit (`pnl_pct IS NULL`) is silently ignored by AVG/MAX/MIN
  but still counts toward `total` and the `expired` bucket.
- **Empty slices return numeric `None`**, never 0.
- **Readiness resolved count = win + loss only** (expired and open excluded).
  Readiness measures accumulated decided data, not trade volume.

---

## Macro regime, VIX, and context score

**Regime (SPY-based):**
- **bull** — SPY > 200 EMA AND 50 EMA > 200 EMA AND 5-day slope of 50 EMA > 0
- **bear** — SPY < 200 EMA AND 50 EMA < 200 EMA AND 5-day slope < 0
- **sideways** — anything else
- **unknown** — yfinance unreachable; signal still fires

**VIX bands:**
- **low** — `VIX < 20`
- **elevated** — `20 <= VIX < 30`
- **high** — `30 <= VIX < 40`
- **extreme** — `VIX >= 40`
- Boundary goes to the higher band (VIX 20.0 = elevated)

**Context score matrix (0-5, long-bias):**

```
              Low VIX    Elevated    High    Extreme
Bull            5           4          3        2
Sideways        4           3          2        1
Bear            2           2          1        1
```

Any `unknown` axis → score = 0. A bear market never exceeds 2.

| Score | Label |
|-------|-------|
| 5 | ideal |
| 4 | favorable |
| 3 | neutral |
| 2 | unfavorable |
| 1 | hostile |
| 0 | unknown |

---

## Prediction engine (Phase 2.2b)

Six voting indicators on the last 30 closed 15-min candles:
1. Candle streak (3 consecutive green/red)
2. RSI(14) slope
3. MACD histogram direction
4. Price vs VWAP
5. Volume confirmation (sided by candle color)
6. Bollinger position (upper/lower half of BB(20,2))

Fires only when **≥5 of 6 indicators agree** AND confidence **≥70%**.

Three off-switches: master switch (`predictions.enabled`), active window
(`08:00–22:00 ET` default), and temporary pause (`predictions.pause_until`).

**No auto-betting.** Robinhood has no public API for event contracts. The
bot notifies; you place the bet manually. This is permanent.

---

## Self-optimisation (Phase 9)

FLAGS AND REPORTS ONLY. Nothing here changes thresholds, statuses, or
gate decisions.

- **Degradation**: compares expectancy in a recent window vs an older baseline
  using `signal_pairs._windowed_stats`. Requires n≥`SO_MIN_SAMPLE` (30) AND
  drop≥`SO_MEANINGFUL_DELTA` (0.15) to flag. `_DELTA_EPSILON = 1e-9` guards
  float boundary.
- **Feature evaluation**: buckets 7 stored context fields against resolved
  outcomes. Actionable only if ≥2 buckets in a feature clear `SO_MIN_SAMPLE`.
- Every run is persisted to `optimization_runs` and replayed by `optimize report`.

---

## Risk management (Phase 7) — advisory only

```
Notional account:     $10 000 (NOTIONAL_ACCOUNT)
Risk per trade:       1.0 % (RISK_PER_TRADE_PCT)
Max portfolio risk:   6.0 % (MAX_PORTFOLIO_RISK_PCT)
Max position:        20.0 % (MAX_POSITION_PCT)
Max cluster risk:    25.0 % (MAX_CORRELATED_CLUSTER_PCT)
ATR stop multiple:    1.5 × (RISK_ATR_STOP_MULTIPLE)
```

`dollar_risk = account × risk_per_trade_pct / 100`.
`shares = dollar_risk / (ATR × stop_multiple)`, capped at MAX_POSITION_PCT.

No capital is at risk. Advisory only. `pnl_dollars` is always None.

---

## Canonical secrets

Set via `python -m trading_bot secrets set <NAME>` locally; in Railway set
in the Railway UI as environment variables.

| Name | Required | Purpose |
|---|---|---|
| `DISCORD_WEBHOOK_URL` | yes | Signal notifications |
| `PUSHOVER_USER_KEY` | yes | Pushover push notifications |
| `PUSHOVER_APP_TOKEN` | yes | Pushover push notifications |
| `NEWSAPI_KEY` | yes | Phase 5 headline fetch |
| `ANTHROPIC_API_KEY` | optional | Phase 5 LLM sentiment scoring; fails soft to neutral when unset |

---

## Railway deployment

- `requirements.txt` is generated from `pyproject.toml` via uv — do not edit
  it by hand.
- `Procfile`: `worker: python tradingbot.py` (3-line shim → `scanner.main()`).
- `TZ=America/New_York` is set in Railway for market-hours detection.
- `PREDICTIONS_ENABLED` (optional): `true` / `false` overrides the
  `predictions.enabled` DB setting at boot. Unset leaves the DB value.
  This is the ONLY way to control predictions on Railway (its DB is separate
  from local).
- Railway DB is separate from local DB. `predictions enable` on your laptop
  does nothing to the deployed instance.

---

## Package layout

```
trading_bot/          new code (Phase 1.1+ migration target)
  __main__.py         CLI dispatcher (python -m trading_bot)
  config.py           all constants and thresholds
  db.py               SQLite layer, schema v16, all queries
  models.py           Signal, Trade, Prediction, DailyPerf dataclasses
  scanner.py          live scan loop (migrated from tradingbot.py in 1.2)
  outcomes.py         win/loss/expired resolver
  performance.py      PerfStats aggregation, all report queries
  regime.py           SPY regime detection + cache + backfill
  vix.py              VIX context + cache + backfill
  context.py          composite context score (regime x VIX)
  predictions.py      15-min direction prediction engine
  settings.py         generic key/value settings layer
  secrets.py          unified secret access (keychain + .env + Railway env)
  discovery.py        universe backtest scan (informational)
  discovery_universe.py  SHADOW_UNIVERSE constant
  shadow_discovery.py live-shadow promotion evaluator
  watchlist_state.py  active<->benched state machine
  signal_pairs.py     per-(ticker, signal_type) mute/enable gate
  indicators.py       Phase 6: five indicator families + IndicatorContext
  risk.py             Phase 7: advisory position sizing + portfolio risk
  self_optimization.py Phase 9: degradation detection + feature evaluation
  readiness.py        Phase 10: unified capability registry + readiness gate
  migrate_csv.py      one-shot CSV importer

tradingbot.py         Railway shim (3 lines)
tests/                pytest suite, mirrors trading_bot/ structure
backtest.py           legacy, excluded from ruff/mypy
backtest2.py          legacy, excluded from ruff/mypy
```

---

## Deferred cleanup

- `backtest.py`, `backtest2.py` — legacy backtester at project root; excluded
  from `ruff` and exempt from strict `mypy`. Migrate to `trading_bot/` later.
- `trading_bot/scanner.py` is exempt from strict `mypy` and omitted from
  coverage. Indicator math and signal-detection rules are validated by forward
  testing; only the wiring added in 1.2+ is unit-tested.

---

## What comes next

**The bot runs itself.** All eight deterministic/ML capabilities are currently
`warming`. Once resolved-trade counts cross each threshold, `evaluate_readiness`
(runs hourly) fires a one-time Pushover notification:

- **Deterministic** (watchlist_rotation, pair_gating, shadow_promotion,
  self_optimization): "switched on automatically" — no human action needed.
- **ML** (ml_pattern_recognition at n=500, ml_predictive_sizing at n=750):
  "ready to build — awaiting the build" — a human must build the model.

Until those Pushover notifications arrive, the loop is self-managing. The next
human decision point is the ML build summons, or an explicit Phase 8 (web
dashboard) implementation request.
