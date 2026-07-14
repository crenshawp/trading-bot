# Signal Field Audit — Computed-but-Dropped Values (Phase 22)

**Scope.** A systematic sweep for one bug pattern, found twice by accident
(hold_estimate_days, fixed in Phase 21; atr, discovered during Phase 21):
a value is **(a)** computed in the signal-fire path, **(b)** has an intended
column or a downstream consumer reading it by name, and **(c)** never reaches
that column on the production INSERT path — always NULL/default in the DB
despite being computed correctly in memory.

This is NOT the full pre-deployment line-by-line audit (a later, broader
phase). It covers only this pattern, in the fire path:
`scanner.detect_stock_signals` / `scanner.detect_crypto_signals` →
`scanner.log_signal` → `db.insert_signal` + `db.insert_trade`, plus the
fire-time advisory bundles (sentiment / indicators / risk).

**Forward-only rule (Phase 21 discipline).** A FIX-NOW field must be a pure
missing pass-through whose persistence cannot change any EXISTING row's
behavior: no backfill ever; NULL keeps meaning exactly what it means today;
each fix carries its own old-row-unaffected regression test.

---

## Fire-path values → signals table

| Field | Computed at | In signal dict? | Reaches column? | Downstream consumer | Category |
|---|---|---|---|---|---|
| `price` → `entry_price` | `detect_stock_signals` / `detect_crypto_signals` locals | yes (`price`) | **YES** | resolver PnL, candidates, plans | **OK** |
| `take_profit` / `stop_loss` | tp/sl locals (ATR-derived), both detectors | yes | **YES** | resolver hit-detection, Phase 19/20 exits | **OK** |
| `hold_days` → `hold_estimate_days` | `estimate_hold_days` (stock) | yes (string) | **YES** (since Phase 21) | resolver settlement window, Phase 20 deadline | **OK** — fixed Phase 21 |
| `atr` | stock + crypto detector locals (the very ATR the signal's TP/SL derive from) | **NO** | **NO — always NULL** | `candidate_source` → `Candidate.atr` → Phase 7 sizing (`allocation._size_candidate`); NULL = "unsizeable", candidate filtered from every plan | **FIX-NOW** (mandated). Resolver never reads it — settlement uses tp/sl/hold only — so persisting cannot change any existing trade's resolution. Consequence today: **live SWING signals can never reach execution.** |
| `rsi` | stock + crypto detector locals | **NO** | **NO — always NULL** | `candidate_source` (fallback when the trade's `ind_rsi` is NULL, i.e. all crypto + shadow) → composite-confidence scoring | **FIX-NOW.** Pure pass-through; read only at candidate-pull time for NEW pulls; existing NULL rows keep today's fallback behavior. |
| `ema21` | stock detector local | **NO** | **NO — always NULL** | none (schema slot from the Phase 1.1c CSV mirror; no reader) | **FIX-NOW.** Zero-risk pass-through; persisted for record completeness (Phase 9 feature bucketing candidate). Stock only — crypto never computes it, stays NULL there. |
| `bb_upper` / `bb_lower` | crypto detector locals | **NO** | **NO — always NULL** | none (schema slots; no reader) | **FIX-NOW.** Same shape as `ema21`, crypto side. Stock never computes them, stays NULL there. |
| `macd` / `macd_signal` | **not computed in the swing fire path** (only the Phase 2.2b prediction engine computes MACD, persisting to `predictions`, its own table) | — | always NULL | none | **OK — not the pattern.** Criterion (a) fails: legacy CSV-mirror columns with no fire-path computation. If a MACD-based swing setup is ever added, wire it then. |
| `earnings_risk` (string → bool column) | `check_earnings_risk` (stock) | only as warning-entry `detail` | **NO — always 0** | `candidate_source` → `Candidate.earnings_blackout` → allocation eligibility gate | **FLAG-FOR-LATER.** Not a pure pass-through: the computed value is a STRING ("HIGH —…"/"LOW"), the column a boolean, and HIGH already suppresses the ticker's signals entirely at fire (warning entry, no trade signal) — so `False` is arguably correct for every signal that fires. Whether MEDIUM/other grades should set the flag (and thereby activate a live allocation gate) is a design decision, not a plumbing fix. |
| `news_risk` (string → bool column) | `check_news_risk` (stock) | only as warning-entry `detail` | **NO — always 0** | none (column only) | **FLAG-FOR-LATER.** Same string→bool semantic question as `earnings_risk` (HIGH suppresses at fire; MEDIUM fires without the flag). No consumer today, so no behavioral urgency. |
| `raw_indicators_json` | full Phase 6 indicator bundle exists at fire | — | **NO on log_signal** (written by `migrate_csv` with CSV extras and by `long_term.persist_candidates` with the entry rationale — two different payload shapes already) | none | **FLAG-FOR-LATER.** Needs a payload-schema decision before a third writer piles in; no reader exists. |
| `detail` / `confidence` / `volume` / `vol_ma` / `ema50` / `slope` / `roc_accel` / `slope_accel` / `high_20` / `recent_high` / `recent_low` / `atr_ma` / `prev_rsi` / `prev_atr` | detector locals / dict fields | some | no column exists | none reads them by name | **OK — not the pattern.** Criterion (b) fails: no intended column, no consumer. (`trades.notes` is generic free text, not an intended slot for `detail`.) |

## Fire-time values → trades table

| Field group | Computed at | Reaches column? | Category |
|---|---|---|---|
| `market_regime`, `vix_level`, `vix_band`, `context_score` | `log_signal` (Phase 2.1–2.3) | **YES** (fail-soft to 'unknown') | **OK** |
| `sentiment_*` (Phase 5) | scan loop bundle → `log_signal(sentiment=…)` | **YES** for active stock signals; **deliberately None** for shadow/crypto (documented design — the bundle is not computed for those scans, not computed-then-dropped) | **OK** |
| `ind_*` (Phase 6) | scan loop bundle → `log_signal(indicators=…)` | **YES** for active stock; deliberately None for shadow/crypto (same) | **OK** |
| `risk_*` (Phase 7) | scan loop bundle → `log_signal(risk=…)` | **YES** for active stock; deliberately None for shadow/crypto (same) | **OK** |

## Execution-side tables (Phases 13/19/20 wiring)

`option_positions` (tp/sl/deadline/greeks/order_id) and `long_term_positions`
(source/direction/tp/sl/deadline) receive every value their writers compute —
verified in Phases 19–20 test suites. No instance of the pattern found.

---

## Summary

- **FIX-NOW (this phase):** `atr` (the mandated case — currently blocks every
  live SWING signal from execution), `rsi`, `ema21`, `bb_upper`+`bb_lower`.
  All are pure pass-throughs read (if at all) only at candidate-pull time for
  new rows; the resolver reads none of them, so no existing trade's outcome
  can change. Each gets its own forward-only regression test and commit.
- **FLAG-FOR-LATER (reported, deliberately NOT fixed here):**
  1. `earnings_risk` — string-grade → boolean mapping needs a design decision,
     and its consumer is a live allocation gate;
  2. `news_risk` — same semantic question, no consumer yet;
  3. `raw_indicators_json` — needs one agreed payload schema (two writers
     already use it for different shapes).
- **No backfill of any existing row, for any field, ever.** NULL keeps meaning
  today's default/fallback everywhere.
