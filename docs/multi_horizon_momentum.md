# Multi-Horizon Momentum

`multi_horizon_momentum` is the TradingLab-style weekly trend strategy. It is a
normal active stock signal: default-enabled pairs alert, open tracked trades,
feed the SWING allocation/execution path, resolve into outcomes, and appear in
the existing performance/readiness reports.

## Signal

The scanner uses only completed daily candles and compares the latest completed
close with 5, 10, 21, and 42 trading sessions earlier. Each positive return is
`+1`, each negative return is `-1`, and an exactly unchanged return is `0`.
Votes are summed:

| Score | Direction | Normal-size fraction |
|---:|---|---:|
| `+4` | call/long | 100% |
| `+2` | call/long | 50% |
| `0` | flat; no signal | 0% |
| `-2` | put/short | 50% |
| `-4` | put/short | 100% |

An exact unchanged horizon can produce an odd score; its size remains
`abs(score) / 4` instead of inventing a directional vote.

The first successful evaluation per ticker in each New York strategy week is
stored in `multi_horizon_evaluations`. The job checks every stock session at
09:32 ET, so market holidays and transient per-ticker data failures retry on the
next session without evaluating successful tickers twice. Flat scores are also
stored to preserve the weekly cadence.

## Risk and lifecycle

The video's volatility statistic—the mean absolute daily percentage move over
30 sessions, annualized with `sqrt(252)` for stocks—is stored as audit metadata.
The bot's existing audited SWING risk authority remains in force:

- ATR-normalized sizing and portfolio/exposure caps;
- score-based 50%/100% scaling applied to advisory and executable size;
- take profit at 2 ATR and stop loss at 1.5 ATR;
- seven-calendar-day hold deadline;
- earnings blackout, watchlist state, and per-pair gating;
- paper-order allocation, execution, reconciliation, and exit watchers.

The ATR target/stop and seven-day deadline are explicit system adaptations. The
source video did not provide a deterministic exit, and its discretionary example
cannot produce reproducible outcomes without one.

The strategy is stock-only because the current executable crypto path is
long-term buy-only; routing short crypto momentum signals through it would invert
their intent. Existing crypto swing signals remain data-only as designed.
