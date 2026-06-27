"""Offline demo of the Phase 9 self-optimization layer.

Run with: ``python demo_optimize.py``

Pure synthetic data, no network. It shows:

* degradation that FIRES (a meaningful expectancy drop on a sample that clears
  the floor) vs a cold streak that does NOT fire (a real drop, but too few
  trades to be actionable);
* feature evaluation that is SUGGESTIVE but explicitly NOT actionable, because
  one bucket sits below the sample floor (the multiple-comparisons discipline);
* persisting a run and replaying it through the report renderer.

FLAGS ONLY — nothing here changes a threshold or any pair/ticker status. Acting
on a finding is a deliberate, later, sample-gated step.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from trading_bot import config

config.DB_PATH = Path(tempfile.mkdtemp()) / "demo_optimize.db"

from trading_bot import db  # noqa: E402
from trading_bot import self_optimization as so  # noqa: E402


def _row(outcome: str, pnl: float, **ctx: object) -> dict[str, object]:
    base: dict[str, object] = {
        "outcome": outcome, "pnl_pct": pnl,
        "signal_type": "ema21_pullback", "ticker": "GOOGL",
    }
    base.update(ctx)
    return base


def _mixed(n_win: int, n_loss: int, **ctx: object) -> list[dict[str, object]]:
    """n_win winners (+2.0%) and n_loss losers (-1.0%), with optional context."""
    return (
        [_row("win", 2.0, **ctx) for _ in range(n_win)]
        + [_row("loss", -1.0, **ctx) for _ in range(n_loss)]
    )


def main() -> None:
    db.init_db()

    print("=== Degradation: fires only with sample AND a meaningful drop ===")
    baseline = _mixed(25, 10)     # 35 trades, expectancy ~ +1.14%
    recent_bad = _mixed(12, 23)   # 35 trades, expectancy ~ +0.03% (a real drop)
    fired = so.compute_degradation(baseline, recent_bad)
    overall = next(f for f in fired if f.scope == "overall")
    print(f"  drop, n>=floor : {overall.verdict.upper()} - {overall.note}")

    recent_cold = _mixed(6, 9)    # only 15 trades — a cold streak, not a signal
    cold = so.compute_degradation(baseline, recent_cold)
    overall_cold = next(f for f in cold if f.scope == "overall")
    print(f"  drop, n<floor  : {overall_cold.verdict.upper()} - {overall_cold.note}")

    print("\n=== Feature evaluation: suggestive but NOT actionable ===")
    sentiment_rows = (
        _mixed(8, 14, sentiment_label="bearish")    # 22 trades, ~36% win
        + _mixed(21, 19, sentiment_label="bullish")  # 40 trades, ~53% win
    )
    feats = so.compute_feature_evaluations(sentiment_rows)
    sent = next(f for f in feats if f.feature == "sentiment")
    print(f"  sentiment [{'actionable' if sent.actionable else 'not actionable'}]: "
          f"{sent.note}")
    for b in sent.buckets:
        flag = "" if b.actionable else "  (n<floor)"
        print(f"    {b.label:<9} win_rate={b.win_rate:.0f}%  "
              f"expectancy={b.expectancy:+.2f}  n={b.n}{flag}")

    print("\n=== Persist a run and replay it via the report ===")
    payload = so.build_payload(
        fired, feats, run_timestamp="2026-07-01T09:00:00",
        degrade_window_days=config.SO_DEGRADE_WINDOW_DAYS,
        baseline_window_days=config.SO_BASELINE_WINDOW_DAYS,
    )
    so.persist_run(payload)
    latest = db.get_latest_optimization_run()
    assert latest is not None
    import json
    print(so.render_report(json.loads(latest["findings_json"])))


if __name__ == "__main__":
    main()
