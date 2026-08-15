# Layer-3 Rolling Exit on Layer-2.5 Trades — 2026-04-22

## Verdict

**Promising, and methodologically much cleaner than the old carry-over Layer-3.**

Using the promoted Layer-2.5 patience gate at threshold `0.40`
(`218` trades across `13` rolling windows), an honest rolling Layer-3
exit model trained only on prior windows lifts the same trade set from:

- **Layer-2.5 + time-stop:** `PF 1.795`, `DD 21.4%`, mean `+$262/trade`

to an exploratory best threshold of:

- **Layer-2.5 + rolling Layer-3 @ 0.19:** `PF 2.351`, `DD 21.4%`, mean `+$334/trade`

This is a real architectural improvement because the exit model is now
evaluated on the same rolling-window world as the current champion,
without borrowing from the older 5-fold Layer-3 universe.

## Important caveat

The threshold comparison is still **exploratory**.

The best threshold in this report (`0.19`) is chosen from the same
rolling OOS windows used to evaluate it, so it is **not** yet a
deployment-calibrated threshold. Treat this as a research clue, not a
production setting.

## Setup

- Layer-2.5 source:
  [v3/artifacts/layer25_entry_patience_surface/entry_patience_surface.json](../artifacts/layer25_entry_patience_surface/entry_patience_surface.json)
- Layer-3 script:
  [v3/layer3/train_rolling.py](/Users/gduby/Documents/autoresearch-trading/v3/layer3/train_rolling.py)
- Layer-3 artifact:
  [v3/artifacts/layer3_rolling_entry_patience/rolling_layer3_report.json](../artifacts/layer3_rolling_entry_patience/rolling_layer3_report.json)

## Threshold sweep

| Exit threshold | PF | DD% | Mean $/trade | Mean bars held |
|---:|---:|---:|---:|---:|
| time-stop baseline | 1.795 | 21.4 | +262 | 212.3 |
| 0.15 | 2.345 | 21.4 | +304 | 132.6 |
| **0.19** | **2.351** | **21.4** | **+334** | **153.0** |
| 0.20 | 2.251 | 21.4 | +327 | 158.5 |
| 0.25 | 2.135 | 21.4 | +320 | 173.0 |
| 0.30 | 1.980 | 21.4 | +296 | 192.1 |

Interpretation:

- The exit layer still adds value after we make the entry set stricter.
- The promising band is broader than a single knife-edge point, which
  is better than the earlier “one lucky threshold” pattern.
- Higher thresholds hold longer and give back some of the lift.

## Training coverage

Because this is a true rolling setup, early windows have little or no
prior trade history:

- window `0`: fallback only
- window `1`: fallback only (`19` prior trades, below the `40`-trade minimum)
- window `2+`: learned Layer-3 active

By the later windows the exit model is training on `20k` to `42k`
post-entry bar rows, which is much healthier than the older
chosen-trade-only bottleneck.

## What this means

This is the first version of the stack where the current honest flow is
fully composable:

1. Layer 2 scores bars
2. Layer 2.5 filters impatient / greedy entries
3. Layer 3 learns exits on the cleaner rolling trade set

That does **not** mean the system is deploy-ready. The next honest
questions are:

1. Can we calibrate Layer-3 thresholds without peeking at the same OOS
   windows?
2. Does this Layer-3 lift survive slippage stress and other execution
   realism checks?
3. Can Layer-2.5 labels be improved using trader-defined “stopped out
   before the real move” logic instead of the current MAE/time-stop
   proxy?
