# Simulated-L3 Iteration-2 Window-6 Diagnostic — 2026-04-23

## Question

Why did the iteration-2 simulated-L3 branch improve mean PF while giving
back the PF floor, and why did the simple `+0.05` decision-margin rescue
make seed `42` worse?

This note localizes the seed-42 failure named in the handoff.

## New tooling

Added:

- [v3/analysis/sim_l3_window_divergence.py](/Users/gduby/Documents/autoresearch-trading/v3/analysis/sim_l3_window_divergence.py)

Default command:

```bash
.venv/bin/python -m v3.analysis.sim_l3_window_divergence
```

Outputs:

- [v3/artifacts/analysis/sim_l3_iter2_w6_diagnostic.json](/Users/gduby/Documents/autoresearch-trading/v3/artifacts/analysis/sim_l3_iter2_w6_diagnostic.json)
- [v3/artifacts/analysis/sim_l3_iter2_w6_day_diff.csv](/Users/gduby/Documents/autoresearch-trading/v3/artifacts/analysis/sim_l3_iter2_w6_day_diff.csv)

The script compares saved Layer-2 chosen trades and calibrated Layer-3
replays. It does not retrain anything.

## Main result

Seed `42`, window `6`:

| branch | W6 trades | W6 L3 PF | W6 L3 PnL | L3 mode |
|---|---:|---:|---:|---|
| promoted champion | 9 | 2.724 | +$2,369 | model |
| iteration-2 | 44 | 0.919 | -$1,262 | model |
| iteration-2 +0.05 margin | 29 | 0.618 | -$5,682 | fallback |

The iteration-2 damage is not from common days:

- champion W6 days: `9`
- iteration-2 W6 days: `44`
- common days: `7`
- iteration-2-only days: `37`
- common-day L3 delta: **+$925** for iteration-2
- iteration-2-only L3 result: **37 trades, PF 0.762, PnL -$3,569**

## Diagnosis

The iteration-2 branch did something useful on shared opportunities, but
expanded into a bad extra-day set.

The bad extra-day set is not a low-margin tail that a scalar gate can
cleanly trim:

- iteration-2-only losing trades had median decision margin `0.527`
- their mean predicted clean-entry probability was `0.381`
- their mean predicted stopout risk was `0.267`
- their mean time-stop PnL was `-$810`
- their mean realized Layer-3 hold was `138` bars

Those are not barely admitted trades. A global positive margin has to cut
deep into the sample before touching them.

The `+0.05` margin rescue had a second-order failure mode: it reduced
prior-window training coverage enough that W6 Layer-3 fell below
`min_train_trades=40`.

- champion W6 L3 train trades: `72`
- iteration-2 W6 L3 train trades: `46`
- margin-offset W6 L3 train trades: `25`

So the margin-offset run used fallback time-stop exits for all W6 trades,
which explains the collapse to `-$5,682`.

## Interpretation

The handoff's conclusion is confirmed: the iteration-2 tradeoff is not
fixable by global entry-gate tightening.

More specifically:

- iteration-2 improves some selected trade timing/contract choices on
  days the champion already liked
- the failure comes from extra-day admission in W6
- those extra days are confidently selected, so margin offset is the
  wrong control surface
- trade-count reduction can also starve early Layer-3 windows and force
  fallback exits

## Next structural branch

Do not spend more loops on global decision-margin offsets.

The most direct next branch is candidate-trained Layer-3: train the exit
model on a broader candidate-trade distribution instead of repeatedly
distilling through the currently chosen champion trades. That targets the
remaining mismatch directly: the entry model is choosing candidates whose
simulated-L3 objective can look acceptable, but the downstream rolling
Layer-3 model has not learned exits on that candidate distribution.
