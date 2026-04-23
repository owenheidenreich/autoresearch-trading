# Hold-Aware Horizon-60 Utility Target — Falsified — 2026-04-22

## Hypothesis

Post-cycle-9 addendum directed: design and test the first hold-
horizon-aware composed utility target. The mechanism under test:
the entry policy should be trained against a utility signal that
matches the forward horizon the champion Layer-3 actually holds,
so it stops rewarding late-session peaks that L3 can't capture.

Empirical evidence for horizon choice (cycle 10 diagnostic, champion
`CPU w=0.00 + L3 robust 0.90`, 1201 trades across 3 seeds × 13
windows):

| statistic | value |
|---|---|
| mean hold | 77.7 bars |
| median hold | 67.0 bars |
| 18% of trades | exit within 10 bars (fast L3 fire) |
| 38% of trades | hold `>=150` bars (late / time-stop fallback) |
| by trigger: model mean hold | 61 bars |
| by trigger: time_stop mean hold | 166 bars |

Per-seed medians: seeds 42/43 = 62 bars, seed 44 = 81 bars. Horizon
`60` sits near the lower end, close to the L3-triggered cohort.

## Mechanism (pre-hypothesis)

A target equal to `PnL at exactly bar (entry + 60)` encodes:

- same arcsinh normalization as the existing utility
- flat action gets `0` (no trade)
- contracts get a **bounded** signal — no oracle MFE-peak leak
- variance should be lower than time-stop because 60-bar holds
  experience less theta decay

If this signal is a better proxy for realized L3 exits than time-
stop, the composed stack should lift (or at least match) the
provisional champion.

## Implementation (Cycle 11)

Added to `v3/oracles/exit_headroom.py` a `_horizon_pnl(...)` helper
that exits at `min(entry + horizon_bars, session_end, last_finite_mid)`
using the same spread-model economics as `_time_stop_pnl`.

Added `horizon_pnl` to [ACTION_LABEL_NAMES](/Users/gduby/Documents/autoresearch-trading/v3/layer2/action_surface_dataset.py)
and recorded `utility_horizon_bars` in the dataset manifest.

Re-exported:

```bash
.venv/bin/python -m v3.layer2.export_action_surface_dataset \
  --output v3/artifacts/layer2_action_surface_dataset_h60.pkl \
  --utility-horizon-bars 60
```

`v3/artifacts/layer2_action_surface_dataset_h60.pkl` — 4m17s CPU
rebuild, 424MB.

Wired `--utility-target {time_stop,horizon}` into
[train_unified_policy.py](/Users/gduby/Documents/autoresearch-trading/v3/layer2/train_unified_policy.py).
`--utility-target horizon` swaps `utility_raw` → `horizon_pnl` for
regression + ranking. `chosen_time_stop_pnl` is still derived from
the honest time-stop label, so report/eval PnL accounting is
unchanged.

## Label sanity check

Across tradeable cells:

| label | mean | median | std | p25 | p75 | >0 rate |
|---|---|---|---|---|---|---|
| utility_raw (time-stop) | -28 | -171 | 860 | -446 | +26 | 26.1% |
| **horizon_pnl** | **-30** | **-82** | **480** | **-261** | **+66** | **30.7%** |
| best_exit_pnl (oracle) | +520 | +185 | 915 | +16 | +640 | 78.9% |

`horizon_pnl` is an honest signal: same negative mean as time-stop
(theta decay is real), **half the variance** (bounded by 60-bar
hold), **30.7% winners** (slightly more than time-stop's 26.1%,
because 60-bar holds skip the last hour of session decay). Pearson
correlation with utility_raw: `0.620`. Not oracular.

## Evaluation (Cycle 12)

```bash
for s in 42 43 44; do
  .venv/bin/python -m v3.layer2.train_unified_policy --tier dev --device cpu \
    --seed $s --dataset v3/artifacts/layer2_action_surface_dataset_h60.pkl \
    --run-dir v3/artifacts/layer2_unified_policy_h60_seed$s \
    --utility-target horizon
  .venv/bin/python -m v3.layer3.train_rolling --entry-source unified \
    --chosen-trades v3/artifacts/layer2_unified_policy_h60_seed$s/seed_$s/chosen_trades.pkl \
    --out-dir v3/artifacts/layer3_unified_cpu_h60_seed$s
  .venv/bin/python -m v3.layer3.calibrate_threshold \
    --run-dir v3/artifacts/layer3_unified_cpu_h60_seed$s \
    --policy prior_window_robust --robust-slack 0.90 \
    --out-suffix robust_90
done
```

## Results

| seed | L3 PF (champ) | L3 PF (h60) | Δ | L3 DD (champ) | L3 DD (h60) | trades (champ→h60) |
|---|---|---|---|---|---|---|
| 42 | 1.711 | **1.774** | +0.063 | 20.2% | **30.8%** | 410 → 529 |
| 43 | 1.725 | 1.324 | **-0.401** | 21.7% | **53.1%** | 390 → 565 |
| 44 | 1.913 | 1.288 | **-0.625** | 24.1% | **52.3%** | 401 → 510 |
| **mean** | **1.783** | **1.462** | **-0.321** | **22.0%** | **45.4%** | 1201 → 1604 |
| **min** | **1.711** | 1.288 | **-0.423** | — | — | — |

Entry-only time-stop baselines (no L3) for the h60 entry sets:

| seed | entry PF | entry DD |
|---|---|---|
| 42 | 0.896 | 100.7% |
| 43 | 0.809 | 194.3% |
| 44 | 0.812 | 166.7% |

**All three seeds' entries LOSE MONEY under time-stop hold.** Under
the champion, the entry-only PFs are `1.066 / 1.172 / 1.097`, all
profitable.

## Falsification

The hold-aware horizon-60 target is strictly worse than the
provisional champion across every axis:

- Mean PF: `1.783 → 1.462` (**-0.321**)
- Min PF: `1.711 → 1.288` (**-0.423**)
- Mean DD: `22.0% → 45.4%` (+23pp)
- Max DD: `24.1% → 53.1%`
- Trade count: `1201 → 1604` (+34%)
- Entries lose money at time-stop on all three seeds

Stopping this workstream per the addendum stop condition: "the
first hold-aware target clearly regresses the PF floor."

## Interpretation

### The hypothesis failed for two reasons

1. **Horizon_pnl has a higher winners rate (30.7% vs 26.1%) →
   flat-vs-trade tilts toward trading.** Trade count ballooned
   +34%. The flat ranking loss now sees more "positive" contracts
   to push flat below, so the policy exits the flat bucket more
   often. Overtrading directly hits composed PF / DD.
2. **Training-time horizon ≠ deployment-time horizon.** Champion
   L3 realized exits are *bimodal* — 18% within 10 bars, 38% at
   `>=150` bars. The fixed 60-bar horizon ignores both extremes.
   Entries optimized to look good at bar 60 are not optimized for
   either the fast-L3 cohort (exit within 10 bars, often at a
   small peak) OR the no-trigger cohort (hold to session end).
   In fact, entries with "good at bar 60, bad by session end" are
   *systematically* worse for the no-trigger cohort that holds to
   `165` bars on champion data.

The entry-only time-stop baseline dropping from `~1.1` (all seeds)
to `~0.81-0.90` is the smoking gun: the policy is picking contracts
that look good at bar 60 but die badly by bar 330. L3 partially
recovers seed 42 (close to champion PF on that seed) but cannot
rescue seeds 43 / 44.

### Same failure pattern three times now

The same seed-dependent collapse pattern has now falsified three
different training-target experiments in a row:

| experiment | mean PF Δ | min PF Δ | winner seed | losers |
|---|---|---|---|---|
| side-contrastive (w=0.10, w=0.20) | tied | -0.15 | 42 | 43, 44 |
| oracle blend (α=0.10, α=0.30) | tied | -0.15 | 42 | 43, 44 |
| **horizon-60 target** | **-0.32** | **-0.42** | **42** | **43, 44** |

Seed 42 consistently benefits from training-target perturbations
that seeds 43 / 44 find destructive. This strongly suggests that
the training signal is a **noise-amplifier**, not a signal-
improver, on this stack. No direction of perturbation we have
tried improves all three seeds.

### What this means for the outer-loop retrain workstream

The addendum's good-candidate list ranked by likely
infrastructure cost was:

1. `simulated-L3 exit PnL per candidate contract using the locked
   champion exit policy` (most expensive, most honest)
2. `fixed-horizon utility` (this cycle)
3. `near-horizon MFE/MAE-derived utility` (also bounded)

Option 2 has been falsified. Option 3 would share the core defect
(fixed horizon ignores bimodal L3 hold distribution). **Option 1
is the only composed-utility target that could plausibly transfer
cleanly** — because it directly simulates the champion's deployment
behavior on every candidate contract, not a horizon proxy.

## Repo-Belief Changes

- **Fixed-horizon composed utility is not a productive outer-loop
  target on this stack.** Any proxy that assumes a single hold
  horizon collides with the champion's bimodal exit distribution
  (18% fast, 38% time-stop, ~44% in between).
- **Entry-policy retraining that raises trade count above the
  champion baseline is a red flag.** Both `oracle blend` and
  `horizon-60` increased trade counts materially (`+23%` and
  `+34%`), and both regressed composed PF. Trade-count inflation
  correlates with training targets that have more winners than
  time-stop.
- **The only defensible next outer-loop target is the full
  simulated-L3 oracle.** A bigger infrastructure build but the
  only option that aligns entry supervision with deployment
  reality. Park until a future loop with explicit scope for it.

## Provisional Champion

Unchanged: `CPU w=0.00 + L3 robust 0.90`, mean PF `1.783`, min
`1.711`, mean DD `22.0%`.

## Artifacts

- `v3/artifacts/layer2_action_surface_dataset_h60.pkl`
- `v3/artifacts/layer2_unified_policy_h60_seed{42,43,44}/`
- `v3/artifacts/layer3_unified_cpu_h60_seed{42,43,44}/`
- New flag: `--utility-target horizon` in [train_unified_policy.py](/Users/gduby/Documents/autoresearch-trading/v3/layer2/train_unified_policy.py)
- New helper: `_horizon_pnl` in [v3/oracles/exit_headroom.py](/Users/gduby/Documents/autoresearch-trading/v3/oracles/exit_headroom.py)
- New label: `horizon_pnl` in the action-surface dataset
- Diagnostic: champion L3 hold distribution captured above
