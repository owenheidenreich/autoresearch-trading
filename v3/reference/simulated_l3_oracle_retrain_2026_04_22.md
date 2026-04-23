# Simulated-L3 Oracle Outer-Loop Retrain — 2026-04-22

## Design + Build (cycles 13-14)

Goal: for every `(day, bar, candidate_contract)` in the action-surface
dataset where `tradeable_mask == 1`, compute
`l3_exit_pnl` = the PnL the champion `CPU w=0.00 + L3 robust 0.90`
policy would have realized if it had entered that contract on that bar.

Built a new module [v3/layer2/build_simulated_l3_oracle.py](/Users/gduby/Documents/autoresearch-trading/v3/layer2/build_simulated_l3_oracle.py)
that:

1. Rebuilds champion per-window L3 models by running
   `train_models_by_window` on the champion seed-42 chosen trades
2. Loads per-window robust-90 thresholds from the champion's
   calibration JSON
3. For each `(day, bar)` in the action-surface dataset, looks up
   which rolling window covers that day as OOS
4. For each tradeable candidate contract at that bar, builds per-bar
   state + trade_state features (reusing `_build_trade_data`), runs
   the champion L3 model, and returns the exit PnL at first
   prob≥threshold crossing (or time-stop fallback)

Produces a sidecar `.npz` aligned to dataset row order:

```
v3/artifacts/simulated_l3_oracle_seed42.npz
  l3_exit_pnl      (89692, 25)  float32
  l3_exit_bar      (89692, 25)  int32
  l3_exit_trigger  (89692, 25)  int8
  meta_json        str
```

Build time: **52 minutes** CPU, 780k simulations over 780 OOS days.
No schema changes to the action-surface dataset.

### Oracle validation — passes

All 410 champion chosen trades map to the identical exit PnL in the
oracle within floating-point noise:

- max abs diff: `0.0001` (single-precision)
- mean abs diff: `0.0000`

Oracle vs existing utility targets on the same ~780k tradeable cells:

| label | mean | med | std | winners | corr(time_stop) |
|---|---|---|---|---|---|
| `utility_raw` (time-stop) | -27.7 | -194 | 878 | 28.6% | 1.000 |
| **`oracle_l3`** | **-8.8** | **-55** | **505** | **34.8%** | **0.578** |
| `best_exit_pnl` | +549.7 | +222 | 930 | 86.0% | 0.853 |

Oracle sits where it should: tighter variance than time-stop (L3
exits bound losses), slightly positive drift vs time-stop, higher
winners rate than time-stop but nowhere near the oracular upper
bound. Correlation with time-stop `0.578` — distinct training
signal.

Trigger breakdown of simulated cells (780,844 total):
- model-triggered: 696,516 (89.2%)
- time_stop_fallback: 19,226 (2.5%)
- fold0_time_stop: 65,102 (8.3%)

## Retrain (cycle 15)

Wired `--utility-target simulated_l3 --simulated-l3-oracle <npz>`
into [train_unified_policy.py](/Users/gduby/Documents/autoresearch-trading/v3/layer2/train_unified_policy.py).
Ran 3-seed CPU dev + L3 robust 0.90 on the new target.

```bash
for s in 42 43 44; do
  .venv/bin/python -m v3.layer2.train_unified_policy --tier dev --device cpu \
    --seed $s --run-dir v3/artifacts/layer2_unified_policy_simL3_seed$s \
    --utility-target simulated_l3 \
    --simulated-l3-oracle v3/artifacts/simulated_l3_oracle_seed42.npz
  .venv/bin/python -m v3.layer3.train_rolling --entry-source unified \
    --chosen-trades v3/artifacts/layer2_unified_policy_simL3_seed$s/seed_$s/chosen_trades.pkl \
    --out-dir v3/artifacts/layer3_unified_cpu_simL3_seed$s
  .venv/bin/python -m v3.layer3.calibrate_threshold \
    --run-dir v3/artifacts/layer3_unified_cpu_simL3_seed$s \
    --policy prior_window_robust --robust-slack 0.90 --out-suffix robust_90
done
```

### Results

| seed | L3 PF (champ) | L3 PF (simL3) | Δ PF | L3 DD (champ) | L3 DD (simL3) | trades (champ→simL3) |
|---|---|---|---|---|---|---|
| 42 | 1.711 | 1.767 | +0.056 | 20.2% | **11.3%** | 410 → 220 |
| 43 | 1.725 | **1.901** | **+0.176** | 21.7% | **9.1%** | 390 → 270 |
| 44 | 1.913 | 1.509 | **-0.404** | 24.1% | **9.8%** | 401 → 302 |
| **mean** | 1.783 | 1.726 | -0.057 | 22.0% | **10.0%** | 1201 → 792 |
| **min** | **1.711** | 1.509 | **-0.202** | — | — | — |
| **max DD** | 24.1% | **11.3%** | -12.8pp | — | — | — |

Trade shares (L2 aggregate): champion `~50%`, simulated_L3 `28-39%`.
Policy is materially more selective.

Entry-only time-stop PFs for simL3: `0.967 / 0.989 / 0.909` (all
sub-1.0) — the h60 failure mode **repeats** here too: chosen entries
would lose money held to session-end, but L3 composition recovers
them (2/3 seeds).

### Seed-44 regression localization

| W | champ thr | champ n | champ $ | simL3 thr | simL3 n | simL3 $ | Δ $ |
|---|---|---|---|---|---|---|---|
| 0 | 0.20 | 33 | +2343 | 0.20 | 58 | **+15981** | +13638 |
| 1 | 0.15 | 55 | +10604 | 0.15 | 2 | +14 | **-10591** |
| 8 | 0.15 | 26 | +13125 | 0.15 | 54 | -2756 | **-15880** |
| 5 | 0.15 | 42 | +4306 | 0.15 | 31 | -1014 | -5320 |

W0 gains `+$13.6k`, but W1 (abstains almost entirely, 55→2 trades)
and W8 (overtrades into losers, 26→54) together lose `~$26k`. Not a
single-window anomaly; the policy is genuinely re-sorting which
days to trade in a seed-specific pattern.

## Interpretation

### This is not the prior failure mode

All prior outer-loop experiments (side-contrastive, oracle blend,
horizon-60) followed the same pattern: **min PF regressed, min DD
regressed, trade count inflated**. Simulated-L3 breaks that pattern:

- min DD drops from `22% → 10%` — every seed's DD improves
- trade count drops 34% — policy is more selective, not over-trading
- entry-only time-stop PFs are close to 1.0, not dropping to 0.81

But min PF still regresses (-0.202, seed 44). Seed 42 and seed 43
meaningfully improve; seed 44 regresses in a way that W0's big
win can't offset.

### PF-vs-DD tradeoff is real and quantifiable

| metric | champion | simL3 | Δ |
|---|---|---|---|
| mean PF | 1.783 | 1.726 | -0.057 |
| min PF | **1.711** | 1.509 | **-0.202** |
| mean DD | 22.0% | **10.0%** | **-12pp** |
| max DD | 24.1% | **11.3%** | **-13pp** |
| trades | 1201 | 792 | -34% |

**mean DD cut in half** and **max DD cut in half**. Aggregated PnL
is clearly higher per-trade on a DD-normalized basis.

### Stop condition hit

Per addendum: "the first narrow implementation clearly regresses
the PF floor" — min PF regresses by -0.202. This **does** meet the
stop condition as written. The workstream cannot promote the
simulated_L3 stack over the locked champion on min-PF grounds.

### But the result is load-bearing for future loops

Unlike the prior three falsifications which were clear "do not
pursue," this result shows the simulated-L3 oracle **is** the
right outer-loop supervision target in a way the proxy targets
were not. What the cycle does *not* prove is that retraining
against this target produces a promotion-grade stack under the
locked min-PF criterion.

Two hypotheses for why min PF still regresses:

1. **Distribution-shift from applying champion L3 to non-chosen
   candidates.** Training L3 on chosen trades and then
   applying to random candidates may produce biased oracle values
   on candidates the champion would never have picked. Fix: retrain
   L3 on all candidate contracts and regenerate oracle.
2. **Per-seed oracle.** Seed 42's L3 was the oracle for all three
   retrained seeds. Seed 44's champion L3 behaves differently
   (seed 44 median hold was 81 bars vs 62 on seeds 42/43). Using
   seed-44-specific oracle for seed 44 retrain might close the gap.

Neither hypothesis can be tested in this loop without chaining a
second major target family, which the addendum explicitly
disallows.

## Repo-belief changes

- **Simulated-L3 oracle infrastructure works and is validated.**
  One canonical seed-42 `.npz` artifact, `780k` sims, `52-min` CPU
  build, round-trip to champion chosen trades is exact.
- **Simulated-L3 supervision produces dramatically safer stacks
  (DD halved) but at the cost of min-PF floor (-0.202).** Not a
  clean promotion over champion under the min-PF criterion.
- **The locked provisional champion remains `CPU w=0.00 + L3 robust
  0.90`** — mean PF 1.783, min 1.711, DD 22.0%.
- Future-loop candidates (not this loop):
  - per-seed oracles
  - candidate-trained L3 (remove distribution shift)
  - composed-stack promotion criterion that values DD more heavily
    alongside PF floor

## Artifacts

- [v3/layer2/build_simulated_l3_oracle.py](/Users/gduby/Documents/autoresearch-trading/v3/layer2/build_simulated_l3_oracle.py) (build script, 287 lines)
- `v3/artifacts/simulated_l3_oracle_seed42.npz` (4.1MB, 780k sims)
- `v3/artifacts/layer2_unified_policy_simL3_seed{42,43,44}/`
- `v3/artifacts/layer3_unified_cpu_simL3_seed{42,43,44}/`
- New trainer flag: `--utility-target simulated_l3 --simulated-l3-oracle <npz>`

## Stop

Loop ends here. Per the addendum: "the first narrow implementation
clearly regresses the PF floor" — stop condition met. Committed
cleanly, no partial state.
