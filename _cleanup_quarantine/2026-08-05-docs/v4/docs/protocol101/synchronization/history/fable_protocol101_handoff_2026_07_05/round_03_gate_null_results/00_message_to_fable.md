# Protocol101 Round 3 — Gate-Only / Null Baseline Results And Next Questions

Generated: 2026-07-05

Thank you for the Round 2 experiment specification. We implemented the first falsification step you requested: gate-only strict serial replay plus random-in-gate null baselines.

This was run as an offline diagnostic only:

- no model training;
- no threshold tuning;
- no broker calls;
- no paper-submit;
- no paid downloads;
- no default or promotion changes;
- no recorder/confirmation days used for selection.

## What We Implemented

We added a new offline script:

```text
v4/scripts/run_protocol101_fair_contract_gate_null_baselines.py
```

The script loads the existing `protocol101-live-v1` processed rows from:

```text
v4/audit/autoresearch/protocol101_fair_contract_expanded_jul_dec2025_128_q1_design/summary.json
```

It then runs:

1. Gate-only strict serial replay.
2. Parallel per-minute selected-label diagnostics.
3. Null-U random-in-gate baseline.
4. Null-M random-in-gate baseline matched to the gate-only per-session trade counts.

The main run used your requested basic configuration:

```text
policy_index = 0
policy = ask_to_bid_stop35_target60_hold10m
cooldown = 10 minutes
selection_mode = stable_abs_offset_20
max_trades_per_session = 3
max_daily_loss = 500
starting_cash = 10000
stress_per_trade = 20
null_seeds = 1000
```

We first ran the primary broad base gate:

```text
put_near_after_0940_vwap_m2_10
```

Then we ran the two pre-registered sub-gates you mentioned:

```text
put_near_after_0940_vwap_m2_10_near_vwap
put_near_after_0940_vwap_m2_10_premium_gte_7_5
```

## Results

### Strict Serial Gate-Only Results

| Gate | Validation | Diagnostic |
|---|---:|---:|
| `put_near_after_0940_vwap_m2_10` | -$950, PF 0.838, 23 trades | -$2,400, PF 0.602, 25 trades |
| `put_near_after_0940_vwap_m2_10_near_vwap` | +$3,690, PF 1.893, 25 trades | -$1,950, PF 0.642, 26 trades |
| `put_near_after_0940_vwap_m2_10_premium_gte_7_5` | -$950, PF 0.838, 23 trades | -$2,400, PF 0.602, 25 trades |

### Null Comparison

The gate-only strategies did not beat the random-in-gate nulls.

For the primary broad base gate:

| Split | Gate PnL | Null-U p95 | Null-M p95 | p vs Null-U | p vs Null-M |
|---|---:|---:|---:|---:|---:|
| validation | -$950 | $3,996 | $3,670 | 0.9041 | 0.9600 |
| diagnostic | -$2,400 | $1,916 | $2,195 | 0.9411 | 0.9800 |

For the near-VWAP sub-gate:

| Split | Gate PnL | Null-U p95 | Null-M p95 | p vs Null-U | p vs Null-M |
|---|---:|---:|---:|---:|---:|
| validation | +$3,690 | $5,101 | $5,255 | 0.2537 | 0.3177 |
| diagnostic | -$1,950 | $1,411 | $1,720 | 0.8292 | 0.9311 |

For the premium>=7.5 sub-gate:

| Split | Gate PnL | Null-U p95 | Null-M p95 | p vs Null-U | p vs Null-M |
|---|---:|---:|---:|---:|---:|
| validation | -$950 | $5,220 | $4,791 | 0.8392 | 0.8771 |
| diagnostic | -$2,400 | $3,320 | $3,301 | 0.9291 | 0.9710 |

Interpretation from our side:

```text
Gate-only does not pass.
The near-VWAP sub-gate is validation-positive but diagnostic-negative, which looks regime/selection fragile.
Random-in-gate often beats deterministic first-eligible gate-only.
```

## Parallel Per-Minute Label Diagnostics

The parallel selected-label values are more nuanced. Some gates have small positive winsorized per-minute values even when strict serial first-eligible replay loses money.

| Gate | Split | Winsor600 Mean | Median |
|---|---|---:|---:|
| base | validation | +$5.63 | -$50 |
| base | diagnostic | +$4.92 | -$25 |
| near-VWAP | validation | +$34.27 | -$35 |
| near-VWAP | diagnostic | +$14.33 | -$20 |
| premium>=7.5 | validation | -$8.71 | -$60 |
| premium>=7.5 | diagnostic | +$21.16 | $0 |

This seems to support one of your interpretation-matrix branches:

```text
There may be weak per-minute signal or label skew inside the gate,
but strict first-eligible serial sampling does not monetize it.
```

However, since medians are mostly negative and random-in-gate often outperforms first-eligible, we are not treating this as evidence of a tradable edge.

## Our Current Reading

Your warning appears correct:

```text
The simple gate is not a hidden paper-ready candidate.
```

The current evidence also suggests:

1. The broad base gate is not enough.
2. The near-VWAP gate is likely validation/regime fragile.
3. First-eligible serial sampling may be actively harmful.
4. Random minute choice inside the gate sometimes performs better than the deterministic gate-only sequence.
5. The old learned-model validation success may have been exploiting a narrow timing/selection accident, not a stable edge.

## Questions For You Before We Implement The Next Diagnostic

We plan to proceed to your apples-to-apples label uplift table and lifecycle/exit attribution next. Before implementation, please answer these specific follow-ups.

### 1. Does Gate-Only Failure Change The Next Step?

Given that:

- gate-only strict serial failed,
- near-VWAP is validation-positive but diagnostic-negative,
- random-in-gate often beats first-eligible,
- parallel per-minute winsor means are mildly positive but medians are negative,

should we still implement the apples-to-apples label uplift table exactly as specified, or should we first add your 1D time-slot schedule probe?

In other words, should the next diagnostic priority be:

```text
A. label uplift / model-selected vs all-gated / null-selected,
B. time-slot schedule probe,
C. lifecycle attribution,
D. all three in one packet?
```

### 2. Should We Treat First-Eligible Serial Sampling As Suspect?

The nulls suggest first-eligible gate-only may be a poor deterministic sampler.

Should we now explicitly test alternative causal deterministic samplers before new model training, such as:

- one trade per pre-registered time window;
- wait until a gate persists for N minutes;
- enter only after gate re-entry/retest;
- choose a random-looking but deterministic hash minute inside a window;
- rank minutes by index-only/context-only features but not option microstructure?

Or would that be too close to adding another overfit surface before the uplift/attribution diagnostics?

### 3. How Should We Interpret Positive Parallel Mean But Negative Median?

Several gates show:

```text
winsorized mean slightly positive,
median negative,
strict serial negative.
```

Does this imply:

- rare right-tail winners are carrying the gate;
- labels are still too heavy-tailed;
- lifecycle policy is asymmetric;
- first-eligible sampling misses the right-tail minutes;
- or the apparent mean edge is probably noise until shown across walk-forward folds?

What exact statistic should be primary in the next uplift table:

```text
winsor600 mean,
median,
trim10 mean,
lower quartile,
share positive,
or a composite?
```

### 4. Should Null-M Be The Primary Null?

We implemented both Null-U and Null-M. Null-M matches the gate-only per-session trade count.

For learned candidates, should the primary null be:

```text
Null-M matched to the candidate's per-session trade count,
or Null-M matched to the gate-only per-session trade count?
```

For this gate-only experiment, those are the same, but for learned candidates they will differ.

### 5. Should We Immediately Retire The Near-VWAP Branch?

The near-VWAP gate:

- validation: +$3,690, PF 1.893;
- diagnostic: -$1,950, PF 0.642.

This looks like the exact validation/diagnostic split fragility you warned about.

Should near-VWAP now be treated as:

```text
retired until walk-forward folds prove otherwise,
or still worth including in label-uplift/lifecycle attribution as a known fragile comparison?
```

### 6. What Exact Next Artifact Should We Build?

Our proposed next artifact is:

```text
protocol101_fair_contract_label_uplift_and_lifecycle_attribution
```

It would include:

1. all-gated selected-label stats;
2. gate-only selected-minute stats;
3. random-in-gate selected-minute stats;
4. attempt130 selected-minute stats;
5. attempt131 selected-minute stats;
6. model top-k parallel stats for attempts 130/131;
7. V_sel, V_band, and V_policy0/1/2 decomposition;
8. entry-timing uplift E;
9. selection delta C;
10. lifecycle delta L;
11. serial sampling delta A.

Should this be the next implementation, or should it be split into smaller artifacts?

## Files Attached

This packet includes:

```text
01_fable_round2_response.md
02_gate_null_base_report.md
03_gate_null_base_summary.json
04_gate_null_variant_report.md
05_gate_null_variant_summary.json
06_gate_null_baseline_script.py
```

The important evidence is in:

```text
04_gate_null_variant_report.md
05_gate_null_variant_summary.json
```

