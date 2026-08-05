# Protocol101 Round 4 — Uplift / Lifecycle / Sampler Packet Results

Generated: 2026-07-05

We implemented the combined packet you requested.

This was run as an offline provisional diagnostic only:

- no model training;
- no threshold tuning;
- no broker calls;
- no paper-submit;
- no paid downloads;
- no default or promotion changes;
- no recorder/confirmation days used for selection.

## What We Added

### 1. Lower-tail p-values in the gate/null baseline script

We updated:

```text
v4/scripts/run_protocol101_fair_contract_gate_null_baselines.py
```

It now reports:

```text
lower_tail_p_value_vs_null_u
lower_tail_p_value_vs_null_m
```

We reran the 3-gate / 1,000-seed variant packet.

### 2. Combined uplift/lifecycle attribution script

We added:

```text
v4/scripts/run_protocol101_fair_contract_uplift_lifecycle_attribution.py
```

It produces:

```text
report.md
summary.json
strategy_uplift_rows.csv
stratum_uplift_rows.csv
lifecycle_attribution_rows.csv
candidate_null_m_rows.csv
```

The packet includes:

- all-gated parallel selected-label stats;
- first-eligible strict serial stats;
- 3-window slot-schedule stats;
- attempt130 serial stats;
- attempt130 top-k-parallel stats;
- attempt131 serial stats;
- attempt131 top-k-parallel stats;
- candidate-matched Null-M for attempts 130 and 131;
- gate-age strata;
- momentum5/momentum15 strata;
- vwap-gap strata;
- premium strata;
- V_sel / V_band / policy0/1/2 lifecycle attribution;
- E/C decomposition;
- session-bootstrap CIs for winsor means and E/C components;
- return-on-premium columns.

## Main Results

### A. First-eligible sampler harm is now explicit

The refreshed gate/null packet shows first-eligible is in the lower tail of the random-in-gate null.

| Gate | Split | First-Eligible PnL | Upper p vs Null-M | Lower p vs Null-M |
|---|---|---:|---:|---:|
| base | validation | -$950 | 0.9600 | 0.0410 |
| base | diagnostic | -$2,400 | 0.9800 | 0.0210 |
| near-VWAP | validation | +$3,690 | 0.3177 | 0.6833 |
| near-VWAP | diagnostic | -$1,950 | 0.9311 | 0.0709 |
| premium>=7.5 | validation | -$950 | 0.8771 | 0.1249 |
| premium>=7.5 | diagnostic | -$2,400 | 0.9710 | 0.0300 |

This supports your reading:

```text
first-eligible is not merely weak; it is often materially worse than random-in-gate.
```

### B. Slot schedule is less harmful, but not paper-ready

The 3-window slot schedule materially improved the base/premium diagnostic paths, but it does not create a strong candidate on these March windows.

| Gate | Split | First-Eligible Stressed PnL | Slot-Schedule Stressed PnL |
|---|---|---:|---:|
| base | validation | -$950 | +$235 |
| base | diagnostic | -$2,400 | -$110 |
| near-VWAP | validation | +$3,690 | +$1,490 |
| near-VWAP | diagnostic | -$1,950 | -$630 |
| premium>=7.5 | validation | -$950 | -$310 |
| premium>=7.5 | diagnostic | -$2,400 | +$980 |

Our interpretation:

```text
slot schedule is a plausible sampler repair candidate for future folds,
but not proven on March-only evidence.
```

### C. Candidate-matched Null-M does not clear attempts 130/131

| Candidate | Gate | Split | Candidate Stressed PnL | Null-M p95 | Upper p | Lower p |
|---|---|---|---:|---:|---:|---:|
| attempt131 | base | validation | +$2,540 | +$4,501 | 0.3017 | 0.7013 |
| attempt131 | base | diagnostic | -$330 | +$2,295 | 0.6693 | 0.3327 |
| attempt130 | near-VWAP | validation | +$4,760 | +$4,817 | 0.0549 | 0.9461 |
| attempt130 | near-VWAP | diagnostic | +$50 | +$1,050 | 0.2627 | 0.7393 |

Attempt130 validation is close to the null p95 but still not a clean pass, and diagnostic is not close.

Our interpretation:

```text
attempt130/131 are still diagnostics, not candidates.
The model is not clearly better than candidate-matched random minute choice.
```

### D. Entry-timing uplift explains much of the sampler issue

The E/C attribution now has bootstrap CIs. The clearest harmful component is negative entry-timing uplift for first-eligible in diagnostic.

Examples:

| Gate | Split | Strategy | E | E 95% CI | C | C 95% CI |
|---|---|---|---:|---:|---:|---:|
| base | diagnostic | first-eligible | -151.1 | [-346.7, -7.6] | +64.5 | [19.5, 114.6] |
| premium>=7.5 | diagnostic | first-eligible | -163.9 | [-368.8, -18.6] | +50.9 | [6.8, 96.0] |
| base | validation | first-eligible | -62.0 | [-288.5, 125.3] | +19.8 | [-42.2, 91.6] |
| near-VWAP | diagnostic | first-eligible | -117.9 | [-282.2, 24.9] | +42.5 | [-4.6, 99.0] |

Our interpretation:

```text
The strongest provisional mechanism is bad entry timing from first-eligible,
especially in diagnostic base/premium gates.
But March-only CIs are wide, so this should become a fold-tested hypothesis, not a promoted rule.
```

### E. Gate-age / momentum strata are suggestive but small

For base first-eligible:

- validation mom15_pos: 10 trades, mean -$140, median -$300;
- validation mom15_flat: 8 trades, mean +$238.8, median +$75;
- diagnostic mom15_pos: 15 trades, mean -$34.7, median -$140;
- diagnostic mom15_flat: 9 trades, mean -$118.9, median -$290.

For near-VWAP first-eligible:

- validation is broadly positive across momentum buckets;
- diagnostic mom15_pos is bad: 13 trades, mean -$122.3, median -$210;
- diagnostic mom15_neg/flat are less bad or slightly positive.

Our interpretation:

```text
positive momentum put entries remain suspicious,
but sample sizes are too small to authorize another momentum filter.
```

### F. Lifecycle policies do not provide a clean rescue

The script computed policy1/policy2 deltas for selected trades and designated the better alternative policy from train rows only.

Train-designated alternate policy by gate:

```json
{
  "put_near_after_0940_vwap_m2_10": 1,
  "put_near_after_0940_vwap_m2_10_near_vwap": 2,
  "put_near_after_0940_vwap_m2_10_premium_gte_7_5": 1
}
```

But March selected-trade deltas are mixed:

- attempt131 diagnostic policy1 delta: +$38.2/trade, policy2: -$105/trade;
- attempt130 diagnostic policy1 delta: -$15.3/trade, policy2: -$234.1/trade;
- near-VWAP validation gate-first has large negative policy1/policy2 deltas despite high policy0 PnL.

Our interpretation:

```text
There is no clean evidence yet that changing lifecycle alone solves the problem.
Lifecycle should remain a measured component, not the next hill-climb surface.
```

## What This Changes

Our current state after your Round 3 guidance and this packet:

```text
gate-only candidate branch: closed on March evidence
first-eligible sampler: evidence-backed suspect
slot schedule / dwell-2: plausible pre-registered fold candidates, not March-tuned fixes
attempt130/131: diagnostic references only
new training: still paused
critical path: backfill + fold-aware evaluation
```

We agree with your statement that the backfill is now the formal critical path.

## Questions For You

### 1. Is the next step now backfill first, before L1/L2/L3 training?

Given this packet, do you agree the next implementation should be:

```text
build fold-aware protocol101-live-v1 history first,
then rerun gate/null/uplift/lifecycle across folds,
then only train L1/L2/L3 if the fold evidence supports a real pocket?
```

Or should we implement L1/L2/L3 machinery now but not run model selection until folds exist?

### 2. Should dwell-2 and slot schedule be frozen as the only sampler variants for fold testing?

You proposed:

```text
dwell-2
slot_schedule_3win
```

Given the March slot results, should the sampler menu for folds be exactly:

```text
first_eligible reference
dwell_2
slot_schedule_3win
```

with no other sampler variants allowed?

### 3. Should attempt130 be fully retired from candidate consideration?

Attempt130:

- validation candidate-matched Null-M upper p = 0.0549;
- diagnostic upper p = 0.2627;
- diagnostic PnL = +$50;
- does not clear null or fold requirements.

Should it now be explicitly marked:

```text
diagnostic_reference_only_not_candidate
```

### 4. How should we use top-k parallel findings?

Some top-k-parallel results are interesting but not stable:

- attempt131 diagnostic top-k parallel mean is much better than serial;
- attempt130 diagnostic top-k parallel is worse than serial;
- serial sampling delta signs differ.

Does this point to:

```text
entry timing signal exists but serial path/cooldown damages it,
or
top-k is just another small-sample artifact until folds exist?
```

### 5. What is the exact backfill/fold priority?

If backfill is next, please specify the minimum useful target in implementation terms:

```text
date range:
schemas/data:
fold size:
embargo:
minimum fold count:
minimum pooled trades:
which existing scripts should be run first:
what reports should be produced before model training resumes:
```

### 6. Should we also instrument a broader V_sel opportunity scan now?

You suggested:

```text
recast the two-stage opportunity scan with realizable V_sel values over broader gates, both sides, and all three lifecycle policies.
```

Should that happen:

```text
A. before backfill,
B. during backfill on existing data as machinery prep,
C. only after fold backfill exists?
```

## Files In This Packet

```text
01_fable_round3_response.md
02_uplift_lifecycle_report.md
03_uplift_lifecycle_summary.json
04_strategy_uplift_rows.csv
05_lifecycle_attribution_rows.csv
06_candidate_null_m_rows.csv
07_stratum_uplift_rows.csv
08_uplift_lifecycle_script.py
09_gate_null_variant_report_refreshed.md
10_gate_null_variant_summary_refreshed.json
11_gate_null_baseline_script_refreshed.py
```

