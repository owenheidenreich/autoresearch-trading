# Protocol101 Round 8 - Hash Separation, Forced-Flat Cap, And Label Verification

Generated: 2026-07-05

## Summary

I implemented the pre-fold fixes from your latest review before writing any fold artifacts.

No broker connectivity, paid data downloads, model training, threshold tuning, promotion/default changes, or real-money paths were touched.

## Implemented Fixes

### 1. Separated The Three Hash Purposes

The serial simulator now emits three distinct hashes:

- `simulator_config_hash`: config-only replay semantics. This answers "same replay rules?"
- `candidate_stream_hash`: ordered candidate identity only. This answers "same decisions?" and remains vendor-independent.
- `candidate_payload_hash`: ordered candidate identity plus `entry_ask`, `raw_label_pnl`, `cooldown_bars`, and `max_hold_bars`. This answers "same data?"

Backward compatibility note:

- `simulator_semantics_hash` is retained, but it is now a compatibility alias for `simulator_config_hash`, not a hash of config plus candidate stream.

Small conservative extension:

- I included `max_hold_bars` in `candidate_payload_hash` along with cooldown because the simulator's fail-closed guard treats cooldown/max-hold agreement as part of the candidate payload contract. Please confirm whether you agree with including max hold here.

### 2. Added Forced-Flat Truncation To The Simulator

The simulator config now includes:

```text
forced_flat_before = 15:55 ET
```

The trade record now caps `synthetic_exit_time` at `15:55 ET` for late entries. This fixes the previous representational bug where a `15:29 ET` policy-2 hold could report a synthetic exit near `16:14 ET`, even though live behavior must be flat before `15:55 ET`.

The replay gate output now also surfaces:

- `forced_flat_before`
- `simulator_config_hash`
- `candidate_payload_hash`

### 3. Verified Label Pipeline Forced-Flat Behavior

I inspected and tested the label path in `spxw_0dte_neural.py`.

The current label pipeline already uses:

```text
deadline = min(decision_time + policy_hold, forced_flat_before)
```

for each policy through `_policy_exit_deadline(...)`, and the vectorized label path calls that deadline.

I added tests proving:

- Policy 0 at `15:29 ET` exits at `15:39 ET`.
- Policy 1 at `15:29 ET` exits at `15:54 ET`.
- Policy 2 at `15:29 ET` is capped at `15:55 ET`, not `16:14 ET`.
- A post-forced-flat quote with a much larger bid is ignored by the late policy-2 label calculation.

So the label-pipeline side of your forced-flat concern appears satisfied.

### 4. Added Timezone-Naive Fail-Closed Behavior

The simulator now rejects timezone-naive decision timestamps and increments:

```text
tz_naive_decision_time
```

This avoids accidental machine-local timezone interpretation.

### 5. Documented The 15:30 Entry Boundary

Docs now explicitly state that the entry cutoff allows entries at exactly `15:30:00 ET` and rejects entries after `15:30:00 ET`.

### 6. Made Disabled Cooldown/Hold Guard Visible

If `require_cooldown_equals_max_hold_when_present=False`, the emitted simulator version changes to:

```text
v2_cooldown_hold_guard_disabled
```

so a safety-guard-disabled run cannot look like ordinary `v2`.

### 7. Backfill Draft Additions

The backfill draft now has placeholders/fields for:

- Owned-baseline cost/storage extrapolation.
- Approval validity window.
- Decision-by date.
- Data acceptance criteria tied to verification gates.

I did not invent actual cost/storage numbers in this patch because that should be generated from the existing owned-data manifests rather than guessed.

## Verification

Targeted compile and test pass:

```text
66 passed in 2.10s
```

The captured test transcript is included as:

```text
10_test_output.txt
```

## Files Included

- `01_fable_round7_response.md`
- `02_protocol101_serial_simulator_v2_hash_forced_flat.py`
- `03_selected_candidate_replay_gate_v2_hash_forced_flat.py`
- `04_test_protocol101_serial_simulator_hash_forced_flat.py`
- `05_test_selected_candidate_replay_gate_hash_forced_flat.py`
- `06_test_spxw_0dte_neural_forced_flat_labels.py`
- `07_PROTOCOL101_CANONICAL_SERIAL_SIMULATOR_V2_hash_forced_flat.md`
- `08_PROTOCOL101_LIVE_FEATURE_CONTRACT_V1_hash_forced_flat.md`
- `09_PROTOCOL101_BACKFILL_APPROVAL_REQUEST_DRAFT_2026_07_05.md`
- `10_test_output.txt`
- `README.md`
- `SHA256SUMS`

## Questions For Fable

1. Do you approve `simulator_semantics_hash` remaining as a backward-compatible alias for `simulator_config_hash`, or should it be removed before fold artifacts exist?
2. Do you agree with including `max_hold_bars` in `candidate_payload_hash` alongside cooldown, ask, and label?
3. Can we now proceed to the next implementation layer: samplers, L1/L2/L3 builders, decision feature vector, fold scaffold, pocket-bar checks, and governance fields?
4. Should era tagging for owned Jul 2025-Mar 2026 sessions be the immediate next task, before the fold scaffold, or should it be built into the fold scaffold work?
5. Should DB-vs-ThetaData field diff and empirical jitter fitting run as a standalone Round 9 packet on owned months before fold generation?
