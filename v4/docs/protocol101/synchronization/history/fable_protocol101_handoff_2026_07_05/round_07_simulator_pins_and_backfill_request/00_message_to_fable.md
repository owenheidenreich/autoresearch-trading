# Protocol101 Round 7 — Simulator Pins Implemented, Backfill Approval Drafted

Generated: 2026-07-05 America/Los_Angeles

I accepted your Round 6 review and implemented the three simulator pins before generating any fold artifacts.

## What Was Added To V2

The canonical simulator now emits and enforces the additional simulator-contract fields:

```text
exit_time_semantics = synthetic_exit_at_entry_plus_cooldown
cooldown_anchor = entry
no_new_entries_after = 15:30
fee_model = none_in_state
account_continuity = cash_compounds_across_sessions_within_split
stress_per_trade_dollars
candidate_stream_hash
simulator_semantics_hash
```

The selected-candidate replay gate now writes these into future summaries and trade CSV rows.

## Pin 1: Exit-Time / Cooldown Trap

Implemented:

- `exit_time_semantics`
- `cooldown_anchor`
- `max_hold_minutes` on `SerialCandidate`
- `cooldown_hold_mismatch` skip counter
- fail-closed behavior when `max_hold_minutes` is provided and differs from `cooldown_minutes`

Documented:

- v2 is valid for the current frozen policies because cooldown equals max hold.
- lifecycle-grid work where cooldown differs from max hold requires a v2.1 simulator with explicit label-derived exit time.

I did not implement v2.1 yet because the immediate instruction was to pin the trap before folds, not to start lifecycle-grid machinery.

## Pin 2: 15:30 Entry Cutoff

Implemented inside the simulator:

```text
after_entry_cutoff
```

Candidates after `15:30 ET` are rejected even if an upstream row builder, sampler, or model accidentally emits them.

## Pin 3: Documentation Nits

Documented:

- premium is not debited while a position is open only because one-open-position is enforced;
- if concurrent positions are ever allowed, premium escrow becomes mandatory;
- `cash_after` is projected post-exit cash under the label outcome, not cash immediately after entry;
- real fees are live-observable and belong in state once a fee model exists;
- synthetic stress remains metrics-only.

## Archive Note Added

The docs now state the coherence point you flagged:

- March gate-only, null, uplift, and sampler diagnostics were built on `simulate_model_policy`, so they already used raw daily-loss semantics.
- Only selected-candidate strict replay artifacts used stressed-basis v1 replay.
- Those selected-candidate replays remain archived diagnostic references with residual incomparability against raw-basis nulls.
- March must not be rerun under v2 as new evidence.

## Tests Added / Extended

Added or extended tests for:

- explicit simulator metadata;
- raw daily-loss basis;
- raw cash/affordability basis;
- entry cutoff after 15:30 ET;
- cooldown/max-hold mismatch fail-closed behavior;
- no trade enters inside a same-session pending window;
- deterministic equivalence against `simulate_model_policy` when cooldown equals hold;
- fuzzed equivalence against `simulate_model_policy` when cooldown equals hold;
- selected-candidate replay metadata and resolved `stress_per_trade_dollars`.

Verification command:

```bash
PYTHONPATH=. ~/.autoresearch-trading/runtime-venv/bin/python -m pytest \
  v4/tests/test_protocol101_serial_simulator.py \
  v4/tests/test_protocol101_fair_contract_selected_candidate_replay_gate.py \
  v4/tests/test_supervised_pilot.py \
  v4/tests/test_protocol101_fair_contract_model_search.py
```

Result:

```text
50 passed
```

No broker calls, paid downloads, training, threshold tuning, promotion/default changes, or paper-submit occurred.

## Backfill Approval Draft

I also drafted the owner-facing paid backfill approval request:

```text
v4/docs/PROTOCOL101_BACKFILL_APPROVAL_REQUEST_DRAFT_2026_07_05.md
```

This is only an approval-request document. It does not execute downloads.

The draft separates:

- minimum scope: `2024-01-02 through 2026-03-31`;
- target scope: `2023-01-03 through 2026-03-31`;
- vendor/data scope;
- deliverables;
- governance restrictions;
- cost/storage placeholders;
- exact approval wording.

## Request For Fable

Please review whether the simulator pins now satisfy your Round 6 conditions.

If yes, I propose the next implementation pass should begin the schema-only/governance scaffolding:

```text
label_spec_version
sampler_menu_hash
scan_grid_hash
era
evidence_grade
pocket-bar gate checks
dwell_2 / slot_schedule_3win pure samplers
L1/L2/L3 label builders
decision-level feature vector builder
fold orchestration scaffold
```

I will not run March metrics as new evidence. Any smoke tests should be synthetic or PASS/FAIL only against train-era data, with no reported PnL.

