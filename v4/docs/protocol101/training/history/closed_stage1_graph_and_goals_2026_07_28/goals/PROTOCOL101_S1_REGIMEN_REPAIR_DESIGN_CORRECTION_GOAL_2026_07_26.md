# Protocol101 Goal Prompt: S1 Regimen Repair Design Correction

Paste the text below into a fresh Codex Goal task.

---

GOAL ID:

`S1-REGIMEN-REPAIR-DESIGN-CORRECTION`

OBJECTIVE:

Correct one scientific-contract error in the unsigned Stage-1 regimen repair
design, regenerate the owner decision packet as attempt002, and stop before
implementation.

The attempt001 design incorrectly requires a source quote exactly at the
max-hold/forced-flat deadline and requires:

```text
label_realized_exit_time_ns == label_source_exit_quote_time_ns
```

That is not the frozen label behavior. The current governed labeler uses the
first stop/target quote, otherwise the latest causal quote available at or
before the policy deadline. Requiring an exact deadline quote would silently
change payoff labels, invalidate otherwise usable rows, and turn a timing
repair into a new trading strategy.

This Goal must replace that rule with an explicit two-clock contract:

1. **Occupancy/decision exit time:** when the position is considered closed.
2. **Source quote time:** when the bid used to price that exit was observed.

This is a correction to an unsigned design. Do not modify attempt001.

WORKSPACE AND INTERPRETER:

```text
workspace: /Users/gduby/Documents/autoresearch-trading
interpreter: ~/.autoresearch-trading/runtime-venv/bin/python
```

READ BEFORE ACTING:

- `v4/docs/protocol101/training/README.md`
- `v4/docs/protocol101/training/goals/PROTOCOL101_S1_REGIMEN_REPAIR_DESIGN_GOAL_2026_07_26.md`
- `v4/audit/autoresearch/protocol101_stage1_regimen_repair_design_attempt001/owner_decision_packet.md`
- `v4/audit/autoresearch/protocol101_stage1_regimen_repair_design_attempt001/report.md`
- `v4/audit/autoresearch/protocol101_stage1_regimen_repair_design_attempt001/realized_exit_contract.json`
- `v4/audit/autoresearch/protocol101_stage1_regimen_repair_design_attempt001/serial_simulator_successor_contract.json`
- `v4/audit/autoresearch/protocol101_stage1_regimen_repair_design_attempt001/processed_row_schema_proposal.json`
- `v4/audit/autoresearch/protocol101_stage1_regimen_repair_design_attempt001/immutable_replay_artifact_schema.json`
- `v4/audit/autoresearch/protocol101_stage1_regimen_repair_design_attempt001/equivalence_certificate_schema.json`
- `v4/audit/autoresearch/protocol101_stage1_regimen_repair_design_attempt001/independent_acceptance_specification.json`
- `v4/audit/autoresearch/protocol101_stage1_regimen_repair_design_attempt001/repair_test_matrix.csv`
- `v4/dataset/spxw_0dte_neural.py`
- `v4/scripts/run_protocol101_owned_raw_acceptance_verifier.py`
- `v4/tests/test_protocol101_owned_raw_acceptance_verifier.py`

OUTPUT DIRECTORY:

```text
v4/audit/autoresearch/protocol101_stage1_regimen_repair_design_correction_attempt002/
```

FROZEN ATTEMPT001 INPUT:

Verify attempt001 has:

```text
terminal_route: repair_design_ready_for_owner_signature
summary_hash: bd54a8f62008f50c65352d61156231d3c15f30983dab3c7bc1f414e188d4ae14
amendment_signed: false
```

Run its checksum validation from inside its directory. Record attempt001's
full hash inventory and prove it remains unchanged after this Goal.

CORRECTED EXIT CONTRACT:

For each candidate/policy cell, select the same pricing event as the frozen
label law:

1. Use only same-contract, same-session quote states with source quote time
   strictly after entry and at or before the policy deadline.
2. Sort deterministically by source quote time and fail closed on duplicate
   path quote identities.
3. The first no-bid/stop/target event wins.
4. If no threshold event occurs, use the latest causal quote state at or
   before the policy deadline, exactly as the frozen labeler does.
5. If no causal future quote exists, the cell is invalid.
6. Do not require a new source quote exactly at the deadline.
7. Do not introduce a new quote-age rejection threshold in this correction.

Pin the two clocks:

- For `stop_loss`, `take_profit`, or `no_bid_stop`:
  - `label_realized_exit_time_ns = label_source_exit_quote_time_ns`
- For `max_hold` or `forced_flat`:
  - `label_realized_exit_time_ns = label_policy_deadline_ns`
  - `label_source_exit_quote_time_ns <= label_realized_exit_time_ns`
  - the executable bid comes from the latest causal quote state selected
    above.

The simulator must release occupancy at
`label_realized_exit_time_ns`, not at the source quote's timestamp when a
deadline exit uses an earlier quote state.

Add an audit-only field:

```text
label_exit_quote_age_ms
```

defined as:

```text
(label_realized_exit_time_ns - label_source_exit_quote_time_ns) / 1_000_000
```

It must be nonnegative, reported by policy/session/time-of-day, and forbidden
from model alpha. It is diagnostic only in this correction. A future global
quote-age rule would require a separate owner amendment and may not be chosen
after repaired PnL is visible.

REQUIRED INVARIANTS:

- `decision_time_ns < label_source_exit_quote_time_ns`
- `label_source_exit_quote_time_ns <= label_realized_exit_time_ns`
- `label_realized_exit_time_ns <= label_policy_deadline_ns`
- threshold/no-bid exits have source time equal to realized exit time
- max-hold/forced-flat exits have realized exit time equal to deadline
- entry and both exit clocks belong to the same New York session
- PnL uses the bid from `label_source_exit_quote_time_ns`
- occupancy uses `label_realized_exit_time_ns`
- no missing metadata may fall back to synthetic max hold
- all exit, path, quote-age, and PnL fields remain blocked from model inputs

EQUIVALENCE CONSEQUENCE:

Update E06/E07 and related checks so the intended repair can prove existing
`labels_net_pnl` and `labels_mid_pnl` are bit-identical to the frozen corpus.

The correction must not force a full refit merely because it invented an
exact-deadline quote requirement that the frozen strategy never had.

The all-or-nothing E01-E13 branch remains:

- all checks pass: reuse all 420 frozen models for repaired replay;
- any check fails: refit all 420;
- partial or mixed reuse remains forbidden.

OTHER ATTEMPT001 DECISIONS:

Preserve unless this two-clock correction creates a direct contradiction:

- simulator-v5 actual-exit occupancy;
- v4 account continuity and fee reserve;
- fail-closed duplicate identities;
- D1 complete-block resolution;
- D5 heuristic rebuild and identity hashes;
- D6 signed split-family offline authority and later no-order shadow transfer;
- immutable rebuild/independent acceptance sequence;
- 20,000-replicate synchronized five-session maxT proposal;
- owner choice between hard and report-only multiplicity enforcement.

Do not calculate maxT results in this Goal.

OWNER DECISION PACKET:

Regenerate an unsigned packet that:

- describes the corrected two-clock semantics plainly;
- contains no exact-deadline-source-quote requirement;
- states that latest causal quote state is used at deadline;
- identifies quote age as required audit reporting but not a gate;
- uses the phrase `offline Stage-1 research`, not `offline label-only work`,
  when describing D6;
- retains the hard-versus-report-only multiplicity checkbox;
- includes the actual SHA-256 of `owner_decision_packet.json` in the Markdown
  packet; and
- authorizes only the later bounded machinery implementation after signature.

ALLOWED WORK:

- Read attempt001 and current governed source/tests.
- Create corrected design artifacts under attempt002.
- Create audit-only schema-validation tooling under attempt002.
- Recompute internal design hashes and packet checksums.
- Continue through mechanical output or checksum blockers.

FORBIDDEN WORK:

- Do not modify attempt001.
- Do not modify governed dataset, model, simulator, runner, gate, or runtime
  code.
- Do not train, refit, score, replay economics, aggregate gates, rank, or
  select.
- Do not run seed 45/G9.
- Do not read the holdout or sealed evidence.
- Do not contact a broker, submit paper orders, download paid data, or modify
  promotion/runtime/launchd/real-money state.
- Do not sign on behalf of the owner.

BLOCKER RECOVERY:

Continue through transient and mechanical documentation/schema blockers for
up to three attempts per component. Preserve failed outputs under
`void_outputs/`.

Stop for owner input only if the correction would require changing frozen
payoff labels, policy thresholds, max holds, feature sets, folds, seeds,
model family, G1-G9, or protected-data boundaries.

REQUIRED OUTPUT:

```text
source_inventory.json
attempt001_integrity_before_after.json
correction_diff.json
realized_exit_contract_v2.json
realized_exit_contract_v2.md
processed_row_schema_proposal_v2.json
serial_simulator_successor_contract_v2.json
serial_simulator_successor_contract_v2.md
immutable_replay_artifact_schema_v2.json
equivalence_certificate_schema_v2.json
independent_acceptance_specification_v2.json
repair_test_matrix_v2.csv
owner_decision_packet.json
owner_decision_packet.md
progress.json
summary.json
hashes.sha256
report.md
```

TERMINAL ROUTES:

- `repair_design_correction_ready_for_owner_signature`
- `repair_design_correction_blocked_contract_conflict`
- `repair_design_correction_requires_label_strategy_amendment`
- `repair_design_correction_blocked_missing_nonprotected_evidence`

PASS REQUIREMENTS:

- attempt001 remains byte-identical;
- frozen label selection behavior is reproduced in the corrected design;
- source quote time and occupancy exit time are separate where appropriate;
- no exact-deadline source quote is required;
- existing PnL labels remain the equivalence target;
- quote age is auditable and barred from alpha;
- simulator v5 uses occupancy exit time;
- all other accepted attempt001 design decisions remain pinned;
- the new owner packet contains its actual JSON hash;
- all required files and checksums validate; and
- no forbidden action occurred.

STOP BOUNDARY:

Stop after the corrected unsigned owner packet. Do not implement simulator v5
or begin economic evidence rebuild.

HIGHEST ALLOWED CLAIM:

> Protocol101 Stage-1 regimen repair design corrected and ready for owner
> decision.

FINAL RESPONSE:

Explain:

1. Why exact-deadline quote equality was wrong.
2. How source quote time differs from occupancy exit time.
3. Whether frozen PnL labels remain the equivalence target.
4. Which attempt001 decisions remain unchanged.
5. What the owner must sign.
6. The one next Goal after signature.

