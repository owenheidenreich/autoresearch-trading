# Protocol101 Goal Prompt: S1-H0-RUN

Revision: 2 - iterative blocker recovery and zero-based policy indexing.

Paste the text below into a fresh Codex Goal task.

---

GOAL ID: S1-H0-RUN

OBJECTIVE:
Preregister and execute exactly one complete Protocol101 H0 offline
evidence-producing batch: 7 fixed exit policies x 3 seeds x 5 governed folds
= 105 units. Produce and mechanically verify the raw unit/model artifacts,
then stop before G1-G8 aggregation or any interpretation of trading edge.

OWNER AUTHORIZATION:
This prompt is explicit owner authorization for this one offline H0 research
batch, including fold-local HGB fitting, calibration, threshold selection,
1.0x-noise validation, and strict serial replay. It authorizes no other
training hypothesis or downstream gate.

WORKSPACE AND INTERPRETER:
- workspace: /Users/gduby/Documents/autoresearch-trading
- interpreter: ~/.autoresearch-trading/runtime-venv/bin/python

READ BEFORE ACTING:
- v4/docs/PROTOCOL101_GOAL_SIZED_GATED_TRAINING_SYSTEM_2026_07_25.md
- v4/docs/PROTOCOL101_CANONICAL_V1_STAGE1_TRAINING_DESIGN.md
- v4/docs/PROTOCOL101_SCOPED_SYNCHRONIZATION_DECISION_2026_07_25.md
- v4/docs/PROTOCOL101_TRADER_CHARTER.md
- v4/docs/PROTOCOL101_STAGE1_OBJECTIVE_AND_GATES_PROPOSAL.md
- v4/docs/PROTOCOL101_G4_HOLDOUT_REVISION_2026_07_19.md
- v4/docs/PROTOCOL101_STATUS.md

STARTING CHECKPOINT:
- F0 foundation freeze is complete.
- machinery status must be:
  ready_for_separate_owner_approved_H0
- machinery blockers must be empty.
- runner-freeze status must be:
  core_runner_smoke_passed
- runner freeze hash:
  e1e7e7510aab474e6a280b3cbe7c095386c827121165ce9fd910d9ceae7d448a
- contract:
  protocol101-scoped-canonical-stage1-v1
- simulator:
  protocol101_serial_simulator_v4_account_continuity_fee_reserve

VERIFY THESE FILE SHA-256 HASHES BEFORE TRAINING:
- v4/docs/PROTOCOL101_GOAL_SIZED_GATED_TRAINING_SYSTEM_2026_07_25.md
  319def6ccb592abf6746651fc750062c02126fd404b29846b9f45f8e8ab5a63b
- v4/docs/PROTOCOL101_CANONICAL_V1_STAGE1_TRAINING_DESIGN.md
  db4fc337ec0ca56cfab79af4317273920812694d9be7edd6848bd38abdff10a0
- v4/docs/PROTOCOL101_SCOPED_SYNCHRONIZATION_DECISION_2026_07_25.md
  5c5a44e6bb4053276ef60b710a3788d1ee5ed9930a0aa8f10a4f45bbbe31417d
- v4/docs/PROTOCOL101_TRADER_CHARTER.md
  3dbc1cf45e2200b7fd789c92be7e927c714b476b4b45aec95b5e0d9686c1ed66
- v4/docs/PROTOCOL101_STAGE1_OBJECTIVE_AND_GATES_PROPOSAL.md
  d1528db501ecac1ff6f1ad9e29b1f2199b0de0a3b247ab47bb68342550ccd760
- v4/docs/PROTOCOL101_G4_HOLDOUT_REVISION_2026_07_19.md
  ff554ef5dd34086cd09954b906bbb5ce8455fa467776ae83906b01f80d565da5
- v4/audit/autoresearch/protocol101_scoped_canonical_stage1_machinery_readiness/summary.json
  1190d65ee93e0304cce49c9313b010b414c15bc223252592e8ed2a33b0b23184
- v4/audit/autoresearch/protocol101_scoped_canonical_stage1_plumbing_smoke_validation/runner_freeze.json
  c530760e2e34bacc26680b20200add5075930f650f724487467b59ded8b035ea

FROZEN H0 MODEL-FACING FEATURES:
1. spx_vwap_gap_points
2. spx_vwap_gap_bps
3. spx_vwap_gap_over_session_range
4. session_range_bps
5. momentum_5m_bps
6. momentum_15m_bps
7. momentum_5m_over_session_range
8. momentum_15m_over_session_range
9. omar_clipped_neg3_pos3
10. vwap_side_alignment_flag
11. omar_side_alignment_flag
12. momentum15_side_alignment_flag

No D-family composites, internal Greeks, direct option-price paths, VIX
changes, raw quote microstructure, volume/OI, vendor Greeks, labels, path
data, or fill data may enter H0 model scoring.

OTHER FROZEN INPUTS:
- accepted registry hash:
  6c656cfb2caeaab1d03e78eee39a164f7d169709abeaec5b0c490f1b1e29f0fe
- fold governance hash:
  c02c19feafbac7888a2317ddd7ef6753888b2b10a704d230355b622eb350b920
- 301 accepted sessions; 271 fold-eligible sessions
- 5 chronological expanding-window folds; 1-session embargo; no shuffling
- protected holdout 2025-05-16 through 2025-06-30 excluded
- seven fixed exit shapes, represented by frozen zero-based policy indices
  0 through 6
- seeds 42, 43, and 44
- bounded HistGradientBoosting regressor
- payoff / fee-adjusted return-on-premium target, never win probability
- fold-local training-tail calibration and threshold selection only
- 1.0x measured-divergence noise is primary; 0x/0.5x/2.0x are diagnostics
- k=2 action dead-band and k=2 slot-margin gate
- deterministic score/strike-index/right-index ordering
- score-independent nearest-ATM fallback
- boundary-stable intersection guards
- pessimistic fills
- $3.00 round-trip fee; $2.60/$4.00 sensitivities
- one account, one contract, $10,000 starting cash
- 5% session-starting-equity daily stop and forced flat

PRE-RUN CHECKS:
1. Verify all checkpoint states and SHA-256 hashes above.
2. Verify no H0 runner process is already active.
3. Verify this output directory is absent:
   v4/audit/autoresearch/protocol101_scoped_canonical_stage1_h0_attempt001
4. Verify the runner plan still resolves to H0, 12 exact features, 7 policies,
   3 seeds, 5 folds, the frozen registry/fold/simulator hashes, and no blockers.
5. Do not read protected-holdout, recorder, parity-confirmation, or sealed
   market evidence while performing these checks.

If any check fails or the output directory already exists, enter the BLOCKER
RECOVERY LOOP below. Do not abandon the goal after the first failed check.

ALLOWED WORK:
- Execute the exact H0 command below.
- Monitor the same running process until it completes.
- Read only the generated H0 artifacts needed to verify counts, schemas,
  timestamps, paths, and hashes.
- Compute mechanical completeness/hash checks that do not calculate or
  interpret G1-G8.
- Diagnose and repair non-substantive plumbing blockers needed to complete this
  same H0 batch.
- Modify paths, zero-based index mappings, readiness adapters, deterministic
  resume behavior, artifact writers/verifiers, imports, environment wiring,
  and focused tests when necessary.
- Regenerate affected smoke/readiness/freeze artifacts after a mechanical code
  repair, proving that the scientific contract remains unchanged before
  training resumes.

EXECUTION COMMAND:

~/.autoresearch-trading/runtime-venv/bin/python \
  -m v4.scripts.run_protocol101_scoped_stage1_hgb_runner \
  --mode train-hypothesis \
  --hypothesis H0 \
  --out-dir v4/audit/autoresearch/protocol101_scoped_canonical_stage1_h0_attempt001 \
  --owner-approved-offline-training

Do not add --force.

BLOCKER RECOVERY LOOP:
Continue working toward the original 105-unit outcome until it completes or a
genuinely substantive owner decision is required.

For every blocker:
1. Classify it as transient, mechanical_non_substantive,
   scientific_contract, protected_action, or external_owner_required.
2. Record the diagnosis and evidence under:
   v4/audit/autoresearch/protocol101_scoped_canonical_stage1_h0_recovery/
3. Retry transient failures with bounded backoff.
4. For mechanical_non_substantive failures, implement the narrowest repair,
   add or update focused tests, run them, rerun readiness/smoke/freeze checks,
   and retry S1-H0-RUN.
5. Do not preserve results produced by defective machinery as valid evidence.
   Keep them append-only, write a void/invalidation record, and use either a
   preregistration-preserving resume path or the next immutable attempt
   directory.
6. Do not mark the goal blocked merely because one attempt, test, process, or
   preflight check failed.
7. Ask the owner and wait only if the required resolution would change a
   frozen scientific/trading input, access protected data, spend money, contact
   a broker, or alter paper/real-money state.

Mechanical recovery must not change the H0 feature list, data membership,
folds, embargo, seeds, policy semantics, model family, target, calibration
semantics, noise assumptions, guards, fills, fees, simulator economics, gates,
or protected-data boundary.

INTERRUPTION AND RESUME RULE:
If the process is still running, wait for it. If Codex or the process exits
after writing partial unit evidence, preserve the directory. Before resuming,
verify that the original preregistration and all frozen hashes still match.
The runner must not rewrite preregistration after result artifacts. If current
resume behavior would do that, repair and test resume behavior as a
mechanical_non_substantive blocker, invalidate any tainted partial attempt, and
continue under a clean immutable attempt. Never use --force to overwrite
evidence.

REQUIRED OUTPUT:
v4/audit/autoresearch/protocol101_scoped_canonical_stage1_h0_attempt001/
- runner_plan.json
- preregistration.json
- progress.json
- units/policy{0..6}/seed{42,43,44}/{five folds}/summary.json
- one model.pkl beside every unit summary
- summary.json

MECHANICAL COMPLETION CHECKS:
- runner_plan status is ready with blockers [].
- runner_plan and preregistration select H0 and exactly the 12 listed features.
- preregistration was written before every unit result.
- exactly 105 unique policy/seed/fold unit summaries exist.
- exactly 105 model.pkl files exist.
- policy indices are exactly 0-6; seeds exactly 42/43/44; every pair has
  5 folds.
- every model hash recorded by its unit summary matches the file.
- every unit-summary hash recorded by the batch summary matches the file.
- progress reports completed_units=105 and total_units=105.
- batch summary reports:
  - schema_version Protocol101ScopedStage1HGBHypothesisUnitsV1
  - status unit_execution_complete_pending_preregistered_gate_aggregation
  - contract_id protocol101-scoped-canonical-stage1-v1
  - hypothesis H0
  - unit_count=105 and expected_unit_count=105
- batch code hashes match the preregistered/frozen runner code hashes.
- protected_holdout_read=false.
- recorder_or_confirmation_data_read=false.
- broker_endpoint_called=false.
- paper_submit_allowed=false.
- paid_data_downloaded=false.
- promotion_or_default_changed=false.
- runtime_or_launchd_changed=false.
- real_money_path_changed=false.
- research_model_training_executed=true is expected and authorized.

PASS CONDITION:
S1-H0-RUN is complete only if every mechanical completion check passes and the
terminal batch status is exactly
unit_execution_complete_pending_preregistered_gate_aggregation.

FAILURE ROUTING:
- Transient or mechanical blocker: diagnose, repair, verify, and retry within
  this goal.
- Existing or partial output: preserve it and use the interruption/resume rule.
- Defective or mechanically invalid output: mark it void; do not count it as
  H0 evidence; repair and rerun under a clean immutable attempt.
- Scientific-contract change, protected action, paid data, broker access, or
  paper/real-money change: ask the owner and wait without redefining success.
- After owner input, continue this same goal unless the owner explicitly
  cancels or replaces it.

FORBIDDEN WORK:
- Do not run the G1-G8 gate aggregator.
- Do not inspect, summarize, rank, or interpret PnL, z-scores, drawdown,
  calibration, trade frequency, policy quality, feature importance, or edge.
- Do not run H1, H2, H3, a conservative batch, G9, final fit, or holdout.
- Do not read the protected holdout or recorder/parity/sealed evidence.
- Do not change contracts, model-facing features, data membership, labels,
  thresholds, folds, seeds, policy semantics, fees, guards, fills, simulator
  economics, nulls, heuristic, or gates. Narrow source/test changes are allowed
  only for logged mechanical_non_substantive recovery.
- Do not call IBKR or any broker endpoint.
- Do not submit paper orders, download paid data, promote a model, edit
  runtime/default/launchd state, or touch any real-money path.

FINAL RESPONSE:
Report only:
- S1-H0-RUN complete, or the exact owner/external action still required;
- terminal status;
- 105/105 unit and model counts;
- preregistration hash and batch summary hash;
- frozen contract/fold/registry/simulator identifiers;
- side-effect audit;
- exact blocker if incomplete;
- next allowed goal ID: S1-H0-GATE, but do not start it.

STOP LINE:
Stop after raw H0 evidence and mechanical completeness verification. Do not
aggregate G1-G8, interpret the result, or begin the next goal. Before that stop
line, iterate through transient and non-substantive mechanical blockers until
the 105-unit H0 evidence outcome is genuinely complete.
