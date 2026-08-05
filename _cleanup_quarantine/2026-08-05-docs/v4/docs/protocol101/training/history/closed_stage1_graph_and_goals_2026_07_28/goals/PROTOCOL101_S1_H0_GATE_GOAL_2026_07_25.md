# Protocol101 Goal Prompt: S1-H0-GATE

Revision: 1 - mechanical G1-G8 aggregation with iterative blocker recovery.

Paste the text below into a fresh Codex Goal task.

---

GOAL ID: S1-H0-GATE

OBJECTIVE:
Mechanically aggregate the frozen S1-H0-RUN evidence into the preregistered
G1-G8 results for all seven policy indices and three seeds. Verify the
aggregation packet and report its preliminary mechanical routing in plain
English, then stop before independent acceptance, retraining, H1, attribution,
G9, holdout access, or candidate action.

IMPORTANT INTERPRETATION:
This goal evaluates the completed H0 models; it does not train them.

An aggregator status of `pass` means the calculation completed without
artifact or governance blockers. It does not mean that any H0 policy passed
G1-G8. A valid negative H0 result is still successful completion of this goal.
Only S1-H0-AUDIT may independently accept or void the resulting verdict.

OWNER AUTHORIZATION:
This prompt authorizes only read-only use of the frozen H0 batch, null
reference, heuristic reference, era manifest, and the deterministic G1-G8
aggregator. It authorizes no model fit, threshold selection, protected-data
access, H1-H3 work, G9, broker action, or trading-state change.

WORKSPACE AND INTERPRETER:
- workspace: /Users/gduby/Documents/autoresearch-trading
- interpreter: ~/.autoresearch-trading/runtime-venv/bin/python

READ BEFORE ACTING:
- v4/docs/PROTOCOL101_GOAL_SIZED_GATED_TRAINING_SYSTEM_2026_07_25.md
- v4/docs/PROTOCOL101_STAGE1_OBJECTIVE_AND_GATES_PROPOSAL.md
- v4/docs/PROTOCOL101_G4_HOLDOUT_REVISION_2026_07_19.md
- v4/docs/PROTOCOL101_CANONICAL_V1_STAGE1_TRAINING_DESIGN.md
- v4/docs/PROTOCOL101_STATUS.md
- v4/audit/autoresearch/protocol101_scoped_canonical_stage1_h0_recovery/mechanical_completion_receipt.json

STARTING CHECKPOINT:
- S1-H0-RUN is mechanically complete.
- batch terminal status:
  unit_execution_complete_pending_preregistered_gate_aggregation
- unit summaries: 105
- model artifacts: 105
- contract:
  protocol101-scoped-canonical-stage1-v1
- simulator:
  protocol101_serial_simulator_v4_account_continuity_fee_reserve
- policies: frozen zero-based indices 0 through 6
- seeds: 42, 43, and 44
- folds: five chronological expanding-window folds
- G9 has not run and must remain false.

FROZEN IDENTIFIERS:
- H0 preregistration embedded hash:
  5b44588dfaa27d9c1d072c98f4008d00fbac426d1f421149949d97cb583c24cb
- H0 batch embedded summary hash:
  99baea50b40e3d456ecf1fe868637dcf2df1687a225b7c470ed9f082dbcd4e19
- acceptance registry hash:
  6c656cfb2caeaab1d03e78eee39a164f7d169709abeaec5b0c490f1b1e29f0fe
- fold governance hash:
  c02c19feafbac7888a2317ddd7ef6753888b2b10a704d230355b622eb350b920
- F0 runner freeze hash:
  e1e7e7510aab474e6a280b3cbe7c095386c827121165ce9fd910d9ceae7d448a

VERIFY THESE FILE SHA-256 HASHES BEFORE AGGREGATION:
- v4/docs/PROTOCOL101_GOAL_SIZED_GATED_TRAINING_SYSTEM_2026_07_25.md
  723779e6ba5338cf9fff67bfa28a909aabd57a11514fc9b00d7f0510f5fe77f1
- v4/docs/PROTOCOL101_STAGE1_OBJECTIVE_AND_GATES_PROPOSAL.md
  d1528db501ecac1ff6f1ad9e29b1f2199b0de0a3b247ab47bb68342550ccd760
- v4/docs/PROTOCOL101_G4_HOLDOUT_REVISION_2026_07_19.md
  ff554ef5dd34086cd09954b906bbb5ce8455fa467776ae83906b01f80d565da5
- v4/docs/PROTOCOL101_CANONICAL_V1_STAGE1_TRAINING_DESIGN.md
  db4fc337ec0ca56cfab79af4317273920812694d9be7edd6848bd38abdff10a0
- v4/audit/autoresearch/protocol101_scoped_canonical_stage1_h0_attempt001/summary.json
  f082efc9f12021102fb24f78104c7fe226229c0ddc214461a5e29f4f5a733964
- v4/audit/autoresearch/protocol101_scoped_canonical_stage1_h0_attempt001/preregistration.json
  d00fc3a934eed478c041cfdf8c09014db465f53360d63298b0da96f7c346bb53
- v4/audit/autoresearch/protocol101_scoped_canonical_stage1_h0_recovery/mechanical_completion_receipt.json
  5a5d3afcf17e0c4cbfaf838c6257723e9476033db2c2ba0e69b8da357d22e4a0
- v4/audit/autoresearch/protocol101_scoped_canonical_stage1_null_canary/summary.json
  8c6dece25b11768cf16362305089de2dce4a33976cf3a01457170e9e328ee333
- v4/audit/autoresearch/protocol101_scoped_canonical_stage1_heuristic_baseline/summary.json
  db48c65634305e581772edd71c55f682fbc24a7cfb931d3c72beb204e555adab
- v4/audit/autoresearch/protocol101_session_era_manifest/summary.json
  1c5c215c34aaae3dc672fd0511f7cd41c6fd27a578af85a23f2c2e2d3febc017
- v4/scripts/run_protocol101_scoped_stage1_gate_aggregator.py
  db31b6dbd2242015b21322ab9d96724f5d6bfa6e3fc8cc00d58abb9973451a2a
- v4/audit/autoresearch/protocol101_scoped_canonical_stage1_plumbing_smoke_validation/runner_freeze.json
  c530760e2e34bacc26680b20200add5075930f650f724487467b59ded8b035ea

PREREGISTERED GATE LAW:
- G1: all three seeds are profitable on at least 4 of 5 folds and pooled PnL
  is positive.
- G2: median-seed pooled PnL z-score is at least 3.0 versus the matched policy
  random null.
- G3: median-seed pooled fee-adjusted PnL exceeds the frozen policy-5
  heuristic baseline.
- G4: every seed has pooled Calmar at least 1.0 and every fold maintains at
  least $5,000 equity.
- G5: every seed passes G1 and the worst-seed null z-score is at least 2.0.
- G6: no seed has a negative median test-fold PnL in any governed era.
- G7: every seed/fold averages 0.3 through 6.0 trades per day.
- G8: every seed has weighted out-of-fold ECE at most 0.10.
- G9: false and unexecuted in this goal.

Seed aggregation, thresholds, gate definitions, null bands, heuristic
selection, fees, fills, noise scales, folds, and era definitions are frozen.
Do not reinterpret or change them after seeing results.

PRE-RUN CHECKS:
1. Verify every file and embedded hash above.
2. Verify the H0 summary still references exactly 105 unique unit-summary
   hashes and all match.
3. Verify every unit still references one model hash and all 105 model files
   match.
4. Verify the H0 preregistration predates every unit result.
5. Verify the policy/seed/fold grid is exactly 0-6 x 42/43/44 x five folds.
6. Verify all units use H0, the exact 12 H0 features, simulator v4, 1.0x
   primary validation noise, 0x/0.5x/2x diagnostics, $2.60/$3.00/$4.00 fees,
   and pessimistic fills.
7. Verify protected holdout and recorder/parity sessions are absent.
8. Verify null and heuristic references have status `pass`, the exact
   contract, registry hash, and fold hash.
9. Verify the gate output directory is absent and no H0 aggregator is active:
   v4/audit/autoresearch/protocol101_scoped_canonical_stage1_h0_attempt001_gates

If a check fails, enter the BLOCKER RECOVERY LOOP. Do not abandon the goal
after the first mechanical failure.

ALLOWED WORK:
- Read the frozen H0 batch, null, heuristic, era, and governance artifacts.
- Verify all unit, model, source, reference, and embedded hashes.
- Run the exact deterministic aggregator command below.
- Reconstruct chronological out-of-fold account continuity.
- Compute and report G1-G8, fee/fill/noise diagnostics, and the mechanical
  routing already defined by the aggregator.
- Diagnose, repair, test, and retry non-substantive aggregation plumbing
  failures without altering frozen evidence or gate meaning.

EXECUTION COMMAND:

~/.autoresearch-trading/runtime-venv/bin/python \
  -m v4.scripts.run_protocol101_scoped_stage1_gate_aggregator \
  --batch-dir v4/audit/autoresearch/protocol101_scoped_canonical_stage1_h0_attempt001 \
  --out-dir v4/audit/autoresearch/protocol101_scoped_canonical_stage1_h0_attempt001_gates

Do not add --force.

BLOCKER RECOVERY LOOP:
Continue toward a valid H0 G1-G8 packet until it completes or a genuinely
substantive owner decision is required.

For every blocker:
1. Classify it as transient, mechanical_non_substantive,
   scientific_contract, frozen_evidence_defect, protected_action, or
   external_owner_required.
2. Record diagnosis and evidence under:
   v4/audit/autoresearch/protocol101_scoped_canonical_stage1_h0_gate_recovery/
3. Retry transient failures with bounded backoff.
4. For mechanical_non_substantive failures, implement the narrowest versioned
   repair, add focused tests, and prove the G1-G8 definitions and frozen H0
   evidence are unchanged.
5. Never rewrite the H0 preregistration, unit summaries, models, batch summary,
   null reference, heuristic reference, or era manifest.
6. Preserve defective gate outputs append-only and mark them void. Retry using
   the next immutable gate-attempt directory; never use --force.
7. If aggregator source must change, preserve the preregistered source, create
   a versioned repair plus an aggregation-recovery addendum binding old/new
   hashes, run focused equivalence tests, and keep the independent AUDIT
   requirement. Do not silently replace the preregistered evaluator.
8. Ask the owner and wait if resolution would change any gate formula or
   threshold, unit evidence, feature, data membership, fold, seed, policy
   semantics, null, heuristic, era assignment, fee, fill, noise rule,
   simulator economics, protected access, paid data, broker access, or trading
   state.
9. Do not mark the goal blocked merely because the first aggregation attempt
   or verifier fails.

REQUIRED OUTPUT:
v4/audit/autoresearch/protocol101_scoped_canonical_stage1_h0_attempt001_gates/
- summary.json
- gate_results.json
- report.md

MECHANICAL COMPLETION CHECKS:
- output `summary.json` schema is
  Protocol101ScopedStage1GateAggregationV1.
- output status is `pass` and blockers are [].
- recorded batch/preregistration paths and file hashes match the frozen inputs.
- output summary self-hash verifies.
- `gate_results.json` exactly matches `summary.json.results`.
- results status is `complete` and hypothesis is H0.
- policy results contain exactly indices 0-6.
- each policy contains exactly seeds 42/43/44.
- each seed contains exactly five chronological fold rows.
- every policy has Boolean G1 through G8 results.
- G9 is false for every policy and `G9_executed=false`.
- fee sensitivities, pessimistic-fill edge band, 1.0x primary noise, and
  0x/0.5x/2x diagnostics are present.
- mechanical routing is one of:
  - candidate_selected_for_fresh_G9_confirmation_seed
  - real_signal_failure_attribution_required
  - hypothesis_rejected_no_accepted_real_signal
- a selected policy is present only when the mechanical route permits it.
- no model fit or threshold selection occurred.
- protected_holdout_read=false.
- recorder_or_confirmation_data_read=false.
- broker_endpoint_called=false.
- paper_submit_allowed=false.
- paid_data_downloaded=false.
- promotion_or_default_changed=false.
- runtime_or_launchd_changed=false.
- real_money_path_changed=false.
- no H1-H3, G9, holdout, attribution, independent audit, broker, paper, or
  promotion process was started.

PASS CONDITION:
S1-H0-GATE is complete when the aggregation packet passes every mechanical
completion check. Candidate eligibility is not required for this goal to
complete. A valid mechanically calculated rejection or attribution route is a
successful gate computation.

FAILURE ROUTING:
- Transient or mechanical blocker: diagnose, repair, verify, and retry inside
  this goal.
- Invalid frozen H0 evidence or a requested scientific/gate change: preserve
  evidence, ask the owner, and wait.
- Valid policy gate failures: report them as results, not implementation
  blockers. Do not tune around them.
- Defective gate output: preserve and void it, repair only non-substantive
  plumbing, and rerun in a clean immutable gate-attempt directory.

FORBIDDEN WORK:
- Do not fit or retrain any model.
- Do not select or tune a threshold.
- Do not modify the H0 batch, preregistration, models, units, features,
  policies, seeds, folds, null, heuristic, era rules, gate definitions, fees,
  fills, noise assumptions, or simulator economics.
- Do not independently accept or freeze the H0 verdict.
- Do not run H1, H2, H3, a conservative batch, attribution, G9, final fit, or
  holdout.
- Do not read recorder, parity-confirmation, sealed, or protected-holdout
  market data.
- Do not call IBKR or another broker endpoint.
- Do not submit paper orders, download paid data, promote a model, edit
  runtime/default/launchd state, or touch a real-money path.

FINAL RESPONSE:
Explain in plain English:
- that S1-H0-GATE completed or the exact owner/external action still required;
- whether the aggregation itself was mechanically valid;
- the mechanical route;
- whether any policy passed all G1-G8;
- a compact policy-by-policy G1-G8 pass/fail table;
- the most important numerical reason for each failing route, without changing
  or second-guessing the preregistered gates;
- selected policy index only if the packet mechanically selected one;
- gate summary hash and input batch/preregistration hashes;
- side-effect audit;
- next allowed goal ID: S1-H0-AUDIT, but do not start it.

Always label the result preliminary and pending independent S1-H0-AUDIT.
Do not call H0 accepted, rejected, promotable, or eligible as a final project
verdict in this producing task.

STOP LINE:
Stop after producing, mechanically verifying, and plainly summarizing the H0
G1-G8 packet. Do not accept the verdict, repair scientific shortcomings,
retrain, run attribution, start H1, run G9, open the holdout, or act on a
candidate.
