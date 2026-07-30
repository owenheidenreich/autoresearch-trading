# Protocol101 Goal Prompt: FT1C Gate, Audit, And Selection Machinery Repair

Run this Goal in a fresh Codex task or subagent.

---

GOAL ID:

`FT1C-GATE-AUDIT-SELECTION-MACHINERY-REPAIR`

OBJECTIVE:

Repair the last pre-training Stage-1 foundation:

1. aggregate the fresh 420-unit H0-H3/P0-P6/seed/fold campaign under exact
   simulator-v5 evidence;
2. apply signed G1-G7 as hard gates and G8 as report-only;
3. apply the hard 28-row maxT control and D1/D5/D6 campaign controls;
4. independently reconstruct and audit every row;
5. compare all 28 accepted rows without utility tuning;
6. select at most one candidate by the signed plain-PnL rule; and
7. keep seed 45/G9 behind a separate spend-once authorization boundary.

Implement and validate this machinery on synthetic, audit-local packets only.
Do not fit campaign models, execute campaign/reference economics, inspect old
H0-H3 results, aggregate real rows, rank real rows, select a real candidate,
or spend seed 45.

WORKSPACE AND INTERPRETER:

```text
workspace: /Users/gduby/Documents/autoresearch-trading
interpreter: ~/.autoresearch-trading/runtime-venv/bin/python
```

READ BEFORE ACTING:

- `v4/docs/protocol101/training/README.md`
- `v4/docs/protocol101/training/contracts/PROTOCOL101_STAGE1_OBJECTIVE_AND_GATES_PROPOSAL.md`
- `v4/docs/protocol101/training/contracts/PROTOCOL101_G4_HOLDOUT_REVISION_2026_07_19.md`
- `v4/docs/protocol101/training/contracts/PROTOCOL101_G8_CALIBRATION_REVISION_2026_07_26.md`
- `v4/docs/protocol101/training/contracts/PROTOCOL101_STAGE1_REGIMEN_REPAIR_AMENDMENT_2026_07_26.md`
- `v4/docs/protocol101/training/contracts/PROTOCOL101_CANONICAL_V1_STAGE1_TRAINING_DESIGN.md`
- every top-level artifact under
  `v4/audit/autoresearch/protocol101_full_trader_entry_campaign_preregistration_attempt001/`
- every top-level artifact under
  `v4/audit/autoresearch/protocol101_full_trader_entry_runner_v5_core_independent_acceptance_attempt001/`
- every top-level artifact under
  `v4/audit/autoresearch/protocol101_full_trader_stage1_reference_multiplicity_independent_acceptance_attempt001/`
- the existing gate aggregator, independent audit, cross-hypothesis selector,
  autoresearch graph, and their tests

AUTHORITY ORDER:

When old prose or code conflicts:

1. the owner-signed regimen repair amendment supersedes simulator-v4 clauses;
2. the signed G4 revision supersedes the old drawdown gate;
3. the signed G8 revision makes G8 report-only globally;
4. the signed hard maxT control applies to the frozen 28-row family; and
5. the fresh campaign contract forbids old reference values, old economics,
   old ranking, and old namespaces.

Do not edit signed documents to resolve these already-governed conflicts.

FROZEN ACCEPTANCE INPUTS:

Require:

```text
d6211d8260fe43aadd30037da5d9df373bb53eec2d78ab49866f73dc52b4022f
  v4/audit/autoresearch/protocol101_full_trader_stage1_reference_multiplicity_independent_acceptance_attempt001/acceptance_decision.json

5a6582895ee23396386fc443f6e98dfba72376863d2456a305b8c5bfcf5eca11
  v4/audit/autoresearch/protocol101_full_trader_stage1_reference_multiplicity_independent_acceptance_attempt001/summary.json

58e5fd26ec885b4563b8bd3511e4075e8d41a498d4755f94ff240a45731e0af2
  v4/audit/autoresearch/protocol101_full_trader_stage1_reference_multiplicity_independent_acceptance_attempt001/maxT_independent_acceptance.json

b9247dd06a808adee8bdbbcdb0adf1cdca1b5f0e5536716ff5861473a0af93e1
  v4/audit/autoresearch/protocol101_full_trader_stage1_reference_multiplicity_independent_acceptance_attempt001/hashes.sha256

72cbe1443cbddeb5af6dbb09bf384d9ea479658a4cefacbcda5b9c8463600bd2
  v4/audit/autoresearch/protocol101_full_trader_entry_runner_v5_core_independent_acceptance_attempt001/acceptance_decision.json

b5a70e0dcc6df7c0bd95408a65d1561beaccb615c75ea978060eab15c60b41e0
  v4/audit/autoresearch/protocol101_full_trader_entry_runner_v5_core_independent_acceptance_attempt001/hashes.sha256

7a6f747718419041ca3ce9590fb192c64e915800f3dd0dafac5d9ffdc5ec03f0
  v4/audit/autoresearch/protocol101_full_trader_entry_campaign_preregistration_attempt001/campaign_contract.json

40c3fa07c6fc94aaafdb1abf2b454ede5567c92728f814c38870c8f0eed969c5
  v4/audit/autoresearch/protocol101_full_trader_entry_campaign_preregistration_attempt001/preregistration.json
```

Require routes:

```text
entry_runner_v5_core_independently_accepted
reference_multiplicity_machinery_independently_accepted
```

OUTPUT DIRECTORY:

```text
v4/audit/autoresearch/
protocol101_full_trader_stage1_gate_audit_selection_machinery_attempt001/
```

PRODUCTION SCOPE:

Authorized files:

```text
v4/model/protocol101_stage1_gate_contract.py
v4/scripts/run_protocol101_scoped_stage1_gate_aggregator.py
v4/scripts/run_protocol101_scoped_stage1_independent_audit.py
v4/scripts/run_protocol101_stage1_cross_hypothesis_selection.py
v4/scripts/run_protocol101_stage1_autoresearch_graph.py
v4/scripts/run_protocol101_full_trader_stage1_gate_audit_selection_validation.py
v4/tests/test_protocol101_stage1_gate_contract.py
v4/tests/test_protocol101_scoped_stage1_gate_aggregator.py
v4/tests/test_protocol101_scoped_stage1_independent_audit.py
v4/tests/test_protocol101_stage1_cross_hypothesis_selection.py
v4/tests/test_protocol101_stage1_autoresearch_graph.py
v4/tests/test_protocol101_full_trader_stage1_gate_audit_selection_validation.py
```

The gate contract and terminal validator may be created. Keep edits within this
allowlist and the output directory. Do not edit the accepted runner,
references, maxT engine, simulator, repair contracts, loader, dataset builder,
feature firewall, campaign contract, or signed documents.

PREREGISTRATION:

Before invoking comparisons, write and hash:

```text
preregistration.json
source_inventory.json
progress.json
```

Freeze:

- exact gate laws below;
- exact result schemas and required diagnostics;
- independent-audit responsibilities;
- selection and tie-break law;
- synthetic fixtures and expected routes;
- float tolerances;
- fail-closed conditions;
- changed-file allowlist;
- terminal routes; and
- forbidden actions.

No threshold, gate, diagnostic role, rank criterion, or route may change after
synthetic results are visible.

## A. Fresh campaign evidence contract

The gate stack may accept only the fresh namespace:

```text
protocol101_full_trader_stage1_entry_fresh_attempt001
```

Require:

- exactly H0-H3, P0-P6, seeds 42-44, folds 1-5;
- exactly 420 freshly fitted unit manifests when real execution later occurs;
- exact campaign, fold, registry, feature, simulator, schema, model,
  threshold, epsilon, source, and code hashes;
- exact simulator-v5 version in every replay payload;
- repaired source-exit and realized-exit clocks;
- all five validation folds on the frozen chronological/embargo grid;
- no duplicate or omitted unit/row/seed/fold/session/trade identity;
- primary evidence from 1.0x divergence noise and pessimistic executable fill;
- $3.00 fee primary with $2.60/$4.00 diagnostics;
- no old model, old null, old heuristic, old gate, old audit, or old selection
  packet; and
- no protected/recorder/sealed/G9 evidence.

All required artifacts must be immutable and hash-bound before aggregation.
Any missing or mixed v4/v5 unit invalidates the entire campaign.

## B. Exact G1-G8 interpretation

For each H/P row, first reconstruct each seed from its five validation folds
using continuous one-account simulator-v5 replay.

Per-seed evidence includes:

- fold and pooled fee-adjusted net PnL;
- continuous pooled strict-serial PnL, drawdown, equity, and Calmar;
- per-fold minimum equity and trades/session frequency;
- matched-null z;
- era/fold PnL;
- ECE and observations;
- fees, fill-edge band, and noise diagnostics;
- side/time exposure;
- top-trade/top-day/month concentration;
- churn and skipped opportunity;
- worst day and underwater duration;
- outcome buckets and harvest ratio;
- daily-breaker events; and
- seven-shape usage distribution.

Apply:

```text
G1:
  each seed has >0 fee-adjusted PnL on at least 4 of 5 folds
  and pooled PnL >0; row pass requires all seeds pass.

G2:
  median-seed pooled top-selection PnL z >=3.0 versus the exact matched
  random null for that H/P row.

G3:
  median-seed pooled fee-adjusted strict-serial PnL exceeds the freshly
  rebuilt fixed P5 heuristic on the same folds.

G4:
  every seed has pooled continuous strict-serial
  PnL / max drawdown >=1.0, and every seed/fold minimum equity >=$5,000.
  A positive-PnL zero-drawdown seed has infinite Calmar; nonpositive PnL
  cannot pass. Null-relative drawdown is report-only.

G5:
  all three seeds satisfy G1 and worst-seed matched-null z >=2.0.

G6:
  every seed has median test-fold PnL >=0 in every governed era.
  Any negative era median is a hard gate failure and must emit
  regime_bound_requires_owner_review; do not silently call it ordinary
  no-signal.

G7:
  every seed/fold averages 0.3 to 6.0 trades per session.
  Report the 3/day conservative rail; it is not an extra gate.

G8:
  compute/report frozen 10-bin payoff-score-to-win ECE for every seed.
  ECE <=0.10 is a benchmark only. G8 never changes eligibility, ranking,
  G9 eligibility, or holdout eligibility in this campaign.

G9:
  false/not-run throughout this machinery and initial campaign.
```

Do not refit, recalibrate, retune thresholds, or infer missing metrics during
aggregation.

## C. Campaign controls and eligibility

Require before any row can be eligible:

- D1 campaign diagnostic passes its signed three-part law;
- D5 identity-complete fixed heuristic is independently accepted;
- D6 authority receipt is exact and independently accepted;
- the joint maxT is valid across all 28 rows;
- the row has `p_FWER <=0.05` with exceedance count <=999;
- the same row also passes G2; and
- G1-G7 all pass.

G8 is present and reported but excluded from the hard conjunction.

Define:

```text
hard_gate_eligible = G1 & G2 & G3 & G4 & G5 & G6 & G7
                      & maxT & D1 & D5 & D6

multiplicity_adjusted_real_entry_signal = G2 & G5 & maxT
```

If the joint maxT is invalid, D1 fails, or a required global control is
missing, all 28 rows are ineligible and selection is blocked.

Diagnostics such as fill bands, noise agreement, concentration, churn, and
outcome buckets must be fully reported. Do not invent new hard thresholds for
them. Enforce only existing causal truths:

- 1.0x noise is primary;
- pessimistic executable fill is primary;
- evidence cannot become eligible by substituting the 0x-noise or favorable
  fill diagnostic; and
- nonfinite, incomplete, or identity-inconsistent diagnostics fail closed.

## D. Independent audit path

Repair the existing read-only audit into an implementation independent from
the producer aggregator:

- do not import aggregator result helpers;
- load the frozen units/references directly;
- independently deserialize repaired decisions;
- replay continuous seed streams through simulator v5;
- recompute every G1-G8 number, D1/D5/D6 binding, and maxT eligibility;
- compare unit identities, both clocks, trades, skipped events, folds,
  diagnostics, and hashes;
- publish every one of the 28 rows including failures; and
- bind a freeze only after exact agreement.

The production audit path remains read-only. A later fresh-agent Goal will
still independently accept both implementations.

## E. Selection

Selection may run only after every H0-H3 audit freeze is accepted and all
global controls are valid.

Eligible rows are ranked by:

```text
primary:
  descending median-seed fee-adjusted continuous strict-serial net PnL

tie-break:
  H0, H1, H2, H3 order, then P0 through P6
```

No tunable risk-adjusted utility, ECE, win rate, drawdown, feature count,
model complexity, or post-result preference may alter ranking after
eligibility.

Select at most one candidate.

Routes:

```text
selected_candidate_awaiting_owner_authorized_G9
  one or more eligible rows, one selected by frozen rule

multiplicity_adjusted_real_signal_requires_fixed_exit_binding_attribution
  no eligible row, but at least one row has G2 & G5 & maxT

regime_bound_requires_owner_review
  otherwise eligible/signal evidence is blocked specifically by G6

stage1_no_accepted_edge_stop_and_redesign
  no eligible row and no multiplicity-adjusted real entry signal

campaign_invalid_no_selection
  any artifact, global control, independent-audit, or maxT invalidity
```

The attribution route does not authorize learned-exit training. It only
authorizes a later bounded attribution Goal.

## F. Autoresearch graph and owner boundary

Repair the graph as a dependency/state machine, not an autonomous permission
escalator:

```text
owner execution authorization
  -> fresh 420-unit RUN
  -> real v5 matched null + D5 + D1 evidence
  -> frozen 20,000-replicate maxT
  -> G1-G8 aggregation
  -> independent audit
  -> model-free selection/routing
  -> STOP
```

G9/seed 45 is a separate written spend-once Goal and owner decision after a
selected candidate exists. Holdout, learned exits, transfer, and paper remain
separate later gates.

Without an explicit future owner authorization packet, graph execution must
stop before RUN and report:

```text
full_trader_stage1_machinery_ready_owner_execution_authorization_required
```

This Goal must not create an authorization token or execute the graph.

## G. Synthetic validation matrix

Build complete synthetic 28-row campaign packets with exact 420-unit geometry.
Use no real campaign values.

Required cases:

1. one fully eligible row with G8 above 0.10 is selected;
2. identical economics with G8 below 0.10 yields the same selection;
3. an otherwise eligible row failing maxT is ineligible;
4. maxT invalidity blocks all selection;
5. D1 failure blocks all selection;
6. D5/D6 mismatch blocks all selection;
7. each of G1-G7 independently fails the intended row;
8. a G6-only failure routes to owner review;
9. G2/G5/maxT with fixed-exit gate failure routes to attribution;
10. no adjusted signal routes to stop/redesign;
11. ranking chooses larger plain median strict-serial PnL;
12. exact economic tie uses H then P order;
13. ECE, win rate, drawdown, and model complexity cannot alter rank;
14. v4/mixed/stale namespace packets fail;
15. missing/reordered/duplicate H/P/seed/fold/unit/session/trade fails;
16. favorable fill or 0x noise cannot replace primary evidence;
17. nonfinite/incomplete diagnostics fail;
18. seed 45/G9 input is rejected;
19. protected/sealed/recorder input is rejected;
20. aggregator and production independent-audit path agree exactly;
21. an injected aggregator defect is caught by the audit; and
22. no-owner graph stops before RUN.

Use synthetic expected metrics generated independently of the aggregator.

## H. Tests and readiness

Run:

- compilation for all changed files;
- gate-contract, aggregator, audit, selection, graph, and validator tests;
- accepted runner-v5 core regressions;
- accepted reference/multiplicity regressions;
- simulator-v5, repair identity/artifact, loader, and firewall regressions.

Do not run the entire repository suite.

The no-economic readiness status on success is:

```text
gate_audit_selection_machinery_ready_pending_independent_acceptance
```

SELF-REPAIR LOOP:

Do not stop at the first ordinary implementation or test failure. Reproduce,
repair within the allowlist, rerun the narrow test, then rerun the focused
suite until success or a genuine owner blocker.

A genuine owner blocker is only:

- a signed-authority contradiction not resolved by the stated authority order;
- a frozen required input missing/hash-mismatched before work;
- a requested behavior requiring a new policy choice; or
- unavoidable need for campaign economics, protected/sealed evidence, broker
  access, or paid data.

Stale code, a failing test, long runtime, or implementation complexity is not
an owner blocker.

REQUIRED OUTPUTS:

```text
preregistration.json
source_inventory.json
implementation_manifest.json
changed_files.json
gate_contract.json
evidence_schema.json
gate_truth_table.json
audit_independence_report.json
selection_contract.json
graph_contract.json
synthetic_420_unit_manifest.json
synthetic_case_results.json
readiness_matrix.csv
test_matrix.csv
test_results.json
progress.json
summary.json
routing_decision.json
report.md
hashes.sha256
```

Write `hashes.sha256` last and cover every top-level output except itself.

TERMINAL ROUTES:

Success:

```text
gate_audit_selection_machinery_repair_complete_pending_independent_acceptance
```

Genuine owner blocker:

```text
gate_audit_selection_machinery_owner_blocked
```

Technical failure after bounded self-repair:

```text
gate_audit_selection_machinery_repair_failed
```

On success, the sole next phase is a separately written fresh-agent
independent acceptance Goal. Do not start it here.

FORBIDDEN:

- real campaign fit, score, references, replay, aggregation, audit, ranking,
  selection, or attribution;
- old H0-H3 economic inspection;
- threshold/feature/epsilon/gate tuning;
- seed 45 or G9;
- protected holdout or sealed/recorder evidence;
- learned exits;
- broker/API, paper submit, paid download, promotion/default, runtime,
  launchd, or real-money changes;
- signed-contract edits; and
- edits outside the authorized files and output directory.

HIGHEST ALLOWED CLAIM:

> Stage-1 gate, audit, and selection machinery repair complete; independent
> acceptance is still required.

