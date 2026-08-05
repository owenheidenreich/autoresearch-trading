# Protocol101 Goal Prompt: FT1C Gate, Audit, And Selection Independent Acceptance

Run this Goal in a fresh Codex task or subagent that did not implement FT1C.

---

GOAL ID:

`FT1C-GATE-AUDIT-SELECTION-INDEPENDENT-ACCEPTANCE`

OBJECTIVE:

Independently accept or reject the final pre-training Stage-1 judging stack.
Using a fresh audit-local oracle and newly generated synthetic 420-unit
campaigns, prove that:

- only exact fresh simulator-v5 evidence is accepted;
- G1-G7 and maxT/D1/D5/D6 are hard;
- G8 is required report-only and cannot affect eligibility or ranking;
- the production audit independently catches aggregator defects;
- selection uses only plain median-seed strict-serial net PnL after
  eligibility;
- at most one candidate is selected;
- routes are exact; and
- the graph stops for owner execution authorization before real RUN and again
  before G9.

This is acceptance only. Do not repair production, execute campaign economics,
rank/select a real row, or spend seed 45.

WORKSPACE AND INTERPRETER:

```text
workspace: /Users/gduby/Documents/autoresearch-trading
interpreter: ~/.autoresearch-trading/runtime-venv/bin/python
```

READ BEFORE ACTING:

- `v4/docs/protocol101/training/README.md`
- all signed Stage-1, G4, G8, and regimen-repair contracts
- `v4/docs/protocol101/training/goals/PROTOCOL101_FT1C_GATE_AUDIT_SELECTION_MACHINERY_REPAIR_GOAL_2026_07_26.md`
- every top-level producer artifact under
  `v4/audit/autoresearch/protocol101_full_trader_stage1_gate_audit_selection_machinery_attempt001/`
- all 12 producer-changed files
- independently accepted runner-v5 and reference/multiplicity packets

FROZEN PRODUCER ARTIFACT HASHES:

```text
5cc6fb59469c1013385f1e25aa979446d02403556eaee078b4d9b65eb6d31b71 preregistration.json
6c0f597b0910b8a414f9b3eded29cebd358db0970b271ebb63c13105685f0278 implementation_manifest.json
0fbfe1372d05ede218857b3fc3f20540f4e8057b40a0afa8c63149fff8db5e9f changed_files.json
75fa7dbd0628ec5f7fa9289ec056498ee732cda9e8a40129cb2191b10b017035 gate_contract.json
e417e70dd35b054456862fd6f6bb8b4678633306e29ea5bbe91947c72979e7f0 evidence_schema.json
7ffd667b56670b02a7746d33304605fdcad0c11d467e8b8c53161cde1a4d097a gate_truth_table.json
99e0452a7c0c369a11bf9a67ca090bdd5ef5e43be1387f0aa4ab6cace320a7d8 audit_independence_report.json
14991e240bc8937719f1a581efd7d8c4cf76711f126ce36b80ccf53e718e3a58 selection_contract.json
8f77ba42d1299f40fa80a7c9c9796641495c5791e4fbf316aa57ccadb8f16c27 graph_contract.json
8fbded4e485d1f06475d62773c4a50d214c1840b29b0bf1484d1feb0e89e3b24 synthetic_420_unit_manifest.json
5c211b5b82ec9a860f9dc91ef97bf53615e3d840a81ec24c2a7ac37a17110ef6 synthetic_case_results.json
65c9afb0ccf74c55ef640cafbea791905db217c2bcd98307a3f2040a8703f931 readiness_matrix.csv
3f18fbed9d0713cdeca6190e390bd1604ef3314a6f1108e572a6059a78cec026 test_results.json
49ddbeba656555ed34ef66bd76b227286f0d6812ef0056f489c62047f9be218b summary.json
7b258ee104fffdae9d168b6d46f18900568f3b975326e70992a8bec7f0b41dca routing_decision.json
8f47a472472dbafd99b7da06d8fd59dad13672ce0d9d3c468bbbb281bc8a445d hashes.sha256
```

Paths above are relative to the producer output directory.

FROZEN SOURCE HASHES:

```text
9dc12adb949ccc1fdc12c05209945c655e525b04cc4caf09eb8f3221fa7253cf v4/model/protocol101_stage1_gate_contract.py
5479e22f3a728fe4d9c1f0ed68927672d610d7cf8ac20996d12d231426b3e561 v4/scripts/run_protocol101_scoped_stage1_gate_aggregator.py
663ac627e936ee3625410b1b05990827eb39dd2b4258d3d6adbc49aff65af05e v4/scripts/run_protocol101_scoped_stage1_independent_audit.py
4c224d540f6079560762a2eb75db190892b1b7a84800c60485ebc159d05e611d v4/scripts/run_protocol101_stage1_cross_hypothesis_selection.py
ea13ac4bb979ac46eec2975a3d10dc03162ad23cc8aea6c9266598da1b26369f v4/scripts/run_protocol101_stage1_autoresearch_graph.py
b60998516fc883226124cede5c929a9553ec0281b0c4286dd63ad42ff5a8cac0 v4/scripts/run_protocol101_full_trader_stage1_gate_audit_selection_validation.py
d74f415207d8efe1314cfd6fe6a4f2b13ba2d76e7d2ff11a6d127f9a5df76f1e v4/tests/test_protocol101_stage1_gate_contract.py
2c8f500b281f29c9cf2f2d208c84ff9dc46bb2a127595a1dec9e48bf97b2f4d1 v4/tests/test_protocol101_scoped_stage1_gate_aggregator.py
d5fe20e8d8b47d263761a6a8edd125f5df0b0b7e9a38fd15c50df39151d3ea66 v4/tests/test_protocol101_scoped_stage1_independent_audit.py
0b3952fea20a51eb7fc669dde7691a12b8f537eae3e19fe03baea50feaf6c031 v4/tests/test_protocol101_stage1_cross_hypothesis_selection.py
89dd6c6e63262e346b2f5519771eec3232facbf5e8d3a0c9671d0555d85c7a1a v4/tests/test_protocol101_stage1_autoresearch_graph.py
aab9bd40c45a7dfd8bbdbd86f12de492433e876b8724434111ecf6badc64c487 v4/tests/test_protocol101_full_trader_stage1_gate_audit_selection_validation.py
```

Require producer route:

```text
gate_audit_selection_machinery_repair_complete_pending_independent_acceptance
```

OUTPUT DIRECTORY:

```text
v4/audit/autoresearch/
protocol101_full_trader_stage1_gate_audit_selection_independent_acceptance_attempt001/
```

This is the only writable directory.

INDEPENDENCE CONTRACT:

Create verifier code and fixtures under the output directory.

The verifier must not import:

- the producer terminal validator;
- producer tests;
- producer fixture builders;
- the producer gate-contract calculation helpers; or
- producer expected-output helpers.

It may invoke the public aggregator, production audit, selector, and graph as
systems under test. It must independently implement gate formulas, controls,
routes, and selection in its audit-local oracle. It must not use producer
reports as proof.

Record verifier source/hash and run import/source-similarity checks.

PREREGISTRATION:

Before invoking the systems under test or reading producer result values beyond
hashes/inventory, write and hash:

```text
preregistration.json
source_inventory.json
independent_oracle_manifest.json
progress.json
```

Freeze fixtures, mutation order, tolerances, expected routes, all-or-nothing
acceptance, forbidden actions, and terminal routes.

INDEPENDENT BATTERY:

## A. Integrity and source audit

1. Verify every producer artifact/checksum and all 12 source hashes.
2. Verify accepted runner-v5, simulator-v5, repair, and
   reference/multiplicity sources are unchanged.
3. Parse source independently and reject any fresh v4 replay edge, hard-G8
   conjunction, missing maxT/D1/D5/D6 edge, risk-utility rank, or graph
   permission escalation.
4. Record pre/post hashes.

## B. New independent 420-unit fixtures

Create at least two complete campaigns from scratch:

```text
H0-H3 x P0-P6 x seeds 42-44 x folds 1-5 = 420 units
```

Use different PnL, drawdown, frequency, era, ECE, diagnostic, and maxT patterns
than the producer fixtures. Include repaired two-clock v5 intents and all
required diagnostics.

Independently calculate expected per-seed and per-row metrics. Do not call
producer gate helpers to make expected values.

## C. Exact gate oracle

Prove exact production agreement for:

- G1 all-seed 4/5-fold and pooled profitability;
- G2 median-seed z >=3;
- G3 median-seed pooled PnL above fresh D5;
- G4 all-seed Calmar >=1 and every fold equity >=5000, including
  zero-drawdown edge cases;
- G5 all seeds satisfy G1 and worst z >=2;
- G6 every seed/era median >=0 and the owner-review route;
- G7 every seed/fold frequency in [0.3, 6.0];
- G8 calculation/reporting with no eligibility or ranking effect;
- exact G9 false/not-run; and
- maxT/D1/D5/D6 conjunctions.

Independently recompute simulator-v5 continuous replay from intents and compare
trades, both clocks, PnL, drawdown, equity, skipped events, and fees.

## D. Mutation and reward-hacking battery

At minimum test:

1. each G1-G7 boundary immediately below, at, and above its threshold;
2. G8 on both sides of 0.10 with identical eligibility/selection;
3. maxT p/count at 999 and 1000 exceedances;
4. invalid maxT blocks the whole family;
5. D1, D5, and D6 each independently block;
6. favorable fill cannot replace pessimistic primary;
7. 0x/0.5x/2x noise cannot replace 1.0x primary;
8. missing/nonfinite diagnostics block;
9. v4, stale namespace, old reference, old rank, and old economics block;
10. missing/reordered/duplicated H/P/seed/fold/unit/session/trade identities
    block;
11. seed 45/G9 input blocks;
12. protected/sealed/recorder input blocks;
13. ECE, win rate, drawdown, feature count, and complexity mutations cannot
    alter eligible ranking;
14. larger plain median-seed PnL wins;
15. exact ties use H0-H3 then P0-P6;
16. at most one candidate is returned;
17. no adjusted signal routes stop/redesign;
18. G2/G5/maxT plus fixed-exit failure routes attribution;
19. G6-only failure routes owner review;
20. injected aggregator metric, gate, control, or row omission is caught by
    production audit; and
21. graph without owner authorization emits no commands and stops before RUN.

Use property/fuzz mutations across multiple rows, not only handpicked H0/P0.

## E. Audit and selection independence

Require the production audit:

- does not import aggregator result helpers;
- directly loads units/references/controls;
- replays simulator v5;
- publishes all 28 rows;
- exactly agrees with the independent oracle; and
- rejects every injected aggregator defect.

Require selector input only from accepted independent audit freezes. Reject
producer-only gate summaries, incomplete freezes, changed hashes, and partial
families.

## F. Graph boundary

Independently verify the dependency order:

```text
OWNER AUTH -> RUN -> REFERENCES/D1/D5/D6 -> maxT -> GATES
-> INDEPENDENT AUDIT -> SELECTION -> STOP
```

Without an authorization receipt, exact route must be:

```text
full_trader_stage1_machinery_ready_owner_execution_authorization_required
```

Even with a synthetic authorization receipt, graph must stop after selection
and may not run G9, holdout, learned exits, transfer, or paper.

## G. Regressions

Run:

- compilation;
- exact producer-focused 98-test suite;
- independent oracle tests;
- fresh-process packet/order tests; and
- source hash recheck.

Do not run the entire repository suite.

REQUIRED OUTPUTS:

```text
preregistration.json
source_inventory.json
independent_oracle.py
test_independent_oracle.py
independent_oracle_manifest.json
independence_audit.json
producer_integrity.json
independent_campaign_manifest.json
independent_gate_results.json
mutation_matrix.csv
audit_acceptance.json
selection_acceptance.json
graph_acceptance.json
acceptance_matrix.csv
regression_results.json
progress.json
acceptance_decision.json
summary.json
report.md
hashes.sha256
```

Write `hashes.sha256` last and cover every top-level output except itself.

TERMINAL ROUTES:

All rows pass:

```text
gate_audit_selection_machinery_independently_accepted
```

Any reproducible production defect:

```text
gate_audit_selection_machinery_independent_rejection
```

Missing/changed frozen input:

```text
gate_audit_selection_machinery_acceptance_input_blocked
```

Do not repair production. Preserve exact reproducers on rejection.

FORBIDDEN:

- production/test/doc/contract/producer/campaign edits;
- real campaign fit, reference execution, economic aggregation, ranking,
  selection, or attribution;
- old H0-H3 economic inspection;
- threshold/gate tuning;
- seed 45/G9;
- protected/sealed/recorder evidence;
- learned exits;
- broker/API, paper, paid download, promotion/default, runtime, launchd, or
  real-money changes; and
- acceptance based solely on producer tests/reports.

HIGHEST ALLOWED CLAIM:

> Stage-1 gate, audit, and selection machinery independently accepted.

On pass, stop for the owner decision on whether to authorize the fresh
420-unit campaign execution. Do not write or execute that authorization.

