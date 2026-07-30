# Protocol101 Goal Prompt: FT1C2 Gate, Audit, And Selection Independent Reacceptance

Run this Goal in a fresh Codex task or subagent that did not implement FT1C or
FT1C1 and did not perform the first FT1C independent acceptance.

---

GOAL ID:

`FT1C2-GATE-AUDIT-SELECTION-INDEPENDENT-REACCEPTANCE`

OBJECTIVE:

Independently accept or reject the repaired final pre-training Stage-1 judging
stack. Build a fresh audit-local oracle and new synthetic campaigns. Prove that
the ten first-acceptance defects are actually closed, that the new authority
chain has an external root of trust rather than merely more self-hashes, and
that every original FT1C requirement still holds.

This is independent acceptance only. Do not repair production, execute or
inspect real campaign economics, train models, select a real row, run seed
45/G9, open protected or sealed evidence, contact a broker, or change runtime.

WORKSPACE AND INTERPRETER:

```text
workspace: /Users/gduby/Documents/autoresearch-trading
interpreter: ~/.autoresearch-trading/runtime-venv/bin/python
```

READ BEFORE ACTING:

- `v4/docs/protocol101/training/README.md`
- all signed Stage-1, G4, G8, and regimen-repair contracts
- FT1C producer, first independent acceptance, and FT1C1 repair Goal files
- all top-level artifacts under:
  `v4/audit/autoresearch/protocol101_full_trader_stage1_gate_audit_selection_repair_attempt002/`
- the immutable first rejection packet and its exact reproducers
- all 12 repaired production/test files
- independently accepted simulator-v5, runner-v5, and
  reference/multiplicity packets

REQUIRED PRODUCER ROUTE:

```text
gate_audit_selection_rejection_repair_complete_pending_reacceptance
```

FROZEN REPAIR ARTIFACT HASHES:

```text
b2f8bc23784055a820e8561ae384cf483ee306c7e3e9c55b7ab9f122127dfc07 preregistration.json
bd2ddca23db9cd7584cd37ee75d8407d89e32cf0b5051f77569f81fa6be81b37 input_inventory.json
06eb30739893679d4ab7514336b9fa2cdb2f6da4c432b4e14a0eff987642c28d defect_contract.json
b74827bc8ab60ddc2452c465ec7ecb302584013ed58cc64c4239bc651a9a0d02 after_reproduction.json
97b96073d86562e7e7588ba914e08de1f1871c4b9020d68ffc997db567416f7c repair_manifest.json
b7d3631dbfc35958817fe23943127c6c73b0f09b85c7eab84e58f7d9ea03ae5b authority_contracts.json
165fef0efc0b78bfafd18206de90c0b36ac01b44d4535b10f0a6892d54489497 test_results.json
213c152a669b0f1777fcdb8519999cc3a77d6ea994b7be8d26bc3a670c84537f summary.json
335c585786805c342d9f87a72b907e8a629306f0f57f7a2d874aa9376270e8c2 routing_decision.json
81ca861c13f612be69c38182cee78f28e4144ead36d333d4e2be398c3ae57427 report.md
c5434b6de6964a147b544fc2552f8f853ffa33852ec13a032c25d2777c0a7aef hashes.sha256
```

Paths above are relative to the repair output directory.

FROZEN REPAIRED SOURCE HASHES:

```text
fc65d9ab835dad8d702aac67fff5e972644ea36be9e877c6cad597ab608f2536 v4/model/protocol101_stage1_gate_contract.py
3181577670f26317c16b8405caa8f2d2676a201003621fa9cfcc9708e1c372a4 v4/scripts/run_protocol101_scoped_stage1_gate_aggregator.py
e08a954ca021ec56dc14e00f1d2ac76f66446b435344b1cdd5f0bbc10506c5d5 v4/scripts/run_protocol101_scoped_stage1_independent_audit.py
1f1622c7bd4b02f523d168ae8b65b5ad9338f3dd3f5d7fd487119f711468a338 v4/scripts/run_protocol101_stage1_cross_hypothesis_selection.py
c6dd10405292b7084e9cf7addfe3b416ca0631efb029cb291dbc66deb8808241 v4/scripts/run_protocol101_stage1_autoresearch_graph.py
edfd71b02d33c44345ea1ccd265686536d4ffd012b5573734d53c2ae317849e7 v4/scripts/run_protocol101_full_trader_stage1_gate_audit_selection_validation.py
7f76467d878c26d82d1b28b5ffe774250891b582e2f6106370143394fc2cbcdf v4/tests/test_protocol101_stage1_gate_contract.py
4d0d1155159a072d008ef7e170a877ec8b84db936726932a050a0267351a53bf v4/tests/test_protocol101_scoped_stage1_gate_aggregator.py
ecacd3575d0bc76a0091c68c1087ea66efee1b417fd43cf4963d5531adfb01c9 v4/tests/test_protocol101_scoped_stage1_independent_audit.py
ae1c5f875d3f72abd597d430789e5b3065ac9b4b0c86c349e0d88daa5c0b7bfa v4/tests/test_protocol101_stage1_cross_hypothesis_selection.py
eeb5ffca8b273d9d18361b76e7ab16e880d27117d0ab4504dd46cda8b005cffc v4/tests/test_protocol101_stage1_autoresearch_graph.py
d5862a229f58ff4db2246e662214dfd7c4ff19ea9c0c06ef45bc821e83869a3a v4/tests/test_protocol101_full_trader_stage1_gate_audit_selection_validation.py
```

FROZEN FIRST-REJECTION EVIDENCE:

```text
202b3d705ccff065a2ebff19ffd168fc06a4d70d2fb1f0f1d8d773ea1adbf292 independent_oracle.py
cf7a179c259bead0b83c9fb05bbf8a4ecd3813e5211a0c9c8f3d420d347987da mutation_matrix.csv
c5f7240c9385ccb6266cb09a06787a68391fa122cb08b02335f284fee701b377 reproducers/reproducers.json
71981bbfda2d3afb5d45964e035be2d95f3038c0fe1f2e21049a25b30bc29ffe hashes.sha256
```

These are requirements and historical evidence only. Do not import or copy the
first independent oracle as the new oracle.

OUTPUT DIRECTORY:

```text
v4/audit/autoresearch/
protocol101_full_trader_stage1_gate_audit_selection_independent_acceptance_attempt002/
```

This is the only writable directory. Production and producer packets are
read-only for this Goal.

INDEPENDENCE CONTRACT:

Create all verifier source and fixtures under the output directory.

The verifier must not import:

- the producer validation runner;
- producer tests or fixture builders;
- first-acceptance oracle code;
- gate-contract calculation helpers;
- aggregator result helpers; or
- producer expected-output helpers.

It may invoke the public aggregator, production audit, selector, and graph as
black-box systems under test. Independently implement gate formulas, authority
bindings, routes, and selection. Record source similarity/import audits.

PREREGISTRATION:

Before invoking systems under test or reading producer result values beyond
hashes and inventory, write and hash:

```text
preregistration.json
source_inventory.json
independent_oracle_manifest.json
progress.json
```

Freeze:

- two or more fresh 420-unit campaign designs;
- all expected gates/routes;
- exact numeric comparison behavior;
- every authority-root attack;
- every mutation and mutation order;
- minimum mutation count;
- tolerances;
- all-or-nothing acceptance;
- forbidden actions; and
- terminal routes.

INDEPENDENT BATTERY:

## A. Integrity and implementation audit

1. Verify all repair artifacts, source hashes, and immutable rejection hashes.
2. Verify accepted simulator-v5, runner-v5, and reference/multiplicity source
   and packet hashes remain unchanged.
3. Parse source independently for v4 replay edges, hard-G8 conjunctions,
   missing hard gates/controls, risk-utility ranking, selector use of uncovered
   values, self-authorizing graph edges, or permission escalation.
4. Record pre/post hashes and prove no production write.

## B. Fresh campaigns and exact oracle

Create at least two new complete campaigns:

```text
H0-H3 x P0-P6 x seeds 42-44 x folds 1-5 = 420 units each
```

Use economics and edge patterns different from both producer batteries and the
first independent fixtures. Build authority receipts from a separate
audit-local controller, not with production builder helpers.

Independently prove exact agreement for:

- G1-G7;
- G8 required report-only;
- maxT, D1, D5, D6;
- all 28 rows and all seeds/folds;
- simulator-v5 continuous replay;
- fixed plain median-seed strict-serial PnL selection;
- exact tie order;
- at most one selection;
- all stop/routing outcomes; and
- G9 false/not-run.

## C. Ten exact rejection cases

Recreate all ten first-rejection mutations independently and require:

```text
G4-CALMAR-AT: pass
G6-AT: pass
FRESHNESS-changed_model_hash: reject
FRESHNESS-old_reference_schema: reject
FRESHNESS-old_rank_payload: reject
FRESHNESS-old_economics_marker: reject
CONTROL-D5-CHANGED-HASH: reject
CONTROL-D6-CHANGED-HASH: reject
SELECTOR-CHANGED-FROZEN-PAYLOAD: campaign_invalid_no_selection
SELECTOR-MALFORMED-FREEZE-HASH: campaign_invalid_no_selection
```

Do not rely on the repair packet's compatibility adapter as proof.

## D. Authority root-of-trust attacks

The new authorities must not reduce to attacker-controlled self-hashes. Test at
least:

1. mutate unit model/provenance and reseal unit plus campaign only;
2. mutate unit and rebuild execution authority plus its hash;
3. mutate unit, packet, execution authority, and every embedded self-hash while
   retaining the graph-frozen execution receipt/root;
4. substitute an authority from another complete 420-unit campaign;
5. reorder, duplicate, omit, or add an authority unit;
6. change authority schema, route, campaign namespace, source hashes, or count;
7. pass a valid-looking authority hash not present in the graph state;
8. mutate D1, D5, D6, reference, or maxT and reseal its local receipt only;
9. mutate a control and rebuild the control authority plus its hash while
   retaining the graph-frozen control receipt/root;
10. substitute a control authority from another campaign;
11. change graph node order or inject an authority before RUN;
12. omit owner authorization and inject completed RUN/authority receipts;
13. change owner authorization after authority generation;
14. use malformed, uppercase, truncated, padded, or non-hex hashes; and
15. replay a valid authority receipt under a different campaign identifier.

Acceptance requires the public graph path to bind downstream expected authority
hashes to its frozen state. Supplying a matching mutable packet, authority, and
CLI hash directly must not be mistaken for owner/graph authorization. If direct
component APIs cannot establish graph authorization by design, record that
scope explicitly and prove the graph rejects the forged chain.

## E. Freshness and recursive forbidden-evidence attacks

Insert stale payloads at top level and nested within units, references,
controls, diagnostics, provenance, routes, and arbitrary lists:

- old schemas;
- `prior_ranking`;
- selected-candidate/promotion/G9/holdout fields;
- old H0-H3 economics;
- benchmark economics;
- `OLD_H0_H3_ECONOMICS`;
- stale namespaces and source hashes.

Try case variants and substring lookalikes. Require deterministic fail-closed
behavior for prohibited exact semantic fields/markers without rejecting normal
unrelated strings.

## F. Selector freeze attacks

Independently mutate each freeze-bound component:

- independent result;
- one row;
- campaign packet;
- execution authority;
- control authority;
- producer aggregation;
- independent aggregation;
- route;
- source hashes;
- row order/count;
- family completeness; and
- freeze schema/hash.

Try recomputing inner hashes while retaining the outer freeze and vice versa.
No uncovered value may influence eligibility, ranking, selected row, or route.

## G. Complete original battery

Repeat all original FT1C categories, including:

- every G1-G7 boundary below/at/above;
- G8 both sides with identical eligibility/ranking;
- maxT 999/1000 count boundary and invalid-family blocking;
- D1/D5/D6 independent blocking;
- pessimistic primary/noise invariants;
- nonfinite/missing diagnostics;
- identity mutations at every level;
- seed 45/protected/sealed/recorder blocking;
- ranking mutation immunity;
- route truth table;
- injected aggregator defects caught by production audit; and
- graph stopping before RUN without owner authorization and before G9 after
  synthetic selection.

Use at least 120 total deterministic/property mutations across multiple rows,
families, seeds, folds, controls, authority layers, and selector fields.

## H. Regressions

Run:

- compilation;
- the exact current 109-test FT1C/dependency suite;
- new independent oracle tests;
- fresh-process packet/order tests;
- all packet checksums; and
- final source hash recheck.

Do not run the whole repository suite.

ACCEPTANCE RULE:

All checks are all-or-nothing. Any scientific, provenance, authority, selector,
route, independence, regression, or packet-integrity failure rejects the
machinery. Do not repair production in this Goal.

REQUIRED OUTPUTS:

```text
preregistration.json
preregistration_freeze.sha256
source_inventory.json
independent_oracle.py
independent_oracle_manifest.json
test_independent_oracle.py
independent_campaign_manifest.json
authority_root_audit.json
independent_gate_results.json
mutation_matrix.csv
freshness_audit.json
selector_freeze_audit.json
producer_integrity.json
independence_audit.json
audit_acceptance.json
selection_acceptance.json
graph_acceptance.json
regression_results.json
acceptance_matrix.csv
acceptance_decision.json
summary.json
report.md
hashes.sha256
progress.json
```

Preserve exact deterministic reproducers for every failed case.

TERMINAL ROUTES:

Accepted:

```text
gate_audit_selection_machinery_independently_accepted
```

Rejected:

```text
gate_audit_selection_machinery_independent_rejection
```

HIGHEST ALLOWED CLAIM ON ACCEPTANCE:

```text
Protocol101 Full Trader Stage-1 pre-training machinery is independently
accepted; real campaign execution still requires explicit owner authorization.
```

No profitability, candidate, G9, holdout, learned-exit, transfer, paper, or
live-readiness claim is allowed.

NEXT BOUNDARY:

If and only if accepted, stop for the owner's explicit decision whether to
authorize the fresh 420-unit offline H0-H3 campaign. Do not execute it in this
Goal.
