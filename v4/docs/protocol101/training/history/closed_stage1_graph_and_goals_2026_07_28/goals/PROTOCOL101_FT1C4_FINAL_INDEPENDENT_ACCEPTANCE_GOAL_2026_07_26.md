# Protocol101 Goal Prompt: FT1C4 Final Independent Acceptance

Run this Goal in a fresh Codex task or subagent that did not implement FT1C,
FT1C1, or FT1C3 and did not perform either prior FT1C acceptance.

---

GOAL ID:

`FT1C4-FINAL-INDEPENDENT-ACCEPTANCE`

OBJECTIVE:

Independently accept or reject the complete Protocol101 Full Trader Stage-1
pre-training machinery after the authority-root repair.

Use a newly written audit-local oracle, new synthetic 420-unit campaigns, and
a new root-chain attack harness. Verify the original judging contract, the ten
first-rejection repairs, the thirteen second-rejection repairs, and the strict
owner-rooted graph chain all at once.

This is acceptance only. Do not repair production, execute or inspect real
campaign economics, train models, select a real candidate, run seed 45/G9,
open protected or sealed evidence, contact a broker, or modify runtime.

WORKSPACE AND INTERPRETER:

```text
workspace: /Users/gduby/Documents/autoresearch-trading
interpreter: ~/.autoresearch-trading/runtime-venv/bin/python
```

READ BEFORE ACTING:

- `v4/docs/protocol101/training/README.md`
- all signed Stage-1, G4, G8, and regimen-repair contracts
- FT1C through FT1C3 Goal files
- FT1C producer, both independent rejection, and both repair packets
- independently accepted simulator-v5, runner-v5, and
  reference/multiplicity packets
- all 12 current production/test files in the frozen source list below

REQUIRED PRODUCER ROUTE:

```text
authority_root_chain_repair_complete_pending_reacceptance
```

FROZEN FT1C3 ARTIFACT HASHES:

```text
232e61ad4531f342acfa272c37af370986689779e9a1d3987e37408bb651e038 preregistration.json
c75d9eeebc47ddebbec65a618ca955dec1599e1978df2d46d16b7e9c2426d420 root_chain_contract.json
82add6d3b013a39e0364a7fbcdefa67b679fe989f6e7ff844975432f57f6db8b after_reproduction.json
a48a4e7caafee5eb0d33dbfaaf30dc0a380b2bf3de16d06eab27877cc9970cd5 root_attack_matrix.csv
89c6a3e8ec3e5a16df37365cf0c143225179a80e2b6717f6fa37d96aaf1a50ed repair_manifest.json
37f88aada51153e23d9b02ef4aa3c58de982347a5d2208d860c36a60faf2aa75 test_results.json
00754d18f68711ec0fe6416a86efd8e122e5e60f9e5cef5814cc85d52a7a9c04 summary.json
4e0d792c839b7148bc4d94acec39686a9f5213c633546f279b7c93a9b701298a routing_decision.json
694c3af275b7cd939dfdfa02c2d3de59a189019847491d6fccd3966dc5330e1e report.md
b592d6857f45344a526971a0532cca972f0376c147a03ddacf23d2cbd61936c7 hashes.sha256
```

Paths are relative to:

```text
v4/audit/autoresearch/
protocol101_full_trader_stage1_authority_root_chain_repair_attempt003/
```

FROZEN CURRENT SOURCE HASHES:

```text
fc65d9ab835dad8d702aac67fff5e972644ea36be9e877c6cad597ab608f2536 v4/model/protocol101_stage1_gate_contract.py
3181577670f26317c16b8405caa8f2d2676a201003621fa9cfcc9708e1c372a4 v4/scripts/run_protocol101_scoped_stage1_gate_aggregator.py
e08a954ca021ec56dc14e00f1d2ac76f66446b435344b1cdd5f0bbc10506c5d5 v4/scripts/run_protocol101_scoped_stage1_independent_audit.py
1f1622c7bd4b02f523d168ae8b65b5ad9338f3dd3f5d7fd487119f711468a338 v4/scripts/run_protocol101_stage1_cross_hypothesis_selection.py
97cc5ffa661171ce7547cc1dd789d570168781f88408e81350e276ef47c84d68 v4/scripts/run_protocol101_stage1_autoresearch_graph.py
744415207dee9dd3efbbc04a6e68b2d801cddc52b23be9ddd55680dfc449c711 v4/scripts/run_protocol101_full_trader_stage1_gate_audit_selection_validation.py
7f76467d878c26d82d1b28b5ffe774250891b582e2f6106370143394fc2cbcdf v4/tests/test_protocol101_stage1_gate_contract.py
4d0d1155159a072d008ef7e170a877ec8b84db936726932a050a0267351a53bf v4/tests/test_protocol101_scoped_stage1_gate_aggregator.py
ecacd3575d0bc76a0091c68c1087ea66efee1b417fd43cf4963d5531adfb01c9 v4/tests/test_protocol101_scoped_stage1_independent_audit.py
ae1c5f875d3f72abd597d430789e5b3065ac9b4b0c86c349e0d88daa5c0b7bfa v4/tests/test_protocol101_stage1_cross_hypothesis_selection.py
0601ac0766dbc81fb4dc1d6a3973a61d5f3ac8047dc71f2cf78faf9aa5d4e59a v4/tests/test_protocol101_stage1_autoresearch_graph.py
698d80934b123a2082f6b623b7fb693618393d54d03c259585d0d543821e8df4 v4/tests/test_protocol101_full_trader_stage1_gate_audit_selection_validation.py
```

HISTORICAL REJECTION ORACLE HASHES:

```text
202b3d705ccff065a2ebff19ffd168fc06a4d70d2fb1f0f1d8d773ea1adbf292 FT1C first independent oracle
4a75be67aa90cad8faa2949bb8fc2424ec6c1e43a5e892d89164b1b24a8ae279 FT1C2 second independent oracle
```

These are frozen historical requirements. Do not import either oracle.

OUTPUT DIRECTORY:

```text
v4/audit/autoresearch/
protocol101_full_trader_stage1_gate_audit_selection_independent_acceptance_attempt003/
```

This is the only writable directory. Production and every prior packet are
read-only.

INDEPENDENCE CONTRACT:

Create all verifier code and fixtures under the output directory.

Do not import:

- producer validation/fixture code;
- production gate-formula helpers;
- producer tests;
- either previous independent oracle;
- repair compatibility adapters; or
- producer expected results.

You may invoke public aggregator, production audit, selector, and graph APIs as
black-box systems under test. Independently implement every expected gate,
control, authority, chain, route, and selection result. Record source
similarity/import audits.

PREREGISTRATION:

Before systems-under-test execution or result-value inspection, write and hash:

```text
preregistration.json
source_inventory.json
independent_oracle_manifest.json
progress.json
```

Freeze at least:

- two fresh 420-unit campaigns;
- 28-row expected results;
- gate formulas and boundary tolerances;
- authority/chain schemas and expected routes;
- at least 220 mutations;
- mutation order;
- exact regression commands;
- all-or-nothing acceptance;
- forbidden actions; and
- terminal routes.

BATTERY A: ORIGINAL JUDGING CONTRACT

Using at least two fresh campaigns:

```text
H0-H3 x P0-P6 x seeds 42-44 x folds 1-5 = 420 units each
```

independently verify:

- exact simulator-v5 two-clock serial replay;
- exact 420-unit identities and provenance;
- G1-G7 hard;
- G8 required report-only with no eligibility/ranking effect;
- D1/D5/D6 and 20,000-replicate maxT hard;
- all 28 rows;
- production audit independent agreement;
- plain median-seed strict-serial net-PnL selection only;
- deterministic H0-H3/P0-P6 tie order;
- at most one candidate;
- exact stop/redesign/attribution/owner-review routes;
- seed 45/G9 false and unspent; and
- graph stops at the owner boundary before RUN and at STOP before G9.

Re-run the full original mutation categories across gates, controls, identities,
freshness, diagnostics, selection, routes, and reward-hacking attempts.

BATTERY B: FIRST TEN DEFECTS

Independently recreate and require correct closure of:

```text
G4-CALMAR-AT
G6-AT
FRESHNESS-changed_model_hash
FRESHNESS-old_reference_schema
FRESHNESS-old_rank_payload
FRESHNESS-old_economics_marker
CONTROL-D5-CHANGED-HASH
CONTROL-D6-CHANGED-HASH
SELECTOR-CHANGED-FROZEN-PAYLOAD
SELECTOR-MALFORMED-FREEZE-HASH
```

BATTERY C: OWNER-ROOTED CHAIN

Build the owner authorization and ordered receipt chain independently; do not
use production seal/build helpers to construct expected values.

Require:

1. invalid/missing owner authorization stops before RUN;
2. exact owner payload canonical hash is the root;
3. empty valid chain identifies RUN as next;
4. every exact prefix length zero through eight identifies the correct next
   node;
5. only a complete valid chain reaches STOP;
6. every receipt binds exact schema, node, route, namespace, execution ID,
   owner hash, parent hash, artifact hash, and receipt hash;
7. chain hash covers the complete ordered chain;
8. graph state exposes exact accepted bindings;
9. no unknown or downstream G9/holdout/paper node is accepted; and
10. no graph path executes commands or creates owner authorization.

Independently reproduce all thirteen FT1C2 failed attacks and require rejection
before STOP.

Then attack every owner field, chain field, receipt field, node position,
parent link, artifact binding, campaign identifier, route, schema, and hash.
Include:

- locally self-resealed descendants;
- fully rebuilt descendants under an unchanged owner root;
- mixed/spliced chains from different campaigns;
- wrong execution-provenance/control/aggregation/audit/selection artifacts;
- reordered, duplicated, omitted, extra, and unknown receipts;
- changed owner with old descendants;
- V1 receipt bags;
- malformed and noncanonical hashes;
- complete chain plus forbidden downstream nodes; and
- replay under another namespace/execution ID.

BATTERY D: INDEPENDENT AUDIT AND SELECTOR FREEZE

Verify that:

- production audit loads raw units/references/controls and replays v5;
- aggregator defects are independently caught;
- selector reads only verified freeze-covered values; and
- every mutation of result, row, packet, authority, aggregation, route, source
  hash, order, count, or freeze routes `campaign_invalid_no_selection`.

BATTERY E: REGRESSIONS AND INTEGRITY

Run:

- compilation for all 12 frozen files;
- exact current 109-test FT1C/dependency suite;
- complete current graph/validation tests (`29` expected at freeze time);
- new independent oracle tests;
- fresh-process packet/order/chain tests;
- all prior packet checksum manifests; and
- final source hash recheck.

Do not run the whole repository suite.

ACCEPTANCE:

All checks are all-or-nothing. Any formula, authority, chain, freshness,
selector, route, independence, regression, or integrity failure rejects the
machinery. Do not repair production.

REQUIRED OUTPUTS:

```text
preregistration.json
preregistration_freeze.sha256
source_inventory.json
independent_oracle.py
independent_oracle_manifest.json
test_independent_oracle.py
independent_campaign_manifest.json
independent_gate_results.json
mutation_matrix.csv
authority_chain_audit.json
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

Preserve exact reproducers for every failure.

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
accepted; fresh real campaign execution requires explicit owner authorization.
```

NEXT BOUNDARY:

If and only if accepted, stop. The next action is the owner's explicit decision
whether to authorize the fresh 420-unit offline H0-H3 entry campaign. Do not
execute it, spend seed 45/G9, open holdout data, start learned exits, or touch
paper/live paths in this Goal.
