# Protocol101 Goal Prompt: FT1C3 Authority Root Chain Repair

Run this Goal in a fresh Codex task or subagent.

---

GOAL ID:

`FT1C3-AUTHORITY-ROOT-CHAIN-REPAIR`

OBJECTIVE:

Repair the single root cause behind the 13 authority-root failures from FT1C2:
the Stage-1 graph currently accepts an unordered bag of individually
self-validating receipts instead of one strict receipt chain rooted in the
exact owner authorization and fresh RUN.

Replace that weak receipt model with an ordered, hash-linked, campaign-bound
chain. Preserve all accepted gate, audit, selector, simulator, feature,
multiplicity, and authorization policies.

This is graph machinery repair only. Do not execute or inspect real campaign
economics, train models, select a real row, run seed 45/G9, open protected or
sealed evidence, contact a broker, or change runtime.

WORKSPACE AND INTERPRETER:

```text
workspace: /Users/gduby/Documents/autoresearch-trading
interpreter: ~/.autoresearch-trading/runtime-venv/bin/python
```

READ BEFORE ACTING:

- `v4/docs/protocol101/training/README.md`
- all signed Stage-1 and regimen-repair contracts
- the FT1C, FT1C1, and FT1C2 Goal files
- all top-level artifacts and exact reproducers under:
  `v4/audit/autoresearch/protocol101_full_trader_stage1_gate_audit_selection_independent_acceptance_attempt002/`
- current graph, graph tests, FT1C validation runner, and validation tests

REQUIRED INPUT ROUTE:

```text
gate_audit_selection_machinery_independent_rejection
```

FROZEN REJECTION EVIDENCE:

```text
7b9b41e469ecfe7f7dade37771713c728f1227c5445351709493c3cad5fa1cab preregistration.json
4a75be67aa90cad8faa2949bb8fc2424ec6c1e43a5e892d89164b1b24a8ae279 independent_oracle.py
32e771aabc2bde8c42ed7e6e4dff609a0edb113456f3fc7db796f1e3f4fde572 authority_root_audit.json
358ce8eeeedaf148635f49a35129709790c2e6308b91838b81257edf2dc976ee mutation_matrix.csv
ed51bfcbc446271a3cfdd92af689e47034b2ac0bb0ae933114cb270b52b9cf98 acceptance_decision.json
143b6b3a6a7b673677ad0ce49161a3e348fc5c1f0809a47cec401feb076107f4 summary.json
c0c0e989a0bfb115d8c21e23d5be9278a87d750a8514a0bac03b4ec80dbdc28d report.md
00828699ae03c51974aa629737ff9c968b48af18aaf521e1196c7608329ba2c2 hashes.sha256
78449b6414e91fbbba54dd49bc11d81873141340ceec30c8f4b2b96b94f48d34 reproducers/reproducers.json
```

Paths are relative to the FT1C2 independent acceptance directory. Verify
before editing and after terminalization. The rejection packet is immutable.

FROZEN PRE-REPAIR SOURCE HASHES:

```text
c6dd10405292b7084e9cf7addfe3b416ca0631efb029cb291dbc66deb8808241 v4/scripts/run_protocol101_stage1_autoresearch_graph.py
eeb5ffca8b273d9d18361b76e7ab16e880d27117d0ab4504dd46cda8b005cffc v4/tests/test_protocol101_stage1_autoresearch_graph.py
edfd71b02d33c44345ea1ccd265686536d4ffd012b5573734d53c2ae317849e7 v4/scripts/run_protocol101_full_trader_stage1_gate_audit_selection_validation.py
d5862a229f58ff4db2246e662214dfd7c4ff19ea9c0c06ef45bc821e83869a3a v4/tests/test_protocol101_full_trader_stage1_gate_audit_selection_validation.py
```

AUTHORIZED PRODUCTION FILES:

```text
v4/scripts/run_protocol101_stage1_autoresearch_graph.py
v4/scripts/run_protocol101_full_trader_stage1_gate_audit_selection_validation.py
```

AUTHORIZED TEST FILES:

```text
v4/tests/test_protocol101_stage1_autoresearch_graph.py
v4/tests/test_protocol101_full_trader_stage1_gate_audit_selection_validation.py
```

Do not modify gate formulas, aggregator, production independent audit,
selector, simulator, runner, feature contract, signed documents, or any other
source/test file.

OUTPUT DIRECTORY:

```text
v4/audit/autoresearch/
protocol101_full_trader_stage1_authority_root_chain_repair_attempt003/
```

This is the only writable artifact directory.

PREREGISTRATION:

Before production edits, write and hash:

```text
preregistration.json
input_inventory.json
root_chain_contract.json
progress.json
```

Freeze:

- the 13 exact failed case IDs;
- the owner-root trust model;
- exact schemas and allowed fields;
- node order;
- parent-link rules;
- campaign/run identity rules;
- all comparison/hash behavior;
- allowed files;
- required tests;
- all-or-nothing repair; and
- terminal routes.

TRUST MODEL:

The owner-authorization document is the external governance trust root. This
Goal does not introduce a signing-key infrastructure or claim resistance to a
malicious user who can replace both the owner authorization and all local
files. It must prevent descendants from being changed, substituted, reordered,
or rebuilt without changing the frozen owner/run chain.

The graph must canonical-hash the exact owner-authorization payload and bind
every later receipt to that hash.

REQUIRED GRAPH V2 CONTRACT:

## A. Owner authorization root

Replace the loose authorization schema with a strict current schema requiring
at least:

```text
schema_version
campaign_namespace
campaign_execution_id
authorized = true
routing_decision
owner_signature
owner_decision_date
goal_sha256
preregistration_sha256
contract_bundle_sha256
seed_45_or_G9_authorized = false
```

- Require exact allowed fields and lowercase 64-hex hashes.
- Require a nonempty immutable `campaign_execution_id`.
- Canonical-hash the full payload as `owner_authorization_sha256`.
- Do not create or infer owner authorization in the graph.
- A changed owner payload with unchanged descendants must fail.

## B. Ordered receipt chain

Replace the receipt mapping/bag with one strict current receipt-chain schema:

```text
schema_version
campaign_namespace
campaign_execution_id
owner_authorization_sha256
receipts
chain_sha256
```

`receipts` is an ordered list and may contain only the exact completed prefix
of the canonical nodes:

```text
FRESH_420_UNIT_RUN
EXECUTION_PROVENANCE_AUTHORITY
REAL_V5_REFERENCES_D1_D5_D6
CONTROL_AUTHORITY
FROZEN_20000_REPLICATE_MAXT
G1_G8_AGGREGATION
INDEPENDENT_AUDIT
MODEL_FREE_SELECTION_ROUTING
```

Each receipt must use a strict schema and exact allowed fields including:

```text
schema_version
node
routing_decision
campaign_namespace
campaign_execution_id
owner_authorization_sha256
parent_receipt_sha256
artifact_sha256
receipt_sha256
```

Rules:

1. Receipt zero's parent is exactly `owner_authorization_sha256`.
2. Every later parent is exactly the preceding canonical receipt hash.
3. Every receipt carries the same namespace, execution ID, and owner hash.
4. Every artifact hash is lowercase 64-hex and binds that node's immutable
   output, including:
   - RUN campaign packet;
   - execution-provenance authority;
   - reference/D1/D5/D6 completion packet;
   - control authority;
   - maxT packet;
   - G1-G8 aggregation;
   - independent-audit freeze; and
   - selection/routing packet.
5. Every receipt hash covers all receipt fields with `receipt_sha256 = null`
   during canonical hashing.
6. `chain_sha256` covers the complete chain payload with
   `chain_sha256 = null`.
7. Unknown, extra, duplicate, omitted-middle, reordered, wrong-route,
   wrong-node, malformed, cross-campaign, or trailing injected receipts fail
   closed.
8. A valid prefix advances only to its exact next node.
9. An invalid chain never reaches `STOP`.

## C. Explicit external bindings

- The execution-authority receipt's `artifact_sha256` must be the exact
  externally frozen execution-provenance authority hash.
- The control-authority receipt's `artifact_sha256` must be the exact
  externally frozen control-authority hash.
- Later receipts must bind the exact aggregation, audit-freeze, and selection
  hashes used downstream.
- Do not treat a caller-supplied standalone matching hash as owner/graph
  authorization.
- Expose the accepted owner hash, execution ID, completed prefix, and exact
  artifact bindings in graph state for downstream verification.

## D. Exact routes

No valid owner authorization:

```text
full_trader_stage1_machinery_ready_owner_execution_authorization_required
```

Valid authorization, empty chain:

```text
owner_authorized_fresh_420_unit_RUN_pending_external_execution
```

Valid partial chain: remain on the same pending route and name the exact next
node.

Invalid authorization or chain:

```text
full_trader_stage1_receipt_chain_invalid
```

Complete valid chain:

```text
full_trader_stage1_selection_routed_STOP_before_G9
```

`RUN_executed` and `commands_executed` remain false because this Goal's graph
is a controller/evaluator, not campaign execution. G9 remains false.

REQUIRED REPRODUCTION:

Before repair, reproduce the exact 13 failures:

```text
AUTH-ATTACK-01-packet_resealed
AUTH-ATTACK-02-authority_rebuilt
AUTH-ATTACK-03-all_self_hashes
AUTH-ATTACK-04-cross_campaign_authority
AUTH-ATTACK-05-unit_grid
AUTH-ATTACK-06-authority_schema
AUTH-ATTACK-07-unknown_valid_hash
AUTH-ATTACK-08-control_local
AUTH-ATTACK-09-control_rebuilt
AUTH-ATTACK-10-cross_campaign_control
AUTH-ATTACK-11-node_order
AUTH-ATTACK-12-pre_run_injection
AUTH-ATTACK-14-owner_changed
```

After repair, recreate their semantic attacks against the V2 chain. Require
all 13 to reject before `STOP`. Preserve the original rejection packet and
oracle byte-for-byte.

ADDITIONAL REQUIRED ATTACKS:

Test at least:

- mutate each chain field and each receipt field;
- change each artifact hash independently;
- rebuild one receipt without rebuilding descendants;
- rebuild all descendants while retaining the owner-rooted prior hash;
- splice valid prefixes from two campaigns;
- omit first, middle, or last receipts;
- repeat each node;
- reverse or permute receipt order;
- append unknown receipts;
- valid prefix lengths zero through eight;
- wrong owner hash or execution ID at each depth;
- malformed/uppercase/non-hex/padded hashes;
- malformed chain hash;
- extra fields at every schema layer;
- stale V1 bag input;
- owner authorization field additions/removals;
- changed owner payload with old chain;
- valid chain with G9/seed45/holdout/paper nodes appended; and
- chain reuse under another campaign namespace or execution ID.

Validation must include at least 80 graph/root mutations beyond the 13 exact
reproducers.

SELF-REPAIR LOOP:

A failing test is not automatically terminal. Diagnose and repair within the
four authorized files, then rerun the relevant battery. Stop only when:

- all 13 reproductions and all required new attacks pass; or
- repair would change signed policy, require another production file, inspect
  real economics/evidence, or require an owner decision.

REQUIRED REGRESSIONS:

- compile the four authorized files;
- run the exact current 109-test FT1C/dependency suite;
- run updated graph and validation tests;
- run the validation runner in a fresh process;
- verify the FT1C1 and FT1C2 packets remain byte-identical; and
- verify all non-authorized source hashes remain unchanged.

REQUIRED OUTPUTS:

```text
preregistration.json
preregistration_freeze.sha256
input_inventory.json
root_chain_contract.json
before_reproduction.json
after_reproduction.json
root_attack_matrix.csv
repair_manifest.json
changed_files.json
test_results.json
summary.json
routing_decision.json
report.md
hashes.sha256
progress.json
```

Every output must state that no real campaign economics, training, selection,
seed 45/G9, protected/sealed evidence, broker, paper, promotion, runtime, or
launchd action occurred.

TERMINAL ROUTES:

Success:

```text
authority_root_chain_repair_complete_pending_reacceptance
```

Bounded blocker:

```text
authority_root_chain_repair_blocked
```

Owner-policy blocker:

```text
authority_root_chain_owner_decision_required
```

HIGHEST ALLOWED CLAIM:

```text
FT1C authority-root chain repair implemented; fresh independent reacceptance
required.
```

NEXT PHASE:

Only after success, write and run a fresh independent reacceptance Goal using a
new acceptance agent. Do not begin real campaign execution.
