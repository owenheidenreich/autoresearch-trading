# Protocol101 Goal Prompt: FT1C5 Option A Controller Journal Repair

Run this Goal in a fresh Codex task or subagent.

---

GOAL ID:

`FT1C5-OPTION-A-CONTROLLER-JOURNAL-REPAIR`

OBJECTIVE:

Implement the owner-approved Option A trust boundary: a lightweight stateful,
append-only local controller journal that prevents the training/evaluation
pipeline from replacing its own accepted artifacts or selector freeze.

Repair exactly the five remaining independent-acceptance failures. Do not add
cryptography, signing keys, network services, or attacker-oriented machinery.

This Goal does not train models. Do not execute or inspect real campaign
economics, select a real candidate, run seed 45/G9, open protected or sealed
evidence, contact a broker, or change runtime.

WORKSPACE AND INTERPRETER:

```text
workspace: /Users/gduby/Documents/autoresearch-trading
interpreter: ~/.autoresearch-trading/runtime-venv/bin/python
```

OWNER DECISIONS:

Option A approval:

```text
v4/audit/autoresearch/
protocol101_full_trader_stage1_trust_boundary_owner_decision_2026_07_26/
owner_decision_packet.md
```

Frozen hash:

```text
e122d996686e30275075c398dd856d5d6c4b1da9d419b52df1d4d966c45a88be
```

Offline H0-H3 authorization is on file but is not executable inside this Goal:

```text
v4/audit/autoresearch/
protocol101_full_trader_stage1_offline_training_authorization_2026_07_26/
owner_authorization.md
```

Frozen hash:

```text
a12a21740a7d1113ab35aef24f0c4c747884fc21fa1cdc4061f6bcf707b083c4
```

READ BEFORE ACTING:

- `v4/docs/protocol101/training/README.md`
- all signed Stage-1, G4, G8, and repair contracts
- FT1C through FT1C4 Goal files
- Option A owner-decision packet
- FT1C4 independent rejection packet and exact reproducers
- current graph, selector, validation runner, and focused tests

REQUIRED INPUT ROUTE:

```text
gate_audit_selection_machinery_independent_rejection
```

FROZEN REJECTION EVIDENCE:

```text
99503ee745fc97d107190ebd6e782a4cd7602619756d4375939f4d1ce66e6961 preregistration.json
b44a5f28bbc51a557da315b34ec7cee58ba29b02c946ccf5932c3857273f73f3 independent_oracle.py
e6f126c19a270d00a9da0f15d0da608a30a800cc22e1979b567261f62480d1bc mutation_matrix.csv
34f366f3f72639e06db790e5833996e8a1e8ee507b44d770a9db6271d38bf4bf authority_chain_audit.json
67bcb82fb33018fdbccffe311b6f4a06a27dab35e07ac99489e2deb43db7f2db selector_freeze_audit.json
955df45a26ca85647353faf291c2fc9df058eb4790189c462866f0e11ac6f4f0 acceptance_decision.json
6f797e0997fe3b685cb7920f9f6c65d872a159670c7aac323697c8c09e113a56 report.md
d40ad997688a5e42257c7a88ea175c128613d27613b1988e86de50fde210f923 hashes.sha256
```

Paths are relative to the FT1C4 independent acceptance directory.

Exact reproducer hashes:

```text
1b2015f1a2d8e955c84e70a386cae0a6b47207fd95464fc89a16589deeecd59b AUTH-ATTACK-02-authority_rebuilt.json
a1e4cdeb0393e20b526f4f6964aa40379d33e2b0c9335dd55e4a6a67df09efc4 AUTH-ATTACK-03-all_self_hashes.json
9e81dd545edd61946f08968f82d7aa3720d51e721ad2e3794853d82953304c08 AUTH-ATTACK-09-control_rebuilt.json
22bd5326651ebe73f9f0fc6c9fe55041fb1a2d3021530cef4f00603a4311879c SEL-fully-resealed-result.json
177a275194a75b80defab89b0b4c9d19cc674c00fb68cf110689143e3a683800 SEL-fully-resealed-producer-independent.json
```

FROZEN PRE-REPAIR SOURCE HASHES:

```text
1f1622c7bd4b02f523d168ae8b65b5ad9338f3dd3f5d7fd487119f711468a338 v4/scripts/run_protocol101_stage1_cross_hypothesis_selection.py
97cc5ffa661171ce7547cc1dd789d570168781f88408e81350e276ef47c84d68 v4/scripts/run_protocol101_stage1_autoresearch_graph.py
744415207dee9dd3efbbc04a6e68b2d801cddc52b23be9ddd55680dfc449c711 v4/scripts/run_protocol101_full_trader_stage1_gate_audit_selection_validation.py
ae1c5f875d3f72abd597d430789e5b3065ac9b4b0c86c349e0d88daa5c0b7bfa v4/tests/test_protocol101_stage1_cross_hypothesis_selection.py
0601ac0766dbc81fb4dc1d6a3973a61d5f3ac8047dc71f2cf78faf9aa5d4e59a v4/tests/test_protocol101_stage1_autoresearch_graph.py
698d80934b123a2082f6b623b7fb693618393d54d03c259585d0d543821e8df4 v4/tests/test_protocol101_full_trader_stage1_gate_audit_selection_validation.py
```

AUTHORIZED FILES:

Existing production:

```text
v4/scripts/run_protocol101_stage1_cross_hypothesis_selection.py
v4/scripts/run_protocol101_stage1_autoresearch_graph.py
v4/scripts/run_protocol101_full_trader_stage1_gate_audit_selection_validation.py
```

Existing tests:

```text
v4/tests/test_protocol101_stage1_cross_hypothesis_selection.py
v4/tests/test_protocol101_stage1_autoresearch_graph.py
v4/tests/test_protocol101_full_trader_stage1_gate_audit_selection_validation.py
```

One new focused production module and test are allowed:

```text
v4/model/protocol101_stage1_controller_journal.py
v4/tests/test_protocol101_stage1_controller_journal.py
```

Do not modify any other source, test, signed document, or prior packet.

OUTPUT DIRECTORY:

```text
v4/audit/autoresearch/
protocol101_full_trader_stage1_option_a_controller_journal_repair_attempt004/
```

PREREGISTRATION:

Before production edits, write and hash:

```text
preregistration.json
input_inventory.json
journal_contract.json
progress.json
```

Freeze the five exact failures, trust model, schemas, path/permission rules,
crash behavior, allowed files, tests, forbidden actions, and terminal routes.

OPTION A TRUST MODEL:

The local controller and its journal are trusted. The training subprocess,
result packets, references, independent audit, and selector are untrusted
inputs until the controller records them.

The journal protects against stale/mixed artifacts, accidental rewrites,
self-resealed evidence, cross-campaign substitution, crash/resume confusion,
and pipeline-level reward hacking. It does not claim protection against the
computer owner or a malicious process that can rewrite controller code and the
journal.

REQUIRED JOURNAL:

Implement one JSONL journal per campaign execution.

## Genesis

Create exactly once with an exclusive create. Genesis binds:

- current journal schema;
- campaign namespace and execution ID;
- owner Option A decision hash;
- offline-training authorization hash;
- campaign Goal hash when later available;
- campaign preregistration hash;
- signed contract-bundle hash;
- owner identity/date;
- previous record hash set to the fixed genesis sentinel; and
- record hash over the canonical full record.

Refuse overwrite, truncation, alternate genesis, or campaign reuse.

## Append

Append exactly one canonical next-node checkpoint at a time:

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

Each record binds:

- exact ordinal/node/route;
- campaign namespace/execution ID;
- owner/genesis hash;
- prior journal record hash;
- exact artifact path relative to workspace;
- artifact SHA-256 computed by the journal from disk;
- validator route and validator-receipt SHA-256;
- timestamp; and
- canonical record hash.

Requirements:

- journal computes file hashes itself;
- append uses file locking, flush, and `fsync`;
- create journal and lock with owner-only permissions where supported;
- existing lines are never rewritten;
- every append revalidates the complete prefix;
- duplicate, reordered, skipped, extra, malformed, cross-campaign, changed, or
  self-resealed inputs fail closed;
- resume reads only the validated final checkpoint;
- a partially written final line is reported and never accepted;
- no API accepts caller-supplied replacement trusted bindings.

## Graph integration

- Graph state comes from the validated journal prefix.
- Remove or reject direct caller-supplied trusted artifact maps as an
  authorization mechanism.
- Empty journal after valid genesis identifies RUN as next.
- Exact prefixes identify exact next nodes.
- Only all eight journaled checkpoints reach STOP.
- Graph exposes accepted artifact bindings from the journal.
- Graph still executes no campaign commands and cannot create owner
  authorization.

## Selector integration

- Selector must receive the validated journal path or a controller-produced
  immutable checkpoint receipt tied to that path.
- Selector may accept only the independent-audit payload whose exact freeze or
  packet hash equals the journaled `INDEPENDENT_AUDIT` artifact binding.
- A fully modified and self-resealed audit payload must route:
  `campaign_invalid_no_selection`.
- A fully modified producer plus independent payload must route the same.
- Selector output may be journaled only after selection completes.

The journaled independent-audit binding is the external expected value. The
audit payload cannot provide its own replacement expectation.

EXACT FIVE REPRODUCTIONS:

Before repair reproduce all five failures. After repair require:

```text
AUTH-ATTACK-02-authority_rebuilt: reject before STOP
AUTH-ATTACK-03-all_self_hashes: reject before STOP
AUTH-ATTACK-09-control_rebuilt: reject before STOP
SEL-fully-resealed-result: campaign_invalid_no_selection
SEL-fully-resealed-producer-independent: campaign_invalid_no_selection
```

Recreate semantics against the journal API; do not modify prior reproducers.

FOCUSED ADVERSARIAL BATTERY:

At minimum test:

- all valid prefix lengths;
- crash/resume after every record;
- duplicate and skipped node append;
- changed prior line, artifact, route, validator receipt, or record hash;
- mixed campaign ID/namespace;
- cross-campaign journal;
- caller-provided replacement artifact map ignored/rejected;
- rebuilt authority/control artifacts after earlier journal acceptance;
- rewritten audit payload after journal acceptance;
- fully resealed producer/independent audit payload;
- stale journal from another campaign;
- partial final JSON line;
- concurrent append/lock behavior;
- owner-only permissions;
- exact selector journal binding; and
- forbidden G9/holdout/lifecycle/paper nodes.

Use at least 60 focused journal/selector cases.

SELF-REPAIR LOOP:

Continue within authorized files until all five exact cases, focused cases, and
regressions pass. Stop only if a repair requires changing signed policy,
another source file, real economics, or owner authorization.

REGRESSIONS:

- compile all changed files;
- run the exact current 109-test FT1C/dependency suite;
- run current graph/validation tests;
- run new journal/selector tests;
- run validation in a fresh process and temporary campaign directory;
- verify every prior packet/checksum remains unchanged; and
- verify non-authorized source hashes remain unchanged.

REQUIRED OUTPUTS:

```text
preregistration.json
preregistration_freeze.sha256
input_inventory.json
journal_contract.json
before_reproduction.json
after_reproduction.json
journal_attack_matrix.csv
repair_manifest.json
changed_files.json
test_results.json
summary.json
routing_decision.json
report.md
hashes.sha256
progress.json
```

TERMINAL ROUTES:

Success:

```text
option_a_controller_journal_repair_complete_pending_independent_acceptance
```

Bounded blocker:

```text
option_a_controller_journal_repair_blocked
```

Owner-policy blocker:

```text
option_a_controller_journal_owner_decision_required
```

HIGHEST ALLOWED CLAIM:

```text
Owner-approved Option A controller journal implemented; fresh independent
acceptance required before offline training.
```

Do not claim machinery acceptance, training execution, candidate eligibility,
profitability, G9, or paper readiness.
