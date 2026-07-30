# Protocol101 Goal Prompt: FT1C6 Option A Journal Independent Acceptance

Run this Goal in a fresh Codex task or subagent that did not implement FT1C5.

---

GOAL ID:

`FT1C6-OPTION-A-CONTROLLER-JOURNAL-INDEPENDENT-ACCEPTANCE`

OBJECTIVE:

Independently accept or reject the owner-approved Option A local controller
journal and its graph/selector binding. Verify that the five final FT1C
failures are closed without weakening any previously accepted Stage-1 rule.

Keep this audit practical and local. Test accidental/stale/self-resealed
pipeline evidence and crash/resume behavior. Do not add cryptography or expand
the threat model to a malicious computer owner.

This Goal is acceptance only. Do not modify production or run training.

WORKSPACE AND INTERPRETER:

```text
workspace: /Users/gduby/Documents/autoresearch-trading
interpreter: ~/.autoresearch-trading/runtime-venv/bin/python
```

READ BEFORE ACTING:

- Option A owner-decision packet
- offline-training authorization packet
- FT1C5 Goal and producer packet
- the five FT1C4 reproducers
- current journal, graph, selector, validation sources and tests
- accepted simulator-v5, runner-v5, reference/multiplicity, gate, and audit
  contracts

REQUIRED PRODUCER ROUTE:

```text
option_a_controller_journal_repair_complete_pending_independent_acceptance
```

FROZEN PRODUCER ARTIFACT HASHES:

```text
01a32158dee8fe6e73dacd3068cd23af360d5ceaa8d76cb1d22e99364904681e preregistration.json
91b4a0afddb291cdb691fad41f55e9c5734f7224a0f8a85cca9dc2823d52a364 journal_contract.json
cb333e0e434cc22cc54df7c98cf443710ff05b3f77aea15b7c142e2561782e96 after_reproduction.json
78f001fff6f4ccf48c7a6c1790c622c45c5c795a5b8bb169845f15ce92fa577d journal_attack_matrix.csv
a8c3eaf4a9e6cfa1e6df174f23173a0cfff4ea9c5ba9a3a20f905d0bc7f64fec repair_manifest.json
611f15cf6771b2f771044fc24fdba9b8c30fb6ee8c879248d67bbe1cc83d9bd6 test_results.json
6320df3990d7c57a76252a92cdbc09e51c16bd3de16841c618368a8932ef6f17 summary.json
050de956518112ae72a0305f7993ca4aebeeb37eabded0fb4606abc5b401f838 routing_decision.json
b866f3e9f7764dac4ada8672996b6ba61c7e6672bcc0dec8035af8b90d656f72 report.md
862f030344634eff5b9e23f53546fceae4d68786e4bee529fac6945472b6f261 hashes.sha256
```

Paths are relative to the FT1C5 producer directory.

FROZEN SOURCE HASHES:

```text
204eb06a8b1332a68ab82f01390f9e14e8dda72ed3a0066c12e1653092137937 v4/model/protocol101_stage1_controller_journal.py
233b8b21e6225bab4a00fd0ae5d6008e065c6de8b57def1a8a73f812fb05f232 v4/scripts/run_protocol101_stage1_cross_hypothesis_selection.py
1b910298d34224d3978b0ee34d242f5d95f70cd542000f269458016eadebce4f v4/scripts/run_protocol101_stage1_autoresearch_graph.py
fcda7554b1380db03727ff994699d0903feaac50a3a2bef25a588cfdf20a3774 v4/scripts/run_protocol101_full_trader_stage1_gate_audit_selection_validation.py
94fbe408a1e0c4c7641514a5db9786e34567a12a02d46a1656c7880b20c4d344 v4/tests/test_protocol101_stage1_controller_journal.py
ae1c5f875d3f72abd597d430789e5b3065ac9b4b0c86c349e0d88daa5c0b7bfa v4/tests/test_protocol101_stage1_cross_hypothesis_selection.py
3ebd29c0cd3fb3ecae128702f5709662dd28d9fac78a6e8a64f4e326ef6f9040 v4/tests/test_protocol101_stage1_autoresearch_graph.py
3539ad56f8b78d100abb40f3edcd5aecd7a1ad0fd901e77b3a3e9629b671b89c v4/tests/test_protocol101_full_trader_stage1_gate_audit_selection_validation.py
```

OUTPUT DIRECTORY:

```text
v4/audit/autoresearch/
protocol101_full_trader_stage1_option_a_controller_journal_independent_acceptance_attempt001/
```

This is the only writable directory.

INDEPENDENCE:

Write audit-local fixtures/verifier code. Do not import producer tests,
validation fixtures, or producer expected-output helpers. Public journal,
graph, and selector APIs may be invoked as systems under test.

PREREGISTER before testing:

```text
preregistration.json
source_inventory.json
independent_verifier_manifest.json
progress.json
```

Freeze the five exact cases, at least 60 focused cases, expected routes, crash
cases, regression commands, all-or-nothing acceptance, and forbidden actions.

REQUIRED BATTERY:

1. Verify producer/source/prior-packet hashes.
2. Independently create a temporary campaign journal.
3. Verify exclusive genesis, owner-only mode, exact genesis bindings, canonical
   hashes, and refusal to overwrite.
4. Append every valid node one at a time from real temporary files; verify
   `fsync`, prior-hash linkage, exact artifact hashes, routes, and full-prefix
   validation.
5. Verify resume after every prefix and partial final-line handling.
6. Verify graph state comes only from the journal and rejects caller-supplied
   replacement trusted maps.
7. Verify selector accepts only the journaled independent-audit artifact.
8. Recreate all five exact FT1C4 failures and require:

```text
AUTH-ATTACK-02-authority_rebuilt: reject before STOP
AUTH-ATTACK-03-all_self_hashes: reject before STOP
AUTH-ATTACK-09-control_rebuilt: reject before STOP
SEL-fully-resealed-result: campaign_invalid_no_selection
SEL-fully-resealed-producer-independent: campaign_invalid_no_selection
```

9. Mutate/reorder/duplicate/omit records, artifacts, validator receipts,
   campaign IDs, namespaces, prior hashes, and journal hashes.
10. Test cross-campaign/stale journals, concurrent append, invalid permissions,
    forbidden downstream nodes, and crash recovery.
11. Verify no accepted G1-G8, simulator-v5, maxT, audit, or selection policy
    changed.

REGRESSIONS:

- compile frozen sources/tests;
- exact current `109` FT1C/dependency tests;
- exact current `100` journal/graph/selector/validation tests;
- fresh independent verifier tests;
- fresh-process temporary-journal smoke;
- all packet checksums; and
- final source hash recheck.

Do not run the entire repository suite.

ACCEPTANCE:

All checks pass or the machinery is rejected. Do not repair production.

REQUIRED OUTPUTS:

```text
preregistration.json
preregistration_freeze.sha256
source_inventory.json
independent_verifier.py
independent_verifier_manifest.json
test_independent_verifier.py
journal_acceptance.json
graph_acceptance.json
selector_acceptance.json
mutation_matrix.csv
regression_results.json
acceptance_decision.json
summary.json
report.md
hashes.sha256
progress.json
```

Preserve deterministic reproducers for failures.

TERMINAL ROUTES:

Accepted:

```text
option_a_controller_journal_independently_accepted
```

Rejected:

```text
option_a_controller_journal_independent_rejection
```

HIGHEST ALLOWED CLAIM ON ACCEPTANCE:

```text
Protocol101 Stage-1 pre-training machinery, including the owner-approved Option
A journal, is independently accepted for the authorized offline H0-H3 campaign.
```

NEXT PHASE:

On acceptance, the controller may immediately run the separately written,
checksummed, owner-authorized offline H0-H3 training Goal. Stop before G9 and
HOLD/EXIT training.
