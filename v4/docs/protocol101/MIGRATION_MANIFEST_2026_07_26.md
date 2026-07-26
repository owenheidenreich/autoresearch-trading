# Protocol101 Documentation Migration Manifest

Date: 2026-07-26
Scope: documentation organization only
Canonical training entrypoint: [training/README.md](training/README.md)

## Purpose

This migration separates Protocol101 model-training authority from
synchronization history without changing signed document contents, model
artifacts, datasets, research evidence, runtime state, or broker operations.

The complete file-level mapping is recorded in
[MIGRATION_MANIFEST_2026_07_26.csv](MIGRATION_MANIFEST_2026_07_26.csv). Its 288
rows record:

- old path;
- canonical new path;
- category;
- pre-migration SHA-256;
- post-migration SHA-256; and
- compatibility treatment.

The CSV is authoritative for path migration. This document explains the
policy and high-level result.

## Resulting Layout

```text
v4/docs/protocol101/
├── training/
│   ├── README.md
│   ├── contracts/
│   ├── execution/
│   ├── research/
│   ├── audits/
│   ├── goals/
│   └── history/
├── synchronization/
│   ├── contracts/
│   ├── design/
│   └── history/
├── MIGRATION_MANIFEST_2026_07_26.md
└── MIGRATION_MANIFEST_2026_07_26.csv
```

Recorder operations, IBKR recovery, paper trading, and shadow/launch
instructions remain directly under `v4/docs/`.

## Front-Page Consolidation

The useful navigation, namespace, alpha-preservation, and training-regimen
material from these documents was consolidated into `training/README.md`:

- `v4/docs/PROTOCOL101_TRAINING_CONTROL_CENTER.md`
- `v4/docs/PROTOCOL101_TRAINING_REGIMEN_AND_INTEGRITY_CONTROL_2026_07_26.md`

The control-center file was retired. The original regimen document is
preserved byte-for-byte at:

```text
v4/docs/protocol101/training/history/frozen/
  PROTOCOL101_TRAINING_REGIMEN_AND_INTEGRITY_CONTROL_2026_07_26.md
```

Its old path remains a compatibility symlink because it was an input to the
completed Stage-1 adversarial audit.

The new README reflects the audit's terminal route
`regimen_invalid_redesign_required`; it does not repeat the superseded
pre-audit claim that `S1-H2/policy 5` is ready for reaggregation.

## Compatibility Paths

Legacy symlinks are retained only for signed or machine-consumed audit inputs:

```text
PROTOCOL101_TRADER_CHARTER.md
PROTOCOL101_SCOPED_SYNCHRONIZATION_DECISION_2026_07_25.md
PROTOCOL101_CANONICAL_V1_STAGE1_TRAINING_DESIGN.md
PROTOCOL101_STAGE1_OBJECTIVE_AND_GATES_PROPOSAL.md
PROTOCOL101_G4_HOLDOUT_REVISION_2026_07_19.md
PROTOCOL101_G8_CALIBRATION_REVISION_2026_07_26.md
PROTOCOL101_TRAINING_REGIMEN_AND_INTEGRITY_CONTROL_2026_07_26.md
PROTOCOL101_PRIOR_CAMPAIGN_DISTILLATION.md
PROTOCOL101_PROTOCOL_FARM_LINEAGE_2026_07_19.md
PROTOCOL101_STAGE1_ADVERSARIAL_AUDIT_SPEC.md
PROTOCOL101_S1_REGIMEN_AUDIT_GOAL_2026_07_26.md
```

Existing companion `.sha256` compatibility paths are retained with the same
policy. New code and documentation must use canonical paths under
`v4/docs/protocol101/`.

Other historical paths are intentionally resolved through the CSV and Git
history rather than a second flat compatibility tree.

## Audit Reproducibility

Before migration, the completed adversarial audit reported:

```text
status: complete
terminal_route: regimen_invalid_redesign_required
summary_hash: 770f2d1968270c383d70216e35ca83598988e1ebe2d5e88443e03daa8005a33d
```

All Protocol101 governing-document hashes in its `source_inventory.json`
remain reproducible through the compatibility paths.

Because `AGENTS.md` and the project single-source-of-truth now point future
agents to the canonical training README, their exact pre-migration audit inputs
are preserved at:

```text
v4/docs/protocol101/training/audits/frozen_inputs/
  protocol101_stage1_training_regimen_adversarial_audit_attempt001/
    AGENTS.md
    CURRENT_TRADING_BOT_SINGLE_SOURCE_OF_TRUTH.md
    run_protocol101_stage1_gate_validity_and_integrity_audit.py
```

Those frozen copies have the audit-recorded SHA-256 values:

```text
AGENTS.md:
  69a03d691669811b0156e13c3da30dcef7866037d7457e56fa0fc35e07d0ebd2
CURRENT_TRADING_BOT_SINGLE_SOURCE_OF_TRUTH.md:
  47964607e13f6fee361fa6dc7fb763944cfd78c606b39645576b7661873a02c4
run_protocol101_stage1_gate_validity_and_integrity_audit.py:
  79f276e2a351f412e05da287975e94ded49394f67990c5c3471b1b32158e95af
```

No audit artifact under `v4/audit/autoresearch/` was moved or edited by this
migration.

## Checksum Policy

- Signed Markdown contents were moved without byte changes.
- Existing `.sha256` companions moved beside their canonical documents.
- Only the path field inside those checksum files was updated.
- Two companions were already stale before migration
  (`PROTOCOL101_GOAL_SIZED_GATED_TRAINING_SYSTEM_2026_07_25.md.sha256` and
  `PROTOCOL101_S1_H2_CALIBRATION_TO_G9_GOAL_2026_07_26.md.sha256`); their
  digest fields were corrected to the actual pre-migration Markdown hashes.
- The CSV records pre- and post-migration hashes, including the expected
  checksum-file changes.
- Historical documents were not rewritten to modernize old narrative or
  provenance.

## Going Forward

- Begin every Protocol101 training task at `training/README.md`.
- Treat synchronization contracts as inputs to training, not as training
  results.
- Put detailed evidence under `v4/audit/autoresearch/`.
- Put new owner-signed training amendments under `training/contracts/`.
- Put bounded Goal prompts under `training/goals/`.
- Update the canonical README after independent acceptance instead of creating
  another status/index/handoff document.
