# Stage 1 Snapshot Plan

Date: 2026-05-24
Requested branch: `stage1-control-snapshot-2026-05-24`
Requested tag after approved commit: `v4-protocol101-operational-control-2026-05-24`
Source workspace: `/Users/gduby/Documents/autoresearch-trading`
Proposed clean snapshot worktree: `/Users/gduby/Documents/autoresearch-trading-stage1-control-snapshot`
Proposed local evidence root: `/Users/gduby/Documents/autoresearch-stage1-evidence/v4-protocol101-operational-control-2026-05-24`

## Stop Point

This file is the pre-commit plan, revised after approval to proceed only through staging and pre-commit review. I am approved to create the clean worktree, generate curated include/bundle/exclude lists, create a local-only evidence bundle, stage the proposed snapshot, run safe verification, and write `research_ops/bootstrap/STAGE1_PRE_COMMIT_REVIEW.md`.

I am not approved to commit, tag, push, delete files, stash, clean, run broker scripts, run paid-data scripts, train models, tune thresholds, mutate runtime flags, or change trading behavior.

## Why A Clean Worktree

The current source workspace is the actual local operational Protocol101 state, but it is not clean:

- Current branch: `v4/phase-0`
- Dirty tracked files: `36`
- Untracked files: `11912`
- Untracked `v4/audit` files: `9659`
- Key current-control docs and artifacts are untracked.
- GitHub `origin/main` diverges from local research history and should not be treated as the operational control.

I will not delete, clean, or stash anything. Instead, I propose to create a separate clean worktree from `v4/phase-0`, copy curated snapshot files from the source workspace into it, commit there, and tag only after that clean curated commit exists.

## Proposed Git Tracking

The Git snapshot should contain source, docs, small control metadata, and selected audit reports needed to understand the current Protocol101 operational state. It should not contain model binaries, raw paid data, large generated tables, live runtime state, or paper logs.

### Root Files

Proposed Git tracking:

- `README.md`
- `pyproject.toml`
- `.gitignore` only if a later hygiene line is needed to keep evidence bundle outputs out of Git; no behavior change.

Reason: current dependencies and root documentation are part of reproducing the local project state, even where docs are stale and called out as stale.

### Current Source-Of-Truth Docs

Proposed Git tracking:

- `docs/CURRENT_TRADING_BOT_SINGLE_SOURCE_OF_TRUTH.md`
- `docs/CURRENT_TRADING_BOT_IMPROVEMENT_QUESTIONS.md`
- `docs/TUESDAY_NO_ORDER_EVIDENCE_COLLECTION_PLAN.md`

Reason: these are the operational truth documents that Stage 1 is freezing around.

### Research Ops Bootstrap

Proposed Git tracking:

- `research_ops/README.md`
- `research_ops/STAGE_0_TRANSITION_CHARTER.md`
- `research_ops/AI_AGENT_OPERATING_CONTRACT.md`
- `research_ops/CEO_DASHBOARD.md`
- `research_ops/ASSUMPTION_REGISTRY.md`
- `research_ops/DECISION_QUEUE.md`
- `research_ops/templates/cartography_report.md`
- `research_ops/templates/dashboard_update.md`
- `research_ops/templates/decision_memo.md`
- `research_ops/templates/experiment_rfc.md`
- `research_ops/templates/implementation_patch.md`
- `research_ops/templates/verifier_report.md`
- `research_ops/bootstrap/V4_BASELINE_INVENTORY.md`
- `research_ops/bootstrap/STAGE1_SNAPSHOT_PLAN.md`
- Later, after approval: `research_ops/bootstrap/STAGE1_REDACTED_MANIFEST.md`
- Later, after approval: `research_ops/bootstrap/STAGE1_EVIDENCE_BUNDLE_SHA256.txt`
- Later, after approval: `research_ops/bootstrap/STAGE1_GIT_INCLUDE_PATHS.txt`

Reason: these files are the governance layer around v4.

### v4 Operational Source Code

Proposed Git tracking:

- `v4/README.md`
- `v4/checks/**/*.py`
- `v4/dataset/**/*.py`
- `v4/foundation/**/*.py`
- `v4/ingest/**/*.py`
- `v4/live/**/*.py`
- `v4/model/**/*.py`
- `v4/schema/**/*.py`
- `v4/sim/**/*.py`
- `v4/scripts/**/*.py`

Important note: this includes broker-risk and paid-data-risk source files as inert code only. They will not be run.

Reason: Protocol101 runtime imports cross-cutting v4 code. The current stack is not only one script; Protocol160 imports Protocol158 helpers, Protocol101 entry inference, Protocol051 surface scoring, Protocol066/081 lifecycle inference, paper executor/guard/logging, schema/data helpers, and training/replay modules used by the artifact loaders.

### v4 Ops And Launch Assets

Proposed Git tracking:

- `v4/ops/ibkr/**/*.sh`
- `v4/ops/ibkr/**/*.py`
- `v4/ops/launchd/**/*.plist`
- `v4/ops/launchd/**/*.sh`

Reason: these files define the current scheduled guarded paper default. They are broker-risk files and must not be run in this snapshot workflow.

### v4 Docs And Promotion Metadata

Proposed Git tracking:

- `v4/docs/**/*.md`
- `v4/docs/trading_bot_engineer_strategy_audit_2026_05_24/*.csv`
- `v4/promotion/*.md`
- `v4/promotion/*.json`
- `v4/promotion/.gitkeep`
- `v4/ledger/RESEARCH_LEDGER.md`

Reason: these documents and promotion metadata define naming, role labels, promotion gates, frozen Protocol101/066/081 lineage, stale-readiness contradictions, and current research blockers.

### Current Control Audit Reports And Small Metadata

Proposed Git tracking for selected audit directories:

- `v4/audit/ibkr_live_data_entitlements/report.md`
- `v4/audit/ibkr_live_data_entitlements/summary.json`
- `v4/audit/autoresearch/formal_validation_governance/{report.md,summary.json}`
- `v4/audit/autoresearch/foundation_hardening_review/{report.md,summary.json}`
- `v4/audit/autoresearch/live_no_order_full_action_parity_readiness/{report.md,summary.json}`
- `v4/audit/autoresearch/project_section_readiness/{report.md,summary.json}`
- `v4/audit/autoresearch/truth_grounded_replacement_program_v1/{report.md,summary.json}`
- `v4/audit/autoresearch/untouched_holdout_availability/{report.md,summary.json}`
- `v4/audit/autoresearch/unified_untouched_holdout_reservation/{report.md,summary.json}`
- `v4/audit/autoresearch/unified_neural_training_readiness/{report.md,summary.json}`
- `v4/audit/autoresearch/unified_protocol101_baseline_attachment/{report.md,summary.json}`
- `v4/audit/autoresearch/tuesday_no_order_evidence_packet/{report.md,summary.json}`
- `v4/audit/autoresearch/protocol101_*_v1/{report.md,summary.json}`
- `v4/audit/autoresearch/v4_aplus_hypothesis_075_protocol054_frozen_stack_artifacts/{report.md,report.json}`
- `v4/audit/autoresearch/v4_aplus_hypothesis_081_q4start_residual_sequence_deterministic_artifacts/{report.md}`
- `v4/audit/autoresearch/v4_aplus_hypothesis_084_protocol081_promotion_readiness/{report.md,summary.json}`
- `v4/audit/autoresearch/v4_aplus_hypothesis_090_protocol081_strict_shadow_lifecycle/{report.md,summary.json}`
- `v4/audit/autoresearch/v4_aplus_hypothesis_101_event_history_policy/{report.md,summary.json}`
- `v4/audit/autoresearch/v4_aplus_hypothesis_102_protocol101_readiness/{report.md,summary.json}`
- `v4/audit/autoresearch/v4_aplus_hypothesis_103_protocol101_external_audit_readiness/{report.md,summary.json}`
- `v4/audit/autoresearch/v4_aplus_hypothesis_107_protocol101_q4_2024_external_stress/{report.md,summary.json}`
- `v4/audit/autoresearch/v4_aplus_hypothesis_109_frozen_protocol101_seed_ensemble/{report.md}`
- `v4/audit/autoresearch/v4_aplus_hypothesis_112_protocol101_money_breakdown/{report.md}`
- `v4/audit/autoresearch/v4_aplus_hypothesis_113_protocol101_trade_charts/{report.md,summary.json}`
- `v4/audit/autoresearch/v4_aplus_hypothesis_114_protocol101_skeptical_falsification/{report.md,summary.json}`
- `v4/audit/autoresearch/v4_aplus_hypothesis_115_protocol101_existing_1s_path_audit/{report.md,summary.json}`
- `v4/audit/autoresearch/v4_aplus_hypothesis_116_protocol101_targeted_1s_request/summary.json`
- `v4/audit/autoresearch/v4_aplus_hypothesis_117_protocol101_targeted_highres_path_audit/report.md`
- `v4/audit/autoresearch/v4_aplus_hypothesis_117_protocol101_targeted_highres_validation/{report.md,summary.json}`
- `v4/audit/autoresearch/v4_aplus_hypothesis_118_protocol101_shadow_rehearsal/{report.md,summary.json}`
- `v4/audit/autoresearch/v4_aplus_hypothesis_119_protocol101_live_readiness/{report.md,summary.json}`
- `v4/audit/autoresearch/v4_aplus_hypothesis_121_protocol101_entry_router_smoke/{report.md,summary.json}`
- `v4/audit/autoresearch/v4_aplus_hypothesis_122_protocol101_capital_realism/{report.md,summary.json}`
- `v4/audit/autoresearch/v4_aplus_hypothesis_123_protocol101_order_state_rehearsal/{report.md,summary.json}`
- `v4/audit/autoresearch/v4_aplus_hypothesis_124_protocol101_live_data_parity_checkpoint/{report.md,summary.json}`
- `v4/audit/autoresearch/v4_aplus_hypothesis_125_protocol101_pre_tuesday_readiness/{report.md,summary.json}`
- `v4/audit/autoresearch/v4_aplus_hypothesis_126_protocol101_timing_fragility_hardening/{report.md,summary.json}`
- `v4/audit/autoresearch/v4_aplus_hypothesis_127_protocol101_live_shadow_schema_hardening/{report.md,summary.json}`
- `v4/audit/autoresearch/v4_aplus_hypothesis_128_protocol101_paper_risk_gate/{report.md,summary.json}`
- `v4/audit/autoresearch/v4_aplus_hypothesis_140_ibkr_autostart_prep/{report.md,summary.json}`
- `v4/audit/autoresearch/v4_aplus_hypothesis_146_ibc_credential_readiness/{report.md,summary.json}` if the report is scrubbed and contains no credential material.
- `v4/audit/autoresearch/v4_aplus_hypothesis_147_protocol101_morning_session/{report.md,summary.json}`
- `v4/audit/autoresearch/v4_aplus_hypothesis_150_protocol101_paper_order_enablement_gate/{report.md,summary.json}`
- `v4/audit/autoresearch/v4_aplus_hypothesis_155_protocol101_live_timing_evidence/{report.md,summary.json}`
- `v4/audit/autoresearch/v4_aplus_hypothesis_156_ibkr_autostart_observability/{report.md,summary.json}`
- `v4/audit/autoresearch/v4_aplus_hypothesis_157_protocol101_daily_ops_monitor/{report.md,summary.json}`
- `v4/audit/autoresearch/v4_aplus_hypothesis_158_protocol101_live_entry_paper_bridge/{report.md,summary.json}`
- `v4/audit/autoresearch/v4_aplus_hypothesis_160_protocol101_persistent_paper_trader/{report.md,summary.json}`
- `v4/audit/autoresearch/v4_aplus_hypothesis_161_may2026_historical_replay/{report.md,summary.json}`
- `v4/audit/autoresearch/v4_aplus_hypothesis_162_may2026_serial_lifecycle_replay/{report.md,summary.json}`
- `v4/audit/autoresearch/v4_aplus_hypothesis_163_recent_protocol101_historical_replay/{report.md,summary.json}`
- `v4/audit/autoresearch/v4_aplus_hypothesis_163_recent_protocol101_serial_lifecycle_replay/{report.md,summary.json}`
- `v4/audit/autoresearch/v4_aplus_hypothesis_168_protocol163_threshold_replay/{report.md,summary.json}`
- `v4/audit/autoresearch/v4_aplus_hypothesis_272_fill_model_readiness/{report.md,summary.json}`
- `v4/audit/autoresearch/v4_aplus_hypothesis_273_model_selection_overfit_risk/{report.md,summary.json}`

Reason: these files preserve the control lineage, live/paper readiness path, major blockers, fill-realism/readiness concerns, validation-overfit concerns, and current Protocol101 audit context without adding large generated datasets.

### Current Model And Artifact Metadata

Proposed Git tracking:

- All `manifest.json`, `summary.json`, `report.md`, `report.json`, `scaler.json`, `entry_standardizer.json`, `protocol054_risk_scaler.json`, and `threshold_sweep.json` under:
  - `v4/audit/autoresearch/v4_aplus_hypothesis_101_event_history_policy`
  - `v4/audit/autoresearch/v4_aplus_hypothesis_075_protocol054_frozen_stack_artifacts`
  - `v4/audit/autoresearch/v4_aplus_hypothesis_081_q4start_residual_sequence_deterministic_artifacts`

Proposed Git exclusion from these directories:

- `*.pt`
- large generated trade tables and path tables unless explicitly listed for bundle-only evidence.

Reason: manifests and scalers are needed to identify and load the control artifacts; model binaries are too large and should live in the local evidence bundle.

### Tests

Proposed Git tracking:

- `v4/tests/test_protocol101*.py`
- `v4/tests/test_protocol051*.py`
- `v4/tests/test_protocol066*.py`
- `v4/tests/test_protocol081*.py`
- `v4/tests/test_protocol142_paper_executor.py`
- `v4/tests/test_protocol147_morning_session.py`
- `v4/tests/test_protocol158*.py`
- `v4/tests/test_protocol160*.py`
- `v4/tests/test_paid_data_guard.py`
- `v4/tests/test_optionsdx_ingest.py`
- `v4/tests/test_*paper*.py`
- `v4/tests/test_*replay*.py`
- Any already-tracked modified tests in the source workspace that are part of paper guard, paper log, paper replay, paid-data guard, Protocol101, Protocol051, or Protocol066/081 coverage.

Reason: tests that protect current Protocol101 operational semantics should travel with the snapshot.

## Proposed Evidence Bundle Only

These files should not be Git-tracked, but should be copied to the local evidence bundle because they may be needed to reconstruct the operational control or audit conclusions.

### Runtime And Paper Evidence

- `v4/runtime/protocol101_paper_order_enablement.json`
- `v4/runtime/protocol101_live_index_context.jsonl`
- `v4/logs/paper_trading/2026-05-21/protocol101_persistent-paper_2026-05-21.jsonl`
- `v4/logs/paper_trading/2026-05-21/protocol101_persistent-paper_2026-05-21.csv`
- Earlier paper-smoke logs from `2026-05-14` through `2026-05-20` only if required by a listed report; otherwise manifest only.

Category in local-only full manifest: `runtime_state` or `paper_log`.

### Current Control Model Binaries

- Protocol101 current/default runtime model binaries under:
  - `v4/audit/autoresearch/v4_aplus_hypothesis_101_event_history_policy/model_artifacts/**/model.pt`
- Protocol051/054 surface model binaries needed by the current default surface manifest:
  - `v4/audit/autoresearch/v4_aplus_hypothesis_075_protocol054_frozen_stack_artifacts/model_artifacts/train_through_q4_2025_test_q1_2026/seed_11/entry_model.pt`
  - `v4/audit/autoresearch/v4_aplus_hypothesis_075_protocol054_frozen_stack_artifacts/model_artifacts/train_through_q4_2025_test_q1_2026/seed_11/protocol054_risk_model.pt`
- Protocol081 lifecycle model binary for current default:
  - `v4/audit/autoresearch/v4_aplus_hypothesis_081_q4start_residual_sequence_deterministic_artifacts/model_artifacts/train_q1_2025_q2_2025_q3_2025_q4_2025_test_q1_2026/seed_1/model.pt`

Category in local-only full manifest: `model_binary`.

### Key Non-Git Audit Tables

Proposed bundle-only if present:

- `v4/audit/autoresearch/v4_aplus_hypothesis_101_event_history_policy/serial_policy_trades.json`
- `v4/audit/autoresearch/protocol101_strategy_forensics_packet_v1/score_calibration.csv`
- `v4/audit/autoresearch/protocol101_internal_slot_cost_counterfactual_v1/blocked_protocol101_internal_slot_events.csv`
- `v4/audit/autoresearch/protocol101_internal_slot_cost_counterfactual_v1/hypothetical_flat_protocol101_entries.csv`
- `v4/audit/autoresearch/protocol101_internal_slot_cost_counterfactual_v1/open_trade_slot_cost_summary.csv`
- `v4/audit/autoresearch/protocol101_loss_reversal_full_serial_replay_v1/full_serial_replay_trades.csv`
- `v4/audit/autoresearch/v4_aplus_hypothesis_272_fill_model_readiness/live_and_paper_event_inventory.csv`
- `v4/audit/autoresearch/v4_aplus_hypothesis_273_model_selection_overfit_risk/*` only for small summary/report/diagnostic tables, not giant raw matrices.

Category in local-only full manifest: `generated_audit`.

## Proposed Excluded Entirely

These should not be Git-tracked and should not be copied into the local evidence bundle unless you explicitly request a much larger private archival bundle.

### Paid Data And Rebuildable Market Data

- `data/**`
- `v4/raw/**`
- `v4/normalized/**`
- `v4/normalized_official_context/**/*.parquet`
- `v4/normalized_official_context_smoke/**/*.parquet`
- `v4/normalized_official_context_fix_smoke/**/*.parquet`
- Databento and ThetaData download logs not explicitly needed for current-control evidence.

Category in local-only full manifest: `paid_data` or `raw_data`.

Reason: likely paid/licensed data, large, and not suitable for Git or a default evidence bundle.

### Large Generated Audit Outputs

- Large `*.parquet`, `*.pkl`, `*.html`, `*.jsonl`, `*.csv`, and large `report.json`/`summary.json` files under `v4/audit/autoresearch/**` unless specifically listed above for Git or bundle.
- Very large examples already found:
  - `v4/audit/autoresearch/v4_aplus_hypothesis_211_full_action_history_feature_repair/full_action_surface_edge_with_history.parquet` about 777 MB.
  - `v4/audit/autoresearch/v4_aplus_hypothesis_189_full_coverage_surface_edge_enrichment/protocol185_full_action_with_surface_edge.parquet` about 738 MB.
  - `v4/audit/autoresearch/v4_aplus_hypothesis_270_full_surface_action_advantage_dataset/full_surface_action_advantage.parquet` about 712 MB.

Category in local-only full manifest: `generated_audit`.

### Non-v4 Or Duplicate Historical Material

- `v2 2/**`
- `v2/**/*.pt`
- `v3 2/**`
- `v3/artifacts/**` unless explicitly needed by current v4 docs.
- `archive/**/*.pt`
- duplicate files with names containing `" 2"` from v2/v3/archive copies.
- `gpt-context-bundle/**`

Category in local-only full manifest: `unknown` or `generated_audit`.

Reason: not needed to freeze the v4 Protocol101 operational control and likely increases confusion.

### Local Process Noise

- `.deploy-state*`
- `.claude/*.lock`
- `.sync-pid`
- `__pycache__/**`
- `*.pyc`

Category in local-only full manifest: `runtime_state` or `unknown`.

Reason: local process state, not durable evidence.

## Possible Secrets Or Sensitive Files

These files or directories are sensitive and should not be Git-tracked or copied into the default evidence bundle without manual review:

- `.env`
- `v4/.env`
- `v4/audit/autoresearch/v4_aplus_hypothesis_146_ibc_credential_readiness/**`
- `v4/ops/ibkr/store_ibkr_paper_credentials.sh`
- `v4/ops/ibkr/write_ibc_runtime_config.py`
- `v4/scripts/run_protocol146_ibc_credential_readiness.py`
- `v4/runtime/protocol101_paper_order_enablement.json`
- `v4/runtime/protocol101_live_index_context.jsonl`
- `v4/logs/paper_trading/**`

Plan:

- Code files such as `store_ibkr_paper_credentials.sh` may be tracked only if they contain no actual credentials.
- Reports from credential readiness may be tracked only if scrubbed.
- Runtime state and selected paper logs stay bundle-only.
- `.env` and `v4/.env` are not copied, not Git-tracked, not bundled, and not represented in Git-tracked manifests with hashes.
- `.env` and `v4/.env` may appear only in the local-only full manifest as `sensitive_excluded`.

## Large Files

Observed large untracked files include:

- `v4/audit/autoresearch/v4_aplus_hypothesis_211_full_action_history_feature_repair/full_action_surface_edge_with_history.parquet` about 777 MB.
- `v4/audit/autoresearch/v4_aplus_hypothesis_189_full_coverage_surface_edge_enrichment/protocol185_full_action_with_surface_edge.parquet` about 738 MB.
- `v4/audit/autoresearch/v4_aplus_hypothesis_270_full_surface_action_advantage_dataset/full_surface_action_advantage.parquet` about 712 MB.
- `v3/artifacts/layer2_surface_dataset.pkl` about 210 MB.
- `v4/audit/autoresearch/v4_aplus_hypothesis_274_position_state_action_advantage_dataset/position_state_action_advantage.parquet` about 138 MB.
- `v4/runtime/protocol101_live_index_context.jsonl` about 2.4 MB.
- Current docs plus research_ops are small, about 820 KB total.
- Current Protocol101/075/081 metadata-only files are about 7.1 MB.

Default policy: large generated tables, raw data, and model binaries are excluded from Git. Only current-control binaries and selected audit tables go into the local bundle.

## Paid-Data-Risk Files

Paid-data-risk source files may be Git-tracked as inert code but must not be run:

- `v4/checks/paid_data_guard.py`
- `v4/ingest/databento_opra.py`
- `v4/scripts/build_databento_neural_dataset.py`
- `v4/scripts/download_databento_cbbo_1s_audit.py`
- `v4/scripts/download_databento_cbbo_1s_selected.py`
- `v4/scripts/download_databento_context_proxies.py`
- `v4/scripts/download_databento_es_vwap.py`
- `v4/scripts/download_databento_pilot.py`
- `v4/scripts/download_protocol101_targeted_highres.py`
- `v4/scripts/download_thetadata_index_bars.py`
- `v4/scripts/check_ibkr_live_data_entitlements.py`
- `v4/promotion/PROTOCOL_101_TARGETED_CBBO_1S_DOWNLOAD_MANIFEST.json`
- `v4/promotion/PROTOCOL_161_MAY_2026_REPLAY_DOWNLOAD_MANIFEST.json`
- `v4/promotion/PROTOCOL_163_RECENT_DATA_CATCHUP_REQUEST.json`

Paid/licensed data outputs are not Git-tracked and are not copied to the default evidence bundle.

## Broker, Account, And Runtime Risk Files

Broker-risk source files may be Git-tracked as inert code but must not be run:

- `v4/live/ibkr_paper_executor.py`
- `v4/live/ibkr_paper_guard.py`
- `v4/live/paper_trade_log.py`
- `v4/live/protocol101_live_entry.py`
- `v4/scripts/ibkr_preflight.py`
- `v4/scripts/check_ibkr_live_data_entitlements.py`
- `v4/scripts/run_protocol140_ibkr_autostart_prep.py`
- `v4/scripts/run_protocol146_ibc_credential_readiness.py`
- `v4/scripts/run_protocol147_protocol101_morning_session.py`
- `v4/scripts/run_protocol150_protocol101_paper_order_enablement_gate.py`
- `v4/scripts/run_protocol156_ibkr_autostart_observability.py`
- `v4/scripts/run_protocol157_protocol101_daily_ops_monitor.py`
- `v4/scripts/run_protocol158_protocol101_live_entry_paper_bridge.py`
- `v4/scripts/run_protocol160_protocol101_persistent_paper_trader.py`
- `v4/ops/ibkr/**`
- `v4/ops/launchd/**`

Runtime/account-risk files stay out of Git:

- `v4/runtime/**`
- `v4/logs/paper_trading/**`
- `.env`
- `v4/.env`

## Evidence Manifest Design

After approval, I will generate two manifests.

Local-only full manifest:

- Path: `/Users/gduby/Documents/autoresearch-stage1-evidence/v4-protocol101-operational-control-2026-05-24/STAGE1_FULL_EXCLUDED_ARTIFACTS_MANIFEST.jsonl`
- Not copied into the clean Git worktree.
- Not committed.

Each JSONL row will contain:

- `path`
- `size_bytes`
- `sha256`
- `reason_excluded`
- `category`
- `bundle_action`: `bundle_only`, `excluded_entirely`, or `sensitive_excluded`
- `sensitivity`: `public_path`, `sensitive_path`, `runtime_or_account`, `paper_log`, `paid_or_licensed_data`, or `unknown`

Allowed categories:

- `raw_data`
- `generated_audit`
- `paper_log`
- `model_binary`
- `paid_data`
- `runtime_state`
- `unknown`

The local-only full manifest will include all files in the source workspace that are not selected for Git tracking but are relevant to the dirty snapshot review, including ignored sensitive files like `.env` and `v4/.env` if present. `.env` and `v4/.env` will not be copied anywhere and their hashes will not be placed in a Git-tracked file.

Git-tracked redacted manifest:

- Path: `research_ops/bootstrap/STAGE1_REDACTED_MANIFEST.md`
- Contains category counts.
- Contains total bytes by category.
- Contains bundle-only file count.
- Contains excluded-entirely count.
- Contains sensitive-excluded count.
- Contains evidence bundle path.
- Contains evidence bundle sha256.
- Contains only non-sensitive example paths if needed.
- Contains no `.env` hashes, no credential material, no paper-log contents, no account identifiers, no raw paid-data contents, and no sensitive exact paths.

## Evidence Bundle Design

After approval, I will create a local-only evidence bundle outside Git:

- Directory: `/Users/gduby/Documents/autoresearch-stage1-evidence/v4-protocol101-operational-control-2026-05-24/files`
- Archive: `/Users/gduby/Documents/autoresearch-stage1-evidence/v4-protocol101-operational-control-2026-05-24.tar.gz`
- Bundle hash file in Git: `research_ops/bootstrap/STAGE1_EVIDENCE_BUNDLE_SHA256.txt`

The bundle will include only `bundle_only` files from the local-only full manifest, not all excluded files.

The bundle will not include paid-data parquet files, raw Databento data, ThetaData data, normalized market data, giant generated parquet datasets, rebuildable raw data, `.env`, or `v4/.env`.

## Final Pre-Commit Review Packet

Before any commit or tag, I will produce:

- `research_ops/bootstrap/STAGE1_PRE_COMMIT_REVIEW.md`

It will include:

- branch name
- source workspace
- clean worktree path
- number of files staged
- staged file list
- staged diff stat
- files excluded from Git by category
- evidence bundle path
- evidence bundle sha256
- secret-scan filename-only results
- large-file summary
- runtime/account-risk files excluded
- paid-data files excluded
- tests run
- any failures
- explicit recommendation: commit / do not commit

## Exact Commands I Intend To Run After Approval

These commands are proposed, not yet run.

```bash
SOURCE=/Users/gduby/Documents/autoresearch-trading
SNAP=/Users/gduby/Documents/autoresearch-trading-stage1-control-snapshot
BRANCH=stage1-control-snapshot-2026-05-24
TAG=v4-protocol101-operational-control-2026-05-24
BUNDLE_ROOT=/Users/gduby/Documents/autoresearch-stage1-evidence/v4-protocol101-operational-control-2026-05-24

cd "$SOURCE"
git status --porcelain=v1
git branch --list "$BRANCH"
test ! -e "$SNAP"
git worktree add -b "$BRANCH" "$SNAP" v4/phase-0

# Generate curated path lists, local-only full manifest, and Git-tracked redacted manifest.
# This will write:
# - research_ops/bootstrap/STAGE1_GIT_INCLUDE_PATHS.txt
# - local-only STAGE1_EVIDENCE_BUNDLE_PATHS.txt
# - local-only STAGE1_FULL_EXCLUDED_ARTIFACTS_MANIFEST.jsonl
# - research_ops/bootstrap/STAGE1_REDACTED_MANIFEST.md
python3 -m research_ops.bootstrap.stage1_snapshot_manifest_builder \
  --source "$SOURCE" \
  --git-include-out "$SOURCE/research_ops/bootstrap/STAGE1_GIT_INCLUDE_PATHS.txt" \
  --bundle-paths-out "$BUNDLE_ROOT/STAGE1_EVIDENCE_BUNDLE_PATHS.txt" \
  --full-manifest-out "$BUNDLE_ROOT/STAGE1_FULL_EXCLUDED_ARTIFACTS_MANIFEST.jsonl" \
  --redacted-manifest-out "$SOURCE/research_ops/bootstrap/STAGE1_REDACTED_MANIFEST.md"

# Copy only curated Git files into the clean worktree.
rsync -a --relative --files-from="$SOURCE/research_ops/bootstrap/STAGE1_GIT_INCLUDE_PATHS.txt" "$SOURCE"/ "$SNAP"/

# Create local-only evidence bundle outside Git.
mkdir -p "$BUNDLE_ROOT/files"
rsync -a --relative --files-from="$BUNDLE_ROOT/STAGE1_EVIDENCE_BUNDLE_PATHS.txt" "$SOURCE"/ "$BUNDLE_ROOT/files"/
tar -czf "$BUNDLE_ROOT.tar.gz" -C "$(dirname "$BUNDLE_ROOT")" "$(basename "$BUNDLE_ROOT")"
shasum -a 256 "$BUNDLE_ROOT.tar.gz" > "$SOURCE/research_ops/bootstrap/STAGE1_EVIDENCE_BUNDLE_SHA256.txt"
rsync -a --relative "$SOURCE/research_ops/bootstrap/STAGE1_REDACTED_MANIFEST.md" "$SNAP"/
rsync -a --relative "$SOURCE/research_ops/bootstrap/STAGE1_EVIDENCE_BUNDLE_SHA256.txt" "$SNAP"/

# Review staged candidate before commit.
cd "$SNAP"
git status --short
git add --pathspec-from-file="$SOURCE/research_ops/bootstrap/STAGE1_GIT_INCLUDE_PATHS.txt"
git add research_ops/bootstrap/STAGE1_REDACTED_MANIFEST.md
git add research_ops/bootstrap/STAGE1_EVIDENCE_BUNDLE_SHA256.txt
git diff --cached --name-only
git diff --cached --stat

# Verification only; no broker, paid-data, training, threshold tuning, or launchd commands.
# Filename-only scan; do not print matching lines.
rg -i -l "API_KEY|SECRET|TOKEN|PASSWORD|PASSWD|PRIVATE KEY|BEGIN RSA|BEGIN OPENSSH|DU[0-9]{3,}|account_id" \
  --glob '!research_ops/bootstrap/*FULL*' \
  --glob '!v4/logs/**' \
  --glob '!v4/runtime/**' \
  --glob '!.env' \
  --glob '!v4/.env' \
  .
python3 -m pytest \
  v4/tests/test_paid_data_guard.py \
  v4/tests/test_protocol142_paper_executor.py \
  v4/tests/test_protocol147_morning_session.py \
  v4/tests/test_optionsdx_ingest.py

# Produce final pre-commit review and stop.
python3 -m research_ops.bootstrap.stage1_precommit_review \
  --source "$SOURCE" \
  --snapshot "$SNAP" \
  --bundle "$BUNDLE_ROOT.tar.gz" \
  --full-manifest "$BUNDLE_ROOT/STAGE1_FULL_EXCLUDED_ARTIFACTS_MANIFEST.jsonl" \
  --out "$SOURCE/research_ops/bootstrap/STAGE1_PRE_COMMIT_REVIEW.md"
rsync -a --relative "$SOURCE/research_ops/bootstrap/STAGE1_PRE_COMMIT_REVIEW.md" "$SNAP"/
git add research_ops/bootstrap/STAGE1_PRE_COMMIT_REVIEW.md
git status --short
```

## Commands Explicitly Not Allowed

I will not run:

- `git clean`
- `git stash`
- broker/IBKR runtime scripts
- paid-data download scripts
- model training scripts
- threshold tuning scripts
- challenger promotion scripts
- launchd install/uninstall scripts
- `git push`

## Approval Questions

1. Is it acceptable to Git-track all v4 source code directories listed above, including broker-risk and paid-data-risk scripts, as inert code?
2. Should the local evidence bundle include only current-control binaries and key audit tables as proposed, or should it also include paid-data parquet files? Including paid-data parquet files would make the bundle much larger and potentially licensing-sensitive.
3. Should `.env` and `v4/.env` be included in the evidence manifest with path, size, and sha256 only, while contents remain excluded entirely?
