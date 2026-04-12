# Documentation Surface Cleanup — 2026-04-10

Purpose: reduce live-path context drift inside `v2/` and keep the exact-chain reset documentation surface intentionally small.

## Archived Out Of `v2/`

- `v2/live/` -> `archive/v2_historical/live/`
- `v2/analysis/analyze_whipsaw.py`
- `v2/analysis/contract_drift_audit.py`
- `v2/analysis/diagnostic_direction_signal.py`
- `v2/analysis/diagnostic_oracle_replay.py`
- `v2/pipeline/build_dataset.py`
- `v2/pipeline/download_wide_grid.py`
- `v2/pipeline/extract_raw.py`
- `v2/pipeline/relabel_tier3.py`
- `v2/docs/goal.md`
- `v2/docs/baselines.md`
- `v2/FRESH_SESSION_HANDOFF.md`
- `v2/data/`
- `v2/data.pt.bak`
- `v2/data_harness_repair.pt`
- `v2/data_pre_harness_repair.pt`

## Rationale

- `v2/live/` was a stub-only future surface, not part of the current exact-chain loop.
- The archived `analysis/` scripts were one-off repair diagnostics or stale analyses, not official workflow tools.
- The archived `pipeline/` scripts described superseded ingestion or labeling regimes and competed with `build_v2_dataset.py`.
- `goal.md`, `baselines.md`, and `FRESH_SESSION_HANDOFF.md` duplicated information already covered by the live docs.
- The archived data files were backups or repair-era manifests, not the canonical live dataset.

## Live Docs After Cleanup

- `v2/HANDOFF.md`
- `v2/program.md`
- `v2/COMMANDS.md`
- `v2/LAYOUT.md`
- `v2/docs/README.md`
- `v2/docs/current_state.md`
- `v2/docs/how_training_works.md`
- `v2/docs/data_contract.md`
- `v2/docs/feature_schema.md`
- `v2/docs/labeling.md`
- `v2/docs/contracts.md`
- `v2/docs/evaluator.md`
- `v2/lab_notebook.md`
- `v2/results.tsv`

## Enforcement

`v2/ops/pre_run_gate.py` now fails if the archived live-path files or folders are reintroduced into `v2/`.
