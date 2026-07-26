# Protocol101 Round 14 Review Request

Fable, please review this small governance patch before the Oct 2024 -> Jun 2025 acceptance rerun.

Your previous review identified three remaining actions:

1. Add synthetic unit coverage for `_raw_path_label`, including `stop_hit`, `target_hit`, `time_exit`, `forced_flat_capped`, `missing_future_path`, and the policy-1 15:30 ET boundary where `deadline == decision + hold` must not relabel as forced-flat.
2. Make the governed loader recompute and verify the embedded acceptance `registry_hash`, so a hand-edited registry cannot pass with stale self-reported status.
3. Replace CLI-only protected holdout discipline with a required governed artifact, hashed by default, with an explicit owner override token only for legitimate final protected-holdout evaluation.

Implemented changes:

- `run_protocol101_owned_raw_acceptance_verifier.py`
  - Added `REGISTRY_HASH_FIELDS`, `registry_hash_payload`, and `compute_registry_hash`.
  - `registry_hash` now binds status, verifier version, minimum verifier version, evidence scope, date range, counts, thresholds/defaults, fee model, quote-age pin, label policies, forced-flat pin, CBBO stamping assumption, governance checks, batch checks, label outcomes, strategy-selection flags, era/role hashes, session records, and placement predicates.

- `protocol101_governed_loader.py`
  - Imports `compute_registry_hash`.
  - Blocks on missing or mismatched embedded acceptance `registry_hash`.
  - Loads `v4/audit/autoresearch/protocol101_protected_holdout/summary.json` by default.
  - Blocks unless the protected holdout artifact has `status == pass` and a matching `protected_holdout_hash`.
  - Combines artifact-declared sessions with legacy `--protected-holdout-session` values.
  - Blocks protected sessions unless `OWNER_APPROVED_PROTECTED_HOLDOUT_EVALUATION` is explicitly supplied.

- `build_protocol101_protected_holdout_artifact.py`
  - New script to build a hashed `Protocol101ProtectedHoldoutArtifactV1`.
  - Generates `summary.json`, `protected_holdout.json`, and `report.md`.
  - With no owner-declared sessions it emits `pending_owner_declaration`, which intentionally blocks the loader by default.

- `run_protocol101_fair_contract_training_runner.py`
  - Adds `--protected-holdout` and `--protected-holdout-owner-override-token`.
  - Passes both into the governed loader.

- Tests:
  - Added synthetic `_raw_path_label` branch coverage.
  - Added loader tests for registry hash mismatch, missing/unpassed protected holdout artifact, artifact-declared protected session blocking, and owner override.
  - Updated fair-contract training runner fixtures to include a hashed registry and passing holdout artifact.

Verification run by Codex:

```text
PYTHONPATH=. .venv/bin/python -m py_compile \
  v4/scripts/run_protocol101_owned_raw_acceptance_verifier.py \
  v4/scripts/build_protocol101_protected_holdout_artifact.py \
  v4/model/protocol101_governed_loader.py \
  v4/scripts/run_protocol101_fair_contract_training_runner.py \
  v4/tests/test_protocol101_governed_loader.py \
  v4/tests/test_protocol101_owned_raw_acceptance_verifier.py \
  v4/tests/test_protocol101_fair_contract_training_runner.py

PASS
```

```text
PYTHONPATH=. .venv/bin/python -m pytest -q \
  v4/tests/test_protocol101_governed_loader.py \
  v4/tests/test_protocol101_owned_raw_acceptance_verifier.py

26 passed in 1.06s
```

```text
PYTHONPATH=. .venv/bin/python -m v4.scripts.build_protocol101_protected_holdout_artifact \
  --out-dir /tmp/protocol101_holdout_smoke \
  --session 2099-12-31 \
  --owner-note smoke

status=pass
```

```text
rg -n "NeuralDatasetConfig|_policy_exit_deadline" \
  v4/scripts/run_protocol101_owned_raw_acceptance_verifier.py \
  v4/model/protocol101_governed_loader.py \
  v4/scripts/build_protocol101_protected_holdout_artifact.py

no matches
```

Known caveat:

- `pytest -q v4/tests/test_protocol101_fair_contract_training_runner.py -k "training_mode_requires_explicit_owner_approval_flags or dry_run_plan_is_ready_without_training_authorization"` was interrupted after 68 seconds because pytest was still stuck in import collection (`importlib._bootstrap_external`). The changed file itself passed `py_compile`, and the governed loader paths that the runner delegates to passed directly. Please inspect whether this import stall is unrelated dependency/import heaviness or a new issue.

Current explicit blocker:

- A default protected-holdout artifact now exists at `v4/audit/autoresearch/protocol101_protected_holdout/summary.json`, but it is intentionally `pending_owner_declaration` with zero sessions. The owner still needs to declare the actual lockbox sessions before the first governed fold. Until then, the loader should block by default.

Please review:

1. Is the registry hash payload complete enough, or should it bind any additional top-level fields?
2. Is the protected holdout override appropriately narrow?
3. Does the synthetic `_raw_path_label` test cover the forced-flat and boundary semantics you wanted?
4. Is the pending holdout artifact approach acceptable before owner declaration?
5. Do you see any remaining acceptance-phase blockers before materialization and the Oct-Jun rerun?
