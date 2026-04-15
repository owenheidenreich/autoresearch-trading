# ART2 Pipeline Audit

**Date:** 2026-04-15
**Auditor:** Claude (pipeline integrity session)
**Scope:** End-to-end verification of every pipeline stage, fingerprint chain, and operational command.

---

## Summary

| Stage | Status | Issues Found |
|-------|--------|-------------|
| 1. Data Acquisition | PASS | 0 |
| 2. Dataset Build | PASS (code review) | 1 weakness |
| 3. Supervised Training | PASS (ran locally) | 0 |
| 4. Sequential BC Training | PASS (ran locally) | 0 |
| 5. Trajectory Collection | PASS (ran locally) | 0 |
| 6. AWAC Training | PASS (ran locally) | 0 |
| 7. Supervised Replay | PASS (ran 3-day + full) | 1 bug found |
| 8. Sequential Replay | PASS (ran 3-day) | 0 |
| 9. Visualization | PASS (both tools) | 0 |
| 10. EvalReport | PASS (generated + verified) | 0 |
| 11. Health Checks | PASS (all 4 checks) | 0 |
| 12. Pre-Run Gate | PASS | 0 |
| 13. Fingerprint Chain | PASS (all match) | 0 |
| 14. Deploy Bundle | PASS (new files included) | 0 |
| **Baselines** | **BUG** | **1 real bug** |

---

## Stage-by-Stage Findings

### Stage 1: Data Acquisition (`pipeline/download_full_chain.py`)

**Status:** PASS (code review only — not executed)

- Downloads from Polygon S3 flat files. Credentials from `.env`.
- Stores to `~/.cache/autoresearch-trading/data/spxw_full_chain/`.
- Handles 0DTE day detection correctly (MWF pre-2022-05-11, all weekdays after).
- Error handling: failed days are logged and skipped, not fatal.
- No fingerprinting issues — this is raw data, upstream of everything.

**No issues found.**

### Stage 2: Dataset Build (`pipeline/build_v2_dataset.py`)

**Status:** PASS (code review — not executed because rebuild would overwrite current data.pt)

- Imports from correct modules (`core/chain_data`, `core/features`, `core/policy`, `core/simulator`).
- Produces `data.pt` with metadata dict containing all stage contract fields.
- New fields added: `config_fingerprint`, `lookback`, `build_git_sha`, `feature_names`.
- Feature assertion: `assert X_combined.shape[1] == NUM_FEATURES` (line 939).
- Sidecar digest computed for integrity tracking.

**Weakness W-001:** `lookback` is hardcoded to `30` in the metadata dict (line 991) rather than imported from RuntimeConfig. If LOOKBACK changes in config, the build script would embed the old value. Should be:
```python
"lookback": RUNTIME_CONFIG.lookback,  # instead of hardcoded 30
```
**Severity:** Low. LOOKBACK has been 30 for the entire project and is unlikely to change. But it violates the single-source-of-truth principle.

### Stage 3: Supervised Training (`train.py`)

**Status:** PASS — ran locally (3 epochs, 1 fold, 328,650 train samples)

- Load-time validation: checks dataset metadata against RuntimeConfig. Hard failure on mismatch.
- Checkpoint saves: `config_fingerprint`, `env_overrides`, `dataset_fingerprint`, `score_config_fingerprint`, full hyperparams.
- Env override capture: all 24 training-relevant env vars captured via `_capture_env_overrides()`.
- Produced compatible model.pt (contract_features=22, lookback=30).

**No issues found.** Training pipeline is clean.

### Stage 4: Sequential BC Training (`train_seq.py`)

**Status:** PASS — ran locally (20 epochs, 78.7% val accuracy)

- Loads frozen encoder from model.pt correctly.
- Produces `seq_agent.pt` with `agent_state_dict`.
- Session state dim = 13 (correct, matches `env.py` SESSION_STATE_DIM).

**No issues found.**

### Stage 5: Trajectory Collection (`collect_trajectories.py`)

**Status:** PASS — ran locally (666 days, 29,970 steps)

- Requires `--checkpoint` to be a seq_agent checkpoint (not model.pt — has `agent_state_dict` key).
- Reward audit output: 10.5% of steps have env/train reward divergence (expected with adaptive spread).
- Saved to `v2/trajectories/fold0_bc_fresh.pt` (422 MB).

**No issues found.**

### Stage 6: AWAC Training (`train_awac.py`)

**Status:** PASS — ran locally (10 value pre-training + 100 AWAC epochs)

- Loads encoder + BC checkpoint + trajectory data correctly.
- Policy loss converges (0.095 → 0.017).
- Value loss converges (3.26 → 1.01).
- KL divergence stays low (0.0008 → 0.0017).
- Best epoch: 5 (early, suggesting later epochs may overfit).

**No issues found.**

### Stage 7: Supervised Replay (`replay.py`)

**Status:** PASS — ran both 3-day and full promote_mask (60 days, 207 trades)

- Load-time validation: warns (does not hard-fail) on dataset/config mismatch.
- Dollar-weighted PF correctly computed (PF=0.885 on 3-day, PF=0.61 on full).
- EvalReport generated with stored trades and fingerprints.
- Trades have correct qty=1, strike, right, entry_price.
- Gate failures reported correctly (too_few_trades on 3-day, excessive_drawdown on full).

**No issues found in replay itself.** But see BUG-001 below for baselines.

### Stage 8: Sequential Replay (`replay.py --sequential`)

**Status:** PASS — ran 3-day test

- Loads encoder + seq_agent correctly.
- Produces behavioral report (hold rate, side flips, action distribution).
- Trades have correct intent from `env._last_closed_intent` (qty=1, strike, right populated).
- Dollar PF=1.865 on 3 days (not meaningful sample, but pipeline works).

**No issues found.**

### Stage 9: Visualization

**`plot_trades.py`:** PASS — generated trades.html, equity.html, trades.csv (207 trades on full promote_mask).

**`plot_progress.py`:** PASS — generated progress.png from results.tsv.

**No issues found.** Both tools work end-to-end.

### Stage 10: EvalReport (`core/eval_report.py`)

**Status:** PASS — generated and verified

Stored report contents:
- `report_id`: unique ID
- `evaluator_fingerprint`: matches current `score_config_fingerprint()`
- `config_fingerprint`: matches RuntimeConfig
- `dataset_fingerprint`: matches data.pt
- `trades`: 10 trade dicts with full intent (qty=1, strike, right, entry_price)
- `daily_equity_curve`: correct length
- `baselines`: captured

**No issues found.**

### Stage 11: Health Checks (`ops/health.py`)

**Status:** PASS — all 4 checks pass

```
config   PASS
data     PASS
model    PASS
smoke    PASS
```

**No issues found.**

### Stage 12: Pre-Run Gate (`ops/pre_run_gate.py`)

**Status:** PASS

- Validates required docs, code patterns, results.tsv format, data integrity, harness eval.
- Does NOT validate RuntimeConfig fingerprint chain (relies on health.py for that).

**No issues found.**

### Stage 13: Fingerprint Chain

**Status:** PASS — all fingerprints match across the full chain

```
RuntimeConfig  -> 8f59591c63a8b071
data.pt        -> config_fp: 8f59591c63a8b071 ✓   dataset_fp: b422dcbce1d15cfc
model.pt       -> config_fp: 8f59591c63a8b071 ✓   dataset_fp: b422dcbce1d15cfc ✓   score_fp: e45320cc6094cd1e ✓
eval_report    -> config_fp: 8f59591c63a8b071 ✓   dataset_fp: b422dcbce1d15cfc ✓   eval_fp:  e45320cc6094cd1e ✓
```

**No issues found.** This chain correctly detects staleness when any upstream component changes.

### Stage 14: Deploy Bundle (`ops/deploy.sh`)

**Status:** PASS — verified tar contents

New files `core/config.py`, `core/eval_report.py`, `ops/health.py` are included in the deploy bundle. `archive_quarantine/` is excluded.

**No issues found.**

---

## Bugs Found

### BUG-001: ATM baselines produce 0 trades (MEDIUM severity)

**Location:** `v2/replay.py` lines 756 and 796

**Problem:** `compute_baseline_atm_always()` and `compute_baseline_atm_trailing()` both hardcode `local_bar = 30` as the entry bar. The current policy window is bars 60–105. At bar 30, most contracts either don't exist yet or fail the executability filters (spread too wide, no volume). Result: both baselines return 0 trades and score 0.0.

**Impact:** The promotion gate "beats all baselines" (`G5`) is trivially satisfied because two of the four baselines always score 0. This means a model could be promoted without actually beating a meaningful ATM-always benchmark.

**Evidence:**
```
ATM-Always:   Score=0.0  PF=0.0  Trades=0
ATM-Trailing: Score=0.0  PF=0.0  Trades=0
```

**Fix:** Change `local_bar = 30` to `local_bar = policy.no_trade_before_bar` (which is 60) in both functions. This makes the baselines enter during the actual trading window.

---

## Weaknesses Found

### W-001: Hardcoded lookback in build_v2_dataset.py (LOW)

**Location:** `v2/pipeline/build_v2_dataset.py` line 991

**Problem:** `"lookback": 30` is hardcoded instead of importing from RuntimeConfig. If LOOKBACK changes, the metadata would embed the wrong value.

**Fix:** `"lookback": RUNTIME_CONFIG.lookback`

### W-002: Replay validation is WARNING not ERROR (LOW)

**Location:** `v2/replay.py` line 883

**Problem:** When dataset metadata doesn't match RuntimeConfig, replay prints a WARNING but continues. Training raises a hard RuntimeError for the same check. Inconsistent severity.

**Recommendation:** For safety, replay should also fail hard when the config mismatch is dimensional (num_features, lookback). Warnings are appropriate for fingerprint-only mismatches (dataset was built under a different config fingerprint but dimensions still match).

### W-003: Model trained with only 3 epochs (EXPECTED)

The current model.pt was trained with `TRAIN_EPOCHS=3` (local pipeline verification, not production). It will need a proper 5-fold GPU training run before being used for research conclusions. This is understood and expected — the local training was to verify the pipeline, not to produce a production model.

### W-004: AWAC best epoch is 5 out of 100 (INFORMATIONAL)

AWAC training selected epoch 5 as best based on val_metric (-0.332). Later epochs had lower loss but worse validation performance. This suggests the AWAC fine-tuning may overfit with too many epochs. Consider reducing AWAC epochs or adding early stopping based on val_metric.

---

## Operational Verification Summary

| Command | Result |
|---------|--------|
| `python -m v2.ops.health` | 4/4 PASS |
| `python -m v2.ops.health quick` | 3/3 PASS |
| `python -m v2.ops.pre_run_gate --data v2/data.pt` | PASS |
| `python -m v2.replay --model v2/models/model.pt --mask promote --days 3` | 10 trades, PF=0.885, EvalReport saved |
| `python -m v2.replay --sequential --seq-model v2/models/seq_agent.pt --days 3` | 12 trades, PF=1.865, behavioral report generated |
| `python -m v2.plot_trades --model v2/models/model.pt` | trades.html + equity.html + trades.csv (207 trades) |
| `python -m v2.plot_progress` | progress.png saved |
| Fingerprint chain (config → data → model → eval) | All match ✓ |
| Deploy bundle includes new files | config.py, eval_report.py, health.py ✓ |

---

## Action Items

| Priority | Item | Status |
|----------|------|--------|
| **HIGH** | Fix BUG-001: ATM baselines hardcoded bar 30 → policy.no_trade_before_bar | Open |
| LOW | Fix W-001: hardcoded lookback in build_v2_dataset.py | Open |
| LOW | Fix W-002: replay config validation severity | Open |
| — | W-003: Retrain model on GPU (5-fold, full epochs) | Deferred to next GPU session |
| — | W-004: Investigate AWAC overfitting / early stopping | Deferred to next research cycle |
