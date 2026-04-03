# v2 Migration Plan

## Purpose

Defines what stays active, what is frozen, when root names become available,
and when v1 tests stop gating v2.

---

## Coexistence Model

During v2 development, both systems exist simultaneously:

```
Active v1 (unchanged):
  training/          <- v1 train/replay/prepare
  training/live/     <- v1 IBKR execution
  tools/             <- v1 inner_loop, daily_pipeline
  infra/             <- v1 deployment
  tests/             <- v1 test suite

Frozen reference:
  legacy_v1/         <- snapshot of v1, never imported or modified

Active v2 (under construction):
  v2/core/           <- TradeIntent, features, labels, simulator, metrics
  v2/live/           <- IBKR execution via TradeIntent
  v2/ops/            <- experiment orchestrator
  v2/pipeline/       <- dataset construction
  tests/v2/          <- v2 test suite
  docs/v2/           <- design documents
```

v1 and v2 share:
- Raw data caches (data/spx_1min/, data/spy_1min/, etc.) -- read-only for both
- IBKR connection (not simultaneously -- one system runs at a time)
- docs/domain/ (permanent domain knowledge)

v1 and v2 do NOT share:
- Model weights (different paths)
- data.pt (different paths, different label schemes)
- Results (different paths: results/ vs results/v2/)
- Test suites (tests/ vs tests/v2/)

---

## Cutover Sequence

v2 components become production-ready in this order. Each step has an
explicit "done when" gate.

### Step 1: Core Schema
**Build:** v2/core/schema.py (TradeIntent, RiskAdjustment)
**Done when:** Dataclass is importable, validation methods work, serialization
round-trips correctly. Tests in tests/v2/test_schema.py pass.

### Step 2: Simulator
**Build:** v2/core/simulator.py
**Done when:** Simulator produces identical trade logs to v1 replay.py on the
same input (parity test). Determinism test passes (two runs = identical output).

### Step 3: Metrics
**Build:** v2/core/metrics.py
**Done when:** Score formula produces identical scores to v1 on the same trade
logs (parity test). Score config fingerprinting works.

### Step 4: Baselines
**Build:** v2/core/baselines.py (or integrated into metrics)
**Done when:** Random, ATM-always, and simple-rules baselines are computed and
cached on the validation set.

### Step 5: Features + Labels
**Build:** v2/core/features.py, v2/core/labels.py
**Done when:** Features produce identical output to v1 prepare.py compute_features()
on the same raw data (parity test). Oracle labels pass quality checks (labeling.md).

### Step 6: Dataset Pipeline
**Build:** v2/pipeline/build_dataset.py
**Done when:** Produces a valid v2 data.pt with oracle labels. Fingerprinting works.
Incremental update works.

### Step 7: Training
**Build:** v2/train.py
**Done when:** Model trains, loss decreases, model beats all three baselines on
held-out validation days.

### Step 8: Live Trading
**Build:** v2/live/
**Done when:** Full RTH paper trading session (9:30-16:00 ET) without crashes.
Same TradeIntent flows through decision -> execution -> audit log.

### Step 9: Cutover
**Action:** Promote v2 to primary system.
- Move v2/ contents to root-level paths
- Convert all 38+ bare imports to explicit package paths
- Update CLAUDE.md for v2 contracts
- Move remaining v1 code into legacy_v1/
- Rename tests/v2/ to tests/
- Archive v1 tests to legacy_v1/tests/

**Done when:** All tests pass, paper trading runs for 3+ full days without issues.

---

## When Root Names Become Available

Root-level `prepare.py`, `train.py`, `replay.py` are NOT created until Step 9.

Before that, all v2 code is imported as `v2.core.X`, `v2.live.X`, etc.

The cutover requires converting 38+ bare imports across these files:
- tests/test_training_v5.py (18 bare imports)
- training/replay.py (4 bare imports)
- training/train.py, best_train.py (2 bare imports each)
- training/trading_rules.py (1 bare import)
- training/live/decision.py (1 bare import)
- tools/inner_loop.py (1 bare import)

This is a mechanical refactor: `from prepare import X` -> `from training.prepare import X`.
But it touches active v1 code and requires running the full v1 test suite after.
It happens at cutover, not before.

---

## Test Partitioning

### During Coexistence

- `tests/` -- v1 tests. Gate v1 changes. Run with `pytest tests/ -x`.
- `tests/v2/` -- v2 tests. Gate v2 changes. Run with `pytest tests/v2/ -x`.
- Both suites run before any commit that touches shared infrastructure.

### v2 Test Naming

```
tests/v2/
  conftest.py              # v2-specific fixtures
  test_schema.py           # TradeIntent validation, serialization
  test_simulator.py        # Trade simulation, determinism, parity
  test_metrics.py          # Score formula, baselines
  test_features.py         # Feature computation, normalization
  test_labels.py           # Oracle labeler quality checks
  test_pipeline.py         # Dataset build, fingerprinting
  test_live_decision.py    # Model -> TradeIntent
  test_live_execution.py   # IBKR order lifecycle
  test_import_guard.py     # No v1 imports in v2 code
```

### After Cutover

v1 tests archived to legacy_v1/tests/. No longer run in CI.
v2 tests become the primary test suite at tests/.

---

## Risk Mitigation

### "v2 breaks v1" Risk

Mitigated by:
- All v2 code under v2/ namespace (no path collisions)
- v1 code never modified during v2 development
- Import guard test prevents v2 from depending on v1
- Both test suites run independently

### "v2 never reaches parity" Risk

Mitigated by:
- Step-by-step cutover sequence with explicit gates
- Parity tests (simulator, metrics, features) prove equivalence
- v1 continues to function until v2 is proven

### "Stale v1 diverges from v2" Risk

Mitigated by:
- v1 code is frozen after Phase 0 (no new v1 features)
- v1 experiments may continue but only within existing contracts
- Any v1 finding worth keeping is ported to v2, not bolted onto v1
