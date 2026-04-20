# Current v2 State

Last refreshed: 2026-04-18

This file is a state snapshot -- what IS true right now. For the operating loop, see `v2/ART2_LOOP.md`. For commands, see `v2/COMMANDS.md`.

## Mission And Phase

- Mission: build a trustworthy exact-chain research system that can hill-climb toward a profitable SPX 0DTE long-options model
- Current phase: **slice-first supervised regime with Direction B abstention architecture now implemented**
- Live trading: deferred until cross-validation is stable and profitable
- The current mission is still research-system reliability first, not paper/live deployment.

## Snapshot

- Active manifest: `v2/data.pt` (rebuilt 2026-04-18 after dynamic-slice feature expansion)
- Dataset version: `v5_exact_chain_v2_slice`
- Dataset fingerprint: `6162cf3d83db3586`
- Context features: 79 (legacy market context + local surface + session-structure additions)
- Contract features: 22 (19 base + 3 economic: theta_to_premium, breakeven_bars_est, gamma_dollar)
- Per-day sidecars: `v2/data_sidecars/*.pt` (schema `v5_exact_chain_v2_slice`)
- Unique days: 986
- Scoring: **v4.0 dollar-weighted PF/sortino** (DD gate 25%, penalty-free ≤12%)
- Previous scoring (v3.0 percentage-weighted PF) was found to mask an 81.5% portfolio loss as near-breakeven. See `docs/incidents/2026-04-15-pf-metric-bug.md`.
- **All prior experiment scores are stale** — evaluator fingerprint changed
- Current promoted model is absent locally; fresh Direction B screening is the next official step
- exp_171 is the canonical post-repair supervised baseline; exp_next_a / exp_next_a2 falsified slice-only and scorer-only fixes

## End-To-End Flow

```text
raw SPX/SPY/VIX pickles + full-chain SPXW pickles
    -> build_v2_dataset.py
    -> v2/data.pt + v2/data_sidecars/
    -> train.py + core/policy.py
    -> ops/pre_run_gate.py
    -> ops/run_experiment_wf.py
    -> v2/models/model_candidate.pt + v2/artifacts/exp_NNN/
    -> model_manage.py keep/revert
    -> promoted artifact + v2/models/model_best.pt
    -> replay.py / analysis / plots
```

## What The Repo Currently Does

### Data

- The manifest stores 79 normalized market-context features in `X`, raw replay features in `X_sim`, plus bar metadata and split masks.
- Exact contracts do not live inside the manifest tensor payload.
- Each day sidecar stores exact contract identities, executable snapshots, and per-contract forward P&L labels under the fixed policy.
- Each sidecar also stores a dynamic near-ATM tradable slice (ATM ±10 strikes, updated bar-by-bar) and slice-aware best-contract metadata.
- Hard or incomplete market days are retained. Missing forward paths are flagged at the contract/bar level instead of dropping whole days.

### Training

- Training is from scratch every experiment.
- The default mutable research surface remains:
  - `v2/train.py`
  - `v2/core/policy.py`
- The approved expanded mutable surface for the current Kronos-inspired block is:
  - `v2/train.py`
  - `v2/core/metrics.py`
  - `v2/replay.py`
  - `v2/core/data_integrity.py`
  - `v2/ops/pre_run_gate.py`
  - live docs that must stay in sync with the active protocol
- The current supervised problem is now explicitly two-stage:
  - gate the bar
  - if the gate passes, rank contracts inside the dynamic slice
- Direction B support is implemented:
  - `GATE_ARCH=decoupled_mlp` creates a standalone gate path from raw lookback summaries
  - `OPP_LABEL=sparse_high_conviction` teaches abstention-first bars
  - `POLICY_GATE_THRESHOLD_MODE=quantile` gates by target pass rate instead of a fixed zero logit
- The recommended next screen uses `TRAIN_FEATURE_SET=full79`, `LINEAR_SCORE_HEADS=0`, `GATE_ARCH=decoupled_mlp`, `OPP_LABEL=sparse_high_conviction`, `SEL_TARGET_MODE=soft_pnl`, `SOFT_TEMP=0.40`, and quantile replay gating at 10% pass rate.
- The current official policy window is full supervised day (`bar 30` through `270`). Widened from 60-105 after window audit showed no structural edge concentration.
- Direction mix is now diagnostic output, not a hard score gate.
- Risk is policy-driven, not learned, in the frozen v4 harness.

### Replay

- Replay uses exact contract identity from sidecars.
- Replay trades only inside the dynamic slice by default; full-chain is retained for diagnostics only.
- No ATM/OTM ladder fallback remains in the main scoring path.
- Entry fills are next-bar.
- Trade simulation uses the same stop / target / trailing / hold rules as policy and still charges spread plus commission.
- Replay can now resolve the live gate by quantile, and reports the resolved threshold plus realized pass rate in metrics.

### Scope Boundary

- The live `v2/` path is now only data, training, replay, experiment management, and the docs needed to operate that system.
- Paper/live trading stubs, old dataset builders, and one-off repair diagnostics were archived out of `v2/` to reduce context drift.

### Harness Eval Suite

- `core_regression` protects non-negotiable invariants.
- `optimization` cases are used during harness repair.
- `holdout` cases verify that harness fixes generalize.

## Operating Loop

See [ART2_LOOP.md](../ART2_LOOP.md) for the canonical hill-climbing protocol (preconditions, training, validation, promotion, paper-trading readiness).
