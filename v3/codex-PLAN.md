# Robust Training Regimen for the v3 Trading System

## Summary

Rebuild training around one principle: **the model must train on the richest truthful surface available, and promotion decisions must come only from honest rolling-window evaluation**. The new regimen will replace the old “entry regressor + side regressor + hand-coded overrides + heuristic strike pick” stack with a staged policy system:

1. **Layer 0 stays as execution safety rails only**
2. **Layer 1 becomes feature-only prior information**
3. **Layer 2 becomes a unified action scorer over `flat` + real contract candidates**
4. **Layer 2.5 becomes an auxiliary patience/stopout-learning task, not a separate tiny sidecar**
5. **Layer 3 remains a rolling learned exit, calibrated only on prior-window validation trades**
6. **No RL until the supervised rolling stack clears promotion gates**

This plan optimizes for **walk-forward robustness and live deployability**, not peak in-sample PF.

## Implementation Changes

### 1. Lock the training and promotion contract

- Make the **13-window rolling harness** the only source of truth for champion promotion.
- Keep the old **5-fold harness** only for smoke tests, architecture checks, and fast ablations.
- Define three run tiers:
  - `smoke`: latest fold/window only, CPU, short epoch budget
  - `dev`: full 5-fold or reduced rolling subset, CPU or single-GPU, 1 seed
  - `promotion`: full 13-window rolling run, 3 seeds, slippage stress, no test-time retuning
- Promotion metrics reported on every candidate:
  - aggregate PF, DD, mean/trade, trade share
  - per-window PF mean/std/min/max
  - clean-entry rate, fast-loser rate, shakeout-winner rate
  - direction consistency vs V1 baseline
  - slippage stress at `$0`, `$10`, `$25` round-trip
- Freeze all model selection on **validation only**. No threshold or hyperparameter can be chosen from the same OOS windows used for final reporting.

### 2. Redesign the training surface for all downstream layers

- Replace the flat Layer-2 export as the champion dataset with a **surface dataset** that includes:
  - scalar bar-state features
  - last `20` bars of sequential context as baseline
  - top `12` contract tokens per side from **all contract-valid rows**, not just passing contracts
  - per-contract gate booleans so the model can see guardrail pressure
- For each contract token, export labels for:
  - time-stop PnL
  - best-exit PnL
  - `5/10/20` minute MFE/MAE
  - “stopout before move” proxy labels
- Export all bars in the current execution window, fixed at **`09:45–11:30 ET`** for this phase.
- Keep Layer 0 unchanged for execution, but expose all guardrail-bind information to training.
- Keep teacher outputs in the dataset, but remove them from any champion-time direction override logic.

### 3. Replace the Layer-2 decomposition with one unified policy model

- Deprecate the current champion path based on:
  - separate `entry` and `side` regressors
  - `score_mode`
  - `entry_threshold × side_threshold`
  - `direction_mode`
  - heuristic `select_contract`
  - V0–V4 post-hoc directional variants
- Replace it with one action model over:
  - `flat`
  - `call_contract_1..12`
  - `put_contract_1..12`
- Model architecture for the first unified branch:
  - scalar encoder for tabular bar-state
  - sequential encoder over the `20`-bar history
  - contract-token encoder shared across calls and puts
  - shared trunk over concatenated state + per-side contract summaries
  - action head scoring every candidate action
- Training targets:
  - primary target: `arcsinh(time_stop_pnl / 100)` for each candidate action
  - auxiliary head 1: `clean_entry_prob`
  - auxiliary head 2: `stopout_risk`
- Training loss:
  - pairwise ranking loss so the best action for a day outranks weaker actions and `flat`
  - utility regression on time-stop value
  - BCE losses for `clean_entry_prob` and `stopout_risk`
- Inference policy:
  - pick the highest-scoring action on each bar
  - choose the day’s best bar/action pair
  - execute only if `best_nonflat_score - flat_score >= decision_margin`
- Calibrate exactly one scalar `decision_margin` on validation windows. Remove the old threshold grid from the champion path.

### 4. Fold Layer 2.5 into the policy training regimen

- Keep the current standalone Layer `2.5` scripts as legacy benchmarks only.
- Move patience learning into the unified policy as auxiliary supervision.
- Default labels:
  - `clean_entry`: profitable by time-stop and early MAE stays above the configured floor
  - `stopout_risk`: adverse excursion crosses the floor before favorable excursion reaches the target horizon
- Report these labels on:
  - all scored bars
  - chosen policy trades
  - per-window promotion summaries
- Success condition for the patience component:
  - lower fast-loser share
  - lower shakeout frequency
  - improved PF at comparable or intentionally reduced trade share

### 5. Keep Layer 3 rolling and honest, then make it part of a controlled outer loop

- Train Layer 3 only on trades generated by the current unified policy from **prior rolling windows**.
- Inputs:
  - exact chosen contract path
  - same bar-state sequence context
  - trade-state features
- Exit threshold chosen only on prior-window validation trades from a fixed grid:
  - default grid: `0.10, 0.15, 0.19, 0.25, 0.30`
- Early windows with insufficient trade history fall back to **time-stop**, not a new heuristic.
- Run Layer 3 in a two-stage outer loop:
  1. train unified entry policy against time-stop labels
  2. train rolling Layer 3 on resulting trades
  3. regenerate composed trade outcomes
  4. retrain unified policy once against composed-exit utilities
- Limit the outer loop to **2 iterations per experiment** to prevent endless circular fitting.

### 6. GPU policy for the new regimen

- CPU remains the default for:
  - smoke tests
  - single-fold checks
  - exporter validation
  - label-generation debugging
- GPU becomes mandatory for:
  - any full 13-window unified policy run
  - any 3-seed promotion run
  - any model larger than the current W1 bridge branch
  - any experiment with sequence length `>20`, top-K `>12`, or `>20` epochs
- Do not spend GPU on legacy flat-feature branches anymore.

## Public Interfaces and Artifact Changes

- New canonical training bundle shape:
  - `rows`
  - `sequence_features`
  - `sequence_mask`
  - `contract_features`
  - `contract_mask`
  - `action_labels`
  - metadata describing sequence length, token schema, execution window, and guardrail semantics
- New champion replay artifact must include:
  - chosen action id
  - chosen side
  - chosen strike/right
  - flat-vs-trade decision margin
  - clean-entry and stopout predictions
- Legacy interfaces remain loadable for benchmarking, but the champion path should mark these as deprecated:
  - `direction_mode`
  - `score_mode`
  - `side_score_weight`
  - heuristic contract selection in replay

## Test Plan and Promotion Gates

### Dataset and labeling tests

- Row alignment test: scalar rows, sequence windows, contract tokens, and labels must have identical row count and day/bar identity.
- Contract-token determinism test: token ordering must be stable across runs.
- Guardrail visibility test: blocked contracts must appear in training tokens with correct gate flags.
- Label parity test: for equivalent selected contracts, time-stop labels must match the current PnL convention exactly.

### Model and replay tests

- Smoke test: latest window trains and replays end-to-end on CPU with no file-shape mismatches.
- Unified-action sanity test: replay can choose `flat` and can choose a real contract without teacher override.
- Contract-choice sanity test: chosen strike/right must be guardrail-passing at execution time.
- Exit sanity test: rolling Layer 3 never trains on same-window trades.

### Promotion gates

- **W1 gate, time-stop only**:
  - aggregate rolling PF must beat the current honest `V0 + time-stop` baseline of `1.132`
  - DD must be no worse than the current Layer `2.5` time-stop baseline unless PF improves by at least `+0.15`
  - trade share must stay between `0.25` and `0.70`
- **Patience gate**:
  - fast-loser share must improve by at least `20%` relative to the current chosen-trade baseline
- **Exit gate**:
  - rolling Layer 3 must beat the unified time-stop baseline by at least `+0.10 PF`
  - must survive `$10` and `$25` slippage stress without dropping below the unified time-stop PF
- **Champion gate**:
  - promotion requires the same result shape on all 3 promotion seeds
  - no seed may fail below PF `1.0`

## Assumptions and Defaults

- Primary objective is **robust walk-forward trade quality**, not maximum headline PF.
- Entry window stays at **`09:45–11:30 ET`** until the unified policy is stable.
- Layer 0 rails remain unchanged and execution-enforced.
- Teachers remain **feature-only** in the champion path.
- Sizing remains fixed at **1 contract** during this redesign.
- RL / agent training is explicitly out of scope until the unified supervised rolling stack clears the promotion gates above.
- Existing flat Layer-2, standalone Layer 2.5, and old Layer 3 code stay in the repo as benchmarks only and should not receive new tuning effort except for parity comparisons.
