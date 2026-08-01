# Protocol101 Path-D Pre-Fit Build Correction Report

Date: 2026-08-01

Authority: `v4/docs/protocol101/training/execution/PROTOCOL101_PATHD_BUILD_CORRECTION_PROPOSAL_2026_08_01.md`

Corrected generation: `v4/audit/autoresearch/protocol101_pathd_entry_exit_model_research_corrected_v2_2026_08_01`

Status: `CORRECTED_AND_REFROZEN_AT_PREFIT_PAUSE`

## Executive disposition

The reported fold-1 foundation drift was **BENIGN_TEST_CONTAMINATION**, not a scientific-data or frozen-foundation change. The immutable v1 burn remains quarantined as history, while a distinct corrected-v2 foundation restores pristine namespaces for all five chronological outer folds. The binding acceptance rule remains **pooled AND at least 4 of 5 chronological outer folds**; five available folds are not being mislabeled as a 5/5 acceptance rule.

The field-parity correction is fail closed. `last_causal_open_interest` is dropped because its Databento source is daily/EOD and has no intraday live twin. `last_causal_minute_volume` is also dropped from this run: intraday volume is derivable in principle from trades/OHLCV, but no shared historical/live adapter receipt yet proves the exact completed-minute, sparse/no-trade, exact-contract, and 90-second carry semantics. The exit alpha is therefore corrected from the original 49 features to 47. Entry remains byte-order-identical **signed-17**.

No corpus decode, model fit, evidence-fold opening, protected-holdout opening, live-market operation, broker connection, paper order, model promotion, or default change occurred. The machinery and foundation-stability receipts were deliberately **not sealed** because `COMPOSITE_CALIBRATION_TERMINAL_RULE` remains an owner/Claude verification decision.

## Scope and hard stops

Completed in this correction:

- Root-cause and classify the apparent frozen-foundation drift.
- Preserve the original fold-1 burn receipt and both earlier foundation generations byte-for-byte.
- Create and freeze a distinct corrected-v2 preregistration generation.
- Restore pristine namespaces for outer folds 1 through 5 in that new generation.
- Add a campaign-wide byte-stability preflight before every fold-scoped dispatcher and retain the immediate pre-decode stability check.
- Adopt the Databento field-parity inventory as the live-twin authority.
- Reverify the original entry-17 and exit-49 inventory, then remove the two unresolved exit fields.
- Record `size_imbalance` as a forward-only entry-widening lead.

Still prohibited and absent:

- Any fit or fitted weights.
- Any corpus decode or fold access receipt.
- Any outer/nested evidence result.
- Any protected-holdout access.
- Any live data, broker, paper-submit, or real-money action.
- Any widening of signed-17 in this run.
- Any machinery, lineage-implementation, corpus-integrity, or foundation-stability seal.

## P1 — Drift root cause and fold disposition

### Finding: `BENIGN_TEST_CONTAMINATION`

The burn did not follow a genuine change between a valid freeze and a production decode:

1. The quarantined fold-1 burn was written at `2026-08-01T19:45:37.424780Z`.
2. The original preregistration freeze was written later, at `2026-08-01T19:51:02.310489Z`.
3. The burn therefore predates that freeze by `324.885709` seconds.
4. Its fixed transaction ID, `frozen-gate-transaction`, exactly matches the typed synthetic authorization fixture `_dummy_evidence_authorization` in `v4/tests/test_pathd_entry_exit_gate_frozen.py`.
5. The earlier validator control flow allowed that exact typed test authorization to reach the real fixed-scope validator before process-local capability authentication. It wrote a burn into the then-active fold-1 namespace.
6. No evidence was observed: `access_count=0`, `access_receipt_sha256=null`, `dataset_receipt_sha256=null`, `result_sha256=null`, and `holdout_open_count=0`.
7. The original and corrected session assignments are semantically identical (`597afa94…` semantic assignment hash), so this was not a corpus split or scientific-data change.

The historical burn is preserved at:

`v4/audit/autoresearch/pathd_test_contamination_quarantine_2026_08_01/outer_primary_burned_receipt.frozen-gate-transaction.json`

Its identities are:

- Raw file SHA-256: `48aa120cbe1c9917bc92206bbeb96bb0662a066a8e5e80f02a385713a7625aeb`
- Receipt semantic self-hash: `5aa970c413d4b963641b076d74294432fd4950a1f45301af3615abe8e29b7d58`
- Historical status: `BURNED_NO_REOPEN`
- Historical reason: `FROZEN_FOUNDATION_DRIFT_BEFORE_DECODE`

That v1 receipt is not reopened, deleted, or rewritten. Restoration occurs only in a new generation.

### Corrected-v2 re-freeze and restoration

The corrected-v2 preregistration freeze reports:

- Status: `FROZEN_BEFORE_ANY_MODEL_FIT`
- Frozen at: `2026-08-01T22:51:15.434609Z`
- Preregistration SHA-256: `f5e8ed8b99da21b6a4b1d83cdf3e23729ced1eb0e038da1467b5aa64d833c04e`
- Session assignments SHA-256: `431cd14879ad6a14b278cb2683e4f3b860eef960e6b74a4f0edb3e620cf82475`
- Feature lineage SHA-256: `4521b3e62982f672daaba73d98ce5a9ce72df170b083cd40df7dc979983d1ef0`
- Foundation correction SHA-256: `e6994dc197e826e0df918fac0f141df7b1d80ab0536b692128dfc33f8e0530fb`
- `model_fit_executed=false`
- `holdout_open_count=0`

The restoration receipt reports:

- Status: `RESTORED_FIVE_FOLDS_BENIGN_TEST_CONTAMINATION`
- Restored outer folds: `[1, 2, 3, 4, 5]`
- Every new fold namespace: `PRISTINE_ABSENT`
- Foundation generation SHA-256: `c1a0b31383524197a383281f9fb43526d4a20174b6557def2794b2eb7994b79a`
- Receipt semantic self-hash: `fe3b39d344c5cbf76df8264e78a4965752153c1d81d3b1825e3a1a2166eb6edf`
- Acceptance: `pooled AND at least 4 of 5 chronological outer folds`
- `model_fit_executed=false`
- `holdout_open_count=0`

Both prior generations are immutable, explicitly hash-bound inputs to corrected-v2:

- Original: `protocol101_pathd_entry_exit_model_research_2026_08_01`
- Intermediate correction: `protocol101_pathd_entry_exit_model_research_corrected_2026_08_01`
- Active pre-fit correction: `protocol101_pathd_entry_exit_model_research_corrected_v2_2026_08_01`

### Foundation-stability gate

The gate is implemented at three boundaries:

1. `v4/scripts/run_pathd_entry_exit_research.py` invokes `assert_research_foundation_stable()` before dispatching **any** fold-scoped stage, including invalid-nested-skip, nested-block, outer-fold, and generic fold routes. A mismatch therefore stops before a skip, access, burn, dataset, or result artifact can be written.
2. `_prepare_invalid_nested_skip_claim()` in `v4/research/pathd_evidence_gate.py` independently invokes the same stability assertion before its skip path can write anything.
3. `_validate_claim_foundation()` invokes it again immediately before the one authorized decode/O_EXCL access transaction and binds the claim to both the foundation-generation hash and stability-receipt hash.

The required behavior is fail-loud:

- Missing stability seal: block before fold dispatch.
- Any byte/hash mismatch: block before fold dispatch; do **not** silently burn an unopened fold.
- Drift after a scope has actually opened: preserve conservative transaction semantics and burn only that already-open scope.

A negative test mutates the asserted foundation and proves that fold dispatch is never called and no fold namespace is created. The gate is implemented but its stability receipt is intentionally absent at this pause.

## P2 — Field parity and live-twin ground truth

### Binding source classes

The inventory now treats `FINDING_databento_live_field_parity_2026_08_01.md` as ground truth:

- OPRA current/rolling quote fields: historical `cbbo-1s`; live direct `cbbo-1s` or an exact `cmbp-1` to `cbbo-1s` reconstruction.
- Entry D ladder fields: historical **completed `cbbo-1m`**, matching the actual dataset builder; live direct completed `cbbo-1m` or exact `cbbo-1s`/`cmbp-1` to the same completed-minute contract.
- Official SPX context: completed ThetaData official-SPX minute bars with identical causal clocks and carry.
- Greeks: the same pure calculation/kernel and constants over causal price, spot, contract geometry, and time inputs.
- Causal position/account fields: native hash-chained internal state through the current decision only.
- Databento OI/stat-volume: daily/EOD, barred as an intraday feature.

### OI disposition

`last_causal_open_interest` is **dropped**, not reinterpreted as a 90-second intraday carry. The negative `feature_without_live_twin` fixture rejects it even when a fake nonempty adapter label is supplied. A name or adapter string cannot turn an EOD field into an intraday live twin.

### Minute-volume disposition

`last_causal_minute_volume` is **dropped from this run**. Intraday option volume can be derived from live trades or OHLCV, but the present repository does not contain a sealed shared adapter proving all of the binding semantics:

- exact completed one-minute boundary,
- exact SPXW contract identity,
- sparse/no-trade minute treatment,
- historical/live aggregation identity,
- same-session-only carry,
- exact 90-second freshness/carry rule.

This is a fail-closed removal, not a claim that live intraday volume is impossible. It may return only in a separately reviewed generation after an exact adapter and parity receipt exist.

### Reverified entry feature list — signed-17 unchanged

The exact ordered signed-17 identity hash remains `a3c7d3420309722324ce827ddd833d86ffec517e5b841a7c9b480e172707c469`.

Official-SPX completed-minute context (1–12; live-derivable pending exact shared adapter receipt):

1. `spx_vwap_gap_points`
2. `spx_vwap_gap_bps`
3. `spx_vwap_gap_over_session_range`
4. `session_range_bps`
5. `momentum_5m_bps`
6. `momentum_15m_bps`
7. `momentum_5m_over_session_range`
8. `momentum_15m_over_session_range`
9. `omar_clipped_neg3_pos3`
10. `vwap_side_alignment_flag`
11. `omar_side_alignment_flag`
12. `momentum15_side_alignment_flag`

Completed OPRA CBBO-1m ladder derivations (13–15; live-derivable pending exact completed-minute adapter receipt):

13. `D.near_atm.straddle_mid_spot_bps`
14. `D.near_atm.put_call_mid_ratio`
15. `D.near_atm.side_smile_slope_bps_per_5pt`

Self-computed Greeks (16–17; identical kernel/constant contract over causal inputs):

16. `E.bs.delta`
17. `E.bs.gamma`

### Reverified original exit feature list — 49 reviewed, 47 retained

OPRA CBBO-1s current and rolling quote family (1–22; live-derivable pending exact shared adapter receipt):

1. `option_bid`
2. `option_ask`
3. `option_mid`
4. `option_spread`
5. `option_spread_over_mid`
6. `option_bid_size`
7. `option_ask_size`
8. `option_size_imbalance`
9. `option_quote_age_ms`
10. `option_log_mid_return_1s`
11. `option_log_mid_return_5s`
12. `option_log_mid_return_15s`
13. `option_log_mid_return_30s`
14. `option_log_mid_return_60s`
15. `option_spread_mean_5s`
16. `option_spread_mean_15s`
17. `option_spread_mean_60s`
18. `option_spread_std_15s`
19. `option_spread_std_60s`
20. `option_imbalance_mean_5s`
21. `option_imbalance_mean_15s`
22. `option_imbalance_mean_60s`

Removed after field-parity review:

23. `last_causal_minute_volume` — **DROPPED** until exact shared trades/OHLCV adapter and carry-parity receipt.
24. `last_causal_open_interest` — **DROPPED**; EOD-only, no intraday live twin.

Official-SPX completed-minute context (25–29; live-derivable pending exact shared adapter receipt):

25. `official_spx_close`
26. `official_spx_log_return_1m`
27. `official_spx_log_return_5m`
28. `official_spx_log_return_15m`
29. `official_spx_vwap_gap_bps`

Self-computed option Greeks and causal changes (30–36; same kernel and causal-input contract):

30. `self_computed_iv`
31. `self_computed_delta`
32. `self_computed_gamma`
33. `self_computed_iv_change_5s`
34. `self_computed_iv_change_30s`
35. `self_computed_delta_change_30s`
36. `self_computed_gamma_change_30s`

Native causal position, clock, and account state (37–49; derived through the current decision only):

37. `held_right_is_call`
38. `held_strike_offset_points`
39. `entry_fill_option_price`
40. `current_net_pnl_dollars`
41. `current_return_on_entry_premium`
42. `mfe_dollars_to_now`
43. `mae_dollars_to_now`
44. `giveback_dollars_to_now`
45. `seconds_held`
46. `seconds_to_15:55`
47. `position_occupancy`
48. `remaining_d48_budget_dollars`
49. `realized_session_pnl_dollars`

The corrected exit-47 is the original ordered list with only items 23 and 24 removed. No unresolved live-twin field is permitted in current alpha. Remaining pending adapter receipts are explicit fit gates, not implied proofs.

## P3 — Forward-only widen-entry lead

Recorded for a future, separately preregistered `WIDEN-ENTRY` experiment:

- Primary lead: `size_imbalance`, derived from live-usable `bid_size` and `ask_size`.
- Evidence: `+0.129` incremental candidate-ranking signal over moneyness; clustered-minute `t≈2.59` (reported as approximately 2.6).
- Optional companions: raw `bid_size` and `ask_size`.
- Current-run alpha status: **not allowed**.
- Current entry contract: **exactly signed-17 unchanged**.

The exit model already retains `option_size_imbalance`; this note does not change entry decision #1.

## Verification evidence

The frozen generation and gate logic passed the complete registered pre-fit synthetic suite:

```text
143 passed in 27.04s
```

The suite covered:

- frozen evidence gating and capability hardening,
- entry/exit research contracts,
- evidence and protected-holdout gates,
- fixed-science contracts,
- all entry-17/original-exit-49/intermediate-exit-48/corrected-exit-47 live-twin inventories,
- CBBO-1m entry-D historical lineage,
- rejection of intraday OI with a fake adapter,
- corrected-foundation reconstruction,
- fail-before-dispatch behavior on a foundation mismatch.

At report time, the corrected-v2 root contains only its six frozen foundation files:

1. `feature_lineage.json`
2. `foundation_restoration_receipt.json`
3. `preregistration.json`
4. `preregistration.sha256`
5. `preregistration_freeze_receipt.json`
6. `session_assignments.json`

There is no fold directory, access receipt, dataset receipt, result, model artifact, fitted weight, holdout receipt, machinery receipt, lineage-implementation receipt, corpus-integrity receipt, or foundation-stability receipt.

## Required next decision

Do not seal machinery/stability and do not fit. `COMPOSITE_CALIBRATION_TERMINAL_RULE` remains unresolved: a calibration-valid session-count block can still yield `INVALID_TARGET_COVERAGE` or `INSUFFICIENT_EVIDENCE`, and choosing its durable terminal/minimum-power rule changes scientific topology outside P1/P2/P3.

STOP_FOR_CLAUDE_VERIFICATION
