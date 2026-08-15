# Current Pilot Decision Summary

The v4 data stack is usable and the first time-aware action model found a promising time-window-sensitive structure. A fresh Q4 2025 frozen holdout rejected both current Jan-Mar champions, so the broad data-purchase signal is not cleared. Continue model iteration on the existing data; do not spend more credits until a pre-registered model change survives the protected holdouts.

Environment caveat: Jan-Mar 2026 is real market truth, but only one environment slice. The goal is to train across all market environments over time, not to freeze the system around this slice.

## Data Sufficiency

The current data inventory is enough for a serious prototype and falsification loop, but not enough for a broad-market neural edge claim.

| Source | Coverage | Safe Use | Main Caveat |
|---|---:|---|---|
| v4 Databento OPRA CBBO/OHLCV/statistics | 61 sessions, 11.14M normalized rows | executable ask-entry/bid-exit labels | only Jan-Mar 2026 |
| v4 neural rows | 21,959 decision rows, 638k finite labels per policy | small neural prototype and March holdout | SPX/VIX context is derived/proxy, not official Cboe index data |
| Q4 2025 frozen audit block | 64 sessions, 11.59M normalized rows, 22,719 decision rows | protected out-of-period survival check | audit-only; do not tune against it |
| v2/v3 historical artifacts | 1002 days, 389k bars, 89,692 action rows | regime/time/VWAP/OMAR priors | bid/ask is Polygon OHLC proxy, not observed NBBO |

Full audit: `v4/audit/data_sufficiency/report.md`.

## Candidate Neural Policies

| Policy | Neural PnL | PF | Max DD | Gate |
|---|---:|---:|---:|---|
| policy0_stop35_target60_hold10m | -60 | 0.937 | -464 | False |
| policy1_stop50_target100_hold25m | -401 | 0.773 | -1146 | False |
| policy2_stop65_target150_hold45m | -541 | 0.041 | -564 | False |

## Time-Aware Action Model Policy 2

Single-seed March holdout from the first time-aware action model:

| Filter | Test Trades | Test PnL | Test PF | Test DD | Positive Day Fraction |
|---|---:|---:|---:|---:|---:|
| all_times | 176 | 10473 | 1.168 | -8867 | 0.591 |
| skip_first_30 | 154 | 17427 | 1.372 | -5611 | 0.682 |
| post_open_and_late | 89 | 15107 | 1.559 | -4051 | 0.591 |
| late_afternoon_only | 45 | 8545 | 1.832 | -1880 | 0.591 |
| post_open_only | 44 | 6562 | 1.392 | -6426 | 0.500 |

## Multi-Seed Fixed Time-Window Robustness

No fixed time window passed the broad purchase gate across 10 random seeds. The closest near-misses:

| Policy | Filter | Trades | Median PnL | Median PF | Median DD | Positive Seeds | Positive Days | Miss |
|---|---|---:|---:|---:|---:|---:|---:|---|
| policy2_stop65_target150_hold45m | skip_first_30 | 154 | 6318 | 1.127 | -6904 | 0.80 | 0.59 | PF and DD |
| policy2_stop65_target150_hold45m | late_afternoon_only | 46 | 4584 | 1.357 | -2757 | 1.00 | 0.48 | PnL and positive days |
| policy1_stop50_target100_hold25m | post_open_and_late | 136 | 5106 | 1.169 | -5767 | 0.70 | 0.50 | PF and positive days |
| policy0_stop35_target60_hold10m | post_open_and_late | 47 | 1746 | 1.596 | -1686 | 0.90 | 0.56 | PnL |

Full summary: `v4/audit/broad_data_purchase_signal/fixed_filter_summary.md`.

## Risk-Controlled Purchase Gate

Validation-selected threshold, time window, max trades per day, and daily loss stop did not improve the broad purchase case.

| Policy | Runs | Median PnL | Median PF | Median DD | Trades | Positive Seeds | Positive Days | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| policy0_stop35_target60_hold10m | 10 | 2554 | 1.187 | -2996 | 61 | 0.80 | 0.48 | False |
| policy1_stop50_target100_hold25m | 10 | 2537 | 1.194 | -5517 | 49 | 0.70 | 0.42 | False |
| policy2_stop65_target150_hold45m | 10 | -1318 | 0.944 | -11937 | 60 | 0.40 | 0.41 | False |

Full summary: `v4/audit/risk_controlled_purchase_signal/report.md`.

## Historical-Prior Feature Rerun

I mined v2/v3 only for reusable causal priors, then added explicit v4 features for VWAP relation, OMAR sign, momentum alignment/counteralignment, and side-aware trend/mean-reversion state. These are not hard rules; they are cleaner state variables for the model.

Broad gate with env-prior features still failed, but policy 1 became materially cleaner:

| Policy | Runs | Median PnL | Median PF | Median DD | Trades | Positive Seeds | Positive Days | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| policy0_stop35_target60_hold10m | 10 | 2593 | 1.586 | -1439 | 46 | 0.80 | 0.50 | False |
| policy1_stop50_target100_hold25m | 10 | 2448 | 1.490 | -2544 | 48 | 0.90 | 0.55 | False |
| policy2_stop65_target150_hold45m | 10 | 3505 | 1.154 | -6246 | 48 | 0.80 | 0.50 | False |

Risk-controlled gate with env-prior features also failed, but policy 1 was the nearest miss:

| Policy | Runs | Median PnL | Median PF | Median DD | Trades | Positive Seeds | Positive Days | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| policy0_stop35_target60_hold10m | 10 | 1780 | 1.240 | -1588 | 58 | 0.70 | 0.53 | False |
| policy1_stop50_target100_hold25m | 10 | 3550 | 1.206 | -3620 | 87 | 0.80 | 0.55 | False |
| policy2_stop65_target150_hold45m | 10 | -628 | 0.902 | -8164 | 66 | 0.30 | 0.45 | False |

Validation-calibrated abstention split February into a calibration half and a selection half, then held March untouched. It reduced concentration but did not improve the purchase case:

| Policy | Runs | Median PnL | Median PF | Median DD | Trades | Positive Seeds | Positive Days | Top-Day Share | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| policy0_stop35_target60_hold10m | 10 | 56 | 1.010 | -2508 | 45 | 0.50 | 0.47 | 0.30 | False |
| policy1_stop50_target100_hold25m | 10 | 1679 | 1.113 | -3720 | 46 | 0.70 | 0.42 | 0.27 | False |
| policy2_stop65_target150_hold45m | 10 | -274 | 1.001 | -4874 | 44 | 0.50 | 0.45 | 0.28 | False |

Full reports: `v4/audit/historical_priors/report.md`, `v4/audit/broad_data_purchase_signal_envpriors/report.md`, `v4/audit/risk_controlled_purchase_signal_envpriors/report.md`, and `v4/audit/calibrated_abstention_signal_envpriors/report.md`.

## Autoresearch Loop 001

I added a bounded autoresearch loop inspired by Karpathy's `autoresearch` pattern, but adapted for real options risk. Trials are pre-registered, every trial is logged, selection uses only the second half of February, and March is audit-only. The neural net directly chooses among no-trade, nearest-ATM call, and nearest-ATM put.

Selection-floor-cleared champion:

| Policy | Trial | Selection PnL | Selection PF | Selection Days | March PnL | March PF | March DD | March Positive Seeds |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| policy1_stop50_target100_hold25m | skip_first30_edge0_max4_stop1000 | 1152 | 1.159 | 0.50 | 1687 | 1.181 | -2358 | 1.00 |

This is a useful modeling lead, not a broad-purchase or live-trading signal. The March audit is constructive, but the scale is still too small and the loop used only three seeds.

Full report: `v4/audit/autoresearch/v4_autoresearch_001/report.md`.

## Decision-Aware No-Trade Loss Rerun

I added a fixed decision-aware action loss: executable PnL Huber regression plus best-action classification, with an extra margin penalty when the model predicts a trade in minutes where no-trade is the best action. The same pre-registered autoresearch loop was rerun with no new selection knobs.

Result: the new loss improved February selection metrics but did **not** improve the March audit. The selected champion became more selective and less robust out of sample.

| Objective | Champion | Selection PnL | Selection PF | Selection Trades | March PnL | March PF | March Trades | March Positive Seeds |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Huber Q regression | policy1 / skip_first30_edge0_max4_stop1000 | 1152 | 1.159 | 32 | 1687 | 1.181 | 42 | 1.00 |
| Decision-aware loss v1 | policy1 / post_open_late_edge25_max4 | 1656 | 1.429 | 17 | 160 | 1.119 | 15 | 0.67 |

Interpretation: do not treat decision-aware loss v1 as an improvement. It is useful because it exposed a validation/holdout mismatch, but the older Huber objective remains the stronger current baseline for March audit behavior.

Full report: `v4/audit/autoresearch/v4_autoresearch_001_decision_loss/report.md`.

## Q4 2025 Frozen Holdout Audit

With permission, I downloaded a targeted Q4 2025 block instead of spending the remaining credit balance broadly. The capped download estimated cost was **$31.47** and produced 64 sessions from 2025-10-01 through 2025-12-31. The neural build created 22,719 decision rows. Holidays 2025-11-27 and 2025-12-25 were skipped, and 2025-10-22 carried a Databento degraded-quality warning.

The audit froze the two existing Jan-Mar champions before scoring Q4. No Q4 result was used to select a new trial.

| Champion | Objective | Trial | Q4 PnL | Q4 PF | Q4 DD | Trades | Positive Seeds | Positive Days |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Huber Q baseline | action_v1 | skip_first30_edge0_max4_stop1000 | -13691 | 0.597 | -14343 | 188 | 0.00 | 0.34 |
| Decision-aware loss v1 | action_v2_decision_aware | post_open_late_edge25_max4 | -2337 | 0.702 | -4758 | 51 | 0.00 | 0.36 |

Interpretation: this is the first strong falsification result. Q4 does not say the project is dead; it says the current Jan-Mar signal is not broad enough and should not justify a larger historical purchase or any live-trading path. The remaining credit should be preserved for narrow, pre-registered audits after model changes, not spent on more months by default.

Full reports: `v4/audit/databento_q4_2025_downloads.jsonl`, `v4/audit/q4_2025_neural_build_summary.json`, and `v4/audit/autoresearch/q4_2025_frozen_audit/report.md`.

## Best Validation-Selected Environment Rule Survivors

| Rule | Val PnL | Test Trades | Test PnL | Test PF |
|---|---:|---:|---:|---:|
| `P|above_vwap=True|momentum_15m_sign=positive|time_bucket=late_afternoon` | 1253 | 32 | 3776 | 1.440 |
| `P|above_vwap=True|momentum_15m_sign=flat|time_bucket=late_afternoon` | 330 | 25 | 1815 | 1.302 |
| `C|momentum_15m_sign=flat|time_bucket=post_open_morning` | 1090 | 43 | 894 | 1.048 |
| `P|above_vwap=False|momentum_15m_sign=negative|time_bucket=midday` | 360 | 40 | 650 | 1.052 |
| `C|above_vwap=False|time_bucket=post_open_morning` | 5678 | 37 | 546 | 1.035 |
| `C|above_vwap=False|momentum_15m_sign=negative|time_bucket=post_open_morning` | 6740 | 35 | 250 | 1.018 |
| `P|above_vwap=False|time_bucket=midday` | 136 | 44 | 42 | 1.003 |
| `P|above_vwap=True|momentum_15m_sign=positive|time_bucket=midday` | 1610 | 39 | -138 | 0.987 |
| `C|momentum_15m_sign=negative|time_bucket=post_open_morning` | 4268 | 42 | -524 | 0.972 |
| `C|time_bucket=post_open_morning` | 1304 | 44 | -1038 | 0.948 |

## Interpretation
- Adding explicit causal time-of-day features materially changed the model result, which confirms the model could not fairly discover this structure before.
- The best single-seed model is encouraging, but the multi-seed evidence is not yet stable enough to justify a broad historical data purchase.
- The most credible structure remains: avoid the first 30 minutes, focus on post-open morning and late afternoon, and treat late afternoon as the cleanest standalone window.
- Validation-selected risk controls overfit February and did not survive March strongly enough.
- Validation-calibrated abstention also failed; the best policy 1 result had acceptable concentration and drawdown, but too little median PnL and too few positive days.
- Autoresearch loop 001 found a floor-cleared validation champion with positive March audit across three seeds, but the result is too small for a broad data-purchase signal.
- Decision-aware no-trade loss v1 improved February selection but degraded the protected March audit; this is not a confirmed improvement.
- Q4 2025 frozen holdout rejected both current champions, including the Huber baseline and decision-aware v1. The Jan-Mar result is not yet a generalizable edge.
- v2/v3 is valuable, but mostly as prior-mining/history; v4 CBBO remains the label truth.
- Historical-prior features improved policy 1 robustness, but not enough to clear the purchase gate.
- Late afternoon is the cleanest standalone window across train, validation, and test; post-open morning is useful but more volatile.
- This is a modeling continuation signal, not a broad data-purchase signal.

## Next Steps
- Promote time-aware action modeling as the main prototype path.
- Replace grid-heavy selection with simpler pre-registered priors: skip first 30, late-afternoon-only, and post-open-plus-late.
- Keep Huber Q-regression as the current baseline until a no-trade objective improves both February selection and March audit.
- If trying no-trade loss v2, pre-register it as a new objective variant before looking at March; do not tune loss weights against the March audit.
- Treat Q4 2025 as audit-only. Do not tune model weights, time windows, thresholds, or loss coefficients directly against Q4.
- Use v2/v3 to pretrain or rank candidate features, but continue using v4 CBBO ask-entry/bid-exit labels for all purchase decisions.
- Do not buy more broad historical data until a pre-registered model change survives both March 2026 and the frozen Q4 2025 audit without adding new selection knobs.
