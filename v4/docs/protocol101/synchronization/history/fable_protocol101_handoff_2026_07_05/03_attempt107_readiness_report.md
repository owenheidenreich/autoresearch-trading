# Protocol101 Fair-Contract Attempt107 Readiness Packet

Generated: 2026-07-05

## Decision

- Candidate: `attempt_107_policy0_hgb_blend35_relative_put_near_after0940_vwap_m2_10_cap3_scoreceil50_dailyloss500_plateau_s42`
- Current status: `not_paper_ready`
- Next allowed status: `offline_sync_replay_candidate`
- Paper default changed: `false`
- Broker endpoint called: `false`
- Paper-submit allowed: `false`
- Paid data downloaded here: `false`

This packet does not authorize paper-submit, promotion, default-registry edits, threshold tuning on confirmation days, or real-money trading.

## What Passed

The 128-session fair-contract search selected attempt107 as the current lead under `protocol101-live-v1`.

Evidence:

- Model-search report: `v4/audit/autoresearch/protocol101_fair_contract_model_search_expanded_jul_dec2025_128_q1_vwap_pocket_plateau_threshold/report.md`
- Experiment registry: `v4/audit/autoresearch/protocol101_fair_contract_model_search_expanded_jul_dec2025_128_q1_vwap_pocket_plateau_threshold/experiment_registry.jsonl`
- Training result: `v4/audit/autoresearch/protocol101_fair_contract_model_search_expanded_jul_dec2025_128_q1_vwap_pocket_plateau_threshold/attempts/attempt_107_policy0_hgb_blend35_relative_put_near_after0940_vwap_m2_10_cap3_scoreceil50_dailyloss500_plateau_s42/training_runner/training_result.json`
- Strict replay report: `v4/audit/autoresearch/protocol101_fair_contract_model_search_expanded_jul_dec2025_128_q1_vwap_pocket_plateau_threshold/attempts/attempt_107_policy0_hgb_blend35_relative_put_near_after0940_vwap_m2_10_cap3_scoreceil50_dailyloss500_plateau_s42/selected_candidate_replay_gate/report.md`
- Trade charts: `v4/audit/autoresearch/protocol101_fair_contract_attempt107_128_trade_charts/report.md`

Strict replay metrics after stress:

| Split | Trades | Total PnL | Profit Factor | Max Drawdown % Start | Status |
|---|---:|---:|---:|---:|---|
| validation | 24 | $3,495 | 1.978 | -21.15% | pass |
| diagnostic_test | 26 | $1,355 | 1.389 | -9.70% | pass |

Combined chart packet:

| Metric | Value |
|---|---:|
| Trades | 50 |
| Total PnL | $4,850 |
| Ending equity | $14,850 |
| Win rate | 44.0% |
| Max drawdown | -$2,115 |
| Max drawdown % | -18.19% |
| Worst day | -$940 |
| Max known buying power | 35.0% of equity |

The candidate also satisfies these historical replay checks in the strict replay gate:

- no missing required selected-candidate fields
- no overlap skips
- no unaffordable skips
- validation and diagnostic trade counts above 20
- validation and diagnostic PnL positive after stress
- validation and diagnostic PF above 1.25
- validation and diagnostic drawdown below 35% of starting equity

## What Did Not Pass Yet

Attempt107 is not paper-ready because the goal requires both historical validation and synchronization evidence. The following gates remain open:

1. IBKR-vs-Databento/ThetaData paired replay has been run and failed for this candidate on the threshold-crossing confirmation days.
2. Option-feature data-plane sensitivity has not been repaired or cleared. July 1 action drift is driven by option IV/score-ceiling sensitivity, and July 2 selected-contract drift is driven by option spread/size/ranking sensitivity.
3. Concentration risk has not been cleared. The chart packet reports top 20 trades contributed $11,900 while total PnL was $4,850, which means losses materially offset a small number of large winners.
4. Training-window sensitivity is not cleared. The stability diagnostic shows this same attempt family is sensitive across the 64-session and 128-session training-window comparisons.
5. Same-runner legacy and frozen Protocol101 live-v1 baseline comparisons for this exact candidate packet still need to be summarized in one final selection packet.

## IBKR Same-Input Replay

Attempt107 was replayed over three immutable IBKR recorder days with repeat checks.

Evidence:

- June 30 replay: `v4/audit/autoresearch/protocol101_fair_contract_attempt107_ibkr_capture_replay_2026_06_30/report.md`
- July 1 replay: `v4/audit/autoresearch/protocol101_fair_contract_attempt107_ibkr_capture_replay_2026_07_01/report.md`
- July 2 replay: `v4/audit/autoresearch/protocol101_fair_contract_attempt107_ibkr_capture_replay_2026_07_02/report.md`

| Session | Decisions | Candidates | Entry Intents | Same-Input Exact | Notes |
|---|---:|---:|---:|---|---|
| 2026-06-30 | 360 | 10,772 | 3 | true | session trade cap blocks later entries |
| 2026-07-01 | 360 | 10,188 | 3 | true | session trade cap blocks later entries |
| 2026-07-02 | 360 | 11,598 | 3 | true | session trade cap blocks later entries |

This clears the candidate-specific same-input IBKR replay gate for these three captures. It does not clear cross-vendor paired replay because the same attempt107 decisions still need to be generated from the matching Databento/ThetaData canonical rows and diffed at the same timestamps.

Important caution: attempt107 enters up to its three-trade session cap on all three IBKR replay days. Threshold waits were zero in these IBKR packets, so the candidate is heavily governed by the entry filter, score ceiling, cooldown, and trade cap. That makes paired historical replay and concentration/stability diagnostics especially important before paper-submit.

## Paired Historical Replay

Attempt107 was also replayed over matching historical `protocol101-live-v1` rows for June 30, July 1, and July 2, then diffed against the IBKR capture replay.

Evidence:

- June 30 historical replay: `v4/audit/autoresearch/protocol101_fair_contract_attempt107_historical_replay_2026_06_30/report.md`
- July 1 historical replay: `v4/audit/autoresearch/protocol101_fair_contract_attempt107_historical_replay_2026_07_01/report.md`
- July 2 historical replay: `v4/audit/autoresearch/protocol101_fair_contract_attempt107_historical_replay_2026_07_02/report.md`
- June 30 paired diff: `v4/audit/autoresearch/protocol101_fair_contract_attempt107_paired_diff_2026_06_30/report.md`
- July 1 paired diff: `v4/audit/autoresearch/protocol101_fair_contract_attempt107_paired_diff_2026_07_01/report.md`
- July 2 paired diff: `v4/audit/autoresearch/protocol101_fair_contract_attempt107_paired_diff_2026_07_02/report.md`

| Session | Action Matches | Action Mismatches | Selected Contract Mismatches | Interpretation |
|---|---:|---:|---:|---|
| 2026-06-30 | 360 | 0 | 0 | entry times and contracts match despite feature/score timestamp drift |
| 2026-07-01 | 358 | 2 | 2 | fail: score-ceiling drift at 10:36 ET changes sequence; later trade-cap state diverges |
| 2026-07-02 | 360 | 0 | 1 | partial: entry times match, but 09:57 ET selected put differs by one/more strikes |

The generic paired diff reports every row as `decision_mismatch` because source timestamps, account-state hashes, feature hashes, and score hashes differ across vendors. For paper readiness, that is not automatically fatal, but the action/contract failures on July 1 and July 2 are fatal until explained or repaired.

Key mismatch examples:

- July 1 at `2026-07-01T14:36:00+00:00`: IBKR enters `SPXW-20260701-07510.000-P` with score `25.3371`; historical waits because the same contract scores `55.5838`, triggering the `max_score_ceiling=50` guard.
- July 1 at `2026-07-01T16:53:00+00:00`: historical enters `SPXW-20260701-07500.000-P`; IBKR waits because its earlier 10:36 ET entry already consumed the three-trade session cap.
- July 2 at `2026-07-02T13:57:00+00:00`: both enter, but IBKR selects `SPXW-20260702-07545.000-P` while historical selects `SPXW-20260702-07520.000-P`; both eligible put scores are close enough that cross-vendor feature drift changes the ranking.

Current paired-replay conclusion: `not_synchronized_enough_for_paper_submit`.

## Pair Attribution

The paired replay failures have now been attributed at the candidate-feature level.

Evidence:

- Pair attribution report: `v4/audit/autoresearch/protocol101_fair_contract_attempt107_pair_attribution/report.md`
- Pair attribution summary: `v4/audit/autoresearch/protocol101_fair_contract_attempt107_pair_attribution/summary.json`
- Feature diffs: `v4/audit/autoresearch/protocol101_fair_contract_attempt107_pair_attribution/feature_diffs.csv`
- Group swaps: `v4/audit/autoresearch/protocol101_fair_contract_attempt107_pair_attribution/group_swaps.csv`

Finding:

- The actionable paired-replay drift is concentrated in option-side features, not broad market context.
- July 1 at `2026-07-01T14:36:00+00:00`: IBKR enters `SPXW-20260701-07510.000-P` with score `25.3371`; historical waits because the same contract scores `55.5838` and crosses `max_score_ceiling=50`. The model counterfactual shows the meaningful score movement is almost entirely from option features, especially a very small IV difference.
- July 2 at `2026-07-02T13:57:00+00:00`: both feeds enter, but the selected put differs. The drift is small in headline score terms, and the ranking flip is mainly option spread/size sensitivity among close candidates.

Interpretation:

- Attempt107 is deterministic on the same captured input, but it is too sensitive to cross-vendor option-feature differences at the exact moments that matter.
- The score-ceiling guard is especially brittle because it turns a small option IV difference into an action mismatch.
- This is not evidence that OMAR, VWAP, or the opening-context repair failed; those broad market-context features are not the driver in these two blocking examples.

Current attribution conclusion: `attempt107_not_paper_ready_due_option_feature_data_plane_sensitivity`.

## Microstructure-Mask Robustness Search

Two follow-up robustness attempts were run after the pair attribution showed the blocking drift was concentrated in option IV, spread, and size semantics.

Evidence:

- Robustness search report: `v4/audit/autoresearch/protocol101_fair_contract_model_search_expanded_jul_dec2025_128_q1_microstructure_mask_robustness/report.md`

Both attempts used the same fair `protocol101-live-v1` labels and excluded June/July recorder days from training and threshold selection. They did not call broker endpoints, allow paper-submit, download paid data, change defaults, or promote a model.

| Attempt | Model | Feature transform | Score ceiling | Validation PnL | Diagnostic PnL | Validation PF | Diagnostic PF | Result |
|---|---|---|---:|---:|---:|---:|---:|---|
| `attempt_108` | HGB | `mask_vendor_sensitive_option_microstructure` | 0 | $1,385 | -$2,210 | 1.353 | 0.536 | fail |
| `attempt_109` | MLP ensemble | `mask_vendor_sensitive_option_microstructure` | 0 | $1,650 | -$1,390 | 1.374 | 0.757 | fail |

Interpretation:

- Masking IV, spread, spread fraction, bid/ask size, option volume, and open interest reduces the exact cross-vendor brittleness we saw in the paired replay.
- But the first masked candidates lose enough historical edge that they fail diagnostic profitability.
- This makes the current problem sharper: attempt107 depends on vendor-sensitive option microstructure, while simple masking removes too much edge.

Current robustness conclusion: `no_microstructure_masked_paper_ready_candidate_yet`.

## Microstructure-Bucket Robustness Search

Two additional robustness attempts were run with a less destructive transform: spread, spread fraction, bid/ask size, and IV were bucketed/quantized while option volume and open interest were zeroed. This was meant to preserve coarse option-state information while reducing sensitivity to tiny cross-vendor deltas.

Evidence:

- Bucket robustness search report: `v4/audit/autoresearch/protocol101_fair_contract_model_search_expanded_jul_dec2025_128_q1_microstructure_bucket_robustness/report.md`

Both attempts used the same fair `protocol101-live-v1` labels and excluded June/July recorder days from training and threshold selection. They did not call broker endpoints, allow paper-submit, download paid data, change defaults, or promote a model.

| Attempt | Model | Feature transform | Score ceiling | Validation PnL | Diagnostic PnL | Validation PF | Diagnostic PF | Result |
|---|---|---|---:|---:|---:|---:|---:|---|
| `attempt_110` | HGB | `bucket_vendor_sensitive_option_microstructure` | 0 | -$1,050 | $60 | 0.734 | 1.018 | fail |
| `attempt_111` | MLP ensemble | `bucket_vendor_sensitive_option_microstructure` | 0 | $1,770 | -$1,100 | 1.518 | 0.759 | fail |

Interpretation:

- Bucketed option microstructure is not enough by itself. It avoids the most literal tiny-IV/tiny-spread dependence, but the tested candidates fail either validation profitability/trade-count gates or diagnostic profitability/PF gates.
- This strengthens the current conclusion: the repair should not be an adapter patch around June/July mismatches, nor a simple field deletion/quantization. The next fair-contract search needs robustness-aware training or selection, where candidates are rewarded for keeping edge while remaining stable under realistic option-feature perturbations.

Current bucket robustness conclusion: `no_microstructure_bucketed_paper_ready_candidate_yet`.

## Feature-Jitter Robustness Gate

A development-only feature-jitter gate was added and run against attempt107. It replays the already-trained candidate over validation and diagnostic rows, then perturbs only the option-side fields that caused the paired IBKR-vs-historical drift. This does not train, tune thresholds, use June/July recorder days for selection, contact brokers/vendors, download data, change defaults, or promote a model.

Evidence:

- Feature-jitter gate report: `v4/audit/autoresearch/protocol101_fair_contract_attempt107_feature_jitter_gate/report.md`
- Scenario rows: `v4/audit/autoresearch/protocol101_fair_contract_attempt107_feature_jitter_gate/scenario_rows.csv`

Result: `fail`.

| Scenario | Validation PnL | Diagnostic PnL | Validation Contract Match | Diagnostic Contract Match | Result |
|---|---:|---:|---:|---:|---|
| baseline | $3,495 | $1,355 | 1.000 | 1.000 | pass |
| IV +0.002 | $3,715 | $1,375 | 1.000 | 0.962 | pass |
| IV -0.002 | $3,495 | $1,355 | 1.000 | 1.000 | pass |
| spread widen 0.05 | $5,960 | -$600 | 0.333 | 0.538 | fail |
| spread tighten 0.05 | $2,730 | $1,690 | 0.542 | 0.500 | fail |
| bid size half / ask size double | $3,890 | $705 | 0.833 | 0.769 | fail |
| bid size double / ask size half | $3,135 | $1,205 | 0.833 | 0.885 | fail |

Interpretation:

- The July 1 IV-triggered score-ceiling failure is still real, but attempt107 is not broadly destroyed by +/-0.002 IV alone on the development rows.
- The larger live-readiness problem is spread/size-sensitive ranking. Small spread and quote-size perturbations keep many entry minutes alive, but they often change which contract is selected and can materially change diagnostic PnL.
- This confirms the repair direction: future candidates need an action-stability/contract-stability objective or stress gate around option spread/size. Adapter-patching the live feed to match Databento would be the wrong lesson.

Current feature-jitter conclusion: `attempt107_fails_option_spread_size_robustness`.

## Score-Margin Jitter-Gated Search

Two margin-based follow-up attempts were run after the feature-jitter gate showed spread/size ranking instability. These attempts kept raw fair-contract option features, removed the brittle score ceiling, and required the selected contract to beat the runner-up by either 5 or 10 score points. The new feature-jitter gate was included directly in model-search evaluation.

Evidence:

- Margin+jitter robustness report: `v4/audit/autoresearch/protocol101_fair_contract_model_search_expanded_jul_dec2025_128_q1_margin_jitter_robustness/report.md`

Result: both attempts failed strict replay and feature-jitter robustness.

| Attempt | Margin | Validation PnL | Diagnostic PnL | Validation PF | Diagnostic PF | Jitter Gate | Result |
|---|---:|---:|---:|---:|---:|---|---|
| `attempt_112` | 5 | -$1,615 | -$1,850 | 0.718 | 0.671 | fail | fail |
| `attempt_113` | 10 | -$1,575 | -$2,680 | 0.755 | 0.541 | fail | fail |

Interpretation:

- A simple top-vs-runner-up score-margin filter does not solve the problem. It removes or reshuffles enough trades that historical replay becomes unprofitable, and it still does not make selected contracts stable under option-feature jitter.
- This rules out another tempting adapter-like shortcut. The next experiment should make robustness part of model fitting or candidate scoring, not just a post-hoc score-margin filter.

Current score-margin conclusion: `score_margin_filter_not_sufficient_for_paper_ready_fair_contract_candidate`.

## Feature-Noise Augmentation Search

Two follow-up attempts trained on deterministic fit-only vendor microstructure jitter. The augmentation duplicated only the model-fitting rows with live-plausible IV, spread, and quote-size perturbations. Calibration, validation, diagnostic, and June/July confirmation rows remained unaugmented.

Evidence:

- Noise-augmentation robustness report: `v4/audit/autoresearch/protocol101_fair_contract_model_search_expanded_jul_dec2025_128_q1_noiseaug_jitter_robustness/report.md`

Result: both attempts failed strict replay and feature-jitter robustness.

| Attempt | Model | Noise Augmentation | Validation PnL | Diagnostic PnL | Validation PF | Diagnostic PF | Jitter Gate | Result |
|---|---|---|---:|---:|---:|---:|---|---|
| `attempt_114` | HGB | `vendor_microstructure_jitter_v1` | $2,770 | -$3,300 | 1.500 | 0.497 | fail | fail |
| `attempt_115` | MLP ensemble | `vendor_microstructure_jitter_v1` | $840 | -$1,670 | 1.180 | 0.707 | fail | fail |

Interpretation:

- Fit-only feature noise did not recover a paper-ready model in this narrow put/VWAP pocket.
- Attempt114 preserved validation profitability but failed diagnostic badly. Attempt115 reduced some jitter blockers compared with attempt114, but validation trade count/PF and diagnostic PnL/PF still failed.
- This means the issue is not merely that the model has never seen small vendor perturbations. The current entry pocket/label/model family combination may be selecting a regime that is unstable under the fair live-reproducible game.

Current feature-noise conclusion: `naive_fit_noise_augmentation_not_sufficient`.

## Jitter-Stressed Threshold Search

A new threshold-selection rule, `jitter_stability_stressed`, was added after naive feature-noise augmentation failed. Instead of choosing the threshold from ordinary validation replay only, it evaluates deterministic validation/calibration jitter scenarios for IV, spread, and quote-size perturbations and prefers thresholds with stronger worst-case jitter PnL, profit factor, trade sufficiency, and action-presence stability.

Evidence:

- Jitter-threshold robustness report: `v4/audit/autoresearch/protocol101_fair_contract_model_search_expanded_jul_dec2025_128_q1_jitterthreshold_robustness/report.md`

Result: both attempts failed strict replay and feature-jitter robustness.

| Attempt | Model | Threshold Rule | Validation PnL | Diagnostic PnL | Validation PF | Diagnostic PF | Jitter Gate | Result |
|---|---|---|---:|---:|---:|---:|---|---|
| `attempt_116` | HGB | `jitter_stability_stressed` | $1,630 | -$2,420 | 1.304 | 0.581 | fail | fail |
| `attempt_117` | MLP ensemble | `jitter_stability_stressed` | $1,160 | -$890 | 1.352 | 0.804 | fail | fail |

Interpretation:

- Jitter-aware threshold selection is directionally better than the worst failed branches, but it still does not recover a paper-ready candidate.
- The MLP branch reduced diagnostic loss compared with several prior failures, but it traded too little in validation, remained diagnostic-negative, and still failed quote-size/spread jitter stability.
- This further supports stepping back from the current narrow put-near/VWAP pocket. The next repair should not be another threshold trick on the same pocket; it should evaluate broader entry filters, labels, and candidate objectives under the jitter-stressed gate from the beginning.

Current jitter-threshold conclusion: `threshold_mechanics_not_sufficient_for_paper_ready_candidate`.

## Broader Entry-Filter Jitter-Gated Search

Two broader entry-filter attempts were run after the narrow put-near/VWAP pocket failed masking, bucketing, margin, feature-noise, and jitter-threshold repairs. These attempts kept the fair contract and jitter-stressed threshold rule, but relaxed the candidate universe:

- `attempt_118`: near-offset puts without the after-09:40/VWAP gate.
- `attempt_119`: near-offset calls and puts with right-specialized HGB estimators.

Evidence:

- Broader jitter-gated report: `v4/audit/autoresearch/protocol101_fair_contract_model_search_expanded_jul_dec2025_128_q1_broader_jittergate_robustness/report.md`

Result: both attempts failed strict replay and feature-jitter robustness.

| Attempt | Entry Filter | Model | Validation PnL | Diagnostic PnL | Validation PF | Diagnostic PF | Jitter Gate | Result |
|---|---|---|---:|---:|---:|---:|---|---|
| `attempt_118` | `put_near_10_20_offset` | HGB | $60 | -$2,240 | 1.012 | 0.719 | fail | fail |
| `attempt_119` | `near_10_20_offset` | HGB by right | -$6,140 | -$1,510 | 0.097 | 0.750 | fail | fail |

Interpretation:

- Simply broadening the entry filter does not repair the fair-contract model. It adds more exposure without enough stable edge.
- The right-specialized model did not rescue call/put side structure; it produced a severe validation loss and still failed jitter stability.
- The next repair should likely move away from this Protocol101-style one-step entry scorer in the current feature space and inspect label/objective construction, candidate value targets, or a two-stage robust selector that first learns stable tradability/edge buckets before ranking contracts.

Current broader-filter conclusion: `broader_entry_filters_not_sufficient_under_jitter_gate`.

## Two-Stage Selector Diagnostic

An offline compound selector diagnostic was added after the broader entry-filter attempts failed. It does not train, tune thresholds, use June/July recorder days for model selection, contact brokers/vendors, download data, change defaults, or promote a model. It uses future labels only to ask whether live-causal context gates still contain stable candidate-level opportunity.

Evidence:

- Two-stage selector diagnostic report: `v4/audit/autoresearch/protocol101_fair_contract_two_stage_selector_diagnostic_jul_dec128_q1/report.md`

Result: `pass_diagnostic_only`.

| Gate | Validation Avg Label | Diagnostic Avg Label | Validation Candidates | Diagnostic Candidates | Validation Oracle PnL | Diagnostic Oracle PnL |
|---|---:|---:|---:|---:|---:|---:|
| `base_plus_omar_neg_range_20_45` | $177.33 | $160.25 | 968 | 648 | $7,560 | $1,240 |
| `base_plus_near_vwap` | $96.37 | $40.11 | 1,154 | 1,895 | $17,270 | $14,810 |
| `base_plus_premium_gte_7_5` | $57.63 | $44.07 | 3,117 | 4,166 | $18,500 | $10,420 |
| `base_put_near_after0940_vwap_m2_10` | $42.31 | $41.99 | 3,952 | 4,970 | $18,500 | $10,420 |

Interpretation:

- The fair `protocol101-live-v1` rows still contain stable causal opportunity. This argues against giving up on the live-reproducible contract.
- The highest-average OMAR/range gate is too sparse by itself for the current paper-readiness trade-count requirement.
- The trade-sufficient gates justify learned-model attempts, but the diagnostic itself is not paper-readiness evidence because it uses labels as an oracle.

Current two-stage diagnostic conclusion: `stable_opportunity_exists_but_requires_learned_robust_selection`.

## Compound Selector Jitter-Gated Search

Four follow-up model-search attempts encoded the best diagnostic-supported compound gates as live-causal first-stage filters. These attempts still used the fair contract, excluded June/July recorder days from training and threshold selection, and required the feature-jitter gate.

Evidence:

- Compound selector robustness report: `v4/audit/autoresearch/protocol101_fair_contract_model_search_expanded_jul_dec2025_128_q1_compound_selector_robustness/report.md`

Result: all attempts failed. Best attempt: `attempt_122_policy0_hgb_blend35_relative_put_near_after0940_vwap_m2_10_nearvwap_cap3_dailyloss500_s42_jittergate`.

| Attempt | Entry Filter | Validation PnL | Diagnostic PnL | Validation PF | Diagnostic PF | Jitter Gate | Result |
|---|---|---:|---:|---:|---:|---|---|
| `attempt_120` | `put_near_after_0940_vwap_m2_10_omar_neg` | $5,850 | -$1,980 | 2.851 | 0.490 | fail | fail |
| `attempt_121` | `put_near_after_0940_vwap_m2_10_range_20_45` | $2,390 | -$250 | 1.616 | 0.926 | fail | fail |
| `attempt_122` | `put_near_after_0940_vwap_m2_10_near_vwap` | $5,650 | $605 | 2.859 | 1.110 | fail | fail |
| `attempt_123` | `put_near_after_0940_vwap_m2_10_premium_gte_7_5` | $4,220 | -$930 | 1.783 | 0.849 | fail | fail |

Interpretation:

- Compound first-stage gates improved the evidence but did not produce a paper-ready candidate.
- The near-VWAP branch almost survives strict replay, with 21 validation trades and 24 diagnostic trades, but diagnostic PF is only 1.110 against the required 1.25.
- All four attempts remain fragile under spread and quote-size perturbations. This is the same root problem seen in attempt107 paired replay: selected-contract ranking depends too much on vendor-sensitive option microstructure.
- The next repair should target ranking stability or aggregate candidate selection, not more narrow hand filters.

Current compound-selector conclusion: `compound_context_gates_not_sufficient_without_ranking_stability`.

## Stable-Offset Selection Search

Three follow-up attempts tested whether the near-VWAP branch could keep its entry signal while making exact contract selection less sensitive to tiny cross-vendor score/ranking changes. Instead of always choosing the top-scored eligible contract, these attempts chose the eligible put closest to a fixed absolute offset target: 10, 15, or 20 SPX points from ATM.

Evidence:

- Stable-selection robustness report: `v4/audit/autoresearch/protocol101_fair_contract_model_search_expanded_jul_dec2025_128_q1_stable_selection_robustness/report.md`

Result: all attempts failed. Best attempt within the failed set: `attempt_125_policy0_hgb_blend35_relative_nearvwap_stableabs15_cap3_dailyloss500_s42_jittergate`.

| Attempt | Selection Mode | Validation PnL | Diagnostic PnL | Validation PF | Diagnostic PF | Strict Replay | Jitter Gate | Result |
|---|---|---:|---:|---:|---:|---|---|---|
| `attempt_124` | `stable_abs_offset_10` | $7,555 | $270 | 4.522 | 1.058 | fail | fail | fail |
| `attempt_125` | `stable_abs_offset_15` | $6,615 | $35 | 4.454 | 1.008 | fail | fail | fail |
| `attempt_126` | `stable_abs_offset_20` | $4,925 | $555 | 3.397 | 1.130 | fail | fail | fail |

Interpretation:

- Deterministic fixed-offset selection reduced one form of contract-ranking freedom, but it did not solve paper readiness.
- All three attempts kept strong validation PnL and then collapsed on diagnostic PF below the required 1.25.
- The 20-point selector passed the initial candidate validation gate, but still failed strict replay and the option-feature jitter gate.
- Spread/size perturbations still changed selected contracts or PnL materially, so the remaining problem is not just “pick a fixed strike distance.”
- The next repair should move from per-contract top-score selection to a decision-level or aggregate candidate-value objective. The model should learn whether a minute/setup is tradable and then select a contract through a stable causal policy, rather than relying on tiny per-contract score differences across vendors.

Current stable-selection conclusion: `fixed_offset_selection_not_sufficient_for_paper_ready_candidate`.

## Aggregate Decision-Objective Search

Four follow-up attempts changed the learning target instead of only changing the selected contract. These attempts asked the model to learn whether the decision minute/setup was tradable, then used deterministic stable-offset selection for the exact contract. This was designed to remove the vendor-sensitive per-contract ranking dependence seen in attempt107, while still using only fair `protocol101-live-v1` labels and excluding June/July recorder days from training and threshold selection.

Evidence:

- Aggregate decision-objective report: `v4/audit/autoresearch/protocol101_fair_contract_model_search_expanded_jul_dec2025_128_q1_aggregate_decision_objective/report.md`

Result: all attempts failed strict paper-readiness, but all four passed the option-feature jitter gate. Best attempt within the failed set: `attempt_130_policy0_hgb_decisionpresence_nearvwap_stableabs20_cap3_dailyloss500_s42_jittergate`.

| Attempt | Target | Selection Mode | Validation PnL | Diagnostic PnL | Validation PF | Diagnostic PF | Trades V/D | Jitter Gate | Result |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| `attempt_127` | `decision_best_profit_regression` | `stable_abs_offset_15` | $3,100 | -$2,140 | 1.695 | 0.620 | 23 / 24 | pass | fail |
| `attempt_128` | `decision_best_profit_regression` | `stable_abs_offset_20` | $3,690 | -$2,000 | 1.893 | 0.610 | 25 / 24 | pass | fail |
| `attempt_129` | `decision_profit_presence_classifier` | `stable_abs_offset_15` | $4,280 | $120 | 3.365 | 1.043 | 18 / 17 | pass | fail |
| `attempt_130` | `decision_profit_presence_classifier` | `stable_abs_offset_20` | $4,760 | $50 | 3.850 | 1.020 | 20 / 17 | pass | fail |

Interpretation:

- This is the best-shaped failure so far: every aggregate attempt passed the option-feature jitter gate.
- That supports the hypothesis that exact per-contract ranking, not the broad fair feature contract, was the most vendor-sensitive piece.
- The current aggregate variants are still not paper-ready because diagnostic PF and/or trade count are too weak. The best classifier variant reached validation +$4,760 PF 3.850 with 20 trades, but diagnostic was only +$50 PF 1.020 with 17 trades.
- The next repair should preserve the aggregate decision-level objective and stable contract-selection structure, then broaden or reshape the opportunity filter/target so diagnostic edge and trade count recover without reintroducing vendor-sensitive ranking.

Current aggregate-objective conclusion: `aggregate_objective_repairs_jitter_shape_but_not_diagnostic_edge`.

## Broader Aggregate Decision-Objective Search

Two follow-up attempts kept the best-shaped aggregate classifier structure, kept deterministic 20-point contract selection, and widened the opportunity filter to test whether the near-VWAP gate was starving the diagnostic split. Both attempts still excluded June/July recorder days from training and threshold selection.

Evidence:

- Broader aggregate decision-objective report: `v4/audit/autoresearch/protocol101_fair_contract_model_search_expanded_jul_dec2025_128_q1_broader_aggregate_decision_objective/report.md`

Result: both attempts failed strict paper-readiness, but both passed the option-feature jitter gate.

| Attempt | Entry Filter | Validation PnL | Diagnostic PnL | Validation PF | Diagnostic PF | Trades V/D | Jitter Gate | Result |
|---|---|---:|---:|---:|---:|---:|---|---|
| `attempt_131` | `put_near_after_0940_vwap_m2_10` | $1,970 | -$330 | 1.540 | 0.925 | 21 / 22 | pass | fail |
| `attempt_132` | `put_near_after_0940_vwap_m2_10_premium_gte_7_5` | $1,970 | -$330 | 1.540 | 0.925 | 21 / 22 | pass | fail |

Interpretation:

- Broadening the aggregate classifier did not recover diagnostic edge.
- The jitter pass stayed intact, which confirms the aggregate/stable-selection structure is less vendor-fragile than per-contract top-score selection.
- The blocker has shifted: the remaining issue is not primarily IBKR-vs-Databento option microstructure sensitivity; it is insufficient out-of-sample expectancy under the fair live-reproducible game.
- The next repair should inspect aggregate-selected diagnostic losers and winners, then decide whether the edge loss belongs to the entry target, the stable contract selector, the lifecycle/exit policy, or the current fixed trade-cap/daily-loss/cooldown assumptions.

Current broader-aggregate conclusion: `aggregate_stability_confirmed_but_diagnostic_edge_missing`.

## Aggregate Edge-Loss Diagnostic

Two aggregate attempts were diagnosed with the fair-contract failure diagnostic to explain why jitter-stable candidates still lost diagnostic expectancy.

Evidence:

- Attempt130 edge-loss diagnostic: `v4/audit/autoresearch/protocol101_fair_contract_aggregate_edge_loss_diagnostic_attempt130/report.md`
- Attempt131 edge-loss diagnostic: `v4/audit/autoresearch/protocol101_fair_contract_aggregate_edge_loss_diagnostic_attempt131/report.md`

Key findings:

| Attempt | Validation Selected PnL | Diagnostic Selected PnL | Diagnostic PF | Main Diagnostic Clue |
|---|---:|---:|---:|---|
| `attempt_130` | $5,160 raw label | $390 raw label | 1.167 | positive 15-minute momentum put entries lost -$1,290 |
| `attempt_131` | $2,980 raw label | $110 raw label | 1.027 | positive 15-minute momentum remained weak; broader filter added more weak trades |

Interpretation:

- Aggregate/stable selection repaired much of the option-feature jitter shape, but it did not learn enough out-of-sample expectancy.
- The strongest loser bucket was causal and inspectable: put entries during positive 15-minute momentum.
- This suggested two follow-ups: remove positive-momentum put entries, and test side-aware momentum selection.

Current edge-loss diagnostic conclusion: `aggregate_jitter_fixed_but_expectancy_missing`.

## Momentum-Filtered Aggregate Search

Three follow-up attempts kept the aggregate classifier and stable 20-point selector, then removed positive-momentum put entries through causal filters.

Evidence:

- Momentum-filtered aggregate report: `v4/audit/autoresearch/protocol101_fair_contract_model_search_expanded_jul_dec2025_128_q1_momentum_filtered_aggregate_objective/report.md`

Result: all attempts failed strict paper-readiness, but all passed the option-feature jitter gate.

| Attempt | Entry Filter | Validation PnL | Diagnostic PnL | Validation PF | Diagnostic PF | Trades V/D | Jitter Gate | Result |
|---|---|---:|---:|---:|---:|---:|---|---|
| `attempt_133` | `put_near_after_0940_vwap_m2_10_mom15_nonpos` | $3,430 | -$460 | 2.429 | 0.859 | 21 / 13 | pass | fail |
| `attempt_134` | `put_near_after_0940_vwap_m2_10_omar_pos_mom15_nonpos` | $1,630 | -$150 | 1.668 | 0.947 | 15 / 11 | pass | fail |
| `attempt_135` | `put_near_after_0940_vwap_m2_10_premium_gte_7_5_mom15_nonpos` | $2,730 | -$180 | 1.881 | 0.945 | 21 / 13 | pass | fail |

Interpretation:

- Removing positive-momentum put entries preserved jitter robustness.
- It did not recover diagnostic PnL or trade count.
- The best momentum-filtered attempt still had diagnostic -$180, PF 0.945, and only 13 diagnostic trades.
- This rules out a simple “block positive momentum puts” repair.

Current momentum-filter conclusion: `causal_momentum_block_not_sufficient`.

## Side-Aware Aggregate Search

Two follow-up attempts tested whether the put-only specialist was the remaining issue. Positive 15-minute momentum allowed calls; non-positive momentum allowed puts. Contract selection remained deterministic stable_abs_offset_20.

Evidence:

- Side-aware aggregate report: `v4/audit/autoresearch/protocol101_fair_contract_model_search_expanded_jul_dec2025_128_q1_side_aware_aggregate_objective/report.md`

Result: both attempts failed strict paper-readiness, but both passed the option-feature jitter gate.

| Attempt | Entry Filter | Validation PnL | Diagnostic PnL | Validation PF | Diagnostic PF | Trades V/D | Jitter Gate | Result |
|---|---|---:|---:|---:|---:|---:|---|---|
| `attempt_136` | `near_after_0940_vwap_m2_10_mom15_side` | $3,050 | -$2,700 | 1.753 | 0.636 | 22 / 23 | pass | fail |
| `attempt_137` | `near_after_0940_vwap_m2_10_mom15_side_premium_gte_7_5` | $3,050 | -$2,700 | 1.753 | 0.636 | 22 / 23 | pass | fail |

Interpretation:

- Side-switching on momentum did not recover edge; it materially worsened diagnostic performance.
- This argues against the next fix being another simple side/context gate.
- The remaining problem is likely in the entry label, candidate target, lifecycle/exit assumptions, or how continuation/reversal is priced under the fair live-reproducible contract.

Current side-aware conclusion: `side_switching_not_sufficient`.

## Stability And Failure Diagnostics

Additional offline diagnostics were run after the 128-session pass.

Evidence:

- Stability packet: `v4/audit/autoresearch/protocol101_fair_contract_attempt107_stability_packet/report.md`
- Failure diagnostic: `v4/audit/autoresearch/protocol101_fair_contract_attempt107_128_failure_diagnostic/report.md`

Stability finding:

- The stability packet status is `pass` because it found stable positive buckets.
- Attempt107 itself shows `split_sign_flip` in the cross-window stability comparison.
- The strongest attempt107 stable bucket is premium `7_5_to_15`: validation 18 trades, $2,380 PnL, PF 2.178; diagnostic 28 trades, $810 PnL, PF 1.266.
- This is useful but marginal. It is evidence for a possible gated specialist, not paper-readiness.

Failure-diagnostic finding:

| Split | Decisions | Candidates | Selected | Raw-label PnL | PF | Missed Profitable Decisions | Score/Label Corr |
|---|---:|---:|---:|---:|---:|---:|---:|
| validation | 3,590 | 107,478 | 24 | $3,975 | 2.199 | 3,089 | 0.028 |
| diagnostic_test | 3,590 | 109,463 | 26 | $1,875 | 1.587 | 3,028 | -0.018 |

The model is not broadly ranking all profitable opportunities well. It is finding a narrow causal pocket:

- right: puts only
- offset bucket: near 10-20 only
- strongest time buckets: open and morning
- weakest time buckets: afternoon and late
- stronger contexts: below VWAP, momentum15 negative/flat, lower range buckets
- weaker contexts: momentum15 positive, high range buckets

That makes attempt107 a useful lead, but it should be treated as a fragile specialist until it passes out-of-sample replay against captured IBKR days.

## Current Recommendation

Do not promote attempt107. Do not enable paper-submit.

Proceed with offline synchronization replay:

1. Reject attempt107 as paper-ready in its current form because it fails paired replay through option-feature data-plane sensitivity.
2. Do not try to adapter-patch June/July confirmation days into agreement; that would overfit the synchronization evidence.
3. Continue with a robustness-aware design that can use option information without brittle dependence on tiny spread/size vendor deltas.
4. The first deletion, quantization, hard score-margin, naive fit-noise augmentation, jitter-stressed threshold, broader entry-filter, compound two-stage selector, and fixed-offset contract-selection attempts failed. Aggregate decision-objective attempts improved option-feature jitter stability, but still failed diagnostic PF/trade-count gates. Broader aggregate filters kept the jitter pass but lost diagnostic PnL. Momentum-filtered and side-aware aggregate follow-ups also kept jitter stability but failed diagnostic edge. The next candidate family should stop adding narrow context gates and diagnose whether the repair belongs in the entry label, target construction, lifecycle/exit policy, or continuation/reversal pricing.
5. Any future candidate still needs the same historical gates, same-input IBKR replay, paired IBKR-vs-historical replay, concentration checks, and final baseline comparison before paper-submit review.

Attempt107 can remain the current diagnostic lead, and attempts 122/125/130/131/135/136 are the best failed compound/stable/aggregate branches, but none are paper-ready. The microstructure-masked, microstructure-bucketed, compound-gated, fixed-offset, momentum-filtered, and side-aware attempts prove that simply deleting/coarsening fields, narrowing context, choosing a fixed strike distance, or flipping side on momentum is not enough. The aggregate attempts show the most promising repair shape because jitter stability improved, but the next model still needs real diagnostic expectancy under the fair live-reproducible game.

## Stop-State For This Packet

This packet advances the active goal by identifying a current lead and its exact blockers. It does not satisfy the full goal. The full goal remains active until a candidate passes historical validation, same-input IBKR replay, paired IBKR-vs-historical replay, concentration/drawdown robustness, and a separate owner-approved paper-submit review.
