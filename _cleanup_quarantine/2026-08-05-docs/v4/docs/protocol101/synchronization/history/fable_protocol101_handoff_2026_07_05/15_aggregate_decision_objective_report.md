# Protocol101 Fair-Contract Model Search

## Decision

- Best attempt: `attempt_130_policy0_hgb_decisionpresence_nearvwap_stableabs20_cap3_dailyloss500_s42_jittergate`
- Best score: `-91297.06`
- Reason: `rejected:diagnostic_profit_factor,diagnostic_trade_count`
- Strict replay status: `fail`

## Attempt Summary

| Attempt | Family | Fit mode | Weighting | Feature transform | Noise aug | Jitter gate | Policy | Entry filter | Selection | Margin | Score ceiling | Session cap | Daily loss | Threshold rule | Validation PnL | Diagnostic PnL | Validation PF | Diagnostic PF | Status |
|---|---|---|---|---|---|---|---:|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|
| `attempt_130_policy0_hgb_decisionpresence_nearvwap_stableabs20_cap3_dailyloss500_s42_jittergate` | sklearn_hist_gradient_boosting | full_train | decision_balanced_classifier | none | none | pass | 0 | put_near_after_0940_vwap_m2_10_near_vwap | stable_abs_offset_20 | 0.00 | 0.00 | 3 | 500 | jitter_stability_stressed | 4760.00 | 50.00 | 3.850 | 1.020 | `rejected:diagnostic_profit_factor,diagnostic_trade_count` |
| `attempt_129_policy0_hgb_decisionpresence_nearvwap_stableabs15_cap3_dailyloss500_s42_jittergate` | sklearn_hist_gradient_boosting | full_train | decision_balanced_classifier | none | none | pass | 0 | put_near_after_0940_vwap_m2_10_near_vwap | stable_abs_offset_15 | 0.00 | 0.00 | 3 | 500 | jitter_stability_stressed | 4280.00 | 120.00 | 3.365 | 1.043 | `rejected:diagnostic_profit_factor,diagnostic_trade_count,validation_trade_count` |
| `attempt_128_policy0_hgb_decisionbest_nearvwap_stableabs20_cap3_dailyloss500_s42_jittergate` | sklearn_hist_gradient_boosting | full_train | none | none | none | pass | 0 | put_near_after_0940_vwap_m2_10_near_vwap | stable_abs_offset_20 | 0.00 | 0.00 | 3 | 500 | jitter_stability_stressed | 3690.00 | -2000.00 | 1.893 | 0.610 | `rejected:diagnostic_positive_pnl,diagnostic_profit_factor` |
| `attempt_127_policy0_hgb_decisionbest_nearvwap_stableabs15_cap3_dailyloss500_s42_jittergate` | sklearn_hist_gradient_boosting | full_train | none | none | none | pass | 0 | put_near_after_0940_vwap_m2_10_near_vwap | stable_abs_offset_15 | 0.00 | 0.00 | 3 | 500 | jitter_stability_stressed | 3100.00 | -2140.00 | 1.695 | 0.620 | `rejected:diagnostic_drawdown,diagnostic_positive_pnl,diagnostic_profit_factor` |

## Guardrails

- June/July IBKR recorder days were excluded from training and threshold selection.
- No broker endpoints, paper-submit, paid downloads, default changes, or promotions are performed by this search.
- Registry reuse requires current selected-export and strict-replay implementation versions.
- Failed attempts are still registry entries, because negative evidence is part of hill climbing safely.
