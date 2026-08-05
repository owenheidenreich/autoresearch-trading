# Protocol101 Fair-Contract Model Search

## Decision

- Best attempt: `attempt_135_policy0_hgb_decisionpresence_premiumgte7_5_mom15nonpos_stableabs20_cap3_dailyloss500_s42_jittergate`
- Best score: `-173733.97`
- Reason: `rejected:diagnostic_positive_pnl,diagnostic_profit_factor,diagnostic_trade_count`
- Strict replay status: `fail`

## Attempt Summary

| Attempt | Family | Fit mode | Weighting | Feature transform | Noise aug | Jitter gate | Policy | Entry filter | Selection | Margin | Score ceiling | Session cap | Daily loss | Threshold rule | Validation PnL | Diagnostic PnL | Validation PF | Diagnostic PF | Status |
|---|---|---|---|---|---|---|---:|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|
| `attempt_135_policy0_hgb_decisionpresence_premiumgte7_5_mom15nonpos_stableabs20_cap3_dailyloss500_s42_jittergate` | sklearn_hist_gradient_boosting | full_train | decision_balanced_classifier | none | none | pass | 0 | put_near_after_0940_vwap_m2_10_premium_gte_7_5_mom15_nonpos | stable_abs_offset_20 | 0.00 | 0.00 | 3 | 500 | jitter_stability_stressed | 2730.00 | -180.00 | 1.881 | 0.945 | `rejected:diagnostic_positive_pnl,diagnostic_profit_factor,diagnostic_trade_count` |
| `attempt_133_policy0_hgb_decisionpresence_mom15nonpos_stableabs20_cap3_dailyloss500_s42_jittergate` | sklearn_hist_gradient_boosting | full_train | decision_balanced_classifier | none | none | pass | 0 | put_near_after_0940_vwap_m2_10_mom15_nonpos | stable_abs_offset_20 | 0.00 | 0.00 | 3 | 500 | jitter_stability_stressed | 3430.00 | -460.00 | 2.429 | 0.859 | `rejected:diagnostic_positive_pnl,diagnostic_profit_factor,diagnostic_trade_count` |
| `attempt_134_policy0_hgb_decisionpresence_omarpos_mom15nonpos_stableabs20_cap3_dailyloss500_s42_jittergate` | sklearn_hist_gradient_boosting | full_train | decision_balanced_classifier | none | none | pass | 0 | put_near_after_0940_vwap_m2_10_omar_pos_mom15_nonpos | stable_abs_offset_20 | 0.00 | 0.00 | 3 | 500 | jitter_stability_stressed | 1630.00 | -150.00 | 1.668 | 0.947 | `rejected:diagnostic_positive_pnl,diagnostic_profit_factor,diagnostic_trade_count,validation_trade_count` |

## Guardrails

- June/July IBKR recorder days were excluded from training and threshold selection.
- No broker endpoints, paper-submit, paid downloads, default changes, or promotions are performed by this search.
- Registry reuse requires current selected-export and strict-replay implementation versions.
- Failed attempts are still registry entries, because negative evidence is part of hill climbing safely.
