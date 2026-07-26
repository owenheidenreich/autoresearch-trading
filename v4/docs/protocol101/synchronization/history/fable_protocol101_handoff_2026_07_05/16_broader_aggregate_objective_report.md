# Protocol101 Fair-Contract Model Search

## Decision

- Best attempt: `attempt_131_policy0_hgb_decisionpresence_basevwap_stableabs20_cap3_dailyloss500_s42_jittergate`
- Best score: `-103296.55`
- Reason: `rejected:diagnostic_positive_pnl,diagnostic_profit_factor`
- Strict replay status: `fail`

## Attempt Summary

| Attempt | Family | Fit mode | Weighting | Feature transform | Noise aug | Jitter gate | Policy | Entry filter | Selection | Margin | Score ceiling | Session cap | Daily loss | Threshold rule | Validation PnL | Diagnostic PnL | Validation PF | Diagnostic PF | Status |
|---|---|---|---|---|---|---|---:|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|
| `attempt_131_policy0_hgb_decisionpresence_basevwap_stableabs20_cap3_dailyloss500_s42_jittergate` | sklearn_hist_gradient_boosting | full_train | decision_balanced_classifier | none | none | pass | 0 | put_near_after_0940_vwap_m2_10 | stable_abs_offset_20 | 0.00 | 0.00 | 3 | 500 | jitter_stability_stressed | 1970.00 | -330.00 | 1.540 | 0.925 | `rejected:diagnostic_positive_pnl,diagnostic_profit_factor` |
| `attempt_132_policy0_hgb_decisionpresence_premiumgte7_5_stableabs20_cap3_dailyloss500_s42_jittergate` | sklearn_hist_gradient_boosting | full_train | decision_balanced_classifier | none | none | pass | 0 | put_near_after_0940_vwap_m2_10_premium_gte_7_5 | stable_abs_offset_20 | 0.00 | 0.00 | 3 | 500 | jitter_stability_stressed | 1970.00 | -330.00 | 1.540 | 0.925 | `rejected:diagnostic_positive_pnl,diagnostic_profit_factor` |

## Guardrails

- June/July IBKR recorder days were excluded from training and threshold selection.
- No broker endpoints, paper-submit, paid downloads, default changes, or promotions are performed by this search.
- Registry reuse requires current selected-export and strict-replay implementation versions.
- Failed attempts are still registry entries, because negative evidence is part of hill climbing safely.
