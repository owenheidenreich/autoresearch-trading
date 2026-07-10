# Protocol101 Training-Scope Acceptance Registry

## Decision

- Status: `pass`
- Included pass sessions: `301`
- Excluded sessions: `10`
- Source registry status: `fail`
- Source registry hash: `3adf3fd6357912fa710160de9147757e0c6bb7232485902c6e0043557ed06fbc`
- Training-scope registry hash: `6c656cfb2caeaab1d03e78eee39a164f7d169709abeaec5b0c490f1b1e29f0fe`

## Exclusions

- `2025-04-07`: source_status=`report_only`, report_only_reason=`low_tradable_liquidity`, failed_checks=`mean_tradable_candidates,near_atm_tradable_share`
- `2025-04-08`: source_status=`report_only`, report_only_reason=`low_tradable_liquidity`, failed_checks=`near_atm_tradable_share`
- `2025-04-09`: source_status=`report_only`, report_only_reason=`low_tradable_liquidity`, failed_checks=`mean_tradable_candidates,near_atm_tradable_share`
- `2025-04-10`: source_status=`report_only`, report_only_reason=`low_tradable_liquidity`, failed_checks=`near_atm_tradable_share`
- `2025-04-11`: source_status=`report_only`, report_only_reason=`low_tradable_liquidity`, failed_checks=`near_atm_tradable_share`
- `2025-07-03`: source_status=`report_only`, report_only_reason=`early_close_not_close_aware`, failed_checks=``
- `2025-07-30`: source_status=`report_only`, report_only_reason=`missing_index_context`, failed_checks=`context_lag_exact_one_minute`
- `2025-10-22`: source_status=`fail`, report_only_reason=``, failed_checks=`context_lag_exact_one_minute`
- `2025-11-28`: source_status=`report_only`, report_only_reason=`early_close_not_close_aware`, failed_checks=``
- `2025-12-24`: source_status=`report_only`, report_only_reason=`early_close_not_close_aware`, failed_checks=`context_lag_exact_one_minute`

## Guardrails

- This artifact narrows training inputs; it does not alter verifier thresholds.
- Excluded sessions are not placeable for train, validation, diagnostic gates, or uplift claims.
- Model training, threshold selection, broker calls, and paper-submit remain false.
