# Protocol101 Era Role Policy

- Status: `pass`
- Policy hash: `6f51e65f5b9d271f14c83f6752d30390dca1797d5444c64a710bb9b2214fac7d`
- Session manifest hash: `6767427de34e6ef1f1558ab7854c5f7b599068ed05302629bd5aec083793ce16`

## Policy

- `pre_program_oct2024_jun2025`: roles=`['train', 'test', 'diagnostics_only']`, tier=`owned_pre_program_raw_pending_acceptance`
- `owned_jul_dec2025`: roles=`['train', 'test', 'diagnostics_only']`, tier=`owned_mechanically_clean_for_new_fold_models_with_family_selection_caveat`
- `q1_2026_development`: roles=`['diagnostics_only', 'report_only']`, tier=`development_report_only_until_split_ancestry_audit`
- `post_q1_gap_apr_may2026`: roles=`['report_only']`, tier=`post_q1_nonfold_report_only`
- `confirmation_jun_jul2026`: roles=`['confirmation_one_shot', 'report_only']`, tier=`recorder_confirmation_only`
- `unassigned_requires_decision`: roles=`[]`, tier=`blocked`
- `extension_2024h1`: roles=`['train', 'test', 'diagnostics_only']`, tier=`placeholder_pending_owner_approval_and_acceptance`
- `extension_2023`: roles=`['train', 'test', 'diagnostics_only']`, tier=`placeholder_pending_owner_approval_and_acceptance`

## Role Taxonomy

- `train`: May appear in model-tier training windows when all acceptance and fold predicates pass.
- `test`: May appear in model-tier test windows when all acceptance and fold predicates pass.
- `diagnostics_only`: May appear in diagnostics-tier fold test windows for gates, samplers, nulls, and uplift; never model-tier train/test.
- `report_only`: May be summarized for context but not used for model selection, diagnostics gates, or promotion claims.
- `confirmation_one_shot`: May be used only for one-shot live/parity confirmation, not training or tuning.

## Promotion Guards

- `pre_program_systematic_negative_guard`: if `pooled criteria pass but pre_program_oct2024_jun2025 test folds are systematically negative` then `regime_bound_requires_owner_review`.

## Notes

- The fold scaffold should consume this policy alongside the session era manifest.
- Policy changes should update this artifact without mutating the session facts manifest.
