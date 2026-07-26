# Protocol101 Era Role Policy

- Status: `pass`
- Policy hash: `d9774a2821e5f3a1be6ec1e59ffc614b4f28b7648c297605af874ad7136eb98c`
- Session manifest hash: `6767427de34e6ef1f1558ab7854c5f7b599068ed05302629bd5aec083793ce16`

## Policy

- `pre_program_oct2024_jun2025`: roles=`['train', 'test', 'diagnostics_only']`, tier=`owned_pre_program_raw_pending_acceptance`
- `owned_jul_dec2025`: roles=`['train', 'test', 'diagnostics_only']`, tier=`owned_mechanically_clean_for_new_fold_models_with_family_selection_caveat`
- `q1_2026_development`: roles=`['diagnostics_only', 'report_only']`, tier=`development_report_only_until_split_ancestry_audit`
- `post_q1_gap_apr_may2026`: roles=`['report_only']`, tier=`post_q1_nonfold_report_only`
- `confirmation_jun_jul2026`: roles=`['confirmation_one_shot', 'report_only']`, tier=`recorder_confirmation_only`
- `unassigned_requires_decision`: roles=`[]`, tier=`blocked`

## Notes

- The fold scaffold should consume this policy alongside the session era manifest.
- Policy changes should update this artifact without mutating the session facts manifest.
