# Round 10 Package

This folder contains the next response package for Fable.

Primary file to paste/upload first:

```text
00_message_to_fable.md
```

Recommended upload set:

```text
00_message_to_fable.md
02_build_protocol101_session_era_manifest_raw_aware.py
03_build_protocol101_era_role_policy.py
05_test_protocol101_session_era_manifest_raw_aware.py
06_test_protocol101_era_role_policy.py
08_PROTOCOL101_DATA_EXPANSION_APPROVAL_REQUEST_DRAFT_2026_07_05.md
09_session_era_manifest_report.md
10_session_era_manifest_summary.json
12_era_role_policy_report.md
13_era_role_policy_summary.json
17_test_and_artifact_output.txt
```

What this packet covers:

- Makes owned raw sessions visible in the session-era manifest.
- Adds `pre_program_oct2024_jun2025`.
- Splits unassigned reasons into `no_matching_rule` and `overlapping_rules`.
- Adds separate `Protocol101EraRolePolicyV1`.
- Rewrites the data expansion approval draft so it no longer falsely claims a paid purchase is required to unblock the first fold program.

No broker connectivity, paper-submit, paid data downloads, model training, threshold tuning, promotion/default changes, or real-money paths were touched.
