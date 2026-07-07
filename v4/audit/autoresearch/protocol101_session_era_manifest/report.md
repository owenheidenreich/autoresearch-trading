# Protocol101 Session Era Manifest

- Status: `pass`
- Manifest hash: `6767427de34e6ef1f1558ab7854c5f7b599068ed05302629bd5aec083793ce16`
- Session count: `429`
- Fail-closed default: `unassigned_requires_decision`
- Allow unassigned: `false`

## Counts By Era

- `confirmation_jun_jul2026`: `9`
- `owned_jul_dec2025`: `131`
- `post_q1_gap_apr_may2026`: `35`
- `pre_program_oct2024_jun2025`: `191`
- `q1_2026_development`: `63`

## Era Rules

- `pre_program_oct2024_jun2025`: `2024-10-01` to `2025-06-30`
- `owned_jul_dec2025`: `2025-07-01` to `2025-12-31`
- `q1_2026_development`: `2026-01-01` to `2026-03-31`
- `post_q1_gap_apr_may2026`: `2026-04-01` to `2026-05-31`
- `confirmation_jun_jul2026`: `2026-06-01` to `2026-07-31`

## Notes

- The fold scaffold should consume this manifest and refuse to place `unassigned_requires_decision` sessions.
- Recorder sessions are included as confirmation-era evidence and retain their complete/partial source status.
