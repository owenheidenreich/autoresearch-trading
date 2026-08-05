# Protocol101 Session Era Manifest

- Status: `pass`
- Manifest hash: `55c38f8265068fca323ea8e1b4eae36d3c8c3212b7b8e77192f803037a7aa144`
- Session count: `193`
- Fail-closed default: `unassigned_requires_decision`
- Allow unassigned: `false`

## Counts By Era

- `confirmation_jun_jul2026`: `4`
- `owned_jul_dec2025`: `128`
- `q1_2026_development`: `61`

## Era Rules

- `owned_jul_dec2025`: `2025-07-01` to `2025-12-31`
- `q1_2026_development`: `2026-01-01` to `2026-03-31`
- `confirmation_jun_jul2026`: `2026-06-01` to `2026-07-31`

## Notes

- The fold scaffold should consume this manifest and refuse to place `unassigned_requires_decision` sessions.
- Recorder sessions are included as confirmation-era evidence and retain their complete/partial source status.
