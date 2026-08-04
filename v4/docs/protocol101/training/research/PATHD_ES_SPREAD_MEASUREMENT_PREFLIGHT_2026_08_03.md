# ES BBO-1s Spread Measurement — Paid Preflight (2026-08-03)

## Status

**PAUSED BEFORE SPEND — authorization window does not match its stated RTH purpose on six sessions.**

The read-only exact cost preflight passed at **$1.470344**, below the authorized $3.00 cap. No
`timeseries.get_range` call was made, no bytes were downloaded, and the paid-data guard was not bypassed.

## Scope validation

The separate manifest correctly restricts dataset/symbol/schema to `GLBX.MDP3`, `ES.FUT`,
`stype_in=parent`, `bbo-1s`, 20 unique named sessions, and a $3 cap. It does not alter the earlier OHLCV
manifest. The exact approval text supplied by the owner matches the new manifest.

However, its single window `13:30–20:00 UTC` is:

- 09:30–16:00 EDT on 14 sessions: correct full RTH;
- **08:30–15:00 EST** on 2025-11-19, 2025-12-08, 2025-12-23, 2026-01-13, 2026-01-30, and 2026-02-18.

Those six requests would add one premarket hour and omit the closing RTH hour—the period most likely to
matter for the requested elevated-volatility conditional spread. Silently changing them to
`14:30–21:00 UTC` would exceed the exact authorized window and violate the hard stop. Spending against the
current scope would not produce the promised 20-session RTH measurement.

## Exact read-only estimate

| Session | Cost | Authorized window in New York |
|---|---:|---|
| 2025-08-01 | $0.078366 | 09:30–16:00 EDT |
| 2025-08-20 | $0.075913 | 09:30–16:00 EDT |
| 2025-09-08 | $0.077587 | 09:30–16:00 EDT |
| 2025-09-25 | $0.072255 | 09:30–16:00 EDT |
| 2025-10-14 | $0.074698 | 09:30–16:00 EDT |
| 2025-10-31 | $0.077387 | 09:30–16:00 EDT |
| 2025-11-19 | $0.081119 | **08:30–15:00 EST** |
| 2025-12-08 | $0.076141 | **08:30–15:00 EST** |
| 2025-12-23 | $0.057457 | **08:30–15:00 EST** |
| 2026-01-13 | $0.069479 | **08:30–15:00 EST** |
| 2026-01-30 | $0.078456 | **08:30–15:00 EST** |
| 2026-02-18 | $0.071016 | **08:30–15:00 EST** |
| 2026-03-09 | $0.096264 | 09:30–16:00 EDT |
| 2026-03-26 | $0.065198 | 09:30–16:00 EDT |
| 2026-04-14 | $0.065563 | 09:30–16:00 EDT |
| 2026-05-01 | $0.067193 | 09:30–16:00 EDT |
| 2026-05-20 | $0.073541 | 09:30–16:00 EDT |
| 2026-06-08 | $0.079918 | 09:30–16:00 EDT |
| 2026-06-25 | $0.069196 | 09:30–16:00 EDT |
| 2026-07-14 | $0.063598 | 09:30–16:00 EDT |
| **Total** | **$1.470344** | **PASS vs $3.00 cap** |

## Required correction

A new or superseding owner-approved manifest must explicitly authorize DST-aware per-session windows:
09:30–16:00 America/New_York, which resolves to 13:30–20:00 UTC in EDT and 14:30–21:00 UTC in EST. The
existing manifest must remain unchanged. Once supplied, the cost must be rechecked for the corrected exact
request before the guarded download.

STOP_FOR_CLAUDE_VERIFICATION
