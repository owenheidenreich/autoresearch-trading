# Protocol101 Recorder Startup Reliability Note - 2026-07-20

## Context

The 2026-07-20 rehearsal capture missed the regular-session opening window even though the launchd packet was loaded. The recurring failure mode was not the model or recorder logic; IB Gateway/API readiness arrived too late after a slow authentication/re-login path.

Observed symptom:

- Gateway launch was scheduled at 05:30 PT.
- IBKR API readiness was not stable until after the market open path was already damaged.
- The resulting capture was usable as partial development evidence only, not as complete sealed confirmation evidence.

## Active Mitigation

The launchd packet was shifted earlier and given longer retry windows:

- `gateway`: 04:45 PT
- `preflight`: 05:25 PT, extended retries
- `recorder`: 05:40 PT, extended retries
- `health`: 05:50, 06:00, 06:10, 06:20, 06:28 PT
- `watchdog`: 06:05 PT
- `shutdown`: 13:05 PT
- `finalize`: 13:10 PT
- `audit`: 13:15 PT

Gateway now has `IB_GATEWAY_WAIT_SECONDS=2700` so slow IBKR login/re-authentication can finish without premature restart churn.

Preflight and recorder have extended retry settings so a slow but progressing Gateway does not immediately lose the session.

## Automation

The Codex heartbeat automation `protocol101-recorder-pre-open-check` now runs at:

- 05:55 PT: early startup check
- 06:20 PT: final pre-open readiness check
- 13:30 PT: post-session classification and sealing assignment

Do not remove the 05:55 check while IB Gateway startup remains a known weak point.

## Scope Boundaries

This mitigation changes recorder/Gateway startup infrastructure only. It does not change:

- model thresholds
- training or tuning
- paid-data behavior
- promotion/default state
- paper-submit behavior
- real-money trading paths

