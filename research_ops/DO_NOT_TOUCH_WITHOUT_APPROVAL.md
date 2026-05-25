# Do Not Touch Without Approval

Last updated: 2026-05-24

These files and behaviors require explicit user approval before modification or
execution. The scaffold itself does not authorize any of these actions.

## Trading Behavior

- `v4/live/`
- `v4/scripts/`
- `v4/runner/`
- `v4/sim/`
- `v4/ops/ibkr/`
- `v4/ops/launchd/`
- Protocol101 entry logic.
- Protocol051/054 surface scoring logic.
- Protocol066 and Protocol081 lifecycle logic.
- Candidate filtering, thresholds, position sizing, and promotion rules.

## Runtime State

- Runtime flags.
- Paper-order mode files.
- Broker readiness or entitlement state.
- Launchd plists or scheduling.
- Paper session state.
- Account state files.
- Trade logs and order logs.

## Model And Data Artifacts

- `model.pt`, scaler, manifest, and checkpoint artifacts.
- Paid-data raw files.
- Normalized market-data roots.
- Rebuildable parquet datasets.
- Historical replay datasets.
- Vendor download scripts or credentials.

## External Systems

- IBKR or broker APIs.
- Paid data vendors.
- Cloud compute jobs.
- Model training jobs.
- Threshold tuning jobs.
- Challenger promotion jobs.

## Required Exception Packet

Any exception must include:

1. Explicit user approval for the specific protected surface.
2. Cartography report.
3. Experiment RFC or implementation plan.
4. Verifier report.
5. Decision memo.
6. CEO dashboard update.
