# v4 Operating Memory

This file records project assumptions that must not drift between runs.

## Capital Assumption

- `$500` real cash in IBKR is an account-access and market-data reserve only.
- The paper account starts at `$10,000`.
- The intended future real-money trading bankroll is `$10,000`.
- Do not shrink Protocol101 or future models to fit a `$500` trading account unless the user explicitly changes the capital plan.

## Current Promotion Path

The project is currently in a foundation-hardening pause for new model work.
Use [FOUNDATION_HARDENING_AUDIT.md](FOUNDATION_HARDENING_AUDIT.md) as the
active gate before resuming challenger experiments.

The next approved ML direction is the unified conservative offline policy
foundation in [UNIFIED_CONSERVATIVE_OFFLINE_POLICY.md](UNIFIED_CONSERVATIVE_OFFLINE_POLICY.md).
It abandons Protocol276 as a candidate, keeps Protocol101 as the paper default,
and freezes the target contract around `wait / enter candidate / hold / exit`.
Trajectory extraction status is tracked in
[UNIFIED_POLICY_TRAJECTORY_FOUNDATION.md](UNIFIED_POLICY_TRAJECTORY_FOUNDATION.md).

Do not run open-ended model experiments, threshold sweeps, architecture searches,
or broad paid-data acquisition until these blockers are closed:

- Protocol276 entry/lifecycle alignment fixes from the completed attribution packet.
- Fill-evidence collection for any calibrated stochastic fill model.
- Reservation of a new untouched evaluation block.
- Live no-order parity for challenger full-action candidates/features.
- Protocol270/274 label alignment against deployable simulator assumptions.

Until cash settlement and live market-data subscriptions are ready, focus on:

1. Live-data parity checks.
2. No-order Protocol101 shadow capture readiness.
3. Order-state rehearsal around a `$10,000` paper account.

Do not place live or paper broker orders. Do not buy or download paid market data without explicit approval.

## Model Improvement Discipline

Use [MODEL_IMPROVEMENT_GUIDELINES.md](MODEL_IMPROVEMENT_GUIDELINES.md) for every new challenger, training experiment, and "is this better?" comparison.

Use [HYPOTHESIS_TO_PROMOTION_PROCESS.md](HYPOTHESIS_TO_PROMOTION_PROCESS.md) for the full stage-gate path from hypothesis to experiment to validation to research freeze to runtime parity to paper promotion.

In short:

- Do not call a challenger better unless the metric scope is explicit.
- Do not mix five-seed totals, five-seed medians, attribution subsets, and single-seed equity charts.
- Do not use overlapping independent candidate PnL as headline equity.
- Compare replacement candidates against `PAPER_DEFAULT_PROTOCOL101` under strict one-account serial replay.
- If a model has higher PnL but lower win rate, higher churn, or weaker runtime parity, classify it as research-only until the tradeoff is understood.
