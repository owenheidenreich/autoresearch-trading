# Hypothesis To Promotion Process

This document defines how an idea becomes an experiment, how an experiment becomes a validated research challenger, and how a challenger can eventually replace the paper default.

The purpose is to prevent shortcuts. A model is not better because one backtest looks exciting. A model is better only if it survives the same live-like game it will eventually trade.

## Current Standing

```text
Paper default: PAPER_DEFAULT_PROTOCOL101
Replacement baseline: PAPER_DEFAULT_PROTOCOL101 under strict one-account serial replay
Current challenger status: research-only unless a promotion packet explicitly changes this
```

Every new item must use the naming roles in [NAMING_GUIDE.md](NAMING_GUIDE.md), and every model comparison must follow [MODEL_IMPROVEMENT_GUIDELINES.md](MODEL_IMPROVEMENT_GUIDELINES.md).

## Stage Map

| Stage | Role label | Main question | Possible decision |
|---|---|---|---|
| 0. Observation | `AUDIT_*` or notes | What failed or what opportunity did we observe? | Form hypothesis or do nothing |
| 1. Hypothesis | `EXP_*` draft | What exactly are we testing, and why should it help? | Approve experiment design or revise |
| 2. Experiment | `EXP_*` | Does the change train/run correctly on allowed data? | Reject, debug, or advance to validation |
| 3. Validation | `AUDIT_*` / validation report | Is the result actually better under the correct metric scope? | Reject, iterate, or freeze as challenger |
| 4. Research Freeze | `DECISION_FREEZE_*` | Is this a stable research challenger worth preserving? | Freeze research-only or reject |
| 5. Runtime Parity | `RUNTIME_*` | Can the same model/action surface run live-safe without orders? | Block, fix parity, or advance |
| 6. Paper Promotion | `DECISION_PROMOTE_*` | Should this replace the paper default? | Keep default or promote |
| 7. Paper Operation | `PAPER_DEFAULT_*` | Does it behave correctly in paper trading? | Continue, roll back, or restrict |
| 8. Real-Money Review | `DECISION_REAL_MONEY_*` | Is real capital justified? | Usually no; explicit approval required |

No stage may be skipped because a PnL number looks good.

## Stage 0: Observation

Purpose: identify a real failure mode or opportunity.

Valid inputs:

- equity/trades visual inspection,
- attribution report,
- live/paper logs,
- missed-trade diagnosis,
- churn/re-entry audit,
- runtime parity failure,
- slippage/timing stress failure,
- data-quality finding.

Required output:

```text
Observed behavior:
Why it matters:
Evidence file(s):
Possible hypothesis:
```

Rules:

- Do not start by adding a model knob.
- Do not treat protected holdout behavior as permission to tune directly to that holdout.
- If the observation comes from a chart, confirm the chart's metric scope first.

## Stage 1: Hypothesis

Purpose: pre-register what change is being tested before results are known.

Required fields:

```text
What is this:
Candidate role label:
Hypothesis:
Failure mode addressed:
Baseline it must beat:
Allowed data:
Training splits:
Validation splits:
Protected test splits:
Primary metric:
Secondary metrics:
Required stress checks:
Expected failure mode if wrong:
Does this change paper default: no
Paid data required: yes/no
Broker endpoint called: no
```

Rules:

- A hypothesis must be narrow enough that a failure teaches something.
- The primary metric must be selected before seeing test results.
- Protected test blocks cannot select thresholds, objectives, architectures, filters, or sizing rules.
- Paid data requires a separate explicit data request with source, date range, products, estimate, cap, and reason.

Allowed decisions:

- `hypothesis_ready_to_run`
- `hypothesis_needs_data_request`
- `hypothesis_rejected_too_vague`
- `hypothesis_rejected_not_train_live_reproducible`

## Stage 2: Experiment

Purpose: implement and run the proposed change on allowed data.

Experiment types:

- model architecture,
- objective/loss,
- feature set,
- label construction,
- action space,
- lifecycle/exit behavior,
- candidate generation,
- sizing overlay,
- simulator/accounting change,
- runtime parity harness.

Required controls:

- fixed allowed data list,
- deterministic seed list when applicable,
- train/validation/test separation,
- no future columns in features,
- no paid download unless already approved,
- no broker orders,
- paper default unchanged.

Required outputs:

```text
experiment report
summary.json
model artifact manifest, if trained
trade log, if replayed
invariant checks
```

Minimum invariant checks:

- no overlapping headline trades,
- no unaffordable headline trades,
- all sessions flat by close,
- ask-entry / bid-exit accounting,
- no future/path/exit columns in runtime features,
- feature timestamps are `<= decision_time`,
- SPXW PM-only contracts,
- `$5` strike alignment,
- no stale/malformed quote rows selected.

Allowed decisions:

- `experiment_failed_to_run`
- `experiment_rejected_invariant_failure`
- `experiment_rejected_no_improvement`
- `experiment_passed_to_validation`

## Stage 3: Validation

Purpose: decide whether the experiment is truly better or merely different.

Validation must be apples-to-apples.

Required comparison:

```text
Candidate vs PAPER_DEFAULT_PROTOCOL101
Metric scope: strict one-account serial replay
Starting equity: $10,000
Position limit: current operational limit unless explicitly studying sizing
Fills: ask entry, bid exit
```

Required views:

- PnL by split,
- profit factor by split,
- win rate by split,
- average PnL per trade,
- median premium,
- PnL per premium,
- drawdown,
- worst day,
- slippage stress,
- concentration,
- side exposure,
- moneyness exposure,
- time bucket exposure,
- churn/re-entry behavior,
- large directional move capture,
- skipped opportunity cost.

Metric-scope rule:

Every table must say whether it is:

- single-seed paper account replay,
- five-seed median,
- five-seed total,
- attribution subset,
- directional-move subset,
- overlapping diagnostic,
- live/no-order replay,
- paper-trade replay.

Allowed decisions:

- `validation_rejects_candidate`
- `validation_requires_attribution`
- `validation_supports_research_freeze`
- `validation_supports_runtime_parity_gate`

Promotion is not an allowed Stage 3 decision.

## Stage 4: Research Freeze

Purpose: preserve a stable challenger that is worth studying further.

A freeze packet says:

```text
This candidate is stable enough for continued research.
It does not replace the paper default.
```

Required fields:

```text
Candidate:
Baseline:
Data used:
Metric scope:
Why it passed research freeze:
Known weaknesses:
What it must prove next:
Does it change paper default: no
```

Research freeze can happen when:

- the challenger beats the paper default on core historical replay but has unresolved runtime or behavior risk,
- the challenger reveals a useful feature/objective direction,
- the challenger is not yet safe to run in paper trading.

Research freeze must not hide weaknesses. If win rate drops, churn rises, or drawdown worsens, say so in the freeze packet.

Allowed decisions:

- `freeze_research_challenger`
- `freeze_research_only_with_blockers`
- `reject_do_not_freeze`

## Stage 5: Runtime Parity

Purpose: prove the model can play the same game live that it played historically.

Runtime parity is live-safe and no-order by default.

Required checks:

- same candidate filters,
- same feature names,
- same feature calculations,
- same account-state fields,
- same action space,
- same stale quote rules,
- same SPXW contract universe,
- same affordability masks,
- same output schema,
- no future labels,
- no broker order endpoint,
- latency logged,
- quote freshness logged,
- blocked reasons logged.

Required outputs:

```text
live/no-order JSONL
schema validation report
candidate breadth report
latency/freshness report
feature parity report
```

Allowed decisions:

- `runtime_parity_failed`
- `runtime_parity_blocked_by_entitlements`
- `runtime_parity_passed_no_orders`
- `runtime_parity_passed_ready_for_promotion_review`

Runtime parity is not the same as paper promotion. It only says the candidate can be observed live safely.

## Stage 6: Paper Promotion

Purpose: decide whether a challenger replaces the current paper default.

This stage requires a dedicated promotion decision packet.

Required preconditions:

- Stage 3 validation passed under the correct metric scope.
- Stage 4 research freeze exists.
- Stage 5 runtime parity passed.
- Known weaknesses are documented.
- Paper account risk rules are defined.
- Rollback plan exists.
- User explicitly approves replacement.

Promotion packet must include:

```text
Old paper default:
New proposed paper default:
Why replacement is justified:
Historical validation summary:
Runtime parity summary:
Risk controls:
Known weaknesses:
Rollback trigger:
Does this enable broker orders:
Paper only or real money:
User approval:
```

Allowed decisions:

- `promotion_rejected_keep_current_default`
- `promotion_deferred_need_more_live_observation`
- `promotion_approved_paper_only`

Real-money approval is never implied by paper promotion.

## Stage 7: Paper Operation

Purpose: collect real paper-trading evidence.

Required daily logs:

- startup status,
- IBKR connection status,
- market-data entitlement status,
- model decisions,
- candidate set summary,
- selected contracts,
- risk-gate decisions,
- intended orders,
- submitted orders,
- fills,
- exits,
- account state,
- realized PnL,
- failures and blocked reasons.

Daily review must answer:

```text
Did the bot start?
Did it receive fresh data?
Did it evaluate the correct candidate universe?
Did it place only allowed paper orders?
Were fills close to historical assumptions?
Did logs close cleanly?
Did behavior match the historical simulator?
```

Allowed decisions:

- `continue_paper_default`
- `paper_default_restricted`
- `paper_default_rolled_back`
- `paper_logs_ready_for_offline_analysis`

## Stage 8: Real-Money Review

Purpose: decide whether risking real capital is justified.

This is intentionally separate from paper promotion.

Required preconditions:

- stable paper operation over multiple sessions,
- fill/slippage evidence consistent with assumptions,
- risk controls tested,
- drawdown behavior acceptable,
- logs complete,
- explicit user approval,
- separate runtime flag,
- separate real-money promotion packet.

Allowed decisions:

- `real_money_rejected`
- `real_money_deferred`
- `real_money_approved_with_limits`

Default decision is rejection/defer until evidence is overwhelming.

## State Transitions

```text
Observation
  -> Hypothesis
  -> Experiment
  -> Validation
  -> Research Freeze
  -> Runtime Parity
  -> Paper Promotion
  -> Paper Operation
  -> Real-Money Review
```

Failure transitions:

```text
Experiment failure -> attribution -> revised hypothesis
Validation failure -> attribution -> revised hypothesis
Runtime parity failure -> parity fix, not model promotion
Paper operation failure -> rollback/restrict before new research claims
```

## Stop Conditions

Stop and ask for user approval when:

- paid market data is required,
- broker order behavior changes,
- paper default would change,
- real-money trading would be enabled,
- max contract quantity would increase,
- account-level risk caps would loosen,
- protected holdout results tempt a direct tuning change.

## Current Application To The Premium-Leaning Challenger

`CHALLENGER_PREMIUM_LEANING_BLENDED_UTILITY_V1` has not completed this process.

Current status:

```text
Stage: Research freeze / runtime parity preparation
Paper default changed: no
Reason: higher total PnL but lower win rate, higher trade count, and unresolved runtime/execution-risk profile
Next valid step: no-order live surface/runtime parity, not paper promotion
```

This is the intended behavior of the process. It prevents a research-useful but operationally unproven model from replacing a safer paper default.

