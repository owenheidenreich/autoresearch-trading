# Model Improvement Guidelines

This document is the operating rulebook for v4 model research. Its purpose is to prevent us from accidentally treating a noisy, mismatched, or misleading backtest as a better 0DTE trading model.

The goal is still to train a neural trader that can discover a real, executable edge in SPXW 0DTE long calls and long puts. These guidelines define how a challenger is allowed to claim improvement over the current paper default.

For the full stage-gate process from idea to paper-default replacement, use [HYPOTHESIS_TO_PROMOTION_PROCESS.md](HYPOTHESIS_TO_PROMOTION_PROCESS.md).

## Current Default

```text
Paper default: PAPER_DEFAULT_PROTOCOL101
Current replacement bar: beat Protocol101 under the same live-like serial account game
```

No challenger replaces the paper default unless a dedicated freeze / promotion decision says so explicitly.

## Approved ML Direction

The next best-foundation ML path is [UNIFIED_CONSERVATIVE_OFFLINE_POLICY.md](UNIFIED_CONSERVATIVE_OFFLINE_POLICY.md).
Treat it as a foundation contract, not a trained challenger: Protocol276 is
abandoned as a candidate, Protocol101 remains the paper default, and future
neural work must use `UnifiedDecisionStateV1`, `ExecutionModelV1`, and
`ActionAdvantageLabelV1` unless a later decision packet supersedes them.
Trajectory extraction and coverage are tracked in
[UNIFIED_POLICY_TRAJECTORY_FOUNDATION.md](UNIFIED_POLICY_TRAJECTORY_FOUNDATION.md).

## Current Foundation-Hardening Gate

Project work is split into the sections defined in
[PROJECT_SECTIONS_AND_HILL_CLIMB_GATES.md](PROJECT_SECTIONS_AND_HILL_CLIMB_GATES.md).
Before model hill climbing, Section 1 (architecture/foundation) and Section 2
(data acquisition/preparation) must pass the read-only section gate:

```bash
python3 -m v4.scripts.run_project_section_readiness
python3 -m v4.scripts.run_section3_model_experiment_preflight
```

Before any new challenger training, model tweak, threshold sweep, or architecture
search, check [FOUNDATION_HARDENING_AUDIT.md](FOUNDATION_HARDENING_AUDIT.md)
and the Section 3 preflight output at
`v4/audit/autoresearch/section3_model_experiment_preflight/summary.json`.
If the preflight reports `section3_model_experiment_preflight_blocked`, the only
allowed model-adjacent work is attribution and gate closure.

The active blockers are:

- keep the additional-neural-training pause until the current truth gates are closed;
- collect enough paper/no-order fill observations before using a stochastic fill model;
- collect and freeze the reserved untouched evaluation block before future promotion claims;
- upgrade challenger parity from historical proxy to live no-order full-action parity;
- require preregistered hypothesis packets for any actual model run;
- defer broad historical data purchases until simulator, label, parity, and validation gates are stable.

Formal validation controls now have a read-only strategy matrix and CSCV-style
proxy in `v4/audit/autoresearch/formal_validation_governance/summary.json`.
That is a governance control, not permission to score protected data or promote a
challenger.

## Non-Negotiable Game Definition

Every headline model comparison must use the same trading game the live/paper bot will play:

- `$10,000` starting paper equity unless a report explicitly studies account-size sensitivity.
- SPXW PM-settled 0DTE only.
- Long calls, long puts, or no trade.
- One account, one timeline.
- One open position max for current operational comparisons.
- Ask-entry and bid-exit accounting.
- No fees unless a report is specifically studying fee sensitivity.
- Affordability enforced at entry.
- No overlapping positions in headline equity.
- Flat by close.
- Causal features only; no future/path/exit columns in model inputs.
- Same live-safe candidate-generation contract that the runtime can reproduce.

Independent overlapping candidate PnL is allowed only as a diagnostic. It must not be used as headline equity or as proof that a model is deployable.

## Improvement Loop

Every model-improvement attempt must follow this sequence.

### 1. State The Failure Mode

A new experiment must be motivated by a specific observed failure, not by random knob search.

Valid examples:

- Protocol101 misses full-action-surface opportunities.
- A challenger captures more big moves but churns too much.
- A lifecycle model exits early and then re-enters the same directional idea.
- A premium-leaning model improves capital efficiency but lowers win rate.
- A model has good historical PnL but poor delay/fill robustness.

### 2. Pre-Register The Hypothesis

Before running the experiment, write:

```text
Hypothesis:
Candidate role label:
Baseline:
Data allowed:
Training splits:
Validation splits:
Protected test splits:
Primary metric:
Required stress checks:
Expected failure mode if wrong:
Does this change paper default: no
```

The answer to `Does this change paper default` is `no` unless the work is explicitly a freeze / promotion decision packet.

### 3. Train Only On Allowed Data

Protected blocks must not influence:

- feature selection,
- thresholds,
- objective weights,
- architecture choices,
- sizing rules,
- exit rules,
- candidate filters.

If a protected block reveals a failure, use it for attribution, then design the next experiment using training/validation data only.

### 4. Compare Against The Correct Baseline

The baseline for replacement is not an old overlapping curve and not a five-seed aggregate.

The baseline is:

```text
PAPER_DEFAULT_PROTOCOL101 under strict one-account serial replay.
```

A challenger can be research-useful without being replacement-worthy.

### 5. Report Metric Scope Explicitly

Every metric table must say what scope it represents:

- single-seed paper account replay,
- five-seed median,
- five-seed total,
- attribution subset,
- directional-move subset,
- training split,
- validation split,
- protected holdout,
- live/no-order log replay,
- paper-trade log replay.

Never compare these as if they are the same thing.

The Protocol240/242/246 confusion is the warning example:

```text
Q4 directional >=10 SPX-point capture across all five seeds was not Q4 net account PnL.
Single-seed equity.html PnL was not a five-seed median.
Attribution PnL was not promotion-grade paper-account equity.
```

## Required Evaluation Views

A challenger must produce these views before it can be called better than the paper default.

### Economic Replay

Required:

- per-split PnL,
- per-split profit factor,
- per-split win rate,
- trades,
- average PnL per trade,
- median premium,
- PnL per premium,
- max drawdown,
- worst day,
- concentration by top trades/days/months,
- slippage stress.

Win rate alone is not the objective, but a materially lower win rate must be explained by better expectancy, better capital efficiency, and acceptable drawdown/churn behavior.

### Trade Behavior

Required:

- calls vs puts,
- ITM / ATM / OTM profile,
- premium buckets,
- spread buckets,
- time buckets,
- exit reasons,
- same-side exit/re-entry chains,
- large directional move capture,
- missed-move attribution,
- skipped opportunity cost while holding.

### Account Realism

Required:

- no unaffordable trades,
- no overlapping headline trades,
- all sessions flat by close,
- buying power used,
- daily PnL,
- drawdown,
- skipped/blocked trade reasons.

### Runtime Parity

Required before paper-default replacement:

- live-style candidate generation uses the same feature set as training,
- same filters and masks,
- same account-state fields,
- same stale quote rules,
- same action space,
- same output schema,
- broker endpoint disabled for no-order tests,
- latency and quote freshness logged.

## Minimum Replacement Bar

A challenger is not a better replacement merely because one chart has higher ending equity.

To be considered a replacement candidate, it must:

- beat Protocol101 on strict one-account serial replay across the same comparable period,
- beat Protocol101 on protected historical blocks without tuning to them,
- stay positive under required slippage stress,
- preserve realistic account behavior,
- avoid relying on one trade/day/month,
- have an understood tradeoff profile if win rate is lower,
- pass attribution that explains where the extra PnL comes from,
- pass no-order runtime parity for the same full-action feature set.

If the model has higher PnL but meaningfully worse win rate, drawdown, churn, or timing sensitivity, it is a research challenger, not a replacement.

## Promotion Language Rules

Do not write:

```text
This challenger is better.
```

Write:

```text
This challenger has higher single-seed strict-serial PnL over Q3 2025 through recent 2026, but lower win rate and higher churn than Protocol101. It remains research-only until runtime parity and execution-risk checks pass.
```

Do not write:

```text
Q4 made $910k.
```

Write:

```text
In the five-seed attribution subset, Q4 trades with >=10 SPX-point favorable underlying movement contributed $910k before being offset by other Q4 trades. Single-seed Q4 account PnL was $158.8k.
```

## Chart Rules

`equity.html` and `trades.html` used for model review must be source-of-truth serial account views unless clearly labeled diagnostic.

Required chart labels:

- candidate name,
- baseline name,
- seed,
- date range,
- starting equity,
- whether train/validation rows are included,
- whether overlapping candidate rows are excluded,
- whether intratrade quote-path MFE/MAE backfill is available.

If a chart skips quote-path backfill, say so prominently.

## Failure Handling

If a challenger fails replacement criteria:

1. Do not soften the gate after seeing the result.
2. Do not promote it because one metric looks exciting.
3. Classify the failure mode.
4. Keep useful diagnostics.
5. Design the next experiment from the failure mode.

Examples:

- Lower win rate but higher PnL: audit expectancy, drawdown, churn, and fill sensitivity.
- Big directional winners but bad net PnL: study entry selectivity and lifecycle exits.
- Good historical replay but weak runtime parity: pause model replacement and fix train/live feature parity.
- Better five-seed median but worse single-seed chart: inspect seed variance and chart construction.

## Paid Data Rule

No broad paid historical data expansion is allowed just because a model is interesting.

Before paid downloads:

- state exactly what question the data answers,
- prove existing data cannot answer it,
- estimate cost without billable downloads,
- request explicit user approval,
- set a hard spend cap.

## Report Template

Every model-improvement report should include:

```text
What is this:
Does it change the paper-trading default:
Candidate being tested:
Baseline it must beat:
Data used:
Paid data downloaded:
Broker endpoint called:
Metric scope:
Training data:
Validation data:
Protected test data:
Primary result:
Win-rate / expectancy tradeoff:
Drawdown / worst-day tradeoff:
Churn / lifecycle behavior:
Attribution summary:
Runtime parity status:
Decision:
Next experiment:
```

## Current Standing Interpretation

The premium-leaning blended challenger is a useful research signal, not a paper-default replacement.

It showed higher historical total PnL than Protocol101 in the single-seed chart, but with much lower win rate and higher trade count. That is not automatically better. It means the next step is attribution and runtime parity, not promotion.
