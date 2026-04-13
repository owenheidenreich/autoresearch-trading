# v3 Project Overview

This is the plain-English description of the new `v3/` project.

If `v2/` was the exact-chain supervised research system, `v3/` is the new pure reinforcement-learning track.

## What v3 Is

`v3` is a research system for training an SPX 0DTE options agent that learns directly from replayed outcomes.

The goal is for one model to decide, bar by bar:

- whether to trade
- which exact contract to trade
- how much risk to allocate
- how to manage the position after entry

This is intentionally different from `v2`.

In `v2`, the model mostly learned to score contracts against historical labels while trade management stayed policy-driven. In `v3`, the model is meant to control the full trade expression inside a physically honest replay environment.

## What Changed From v2

The major design change is philosophical as much as technical.

`v2` asked:

- can we rank the historically best contract?
- can we beat fixed baselines under an exact-chain replay harness?

`v3` asks:

- can an agent learn a profitable behavior policy from consequences alone?
- can it choose different contracts, different risk, and different management styles in different regimes without hardcoded time windows or fixed trade rules?

That means `v3` removes the idea that the model should imitate an oracle contract picker and instead treats trading as a sequential decision problem.

## What The Agent Actually Sees

Each decision step is one intraday bar from a single SPX 0DTE trading day.

The observation includes:

- a trailing `90 x 47` one-minute context window from the trusted `v2` dataset
- a causal higher-timeframe 5-minute context stream built from raw intraday bars
- explicit session-state anchors like opening range, initial balance, session high/low distance, and time-of-day volume pressure
- the exact executable contract snapshot visible on that bar
- current account state
- current position state if a trade is open
- current drawdown and realized day PnL

So the model is not just looking at the last 30 minutes anymore. It sees fast tape, slower intraday structure, and explicit trade lifecycle memory at the same time.

## The New Market-State Cache

`v3` now builds a separate derived cache called `v3_market_state_v1`.

That cache does not replace `v2/data.pt`. It sits beside it and adds the new memory surfaces needed by the RL agent:

- 5-minute causal bucket features
- per-bar session-state anchors

This lets `v2` stay frozen and trustworthy while `v3` grows a richer observation contract.

## What The Agent Is Allowed To Do

When flat, the agent can:

- do nothing
- open a new position

When already in a trade, the agent can:

- hold
- close
- adjust the live trade

An open action includes:

- selecting the exact visible contract
- choosing a risk budget fraction
- choosing stop distance
- choosing target distance
- choosing time stop
- choosing an exit style

An adjust action includes:

- tightening or loosening stop/target/time settings
- scaling position size up or down on the same contract
- closing early if the model wants out

## What Is Still Hardcoded And What Is Not

The environment removes strategy rules, but it still keeps physical market rules.

Not hardcoded:

- no morning-only session window
- no cooldown
- no fixed stop
- no fixed target
- no fixed hold time
- no fixed quantity
- no daily loss cap

Still enforced:

- the model can only trade contracts visible in the exact historical snapshot
- entries and adjustments execute on the next bar
- commissions and slippage are charged
- quantity must be an integer
- only one live position can exist at a time
- while in a trade, resizing must stay on the same contract
- every day ends flat because this is 0DTE

So the idea is: no strategy handholding, but no cheating on execution either.

## How Sizing Works

The model does not emit raw quantity directly as its main sizing decision.

Instead it emits a `risk_budget_frac`, which is interpreted as a fraction of account equity the model is willing to risk. The environment then converts that into an executable integer quantity using:

- the chosen stop distance
- estimated worst-case loss to stop
- commissions
- size-aware slippage
- available cash

This keeps sizing dynamic while still grounded in something physically executable.

## How Training Works

Training uses PPO in PyTorch.

One episode is one trading day.

At a high level:

1. Sample training days from the dataset.
2. Let the current policy act through those days.
3. Record rewards, log-probs, values, and done flags.
4. Run PPO updates on that rollout data.
5. Periodically replay the policy deterministically on held-out days.
6. Save the best checkpoint, artifact manifest, replay traces, and evaluation metrics.

There is no teacher policy, no oracle warm start, and no imitation target in the training loop.

## How Reward Works

The main reward signal is dense account change over time.

The environment rewards or penalizes:

- marked-to-liquidation equity change
- additional drawdown
- unnecessary action churn
- unnecessary resizing
- final day PnL
- final day drawdown

This is meant to push the model toward good trading behavior, not just good entry labels.

## How Evaluation Works

Evaluation is deterministic replay on held-out days.

The key outputs are:

- account return
- drawdown
- daily return distribution
- Sortino-style risk-adjusted performance
- trade expectancy
- turnover
- exposure time
- confidence/reward calibration

The model is also compared against several new v3 baselines:

- `NoTrade`
- `ATM-Fixed`
- `SimpleRules-Fixed`
- `V2-Best-Static`

`v3/results.tsv` is the official log for this RL phase.

## What Is Already Implemented

The first full `v3` research surface now exists:

- exact-chain RL environment
- policy/value model
- PPO trainer
- deterministic replay
- artifact saving
- baseline evaluation
- pre-run gate
- GPU deployment wrapper
- tests for core environment and policy behavior

Operationally, the codebase is ready to start a real training run once the GPU session is booted.

## What Is Not Done Yet

This is still an early working research system, not a finished trading product.

Not done yet:

- no trained profitable RL policy yet
- no long experiment history yet
- no live broker execution
- no multi-position portfolio logic
- no market impact model beyond the current size-aware slippage approximation
- no guarantee yet that PPO is the final algorithm

So `v3` is real infrastructure, but it is still in the “prove the research loop works” stage.

## Files To Read First

If you want the shortest useful path through the new project, read these in order:

1. `v3/docs/project_overview.md` — this file
2. `v3/program.md` — the official operating protocol
3. `v3/docs/README.md` — doc index and launch flow
4. `v3/ops/status_report.py` — quick current-state summary

## Short Version

The new project is trying to build a single all-day SPX 0DTE RL agent that can dynamically choose:

- if there is a trade
- which contract expresses that trade best
- how much risk to take
- how to manage the trade after entry

It uses the existing exact-chain dataset and sidecars from `v2`, but the learning method and decision model are fundamentally different.
