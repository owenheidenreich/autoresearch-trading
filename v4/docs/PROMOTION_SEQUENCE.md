# Protocol101 Promotion Sequence

This document preserves the current promotion logic for the SPXW 0DTE trading bot. It exists so we do not accidentally treat a promising historical sizing result as paper-trading or live-trading approval.

For the broader process that every model must follow before reaching promotion, see [HYPOTHESIS_TO_PROMOTION_PROCESS.md](HYPOTHESIS_TO_PROMOTION_PROCESS.md).

## Current Rule

The operational paper bot starts in one-contract mode.

The account-aware multi-contract sizer is a research candidate only. It may be replayed offline against paper/live logs, but it must not submit multi-contract paper orders until it clears the promotion gates below.

Paper money is not real capital, but paper fills are real evidence. Bad evidence created by uncontrolled sizing can mislead the research process, so paper execution must stay simple until timing, fills, and account behavior are proven.

## Why Multi-Contract Paper Orders Are Disabled For Now

Protocol154 concluded:

```text
Decision: blocked_multi_contract_promotion_missing_timing_evidence
```

The sizing overlay was not rejected. It passed the economic replay gates and live-stack compatibility checks. It is blocked because high-resolution timing evidence is incomplete.

Until that timing blocker is cleared, the paper order guard rejects order quantities greater than `1`. This isolates the first live question:

```text
Does the frozen Protocol101 stack produce realistic one-contract paper entries and exits at all?
```

If we allow 2-3 contracts before answering that, losses or gains become harder to interpret. A bad result could be caused by the model, IBKR fills, stale quotes, latency, overleveraging, or ordinary variance. One-contract paper mode keeps the evidence clean.

## Promotion Ladder

### Gate 0: Research Candidate

Purpose: prove the idea is worth studying.

Requirements:

- Historical replay uses executable ask-entry and bid-exit prices.
- Starting paper cash is `$10,000`.
- No real-money trading.
- No paid data expansion unless explicitly approved.
- Multi-contract sizing beats the one-contract baseline in frozen replay.
- Drawdown, daily loss, concentration, and slippage-stress checks do not materially worsen.

Current status:

```text
account_aware_sizer_v1: passed research economics, blocked for promotion by timing evidence.
```

### Gate 1: One-Contract Paper Operation

Purpose: prove the bot can run as a real paper trader without confusing the evidence.

Operational rules:

- Max order quantity: `1`.
- Max concurrent positions: `1`.
- Paper account starts at `$10,000`.
- The `$500` real IBKR reserve is not trading capital.
- All entries and exits must be logged.
- Real-money trading is forbidden.
- Multi-contract execution is disabled.

Evidence captured by Protocol155:

- model score and threshold
- selected contract
- bid, ask, bid size, ask size
- option quote age
- SPX/VIX context age
- intended entry and exit timestamps
- actual paper order/fill timestamps and prices
- decision-to-submit and decision-to-fill latency
- entry slippage versus decision ask
- skipped or blocked reasons
- account cash, equity, daily PnL, open positions

Promotion requirement:

```text
Protocol155 must show clean one-contract paper behavior with closed entry/exit fills.
```

### Gate 2: Live Timing Evidence

Purpose: prove historical timing assumptions survive in live/paper conditions.

Protocol155 must show:

- quote age within budget
- SPX/VIX context age within budget
- decision-to-fill latency inside budget
- entry slippage inside budget
- closed trades available for replay
- no quantity greater than `1`
- no real-money rows
- no unexplained broker endpoint rows

Delay-stress rows should be available for:

```text
1s, 5s, 15s, 30s
```

Promotion requirement:

```text
The one-contract paper path must produce enough clean timing and fill evidence to replay the same decisions under delay stress.
```

### Gate 3: Offline Multi-Contract Replay On Live Logs

Purpose: test account-aware sizing without allowing it to affect live/paper execution.

The same paper session logs are replayed through:

- one-contract baseline
- `account_aware_sizer_v1`
- delay-stressed variants

The multi-contract sizer may advance only if it remains better than one contract on the same live/paper evidence.

Required checks:

- incremental PnL positive versus one contract
- max drawdown not materially worse
- worst day not materially worse
- no loss clustering problem
- no daily loss-stop pathology
- no unaffordable trades
- no overlap
- no stale-data dependency
- no timing fragility under critical delays

### Gate 4: Controlled Multi-Contract Paper Trial

Purpose: test real paper execution with limited sizing after offline replay earns it.

This gate requires a new explicit promotion decision.

Default first trial:

```text
max_order_quantity = 2
paper only
real money disabled
same Protocol155 logging
same daily loss and drawdown checks
```

Do not jump directly from one-contract paper to unrestricted 3-contract or 20-contract sizing.

### Gate 5: Expanded Paper Sizing

Purpose: scale only after controlled multi-contract paper evidence is clean.

Possible future steps:

- raise cap from `2` to `3`
- test account-size tiers
- replay with stricter drawdown and daily stop rules
- require repeated clean sessions before each increase

Any increase must be treated as a new promotion decision, not an automatic consequence of paper-account growth.

### Gate 6: Real-Money Consideration

Purpose: decide whether the bot can risk real capital.

This is separate from paper approval.

Real-money trading requires:

- explicit user approval
- separate runtime flag
- separate promotion packet
- proven paper execution parity
- stable logs across multiple sessions
- demonstrated risk controls
- strict capital limits

Nothing in Protocol155, Protocol154, or the current multi-contract research work approves real-money trading.

## Current Standing Decision

```text
Operational/paper default: one contract only.
Multi-contract status: strong research candidate, not paper-execution approved.
Next evidence step: run one-contract paper sessions and let Protocol155 produce timing/fill reports.
```

## Do Not Forget

The model is not being punished by one-contract mode. One-contract mode is how we protect the research signal.

Once live/paper evidence proves that fills, quote freshness, and timing are real, multi-contract sizing can be promoted deliberately.
