# Outside Engineer Brief

## What We Need Help With

We do not primarily need a bigger neural network. We need help finding the right trading questions.

Protocol101 is the current paper default because it is conservative, deployable, and passes strict serial replay. But it was built with trader-level beliefs embedded directly into the candidate selection and timing structure. Later ML systems that treated the problem as generic policy learning often found more backtest PnL but worse execution risk, drawdown, churn, lifecycle behavior, or validation credibility.

## Current Best Hypothesis

Protocol101 monetizes a narrow, high-quality 0DTE SPXW long-option pattern:

- high-premium, mostly ITM contracts,
- post-open morning and late-afternoon timing,
- short holding periods,
- strong A+/surface-edge candidate stream,
- quick capture of favorable movement,
- strict abstention when the event is not good enough.

It may be leaving money on the table in runners, missed convex winners, side-specific regimes, or low-premium playbooks. But past attempts to broaden the system usually made it lower quality.

## What We Want The Engineer To Challenge

Please challenge these assumptions:

1. Is Protocol101 really a scalp strategy, a directional continuation strategy, or a hybrid?
2. Are its exits rational, or inherited artifacts that accidentally work?
3. Should runner logic be learned only after the trade proves itself?
4. Are calls and puts separate strategies?
5. Is Protocol101's ITM/high-premium bias a strength or a blind spot?
6. Are low-premium/OTM challengers fundamentally worse, or only under-filtered?
7. Is the timing edge executable after realistic latency and fill probability?
8. Does Protocol101 block better later opportunities inside the same session?
9. Does recent 2026 weakness reveal regime drift?
10. What should the next model be forbidden to trade?

## What We Should Not Do Yet

- Do not train a new neural model immediately.
- Do not tune thresholds on Q3/Q4/Q1/March/recent.
- Do not promote any challenger that lacks live no-order parity and fill evidence.
- Do not buy broad historical data until the strategy questions and validation gates are clearer.
- Do not optimize raw PnL alone.

## Most Useful Near-Term Deliverable

The best next deliverable is a **Protocol101 strategy-forensics packet**:

- trade archetype taxonomy,
- losing-day autopsy,
- runner opportunity audit,
- missed-winner/abstention audit,
- Protocol101 slot-cost audit,
- call/put asymmetry audit,
- execution realism evidence plan,
- validation readiness plan.

After that, the next model should have a name that sounds like a trading hypothesis, for example:

- `PROTOCOL101_RUNNER_OVERLAY_FOR_CONFIRMED_MFE_V1`
- `PROTOCOL101_EARLY_WEAK_ENTRY_DEFER_V1`
- `PROTOCOL101_HARD_STOP_AVOIDANCE_GATE_V1`
- `LOW_PREMIUM_PUT_PLAYBOOK_FILTER_V1`

That is the standard the next experiment should meet.

