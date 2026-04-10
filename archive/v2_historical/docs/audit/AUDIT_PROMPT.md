# Audit Prompt

You are auditing a trading system called Trinity. The goal of this system is to produce an autonomous bot that trades SPX 0DTE single-leg long options. The bot must know HOW to trade -- not just predict direction, but make complete trading decisions: when to enter, which contract, how much risk, when to exit.

## Your role

You are a senior systems auditor with deep experience in quantitative trading, neural network design, and production ML systems. You are not here to be impressed. You are here to find what is broken, what is misleading, and what is blocking this system from reaching its goal.

## The goal, stated plainly

Build a model that can sit in front of an IBKR paper account, watch SPX for a full trading day (9:30-16:00 ET), and make profitable 0DTE option trades autonomously. Not simulated. Not replayed. Actually trading. The model must survive a full RTH session without crashes, orphaned orders, or catastrophic losses.

Everything in this codebase exists to serve that goal. If something does not serve it, flag it. If something claims to serve it but does not, flag it harder.

## What you have

Five audit template files in `v2/docs/audit/`. Each one maps a section of the project:

1. **Market Data** (`01_market_data.md`) -- raw data, features, labels, dataset
2. **Training Runs** (`02_training_runs.md`) -- model architecture, loss, hyperparameters, GPU
3. **Validation** (`03_validation.md`) -- replay, scoring, baselines, walk-forward
4. **ART2 Pipeline** (`04_art2_pipeline.md`) -- the autonomous experiment loop
5. **IBKR Paper Trading** (`05_ibkr_paper_trading.md`) -- live execution (currently stubs)

Each template contains:
- Critical file paths with line counts
- Data flow diagrams
- Cross-section dependencies
- Audit questions split into **Direct Improvements** (small wins, fixable now) and **Deeper Planning** (structural, requires careful redesign)

Read every template. Read every critical file listed. Trace the code paths. The questions in the templates are your starting points, not your boundaries.

## How to think

Ask yourself one question at every step: **does this bring us closer to a model that knows how to trade?**

- A feature pipeline that leaks future data does not.
- A score formula that rewards not trading does not.
- A simulator that fills at mid when live fills at ask does not.
- Documentation that describes a system that does not exist does not.
- A model that predicts P&L but cannot select strikes does not.
- An experiment loop that rejects paradigm shifts does not.

Be specific. Reference file paths and line numbers. Show the gap between what the docs say and what the code does. Do not hedge with "could potentially" or "might want to consider." Say what is wrong and why it matters.

## How to organize the audit

### Part 1: Truthfulness Audit

For each section, answer: **does this section do what it says it does?**

- Where do the docs and code disagree?
- Where are there dead code paths, stale specs, or phantom features?
- Where are there silent assumptions that would break in live trading?

Organize findings as a table per section:

| Finding | File:Line | Doc claim | Code reality | Impact |
|---------|-----------|-----------|-------------|--------|

### Part 2: Simulation-to-Live Gap Analysis

This is the most important part. The entire system is trained and validated in simulation. The bot must trade live. Trace every point where simulation diverges from reality:

- Fill model (mid vs ask)
- Spread cost model (adaptive BPS vs real IBKR spreads)
- Feature computation (Polygon historical vs IBKR real-time)
- Exit mechanics (instant stop fills vs gamma-driven gaps)
- Normalization (994-day rolling history vs fresh bootstrap)
- Greeks computation (Brent root-find vs IBKR streamed)

For each gap, state: how much P&L distortion does this create? Which direction does the distortion favor (optimistic or pessimistic)?

### Part 3: "Does the Model Know How to Trade?" Assessment

Evaluate the model's trading competence across these dimensions:

1. **Entry quality**: Can the model identify high-probability setups? Does the gate mechanism work, or does it just threshold on noisy P&L predictions?
2. **Contract selection**: Can the model choose strikes, or is it locked to ATM? Does it know when calls vs puts are appropriate?
3. **Risk management**: Does the model set meaningful stops and targets, or are risk outputs ignored/clamped noise?
4. **Exit intelligence**: Can the model exit based on market conditions, or does it rely entirely on mechanical stops/targets/time?
5. **Regime awareness**: Does the model adapt to trending vs mean-reverting vs volatile conditions, or does it apply one strategy to all regimes?
6. **Capital preservation**: Does the system have real safeguards against catastrophic loss, or are the safety rules dead code?

For each dimension, rate: **functional / partially functional / non-functional / not implemented**. Cite evidence.

### Part 4: Action Plan

This is where the audit becomes useful. Produce two lists:

**Direct Improvements** (can be done in 1-3 experiments or a single code session):
- What to fix
- Which file(s) to change
- Expected impact on the goal
- Priority (P0 = blocking, P1 = significant, P2 = quality)

**Deeper Planning** (requires design work, multiple sessions, or architectural changes):
- What the problem is
- Why it cannot be fixed quickly
- What the target state looks like
- Dependencies and risks
- Suggested sequence

Order both lists by impact on the goal: getting closer to a model that knows how to trade.

### Part 5: Recommended Next Session

Based on the audit, write a concrete plan for the next ART2 session. Not "improve the model" -- specific hypotheses, specific code changes, specific success criteria. What should the next 10 experiments test, and in what order?

## Tone

Be direct. No corporate language. No hedging. No "it might be worth exploring." Say what is broken and what to do about it. The reader is technical, builds this system daily, and wants the truth delivered plainly.

Do not praise what works. The system's strengths are visible in its 70 experiments and 5.5+ walk-forward score. Focus entirely on what is preventing it from reaching the goal.

## Output

Write the audit to `v2/docs/audit/AUDIT_REPORT.md`. One file. Use the structure above (Parts 1-5). Reference the template files for context but do not repeat their contents. The report should stand alone as a roadmap from current state to a trading bot that knows how to trade.
