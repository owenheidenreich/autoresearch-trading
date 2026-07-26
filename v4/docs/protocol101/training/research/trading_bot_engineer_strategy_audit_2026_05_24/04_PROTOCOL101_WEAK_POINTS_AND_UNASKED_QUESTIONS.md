# Protocol101 Weak Points And Unasked Questions

This is the most important document for the next research phase.

Protocol101 has earned default status, but it has not proven that its strategy is complete. The open work is to understand exactly where its trader beliefs are right, where they are too narrow, and where they are missing a second playbook.

## Weak Point 1: Timing And Execution

Protocol101 is timing-sensitive. Protocol114 and Protocol126 show delayed entry can destroy much of the replay edge.

Open questions:

- How much of Protocol101's PnL survives realistic quote age, order latency, and fill probability?
- Are the best trades also the most timing-sensitive?
- Are losing days associated with stale quotes, wide spreads, or fast quote movement?
- Does the ask-entry/bid-exit replay still overstate executable fills?

Needed diagnostic:

- Live no-order parity log for every decision time.
- Stratified fill observations by side, premium, spread, quote age, and time bucket.
- Replay stress by realistic latency distribution, not only fixed delay.

## Weak Point 2: It Is Entry-Owned, Not Lifecycle-Owned

Protocol101 learns entry/event selection. It inherits frozen exit behavior.

Open questions:

- Which Protocol101 exits are too early?
- Which holds are too long?
- Which `protocol054_fallback` exits are actually runner candidates?
- Which `hard_stop` losers showed early MFE but were not managed correctly?
- Should the bot have separate states such as "initial scalp paid," "runner allowed," "giveback guard," and "failed continuation"?

Needed diagnostic:

- Protocol101 runner opportunity audit from entry to close.
- Exit-reason-specific MFE/MAE/giveback analysis.
- Counterfactual exit ladders for current exit, delayed exit, trailing giveback, target, and forced-flat.

## Weak Point 3: It May Be Too Narrow

Protocol101 is mostly high-premium, slightly ITM, post-open/late-afternoon. It avoids much of the full surface.

Open questions:

- Is Protocol101 correctly avoiding low-premium OTM noise, or missing convex winners?
- Are there specific lower-premium regimes where challenger-style trades are valid?
- Does recent 2026 weakness indicate a regime where Protocol101's playbook is less effective?
- Are late-morning or midday continuation trades actually bad, or merely not represented in the Protocol101 belief set?

Needed diagnostic:

- Missed-winner and abstention audit over full candidate surfaces.
- Compare rejected candidates by time, premium, moneyness, spread, delta/gamma/theta, and realized path.
- Find "Protocol101-like but not selected" candidates versus truly different playbooks.

## Weak Point 4: Calls And Puts May Be Different Strategies

Protocol101 trades more calls than puts, but puts had higher average PnL in the seed-1 paper replay. Older diagnostics also suggest side-specific regime differences.

Open questions:

- Do calls and puts need separate playbooks?
- Are high-confidence puts sometimes exhaustion/crowding trades?
- Do puts need different premium, IV, spread, and time-of-day filters?
- Are call winners more trend-continuation and put winners more volatility/IV-regime dependent?

Needed diagnostic:

- Side-specific win/loss path taxonomy.
- Side by time bucket, premium, moneyness, IV proxy, spread, and exit reason.
- Side-specific missed-winner audit.

## Weak Point 5: It May Block Better Future Opportunities

The learned-defer work showed challengers can block Protocol101. The same question should be asked inside Protocol101.

Open questions:

- Which Protocol101 entries block better later Protocol101 entries?
- Are some low-edge early entries not worth spending the slot?
- Is there a "wait for a better second signal" pattern?
- Does the one-contract constraint hide cases where multi-contract or staged exits would help?

Needed diagnostic:

- Protocol101 blocked-opportunity audit.
- Compare taken trade PnL against best later same-session feasible Protocol101 candidate while the slot was occupied.
- Measure opportunity cost by entry score, premium, side, time, spread, and duration.

## Weak Point 6: The Edge Margin Is Thin In Places

Protocol101 beat strict serial baselines, but some margins were small:

- Q4 2025 baseline margin was only `$140`.
- Q3 2025 margin was only `$1,340`.
- Some seed-level margins were fragile.

Open questions:

- Is Protocol101's edge broad, or is it a narrow improvement over a strong upstream candidate stream?
- Which feature/history additions actually add value?
- Does Protocol101's score calibrate realized value, or mostly act as a gate?

Needed diagnostic:

- Score/threshold calibration.
- Ablation of short-history features.
- Compare Protocol101 selected rows against first-affordable, matched random, and same-scope strict baseline by day and regime.

## Weak Point 7: Strategy Identity Is Still Blurry

The user-level trading question has not been formalized enough.

Protocol101 sometimes behaves like a quick high-probability capture strategy. Some later experiments tried to capture fuller moves. These are different playbooks.

Open questions:

- Is the bot meant to scalp, capture directional continuation, or operate as a hybrid?
- If hybrid, what is the state transition from scalp to runner?
- What evidence tells us at entry that a trade deserves more room?
- What evidence after entry tells us to stop treating it as a scalp?
- Should the model optimize dollars, return on premium, drawdown-adjusted utility, MFE capture, or baseline-relative incremental utility?

Needed diagnostic:

- Winner path taxonomy: scalp, runner, failed runner, giveback winner, giveback loser, immediate failure.
- Define strategy labels before training.

