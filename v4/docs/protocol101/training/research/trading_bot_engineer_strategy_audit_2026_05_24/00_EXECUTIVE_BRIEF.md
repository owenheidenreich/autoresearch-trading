# Executive Brief

## Bottom Line

Protocol101 is still the operational paper default, but it should not be treated as the final strategy. It is best understood as the strongest deployable expression of a specific trader belief set:

- Trade SPXW PM 0DTE long options only.
- Wait by default.
- Enter only when a curated A+/surface-edge candidate is strong enough.
- Prefer post-open morning and late-afternoon windows.
- Prefer near-ATM or slightly ITM, premium-rich contracts.
- Use one account, one open position, one contract.
- Exit through frozen inherited lifecycle logic rather than a fully learned hold/exit policy.

The important correction from this audit is that Protocol101 is not winning because generic ML is insufficiently clever. Protocol101 is winning because it embeds trader-level constraints that many later challengers violated.

## Why Protocol101 Is The Default

Protocol101 became default because it combined four things most challengers did not combine:

1. **A real trading hypothesis.** It is a conservative event-selection model over a curated candidate stream, not an unconstrained rowwise return maximizer.
2. **Strict serial replay.** It respects one account, one contract, one open position, ask-entry, bid-exit, affordability, and flat-by-close semantics.
3. **Deployable artifacts and runtime bridge.** It has frozen model/scaler manifests, live entry plumbing, risk gates, and operational docs.
4. **Better trade quality.** Compared with the strongest broad challenger audit, Protocol101 had lower raw PnL but much better win rate, profit factor, drawdown, and churn profile.

Protocol101 is not unbeatable. Protocol194, Protocol215, Protocol240, Protocol260, and Protocol265-style research lines all found stronger exposed-split PnL. They stayed research-only because their evidence failed on timing, quality, runtime parity, validation cleanliness, lifecycle stability, or promotion readiness.

## What The Project Has Been Testing

The project has tested several distinct ideas:

- Is the A+ event stream profitable under serial one-account replay?
- Can causal event-history features improve entry selection?
- Can broader full-action candidate surfaces find more opportunity than Protocol101?
- Can premium/capital-efficiency objectives capture more convexity?
- Can lifecycle models hold winners longer without overholding losers?
- Can routers combine Protocol101-like safety with challenger upside?
- Can unified wait/enter/hold/exit labels solve the whole serial game?
- Can a learned-defer overlay price the cost of blocking Protocol101?

The repeated answer is: broader systems often find more gross opportunity, but they also introduce more churn, lower hit rate, worse drawdown, timing fragility, or slot-cost damage.

## Why The Current Question Is Strategy, Not Architecture

The next useful model will probably not come from "try a bigger MLP" or "train longer." The next useful model needs a sharper trading question.

Examples:

- Are we trying to scalp high-probability premium-rich ITM contracts?
- Are we trying to capture fuller directional 0DTE moves?
- Should the bot classify a trade as scalp vs runner after MFE appears?
- Are calls and puts different strategies?
- Are Protocol101 exits leaving runner value on the table, or are they correctly avoiding giveback?
- Which Protocol101 entries block better later opportunities?
- Is the apparent edge executable after latency and fill realism?

Until these questions are answered, new model experiments risk optimizing the wrong objective.

## Immediate Recommendation

Freeze new neural/model experiments. Do a Protocol101-centered diagnostic campaign:

1. Protocol101 trade atlas by side, time, premium, moneyness, exit reason, MFE, MAE, duration, and quote freshness.
2. Losing-day and hard-stop autopsies.
3. Missed-winner and abstention audit.
4. Runner/continuation opportunity audit on Protocol101 entries.
5. Protocol101 slot-opportunity audit.
6. Timing/fill/live-no-order parity audit.
7. Formal validation and untouched-holdout discipline.

Only after this should the next model be specified. The next model should be a named strategy hypothesis, not a generic challenger.

