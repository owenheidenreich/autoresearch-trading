# The opportunities exist; the clock minute is not the signal

**Finding, 2026-08-14. Model-free. No policy or threshold was fitted.**

## What changed for the bot

The repository can now replay the problem the owner actually described: each day is a causal episode, the
09:35 state uses the completed 09:34 ES bar and the option snapshot stamped 09:35, the policy sees every
live two-sided strike, and every affordable near-OTM call/put action is followed for 60/90/120 minutes to
strike crossing and 10/20/30-point ITM depth. The $10,000 simulator enforces one position, exact contract choice,
ask-in/bid-out fills, origin-regime exit ownership, 1/2/3 trade caps and deterministic terminal handling.

The opportunity the owner sees is real and sparse. Across all 698,231 eligible contract-minutes, an OTM
contract reached 30 points ITM **1.57% within 60 minutes, 2.93% within 90, and 4.21% within 120**. But the
exact 10:00 and 13:30 minutes do not contain more of those paths than their immediate neighbourhoods after
the declared 24-cell correction. The useful formulation remains conditional: read the structure arriving
near those areas; never buy because the clock names a minute.

No model comparison was run. Current `STATUS.md` and the later do-not-retest closure independently prohibit
another long-side selective fit on this quote corpus, and the neural registry separately requires 1,140
independent sessions against 243 complete sessions here. Owner fit authorization satisfies the action tier;
it does not silently rewrite those evidence rules.

## Coverage: what “every day” honestly means

The owned quote root contains 251 base sessions and 251 `_official_context` companions, 47,707,186 rows in
all. The corrected coverage audit admits **243 complete primary episodes** from 2025-08-01 through
2026-07-30:

- four quote days have an empty matching ES file: 2025-09-19, 2025-12-19, 2026-03-20 and 2026-06-18;
- two full-day quote clocks contain material gaps: 2025-10-22 and 2026-07-31; and
- 2025-11-28 and 2025-12-24 are early-close/partial-clock sessions, not silently treated as normal 16:00
  days.

Thirty included sessions have no affordable OTM contract at 09:35. They remain episodes and can become
no-trade days; the original audit incorrectly excluded them and is preserved as attempt 001. Attempt 002
corrects that post-entry inclusion error.

At 10:00 the median eligible action set has **8 contracts**; at 13:30 it has **10**. Vendor IV and greeks
are absent from the historical files, so the dataset recomputes IV/delta/gamma/theta/vega from the current
mid, underlying snapshot, strike and expiry. Bid/ask size and underlying snapshot coverage are 100%.
Historical `quote_age_ms` is zero everywhere and is explicitly not read as measured arrival latency.

Generated causal substrate:

| Artifact | Rows |
|---|---:|
| ES candles with knowable-at boundary | 94,770 |
| causal minute states | 93,798 |
| compact near-band label-table rows | 1,880,427 |
| affordable OTM candidate contracts | 698,231 |
| all-minute atlas rows | 79,218 |
| every-day 10:00/13:30 band rows | 10,206 |

## The OTM-to-ITM opportunity surface

Candidate-level prevalence, including every failure:

| Maximum horizon | crosses strike | reaches 10 ITM | reaches 20 ITM | reaches 30 ITM |
|---:|---:|---:|---:|---:|
| 60m | 33.18% | 11.33% | 4.01% | **1.57%** |
| 90m | 39.08% | 16.08% | 6.75% | **2.93%** |
| 120m | 42.92% | 19.60% | 9.02% | **4.21%** |

These are labels and ceilings, not precision. At one decision minute there may be several calls and puts;
a causal policy must select one before the future path is known and must pay for all false starts.

At the session-minute level, asking whether **any** eligible contract eventually reached the depth:

| Minute | 30 ITM by 60m | local-band rate | named − local, corrected CI | 30 ITM by 120m | local-band rate | named − local, corrected CI |
|---|---:|---:|---:|---:|---:|---:|
| 10:00 | 9.88% | 8.99% | +0.88pp [−2.08,+4.34] | 19.75% | 20.35% | −0.60pp [−4.51,+3.74] |
| 13:30 | 5.76% | 5.10% | +0.66pp [−2.09,+3.61] | 16.87% | 16.28% | +0.60pp [−2.84,+4.14] |

Every interval contains zero. The same is true for the declared crossing/10/20-point cells. So **10:00
and 13:30 are useful places to display and diagnose structure, not unconditional entry edges**.

## The surprising 10:00 clock diagnostic, audited

Buying every eligible contract is impossible under one-position occupancy. Its average is nevertheless a
useful instrument check and equals the expectation of selecting one eligible contract uniformly at random.
At 10:00, including no-contract sessions as zero, that session-equal expectation is:

| Hold | mean bid P&L/session | family-corrected interval |
|---:|---:|---:|
| 60m | +$29.94 | [−$51.05,+$122.67] |
| 90m | +$55.52 | [−$55.53,+$187.66] |
| 120m | +$75.70 | [−$43.04,+$218.92] |

At 13:30 it is −$34.77, −$27.52 and −$24.23 respectively after validated terminal cash accounting.
The old +$3.43 120-minute point estimate omitted 101 contracts that had no later bid; it was an
executable-bid-only diagnostic, not the complete economic population. **All six corrected intervals
contain zero.** Nothing clears as an unconditional clock policy.

The positive 10:00 point estimates passed the mechanical bug audit but remain high-variance and one-year:

- no 10:00 candidate lacks its 60/90/120-minute executable bid, so survivorship cannot create the sign;
- net P&L reproduces `exit bid × 100 − entry ask dollars − $3.08` exactly;
- mid-to-mid is $17.54 above bid economics at 60 minutes, the charged spread rather than a hidden fill;
- the cheapest `$0–400` bucket is negative, so percentage-return/cheap-contract selection is not the
  mechanism;
- puts carry the point estimate (+$88.53 at 60 minutes) while calls are negative (−$33.76), so this is a
  side/regime concentration, not a general contract effect;
- no contract was selected using its outcome, and hindsight best-contract values remain labelled oracles;
  and
- future-candle and future-quote mutation tests leave the earlier state and trade unchanged.

The median 10:00 contract still loses $153 at 60 minutes; only 35.5% are positive. A few large put winners
create the positive mean, which is exactly the sparse-payoff problem a causal entry would have to solve.

## Simulator and terminal accounting

The simulator has sixteen known-answer tests and an 18-cell real-data no-trade smoke across early/middle/late
sessions, both risk readings and all 1/2/3 trade caps. It routes:

- flat + morning → morning entry;
- morning-origin holding → morning exit at every later clock;
- flat + afternoon → afternoon entry; and
- afternoon-origin holding → afternoon exit.

A marketable exit with no bid waits for the first later executable bid and never takes the later best.
The policy receives the whole current live chain while a separate action mask permits only affordable
`[-25,0)` OTM entries. A held contract remains separately visible if its quote becomes one-sided. Stateful
policies must reset at every episode.

Terminal accounting is now resolved from owned data. Across all 243 included sessions, the separately
owned official SPX minute source is marked official, non-derived and non-proxy; its final 16:00 timestamp
equals the normalized same-day SPXW PM settlement timestamp, and its close exactly equals the aligned
terminal underlying on every session. When no executable bid remains, terminal intrinsic value is therefore
recorded as **`validated_cash_settlement`**, never as a bid fill, with zero-recovery retained as a paired
sensitivity. This resolves **16,485 / 71,136 / 124,002** formerly blocked 60/90/120-minute candidate-clock
rows without dropping one. It is a validated accounting identity, not a claim that a bid was executable or
a separately downloaded OCC exercise statement.

## Architecture answer and implementation boundary

The full unfitted architecture family now implements one common forward interface and exact role routing:

1. shallow time-aware joint policy: 676 trainable parameters;
2. shallow shared four-head policy: 720;
3. causal sequence joint policy: 1,252;
4. causal sequence shared four-head policy: 1,296; and
5. four independent sequence specialists: 4,920, conditional on prior shared-head specialization.

The first interface attempt was too narrow despite passing its tests: it retained only 120 candles and the
±25-point chain. Declaration v2 corrects that before any fit. The deterministic tensor bridge now exposes
the whole completed session prefix (up to 390 candles), the exact whole current live chain, a separate
entry-action mask, causal account/position state and clock encoding. On 12 real first/middle/last-session
cells it preserves 229–365 live contracts and matches the source count exactly; 209–345 deep strikes remain
context. Masked-future mutation tests leave current outputs unchanged.
The atlas still does not decide one versus four: it shows why time must be state and the four roles must be
routed. No weights were fit and no economic result was produced. The 1,296-parameter shared neural design
would require 25,920 independent sessions under the frozen 20-sessions-per-parameter rule; only 243 exist.
A valid comparison also needs explicit reopening of G1 and the later do-not-retest closure.

Replay and reporting plumbing is ready for that eventual comparison. It captures every considered contract,
probabilities, actions, reasons, fills, spreads, fees, account state and trade path. Four examples selected by
a structural first/middle/last/no-candidate rule prove export and accounting only; their fixed 09:35/60-minute
policy was not fitted and is **not model-skill evidence**. The reporting contract covers abstention, trade
frequency, premium at risk, capital utilization, spread, fees, holding time, entry OTM depth, maximum/final
ITM depth, conversion and time-to-cross, option and underlying MFE/MAE, win/payoff statistics, drawdown,
family-corrected session intervals and $10,000-account return. Planned loss remains explicitly unknown unless
a future policy declares it.

## Evidence

- coverage attempt 002: `v4/audit/autoresearch/causal_day_trader_coverage_2026_08_14_attempt002/receipt.json`
- terminal settlement audit: `v4/audit/autoresearch/causal_day_terminal_settlement_2026_08_14/receipt.json`
- settlement-complete causal dataset: `v4/audit/autoresearch/causal_day_dataset_settlement_validated_2026_08_14/receipt.json`
- settlement-complete every-day atlas: `v4/audit/autoresearch/causal_day_atlas_settlement_validated_2026_08_14/receipt.json`
- settlement-complete economic audit: `v4/audit/autoresearch/causal_day_atlas_profit_settlement_validated_2026_08_14/receipt.json`
- whole-day/whole-chain observation audit: `v4/audit/autoresearch/causal_day_observation_contract_2026_08_14/receipt.json`
- deterministic simulator, attempt 005: `v4/audit/autoresearch/causal_day_simulator_2026_08_14_attempt005/receipt.json`
- unfitted architecture/tensor interfaces, attempt 003: `v4/audit/autoresearch/causal_day_architecture_interfaces_2026_08_14_attempt003/receipt.json`
- replay/report plumbing, attempt 005: `v4/audit/autoresearch/causal_day_replay_plumbing_2026_08_14_attempt005/receipt.json`
- machine-enforced model block, attempt 003: `v4/audit/autoresearch/causal_day_model_block_2026_08_14_attempt003/receipt.json`
- requirement completion audit, attempt 002: `v4/audit/autoresearch/causal_day_completion_audit_2026_08_14_attempt002/receipt.json`
- corrected fixed family: [`work/entry-exit-attribution/DECLARATION_V2.json`](../../work/entry-exit-attribution/DECLARATION_V2.json),
  SHA-256 `811cb21d2c5bab3c2f18b58833d7e9043ed1621d0c1baf722ab8bf199ca516b4`; declaration v1 remains preserved

Nothing here contacted a vendor, broker or runtime; fit a model; tuned a threshold; used post-cutoff data;
or authorized promotion.
