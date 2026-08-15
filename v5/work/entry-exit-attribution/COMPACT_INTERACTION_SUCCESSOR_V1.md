# Compact interaction successor V1 — design and single authorization packet

**Status: architecture, full action-value labels, identical real/null fit path and complete economic
evaluation are implemented, tested and machine-declared; fitting is refused by the current signed gate.**

## Decision this packet resolves

The highest-information next experiment is not another magnitude cutoff. It is one compact test of the
actual trading decision:

> At this causal minute, is buying this exact contract worth more than preserving the only trade slot and
> waiting for a later setup?

If this fails the same chronological and executable gates, close the current long-call/long-put selector
branch and prepare the strategic different-game packet. Do not retry a nearby target, horizon, width,
seed, threshold or specialist split.

## Frozen policy design

- architecture name: `compact_interaction_entry`
- implementation: `v5/research/causal_day_compact_interaction.py`
- computed trainable parameters: **48**
- conservative permitted design range: **29--50**
- entry roles: shared model with the complete five-field causal clock, including the morning/afternoon
  boundary; no independent morning or afternoon network
- actions at a flat minute: `[WAIT, every currently eligible affordable contract]`
- trade cap for this decisive entry experiment: **one per session**
- position cap: one contract, one position
- automatic liquidation: 120 minutes, session close, or first later executable bid after the request,
  using the existing terminal-settlement law
- account: $10,000, with current-minute affordability

The one-trade cap is deliberate. It makes the entry target exact without inventing a continuous cash-state
approximation for later trades. One clean trade is within the owner's “few trades per day” thesis; a second
trade is a follow-up only after a stable entry edge exists.

### Structural no-trade floor

`Q(wait)` includes the option to make no trade and earn $0. A regression estimate can nevertheless fall
below zero. Inference therefore compares each contract against `max(0, predicted Q(wait))`; a negative
WAIT error may never force a knowingly negative contract. This is the target's feasibility boundary, not
an operating-point threshold, and it adds no parameter. The policy takes the **first** current contract
whose prediction strictly beats feasible WAIT, with contract identity breaking an exact score tie.

## Forty-eight parameters, computed rather than transcribed

The state is eight fields from the last completed candle plus all five clock fields:

- candle body and upper/lower wicks;
- close from session open, running high and running low;
- range position and one-minute return; and
- elapsed/remaining time, morning/afternoon indicator and cyclical clock.

The contract base uses contemporaneous ask, spread, IV, gamma and theta. Two products provide the missing
conditionality:

`contract_score = base(contract) + direction(state) * is_call + depth(state) * moneyness`

WAIT has its own state value. Three 13-to-1 state maps contribute 42 parameters and one 5-to-1 contract
base contributes six, for **48 built parameters**. Tests prove that changing candle state can reverse the
call/put argmax and that the WAIT head receives gradient.

## Frozen proposed learning target

Label name: `serial_action_advantage_120m`.

For every eligible contract `i` at causal minute `t`, derive one executable terminal reward `R(t,i)` under
the frozen automatic liquidation law:

- buy at the ask;
- sell at the first eligible bid at or after 120 minutes, or use the existing close/settlement convention;
- subtract the measured round-trip fee; and
- do not use best price, MFE, ITM depth or a post-entry exit choice.

With one trade available for the session:

`Q_enter(t,i) = R(t,i)`

`Q_wait(t) = max(0, max Q_enter(s,j) for every later eligible minute s and contract j)`

The model jointly regresses the contract `Q_enter` values and minute-level `Q_wait` value using a frozen
robust loss and fold-fitted scaling. At inference it chooses the largest action logit. There is no score
cutoff, top-N selector, current-day rank or forced 10:00/13:30 trade.

This target uses future outcomes only as supervised labels on training sessions. All scored actions see
causal features and frozen earlier-fold parameters only.

The V2 target build is complete on all 243 sessions: 79,218 explicit decision minutes and all 698,231
candidate actions. It preserves 1,964 WAIT-only minutes that have no eligible contract. The median
executable Q(enter) is -$173, median Q(wait) is +$1,437, and only 0.72% of contract actions beat waiting.
Those frozen diagnostics make the eventual fit intentionally difficult; they did not change its law.

## Frozen evaluation law

- five chronological OOF folds and the existing embargo/timestamp law;
- real model and session-surface shuffled-label null fitted identically;
- outcome-blind control matched on session opportunity, regime, minute, side, delta, premium and trade
  count;
- mid-to-mid gross reported first as the decisive friction-free existence check;
- only if positive: ask-to-bid plus measured fees and exact $10,000 replay;
- absolute, matched and shuffled deltas positive in at least four of five folds;
- the complete declared family retained in corrected confidence; and
- all seven existing kill conditions remain binding.

The complete evaluator is frozen in
[`ACTION_VALUE_EVALUATION_DECLARATION_V1.json`](ACTION_VALUE_EVALUATION_DECLARATION_V1.json), self-hash
`33ef2e0d96d73370ab529f91fd3b3fb47ad7f9a9208fc4a12920c5c3e0d08bef`:

- primary unit is all 150 scored sessions, with an abstention worth zero;
- executable outcomes and controls remain unread unless gross midpoint P&L is first positive;
- the outcome-blind comparator matches exact session, opening regime and side, then nearest minute,
  absolute delta and premium, preserving trade count;
- the shuffled surface is fit and walked under the identical law;
- the 648-member Job 39 family is retained and this one successor raises it to **649**;
- real executable net, real-minus-matched and real-minus-shuffled must each have a corrected lower bound
  above zero and at least four of five positive chronological folds; and
- because this fixed 120-minute entry experiment has no proven early stop, selected premium plus fee and
  the worst realised trade must each remain inside the signed **$500 daily breaker**.

If mid-to-mid gross is non-positive, or chronology is below four of five, the result is negative and the
long selector branch closes. No operating-point search follows because action selection is intrinsic to
the learned action values.

## Why this precedes more data or a different game

V5's failure is first architectural: its chart state could not change contract ordering and WAIT was not
trained. The 48-parameter successor fits the conservative evidence budget, so buying more history before
testing the correct small hypothesis is not justified. V5's pooled point estimate also means the current
long game deserves exactly one decision-valid action test before a strategic pivot—not an open-ended
model search.

## Exact gate result

The current gate correctly refuses the design:

- `compact_interaction_entry` is not in the hashed architecture family; and
- `serial_action_advantage_120m` is outside the signed `itm_depth_magnitude` label reopening.

No gate, status, knob, statistics file or ledger was edited. The single owner-controlled decision is
whether to issue a narrow signed re-ruling adding this one 48-parameter architecture and this one target,
with the unchanged corpus, folds, controls and seven kill conditions. Authorization would not promote,
paper-trade or live-trade anything.

Both the trainer and evaluator have been invoked under the current gate. Each refused on
`compact_interaction_entry` before a target session, optimizer, fit receipt, prediction or economic output
was opened or created. The four declared output/evidence paths remain absent.

## Exact scope of the one owner re-ruling

Permit only this tuple:

- architecture `compact_interaction_entry`, computed count **48**;
- label `serial_action_advantage_120m`, horizon **120**;
- corpus `causal_day_quote_243`;
- fit declaration self-hash
  `04b29a248289bc8a35433c1d734ade09a91ffe82c871a39407fca7423afd3e49`;
- evaluation declaration self-hash
  `33ef2e0d96d73370ab529f91fd3b3fb47ad7f9a9208fc4a12920c5c3e0d08bef`; and
- all seven existing kill conditions plus executable-net and $10,000-risk completion requirements.

Every other architecture, target, horizon, corpus and neighboring retry remains refused. This is the last
safe in-scope experiment, not a general reopening of option fitting.

## Why no other owned route supersedes this decision

A final inventory checked every alternative the ledger still calls genuinely new:

- **Sub-minute:** 189 pre-cutoff sessions of `exit_features` exist, but the registered source is Tier-S
  probe/skeleton evidence, not trusted promotion evidence, and it consists of held-position exit
  trajectories. It cannot establish which contract to enter; the ledger correctly routes it only after a
  skilled entry exists.
- **Economic-event context:** no owned FOMC/CPI/NFP or general macro-event corpus is registered. The only
  calendar artifact is an exchange-session calendar. Building the missing context requires external data
  or owner-approved acquisition.
- **Longer tenor:** the project owns same-day SPXW for this study, not a declared 1DTE/weekly causal
  ladder. This needs new data and a charter change.
- **Different underlying:** the frozen ES direction family is underpowered on 243--247 sessions; its
  known-answer floor is 8--16 points/session. The ledger estimates roughly 988 sessions even to halve the
  best floor to four points. Rerunning it locally is explicitly closed.
- **Defined-risk short premium:** the one predeclared $10,000-compatible translation just failed at
  -$60.57/session with 0/243 winners. Its declaration forbids a nearby width/time/horizon retry.

Accordingly, the narrow compact fit re-ruling is not merely the first item in a queue. It is the only
remaining action that can change the project using the current owned evidence without spending, external
contact, reserved data, a charter expansion or a prohibited neighboring search.
