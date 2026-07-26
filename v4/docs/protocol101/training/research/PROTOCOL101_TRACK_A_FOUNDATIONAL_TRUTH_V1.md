# PROTOCOL101_TRACK_A_FOUNDATIONAL_TRUTH_V1

What is this: Track A stopping-point synthesis for building a better 0DTE trading bot
Does it change the paper-trading default: no
Paid data downloaded: no
Broker endpoint called: no
Model training: no
Untouched holdout scored: no
Decision: `track_a_foundational_truth_complete_manual_strategy_selection_required`

## Bottom Line

Protocol101 is still the paper default because it is the best current expression of a real trader playbook: wait most of the time, trade one premium-rich SPXW 0DTE contract when the surface/event signal is strong enough, enforce one-position serial exposure, and use proven operational plumbing.

The path to improving it is now clearer. The next useful model is not a generic bigger neural network. It should be a narrow playbook-aware policy that changes one specific Protocol101 decision in one specific regime, with Protocol101 preserved as the control.

Track A has reached a stopping point for automated diagnostics. A strategy-selection packet now recommends one first hypothesis for manual review and diagnostic replay: `PROTOCOL101_LOSS_REVERSAL_EXIT_GATE_V1`, with losing-plus-opposite-side rows as the first subcase.

## What We Know

1. Protocol101 has a real but narrow edge.
   - Seed-1 Protocol101: `1,028` selected trades, `$319,050` PnL, `0.761` win rate, `4.421` PF.
   - It is not just a model artifact; it encodes trader-level beliefs and deployable constraints.

2. Naive "hold longer" is false.
   - Exact selected-trade runner paths cover `709 / 1,028` trades.
   - Hindsight post-exit best delta is `$1,147,620`, but naive forced-flat delta is `-$123,260`.
   - There are real runner opportunities, but extension without a giveback/exit guard destroys value.

3. Hold/exit must be action advantage, not duration.
   - The hold/exit foundation built `198,261` state rows.
   - At Protocol101 exits, oracle-best hold fraction is `0.917`, but one-step hold fraction is only `0.258`.
   - The correct question is: when does continuation advantage remain positive after slot opportunity cost, switching cost, fill/latency uncertainty, and tail risk?

4. Protocol101 blocks itself.
   - Counterfactual-flat replay found `8,179` hypothetical flat Protocol101 entries.
   - `3,723` were blocked while actual Protocol101 was already holding.
   - `2,291` open trades had at least one blocked entry.
   - `1,109` open trades had a best blocked entry whose PnL exceeded the actual open trade.
   - Best-blocked-minus-open total was `$399,510`, mostly Q1/March.

5. The slot-cost weakness is specific enough to become hypotheses.
   - Losing open trade blocks positive later signal: `485` rows, `$635,240` slot-cost total.
   - Opposite-side later signal blocked: `428` rows, `$572,430`.
   - Sequence residual slot cost: `741` rows, `$472,930`.
   - Long-duration slot cost: `396` rows, `$409,130`.
   - Same-side later signal blocked: `681` rows, `$287,560`.

## What We Do Not Know Yet

- Whether the top slot-cost rows are causally identifiable live or hindsight artifacts.
- Whether losing open trades were avoidable at entry or only after path deterioration.
- Whether opposite-side blocked signals represent real regime flips, model overreaction, or fill-sensitive noise.
- Whether same-side blocked signals call for runner holding, contract switching, or earlier exit.
- Whether any of this survives live quote freshness, latency, and fill behavior.
- Whether the exposed diagnostic splits are overfit from repeated research attention.

## Foundational Truth

The problem is no longer "can ML find more profitable trades?" It can. Many experiments found more PnL somewhere. The problem is:

> Can a model improve the trader playbook without breaking the single-slot, live-executable, low-drawdown behavior that makes Protocol101 valuable?

That means the next ML/NN model should be playbook-aware and conservative:

- It should predict baseline-relative utility, not raw PnL.
- It should decide between `keep Protocol101`, `defer`, `exit`, `hold`, `switch`, or `enter challenger`.
- It should charge for slot opportunity cost and switching cost.
- It should include downside/tail/giveback/fill sensitivity, not just expected return.
- It should only act when the declared strategy hypothesis says it is allowed to act.

## Strategy Selection Update

The follow-on strategy-selection packet ranked the four immediate choices and selected `PROTOCOL101_LOSS_REVERSAL_EXIT_GATE_V1` as the first diagnostic, not as a trainable model.

- It has the largest raw internal slot-cost evidence: `485` rows and `$635,240` positive slot-cost total.
- It overlaps heavily with the regime-flip idea: `325` rows and `$517,980` of positive slot-cost are both loss-reversal and opposite-side blocked-signal cases.
- The first manual review bucket should therefore be: current Protocol101 trade is losing, and a later opposite-side Protocol101-approved signal appears while the slot is still occupied.

This does not mean "cut losers mechanically." It means the next diagnostic should ask whether a losing/stale Protocol101 position has negative continuation advantage once the later baseline-approved opportunity, switching cost, fill latency, spread tax, and quote realism are charged.

Primary packet: `v4/docs/PROTOCOL101_STRATEGY_SELECTION_PACKET_V1.md`

## Loss-Reversal Diagnostic Update

The first selected hypothesis was then checked against the existing hold/exit action-advantage foundation.

- Loss-reversal candidates: `485`, with `$635,240` positive slot-cost total.
- Causal holding state could be matched at the later blocked-signal time for only `65 / 485` rows.
- Among matched rows, `46` were actually losing at the blocked-signal time and `19` were not.
- The highest-priority live-test bucket is current-loss plus opposite-side plus one-step-hold-negative: `20` rows and `$32,790` positive slot-cost.

This is the critical label lesson: a final losing trade is not the same thing as a live losing/stale trade. A neural model trained on the current loss-reversal rows would learn hindsight unless the label is rebuilt around current state at the later signal time.

Primary packet: `v4/docs/PROTOCOL101_LOSS_REVERSAL_EXIT_GATE_DIAGNOSTIC_V1.md`

## Causal State And Local Pricing Update

The loss-reversal branch now has two additional no-training diagnostics:

- Direct quote-path causal state matched `441 / 485` rows after falling back from unreadable official-context files to readable normalized files.
- Among matched rows, `293` were losing at the later blocked-signal time and `148` were not.
- The strongest live-state bucket is current-loss + opposite-side + negative next quote + Protocol101 exit deterioration: `145` rows.
- Local action pricing on matched rows shows switch-minus-keep of `$511,500`, or `$489,450` after `$0.25` two-side stress.
- In the priority-1 bucket, switch-minus-keep remains `$223,120` after `$0.25` stress.

This was promising, but not yet a model label. It was local event pricing, not full serial account replay. Switching one trade changes later account state, slot occupancy, and future Protocol101 opportunities. That is why the next artifact was the full serial replay summarized below.

Primary packets:

- `v4/docs/PROTOCOL101_LOSS_REVERSAL_CAUSAL_STATE_ATTACHMENT_V1.md`
- `v4/docs/PROTOCOL101_LOSS_REVERSAL_LOCAL_ACTION_PRICING_V1.md`

## Full Serial Replay Update

The priority-1 loss-reversal playbook was then tested under a full serial flat-stream replay. This replay uses the frozen hypothetical-flat Protocol101 proposal stream, reproduces the baseline serial trade count (`4,456` baseline trades at zero stress across the diagnostic fold/seed/split scope), and applies the priority-1 rule only when the current trade is losing, an opposite-side Protocol101 signal appears, the next quote is adverse, and the original hold later deteriorates.

Result:

- Zero stress: `+$201,350` versus same-scope baseline.
- `$0.10` per-side stress: `+$198,990`.
- `$0.25` per-side stress: `+$195,450`.
- All diagnostic splits have positive delta under all stress levels.
- The overlay fires `145` loss-reversal exits.

This is the first Track A result that looks like a concrete Protocol101 improvement playbook. It is still research-only because the splits are exposed, fill/latency/quote-freshness are not calibrated, missing quote-state rows are excluded, and no live no-order parity exists for this gate.

Primary packet: `v4/docs/PROTOCOL101_LOSS_REVERSAL_FULL_SERIAL_REPLAY_V1.md`

## Other Hypothesis Choices

Pick exactly one:

1. `PROTOCOL101_LOSS_REVERSAL_EXIT_GATE_V1`
   - Thesis: losing Protocol101 positions sometimes occupy the slot while a stronger positive Protocol101 signal appears later.
   - First test: manual review top losing-open-trade rows to decide if losses are entry-avoidable or path-management-avoidable.

2. `PROTOCOL101_REGIME_FLIP_EXIT_GATE_V1`
   - Thesis: opposite-side Protocol101 signals appear while the current trade is stale; these are regime flips, not runner opportunities.
   - First test: review opposite-side blocked rows for underlying reversal, MFE giveback, quote quality, and timing.

3. `PROTOCOL101_SAME_SIDE_SWITCH_OR_RUNNER_V1`
   - Thesis: same-side blocked signals imply Protocol101 stayed in the wrong contract or exited/re-entered poorly.
   - First test: distinguish same-contract continuation from same-side contract replacement, then add switching-cost economics.

4. `PROTOCOL101_LONG_DURATION_FALLBACK_EXIT_V1`
   - Thesis: long fallback/sequence holds block better later signals.
   - First test: mutually exclusive replay for only long-duration positive-slot-cost rows.

## Training Stop

No neural training should start from these diagnostics yet. The blocking issues are:

- counterfactual slot-cost labels are upper-bound and non-additive,
- fill/latency/quote freshness is not calibrated,
- the top rows need human strategy labeling,
- no causal rule has been selected,
- exposed splits are not promotion-grade,
- and Protocol101 remains the control and paper default.

The next correct action is a manual strategy review of `slot_cost_top_open_trades.csv`, then a single preregistered playbook test.

## Primary Artifacts

- Track A packet: `v4/docs/PROTOCOL101_TRACK_A_FORENSICS_V1.md`
- Slot-cost decomposition: `v4/docs/PROTOCOL101_SLOT_COST_ARCHETYPE_DECOMPOSITION_V1.md`
- Internal slot-cost counterfactual: `v4/docs/PROTOCOL101_INTERNAL_SLOT_COST_COUNTERFACTUAL_V1.md`
- Hold/exit foundation: `v4/docs/PROTOCOL101_HOLD_EXIT_ACTION_ADVANTAGE_FOUNDATION_V1.md`
- Exact runner paths: `v4/docs/PROTOCOL101_EXACT_SELECTED_TRADE_RUNNER_PATHS_V1.md`
