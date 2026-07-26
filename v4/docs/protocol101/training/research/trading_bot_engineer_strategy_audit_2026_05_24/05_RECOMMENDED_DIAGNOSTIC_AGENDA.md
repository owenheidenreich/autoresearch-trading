# Recommended Diagnostic Agenda Before New Model Training

The next phase should produce diagnostics, not a new challenger model. Each diagnostic should answer a trader question and end with a decision about what model, if any, should be built.

## Priority 1: Protocol101 Trade Atlas

Question: What trade archetypes does Protocol101 actually monetize?

Scope:

- Side: call/put.
- Time bucket.
- Premium bucket.
- Moneyness/offset.
- Spread and quote freshness.
- Entry score and margin.
- Exit reason.
- Duration.
- MFE, MAE, giveback, MFE capture.
- Day/session context.

Output:

- A trade archetype taxonomy.
- Top profitable and top losing archetypes.
- A list of regimes where Protocol101 should be left alone versus challenged.

Decision enabled:

- Whether the next model should patch exits, entries, or abstentions.

## Priority 2: Losing-Day And Hard-Stop Autopsy

Question: Are Protocol101 losses random cost of doing business, or are they avoidable regimes?

Known evidence:

- In Protocol114, `hard_stop` was the worst entry-reason group: 55 trades for `-$69,760`.
- In the seed-1 paper replay, `hard_stop` had 11 trades, 0% win rate, and `-$14,470`.

Scope:

- Worst days and worst hard-stop trades.
- Directional context before entry.
- Quote/spread behavior.
- Whether trades had early MFE.
- Whether loss followed same-side churn, reversal, or late entry.

Decision enabled:

- Whether to build a loss-avoidance gate, a stop-management patch, or no model change.

## Priority 3: Runner Opportunity Audit

Question: Where does Protocol101 exit too early, and where would holding longer destroy value?

Scope:

- For every Protocol101 entry, replay bid path from actual exit to forced flat.
- Evaluate fixed delayed exits, trailing giveback, target expansion, and flat-by-close.
- Segment by exit reason, side, premium, moneyness, time, and MFE before exit.

Output:

- `runner_candidate` archetypes.
- `do_not_extend` archetypes.
- `giveback_guard` archetypes.

Decision enabled:

- Whether to build a Protocol101 exit overlay instead of a new entry model.

## Priority 4: Missed-Winner And Abstention Audit

Question: Is Protocol101 too conservative, and if so where?

Scope:

- Full live-feasible candidate surface.
- Rejected Protocol101 candidates.
- Later winners in the same event/session.
- Compare rejected winners to Protocol101-selected trades.

Output:

- Missed winner clusters by time, side, premium, moneyness, spread, and surface edge.
- Distinguish "Protocol101-like but below threshold" from genuinely different playbooks.

Decision enabled:

- Whether to widen Protocol101, add a second playbook, or keep abstention strict.

## Priority 5: Protocol101 Slot Opportunity Audit

Question: Do some Protocol101 trades block better later Protocol101-like trades?

Scope:

- For every Protocol101 open interval, identify later feasible Protocol101 entries blocked by the open slot.
- Attribute blocked PnL using `entry_dt <= later_entry_dt < exit_dt`.
- Charge blocked opportunity by score, side, premium, duration, and exit reason.

Output:

- Slot-cost profile internal to Protocol101.
- Early-entry archetypes that should defer.

Decision enabled:

- Whether a learned-defer style overlay should be applied to Protocol101 itself.

## Priority 6: Side-Specific Strategy Audit

Question: Are calls and puts separate playbooks?

Scope:

- Calls vs puts by time, premium, offset, spread, MFE, MAE, exit reason, and market context.
- Identify whether put losses have different causes than call losses.

Output:

- Side-specific rules or confirmation that one shared policy is adequate.

Decision enabled:

- Whether next labels/features/models should be side-conditioned.

## Priority 7: Execution Realism

Question: Is the edge executable?

Scope:

- Live no-order full candidate surface logging.
- Quote freshness, Greeks freshness, masks, affordability, scores, and latency.
- Paper/no-order fill observations across side, premium, spread, quote age, and time.

Output:

- Fill-readiness packet.
- Timing/fill stress calibrated from observations.

Decision enabled:

- Whether any replay edge is eligible for untouched holdout or paper-default discussion.

## Priority 8: Validation Integrity

Question: Are we selecting the best exposed backtest, or finding a real strategy?

Scope:

- Strategy matrix across protocol families, seeds, thresholds, routers, overlays, and splits.
- Daily block bootstrap.
- Concentration.
- CSCV/PBO-style diagnostics where comparable series exist.
- Newly reserved untouched block.

Output:

- Formal validation readiness packet.

Decision enabled:

- Whether the next frozen candidate may be scored once on untouched data.

## Priority 9: Define The Next Model As A Strategy Hypothesis

Only after Priorities 1-8 should a new model be specified.

Acceptable model statements:

- "Protocol101 exits too early on this identified runner archetype; train an exit overlay only for that archetype."
- "Protocol101 takes early weak entries that block stronger later same-session opportunities; train a defer overlay for those states."
- "Protocol101 misses a low-premium put playbook only when spread/IV/time/momentum conditions match this cluster."
- "Protocol101 hard-stop losers show a pre-entry reversal signature; train a conservative rejection gate."

Unacceptable model statements:

- "Try a bigger network."
- "Train a model to make more money."
- "Make it hold longer."
- "Make it scalp less."
- "Tune thresholds until the exposed splits improve."

