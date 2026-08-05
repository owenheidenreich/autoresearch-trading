# Codex Handoff — Options-Chain H1 Raw Screen and `NO_EDGE`

Date: 2026-08-04  
Purpose: independent scientific and implementation review  
Requested reviewer behavior: attack the work; do not write code or rerun the experiment

## 1. Bottom line

The owner’s goal remains a one-account bot that buys SPXW 0DTE calls or puts.
That goal was not changed or replaced with futures, short premium, spreads, or
longer-dated options.

This conversation produced and ran one new, preregistered raw directional
screen:

```text
H1: Does the five-minute change in a symmetric ±20-point SPXW 0DTE
risk reversal predict the next five-minute signed SPX move?
```

The frozen verdict is **`NO_EDGE`**.

- All 215 development sessions were processed.
- Zero sessions were dropped.
- The strict gate required four economic inequalities in all five chronological
  folds.
- H1 passed only 3/5 folds and failed folds 0 and 1.
- The spent 36-session holdout was not opened.
- No null, learner, model fit, option wrapper, broker connection, paper order,
  promotion, or paper-default mutation followed.
- The exact H1 is now recorded in the canonical do-not-retest table.

This result does **not** establish that every possible option-chain observable
lacks directional information. It closes this exact risk-reversal-change family
under its frozen sign, wing construction, horizon, clock, quote-quality law,
occupancy law, and comparator gate. The long-call/put wrapper remains closed
because the required underlying-direction reopening gate did not pass.

## 2. Where the conversation started

The owner began from the committed handoff
`v4/docs/protocol101/training/execution/CODEX_HANDOFF_MECHANISM_FIRST_2026_08_04.md`
at commit `cef2f718`. The instruction was to attack the mechanism plan and
answer six scientific questions before code.

The controlling facts were:

- Five independent negatives had closed the existing directional programme.
- The terminal negative was raw economics with no null involved.
- Buying SPXW 0DTE premium had mean gross expectancy of approximately
  **−$13/trade with all friction removed**, negative in all five folds.
- The 36-session holdout was spent. There was no confirmation firewall.
- Measured ES friction was 0.358 points per trade, but the owner correctly
  challenged per-trade economics for a one-account strategy with serial
  occupancy.
- The prior learned exit was degenerate: first-step exit in about 95.1% of
  trajectories, p99 roughly $86 versus roughly $3,422 for hold-to-close.
- A compatible future entry-plus-exit system would need one objective rewarding
  both loss truncation and convex-tail capture, a deterministic catastrophic
  floor, training/validation on the entry policy’s own out-of-fold distribution,
  strict serial occupancy, and a fresh composed-system gate.
- Prior-art checking was mandatory before every hypothesis. A do-not-retest row
  or rejecting verdict anywhere in the protocol decoder was a stop.
- Raw economics had to come before nulls, models, or promotional machinery.
- `NO_EDGE` was explicitly an acceptable scientific result.

The six original questions were:

1. Is the mechanism list correct, including independent verification that prior
   screens excluded the open/close and that the broad “VIX never entered a
   model” claim was false?
2. Should survival be measured per session under one-account serial occupancy
   rather than by an unlimited per-trade average?
3. Can entry and exit be trained jointly under one objective that avoids both
   always-exit degeneracy and the mirror 98%-dominant-label failure?
4. Which nodes of the 47-node/104-edge trader graph survive, and how must each
   downstream acceptance test explicitly reference the distribution it consumes?
5. What is the smallest wave, with declared count equal to the maxT family?
6. Where does the plan fail under adversarial review?

No new trader-graph design or joint entry/exit model was implemented in this
conversation because the raw upstream directional gate failed first.

The owner then restated the product objective directly: build a 0DTE long-call
or long-put bot. The proposed reopening path was the exception already written
in the do-not-retest ledger: demonstrate directional skill on SPX over a
holdable horizon, then test the option wrapper separately.

The proposed data source was the owned SPXW chain: 251 sessions of Databento
OPRA CBBO-1m/CBBO-1s with prices, displayed sizes, and interval last-sale
fields. Candidate mechanisms initially discussed were dealer positioning/gamma,
skew, signed flow, and quote-size imbalance.

The immediate options-chain context was also preserved in the committed handoff
`v4/docs/protocol101/training/execution/CODEX_HANDOFF_OPTIONS_CHAIN_DIRECTION_2026_08_04.md`
at commit `aca9e8b1d923e818de319fde2d9c6cc1b50e1e7b`.

## 3. Corrections made before the experiment

Several broad claims were corrected during the review process.

### Options information had entered models before

It was false that the options chain had “never entered any model.” The canonical
contract already contains near-ATM straddle, put/call-mid, smile-slope, delta,
and gamma fields. The genuinely untested object was a clean **full-chain causal
directional screen against future SPX moves**.

### Dealer GEX and signed flow were not identifiable from the owned substrate

The owned history does not establish participant-labelled dealer inventory,
position sign, queue history, or a complete historical trade-by-trade signed-flow
substrate. Displayed bid/ask sizes do not reveal who owns gamma. A gamma-weighted
size statistic could only be called a displayed-quote-pressure proxy, not dealer
GEX or dealer hedge demand.

### VIX had entered prior research

The earlier statement “VIX has never entered any model” was also false. The
options-chain handoff records `vix_level`, `vix_change_5m`, `vix_change_15m`,
and `vix_minus_spx_rv15_pct_points` in the prior lean-autoresearch harness. That
campaign returned `NULL_NO_NEW_ENTRY_EDGE`. No new VIX experiment was run in
this conversation.

### The model firewall and a raw screen are different boundaries

`assert_model_alpha_firewall` governs the Protocol101 model-matrix boundary.
`admitted_feature_matrix` refuses non-`ADMITTED` features for fitting. Those
rules prevent an uncertified statistic from becoming model-facing alpha; they
do not by themselves prohibit a model-free historical measurement.

The raw H1 screen therefore used owned bid/ask prices without claiming feature
admission. Its outputs remain barred from model fitting, live inference, and
deployment until the full historical/live parent chain is separately admitted.

### The IV constants conflict was resolved explicitly

`v4/research/pathd_opra_parity_features.py` uses a 4% risk-free rate; an older
canonical contract uses 5%. H1 used the shared Path-D parity/IV implementation
at 4%, 0% dividend yield. Its source SHA-256 was
`0456e788e604fd5609b3f48cddbaee9633660fdbe63a7f62952d5d6cd93d2f1b`,
which exactly matched the existing passing solver/constants receipt. The 5%
implementation was not used.

## 4. The 20-file ChatGPT Pro review package

The owner asked for an exactly 20-file end-to-end package with the first file
containing scientific questions. Codex created:

```text
v4/docs/protocol101/training/execution/
  CHATGPT_PRO_SCIENTIFIC_HANDOFF_2026_08_04/
```

and the uploadable archive:

```text
v4/docs/protocol101/training/execution/
  CHATGPT_PRO_SCIENTIFIC_HANDOFF_2026_08_04.zip
```

The directory contained exactly 20 files and the ZIP contained exactly 20
entries. The ZIP SHA-256 was:

```text
e1aaf8bca9cc5acf07a3b11dbbedf2f32fc6d017d6fef21962e54670f75e3b99
```

File 01 was `01_SCIENTIFIC_QUESTIONS.md`. It instructed ChatGPT Pro Thinking to
preserve the long-call/put objective, attack the proposed K=2 screen, distinguish
evidence/inference/assumption/unknown, cite package files, freeze the smallest
valid experiment, accept `NO_EDGE`, and stop before code.

The remaining 19 files covered project authority, the current source of truth,
training authority, data contract, campaign history, the options-chain handoff,
recent closeouts, feature certification, the paper-default registry, entry and
feature code, the prior raw SPX screen, the research loop/gate, live feature
catalog, current entry inference, and persistent paper runtime.

The two-line prompt given to the owner for ChatGPT Pro was:

```text
Read all 20 files, starting with 01_SCIENTIFIC_QUESTIONS.md, and act as an
adversarial scientific reviewer: attack the proposed K=2 options-chain
directional screen before any code or experimentation.

Preserve the SPXW 0DTE long-call/put objective, answer every question with file
citations and evidence labels, freeze the smallest valid test, and end with GO,
REVISE, or NO-GO.
```

The in-app ChatGPT page was not signed in. The owner chose to drag and drop the
ZIP manually and saved the resulting Pro review at:

```text
v4/docs/deep-research-report.md
```

That file is untracked at handoff time.

## 5. What the ChatGPT Pro review concluded

The Pro review’s headline verdict was **`REVISE`**.

Its core recommendations were:

1. Reject H2 before outcomes.
2. Reduce the declared family from K=2 to K=1.
3. Keep only H1, the five-minute risk-reversal-change signal.
4. Replace asymmetric nearest-wing selection with fixed symmetric ±20-point IV
   interpolation.
5. Use one shared parity/IV implementation and constants hash.
6. Freeze an exact causal clock and strict one-account serial occupancy.
7. Make signed SPX points per session the primary raw statistic and points per
   executed interval secondary.
8. Compare against frozen five-minute SPX momentum on common-valid boundaries.
9. Require positivity and comparator outperformance on both metrics in all five
   folds, with no pooled rescue.
10. Treat a raw directional pass only as permission for a separate fixed option-
    wrapper screen.

The review also proposed a future compatible entry/exit objective that combines
mean net P&L, a loss-floor penalty, and a positive-tail-forfeiture penalty while
retaining a deterministic catastrophic floor. That objective was not
implemented because H1 failed upstream.

The report contains ChatGPT internal `filecite` markers. Its reasoning is useful,
but those markers are not independently resolvable outside the original chat.

## 6. Prior-art work before outcomes

The executable `prior_art_check` in
`v4/research/pathd_research_loop.py` was run before code and return inspection.

### H1 result before the run

Mechanism:

```text
five-minute symmetric SPXW 0DTE risk-reversal change predicts
next-five-minute signed SPX points
```

Exact terms:

```text
risk reversal
skew change
option-surface direction
implied volatility skew
put skew
smile slope
```

A broader pass also used `skew`, `smile`, `volatility surface`, `put-call`, and
`put call`. The exact pass found one nonblocking reference. The broad pass found
four nonblocking references. Neither found a do-not-retest row or rejecting
decoder verdict. H1 was therefore eligible before outcomes.

### H2 result before the run

H2 was the proposed gamma-weighted displayed quote-pressure statistic. Its exact
search found no hit. A broad semantic pass using `gamma`, `imbalance`, `bid
size`, `ask size`, `book pressure`, and `order book` found a blocking rejected
Protocol010 gamma/theta-veto row.

H2 was removed without inspecting outcomes for three independent reasons:

- the broad prior-art block;
- lack of participant/inventory sign, making the dealer mechanism
  non-identifiable; and
- displayed-size and parent-chain live-parity receipts were not admitted.

H2 was never coded or run. The H1 result does not falsify H2; H2 remains blocked.

## 7. Frozen H1 specification

The preregistration is:

```text
v4/docs/protocol101/training/research/
  PATHD_OPTIONS_CHAIN_H1_PREREGISTRATION_2026_08_04.md
```

It was written before any H1 return was inspected.

### Signal

At completed OPRA CBBO-1m boundary `b`:

```text
put_target  = implied_spot_b - 20 points
call_target = implied_spot_b + 20 points

RR_b = interpolated_put_IV(put_target)
       - interpolated_call_IV(call_target)

score_b = -(RR_b - RR_(b-5m))
```

Positive score mapped to CALL/+1, negative to PUT/−1, and exact zero or missing
input to WAIT/0. There was no threshold, deadband, magnitude filter, learned
parameter, or post-result sign selection.

Wing IVs were linearly interpolated from immediately bracketing five-point
strikes. Exact target strikes used one strike. Extrapolation and brackets wider
than five points failed closed.

### Quote law

- Same-session SPXW OSI contracts only.
- `ts_recv` had to equal the completed minute boundary.
- `ts_event <= ts_recv`.
- Quote age could not exceed 90 seconds.
- Nonfinite, crossed, missing, or late quotes were unavailable.
- Missing boundaries, parity pairs, brackets, or IV solutions produced WAIT;
  nothing was carried forward.

### Clock

- Required option snapshots began at 09:55 ET to support the first 10:00 score.
- Candidate boundaries were every minute from 10:00 through 15:50 ET.
- Decision emission was the completed boundary plus **2,336 ms**.
- The label start was the latest official completed SPX close available at the
  decision under the existing bar-open-plus-one-minute law and 90-second SPX
  staleness cap.
- The label end used the same law at decision plus five minutes.
- Datetime integer conversion explicitly pinned nanoseconds, avoiding the known
  1,000× `datetime64[us]` defect.
- The open before 10:00 remains untested. The late session was extended from the
  earlier 15:20/15:00 research windows through 15:50, with flatness by 15:55.

### Occupancy and comparator

H1 and five-minute SPX momentum each replayed the same common-valid boundary
set through its own one-account state. A trade at `t` occupied `[t,t+5m)`.
Signals before `t+5m` were skipped. A new trade could enter exactly at `t+5m`.

Primary metric:

```text
mean signed SPX points per session under serial occupancy
```

Secondary metric:

```text
mean signed SPX points per executed interval
```

Comparator:

```text
sign(SPX_available_at_t - SPX_available_at_t-5m)
```

### Gate

H1 passed only if all four were true in every fold:

1. H1 points/session > 0.
2. H1 points/executed interval > 0.
3. H1 points/session > momentum points/session.
4. H1 points/executed interval > momentum points/executed interval.

One failed condition in one fold was terminal `NO_EDGE`. The pooled result could
not rescue it.

## 8. Implementation and tests

New implementation:

```text
v4/research/pathd_options_chain_direction_screen.py
```

New targeted tests:

```text
v4/tests/test_pathd_options_chain_direction_screen.py
```

The implementation reused, rather than replaced:

- development-session selection and chronological folds;
- official SPX provenance reader;
- OSI parsing;
- OPRA call/put parity estimator;
- shared 4% IV solver and expiry clock;
- official SPX availability functions; and
- the executable prior-art checker.

The new code added:

- symmetric wing interpolation with saved strike/symbol/weight identities;
- causal quote filtering;
- exact five-minute RR differencing;
- CALL/PUT/WAIT mapping;
- separate serial replays for H1 and momentum;
- fold-level gates;
- source and artifact hashes;
- a future-chain-mutation invariance test; and
- exact sign-reversal identity.

One preflight bug was caught and fixed before the full run: the mutation check
attempted to modify a read-only NumPy view. The array is now copied before
mutation. No H1 return was inspected to make that repair.

Preflight on early, middle, and late development sessions produced 266–355 of
356 required RR snapshots. The future mutation on the first session changed
82,302 later source rows while leaving all 180 earlier snapshots identical.

Targeted verification after the ledger update:

```text
38 passed in 1.10s
```

The group included research-loop prior-art tests, feature-admission tests,
Phase-0b certification tests, and the new H1 tests.

`v4/research/pathd_opra_parity_features.py` was already an untracked working-tree
file and was not modified in this conversation. Its bytes matched the existing
solver receipt exactly.

## 9. Frozen result

Result directory:

```text
v4/audit/autoresearch/
  pathd_options_chain_h1_direction_screen_2026_08_04/
```

Headline fold ledger:

| Fold | Sessions | H1 points/session | H1 points/interval | Momentum points/session | Momentum points/interval | Gate |
|---:|---:|---:|---:|---:|---:|:---:|
| 0 | 34 | −10.274118 | −0.153953 | +11.997059 | +0.179850 | FAIL |
| 1 | 33 | −5.257576 | −0.087055 | −0.786667 | −0.013026 | FAIL |
| 2 | 33 | +0.313030 | +0.004653 | +0.011212 | +0.000167 | PASS |
| 3 | 33 | +22.862424 | +0.340614 | +0.071515 | +0.001066 | PASS |
| 4 | 33 | +1.567273 | +0.022855 | +1.328182 | +0.019368 | PASS |

The existing expanding-fold routine evaluates 166 sessions across the five
folds (`34/33/33/33/33`). The initial 44-session prefix and five one-session
embargoes are not fold outcomes. All 215 development sessions were nevertheless
materialized and included in diagnostics.

The gate failed because folds 0 and 1 were negative and below momentum on both
metrics. Fold 3 was large and positive, showing regime concentration, but the
frozen rules forbid pooled or regime rescue.

Additional facts:

- H1 serial trades checked by sign identity: 14,064.
- Sign reversal was the exact negative for every checked trade.
- Future mutation prefix snapshots checked: 180; passed.
- Common-valid boundaries/session: min 72, median 286, mean 276.73, max 351.
- Mean wing-pair turnover over five minutes: approximately 52.18%.
- Across all 215 materialized sessions, H1 summed to +106.09 SPX points and
  momentum to +210.11. This pooled number is **not** the gate and cannot rescue
  the two failing folds.
- Development-prefix chain inventory: 392–1,180 distinct contracts/session,
  median 468, mean 523.74.
- Development-prefix rows/session: 86,039–433,530, median 172,450, mean
  186,766.07.

The earlier all-251-session census reported different maxima/medians because it
covered the full corpus inventory. Do not conflate that census with this
development-only run; the held-out session contents were not read here.

## 10. Receipt and artifact verification

Primary receipt:

```text
v4/audit/autoresearch/pathd_options_chain_h1_direction_screen_2026_08_04/
  receipt.json
```

Receipt self-hash:

```text
1e9e69cfc8c7985dcbc4f374119d6fe649ef08db72fa692ca4c756f7aba73724
```

The self-hash was independently recomputed with `stable_hash` after removing
the `receipt_sha256` field and matched. Every artifact hash embedded in the
receipt also matched its file.

Result artifacts:

```text
decisions.parquet
fold_results.csv
receipt.json
results.md
session_results.csv
trades.csv
```

The receipt records all of the following as false:

```text
protected_holdout_opened
model_trained
null_run
option_wrapper_run
paper_order_submitted
paper_default_modified
```

## 11. Post-result closure

After sealing and verifying the result, one row was appended to §4 of:

```text
v4/docs/protocol101/training/history/
  PROTOCOL101_PRIOR_CAMPAIGN_DISTILLATION.md
```

The row closes this exact H1 and forbids rescue through opposite sign,
deadbands, thresholds, smoothing, other wing offsets, adjacent horizons,
intraday-window selection, looser freshness, or pooled folds.

The executable prior-art check was rerun after the ledger update. It now returns
a blocking hit on the new do-not-retest row, as intended. The new unit test also
asserts that a future H1 rerun is blocked.

This creates an important reproducibility sequence:

1. Pre-run H1 prior-art receipt: clear and embedded in the experiment receipt.
2. Frozen run: `NO_EDGE`.
3. Post-run canonical ledger update: H1 now blocks.

Do not mistake the current blocking result for evidence that prior art blocked
the original run. It did not; the block was added only after the sealed result.

## 12. What was deliberately not done

- No option payoff or 0DTE wrapper was evaluated after H1 failed.
- No model was trained.
- No threshold, sign, wing, quote-age rule, horizon, or time window was changed.
- No null or surrogate was run.
- No attempt was made to rescue the strong fourth fold.
- No H2 result was generated.
- No live data capture, broker access, paper order, runtime flag edit, launch
  scheduling change, promotion packet, or paper-default replacement occurred.
- The existing Protocol101 paper default remains unchanged.
- No commit was created.

## 13. Working-tree state at handoff

Relevant state when this handoff was written:

```text
 M v4/docs/protocol101/training/history/PROTOCOL101_PRIOR_CAMPAIGN_DISTILLATION.md
?? v4/docs/deep-research-report.md
?? v4/docs/protocol101/training/execution/CHATGPT_PRO_SCIENTIFIC_HANDOFF_2026_08_04.zip
?? v4/docs/protocol101/training/execution/CHATGPT_PRO_SCIENTIFIC_HANDOFF_2026_08_04/
?? v4/docs/protocol101/training/execution/CLAUDE_ATTACK_ON_MECHANISM_FIRST_PLAN_2026_08_04.md
?? v4/docs/protocol101/training/research/PATHD_OPTIONS_CHAIN_H1_PREREGISTRATION_2026_08_04.md
?? v4/research/pathd_options_chain_direction_screen.py
?? v4/tests/test_pathd_options_chain_direction_screen.py
```

The audit result directory exists but may be hidden by repository ignore rules.
Do not assume an omitted `git status` entry means the evidence is absent.

The untracked Claude attack document pre-existed this implementation and was
not modified. The broader working tree contains other owner changes; preserve
them.

## 14. Review order

Read these in order:

1. This handoff.
2. `v4/docs/deep-research-report.md`.
3. `v4/docs/protocol101/training/research/PATHD_OPTIONS_CHAIN_H1_PREREGISTRATION_2026_08_04.md`.
4. `v4/research/pathd_options_chain_direction_screen.py`.
5. `v4/tests/test_pathd_options_chain_direction_screen.py`.
6. `v4/audit/autoresearch/pathd_options_chain_h1_direction_screen_2026_08_04/receipt.json`.
7. The other five result artifacts in the same directory.
8. The new H1 row in `PROTOCOL101_PRIOR_CAMPAIGN_DISTILLATION.md` §4.
9. `v4/research/pathd_opra_parity_features.py` and its Phase-0b receipts.
10. `v4/research/pathd_phase1_entry.py`, especially session selection, SPX
    provenance, timestamp handling, and fold construction.

## 15. Questions for the independent reviewer

Answer these in writing and stop. Do not modify code or rerun outcomes.

1. Is the `NO_EDGE` verdict a correct mechanical consequence of the frozen gate?
2. Is `score = -Δ5m(RR)` mapped to CALL/PUT with the intended sign, or was the
   economic sign misconstrued before outcomes?
3. Does the combination `ts_recv == boundary`, `ts_event <= ts_recv`, 90-second
   quote age, and decision at boundary + 2,336 ms establish the intended causal
   historical clock? Identify any hidden interval-label or snapshot semantics.
4. Is symmetric linear IV interpolation at implied spot ±20 points implemented
   exactly as preregistered, including exact-strike and missing-bracket behavior?
5. Does call/put parity use only causal option inputs, and is the 4% constants
   choice correctly tied to the existing receipt?
6. Are H1 and momentum compared fairly under their independent serial account
   states and common-valid boundary set? Do the slightly different execution
   counts create any gate defect?
7. Is mean points/session the right primary harvestability statistic, and is
   the points/interval secondary computed correctly?
8. Does the five-fold routine use the intended 166 evaluation sessions without
   leaking the 44-session prefix or five embargo sessions into the gate?
9. Are the future-mutation and sign-identity checks strong enough? What validity
   test is missing that could turn this into `INVALID_EXPERIMENT` rather than
   `NO_EDGE`?
10. Is the H1 do-not-retest row scoped correctly, or does it overclaim by barring
    adjacent horizons, alternate wing distances, or the opposite sign?
11. Was H2 correctly removed before outcomes? Separate the weak broad prior-art
    text match, the model-admission issue, and the genuine non-identifiability
    problem.
12. Does any evidence justify the conditional option-wrapper gate despite H1’s
    3/5 result? Under the frozen rules the answer is no; identify any scientific
    reason that conclusion could be wrong without post-hoc widening.
13. What remains genuinely untested and materially different after this result?
14. Where is the most likely convincing false negative or false positive in this
    exact implementation?

Required final reviewer verdict:

```text
AFFIRM_NO_EDGE
RECLASSIFY_INVALID_EXPERIMENT
or
REOPEN_FOR_A_SPECIFIC_PRE-OUTCOME_DEFECT
```

If selecting the third option, name one concrete defect that existed before
outcomes and explain why repairing it is not post-hoc hypothesis widening.

## 16. Current stopping point

The correct current state is:

```text
H1 risk-reversal change: CLOSED / NO_EDGE
H2 gamma-weighted quote pressure: NOT RUN / BLOCKED
underlying directional reopening gate: NOT CLEARED
SPXW 0DTE long-call/put wrapper: REMAINS CLOSED
model training: NOT AUTHORIZED BY THIS RESULT
paper default: UNCHANGED
```

Stop for independent review.
