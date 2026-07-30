# Goal: Protocol101 FT2 Stage-0 P5 HOLD/EXIT Feasibility Attempt002

## Objective

Correct the Stage-0 standardized-entry episode construction after attempt001
validly returned `stop_insufficient_data`, then rerun the same bounded,
non-promotable HOLD/EXIT feasibility pilot.

The scientific question is unchanged:

> At an exact governed opportunity where deterministic P5 emits one
> VWAP-side nearest-ATM contract, can a causal HGB model improve the exit of
> that position relative to P5's fixed exit, the best training-selected fixed
> exit, and feature-independent random exit?

This Goal may conclude that a full owner-signed lifecycle campaign is worth
running. It must not launch that campaign.

## Why Attempt001 Stopped

Attempt001 used only P5 trades actually admitted by simulator-v5 serial
replay. P5's fixed policy occupies the account for most of a session, so the
20-session nested sample produced:

- 15 training episodes;
- 5 validation episodes; and
- 7,648 causal lifecycle rows.

Every machinery check passed, but the frozen minimum was 100 validation
episodes. Attempt001 correctly stopped before fitting, scoring, or
interpreting performance.

The defect is in the experimental control, not the market data:

- Exit isolation needs many fixed entry episodes to learn from.
- Actual serial P5 trades are appropriate for the serial consequence ledger
  but too sparse for the isolated episode ledger.
- The governed corpus already contains thousands of exact P5-permitted
  opportunities on the same 20 sessions.

Attempt002 therefore changes only the standardized episode source:

- Ledger A uses every exact P5-permitted opportunity as an independent
  standardized entry episode.
- Ledger B retains strict simulator-v5 serialization over the common ordered
  P5 opportunity stream.

No other scientific choice may change.

## Frozen Inputs

Hash and record:

- `v4/docs/protocol101/training/goals/PROTOCOL101_FT2_STAGE0_P5_HOLD_EXIT_FEASIBILITY_GOAL_2026_07_28.md`
- `v4/docs/protocol101/training/contracts/PROTOCOL101_STAGE2_OBJECTIVE_AND_GATES_PROPOSAL.md`
- `v4/audit/autoresearch/protocol101_ft2_stage0_p5_hold_exit_feasibility_attempt001/preregistration.json`
- `v4/audit/autoresearch/protocol101_ft2_stage0_p5_hold_exit_feasibility_attempt001/decision.json`
- `v4/audit/autoresearch/protocol101_ft2_stage0_p5_hold_exit_feasibility_attempt001/machinery_checks.json`
- `v4/audit/autoresearch/protocol101_ft2_stage0_p5_hold_exit_feasibility_attempt001/entry_identity_receipt.json`
- `v4/audit/autoresearch/protocol101_policy_neutral_contract_selector_stage0_feasibility_attempt002/opportunity_filter_receipt.json`
- `v4/audit/autoresearch/protocol101_policy_neutral_contract_selector_stage0_feasibility_attempt002/risk_set_receipts.json`
- `v4/model/protocol101_stage1_reference_multiplicity.py`
- `v4/model/protocol101_serial_simulator_v5.py`
- The governed fold, corpus, fee, feature, target, and identity authorities
  already frozen by attempt001.

Preserve attempt001 byte-for-byte. It remains valid
`stop_insufficient_data` evidence. Do not reinterpret its five validation
episodes.

## Output

Write results only to:

`v4/audit/autoresearch/protocol101_ft2_stage0_p5_hold_exit_feasibility_attempt002/`

Mark every result artifact:

`exploratory_non_promotable_stage0`

## Step 1: Preregister The Correction

Create and hash `preregistration.json` before building attempt002 lifecycle
rows or inspecting any attempt002 validation economics.

Inherit attempt001 unchanged:

- the same expanding fold 1;
- the same 15 nested training sessions;
- the same 5 nested validation sessions;
- zero final-OOF, G9, holdout, recorder, sealed, shadow, or paper overlap;
- deterministic P5 timing, side, and nearest-ATM contract selection;
- one bounded depth-3 HGB regressor, seed 42, at most 200 iterations;
- one-minute executable-bid hold advantage;
- zero HOLD/EXIT threshold;
- strong-shuffle seeds 8700 and 8701;
- causal features and feature firewall;
- group-balanced weights;
- fixed, exact-random, constant, reversed, and shuffled controls;
- Ledger A and Ledger B definitions;
- minimum sample, performance gates, terminal routes, and forbidden actions.

Change only:

```text
attempt001 Ledger-A episode:
  P5 trade admitted by serial replay

attempt002 Ledger-A episode:
  exact opportunity where byte-equivalent fixed_heuristic_candidates emits
  exactly one P5 contract
```

This is an owner-approved experimental-control correction. It is not
permission to change the target, features, model, sessions, controls, or pass
criteria.

## Step 2: Build The Attempt002 P5 Opportunity Episodes

Rebuild from governed source artifacts. Do not copy attempt001 lifecycle
Parquet or model artifacts.

For each nested session:

1. Load all governed policy-5 `ReferenceOpportunity` records.
2. Apply byte-equivalent `fixed_heuristic_candidates`.
3. Retain a decision only when P5 emits exactly one contract.
4. Freeze that decision/contract as one independent Ledger-A entry episode.
5. Use the same entry ask, causal entry quote timestamp, canonical slot,
   policy deadline, path source, and source hashes as the accepted machinery.
6. Exclude a P5 abstention before lifecycle-row construction; assign no
   synthetic utility.

Expected smoke anchor from selector Stage-0 attempt002:

- 6,414 retained P5 opportunities across all 20 sessions;
- 1,354 retained P5 opportunities across the 5 validation sessions.

These numbers are an identity cross-check, not a license to coerce results.
If source evolution changes them, stop and explain the exact identity delta
before fitting.

Fail closed when:

- P5 emits more than one contract;
- the P5 contract is absent or duplicated in the governed risk set;
- decision, contract, slot, episode, or path identities duplicate;
- entry ask or causal quote time is missing;
- path timestamps are missing or non-monotonic;
- any protected role appears; or
- comparator entry identities differ.

## Step 3: Preserve The Attempt001 Causal Contract

Use the attempt001 lifecycle feature contract, firewall, one-minute target,
deadline/two-clock semantics, missing-bid behavior, manual target
reproduction, HGB configuration, weighting, controls, and reporting without
change.

In particular:

- future-best and oracle labels remain diagnostic-only;
- future exit time, future PnL, future MFE/MAE, target aliases, and action
  advantages remain blocked from features;
- no-bid minutes cannot execute an exit and remain visible in replay;
- missing one-minute targets are not bridged or backfilled;
- minute rows receive nested episode/session-balanced weights; and
- train/validation separation is by complete session, never path row.

Because Ledger-A episodes overlap heavily in wall-clock time:

- report overlap counts and concurrency distributions;
- aggregate economic comparisons by session and episode;
- do not treat episodes or minute rows as independent statistical samples;
  and
- do not call Ledger A executable serial economics.

## Step 4: Ledger A

Fit and evaluate on every retained independent P5 episode using the frozen
attempt001 rules.

Evaluate:

1. Original P5 fixed exit.
2. `EXIT_NOW` at the first executable lifecycle minute.
3. Hold to deadline.
4. Best preregistered fixed exit selected on nested training sessions only.
5. Exact feature-independent random exit with elapsed-minute hazard frozen
   from nested training only.
6. Constant-score behavior.
7. Real HGB seed 42.
8. Reversed real-HGB behavior.
9. Strong-shuffle HGB seed 8700.
10. Strong-shuffle HGB seed 8701.

Do not tune the zero action threshold or any feature/model parameter.

## Step 5: Ledger B

Use the common ordered P5 opportunity stream for each validation session.
For every comparator:

- present identical ordered P5 entry intents;
- let its actual exit determine simulator-v5 occupancy;
- skip later intents while occupied;
- preserve continuous cash, affordability, one contract, fees, daily stop,
  and forced flat; and
- persist admitted and skipped identities.

Ledger B is the directional check against slot-opportunity reward hacking.
The model must not pass merely by holding an isolated episode longer while
blocking more valuable later P5 entries.

## Step 6: Terminal Decision

Return exactly one:

- `proceed_to_full_lifecycle_contract_signature_and_campaign`
- `stop_no_preliminary_exit_signal`
- `stop_mechanical_blocker`
- `stop_scientific_contract_defect`
- `stop_insufficient_data`

Return `proceed_to_full_lifecycle_contract_signature_and_campaign` only if:

1. Every identity, causality, feature-firewall, weighting, target, role, and
   two-clock check passes.
2. At least 100 validation episodes and all 5 validation sessions have usable
   evidence.
3. Real HGB has strictly positive pooled fee-adjusted PnL improvement versus
   original P5, the training-selected best fixed exit, and exact random exit
   in both Ledger A and Ledger B.
4. Real HGB improves on original P5 and the training-selected best fixed exit
   on at least 4 of 5 validation sessions in Ledger A.
5. Real HGB does not lose to either baseline on more than 1 of 5 validation
   sessions in Ledger B.
6. Real HGB's minimum lift over the three baselines strictly exceeds the
   minimum lift of reversed and both strong-shuffle controls.
7. No result requires changing any frozen scientific choice.

Passing is directional feasibility only. It returns to the owner before a
full campaign.

If fewer than 100 validation episodes remain after fail-closed construction,
return `stop_insufficient_data` without performance interpretation.

## Bounded Repair Loop

Allow at most three attempt002 mechanical repair iterations, limited to:

- opportunity and path identity joins;
- causal timestamps;
- native missing-value plumbing;
- target arithmetic;
- fee application;
- exact-random arithmetic;
- simulator-v5 adapter behavior;
- serialization; and
- tests/reporting.

Do not change sessions, episode definition, target, horizon, features, HGB
configuration, threshold, controls, pass criteria, or routes after results
are visible.

Persist every repair in `repair_log.json`. Stop if the same blocker remains
after three iterations.

## Required Artifacts

Produce the same required artifacts as attempt001 under the attempt002 output
directory, including:

- preregistration and hash;
- progress and repair log;
- independent P5 opportunity episodes;
- entry and path identity receipts;
- lifecycle rows and feature firewall audit;
- manual target reproduction;
- machinery checks;
- model receipts and predictions;
- Ledger-A episode replays;
- Ledger-B serial replays and identity receipts;
- fixed/random/reversed/shuffled control results;
- pilot results and resource projection;
- terminal decision and report; and
- complete hashes.

The report must plainly distinguish:

- independent overlapping episode evidence;
- serial one-account consequence evidence;
- absolute P5 profitability; and
- incremental exit-policy uplift over identical baselines.

## Hard Boundaries

Do not:

- change P5 opportunity timing, side, contract choice, or risk set;
- train an entry model or contract selector;
- use more than one real HGB seed or two shuffles;
- train a neural or sequence model;
- optimize the HOLD/EXIT threshold;
- run the full five-fold lifecycle campaign;
- perform multiplicity or independent final acceptance;
- access final OOF, G9, holdout, recorder, sealed, shadow, or paper evidence;
- select, promote, or freeze a Full Trader;
- contact a broker, submit paper orders, or download paid data; or
- modify promotion, runtime, launchd, recorder, or real-money state.

## Highest Allowed Claim

`Exploratory non-promotable P5 HOLD/EXIT Stage-0 attempt002 feasibility decision complete.`
