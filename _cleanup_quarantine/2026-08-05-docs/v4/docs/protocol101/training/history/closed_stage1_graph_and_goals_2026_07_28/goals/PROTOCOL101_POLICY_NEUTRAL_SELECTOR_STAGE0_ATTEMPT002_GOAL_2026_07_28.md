# Goal: Protocol101 Policy-Neutral Selector Stage-0 Attempt002

## Objective

Correct the Stage-0 opportunity-grid implementation so it matches the
owner-approved scientific question, then rerun the bounded, non-promotable
feasibility pilot from clean artifacts.

The scientific question is unchanged:

> At a minute when frozen P5 already permits an entry, can M0 or M1 choose a
> better eligible SPXW contract than the deterministic P5 contract selector
> and exact feature-independent random selection?

This Goal may return `proceed_to_full_campaign`, but it must not launch the
full campaign.

## Why Attempt001 Stopped

Attempt001 correctly stopped with `stop_scientific_contract_defect` before
interpreting model scores. Its runner admitted every decision with at least
one eligible contract, including 438 of 1,792 validation decisions where the
frozen P5 VWAP-side selector emitted no contract. That made the deterministic
baseline undefined on 24.44% of the evaluation grid.

The original owner-approved selector Goal already resolves this ambiguity:

- Entry timing is frozen to P5.
- An opportunity exists only when the byte-equivalent frozen P5 selector
  emits exactly one contract.
- At those P5-permitted minutes, the learned and random selectors may choose
  from all governed eligible calls and puts.
- Only the selected contract may differ.

Therefore this is a correction of the attempt001 implementation and
preregistration namespace, not permission to invent a new trading policy.

## Frozen Inputs

Use and hash these authorities:

- `/Users/gduby/.codex/attachments/fff2e3d8-66e1-4cd2-99a9-27b44962e5be/pasted-text.txt`
- `v4/audit/autoresearch/protocol101_entry_objective_audit_attempt001/recommended_entry_objective.json`
- `v4/audit/autoresearch/protocol101_policy_neutral_contract_selector_stage0_feasibility_attempt001/decision.json`
- `v4/audit/autoresearch/protocol101_policy_neutral_contract_selector_stage0_feasibility_attempt001/scientific_contract_audit.json`
- `v4/audit/autoresearch/protocol101_policy_neutral_contract_selector_stage0_feasibility_attempt001/preregistration.json`
- `v4/model/protocol101_stage1_reference_multiplicity.py`
- The governed fold, feature, target, noise, and dataset authorities already
  hashed by attempt001.

Preserve attempt001 byte-for-byte as invalid, uninterpreted evidence. Do not
inspect, summarize, compare, or reuse its persisted predictions or fitted
models.

## Output

Write only to:

`v4/audit/autoresearch/protocol101_policy_neutral_contract_selector_stage0_feasibility_attempt002/`

Mark every JSON, table, model receipt, and report:

`exploratory_non_promotable_stage0`

## Step 1: Preregister Before Computing Results

Create and hash `preregistration.json` before fitting or scoring.

Freeze all attempt001 choices unchanged:

- Governed fold: expanding fold 1.
- Training sessions: the same 15 nested training sessions.
- Validation sessions: the same 5 later nested validation sessions.
- Final OOF overlap: zero.
- Models: M0 and M1 only.
- Model seed: 42 only.
- Strong-shuffle seeds: 8600 and 8601 only.
- Signed 17-feature firewall.
- Frozen target and noise law.
- Group-balanced weights.
- Ledger A decision-local evaluation only.
- Deterministic, exact-random, constant-score, reversed, and two
  strong-shuffle controls.
- Existing directional metric and terminal routing rules.

Correct and freeze only the opportunity definition:

1. Load governed policy-5 `ReferenceOpportunity` records.
2. Apply the byte-equivalent `fixed_heuristic_candidates` implementation.
3. Include a decision if and only if P5 emits exactly one deterministic
   candidate.
4. Exclude P5 abstentions before fitting, scoring, target aggregation, or
   evaluation.
5. Fail closed if P5 emits more than one candidate, the deterministic
   candidate is absent from the governed risk set, decision identities
   duplicate, or a retained risk set is empty.
6. For every retained decision, expose all governed eligible calls and puts
   to M0, M1, exact random, constant score, reversed, and shuffle selectors.

Persist the excluded P5-abstention count by session and reason. Exclusions are
not failed predictions and receive no synthetic no-trade utility.

## Step 2: Rebuild Clean Attempt002 Risk Sets

Rebuild all 20 sampled sessions from governed source artifacts. Do not copy
attempt001 Parquet files or model artifacts.

Verify before fitting:

- Every retained decision has exactly one deterministic P5 baseline.
- Every deterministic baseline contract belongs to the exact risk set.
- Every selector sees the same ordered candidate identities per decision.
- The 12-feature reference view joins one-to-one to the 17-feature selector
  view without changing opportunity or candidate identities.
- No final OOF session is present.
- No future path, target, or label column enters model features.
- Missing feature values remain native HGB missing values; no future-derived
  imputation is used.
- Group-balanced decision/session weights pass the frozen tolerances.
- Sampled target calculations reproduce manually.

If any machinery check fails, diagnose and repair the local attempt002
implementation. Allow up to three bounded repair iterations that do not
change sessions, features, target, hyperparameters, controls, metrics, or
gates. Record every repair. If the same blocker remains, return
`stop_mechanical_blocker`.

## Step 3: Run The Bounded Pilot

After all machinery checks pass, train from scratch:

- Real M0, seed 42.
- Real M1, seed 42.
- M0 and M1 under strong shuffle 8600.
- M0 and M1 under strong shuffle 8601.

Evaluate on the five nested validation sessions:

- Deterministic P5 VWAP-side nearest-ATM baseline.
- Exact random expected utility over the complete retained risk set.
- Constant-score selector using the frozen deterministic candidate ordering.
- Real M0 and M1.
- Reversed M0 and M1.
- Both shuffled M0 and M1 controls.

Use Ledger A only. Do not run simulator-v5 serial economic replay.

Report paired decision-local and session-level results without treating
candidate rows as independent observations. Report results per session and
pooled, plus opportunity and candidate counts.

## Step 4: Resource Projection

Measure and report:

- Risk-set build time.
- Model-fit and scoring time.
- Peak memory.
- Artifact disk usage.
- Directional projection for the full campaign.

Separate core build/training projections from later full-campaign inference,
20-shuffle, multiplicity, simulator, and independent-verification costs. Do
not describe the core projection as total end-to-end campaign runtime.

## Step 5: Terminal Decision

Return exactly one:

- `proceed_to_full_campaign`
- `stop_no_preliminary_signal`
- `stop_mechanical_blocker`
- `stop_scientific_contract_defect`

Return `proceed_to_full_campaign` only when:

1. Every machinery and identity check passes.
2. M0 or M1 has positive primary-utility improvement versus both the
   deterministic P5 selector and exact-random expectation.
3. The best real model's minimum lift over those baselines strictly exceeds
   every reversed and shuffled control under the frozen comparison rule.
4. No result requires another scientific-contract change.

Otherwise stop and identify whether the result is no signal, insufficient
data, a mechanical blocker, or a scientific-contract defect.

## Required Artifacts

- `preregistration.json`
- `preregistration.sha256`
- `progress.json`
- `opportunity_filter_receipt.json`
- `risk_sets.parquet`
- `risk_set_receipts.json`
- `machinery_checks.json`
- `target_manual_reproduction.json`
- `model_receipts/`
- `predictions.parquet`
- `selector_results.json`
- `control_results.json`
- `resource_projection.json`
- `decision.json`
- `report.md`
- `hashes.sha256`

The final report must state plainly:

- How many decisions P5 permitted and how many it abstained from.
- Whether M0 or M1 beat deterministic P5 and exact random.
- Whether reversed or shuffled controls showed a comparable effect.
- Why the terminal route was chosen.

## Hard Boundaries

Do not:

- Inspect or reuse attempt001 model predictions.
- Run the full five-fold campaign.
- Use seeds 43 or 44.
- Run more than two strong shuffles.
- Run simulator-v5 serial replay.
- Run 20,000-replicate inference or campaign multiplicity.
- Perform independent final verification.
- Access G9 or the protected holdout.
- Train entry timing or HOLD/EXIT models.
- Contact a broker or submit paper orders.
- Download paid data.
- Modify promotion, runtime, launchd, or real-money state.
- Change the frozen target, features, model configuration, controls, metric,
  sessions, or gates after results are visible.

## Highest Allowed Claim

`Exploratory non-promotable Stage-0 feasibility decision complete on the corrected frozen-P5 opportunity grid.`
