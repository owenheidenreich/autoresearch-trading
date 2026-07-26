# Protocol101 Current Endeavor External AI Handoff

Generated: 2026-07-09

## Purpose Of This Handoff

This document is for an external AI reviewer that needs to understand the current Protocol101 effort without reading the entire codebase or the full month-long conversation.

The end goal is still:

```text
Build a profitable SPXW 0DTE model that is synchronized between historical training/replay and IBKR live/paper trading, and is eventually ready for guarded IBKR paper-trade validation.
```

The current endeavor is not yet paper trading. It is a gated sequence of small Codex `goal` prompts that remove one blocker at a time:

```text
prove historical/live game parity
  -> recover only parity-safe features
  -> train/uplift-test only after parity is clean
  -> route to entry training, learned exits, or data-plane redesign
  -> only later consider IBKR shadow/paper-submit validation
```

The user wants critique on whether this path is sensible, whether we are missing a better solution, and how to avoid repeating the original mistake: hill climbing a model on a historical game that does not match what the model can see in IBKR.

## One-Sentence Current State

The project is currently testing whether any parity-safe feature subset can recover real out-of-sample entry signal under `protocol101-live-v2-microstructure-masked`; Group 2 geometry/moneyness has been rejected as no real signal, and a separate Codex goal is currently working on Group 1 non-VIX index/context uplift.

## The Original Problem

The project originally had historically strong-looking Protocol101 equity results, but IBKR live/paper behavior did not match historical replay. In early June 2026, some Databento/ThetaData historical replays showed trades on days where IBKR live/paper sessions showed no trades.

That raised the core concern:

```text
The model may have been trained and validated on a different game than the one it sees through IBKR.
```

The project therefore pivoted from "train harder" to:

```text
Synchronize the historical and live decision game first.
Only then hill climb.
```

## Important Conceptual Decisions Already Made

### 1. The Old High-PnL Curve Is Not The Trustworthy Target

Earlier strong results are no longer treated as a live-relevant benchmark unless they can be reproduced causally under the same contract used live.

Reasons:

- Older official results mixed different runner/simulation semantics.
- Some feature semantics did not survive live/IBKR parity checks.
- Strict one-account serial replay and live-reproducible feature contracts materially reduced performance.

### 2. Live-Reproducibility Is The Law

The project now prioritizes:

- Causal features only.
- No labels/path/future values in model inputs.
- Same decision timestamp convention historical/live.
- Same candidate universe semantics.
- Same replay contract as live/paper.

### 3. Masked-v2 Is A Conservative Fair Baseline, Not The Final Feature Philosophy

The active fair contract is:

```text
protocol101-live-v2-microstructure-masked
```

The required model-facing transform is:

```text
mask_vendor_sensitive_option_quote_greek_microstructure
```

This mask blocks vendor-sensitive quote/Greek/microstructure fields from model alpha. Raw bid/ask/mid/spread/Greeks/volume/OI are still allowed for labels, fills, PnL, tradability, and audit, but not as model-facing alpha unless a feature group passes parity plus uplift evidence.

This is intentional: masking likely sacrifices some signal, but prevents training on features that looked good historically while drifting live.

## Current Governing Documents

Read these first:

- [AGENTS.md](/Users/gduby/Documents/autoresearch-trading/AGENTS.md)
- [Protocol101 current phase/training handoff](/Users/gduby/Documents/autoresearch-trading/v4/docs/PROTOCOL101_CURRENT_PHASE_AND_TRAINING_HANDOFF_2026_07_08.md)
- [Fair-contract training and feature recovery plan](/Users/gduby/Documents/autoresearch-trading/v4/docs/PROTOCOL101_FAIR_CONTRACT_TRAINING_AND_FEATURE_RECOVERY_PLAN_2026_07_08.md)
- [Stage-1 objective and gates proposal](/Users/gduby/Documents/autoresearch-trading/v4/docs/PROTOCOL101_STAGE1_OBJECTIVE_AND_GATES_PROPOSAL.md)
- [Parity certification v2 microstructure masked](/Users/gduby/Documents/autoresearch-trading/v4/docs/PROTOCOL101_PARITY_CERTIFICATION_V2_MICROSTRUCTURE_MASKED_2026_07_07.md)
- [Synchronization resolution](/Users/gduby/Documents/autoresearch-trading/v4/docs/PROTOCOL101_SYNCHRONIZATION_RESOLUTION.md)
- [Recorder-first parity](/Users/gduby/Documents/autoresearch-trading/v4/docs/PROTOCOL101_RECORDER_FIRST_PARITY.md)
- [Trade shape menu v2 proposal](/Users/gduby/Documents/autoresearch-trading/v4/docs/PROTOCOL101_TRADE_SHAPE_MENU_V2_PROPOSAL.md)

## Active Constraints

Unless explicitly authorized by the user in a specific goal prompt:

- Do not call broker/IBKR APIs.
- Do not paper-submit.
- Do not download paid data.
- Do not change promotion/defaults.
- Do not edit runtime flags.
- Do not edit launchd/automation.
- Do not touch real-money paths.
- Do not use protected holdout/recorder/parity-confirmation sessions for training or threshold selection.
- Do not loosen gates because PnL is weak.
- Do not unmask quote/Greek/microstructure alpha without parity evidence.

Use this interpreter for governed/offline work:

```text
~/.autoresearch-trading/runtime-venv/bin/python
```

## The Goal-Prompt Workflow

The user is now using this workflow:

1. Codex is given one narrow `goal` prompt for the next blocker.
2. Codex works until that goal reaches a definitive outcome.
3. The user reports the result back.
4. A new narrow goal is chosen based on evidence.
5. Repeat until a model is paper-ready, or the project hits a true design/data-plane dead end.

This is deliberate. Large "do everything" goals previously caused confusion because they mixed parity repair, feature recovery, training, and paper readiness.

## Recent Evidence Chain

### A. Stage-1 Masked Baseline Failed

Artifact:

- [Masked-v2 Stage-1 failure packet](/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_live_v2_microstructure_masked_stage1_attempt001/failure_packet.json)

Summary:

- No Stage-1 masked-v2 candidate was eligible for paper-readiness validation.
- This did not prove the project is impossible; it showed the conservative masked baseline does not have enough robust signal under the current fixed-exit setup.

### B. Initial Feature Recovery Ladder Was Provisionally Blocked

Artifacts:

- [Feature recovery ladder terminal packet](/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_live_v2_feature_recovery_ladder_terminal_packet/report.md)
- [Terminal summary](/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_live_v2_feature_recovery_ladder_terminal_packet/summary.json)

Initial result:

- Groups 1-4 were rejected before uplift because of candidate-membership parity failures.
- Group 5 volume/OI was blocked by missing certified paired trace fields.
- Group 6 raw microstructure was blocked because normalized quote/spread parity had not passed.

Important interpretation:

```text
This did not prove the feature groups were useless.
It proved the parity evidence layer was not clean enough to test them.
```

### C. Root-Cause Triage Found Candidate-Universe Mismatch

Artifacts:

- [Feature recovery parity root-cause triage report](/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_live_v2_feature_recovery_parity_root_cause_triage/report.md)
- [Feature recovery parity root-cause triage summary](/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_live_v2_feature_recovery_parity_root_cause_triage/summary.json)

Findings:

- The "48 mismatches" were 48 decision minutes, not 48 contracts.
- They decomposed into historical-only candidates, IBKR-only candidates, and ladder metadata deltas.
- A major source was one-strike ATM/ladder shifts, especially on 2026-07-02.
- Other issues included quote freshness/tradability and source-field drift.

### D. Candidate-Universe Parity Repair Fixed Ladder/ATM Drift

Artifacts:

- [Candidate-universe parity repair report](/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_live_v2_candidate_universe_parity_repair/report.md)
- [Candidate-universe parity repair summary](/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_live_v2_candidate_universe_parity_repair/summary.json)

Result:

- Source-aligned SPX/VIX context fixed the ladder problem.
- Removed all 6 one-strike ATM shifts.
- Removed all 184 shared-contract ladder metadata deltas.
- Removed all 571 shared accepted-contract metadata delta items.
- Remaining blocker dropped from 48 mismatch rows to 44.

Interpretation:

```text
The ladder/context problem was real and repairable.
The remaining 44 rows are option quote/tradability source differences.
```

### E. Option-Quote/Tradability Source Policy Resolution

Artifacts:

- [Option-quote source policy report](/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_live_v2_option_quote_source_policy_resolution/report.md)
- [Option-quote source policy summary](/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_live_v2_option_quote_source_policy_resolution/summary.json)
- [Policy options](/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_live_v2_option_quote_source_policy_resolution/policy_options.json)
- [Residual mismatch classification CSV](/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_live_v2_option_quote_source_policy_resolution/residual_mismatch_classification.csv)

Result:

- Remaining 44 mismatch rows are legitimate cross-vendor/causal quote-source differences, not an adapter/filter bug.
- Causes include premium cap, spread cap, stale quote, invalid bid/ask timing, and quotes arriving after the decision boundary.
- Exact cross-vendor candidate parity is not currently feasible using current Databento/ThetaData historical replay versus IBKR recorder replay.

Selected policy direction:

```text
boundary-stable tradability + static-ladder model universe for non-quote features
```

This means:

- Model-facing universe is the shared 42-slot SPXW 0DTE strike ladder.
- Quote/tradability/freshness/affordability remain causal guard/audit fields.
- Quote/liquidity/Greek alpha remains blocked.
- Non-quote feature groups can be audited under a stable model-universe policy.

### F. Static-Ladder Boundary-Stable Policy Passed

Artifacts:

- [Static-ladder boundary-stable policy audit report](/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_live_v2_static_ladder_boundary_stable_policy_audit/report.md)
- [Static-ladder boundary-stable policy audit summary](/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_live_v2_static_ladder_boundary_stable_policy_audit/summary.json)
- [Static-ladder policy replay metrics](/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_live_v2_static_ladder_boundary_stable_policy_audit/policy_replay_metrics.json)

Result:

- Static-ladder universe parity passed.
- 1077 rows checked, 359 per recorder day.
- 45234 static slot pairs checked.
- Universe match rate: 1.0.
- Model-facing candidate count parity: true.
- Guard differences preserved and not hidden.

Feature consequences:

- Group 2 candidate geometry/moneyness became eligible for uplift.
- Group 1 full set remained blocked by VIX-change finite coverage.
- Quote/liquidity/Greek alpha remained blocked.

### G. Group 2 Geometry/Moneyness Uplift Was Rejected

Artifacts:

- [Group 2 attempt 002 report](/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_live_v2_group2_geometry_uplift_attempt002/report.md)
- [Group 2 attempt 002 summary](/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_live_v2_group2_geometry_uplift_attempt002/summary.json)
- [Group 2 attempt 002 routing decision](/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_live_v2_group2_geometry_uplift_attempt002/routing_decision.json)

Result:

- `group2_rejected_no_real_signal`
- Attempt002 reproduced attempt001 after adding full diagnostics.
- 16,958,088 prediction rows persisted.
- 14,352 trade rows and path diagnostics persisted.
- No policy passed all gates.
- G4 drawdown passed 0/21 attempts in primary and 0/21 in conservative.
- No-skill/null evidence was weak.
- Positive PnL in some policies was materially concentrated in a few days/months.
- Salvageable MFE existed on many losing trades, but this was not enough for Stage-2 learned exits because the entry signal was not robust.

Interpretation:

```text
Group 2 is done for now. Do not spend more cycles on geometry/moneyness unless the broader contract or feature set changes materially.
```

### H. Group 1 Index/Context Parity Resolution

Artifacts:

- [Group 1 index/context parity report](/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_live_v2_group1_index_context_parity_resolution/report.md)
- [Group 1 index/context parity summary](/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_live_v2_group1_index_context_parity_resolution/summary.json)
- [Group 1 feature coverage CSV](/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_live_v2_group1_index_context_parity_resolution/feature_coverage.csv)
- [Group 1 parity result](/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_live_v2_group1_index_context_parity_resolution/parity_result.json)

Result:

- Full Group 1 remains blocked.
- VIX-change features were excluded because certified paired traces had only one pre-window row at 09:31 ET.
- VIX changes use exact `decision_ts - 5m/15m` lookup, so:
  - `vix_change_5m` missing 09:32-09:35.
  - `vix_change_15m` missing 09:32-09:45.
- Missingness was symmetric on IBKR and historical replay.
- Non-VIX Group 1 subset passed parity.
- 12 subset-eligible features.
- 4 VIX-change features excluded.

Interpretation:

```text
The VIX issue is a paired decision-trace warm-up/history gap, not measured VIX drift.
The non-VIX index/context subset is eligible for uplift testing.
```

## Current Active Goal Prompt

At the time of this handoff, a separate Codex goal prompt has been started or is expected to be running:

```text
Run preregistered uplift testing for Protocol101 parity-safe non-VIX Group 1 index/context subset.
```

Expected artifact directory:

- [Group 1 non-VIX uplift attempt 001 directory](/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_live_v2_group1_nonvix_uplift_attempt001)

Expected key outputs:

- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_live_v2_group1_nonvix_uplift_attempt001/summary.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_live_v2_group1_nonvix_uplift_attempt001/report.md`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_live_v2_group1_nonvix_uplift_attempt001/routing_decision.json`

Important: verify freshness/status before interpreting that directory. Do not assume completion from file existence alone.

The current goal is allowed to train/uplift-test only the parity-safe non-VIX Group 1 subset, under the same fair-contract constraints. It must not add VIX-change features, Group 2 features, Greeks/IV, quote/liquidity/spread alpha, raw microstructure, volume/OI, broker calls, paper-submit, paid data, runtime changes, launchd changes, promotion/default changes, or real-money paths.

The valid routing outcomes for this active goal are:

- `eligible_offline_candidate`
- `stage2_learned_exits_candidate`
- `group1_nonvix_rejected_no_real_signal`
- `rerun_required_artifact_or_reporting_issue`

## Current Feature Recovery Status

| Feature group | Current status | Notes |
|---|---|---|
| Masked-v2 baseline | Failed | No paper-ready candidate. |
| Group 1 non-VIX index/context subset | Active uplift test | Parity passed; current goal is testing uplift. |
| Group 1 full set with VIX changes | Blocked | Needs paired pre-window/warm-up evidence or changed trace construction. |
| Group 2 candidate geometry/moneyness | Rejected for now | Instrumented attempt002 says no real signal. |
| Group 3 Greeks/IV | Blocked | Quote/source/repair-input parity not proven. |
| Group 4 quote/liquidity/spread | Blocked | Cross-vendor quote source differences remain. |
| Group 5 volume/OI | Blocked | Volume may be traceable, but OI not live-proven in current recorder days. |
| Group 6 raw microstructure | Blocked | Exceptional add-back unjustified while normalized quote/spread parity fails. |

## Current Training/Gate Setup

Training goals use:

- Contract: `protocol101-live-v2-microstructure-masked`
- Transform: `mask_vendor_sensitive_option_quote_greek_microstructure`
- 15-month accepted training-scope corpus
- 5-fold chronological expanding-window CV
- 1-session embargo
- Strict one-account serial replay
- All 7 menu-v2 fixed-exit shapes
- Bounded HGB/tabular first
- No neural/MLP unless a later evidence note justifies it

Key training artifacts:

- [15-month training scope acceptance](/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_live_v2_microstructure_masked_15mo_training_scope_acceptance/summary.json)
- [15-month training design](/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_live_v2_microstructure_masked_15mo_training_design/summary.json)
- [15-month training runner](/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_live_v2_microstructure_masked_15mo_training_runner/report.md)

Gate set:

- G1 Profitability
- G2 Beats no-skill/null
- G3 Beats heuristic
- G4 Drawdown
- G5 Seed robustness
- G6 Era guard
- G7 Frequency
- G8 Calibration
- G9 Confirmation seed

See:

- [Stage-1 objective and gates proposal](/Users/gduby/Documents/autoresearch-trading/v4/docs/PROTOCOL101_STAGE1_OBJECTIVE_AND_GATES_PROPOSAL.md)

## What We Need Help Thinking Through

The user wants an AI reviewer to look for better approaches or hidden flaws. Useful review questions:

1. Is the static-ladder model-universe + separate tradability guard policy a sound compromise, or does it create label/fill/model mismatch risk?
2. Are we being too conservative by blocking quote/Greek/liquidity alpha, or appropriately conservative given cross-vendor source drift?
3. If Group 1 non-VIX also fails, what is the best next path?
   - Repair VIX warm-up trace history?
   - Collect more IBKR recorder evidence?
   - Evaluate same-vendor live/historical model feed?
   - Build a new model family/label strategy on the masked baseline?
   - Move to learned exits despite weak entry evidence?
4. Does the current G1-G9 gate set correctly distinguish "no signal" from "signal but poor exits"?
5. Should quote/liquidity/Greeks be added back through a same-vendor data plane rather than cross-vendor approximation?
6. Are the rejected Group 2 conclusions credible, or could concentration/no-skill metrics be too harsh for a convex 0DTE strategy?
7. Is the current feature-recovery ladder too linear? Should interactions between safe feature groups be tested only after individual groups pass, or is there a principled way to test pooled feature subsets without overfitting?
8. Is the current 09:32 ET / 359-row convention appropriate, or should paired traces carry longer pre-open/pre-window context so VIX/SPX lag features can be evaluated fairly?
9. Are there safer ways to use raw bid/ask/spread/volume/OI as guards or context without making them model alpha?
10. What additional evidence would be required before allowing a candidate into IBKR shadow/paper-readiness validation?

## High-Level Forks After The Active Goal

Once the Group 1 non-VIX uplift goal finishes:

### If It Produces `eligible_offline_candidate`

Next likely step:

```text
Run recorder/parity validation of that actual candidate on IBKR recorder days and matching historical days.
```

Do not claim paper readiness yet.

### If It Produces `stage2_learned_exits_candidate`

Next likely step:

```text
Train/test learned exits using the entry candidate as fixed input, with strict diagnostics and no gate softening.
```

### If It Produces `group1_nonvix_rejected_no_real_signal`

The safest non-quote feature recovery branch is likely exhausted for now. Next decision may be:

- Repair full Group 1 VIX warm-up/history evidence.
- Investigate Group 3 Greeks/IV through a same-source/internal-Greek parity path.
- Investigate Group 5 volume/OI evidence collection.
- Evaluate a same-vendor live/historical model data plane.
- Rethink labels/model family under the masked baseline.

### If It Produces `rerun_required_artifact_or_reporting_issue`

Fix the runner/artifact issue before any further modeling decision.

## Important Warnings For The Reviewer

Do not treat "historical PnL went down" as proof the path is wrong. Much of the old edge may have been non-causal, unreproducible, or tied to incompatible runner semantics.

Do not treat "features are masked" as final. The plan is to recover useful features through parity plus uplift evidence.

Do not treat feature groups as useless merely because earlier ladder packets rejected them before parity infrastructure was repaired.

Do not recommend paper-submit yet. No current candidate is paper-ready at this handoff point.

Do not recommend loosening gates to get a model. The user wants a model that can actually survive IBKR paper validation, not another beautiful historical curve.

## Minimal File Bundle For External Review

If uploading only a small set of files to an external AI, upload these first:

1. `/Users/gduby/Documents/autoresearch-trading/v4/docs/PROTOCOL101_CURRENT_PHASE_AND_TRAINING_HANDOFF_2026_07_08.md`
2. `/Users/gduby/Documents/autoresearch-trading/v4/docs/PROTOCOL101_FAIR_CONTRACT_TRAINING_AND_FEATURE_RECOVERY_PLAN_2026_07_08.md`
3. `/Users/gduby/Documents/autoresearch-trading/v4/docs/PROTOCOL101_STAGE1_OBJECTIVE_AND_GATES_PROPOSAL.md`
4. `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_live_v2_option_quote_source_policy_resolution/report.md`
5. `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_live_v2_static_ladder_boundary_stable_policy_audit/report.md`
6. `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_live_v2_group2_geometry_uplift_attempt002/routing_decision.json`
7. `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_live_v2_group2_geometry_uplift_attempt002/report.md`
8. `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_live_v2_group1_index_context_parity_resolution/report.md`
9. The active Group 1 non-VIX result once complete:
   `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_live_v2_group1_nonvix_uplift_attempt001/report.md`
10. This handoff:
   `/Users/gduby/Documents/autoresearch-trading/v4/docs/PROTOCOL101_CURRENT_ENDEAVOR_EXTERNAL_AI_HANDOFF_2026_07_09.md`

If source code is needed, start with:

- `/Users/gduby/Documents/autoresearch-trading/v4/live/protocol101_feature_contract.py`
- `/Users/gduby/Documents/autoresearch-trading/v4/dataset/spxw_0dte_neural.py`
- `/Users/gduby/Documents/autoresearch-trading/v4/scripts/run_protocol101_static_ladder_boundary_stable_policy_audit.py`
- `/Users/gduby/Documents/autoresearch-trading/v4/scripts/run_protocol101_group2_geometry_uplift_attempt.py`
- `/Users/gduby/Documents/autoresearch-trading/v4/scripts/run_protocol101_group1_nonvix_uplift_attempt.py`
- `/Users/gduby/Documents/autoresearch-trading/v4/scripts/run_protocol101_stage1_bounded_hgb_search.py`

## Final Summary

The project is trying to earn its way back to hill climbing by making the live and historical games match. The original high-PnL historical model is no longer trusted as a live benchmark. The current fair contract masks vendor-sensitive microstructure, then adds features back only if they pass parity and uplift. Candidate-universe parity was repaired by source-aligning SPX/VIX context and adopting a static-ladder model universe with separate tradability guards. Group 2 geometry/moneyness passed parity but failed uplift and is rejected for now. Group 1 full context remains partially blocked by VIX warm-up history, but the non-VIX subset passed parity and is the current active uplift test. No model is paper-ready yet.
