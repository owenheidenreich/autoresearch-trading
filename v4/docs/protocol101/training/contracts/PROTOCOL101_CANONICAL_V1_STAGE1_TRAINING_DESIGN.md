# Protocol101 Scoped Canonical Stage-1 Training Design - SIGNED FINAL

Original draft: 2026-07-09. Scoped amendment: 2026-07-25.
Signed: 2026-07-25. G8 revision incorporated: 2026-07-26.

This amendment replaces the obsolete requirement for an all-family sealed-day
confirmation with the owner-approved scoped synchronization decision. Execution
remains blocked until:

1. `PROTOCOL101_SCOPED_SYNCHRONIZATION_DECISION_2026_07_25.md` is signed;
2. `PROTOCOL101_TRADER_CHARTER.md` is signed;
3. this amended design is signed; and
4. the exact-contract readiness preflight and `plumbing_smoke_only` packet pass.

The G4/holdout revision and G8 calibration revision are signed and complete.

Current execution state: the exact-contract preflight, disposable H3 plumbing
smoke, independent smoke validation, runner freeze, and deterministic gate
aggregator are complete. Historical H0-H3 execution is governed by the
subsequent frozen artifacts and the signed G8 revision.

## Objective

Root reference: `PROTOCOL101_TRADER_CHARTER.md`.

Maximize fee-adjusted net PnL through the strict one-account serial simulator:
$10,000 starting cash, one contract, premium plus fee affordability enforced,
5% of session-starting-equity daily circuit breaker, forced flat 15:55 ET, and
$3.00 round-trip fee overlay with $2.60/$4.00 sensitivity. G1-G9 and signed G4
v2 remain binding.
Win rate is diagnostic, never the optimization target.

Stage-1 trains entries against all seven fixed menu-v2 exit shapes. Learned
exits remain Stage-2 and require their own owner-signed objective after the
entry evidence satisfies the routing rule below.

Every packet also reports the charter diagnostics: four-bucket outcome
distribution, harvest ratio, underwater duration, worst day, and daily-breaker
events.

## Feature Contract

- Contract ID: `protocol101-scoped-canonical-stage1-v1`.
- Model-facing features are exactly the 17 fields authorized in
  `PROTOCOL101_SCOPED_SYNCHRONIZATION_DECISION_2026_07_25.md`:
  12 certified non-VIX context features, three near-ATM D composites, and
  internally computed `E.bs.delta` / `E.bs.gamma`.
- Quarantined from alpha: all Family C direct per-slot price paths,
  `E.bs.iv` and IV expansion/compression, VIX changes, raw quote/size/age,
  volume/OI, and vendor Greeks.
- Raw quote fields remain available for guards, fills, labels, PnL, and audit.
- The legacy masked-v2 transform is an upstream data identifier, not the
  Stage-1 model feature contract. A generic `market_window + option_ladder`
  feature matrix is forbidden.
- D aggregates use only `abs_offset <= 19`. Wing slots remain in the candidate
  universe, guards, labels, fills, and audit.
- Internal delta/gamma use the same deterministic canonical calculation on
  both historical and live paths: tick-quantized mid, SPX, strike, time to
  expiry, `r=0.05`, `q=0`.

## Decision Semantics

Models emit payoff / return-on-premium scores. Selection inherits the v1.4
mechanics:

- k=2 action dead-band;
- k=2 slot-margin gate;
- deterministic ordering by score, strike index, then right index;
- score-independent nearest-ATM fallback when no slot preference clears the
  noise floor.

The old v1.4 numeric epsilons came from classifier/probe score scales and may
not be copied onto a payoff regressor. For each fold/model/seed:

1. score training-only calibration rows at 0x and 1x measured divergence noise;
2. set epsilon to the p95 absolute paired score drift;
3. freeze epsilon and the action threshold;
4. only then score that fold's unseen validation sessions.

No validation/test value may influence epsilon or the action threshold.
Feature parity does not certify an arbitrary fitted model: every frozen
candidate must later pass candidate-specific no-order decision-shadow transfer.

## Robustness Machinery

All runs require:

- measured divergence-noise injection, 1.0x primary with 0x/0.5x/2.0x
  diagnostics;
- the exact boundary-stable intersection guard policy;
- labels and PnL gated on the pessimistic fill rung, with the remaining fill
  rungs reported as an edge band;
- strict serial replay with one account and no overlapping headline trades;
- five chronological expanding folds and a one-session embargo;
- no protected holdout, recorder, parity, or confirmation dates in training.

A candidate whose gate result depends on removing measured noise is rejected as
drift-mining.

## Prior Campaign Guidance

`PROTOCOL101_PRIOR_CAMPAIGN_DISTILLATION.md` and
`PROTOCOL101_PROTOCOL_FARM_LINEAGE_2026_07_19.md` are methodology inputs, not
inherited performance evidence.

Binding lessons:

1. Do not repeat an unchanged mechanism on the do-not-retest list.
2. Start with shallow/bounded trees and chronological tests.
3. Keep entry learning separate from lifecycle/exit learning.
4. Treat opening structure as the strongest prior hypothesis family, not as a
   proven feature in the current contract.
5. Exact opening-gap / first-15 acceptance features require a separate parity
   audit and owner-signed amendment before they may enter alpha.
6. Overlap, cached paths, narrow samples, and pre-parity curves cannot support
   a current edge claim.

The current 12 context features overlap opening structure only partially. This
design does not falsely claim to reproduce the April feature vector.

## First Four Hypotheses

| ID | Features | Purpose |
|---|---|---|
| H0 | 12 certified non-VIX context features | Parity-stable context baseline |
| H1 | H0 + three near-ATM D composites | Implied-move and put/call skew |
| H2 | H0 + internal delta and gamma | Deterministic Greek geometry |
| H3 | H0 + D + internal delta/gamma | Full authorized 17-feature set |

These four make up the complete first Stage-1 search. All four run, one batch at
a time, regardless of interim results. No extra subset, interaction, threshold,
or feature engineering may be added without a preregistered owner-signed
amendment.

## Training Structure

- Corpus registry: 301 accepted pass-only sessions. The frozen folds use 271
  unique sessions after the 30 protected-holdout sessions are excluded,
  exclusively through the governed loader.
- CV: five chronological expanding-window folds, one-session embargo, frozen
  fold boundaries.
- Protected holdout: 2025-05-16 through 2025-06-30 remains locked and absent
  from training and threshold selection.
- Model family: bounded HGB/tabular first. Neural or sequence work requires a
  separate learning-curve evidence note and owner approval.
- Target: payoff / return-on-premium, never win probability.
- Calibration: G8 uses an out-of-fold confidence map derived from payoff
  scores and remains required reporting for every seed. Under the owner-signed
  2026-07-26 revision it is report-only until confidence controls abstention,
  contract selection, sizing, routing, exits, or another trading behavior. It
  does not change the payoff training target.
- Batch: seven exit policies x three seeds, evaluated on all five folds,
  followed by at most one preregistered conservative batch.
- Ledger: every attempt is registered before results are inspected.
- G2/G3: nulls and heuristics must be recomputed under this exact feature,
  candidate, guard, fill, selection, and simulator contract. Older masked-v2
  and broad-canonical bands are provenance inputs, not final gates.
- Serial simulator:
  `protocol101_serial_simulator_v4_account_continuity_fee_reserve`. It realizes
  each session's final pending trade before the next session receives buying
  power and reserves the $3 fee during affordability checks. Any packet naming
  simulator v2/v3 or lacking the fee reserve is stale for this generation.
- Exact current references:
  `protocol101_scoped_canonical_stage1_null_canary` and
  `protocol101_scoped_canonical_stage1_heuristic_baseline`. The frozen G3
  baseline is policy 5 (policy 6 tied; lower index wins the deterministic
  tie-break) at `$4,592` pooled fee-adjusted PnL. It is a comparison bar, not
  an eligible candidate: only 3/5 folds are profitable and its weak-fold
  equity violates signed G4 v2.

The legacy `run_protocol101_stage1_bounded_hgb_search.py` is retained only for
historical masked-v2 evidence. New runs must use the exact-contract canonical
runner whose preflight asserts this allowlist.

## Routing

- `eligible_offline_candidate`: G1-G7 pass, G8 is reported, independent
  cross-hypothesis selection is frozen, and the candidate then passes G9,
  one-shot holdout, candidate freeze, and candidate-specific recorder/shadow
  transfer.
- `stage2_learned_exits_candidate`: G2 and G5 demonstrate real entry signal,
  but fixed exits fail with path evidence that a causal exit model can
  plausibly recover. This authorizes drafting, not executing, Stage-2.
- `rejected_no_real_signal`: no accepted null-relative entry edge.
- All H0-H3 reject: the scoped 17-feature contract has no accepted Stage-1
  alpha under this frozen search. Review the separately parity-tested opening
  structure proposal, a same-vendor data plane, or another strategy class.
  Do not loosen gates.

## Owner Checklist

- [ x] Scoped synchronization decision accepted.
- [ x] Exact 17-feature allowlist and quarantined-feature list confirmed.
- [x ] Four-hypothesis set confirmed as complete.
- [ x] 1.0x measured divergence noise is mandatory.
- [ x] Fold-local training-only epsilon calibration confirmed.
- [ x] Intersection guards and pessimistic-fill gating confirmed.
- [ x] Stage-2 unlock criteria confirmed.
- [ x] Candidate-specific no-order shadow transfer required before paper.
- [ x] Execution blocked until signatures, exact-contract preflight, and
      plumbing smoke pass.

Owner signature: Owen Heidenreich

Date: 07/25/2026
