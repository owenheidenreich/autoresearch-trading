# Seat 2 — ML/statistics delta-scoped review attempt 002

## Verdict

`DELTA_REVIEW_CLEAN`

No in-scope blocking issue remains in FT2-DELTA-B3 or its direct
variance/MDE and semantic-receipt consequences. This is a delta-review verdict,
not approval or routing to FT2-21. The required next state remains
`STOP_FOR_OWNER_DECISION`.

## Scope and safety

- Baseline commit:
  `a7602fdcce541589440b4aa2bc0bd0e2be6d1bbd`.
- Repair diff manifest:
  `v4/audit/autoresearch/protocol101_ft2_20_delta_scoped_review_fixes_attempt001/diff_manifest.json`.
- Diff-manifest SHA-256:
  `2aa1ff2d307d9eccdd483484cde882dee230cc2703869b05dc8561eb41f18ee7`.
- All 41 manifest `after_sha256` values reproduce with zero mismatch.
- Review was restricted to manifest-declared changed files/JSON pointers and
  the direct RLAC, variance/MDE, path/hash, test, and receipt consequences.
- No other attempt-002 seat directory was read.
- No broker, recorder, paid-data, live, protected/outer/holdout, real-training,
  simulator-mutation, runtime, paper-default, promotion, launchd, or order path
  was used.
- All durable writes were confined to this Seat 2 directory.

## FT2-DELTA-B3 assessment

### One-way RLAC remains genuinely noncircular

The RLAC at
`realized_label_audit_composer_spec.json` still constructs its target
population and labels before any model head is fit. Its allowed ancestors are
frozen FT2-04 labels, causal time-t intent masks, fit-role nested CDFs, frozen
anchor thresholds, and deterministic RLAC transforms/tie-breaks. Model and
runtime-composer ancestors remain forbidden. Fit and disjoint calibration roles
build their own target tables, freeze heads/calibrators/manifests, and only then
run the runtime composer.

The independent T1 fixture passed all five assertions: targets compute with the
composer stubbed, target bytes are identical under a composer swap, gate
metrics reproduce, positive provenance is accepted, and a planted
composer-ancestor circularity is rejected. This supports the contract
inspection; it is not being used as a substitute for inspecting the live
specifications.

### WAIT target and population are now identical and testable

`forecast_heads.json` and `calibration_spec.json` contain byte-identical
`target_authority` objects. Both identify:

```text
path = realized_label_audit_composer_spec.json
json_pointer = /targets/WAIT_head
sha256 = adcbec44bc254fe9a968562df77ca6a5736aa5d39d9f8e69dcf477a378df4f09
formula = y_wait(t)=1 if WAIT_justified(t) else 0
scalar_U_label_threshold_allowed = false
zero_intent_eligible_minutes = excluded_and_counted
```

The referenced RLAC pointer exists and defines the same formula over all
label-auditable governed flat minutes having at least one time-t
intent-eligible contract. Its `WAIT_justified` verdict is the absence of any
exact contract satisfying the complete conjunction: time-t intent eligibility,
complete required labels, every active realized guardrail, and realized
fee-cleared upside. The forecast target explicitly imports that conjunction.
The calibration outcome imports the same target; its denominator is the same
RLAC population, and it explicitly excludes and counts both zero-intent and
label-incomplete minutes. The former incompatible scalar-`U_label` threshold
definition is explicitly forbidden.

The bounded checker independently passed the exact shared authority object and
population-denominator checks.

### Marginal and selected-subset claims remain correct

The hard conformal claim remains marginal over the complete RLAC regret
population: every label-complete time-t intent-eligible contract on every
included minute, including RLAC-WAIT minutes. The runtime-composer-selected
exact-contract subset remains a post-hoc, report-only conditional diagnostic.
No conditional-coverage guarantee is claimed, and selected-subset results may
not refit, recalibrate, relabel, or alter target membership. A selected-subset
collapse still routes to owner decision.

## Variance and MDE direct consequences

The changed oracle results correctly flow into the planning-only census
variance and MDE artifacts.

- All 63 `session_variance_components.csv` rows independently reproduce from
  session-level `oracle - P5` PnL differences in
  `oracle_session_results.csv`.
- All 504 session-cluster MDE rows reproduce the frozen
  `z_(0.975)+z_(0.80)` formula. Maximum absolute numerical differences were
  `4.55e-13` for mean-session MDE and `2.91e-11` for total-PnL MDE.
- The exact overall/all `hold_to_forced_flat` planning row is 45 sessions,
  mean `$2,042.311111111111/session`, sample variance
  `8,346,744.491919192`, and sample SD `$2,889.073292929619`.
- `mde_spec.json` pins those rebuilt values and uses the mean only as a
  deliberately generous planning ceiling. It remains explicitly not a target,
  not expected-edge evidence, not a pilot MDE, and not automatic spend
  authorization.
- Actual tranche gating still requires training-side paired
  candidate-minus-P5 and candidate-minus-matched-random session deltas,
  statistic-specific SD/LRV/ESS, analytic and block-power MDEs, both fee
  trajectories, and the frozen multiplicity ledger. Census variance may not
  substitute for that pilot evidence.

T6 also reproduced the frozen studentization fixture, all three terminal
examples, and 22/22 FT2-08 validation checks.

## Semantic path/hash and receipt re-seal

The current generic semantic `{path, sha256}` and
`{source, source_sha256}` scan reports zero mismatches across the active
FT2-04/05/08/10/11 JSON packets. Direct historical `repair_of` pins were also
verified against the exact baseline commit bytes for FT2-05, FT2-08, FT2-10,
and FT2-11.

The five current packet receipts all pin authority
`d115b953d8959fe777923ca5c1e375246754a181847ae77b57d37d24f0a279ca`.
All 157 receipt-declared deliverables reproduce:

| Packet | Receipt SHA-256 | Deliverables | Mismatches |
|---|---|---:|---:|
| FT2-04 | `b4b29dcf9b96b0172adb614bf0aa975ab1c86ef9c92ba2d91423a14c4b52cffe` | 7 | 0 |
| FT2-05 | `ecaa16c923d4834c314816bca471597e09b9668ec5f1fe76d956be9c9d97d168` | 117 | 0 |
| FT2-08 | `c4c9c729c8d19ae76f827ed280f66e7ee1143f7f51fd5b753615d6d968adb726` | 14 | 0 |
| FT2-10 | `b0504ea7d78c041ed066b238229fd5bdda5584da2d1a0854adeefba91b63a3a2` | 9 | 0 |
| FT2-11 | `3cb4693185bc2d1c77c5ecea38817766822b684b33679ba021c0446fade1534c` | 10 | 0 |

The FT2-08/10/11 receipts pin the actual rerun002 raw receipt bytes at
`17cc9983c38ef4c75b8fece0cd67d4662e559f84a27de8eba2a84a34f85d96ac`.
The active census builder pins the current authority and passes its own
authority verifier. The producer aggregate receipt's canonical self-hash also
reproduces exactly.

## Mechanical gates

| Gate | Independent result |
|---|---|
| Bounded-fix checker | pass, 20/20 |
| Targeted census pytest | pass, 12/12 |
| T1 RLAC one-way fixture | pass, 5/5 |
| T3/T4 census reconciliation | pass, 11/11 |
| T5 graph structure (checker) | pass, 47 nodes / 104 edges / 47 reachable |
| T6 regression | pass, 4/4; FT2-08 22/22, 3 terminal fixtures, SE fixture |

T6's validator has a fixed write to the reviewed FT2-08 `validation.json`.
A seat-local `sitecustomize.py` redirected that exact write and subsequent read
to `t6_ft208_validation.json`. The private recomputation and reviewed file are
byte-identical at SHA-256
`d0fd65b55ed38d88d82541e7a2fe1ecec3903b0bb61c6bc41e3c24368cd5fd6e`;
the reviewed artifact was not mutated.

## Documentation-only observation

The duplicated historical authority amendment label `A4` remains unchanged.
It is documentation-grade and outside the mechanical repair's blocking
requirements; it does not change this verdict.

## Route

`DELTA_REVIEW_CLEAN` -> `STOP_FOR_OWNER_DECISION`.

This review does not approve or route FT2-21.
