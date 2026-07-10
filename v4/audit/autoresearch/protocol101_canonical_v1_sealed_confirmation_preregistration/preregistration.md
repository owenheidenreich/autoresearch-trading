# Protocol101 Canonical v1.4 Sealed-Day Confirmation Battery — PREREGISTRATION

Preregistered: 2026-07-09, before any sealed session exists.
Amended: 2026-07-10 (owner-directed, sealed set still EMPTY — no sealed data
existed at amendment time): sealing rule v2 introduces a development slice
(2026-07-13, 2026-07-14 CPI session, 2026-07-20) for rehearsal and repair;
sealed-count requirement adjusted for the smaller sealed set; rehearsal gate
added. FOMC sessions (07-28/29) remain sealed. No pass criteria, thresholds,
epsilons, or battery contents changed.
Status: frozen. Executing this battery requires zero design decisions.

## Purpose

One-shot confirmation that the canonical minute-game transform and the frozen
v1.4 selection contract — designed and certified entirely on burned days
(2026-06-30/07-01/07-02) — reproduce on fresh sealed recorder days that no
tool or human inspected during design. A pass upgrades the transferability
claim from design-grade to confirmed and unlocks gate-graded Stage-1 training.

## Frozen inputs (verify hashes before running; any mismatch = abort)

| Artifact | sha256 |
|---|---|
| selection_contract_v1_4.json (attempt005) | `602fd8eff564a059ad114dd051b6793cb50bcc50269c83b81bdfe25aa119ef57` |
| reconciliation attribution_summary.json (epsilons) | `7919a68854169fb2900e4e72785da897777415fb93dac2502e2a44c168fadcd2` |
| L0/L2 design-audit preregistration.json (L0 thresholds) | `a3021f4945603bab236cc0ea28c34fb9304299d3e0d24fe6fbc15efee9b71614` |
| canonical_feature_definition.json (22 features) | `628b162b66fcce50c0bd08e7bd34606ecf9a4a95afc0cd566f12bb79429d5bb8` |
| sealing_rule.json (sealed-day assignment, rule v2) | `a9fb2c25e0b31c42b6846925f198930f44594cf7567a0970c767169ed468a89e` |

All epsilons, thresholds (including the v1.4-recalibrated band-probe
thresholds recorded inside selection_contract_v1_4.json), k = 2 dead-band and
slot-margin gates, deterministic ordering, score-independent nearest-ATM
fallback, near-tie definition, and the $3.00/contract materiality line are
taken from these artifacts verbatim. Nothing is remeasured or re-derived.

## Evidence requirements (before the battery may run)

- Sealed sessions: >= 9 sessions sealed on arrival per sealing rule v2
  (expected 11: 07-15..17, 07-21..24, 07-27..30), seal manifests green
  (check mode passes).
- Regime coverage: >= 1 high-volatility/event session IN THE SEALED SET
  (the 07-28/29 FOMC sessions, or realized SPX daily range in the top
  tercile of the trailing year). If absent: routing =
  `insufficient_regime_coverage`; keep collecting. Do not run a partial
  battery.
- **Rehearsal gate**: before the sealed battery may run, the IDENTICAL
  battery (same code, same frozen inputs, same pass criteria) must have been
  run on the development days (2026-07-13, 2026-07-14, 2026-07-20) and
  passed. Rehearsal results are design-grade and confer no confirmation
  claim. A rehearsal failure routes to diagnosis and repair on development +
  burned + validation days only; any repair produces a NEW owner-signed
  frozen contract version, which must itself pass rehearsal before taking
  the sealed exam. The sealed set is never spent on a recipe that has not
  passed rehearsal.
- The paired historical (Databento/ThetaData) sessions for the same dates
  built with the SAME source-aligned pipeline used for burned days.
- 2026-07-10 (validation), 2026-07-13/14/20 (development), and all burned
  days are EXCLUDED from confirmation scoring.

## The battery (identical to attempt005 semantics, one shot, all sealed days)

1. **L0 field divergence + bias**: all 22 canonical features, pooled AND
   per-day, stratified by moneyness band / time-of-day / opportunity tercile;
   admit thresholds from the L0 design preregistration (coverage >= 0.80,
   standardized p95 |drift| <= 0.10 under historical_reference_stddev_with
   _floor_v1, bias |rho| <= 0.05 with CI upper <= 0.15 — CIs now day-clustered,
   which >= 10 sealed days finally makes meaningful).
2. **L2 source discriminator**: bounded HGB (depth 3, <= 200 trees) +
   standardized logistic, leave-one-day-out over sealed days; pass AUC <= 0.55;
   null control (odd/even minutes, historical plane) in [0.45, 0.55];
   positive control (raw bid/ask/spread/quote_age) >= 0.80 with price-only
   diagnostic reported.
3. **L1/L3 19-probe battery** under frozen v1.4 semantics: 9 fixed probes
   (band-aggregate momentum and internal-IV with abs_offset <= 19 restriction)
   + 10 transfer probes (5 subsets x logistic/HGB, retrained on burned-day
   historical rows only — sealed rows are never training data — scored on
   sealed paired rows).

## Pass criteria (absolute; no reduction-vs-baseline gates on sealed days)

- Every L1 non-random probe and every L3 subset/model: action agreement >= 0.98.
- Post-gate selected-slot agreement >= 0.99 per probe/subset on mutually
  confident enter decisions (denominator < 30 => descriptive_only for that
  probe, never a run-level verdict by itself).
- Material (>$3.00/contract) true-score reorderings: 0 expected; any
  occurrence is classified, counted, and fails the run only if the count
  exceeds 2 per 10 sealed sessions (tail allowance at burned-day rates).
- Split-confidence and split-action classes reported with the preregistered
  materiality definition (>= 30 material minutes across the battery = material;
  note burned-day baseline was 34 over 3 days — report per-session rate).
- No low-n concentration claim used as rejection evidence (denominator < 30 =>
  descriptive_only).
- Per-day reporting mandatory; a pooled pass may not hide a failing
  high-volatility day: the event-day sessions must individually satisfy the
  action-agreement gate.

## Preregistered watch-item expectations (from burned-day evidence)

- internal_iv_expansion_compression: full gate again; expected ~0.996 action
  agreement post-v1.4 (burned-day value 0.9963).
- Split-action material minutes: burned-day rate ~11/session-triplet;
  concentrated in CDE subsets. Materially worse on sealed days => investigate
  before interpreting the headline verdict.
- internal_delta_geometry slot agreement: burned-day 0.9879 (closest passing
  number). Degradation below 0.98 => probe-level finding, not auto-fail.

## Routing outcomes

- `canonical_v1_4_sealed_confirmed` — all gates pass. Unlocks gate-graded
  Stage-1 training per the Stage-1 training design doc. The sealed days are
  now BURNED (they have been read); mark them so in the assignment manifest.
- `canonical_v1_4_confirmation_failed_<component>` — name the failing layer
  (l0_feature / l2_discriminator / probe name). ALL diagnosis happens on the
  burned + validation set only; the failed sealed days are burned by the run
  and NEW sealed days must accumulate for any retest. No repair may touch the
  frozen contract without an owner-signed v2 cycle.
- `insufficient_regime_coverage` — keep collecting; battery not run.
- `insufficient_artifacts` — hash mismatch or missing paired sessions; fix
  inputs, battery not run.

## Prohibitions

No threshold/epsilon/feature adjustment at run time. No partial or exploratory
peeks at sealed data before the full battery executes. No second run against
the same sealed days. No training on sealed rows, ever. Standard scope
prohibitions (no broker/paid/promotion/runtime/launchd changes) apply.
