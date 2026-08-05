# SUPERSEDED — DO NOT SIGN OR EXECUTE

This July 9 design is retained only as provenance. Its five-family H0-H4
contract includes Family C and internal IV, relies on a sealed-day prerequisite,
names stale null/heuristic values, and predates the serial-accounting repair.
It was superseded on 2026-07-25 by:

`PROTOCOL101_CANONICAL_V1_STAGE1_TRAINING_DESIGN.md`

Only the successor's exact 17-feature H0-H3 contract is eligible for owner
sign-off or new training.

# Protocol101 Canonical v1 Stage-1 Training Design — HISTORICAL

Drafted: 2026-07-09, before sealed-day confirmation results exist.
Execution is BLOCKED until: (1) `canonical_v1_4_sealed_confirmed` routing from
the sealed confirmation battery (preregistration sha256
`13176551ed26f4a42f4a9fce204f3059591a55d24d2833caf1c29b8401ed06b9`, amended
2026-07-10 to add the development-day rehearsal gate while the sealed set was
still empty), and
(2) the owner-signed G4/holdout gates revision
(`PROTOCOL101_G4_HOLDOUT_REVISION_2026_07_19.md`: Calmar >= 1.0 pooled +
$5,000 per-fold equity floor, calibrated by the forced-oracle and
slot-skill-frontier measurements) is signed. Preregistering now removes the
"designed after seeing what passed" critique.
Status note: the canonical v1.4 rehearsal battery PASSED on fresh days
2026-07-10/13/14 incl. the CPI session (design-grade); the sealed exam
remains the gate for execution.

## Objective

Root reference: `PROTOCOL101_TRADER_CHARTER.md` — the owner's plain-language
definition of the trader being built (convex hunter, flat droughts, SPY
floor, 5%-of-equity daily circuit breaker, survival floor). Every gate
below traces to a charter commitment. New charter-mandated metrics for
every Stage-1 packet: harvest ratio (realized/peak available PnL) and
underwater-duration (longest time below high-water mark), both
report-only. Daily circuit breaker set to 5% of current equity in the
simulator config.

Unchanged from the owner-approved Stage-1 objective: maximize fee-adjusted net
PnL through the strict one-account serial simulator
(`v4/model/protocol101_serial_simulator.py`; $10k, single contract, forced
flat 15:55 ET, $3.00/contract fee overlay with $2/$5 sensitivity), subject to
gates G1–G9 with the revised G4. Win rate is a diagnostic, never an objective.
Stage-1 trains ENTRY only, paired with the 7 fixed menu-v2 exit shapes
(`v4/dataset/spxw_0dte_neural.py` label_policies). Learned exits are Stage-2
and require their own owner-signed objective document, unlocked only by the
evidence packet defined below.

## Feature contract and decision semantics (frozen, inherited)

- Features: the 22 canonical v1 admitted features
  (canonical_feature_definition.json sha256 `628b162b…9d5bb8`) plus the 12
  certified Group 1 non-VIX context features. Nothing else. No VIX changes,
  no raw quote/size/age alpha, no vendor Greeks, no volume/OI.
- Decision semantics: models emit scores; actions and slot selection pass
  through the frozen v1.4 selection contract verbatim (sha256
  `602fd8ef…ef57`): k=2 noise-aware action dead-band, k=2 slot-margin gate
  with frozen epsilons, deterministic ordering (score, strike_idx, right_idx),
  score-independent nearest-ATM fallback. Trained policies inherit transfer
  stability by construction; it is not re-proven per candidate.
- Design rules paid for in the probe program:
  1. No entry or side rule may threshold or sign a low-variance signal near
     its distribution bulk (the IV-saga rule). Score constructions must be
     checked against this before training, not after.
  2. Any band-aggregate score uses near-ATM bands only (abs_offset <= 19).
     Wing slots stay in the universe/guards/labels/fills/audit but never in
     aggregate alpha scores (measured 20x ATM->wing IV drift gradient).

## Robustness machinery (both mandatory in every run)

- **Divergence noise injection** (work-queue Step 8 module): training-time
  perturbation of canonical features drawn from the measured cross-plane
  divergence distributions (L0 audit divergence_distributions.parquet),
  conditioned on moneyness band. Primary level 1.0x measured; 0.5x and 2.0x
  as sensitivity arms. A candidate whose gate outcomes flip between 1.0x and
  0x injection is rejected as drift-mining.
- **Intersection guards** (work-queue Step 9 module): pessimistic tradability
  calibrated to the measured ~4.7% IBKR-vs-historical candidate asymmetry.
  All training, CV, and gate evaluation run under intersection guards.
- **Fill ladder**: labels and PnL computed at mid / mid+0.25*spread / touch;
  all gates evaluated on the pessimistic rung; the other rungs are reported
  as the edge band. Fees $3.00 primary (schedule-grounded), $2.60/$4.00 sensitivity.

## Prior-knowledge inputs (preregistered before results exist)

Two mined documents are formal inputs to this design:
`PROTOCOL101_PRIOR_CAMPAIGN_DISTILLATION.md` (April edge ledger,
do-not-retest list, family mapping) and
`PROTOCOL101_PROTOCOL_FARM_LINEAGE_2026_07_19.md` (transcript-tier farm
lineage). Binding consequences:

1. **The do-not-retest list is a constraint**: no Stage-1 attempt may
   re-implement an unchanged falsified mechanism (28 items; e.g.
   always-put default, vwap_reclaim_state as premise, post-hoc score
   coverage thresholds). A retest requires a materially different causal
   variable, target, or game, stated in the ledger entry.
2. **H0 is dual-role, not a mere control.** April's strongest surviving
   edge — opening structure (opening gap + first-15 acceptance/range,
   PF 1.32 -> 1.54 gated, bootstrap CIs excluding zero) — expresses
   through exactly H0's B/G1 features. An H0 gate-pass is a legitimate,
   prior-supported outcome, not a harness anomaly. H0 failing while
   H1-H4 pass would conversely indicate the edge needs option-surface
   information — also informative.
3. **Directional priors:** the honest April baseline was directional
   V0 PF 1.132 over 780 OOS days; the current heuristic scan's best
   reading is put/call skew (+$15,953, z=2.15, gates-fail). Both point
   to H0 (side context) and H2/H4 (skew composites) as the highest-prior
   hypotheses. Priors inform expectations, never thresholds.

## The five preregistered hypotheses (mirroring the certified L3 subsets)

| ID | Features | Role |
|---|---|---|
| H0 | A/B + Group1 non-VIX | Dual role: harness control AND prior-supported opening-structure candidate |
| H1 | H0 + Family C (per-slot mids/momentum/path) | Option price dynamics |
| H2 | H0 + Family D (straddle/skew composites) | Implied-move/skew |
| H3 | H0 + Family E (internal IV/delta/gamma) | Greek structure |
| H4 | H0 + C + D + E | Pooled (one look, no subset search) |

These five are the complete Stage-1 hypothesis set. No additional subsets,
interactions, or feature engineering without an owner-signed amendment. H0–H4
are all run regardless of interim results (no early stopping on peeking).

## Training/validation structure (inherited from Stage-1 law)

- Corpus: 15-month accepted non-holdout sessions via
  `v4/model/protocol101_governed_loader.py`, exclusively.
- 5-fold chronological expanding-window CV, 1-session embargo, frozen fold
  boundaries; protected holdout untouched (one shot per promoted candidate,
  owner token, after the gates revision fixes the stale $1,500 cap).
- Model family: bounded HGB first (existing runner
  `v4/scripts/run_protocol101_stage1_bounded_hgb_search.py`, adapted to the
  canonical contract). Sequence/neural models only if learning curves show
  representation, not capacity, is binding — per the standing model ladder.
- Batch structure: 21-attempt batches (policies x seeds x folds) per
  hypothesis, primary + conservative, as in the Group 2 program.
- Gates: G1–G9 with G4 v2 (Calmar >= 1.0 pooled + $5,000 per-fold equity
  floor per the signed revision; null-relative drawdown report-only),
  evaluated against the RECALIBRATED canonical-contract nulls (the
  masked-era bands are void). G3 baseline = put/call skew heuristic at
  +$15,953 pooled (the standing best from the heuristic scan).
- Ledger entry before results are inspected, for every attempt. No
  threshold/feature tuned on test folds. Sealed days are never inputs.

## Routing outcomes per hypothesis

- `eligible_offline_candidate` — passes G1–G8 => G9 confirmation seed =>
  one-shot holdout => recorder-day replay validation => shadow mode proposal.
  Thresholds remain provisional until re-confirmed on recorder-native
  evidence before any paper-submit.
- `stage2_learned_exits_candidate` — the Stage-2 evidence packet: entry
  signal real (G2 z >= 3.0 pooled and G5 worst-seed z >= 2.0) but full-gate
  failure attributable to exit shape, with salvageable-MFE path diagnostics
  materially exceeding the no-skill band. This packet, and only this packet,
  justifies drafting the Stage-2 owner objective.
- `rejected_no_real_signal` — fails null/no-skill gates; recorded, next
  hypothesis proceeds.
- If H0–H4 ALL reject: canonical-game alpha is empty at Stage-1. Preregistered
  fork: evaluate same-vendor live data plane (Path B) or a different strategy
  class. Gates are not loosened; that outcome is reportable and final for
  this feature set.

## Honest expectations

Two prior feature sets (masked baseline, Group 2 geometry) had no real
signal. The restored quote-derived families are the first feature set
containing the information a discretionary 0DTE trader actually uses; that is
grounds for the attempt, not a prediction of success. The most likely
constructive outcome is a `stage2_learned_exits_candidate` from H2/H4 —
entries worth taking whose profit lives in exit policy — which is exactly the
staged architecture's designed path.

## Owner sign-off checklist

- [ ] Five-hypothesis set confirmed as complete (no additions without amendment)
- [ ] Noise-injection primary level 1.0x measured divergence confirmed
- [ ] Intersection guards mandatory for all gate evaluation confirmed
- [ ] Pessimistic-rung fill gating confirmed
- [ ] Stage-2 unlock criteria (evidence packet definition) confirmed
- [ ] Execution blocked until sealed confirmation + signed gates revision
