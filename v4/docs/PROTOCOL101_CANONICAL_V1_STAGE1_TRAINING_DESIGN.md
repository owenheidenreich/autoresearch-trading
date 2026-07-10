# Protocol101 Canonical v1 Stage-1 Training Design — PREREGISTERED (awaiting owner sign-off)

Drafted: 2026-07-09, before sealed-day confirmation results exist.
Execution is BLOCKED until: (1) `canonical_v1_4_sealed_confirmed` routing from
the sealed confirmation battery (preregistration sha256
`da9f53cadca9cf93281ca150e26ead0aacaf31f588e884f13f9658b2ce508511`), and
(2) the owner-signed G4/holdout gates revision (Step 3 of the recorder-window
work queue) is signed. Preregistering now removes the "designed after seeing
what passed" critique.

## Objective

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
  as the edge band. Fees $3.00 primary, $2.00/$5.00 sensitivity.

## The five preregistered hypotheses (mirroring the certified L3 subsets)

| ID | Features | Role |
|---|---|---|
| H0 | A/B + Group1 non-VIX | Control — expected to fail like masked-v2; validates the harness |
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
- Gates: G1–G9 with revised G4, evaluated against the RECALIBRATED nulls
  (work-queue Step 5 — canonical-contract null/canary bands; the masked-era
  bands are void). G3 baseline = best heuristic from work-queue Step 6.
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
