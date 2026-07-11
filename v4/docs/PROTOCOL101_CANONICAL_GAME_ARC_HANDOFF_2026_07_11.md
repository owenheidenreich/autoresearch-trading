# Protocol101 Canonical-Game Arc — Full Handoff (2026-07-11)

Audience: Codex, external AI reviewers, or future sessions picking this up
cold. This supersedes the 2026-07-09 "current endeavor" handoff for the
synchronization workstream. It condenses a two-day design/verification arc
(2026-07-09 → 07-11) into current state, evidence, rails, and next actions.

## The end goal (unchanged)

Build a profitable SPXW 0DTE model trained on Databento/ThetaData history
that plays the SAME game live on IBKR, validated enough to enter guarded
paper trading. Two independent claims, tested in order:

1. **Transferability**: the model optimizes the same game historically and
   live. (This arc — solved at design level, confirmation pending.)
2. **Profitability**: that game contains real edge. (Stage-1 training,
   starts only after claim 1 is confirmed.)

## The pivot that defined this arc

Months of trying to prove raw Databento/ThetaData quotes equal raw IBKR
quotes ended with proof that exact cross-vendor parity is impossible
(44 irreducible boundary rows; option-quote source policy resolution).
The replacement hypothesis, now designed and burned-day-certified:

> Transform both feeds into a shared canonical minute-level decision game.
> If a classifier cannot tell which feed a canonical row came from, and a
> battery of decision-makers behaves identically on both, a model trained on
> vendor history plays the same game live.

Field-by-field parity was the wrong proof; behavioral equivalence of the
transformed game is the right one.

## Evidence chain (all committed; verify hashes before relying on them)

| Step | Result | Artifact dir (under v4/audit/autoresearch/) |
|---|---|---|
| L0/L2 design audit | PASS — 22 features admitted incl. new Families C (option mids/momentum/path), D (straddle/skew composites), E (internal BS Greeks); worst standardized drift 0.042; bias ~0; discriminator AUC 0.52 (Family C alone 0.5265) | `protocol101_canonical_v1_l0_l2_design_audit_attempt001` |
| L1/L3 probe attempt001 | REJECTED — trigger for the reconciliation | `protocol101_canonical_v1_l1_l3_probe_audit_attempt001` |
| Rejection reconciliation | 438 disagreements decomposed: 361 score-collision classes, 84/86 flips threshold-adjacent, only 32 material true reorderings; "nondeterministic tie-break" hypothesis DISPROVED (harness was deterministic; cause = tick-quantized score collisions + cross-plane tie-set membership changes) | `protocol101_canonical_v1_probe_rejection_reconciliation_attempt001` |
| v1.1 (attempt002) | repair_needed — two artifacts found: old thresholds applied to new band scores; set-dependent fallback | `protocol101_canonical_v1_1_selection_contract_probe_attempt002` |
| v1.2 (attempt003) | repair_needed — 438→135; residual traced to side/action flips of a near-zero band-IV signal | `protocol101_canonical_v1_2_selection_contract_repair_attempt003` |
| v1.3 (attempt004) | repair_needed — action dead-band added; proved dead-bands cannot fix boundary-dense scores (two edges ≈ one) | `protocol101_canonical_v1_3_action_deadband_attempt004` |
| **v1.4 (attempt005)** | **PASS — all 19 probes green; 438→77 (82.4%); material reorderings 32→0; root cause was wing-band IV hypersensitivity (drift p95 0.0016 ATM → 0.0122 far wing; max-over-bands amplifies tails); fix = band-aggregate scores restricted to abs_offset <= 19** | `protocol101_canonical_v1_4_near_atm_band_restriction_attempt005` |

**Frozen canonical v1.4 selection contract sha256:**
`602fd8eff564a059ad114dd051b6793cb50bcc50269c83b81bdfe25aa119ef57`
**HARD STOP: no further burned-day iterations under any justification.**
The burned days (2026-06-30/07-01/07-02) are design-spent; five iterations
ran against them and they can prove nothing further.

### Design lessons encoded as permanent rules

1. Never let per-slot argmax select among tick-quantized near-tied scores;
   margin-gate slot selection at k=2 x measured noise, deterministic
   fallback must be score-independent (nearest-ATM on chosen side).
2. Never build entry/side rules that threshold or sign a low-variance signal
   near its distribution bulk (four attempts died on this with internal-IV).
3. Band-aggregate alpha scores use near-ATM bands only (abs_offset <= 19);
   wings stay in universe/guards/labels/fills/audit but never aggregate alpha
   (IV inversion is hypersensitive where vega ~ 0).
4. Every gate creates a boundary and every boundary creates cross-plane
   flips; measure each new split class (split-confidence, split-action) with
   min-n discipline (denominator < 30 = descriptive only).
5. Report reductions against primary-class counts, not multi-label counts.

## Day taxonomy (sealing rule v2, hash `05bce7c5…d625b`; file `a9fb2c25…a89e`)

Script: `v4/scripts/run_protocol101_sealed_day_assignment.py` (assign runs
daily via the 13:30 PT Codex automation after finalize+audit).

| Class | Days | Meaning |
|---|---|---|
| Burned | 06-30, 07-01, 07-02 | Design-spent; repair surface only |
| Validation | 07-10 | Capture-quality check; PASSED all 13 checks (3.5GB, 390/390 checkpoints, deterministic same-input replay, 360 rows 09:31–15:30 ET) |
| Development | 07-13, 07-14 (CPI session), 07-20 | Openly inspectable; rehearsal + diagnosis + repair surface |
| Sealed | 07-15..17, 07-21..24, 07-27..30 (11 expected; FOMC 07-28/29 deliberately sealed) | Never read by tool or human; spent exactly once by the confirmation battery; live in `~/.autoresearch-trading/live_runtime/ibkr_capture_sealed/` |

Rule v2 was adopted 2026-07-10 while the sealed set was still empty (nothing
tainted). Recorder allowlist runs through 07-30; recorder-only, gate_mode
none, no broker/order paths.

## The exam structure (why failure no longer costs the whole sealed set)

Confirmation preregistration (amended hash `13176551…d06b9`):
`v4/audit/autoresearch/protocol101_canonical_v1_sealed_confirmation_preregistration/`

1. **Rehearsal gate (dev days, after 07-20):** the IDENTICAL battery — L0
   divergence/bias, L2 discriminator with null+positive controls, all 19
   L1/L3 probes under the frozen contract — runs on the three development
   days. Design-grade, no confirmation claim. A failure here routes to
   diagnosis on dev+burned days; any repair becomes a NEW owner-signed
   contract version that must itself pass rehearsal. The sealed set is never
   spent on a recipe that has not passed rehearsal.
2. **Sealed exam (~07-30):** one shot, zero adjustable parameters, >= 9
   sealed sessions with the FOMC days individually passing (pooled results
   may not hide a failing event day). Pass criteria: every probe >= 0.98
   action agreement; >= 0.99 mutually-confident slot agreement (min-n
   discipline); material true reorderings ~0 (allowance 2 per 10 sessions);
   L0 drift/bias within design gates; discriminator <= 0.55 with green
   controls. Requires paired Databento/ThetaData sessions for the sealed
   dates (paid download — owner authorization needed, schedule week of
   07-20). Sealed days become burned the moment the battery reads them.

**Why probes can test decision parity without a trained model:** parity is a
property of the game, not the player. Nineteen diverse stand-in players
(fixed rules + small trained models spanning all feature families) certify
the arena; the frozen selection contract then mechanically prevents ANY
future model from expressing decisions finer than measured cross-feed noise;
and the eventual real candidate still gets its own preregistered
recorder-replay decision check before shadow/paper. If Stage-1 ever
escalates model class (e.g. sequence models), add a matching transfer probe
to the battery first.

## Training-phase prerequisites (all built and committed 07-09/07-10)

| Item | Status | Key artifact |
|---|---|---|
| Null/canary bands under canonical contract | DONE — 271 fold-native sessions, 100 random runs/fold/policy, all 7 menu-v2 shapes | `protocol101_canonical_v1_null_canary_training_scope` |
| Heuristic baselines (G3 bar) | DONE — best: put/call skew +$15,953, z=2.15, DD 151%, gates fail. First positive signal reading in project history; suggestive, not significant (G2 needs z>=3.0) | `protocol101_canonical_v1_heuristic_baselines_training_scope` |
| Divergence noise-injection module | DONE — band-conditioned, injected >= measured (conservative) | `v4/model/protocol101_divergence_noise.py` + calibration packet |
| Intersection guards | DONE — 7.16% haircut to exact intersection (31,246/29,791/29,008); vendor-only pessimistic policy recorded | `protocol101_canonical_v1_intersection_guard_audit` |
| VIX warm-up infra | BLOCKED by sources (historical index files start 09:30; IBKR checkpoints 09:31). Future recorder cycles should capture from ~09:15. No feature admitted. | `protocol101_vix_warmup_trace_infra_audit` |
| G4 feasibility measurement | DONE — random DD $10.7k–38k/fold vs $2.5k nominal cap: infeasibility proven. Unconstrained oracle DD $0 (useless anchor). | `protocol101_stage1_g4_feasibility_training_scope_v2` |
| Stage-1 training design (preregistered) | Hashed `38ed2dcb…81c9d`, awaiting owner signature. Five hypotheses H0 (control) / +C / +D / +E / +CDE; bounded HGB; training INSIDE frozen v1.4 semantics; noise injection + intersection guards + pessimistic fill rung mandatory. | `v4/docs/PROTOCOL101_CANONICAL_V1_STAGE1_TRAINING_DESIGN.md` |

## Open items (the complete list)

1. **G4 revision incomplete** — the drafted text only fixes the stale $1,500
   holdout sentence. Still needed: a frequency-forced oracle rerun (must
   trade in the G7 band even on losing days, to get a skilled-trader DD
   anchor) and a signable G4 shape — recommended direction: profit-relative
   drawdown (Calmar-style, e.g. max DD <= 0.5 x pooled net PnL) plus an
   absolute ruin backstop. Even the profitable heuristic drew down 151% of
   peak: for convex 0DTE long options, DD must be judged against profit, not
   equity. Blocks gate-graded training; owner signature required.
2. **Vendor historical download for dev+sealed dates** (07-13..07-30) —
   paid-data, owner authorization; needed before rehearsal (dev dates) and
   the sealed exam. Schedule week of 07-20 so it is not the critical path.
3. **Sealed-day automation reporting** — the 13:15 audit's same-input replay
   writes decision-level output (entry actions etc.). Outputs are sealed
   with the capture dir (harmless), but the 13:30 human-facing automation
   report must exclude decision-level fields for sessions >= 07-15;
   capture-quality booleans/counts/sealing result only. One-line change to
   the automation, no deployed-bundle edits.
4. **Null summary lacks a selection-semantics field** — heuristics summary
   records `canonical_selection_contract: v1.4`; nulls summary should state
   the same (documentation fix, not a rerun; random selection has no scores
   so semantics mostly don't bind).
5. **Codex must commit its packets** — untracked-artifact drift happened
   twice; every goal should end with a scoped commit.

## Hard rails (unchanged; violating any damages irreplaceable evidence)

- Never read `~/.autoresearch-trading/live_runtime/ibkr_capture_sealed/`
  (health = `run_protocol101_sealed_day_assignment.py check`, manifest-only).
- Never modify the frozen v1.4 contract, epsilons, preregistrations, sealing
  rule, or certified audit packets. Repairs create NEW owner-signed versions.
- No burned-day probe/contract iterations. Dev days are the repair surface.
- Don't touch the recorder deployment (plists/bundle/packet scripts).
- No gate-graded Stage-1 training until sealed confirmation passes AND the
  G4 revision is signed.
- Standing law: no broker calls, paper-submit, paid downloads (without
  explicit owner authorization), promotion/default/runtime/launchd changes,
  real-money paths; interpreter `~/.autoresearch-trading/runtime-venv/bin/python`;
  ledger/preregister before results; never loosen gates.

## Calendar

- **Now → 07-14:** recorder runs itself. Optional: finish open items 1, 3, 4.
- **Week of 07-20:** dev days complete (07-20) → run the rehearsal battery.
  Authorize + download vendor data for dev/sealed dates.
- **~07-30:** collection ends (FOMC 07-28/29 sealed). If rehearsal passed:
  run the sealed exam.
- **If confirmed:** Stage-1 training starts the same day (H0–H4, everything
  pre-built). If a hypothesis passes gates: G9 seed → one-shot holdout →
  recorder replay of the actual candidate → shadow mode → guarded paper.
  If entries are real but PnL weak with salvageable MFE: that evidence
  packet unlocks the Stage-2 learned-exits objective. If H0–H4 all reject:
  canonical-game alpha is empty; preregistered fork to same-vendor live
  data plane or a different strategy class — without loosening gates.

## One-paragraph summary

The month-long synchronization problem is solved at design level: both feeds
now map into a canonical minute-game whose features, and the decisions built
on them, are statistically indistinguishable across Databento/ThetaData and
IBKR on the design days — with every fragility found, root-caused, and fixed
under a frozen, hashed contract. Fresh evidence is accumulating on
autopilot, split into development days (rehearse, diagnose, repair) and
sealed days (one-shot exam, FOMC included). Every training-phase
prerequisite except the G4 gate revision is built, calibrated, and
committed. Passing the sealed exam confirms the model will play the game it
was trained on; whether that game can be won — the question the project
exists to answer — is Stage-1's job, starting the day the exam passes.
