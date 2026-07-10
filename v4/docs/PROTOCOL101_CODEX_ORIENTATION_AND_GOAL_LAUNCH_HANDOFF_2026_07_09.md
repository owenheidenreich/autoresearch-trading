# Protocol101 — Codex Orientation & Goal-Launch Handoff (2026-07-09 evening)

This document orients Codex after a major project-state change today. Read it
before starting any new goal. The step-by-step goal specifications live in
[PROTOCOL101_RECORDER_WINDOW_WORK_QUEUE_2026_07_09.md](PROTOCOL101_RECORDER_WINDOW_WORK_QUEUE_2026_07_09.md);
this document is the context layer: what just happened, what is frozen, what
is safe to build, and why.

## Where the project is now (plain language)

The month-long synchronization problem is solved at design level. The
canonical minute-game transform plus the v1.4 selection contract make
Databento/ThetaData history and IBKR recorder data behave as the same game:
all 19 probes pass, 438 -> 77 disagreements on the burned design days, zero
economically material divergence. That work is **frozen** — contract sha256
`602fd8eff564a059ad114dd051b6793cb50bcc50269c83b81bdfe25aa119ef57` — under a
hard stop: no further burned-day iterations under any justification.

What remains before training can be trusted is **confirmation on fresh
evidence**: the parity recorder is collecting sessions (07-10 validation day,
then 07-13 onward sealed on arrival) and a preregistered one-shot battery
will be run on those sealed days around the end of July. Nothing about that
exam may be designed, tuned, or touched between now and then — it is already
fully specified and hashed.

**The purpose of the goals below is simple: build everything Stage-1 training
needs, now, using only the 15-month vendor corpus and the burned days, so
that if the sealed exam passes, hill climbing starts the same day.** None of
these goals touch the recorder, sealed data, or any frozen artifact. They
cannot break the parity work if the rails below are respected.

## Completed today (do not redo, do not modify)

| Item | Where | Hash |
|---|---|---|
| Canonical v1.4 pass (attempt005) | `v4/audit/autoresearch/protocol101_canonical_v1_4_near_atm_band_restriction_attempt005/` | contract `602fd8ef…ef57` |
| Seal-on-arrival rule + script | `v4/scripts/run_protocol101_sealed_day_assignment.py`, governance in `v4/audit/autoresearch/protocol101_sealed_day_assignment/` | rule `9be032d7…157b4d` |
| Sealed confirmation battery preregistration | `v4/audit/autoresearch/protocol101_canonical_v1_sealed_confirmation_preregistration/` | `da9f53ca…508511` |
| Stage-1 training design (awaiting owner sign-off) | `v4/docs/PROTOCOL101_CANONICAL_V1_STAGE1_TRAINING_DESIGN.md` | `c41f9e3f…64a9d` |
| Recorder repo/bundle drift fix | commits `730042af`, `cc98ec91`, `3de53290` | — |
| 07-30 recorder extension (STAGED, not run) | `v4/ops/ibkr/extend_protocol101_recorder_sessions_to_0730.sh` | owner runs it AFTER the 07-10 validation session reviews green |

## Hard rails (violating any of these damages irreplaceable evidence)

1. **Never read anything under** `~/.autoresearch-trading/live_runtime/ibkr_capture_sealed/`.
   Sealed days are spent exactly once by the preregistered battery. Health
   automation may only call
   `run_protocol101_sealed_day_assignment.py check` (manifest-level).
2. **Never modify** the frozen v1.4 contract, its epsilons/thresholds, the two
   preregistration documents, the sealing rule, or anything inside the
   attempt001–005 / reconciliation audit directories. If a goal seems to
   require it, the goal is wrong — stop and report.
3. **No burned-day probe/contract iterations.** The burned days
   (06-30/07-01/07-02, plus 07-10 after validation) may be used as *inputs*
   (smoke tests, audits, noise calibration) but never to re-tune selection
   semantics, epsilons, or probe definitions.
4. **Do not touch the recorder deployment** — plists, bundle, packet scripts,
   launchd labels `com.autoresearch.protocol101.parityrecorder.*`. The only
   sanctioned change is the staged extension script, run by the owner.
5. **No gate-graded Stage-1 model training** until the sealed battery routes
   `canonical_v1_4_sealed_confirmed` AND the G4/holdout gates revision is
   owner-signed. (Goals 5 and 6 below are exempt: nulls and fixed heuristics
   carry no tunable model state.)
6. Standing law unchanged: no broker calls, paper-submit, paid downloads,
   promotion/default/runtime/launchd changes, real-money paths;
   interpreter `~/.autoresearch-trading/runtime-venv/bin/python`;
   ledger/preregister before results; never loosen gates.

## The six launchable goals (independent of the recorder; safe tonight)

Run each as one narrow goal. Full specs are Steps 3, 5, 6, 8, 9, 10 of the
work-queue document. Purpose and launch notes:

**Goal A (= Step 3): G4 drawdown feasibility + stale holdout cap — measurement + doc draft.**
Why: gate G4 has passed 0/42 attempts ever and is likely mathematically
infeasible as written (random-noise drawdown floor ~$7,800 vs ~$2,500 cap);
the holdout doc still contains the retired $1,500 cap. Every future training
verdict is contaminated until this is measured and revised. Output: oracle/
heuristic/random drawdown distributions through the serial simulator + a
drafted owner-sign revision of the gates doc. No training, no tuning.
Launch first: it needs owner-signature turnaround.

**Goal B (= Step 5): Null/canary recalibration under the canonical contract.**
Why: project law — nulls rerun when the feature contract changes. The
existing G2 bands belong to the dead masked contract; all gate verdicts are
void without new bands. Longest compute; launch early. Uses the governed
15-month corpus + frozen v1.4 semantics.

**Goal C (= Step 6): Heuristic baselines through the simulator and gates.**
Why: sets the honest G3 bar, and the passing probes are the first candidate
rules ever run on features that can actually see option prices — a cheap
shot at edge discovery. Depends on Goal B's bands (launch after B completes,
or evaluate provisionally against old bands and re-score when B lands —
state which in the preregistration).

**Goal D (= Step 8): Divergence noise-injection module + tests.**
Why: hard Stage-1 dependency — a model must keep its edge under the
*measured* cross-feed noise (per-moneyness-band, from
`protocol101_canonical_v1_l0_l2_design_audit_attempt001/divergence_distributions.parquet`)
or it is drift-mining. Pure build + unit tests + burned-day smoke.

**Goal E (= Step 9): Intersection guards + burned-day audit.**
Why: historical replay admits ~4.7% more tradable candidates than IBKR
(measured); uncorrected, every backtest flatters itself. Build the
pessimistic guard mode, audit that it closes the measured gap.

**Goal F (= Step 10): VIX warm-up trace fix — infrastructure only.**
Why: 4 VIX features are blocked purely by trace construction (one pre-window
row). Extend pre-decision context ~20 minutes, rebuild burned-day traces
under a NEW audit prefix (never overwrite the source-aligned traces the
frozen contract was certified on). Explicitly NO feature admission and NO
L0 re-audit — that is a future v2-features cycle.

Suggested launch order tonight: A and B in parallel (A is small + needs the
owner; B is long compute), then D and E (independent builds), then F, then C
when B's bands exist.

## Why none of this can break the accomplished work

Every goal above reads from frozen, hashed artifacts and writes only NEW
files: new audit directories, new modules, new tests, a new trace prefix, a
drafted doc revision. The parity result lives in frozen contracts and
already-committed audit packets; the sealed evidence lives behind a directory
no goal is allowed to read; the recorder runs untouched on its own launchd
schedule. The failure modes that could actually hurt the project are exactly
the six rails above — everything else is additive.

## After the sealed exam (for orientation only; not tonight's work)

If `canonical_v1_4_sealed_confirmed`: Stage-1 training per the signed design
doc (five hypotheses H0–H4, bounded HGB, noise injection, intersection
guards, pessimistic fills, gates vs new nulls). If confirmation fails: all
diagnosis on burned/validation days only; new sealed days must accumulate
for any retest. If Stage-1 finds entries but weak PnL with salvageable MFE:
that evidence packet — and only that — unlocks the Stage-2 learned-exits
objective the owner's end vision describes.
