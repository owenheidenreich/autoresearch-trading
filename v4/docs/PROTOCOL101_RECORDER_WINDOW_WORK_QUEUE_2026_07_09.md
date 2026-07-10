# Protocol101 — Recorder-Window Work Queue (Codex Handoff Plan)

## Context

Canonical v1.4 selection contract **passed** on 2026-07-09 (attempt005, all 19 probes green, 438→77 disagreements, 0 material reorderings, frozen hash `602fd8eff564a059ad114dd051b6793cb50bcc50269c83b81bdfe25aa119ef57`). The burned-day design phase is complete under a **hard stop: no further burned-day iterations under any justification**.

The IBKR recorder is now collecting fresh sessions (currently configured through 2026-07-24). The next milestone is a **one-shot sealed-day confirmation** of the canonical transform + v1.4 contract on days nobody touched. Everything in this plan either (a) protects/extends that sealed evidence, or (b) completes training-phase prerequisites that use only the 15-month vendor corpus and burned days — so that if confirmation passes, Stage-1 hill climbing starts the same day with zero added calendar.

Each step below is one narrow Codex `goal`. Order matters for Steps 0–2 (urgent, protect unrecoverable evidence); Steps 3–9 can interleave.

**Constraints that apply to every step** (from AGENTS.md + project law):
- Interpreter: `~/.autoresearch-trading/runtime-venv/bin/python`
- No broker calls, paper-submit, paid downloads, promotion/default changes, real-money paths.
- Runtime-flag/launchd changes ONLY where a step explicitly grants owner authorization.
- No burned-day probe/contract iterations (v1.4 frozen). No reading sealed market data.
- No gate-graded Stage-1 model training until sealed confirmation passes (sunk-cost discipline).

---

## Step 0 (TODAY, urgent): Create the seal-on-arrival rule

**Why:** Confirmed by repo search: **no seal/sealed-day mechanism exists anywhere in v4/**. The recorder is already collecting, but without a sealing rule the incoming days are inspectable, which would let a future skeptic (or us) argue the confirmation was contaminated. Days sealed retroactively are arguably tainted; days sealed on arrival are not. Every session that lands before this exists weakens the confirmation.

**What:**
- Add a small module + script (e.g. `v4/scripts/run_protocol101_sealed_day_assignment.py`) implementing this fixed rule, written before 07-13: **2026-07-10 is a designated VALIDATION day** — openly inspected to verify capture quality (the owner's planned review of tomorrow's session happens on this day, at full market-data depth, with no sealing claim); **all sessions from 2026-07-13 onward are SEALED on arrival.** Burned days (06-30/07-01/07-02) + 07-10 form the repair/validation set; sealed days are confirmation-only.
- Sealed-day capture output moves/lands in a dedicated directory (e.g. `v4/audit/sealed/`) that no analysis/audit tooling reads. Enforce by convention + a guard check in the assignment script; document in the manifest.
- Write `sealing_rule.json` (rule, hash, effective date, directory policy, the 07-10 validation-day designation) and hash it.

**Key paths:** recorder entrypoint `v4/ops/ibkr/run_protocol101_ibkr_recorder.py`, control wrapper `v4/ops/ibkr/protocol101_recorder_control.py`, capture engine `v4/live/ibkr_market_capture.py`, recorder packet `v4/ops/ibkr/run_protocol101_recorder_packet.sh` (patched 2026-07-09, currently untracked in git).

**Done when:** sealing_rule.json exists and is hashed before 07-13's session; 07-13+ sessions land under the sealed directory; 07-10 is explicitly recorded as validation-only.

---

## Step 1 (TODAY): Extend recorder collection through 2026-07-30

**Why:** The confirmed schedule (07-10, 07-13→17, 07-20→24; 11 sessions) ends before the July FOMC (07-28/29). Every fragility found in the design phase was worst on high-volatility days; an FOMC session is the single most valuable confirmation day available. Collection is passive — extending costs nothing. Also: with 07-10 reclassified as a validation day (Step 0), only 10 sealed sessions remain — the bottom of the 10–15 target; extending to 07-30 yields ~14 sealed days including FOMC.

**What:** Extend the recorder-only schedule through 2026-07-30 (add 07-27→07-30) in the deployed setup: `v4/scripts/deploy_protocol101_recorder_parity.py` + the `com.autoresearch.protocol101.parityrecorder.*` launchd labels, runtime bundle `~/.autoresearch-trading/runtime-bundles/protocol101-parity-v1/6364222ef3fae132`. Keep `gate_mode none` (record-every-day semantics). Explicit owner authorization: scheduling config/launchd for the parity recorder only.

**Done when:** launchd labels cover sessions through 07-30 and a fresh health manifest confirms capture still green.

---

## Step 2: Verify existing recorder health checks + fix git drift

**Why:** Codex already built health infrastructure (launchd checks at 05:50/06:00/06:15/06:28 PT, a watchdog, a 13:15 audit, and twice-daily Codex automation `protocol101-recorder-pre-open-check`). Don't rebuild it — verify it, and close two gaps: (a) health checks must read **manifests only** once sealing starts on 07-13 (never sealed market data); (b) the two patched files (`v4/ops/ibkr/run_protocol101_recorder_packet.sh`, `v4/scripts/deploy_protocol101_recorder_parity.py`) are **untracked in git** while the installed bundle runs the patched versions — a repo-vs-deployed drift that will bite the first time anyone redeploys from the repo.

**What:** (1) Using the 07-10 validation day, confirm the existing checks catch what matters: rows/session ≈ expected, gap detection, disk, schema hash, and — from 07-13 — that outputs land in the sealed directory and the 13:15 audit does not open sealed market data. (2) Commit/track the two patched files so the repo matches the deployed bundle; record the bundle hash `6364222ef3fae132` in the commit message.

**Done when:** 07-10's manifests are green and reviewed; the audit path is confirmed manifest-only for sealed days; git status is clean for the recorder files.

---

## Step 3: G4 drawdown feasibility repair + stale holdout cap fix (owner-signed doc change)

**Why:** G4 (max drawdown ≤ 25% of peak equity) passed **0/42 attempts** in Group 2 and the prior feasibility artifact measured the game's random-noise drawdown floor at ~$7,800 on $10k — the relative cap (~$2,500) is likely below what even a *profitable oracle* can achieve at the required trade frequency. A gate no strategy of the desired class can pass manufactures "no signal" verdicts. Separately, the holdout protocol in the gates doc still says "max DD ≤ $1,500" — the exact absolute number the G4 revision retired; as written, any candidate reaching holdout auto-burns. This is not gate-loosening: it is repairing a gate proven infeasible by the project's own artifact, before training, with owner signature.

**What:**
- Rerun/extend the feasibility measurement in `v4/audit/autoresearch/protocol101_stage1_g4_feasibility/` (it contains `g4_feasibility.py`): measure max drawdown distribution at G7-required frequency for (a) random policies, (b) the fixed Pickle heuristic, (c) a synthetic peek-ahead oracle with same exits/frequency — through `v4/model/protocol101_serial_simulator.py` with the $3 fee overlay.
- Propose a revised G4 (e.g. cap = oracle p95 drawdown × margin, or a floor-aware relative cap) in an owner-signed revision of `v4/docs/PROTOCOL101_STAGE1_OBJECTIVE_AND_GATES_PROPOSAL.md`; fix the holdout paragraph's $1,500 in the same revision.
- No training, no threshold selection — this is measurement + document revision for owner review.

**Done when:** feasibility artifact shows what an oracle can achieve; revised G4/holdout text drafted and awaiting owner signature. (Blocks all gate-graded training until signed.)

---

## Step 4: Draft the preregistered sealed confirmation battery (runnable ~07-30)

**Why:** The confirmation exam must be a button-press with zero adjustable parameters, hashed **before** any sealed day is opened. Preregistering now, weeks early, is what makes the eventual pass trustworthy.

**What:** One goal that writes and hashes the confirmation preregistration (no execution): frozen v1.4 contract hash `602fd8ef…ef57`; the full 19-probe L1/L3 battery + L0 field-divergence/bias tests + L2 source discriminator, run **one-shot** on all sealed days; identical pass criteria to attempt005; all epsilons/thresholds frozen; regime requirement (≥1 high-vol/event day present, else wait); watch-items preregistered with expectations — 34 material split-action minutes (CDE subsets), IV probe as full gate again (expected ≥0.99 post-v1.4), delta-geometry slot 0.9879; per-day AND pooled reporting; routing outcomes (`confirmed` / `confirmation_failed_specific_probe` / `insufficient_regime_coverage`); explicit rule that a failure routes to analysis on repair-set days only — sealed days are never used for repair.

**Key inputs:** attempt005 packet `v4/audit/autoresearch/protocol101_canonical_v1_4_near_atm_band_restriction_attempt005/`, reconciliation epsilons, L0 audit thresholds.

**Done when:** preregistration doc + hash exist; running it later requires only the sealed-day paths.

---

## Step 5: Null/canary recalibration under the canonical v1.4 contract

**Why:** Project law: nulls rerun whenever the feature contract changes. The existing G2 null bands (`protocol101_live_v2_microstructure_masked_null_canary_15mo_cv_policy0..6`) were computed under the **masked** contract; canonical v1.4 is a new contract with new features and new selection semantics. Every future gate verdict is invalid without recalibrated nulls. Uses the 15-month corpus + burned days only — sealed days untouched.

**What:** Rerun the null/canary machinery — `v4/scripts/run_protocol101_fair_contract_null_canary.py` and `v4/scripts/run_protocol101_fair_contract_gate_null_baselines.py` — under: canonical v1.4 feature set (22 admitted + 12 G1 non-VIX), frozen v1.4 selection semantics (margin gate, dead-band, deterministic fallback), all 7 menu-v2 shapes (`v4/dataset/spxw_0dte_neural.py` label_policies), 15-month governed corpus via `v4/model/protocol101_governed_loader.py`, serial simulator, $3 fee overlay. Emit new per-policy null-canary artifacts + updated G2 bands, hashed.

**Done when:** new null artifacts exist for all 7 policies with a summary comparing old vs new bands.

---

## Step 6: Heuristic baselines through the full simulator and gates

**Why:** Sets the G3 "beats-heuristic" bar honestly before any model exists — and the passing L1 probes are now real candidate rules on features no prior attempt could see. If a dumb rule already shows signal, that's the cheapest possible edge discovery; if none do, that's the honest G3 floor. Required before Stage-1 either way, so the wait costs nothing.

**What:** Run the four passing signal probes (straddle-mid expansion, put/call ratio skew, near-ATM band momentum, internal-delta geometry — definitions in `v4/scripts/run_protocol101_canonical_v1_4_near_atm_band_restriction_probe.py` lineage) as fixed strategies over the 15-month corpus: v1.4 selection semantics, all 7 shapes, serial simulator, $3 fee + $2/$5 sensitivity, gates G1–G9 evaluated against Step 5's recalibrated nulls (G4 per Step 3's revision if signed, else report both). Preregister before results; ledger entries first; report per-gate outcomes without tuning anything.

**Done when:** per-heuristic gate scorecards exist; the best heuristic's pooled fee-adjusted PnL is recorded as the standing G3 baseline.

---

## Step 7: Stage-1 training design document (owner-signed, hashed before confirmation results exist)

**Why:** Preregistering the training design before the confirmation exam eliminates the "we designed the training after seeing what passed" critique, and encodes the design lessons paid for this week.

**What:** Write `v4/docs/PROTOCOL101_CANONICAL_V1_STAGE1_TRAINING_DESIGN.md` covering: the five preregistered feature-subset hypotheses (AB+G1 control / +C / +D / +E / +CDE); training inside frozen v1.4 selection semantics (margin-gated slots, action dead-band, score-independent nearest-ATM fallback); noise injection at measured divergence (Step 8 machinery); intersection guards (Step 9); conservative fill ladder (mid / mid+spread-fraction / touch) with gating on the pessimistic rung; bounded HGB first per the model-family ladder; 5-fold chronological CV, 1-session embargo, 21-attempt batch structure; design rule from the IV saga — **no entry/side rules that threshold or sign low-variance signals near their distribution bulk**; the wing-noise rule — band-aggregate scores use abs_offset ≤ 19; routing outcomes incl. the Stage-2 learned-exits evidence packet. Reference `v4/scripts/run_protocol101_stage1_bounded_hgb_search.py` as the runner to adapt.

**Done when:** doc exists, hashed, awaiting owner signature.

---

## Step 8: Noise-injection machinery (build + unit-test, no training)

**Why:** Hard dependency of Stage-1: a model that keeps its edge when features are perturbed by the *measured* cross-feed divergence is robust to the drift that actually exists. The distributions are already on disk.

**What:** New module (e.g. `v4/model/protocol101_divergence_noise.py`) that loads `v4/audit/autoresearch/protocol101_canonical_v1_l0_l2_design_audit_attempt001/divergence_distributions.parquet` and injects per-feature noise **conditioned on moneyness band** (the measured 20× ATM→wing gradient must be preserved). Deterministic given seed. Unit tests: injected-noise marginals match measured distributions per band; zero-divergence features (Family A/B) stay untouched; burned-day smoke shows a v1.4 probe's decisions are stable under injection at 1× measured level.

**Done when:** module + tests pass; a short calibration report compares injected vs measured distributions.

---

## Step 9: Intersection guards (build + burned-day audit)

**Why:** Historical replay admits ~4.7% more boundary-stable candidates than IBKR (measured: 31,246 vs 29,791). Uncorrected, every backtest overstates opportunity and frequency. Training must see the pessimistic (intersection) tradability so historical results transfer down, not up.

**What:** Implement an intersection-guard mode (candidate tradable only if it passes guard state on BOTH planes where paired data exists; on vendor-only history, apply IBKR-calibrated tightened caps derived from the measured asymmetry). Base it on the guard logic in `v4/scripts/run_protocol101_static_ladder_boundary_stable_policy_audit.py` (boundary-stable margins). Burned-day audit: report candidate counts under normal vs intersection guards, confirm the ~4.7% gap closes, and record the tightened-cap derivation for vendor-only sessions.

**Done when:** guard mode exists with an audit artifact quantifying its effect; Step 7 doc references it.

---

## Step 10: VIX warm-up trace fix (infrastructure only; feature admission deferred)

**Why:** `vix_change_5m/15m(_bps)` (4 features) are blocked only because paired traces carry a single pre-window row (09:31), so exact `decision_ts − 5m/15m` lookups are missing for the first minutes. Fixing trace construction now converts a known blocker into a ready option — but the features are NOT admitted in this cycle, so the frozen v1.4 contract and the pending confirmation are untouched.

**What:** Extend paired-trace construction to carry ~20 minutes of pre-decision context: `v4/live/protocol101_capture_replay.py` (window anchored at `09:31` around line 163; decision gating around line 366) and the `market_window_minutes: int = 30` parameter in `v4/dataset/spxw_0dte_neural.py` (context computed as `decision_time − (market_window_minutes − 1)`, ~line 288). Rebuild burned-day traces under a NEW audit prefix (do not overwrite the source-aligned traces the frozen contract was certified on). Validate: 09:32 rows now have finite 5m/15m VIX lookups, symmetric across planes. Explicitly no L0 re-audit and no feature admission — that is a v2-features cycle after confirmation.

**Done when:** new-prefix burned-day traces exist with pre-window depth; a coverage report shows the VIX gap closed; frozen-contract inputs untouched.

---

## What deliberately does NOT happen during this window

- No burned-day probe/contract iterations (v1.4 hard stop is absolute).
- No reading sealed market data — health checks read manifests only.
- No gate-graded Stage-1 model training until sealed confirmation passes (prevents sunk-cost pressure on the one-shot exam). Steps 5–6 are exempt: nulls and fixed heuristics carry no tunable model state.
- No VIX feature admission, no unmasking, no new epsilons/thresholds outside the two owner-signed doc revisions (Step 3).

## Verification

- Steps 0–2: today's session lands sealed with a green manifest; recorder end date ≥ 07-30.
- Steps 3–4: two hashed documents exist (gates revision draft, confirmation preregistration) with no execution side effects.
- Steps 5–6: new artifacts under `v4/audit/autoresearch/` with side-effect flags all false except the intended computation; old-vs-new null band comparison sanity-checked.
- Steps 8–10: unit tests pass via targeted pytest; burned-day smoke/audit artifacts written; `git diff` confirms no changes to frozen-contract inputs or runtime posture.
- End state test: on the day collection completes, the only remaining action is running the Step 4 preregistration against the sealed directory.
