# Path-D Phase-1 → Live-Market Roadmap & Living Status

**Living planning + handoff document.** Tracks every gate from the causal Phase-1 rebuild to a model
that could trade the live market (guarded paper first; real money is a separate owner+governance step).
Each phase is SIGNED with status so Codex/Claude can resume without re-deriving context. Update the
status + signature block whenever a phase advances.

## Honest framing (do not lose this)
- This is a **gated** roadmap. `NO_INCREMENTAL_EDGE` at Stage 1 is the **likely** honest outcome — the
  prior "confirmed edge" (signed18) was a 60 s SPX look-ahead, invalidated by the parity gate.
- The **protected 36-session firewall is SPENT** (signed18 opened `holdout_open_count=1`). There is no
  clean *historical* confirmation left; the forward out-of-sample test is **fresh live paper data**.
- **Paper proves execution, not alpha.** Real money is a distinct, owner-authorized, governance-heavy
  decision this roadmap gets you *to the door of*, not through.

## Status legend
`DONE` · `READY` (built + fixture-validated, awaiting a prerequisite) · `BLOCKED_ON_DRIVE` ·
`PENDING_CODEX` · `NOT_STARTED` · `GATE` (a stop/continue decision point) · `CLOSED`.

> # PHASE-1 IS CLOSED — `NO_INCREMENTAL_EDGE` (2026-08-03)
>
> Terminal record: **[`PATHD_PHASE1_CLOSEOUT_2026_08_03.md`](PATHD_PHASE1_CLOSEOUT_2026_08_03.md)** —
> read that first. Stage 2 is **not authorized** and **no further training is recommended on this
> strategy class**.
>
> **The terminal finding:** with **all** friction removed (buy at bid, sell at bid, zero fees) the average
> trade still loses **−$13.00**, median −$90, win 35.7%, **negative in 5/5 folds**. Buying SPXW 0DTE
> premium at minute cadence is **negative-expectancy before any cost is paid**. That is theta — no model,
> feature, or execution fix repairs it.
>
> **Do not open a new training round against this class.**
>
> **Owner-authorized Wave-1 exit-repair exception is complete — `NO_EDGE` (2026-08-03).** The bounded
> eight-member recovery/balance/fallback/LCB family did not reopen the strategy class: two arms failed
> pre-fit label balance; all six trained arms lost to their best comparator; no bootstrap LCB was
> positive; no member survived maxT. See
> [`PATHD_WAVE1_EXIT_REPAIR_RESULTS_2026_08_03.md`](PATHD_WAVE1_EXIT_REPAIR_RESULTS_2026_08_03.md).
> This strengthens the closure. Stage 2 remains unauthorized.
>
> **Owner-authorized Wave-2 causal 60-minute discovery is complete — `NO_SIGNAL` (2026-08-03).** Three
> fixed, non-fitted scores covering full-ladder surface continuation, option microstructure, and
> SPX/ES/VIX alignment all lost the best 60-minute comparator, failed fold/LCB/maxT/decile/big-loss
> gates, and produced no accepted negative control. See
> [`PATHD_WAVE2_CAUSAL_60M_SIGNAL_DISCOVERY_RESULTS_2026_08_03.md`](PATHD_WAVE2_CAUSAL_60M_SIGNAL_DISCOVERY_RESULTS_2026_08_03.md).
> No model-training gate was earned.
>
> **Adversarially reviewed by Codex (`321b3bbd`):** the Phase-1 closure and the friction accounting are
> UPHELD; **two feasibility claims were REFUTED and withdrawn** (the "116.2% = impossible" figure was a
> median-in-an-EV-formula artefact — the mean-payoff screen gives 80.19%; and ES is **not** the only
> reachable cell). **What survived: horizon dominates instrument choice.** Also withdrawn: "no
> subpopulation is positive" (10:30–10:59 ET is +$5.99 gross, 4/5 folds — though net is still −$20.08).
> Read the correction banners on the close-out and feasibility documents before using any number here.
>
> **The feasibility study is now DONE** —
> [`PATHD_GROSS_EXPECTANCY_FEASIBILITY_STUDY_2026_08_03.md`](PATHD_GROSS_EXPECTANCY_FEASIBILITY_STUDY_2026_08_03.md)
> (`ac485174`, amended). **The fatal choice was the CADENCE, not the instrument** — this survived every
> attack. The same option held 60 minutes rather than 1 clears a far lower bar on all three estimators:
> 116.2%→57.2% (median), 80.2%→53.9% (mean-payoff), 76.5%→53.8% (first-window). VX is dead everywhere.
> **ES's advantage over passive options is narrow and contingent on an unmeasured friction constant** — at
> a 2-tick ES spread it disappears. Establishing a hurdle is clearable does **not** claim profitability and
> does **not** license a training run.

## Status board

| Item | Status | Blocking dependency |
|---|---|---|
| Causal Phase-1 implementation (entry/exit/replay/storage) | DONE (audited PASS, committed `cba15b1a`) | — |
| Footprint estimate (89.69 GB < 150 GB cap) | DONE | — |
| SSD setup runbook (fail-closed) | DONE (executed on verified `/dev/disk4`) | — |
| Phase-1 fixture suite | DONE (green: entry/exit/replay/storage 19 passed) | — |
| Milestone commit + memory/ledger | DONE (`cba15b1a`, `9ae807f5`) | — |
| Suite-green (retire stale v3.2 reconciliation) | DONE (`fa1d9b08`; governance file 7/7 passed) | Claude independent verification |
| Stage 0 — drive setup | DONE (preflight + relocation + independent hashes passed) | Owner waived Claude verification |
| Stage 1 — causal training + four-box | DONE — `UNDERPOWERED`; GATE 1 STOP | — |
| Failure-mode diagnostic (Claude) | DONE — loss flat across 5/5 folds; no subpopulation positive **after costs** (10:30 ET is +$5.99 gross, −$20.08 net) | — |
| Fix research: execution / holding / direction | DONE — `FIX_CANDIDATE` (execution) / `NO_FIX` (profitability) (`d05f0137`) | — |
| **Phase-1 close-out** | **CLOSED — `NO_INCREMENTAL_EDGE`** | — |
| Stage 2 — runtime decision-parity | NOT AUTHORIZED | Stage 1 did not establish edge |
| Stage 3 — live-shadow orchestration | NOT_STARTED (needs building) | Stage 2 pass + live days |
| Stage 4 — guarded paper submit + confirmation | NOT_STARTED (needs building) | Stage 3 + live days |
| Stage 5 — real-money decision | OUT OF SCOPE | separate owner+governance packet |
| **Gross-expectancy feasibility study** | **DONE (`ac485174`)**, then **AMENDED** — cadence-dominates UPHELD; "impossible" + "ES only" REFUTED | — |
| **Codex adversarial review** | **DONE (`321b3bbd`)** — 3 upheld / 3 weakened / 2 refuted; Claude reproduced the refutations | — |
| **Matched execution-aware feasibility gate** | **EXECUTED — `INVALID`** ([results](PATHD_MATCHED_FEASIBILITY_RESULTS_2026_08_03.md)); 9/9 individually `NOT_FEASIBLE`, but H2-30 reversed control cleared, so family discarded | New preregistration required for any rerun |
| **Wave-1 exit repair (P065 / balance / fallback / LCB)** | **DONE — `NO_EDGE`** ([results](PATHD_WAVE1_EXIT_REPAIR_RESULTS_2026_08_03.md)); 8/8 family spent, 2 preflight rejects, 6 trained, 0 pooled wins, controls clean, 0 maxT survivors | Closed; materially different target/representation/horizon/game required |
| **Wave-2 causal 60-minute signal discovery** | **DONE — `NO_SIGNAL`** ([results](PATHD_WAVE2_CAUSAL_60M_SIGNAL_DISCOVERY_RESULTS_2026_08_03.md)); 3/3 fixed scores lost the +$7,053 best comparator, only 1-2 positive folds, negative LCBs, maxT p 0.907-0.963, non-monotonic deciles, 58-60% big-loss shares; controls clean | No training gate; exact surface/microstructure/cross-market composites closed on entry-v2 OOF |
| **Path-D Phase 0 feature certification** | **BUILT — `STOP_FOR_CLAUDE_VERIFICATION`** ([report](PATHD_PHASE0_FEATURE_CERTIFICATION_REPORT_2026_08_04.md)); 1/9 OPRA-only families `FIT_READY`, 8/73 features `ADMITTED`, 65/73 `BARRED`; signed ledger and fail-closed training-matrix enforcement implemented; no model fit | Independent hash/receipt/enforcement verification; no training until separately authorized |
| **Path-D Phase 0b unblock certification / Track A** | **OFFLINE TRANSFORMS BUILT; ARRIVAL PARENT CORRECTION APPLIED; LIVE WINDOW DECLARED** ([report](PATHD_PHASE0B_UNBLOCK_CERTIFICATION_REPORT_2026_08_04.md)); the transform receipts remain valid but the corrected parent law bars them until native CBBO-1m arrival is certified, so the current ledger is 8/73 `ADMITTED`; Track-A sessions are frozen as 2026-08-05, 06, 07, 10, 11 with per-message receipt recording; zero marginal cost confirmed; no connection attempted | Explicit owner authorization in the current conversation before the first Databento Live connection; then collect every declared session and certify without weakening tolerances |
| ES friction measurement | OWNER-AUTHORIZED, PREFLIGHT PAUSED ([preflight](PATHD_ES_SPREAD_MEASUREMENT_PREFLIGHT_2026_08_03.md)); exact cost $1.470344 < $3 cap, no download | Manifest's fixed UTC window misses final RTH hour on six EST sessions; corrected authorization required |

### Durable assets carried out of Phase-1

1. **The causal t−60s pipeline** — audited, works end-to-end, and correctly produced a negative answer.
   Reusable for any future hypothesis.
2. **The passive-execution costing correction** — posting at the bid instead of crossing at ask+tick is a
   real mechanical saving (positive in 5/5 folds), but its **magnitude is unidentified without queue
   data**. Honest range from Codex's fill ladder: **+$22.76 (optimistic offer-touch) → +$12.57 (one-tick
   penetration) → +$2.31 (two-tick)**. Quote the range, never the headline. This is a *costing*
   correction, not a strategy; the frozen `FILL_LAW` was NOT modified.
3. **The knowledge that this class is negative-EV before costs** — it closes a direction permanently
   rather than leaving it to be re-litigated.

### Preconditions for any future training round (gates, not suggestions)

1. **Change the position** — negative gross expectancy cannot be fixed by predicting it better.
2. **Change the features** — the 18-feature contract has zero ranking power (decile curve is flat and
   non-monotonic: top −$16.37, bottom −$17.75, middle best at −$11.48). Same-feature retrains are wasted
   compute.
3. **Re-pose the target** — predict the gross move and subtract the observable known cost separately;
   the current label buries a small noisy signal under a large deterministic cost. Second-order.
4. **Governance** — holdout SPENT: pre-registration + hard budget under family maxT, forward validation on
   fresh live paper only, never on these 215 sessions.

---

## Completed prep (drive-independent) — 2026-08-03
- **Causal implementation audited PASS** (Claude): t−60s SPX clock (bar stamped t structurally excluded),
  conservative one-tick-through fill/label law, entry→exit OOF firewall (only `OUTER_FOLD_OOF` artifacts
  create trajectories; initial-44 prequential training-only; full-dev shadow cannot make trajectories).
- **Footprint**: `PHASE1_STORAGE_FOOTPRINT_ESTIMATE.md` — 89.69 GB (corrected for full-to-15:55
  trajectories = 27.87 M rows and 9 label copies); fits < 150 GB cap with 60.31 GB headroom.
- **Fixtures green**: `test_pathd_phase1_entry/exit/replay/storage` 17 passed; Phase-1 subset 20/20. Only
  the 4 legacy v3.2-release-reconciliation tests fail (unrelated — see Suite-green goal).
- **Runbook** `PHASE1_SSD_SETUP_RUNBOOK.md`: 5-guard fail-closed destructive erase (whole-disk regex,
  `Whole==true`, `Internal==false`, 1.8–2.2 TB range, typed confirmation phrase). Not executed.
- **Committed** `cba15b1a` (Phase-1 rebuild + live-OPRA parity + docs); ledger `9ae807f5`; memory updated
  (signed18 invalidation recorded).

*Signed: Claude Opus 4.8 — 2026-08-03 — status: all drive-independent prep COMPLETE.*

---

## Stage 0 — Drive setup (owner-supervised) — ENGINEER SPEC
Physical + DESTRUCTIVE steps are in `PHASE1_SSD_SETUP_RUNBOOK.md` (5-guard fail-closed erase: whole-disk
regex, `Whole==true`, `Internal==false`, 1.8–2.2 TB range, typed phrase `ERASE diskN FOR AR_TRADING_DATA`;
owner enters the disk-ID + phrase). After the drive is erased→encrypted-APFS as `AR_TRADING_DATA`, dirs
created, and env set (section B block), run:
```bash
PYTHONPATH=. .venv/bin/python -m v4.scripts.run_phase1_exit_model storage-preflight --create-roots
#   POST: {"status":"PASS", encrypted:true, external:true, volume_name:"AR_TRADING_DATA",
#          capacity.free_fraction_after >= 0.25, phase1_allocation.projected_bytes <= 150000000000}
PYTHONPATH=. .venv/bin/python -m v4.scripts.run_phase1_exit_model relocate-corpus \
  --source /Users/gduby/.autoresearch-trading/pathd_2025-08-01_2026-07-31
#   dest = /Volumes/AR_TRADING_DATA/vendor/pathd_2025-08-01_2026-07-31 (default --destination-name)
#   POST: {"status":"COPIED_AND_VERIFIED", source_preserved:true, manifest_sha256:<hex>, total_bytes:~20e9}
```
**Verify (Claude):** preflight `PASS` (encrypted external APFS / name `AR_TRADING_DATA` / ≥25% free /
≤150 GB cap); `relocate-corpus` `COPIED_AND_VERIFIED` + `source_preserved:true` + matching manifest sha;
the original corpus at `~/.autoresearch-trading/...` is byte-unchanged (re-hash a sample file).
**Abort if:** volume is internal, not encrypted, wrong name, <25% free, or manifest mismatch.

**Execution evidence (Codex, 2026-08-03):** verified external 2 TB `/dev/disk4`; encrypted ordinary
APFS volume `AR_TRADING_DATA` with no Time Machine destination or Backup role; post-copy preflight
`PASS` with `encrypted:true`, `external:true`, `free_fraction_after:0.9870410083`, and
`phase1_allocation.projected_bytes:25609636917`. Relocation returned `COPIED_AND_VERIFIED`,
`source_preserved:true`, 3,708 files / 21,484,792,678 bytes, manifest SHA-256
`7929a43e6b3e3398991b78ba9e937e006531b76b1b0cd1e5480b35b12cb550d6`; an independent full source
and destination re-hash matched, including sample `aligned/exit_labels/pathd_exit_labels_12m.parquet`
SHA-256 `1959abf3510ba06db040722dc1c81cd435cf733676c6caf6fa16a3789ab40f94`.

An accidental mid-copy eject was fail-closed and preserved at
`/Volumes/AR_TRADING_DATA/reports/quarantine/relocation_interrupted_20260804T002438Z/`; APFS
verification passed before the clean retry. Current-macOS `diskutil` compatibility fixes are committed
as `b0c5c73d` and the 19-test Phase-1 fixture set passes.

*Signed: Codex — 2026-08-03 — status: STAGE 0 COMPLETE; STOP_FOR_CLAUDE_VERIFICATION.*

---

## Stage 1 — Causal training + four-box development verdict (GATE 1)
The owner explicitly waived independent Claude verification of Stage 0 and authorized Stage 1. Codex
completed the full offline sequence over exactly the 215 development sessions. The 36-session firewall
remained closed; no broker, paper order, paid download, runtime flag, default, or promotion path ran.

**Authoritative verdict: `UNDERPOWERED` — STOP.** The learned-entry subset contains 153 trajectories
across 60 sessions but only 3 distinct outer folds, so the code-defined power condition at
`pathd_phase1_replay.py:311-313` fails. The full evaluation index has 1,031 trajectories over 166
sessions and all 5 folds, but that is not the population the gate tests. Negative controls were all
rejected, so `UNDERPOWERED` wins under the frozen verdict precedence.

The result also fails on economics independently of the power stop: learned-entry/learned-exit pooled
PnL was `-$7,024` versus best comparator `matched_random_3` at `-$4,474`; learned one-sided 95% session
bootstrap LCB was `-$150.20`; paired fold deltas were `+$1,955`, `-$4,195`, and `-$310`; and all eight
fee/latency cells were directionally negative (`-$2,450` at 0 s and `-$2,550` at 1/2/5 s). Exit target
skill was positive in 4/5 folds and no negative control cleared the gate, but only 2/7 Gate 1 conditions
passed.

Evidence:
- entry campaign SHA-256 `c7a9ae05b5c66f115a29bc08b7e6abfbde54ff38794cae7e503531964947e71c`;
- exit campaign SHA-256 `69540b7570ba0cc11b2a961062579c4f90169d3972248cf1d051e5e13a443a36`;
- replay semantic SHA-256 `b07784a0280344da8bce5a445804e2eae2cf2bf6bd0375080b6780e55df25807`;
- `replay.json` file SHA-256 `b115b00a68da63f339fc2482a7010ad75b03b9a60c241cf63c2adeda56cc4cf9`;
- `trajectory_outcomes.parquet` SHA-256
  `9fa99aa69f80f3b8d0b333a4ce89d4879dcba130b84f40a341d089a04d53691b`.

Independent reproduction under `reports/phase1_four_box_reproduction/` produced byte-identical replay
and outcome files. All 1,167 baseline trajectories and 9,336 sensitivity partitions passed full
identity/Parquet validation; 1,031 OOF prediction partitions matched the evaluation index exactly. The
focused Phase-1 suite passes 33/33. Two corpus-exposed implementation contradictions were repaired
without dropping receipts or relaxing economics: unique raw-symbol mapping across 18 schema-local
Databento instrument-ID differences, and the frozen zero terminal write-down for 47 stale-at-boundary
paths (`dd4b21bc`). Sensitivity repricing was made calculation-identical but faster (`0b2da32b`).

**Do not proceed to Stage 2.** This result does not establish causal development edge, and the spent
historical holdout must not be reopened to rescue it.

*Signed: Codex — 2026-08-03 — status: STAGE 1 `UNDERPOWERED`; GATE 1 STOP;
STOP_FOR_CLAUDE_VERIFICATION.*

---

## Stage 2 — Runtime decision-parity — ENGINEER SPEC (runnable; the gate signed18 FAILED)
**Objective:** prove the frozen Stage-1 causal model reproduces bit-identical decisions on the
Databento-live-OPRA feature path. Reuse `v4/research/autoresearch_v2/runtime_decision_parity.py` (the
exact gate that invalidated signed18 at 18.44% match).
- Bind the Stage-1 frozen model + its 18-feature contract by SHA. Replay ≥10 **non-firewall** sessions
  through BOTH the training feature path and the live-twin (t−60s) path on matched decision timestamps.
- **GATE 2:** `complete_per_decision_match == 100%` (score, ENTER/WAIT, side, contract all identical);
  feature cells bit-identical within the declared float tolerance. Emit `parity_result.json` (same schema
  as `.../frozen_entry_runtime_decision_parity_.../parity_result.json`). If <100% → name the diverging
  feature and fix or STOP. Expected PASS (the causal model IS built at the t−60s clock the live path uses).
- **Hard stops:** no firewall/holdout; no live-order/broker/paper; offline replay only.

*Signed: Claude Opus 4.8 — 2026-08-03 — status: READY (runnable once Stage 1 = edge).*

## Stage 3 — Live-shadow orchestration — ENGINEER SPEC (NEEDS BUILDING; gated on Stage 2 + live days)
Build a scheduled session-day runner driving `run_pathd_candidate` (databento-no-order) +
`ibkr_paper_dry_run`: the frozen model decides on live Databento OPRA, qualifies the exact SPXW contract,
builds IBKR paper order PREVIEWS, submits NOTHING (assert `paper_order_submitted==false` and
`broker_submit_endpoint_called==false` every decision). Capture the live-only risks the offline gate
cannot: feed timing, quote age, CBBO consolidation, reconnects/dupes, early-close/DST, deterministic ties,
numeric environment.
- **GATE 3:** live decisions == the offline model on identical live inputs; previews correct;
  latency/feed-health within budget. Begins accumulating fresh out-of-sample decisions.
- **Hard stops:** DU paper account only; NO submission; no promotion/default change.

*Signed: Claude Opus 4.8 — 2026-08-03 — status: NOT_STARTED (spec ready; do not build ahead of a Stage-1 edge).*

## Stage 4 — Guarded paper submit + forward confirmation — ENGINEER SPEC (NEEDS BUILDING; gated on Stage 3 + live days)
Integrate a governed paper-SUBMIT path (beyond today's dry-run) via the guarded paper spine
(`v4/live/ibkr_paper_guard.py` + `ibkr_paper_executor.py`): the frozen model submits PAPER orders under
the fail-closed guards (DU account, affordability, daily-loss, forced-flat, no-order-by cutoffs).
PRE-REGISTER the evaluation (session count, power/MDE, practical bar) BEFORE the first paper session.
- **GATE 4:** over the pre-registered fresh-live paper window, does performance confirm the Stage-1 edge
  with adequate power? Paper fills prove EXECUTION; accumulated fresh out-of-sample PnL is the forward
  confirmation substituting for the spent holdout. Honest bound: not proof of real-money profitability.
- **Hard stops:** paper only; NO real money; no paper-DEFAULT or promotion change without a separate owner packet.

*Signed: Claude Opus 4.8 — 2026-08-03 — status: NOT_STARTED (spec ready; gated).*

## Stage 5 — Real-money decision (OUT OF SCOPE)
Separate, owner-authorized, governance-heavy step, only after a sustained pre-registered paper edge + a
formal promotion / real-money-safety packet. This roadmap does not execute it.

*Signed: Claude Opus 4.8 — 2026-08-03 — status: OUT OF SCOPE.*

---

## Ready-to-fire Codex goals

### A. Suite-green (drive-independent; can run now)
See the drafted goal: retire/supersession-gate the v3.2 release-reconciliation (`pathd_entry_dataset.py:403`,
`CORRECTED_V32_CLAUDE_RELEASE_PATH`) so the 4 governance tests run their real assertions and pass; do NOT
xfail (suppresses coverage), do NOT rewrite the immutable receipt, do NOT weaken invariants. Ends
STOP_FOR_CLAUDE_VERIFICATION.

### B. Stage 1 causal training — ENGINEER SPEC (fire AFTER Stage 0 verifies)

**Objective:** train the causal Phase-1 entry+exit models on the mounted drive and emit the four-box
development verdict. Ends `STOP_FOR_CLAUDE_VERIFICATION`.

**Precondition (assert, else abort):** Stage 0 verified — `storage-preflight` `status:"PASS"`,
`relocate-corpus` `status:"COPIED_AND_VERIFIED"` with `source_preserved:true`, corpus present at
`/Volumes/AR_TRADING_DATA/vendor/pathd_2025-08-01_2026-07-31`.

**Environment (persisted in zsh by the runbook; every stage reads these):**
```bash
export AR_TRADING_DATA_ROOT=/Volumes/AR_TRADING_DATA
export AR_TRADING_SCRATCH_ROOT=/Volumes/AR_TRADING_DATA
export AR_TRADING_ARTIFACT_ROOT=/Volumes/AR_TRADING_DATA/artifacts
```

**Run in order (each resumable; run `storage-preflight` before AND after every long stage — the cap is
NOT re-checked per-partition mid-stage):**
```bash
# 1. Materialize the 215 causal entry sessions -> /Volumes/AR_TRADING_DATA/canonical/entry_v2/session=*
PYTHONPATH=. .venv/bin/python -m v4.scripts.run_phase1_exit_model materialize-entry --resume
#   POST: {"status":"COMPLETE"}, written_sessions + skipped_sessions == 215; NO firewall session decoded
#   (development_sessions() = first 215 only; requesting a firewall session raises).

# 2. Train the 5 OOF entry folds + full-dev shadow -> /Volumes/AR_TRADING_DATA/artifacts/entry_v2/
PYTHONPATH=. .venv/bin/python -m v4.scripts.run_phase1_exit_model train-entry
#   Fails closed unless the dataset is EXACTLY the 215 development sessions.
#   POST: artifacts/entry_v2/{campaign.json, trajectory_index.parquet, evaluation_trajectory_index.parquet}.

# 3. Build the 1-second exit trajectories (baseline + 8 fee/latency sensitivities: fee{1.5,2.0} x lat{0,1,2,5})
PYTHONPATH=. .venv/bin/python -m v4.scripts.run_phase1_exit_model build-trajectories --resume \
  --entry-campaign /Volumes/AR_TRADING_DATA/artifacts/entry_v2/campaign.json \
  --emission-lag-receipt v4/audit/autoresearch/thetadata_completed_minute_timing_2026_08_03/shared_emission_lag.json
#   POST: exit_features/session=*/<traj>.parquet + exit_labels/... ; no partial partitions.

# 4. Train the 5 exit folds + full-dev 52-feature exit artifact -> artifacts/exit_v1/campaign.json
PYTHONPATH=. .venv/bin/python -m v4.scripts.run_phase1_exit_model train-exit \
  --entry-campaign /Volumes/AR_TRADING_DATA/artifacts/entry_v2/campaign.json

# 5. Four-box replay -> development verdict
PYTHONPATH=. .venv/bin/python -m v4.scripts.run_phase1_exit_model replay \
  --entry-campaign /Volumes/AR_TRADING_DATA/artifacts/entry_v2/campaign.json \
  --exit-campaign  /Volumes/AR_TRADING_DATA/artifacts/exit_v1/campaign.json
```

**GATE 1 acceptance (exact, from `run_four_box_replay`):** verdict ∈
`{INVALID, UNDERPOWERED, TIER_S_SUPPORTED, TIER_S_NOT_SUPPORTED}`.

> **Corrected 2026-08-03 (Claude).** An earlier revision of this section listed only conditions 1–4 and
> labelled them "exact". The implementation requires **seven**. The code at
> `v4/research/pathd_phase1_replay.py:311-330` is authoritative; this list is now reconciled to it.
> **If doc and code ever disagree again, fix the doc — never loosen the gate to match it.**

`TIER_S_SUPPORTED` requires ALL of:
1. learned-integrated `pooled_net_pnl_dollars` > the **best comparator** `pooled_net_pnl_dollars` (best
   of the fixed exits `stop50_target100`/`stop25_target50` + `matched_random_0..7`);
2. learned `one_sided_95pct_session_bootstrap_lcb_dollars` > 0;
3. positive per-fold delta vs the best comparator in **≥4 of 5** folds;
4. `negative_control_accepted == false` (constant / sign-reversed / shuffled must NOT clear the gate);
5. **`not underpowered`** — `learned_integrated["sessions"] >= 30` AND `outer_fold.nunique() == 5`;
6. **`positive_skill`** — `exit_campaign["positive_target_skill_folds"] > 0`;
7. **all 8 fee/latency sensitivities `directionally_positive`** — every
   `fee{1.5,2.0} x latency{0,1,2,5}` cell must have `delta_dollars > 0`. A single negative cell fails the
   gate.

Verdict precedence: any accepted negative control → `INVALID` (checked first, overrides everything);
else underpowered → `UNDERPOWERED`; else `TIER_S_SUPPORTED` / `TIER_S_NOT_SUPPORTED`.
Any of `INVALID` / `UNDERPOWERED` / `TIER_S_NOT_SUPPORTED` → **STOP** (honest no-edge; the likely outcome).

**Condition 5 result correction (Codex, 2026-08-03):** the 1,031-row evaluation index does cover 166
sessions and all 5 folds, but line 312 tests `learned_rows`, not the full index. The learned-entry subset
has 60 sessions and only folds 0–2, so condition 5 fails and the verdict is `UNDERPOWERED`. Condition 7
also fails: zero of eight fee/latency cells are directionally positive.
Claude concurs — the earlier "power floor already satisfied" note in this section was Claude's error
(it read the full evaluation index instead of the `LEARNED_OOF` subset) and has been retracted.

**Condition 7 is weaker than it looks (Claude, 2026-08-03).** The sensitivity metric is
`delta = learned − comparator`, and BOTH legs carry the same `fee_per_side`, so the fee term cancels
exactly. This is confirmed empirically in `replay.json`: the `fee=3.00` and `fee=4.00` rows are
bit-identical (`-2450, -2550, -2550, -2550` in both). **The 8-cell grid is therefore 4 distinct latency
tests run twice, not 8 independent tests.** Fee robustness is NOT actually being measured by condition 7.
If a future run needs a genuine fee-stress test, it must compare an *absolute* metric (e.g. learned
pooled PnL at each fee) rather than a delta against a same-fee comparator. This did not affect the
2026-08-03 verdict (all cells negative on the latency axis alone), but it must not be mistaken for
fee robustness in any future `TIER_S_SUPPORTED` claim.

**Hard stops:** decode ONLY the 215 development sessions; 36-firewall CLOSED (`holdout_open_count=0`); no
broker/paper-order/promotion/default/cmbp-1; ABORT if allocation would exceed 150 GB or the drive would
fall below 25% free (do NOT delete source or open the firewall to make space). No reward-hacking on a
large number. Highest claim: "Tier-S DEVELOPMENT evidence for a causal Phase-1 entry+exit model; NOT
confirmed (holdout spent), NOT live." STOP_FOR_CLAUDE_VERIFICATION.

**Claude verification checklist (Stage 1):** reproduce the verdict + all four acceptance components;
confirm mutate-future invariance clean; OOF firewall honored (every exit trajectory from an
`OUTER_FOLD_OOF` receipt; full-dev shadow generated none; initial-history receipts training-only);
`holdout_open_count=0`; negative controls FAILED; matched-random exposure sane; skepticism on any large
pooled number.

## Handoff notes for Codex
- The causal clock/fill/label/OOF-firewall are FROZEN and audited — do not modify them.
- The holdout is spent; never reopen the 36-session firewall. Forward confirmation = fresh live paper.
- Update this doc's status board + add a signature line whenever a phase advances.

*Document owner: Claude Opus 4.8. Last updated 2026-08-03.*
