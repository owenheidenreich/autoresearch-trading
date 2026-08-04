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
`PENDING_CODEX` · `NOT_STARTED` · `GATE` (a stop/continue decision point).

## Status board

| Item | Status | Blocking dependency |
|---|---|---|
| Causal Phase-1 implementation (entry/exit/replay/storage) | DONE (audited PASS, committed `cba15b1a`) | — |
| Footprint estimate (89.69 GB < 150 GB cap) | DONE | — |
| SSD setup runbook (fail-closed) | DONE (executed on verified `/dev/disk4`) | — |
| Phase-1 fixture suite | DONE (green: entry/exit/replay/storage 19 passed) | — |
| Milestone commit + memory/ledger | DONE (`cba15b1a`, `9ae807f5`) | — |
| Suite-green (retire stale v3.2 reconciliation) | DONE (`fa1d9b08`; governance file 7/7 passed) | Claude independent verification |
| Stage 0 — drive setup | DONE (preflight + relocation + independent hashes passed) | Claude independent verification |
| Stage 1 — causal training + four-box | READY | explicit owner training authorization |
| Stage 2 — runtime decision-parity | NOT_STARTED | Stage 1 = edge |
| Stage 3 — live-shadow orchestration | NOT_STARTED (needs building) | Stage 2 pass + live days |
| Stage 4 — guarded paper submit + confirmation | NOT_STARTED (needs building) | Stage 3 + live days |
| Stage 5 — real-money decision | OUT OF SCOPE | separate owner+governance packet |

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
Codex goal is drafted (below), ready to fire once Stage 0 verifies.
**GATE 1:** learned entry/exit beats P5 / matched-random / nearest-ATM pooled AND ≥4/5 folds; negative
controls FAIL; mutate-future clean; latency/fee sensitivities hold. `NO_INCREMENTAL_EDGE` → STOP/reconsider.
Edge → freeze model + 18-feature contract by SHA; proceed to Stage 2.

*Signed: Codex — 2026-08-03 — status: READY, awaiting explicit owner authorization for model training
and independent Claude verification of Stage 0.*

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
`{INVALID, UNDERPOWERED, TIER_S_SUPPORTED, TIER_S_NOT_SUPPORTED}`. `TIER_S_SUPPORTED` requires ALL of:
1. learned-integrated `pooled_net_pnl_dollars` > the **best comparator** `pooled_net_pnl_dollars` (best
   of the fixed exits `stop50_target100`/`stop25_target50` + `matched_random_0..7`);
2. learned `one_sided_95pct_session_bootstrap_lcb_dollars` > 0;
3. positive per-fold delta vs the best comparator in **≥4 of 5** folds;
4. `negative_control_accepted == false` (constant / sign-reversed / shuffled must NOT clear the gate).
Any of `INVALID` / `UNDERPOWERED` / `TIER_S_NOT_SUPPORTED` → **STOP** (honest no-edge; the likely outcome).

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
