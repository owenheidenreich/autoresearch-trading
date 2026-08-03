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
| SSD setup runbook (fail-closed) | DONE (staged, not executed) | — |
| Phase-1 fixture suite | DONE (green: entry/exit/replay/storage 17 passed; Phase-1 subset 20/20) | — |
| Milestone commit + memory/ledger | DONE (`cba15b1a`, `9ae807f5`) | — |
| Suite-green (retire stale v3.2 reconciliation) | PENDING_CODEX (goal drafted below) | Codex (frozen v3.2 machinery) |
| Stage 0 — drive setup | BLOCKED_ON_DRIVE | physical 2 TB SSD + owner disk-ID confirm |
| Stage 1 — causal training + four-box | BLOCKED_ON_DRIVE | Stage 0 |
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

## Stage 0 — Drive setup (owner-supervised, on arrival)
Run `PHASE1_SSD_SETUP_RUNBOOK.md`: identify disk → **owner confirms exact disk ID** → erase encrypted
APFS `AR_TRADING_DATA` → encrypt → create dirs → env → `storage-preflight --create-roots` →
`relocate-corpus --source ~/.autoresearch-trading/pathd_2025-08-01_2026-07-31` → verify.
**Verify (Claude):** preflight PASS (encrypted external APFS / name / ≥25% free / ≤150 GB);
`relocate-corpus` = `COPIED_AND_VERIFIED`, `source_preserved: true`, manifest sha matches; source
byte-unchanged.

*Signed: Claude Opus 4.8 — 2026-08-03 — status: BLOCKED_ON_DRIVE. Next: owner runs runbook when SSD lands.*

---

## Stage 1 — Causal training + four-box development verdict (GATE 1)
Codex goal is drafted (below), ready to fire once Stage 0 verifies.
**GATE 1:** learned entry/exit beats P5 / matched-random / nearest-ATM pooled AND ≥4/5 folds; negative
controls FAIL; mutate-future clean; latency/fee sensitivities hold. `NO_INCREMENTAL_EDGE` → STOP/reconsider.
Edge → freeze model + 18-feature contract by SHA; proceed to Stage 2.

*Signed: Claude Opus 4.8 — 2026-08-03 — status: BLOCKED_ON_DRIVE (goal staged). Next: fire after Stage 0.*

---

## Stages 2–5
Per `~/.claude/plans` roadmap / this doc's status board. NOT_STARTED; each gated on the prior passing and
(for 3–4) on live market days. Stage 3 (live-shadow orchestration) and Stage 4 (governed paper submit)
require building; do NOT build ahead of a Stage 1 edge.

*Signed: Claude Opus 4.8 — 2026-08-03 — status: NOT_STARTED, correctly gated.*

---

## Ready-to-fire Codex goals

### A. Suite-green (drive-independent; can run now)
See the drafted goal: retire/supersession-gate the v3.2 release-reconciliation (`pathd_entry_dataset.py:403`,
`CORRECTED_V32_CLAUDE_RELEASE_PATH`) so the 4 governance tests run their real assertions and pass; do NOT
xfail (suppresses coverage), do NOT rewrite the immutable receipt, do NOT weaken invariants. Ends
STOP_FOR_CLAUDE_VERIFICATION.

### B. Stage 1 causal training (fire AFTER Stage 0 verifies)
```text
Goal: PHASE-1 CAUSAL TRAINING + FOUR-BOX DEVELOPMENT VERDICT (on the mounted AR_TRADING_DATA drive).
Ends STOP_FOR_CLAUDE_VERIFICATION. PRECONDITION: Stage 0 verified (storage-preflight PASS, corpus
relocated + checksum-verified, source preserved).

Run, in order, resumable, on the external drive (env set to /Volumes/AR_TRADING_DATA):
  run_phase1_exit_model materialize-entry --resume
  run_phase1_exit_model train-entry
  run_phase1_exit_model build-trajectories --resume --entry-campaign <.../entry_v2/campaign.json>
    --emission-lag-receipt v4/audit/autoresearch/thetadata_completed_minute_timing_2026_08_03/shared_emission_lag.json
  run_phase1_exit_model train-exit --entry-campaign <.../entry_v2/campaign.json>
  run_phase1_exit_model replay --entry-campaign <...> --exit-campaign <.../exit_v1/campaign.json>

Emit the four-box development verdict from the frozen vocabulary. Report: per-fold + pooled learned
entry/exit vs P5 / matched-random / nearest-ATM; negative-control results (must FAIL); mutate-future
audit; latency/fee sensitivities; the storage allocation stayed <150 GB with a preflight before AND
after each long stage.

HARD STOPS: decode ONLY the first 215 development sessions; the 36-session firewall (incl. protected 30)
stays CLOSED (holdout_open_count=0); no broker/paper-order/promotion/default/cmbp-1; no reward-hacking on
a large number. Highest claim: "Tier-S development evidence for a causal Phase-1 entry+exit model; not
confirmed (holdout spent), not live." STOP_FOR_CLAUDE_VERIFICATION.
```

## Handoff notes for Codex
- The causal clock/fill/label/OOF-firewall are FROZEN and audited — do not modify them.
- The holdout is spent; never reopen the 36-session firewall. Forward confirmation = fresh live paper.
- Update this doc's status board + add a signature line whenever a phase advances.

*Document owner: Claude Opus 4.8. Last updated 2026-08-03.*
