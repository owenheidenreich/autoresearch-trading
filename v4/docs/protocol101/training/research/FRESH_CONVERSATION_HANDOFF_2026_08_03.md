# Fresh-Conversation Handoff — Path-D Phase-1 (2026-08-03)

Seed a new Claude or Codex conversation with this. It captures the current state, the division of
labor, the immediate next actions, and the hard-won rules so a cold session does not re-derive context
or repeat past mistakes. **Single source of truth for the plan:**
[`PATHD_PHASE1_TO_LIVE_ROADMAP_AND_STATUS_2026_08_03.md`](PATHD_PHASE1_TO_LIVE_ROADMAP_AND_STATUS_2026_08_03.md)
(engineer-grade, signed status board). Full history: the learnings ledger
(`v4/docs/protocol101/training/execution/PROTOCOL101_WALKING_SKELETON_LEARNINGS_LEDGER_2026_07_30.md`).

## Division of labor (owner's setup)
- **Claude** — plans, writes engineer specs + Codex goals, reviews/audits, and **verifies-don't-rubber-
  stamp** (reproduce hashes, re-run suites, hunt leakage/reward-hacking). Lighter compute.
- **Codex** — implements and runs the heavy/long-compute work (training campaigns, large test suites,
  multi-hour goals). Has the compute budget.
- **Loop:** Claude drafts a goal/spec → Codex executes and ends with `STOP_FOR_CLAUDE_VERIFICATION` →
  Claude verifies independently → repeat. Cross-review both directions (Claude writes / Codex reviews and
  vice-versa). The owner drives Codex; Claude does not launch Codex.

## Current state (honest one-paragraph)
Path-D = a maximally-honest **learned SPXW 0DTE long-options trader** (minute-cadence entry + 1-second
exit), trained on **Databento OPRA + ThetaData**, deciding live on **Databento live OPRA**, executing on
**IBKR (execution-only)**. The autoresearch_v2 "confirmed edge" (`signed18`, +$540/session) was
**INVALIDATED** by the runtime decision-parity gate: training used a **60-second SPX look-ahead**
(ThetaData bar stamped `t`, whose close is only available at `t+60s`). It is quarantined
`INVALID_EXPERIMENT`; the protected holdout was opened on it and is **SPENT**. A **causal Phase-1
rebuild** (entry/exit/replay/storage) now rebuilds features from raw at the **t−60s** clock; it is
implemented, **Claude-audited PASS** (clock, conservative fill/label law, entry→exit OOF firewall),
fixture-green, and **purchase-ready** (footprint 89.69 GB < 150 GB cap). **Stage 0 is complete:** the
2 TB external encrypted-APFS SSD is mounted at `/Volumes/AR_TRADING_DATA`, the already-owned 12-month
corpus was copied and independently checksum-verified (manifest `7929a43e...550d6`), and the internal
source remains preserved. The SSD is the training WORKSPACE, not a data download. **No causal model has
been trained yet.** Stage 1 is ready but still requires explicit owner training authorization;
`NO_INCREMENTAL_EDGE` is the likely honest outcome.

## Immediate next actions (gated; see the roadmap for exact commands)
1. **Stage 0 — COMPLETE, pending Claude verification.** Preflight and relocation passed; source and
   destination independently match 3,708 files / 21,484,792,678 bytes.
2. **Stage 1 — causal training (Codex; explicit owner authorization required).** `materialize-entry → train-entry →
   build-trajectories → train-exit → replay` → four-box verdict. **GATE 1** = `TIER_S_SUPPORTED` (beat best
   comparator pooled + bootstrap LCB>0 + ≥4/5 positive fold deltas + no negative control accepted); else STOP.
3. **Suite-green — COMPLETE:** superseded v3.2 release reconciliation retired in `fa1d9b08`; immutable
   evidence preserved and the governance test file passes 7/7.
4. Stages 2–5 (parity → live-shadow → guarded paper → real-money) are specified + gated in the roadmap;
   Stage 3–4 need building; **do not build ahead of a Stage-1 edge.**

## Critical rules & hard-won lessons (do not repeat these)
- **The holdout is SPENT.** Never reopen the 36-session firewall (protected 30 + 6 smoke). Forward
  confirmation = **fresh live paper data**, not the historical holdout.
- **Causal clock is t−60s.** The SPX bar stamped `t` is a look-ahead (the exact bug that killed signed18).
  Never source SPX context from the bar stamped `t`. Verify **feature-availability-clock parity**, not just
  future-outcome guards — the mutate-future lint did NOT catch the clock leak; the runtime-parity gate did.
- **Verify-don't-rubber-stamp — and verify the right thing.** Past misses: Claude's lean loop had a
  non-causal `gt_session_q60` (future within-session scores); v3.2 reported "175 tests passed" but the real
  fit was faked (consumer-only tests), caught only on an independent full-suite rerun; the signed18 edge was
  a clock leak Claude's first sign-off missed. Reproduce hashes, re-run suites yourself, and check parity.
- **No reward-hacking.** A big improvement is a reason for *more* skepticism. Don't optimize a proxy that
  diverges from generalizing profit. NULL / `NO_INCREMENTAL_EDGE` is a first-class, respectable outcome.
- **Governance ≠ progress.** The heavy governed Path-D pipeline hit 6 frozen generations + ~30k lines and
  never once ran a real fit; the lean/Phase-1 path answered the real question. Cheap real experiments
  before governance; run the real fit before elaborate fake-tested pipelines.
- **Paper proves execution, not alpha. Real money is separate** and owner-authorized (a distinct governance
  packet). Don't plan a straight shot to real money.
- **Don't build ahead of a gate**, don't rewrite frozen receipts to pass tests, don't spawn subagents
  unless the owner asks.

## Safety hard-stops (from CLAUDE.md — require explicit owner authorization + a fresh safety read)
IBKR/broker/order/live/paper-submit/market-data; paid Databento/Polygon downloads or broad backfills
(incl. **cmbp-1 ~300 GB Phase-2**); model training/threshold-tuning/promotion; runtime-flag/launchd/plist
edits; cleanup outside a manifest-backed quarantine. The Stage-0 disk erase is **DESTRUCTIVE + owner-
supervised** (manual disk-ID confirmation). No paper-default or promotion change without an owner packet.

## Where a fresh session should start
- **Claude:** CLAUDE.md loads automatically; then read this handoff → the roadmap/status doc → the ledger
  tail → auto-memory (`project-protocol101-walking-skeleton`, `feedback-protocol101-conversation-style`).
- **Codex:** read this handoff → the roadmap/status doc (engineer-grade Stage 0/1 commands + GATE 1
  acceptance) → CLAUDE.md safety rules. Execute only the goal the owner hands you; end with
  `STOP_FOR_CLAUDE_VERIFICATION`.
- **Git:** branch `v4/phase-0`; local commits include `fa1d9b08` (v3.2 supersession) and `b0c5c73d`
  (current-macOS storage-preflight compatibility); pushing is a separate owner action.

## Working style (owner)
Plain language + trader analogies; `AskUserQuestion` for genuine forks (recommendation first); challenge
owner assumptions with evidence; own your own errors and correct them; update the ledger + this handoff +
the status board as phases advance; keep commits curated (no data/secrets/checkpoints/backlog).

*Signed: Codex — 2026-08-03 — Stage 0 complete; STOP_FOR_CLAUDE_VERIFICATION. Keep this current as phases advance.*
