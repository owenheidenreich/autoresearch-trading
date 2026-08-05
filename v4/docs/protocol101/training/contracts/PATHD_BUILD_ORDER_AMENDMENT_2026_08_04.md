# Path-D Build Order Amendment — Freeze the Certification Queue

> **SUPERSEDED 2026-08-05 — see [`STATUS.md`](../../../../../STATUS.md) for current status.**
> **§2 is SUPERSEDED — do NOT run its `bootout` block.** The Track-A capture was reopened by owner decision; see the [restart record](PATHD_PROGRAMME_RESTART_RECORD_2026_08_05.md).

Status: **OWNER-AUTHORIZED AND BINDING**

Effective date: 2026-08-04

Signature record: owner signed 2026-08-04 via explicit sign-off in the Fable review session
(recorded by Claude Fable 5). The §2 stand-down was executed by the owner the same day and
independently verified: no `com.autoresearch.tracka.*` labels loaded in launchd, no tracka plists
remaining in `~/Library/LaunchAgents/`.

Authorization source: the owner-approved
[`PATHD_FABLE_REVIEW_IMPLEMENTATION_PLAN_2026_08_04.md`](../execution/PATHD_FABLE_REVIEW_IMPLEMENTATION_PLAN_2026_08_04.md),
Deliverable 4, arising from the 2026-08-04 outside review of
[`FABLE_REVIEW_BRIEF_2026_08_04.md`](../research/FABLE_REVIEW_BRIEF_2026_08_04.md).

This amendment supersedes the Phase-0 → Phase-1 sequencing in
[`PATHD_BUILD_ORDER_2026_08_04.md`](PATHD_BUILD_ORDER_2026_08_04.md). Existing Phase-0 artifacts, receipts,
and the signed admission ledger remain unchanged and continue to describe the law in force when they were
created. Nothing below invalidates them; they simply stop being a work queue.

---

## Why the queue is frozen

The Build Order sequences **certify → enforce → IBKR → train**. That order is still correct. The problem is
what it is certifying *for*.

The 65 features sitting at `missing_required_receipts` / `parent_family_not_admitted` exist to feed an entry
and exit model on the **0DTE long-premium class**. As of the Phase-1 closeout that class is
**closed, structurally**: gross directional expectancy is **−$13.00/trade with all friction removed**,
negative in 5 of 5 folds, before any model, feature, or execution question is asked. Certifying 65 features
to train a model that is barred from being trained is work with no consumer.

The certification apparatus itself is sound and stays exactly as it is. What changes is that it stops
running ahead of a strategy question it cannot answer.

## 1. The Phase-0 certification queue is FROZEN

Until an owner-authorized strategy class exists **with measured friction it can clear**:

- No certification work on the 65 blocked features.
- No ledger regeneration.
- No wave re-freezing (Wave 3 as written specifies uncertified greeks and stays unfrozen).
- No capture consumption.

**What does not change.** `feature_admission_ledger.json` and the `admitted_feature_matrix` fail-closed
enforcement stay **exactly as-is**. The ledger's vocabulary is binary — `ADMITTED` / `BARRED` — and has no
term for "frozen"; the freeze therefore lives in this document, not in the ledger. Enforcement continues to
refuse any non-admitted feature, which is the behaviour we want preserved. The 8 `ADMITTED` calendar and
geometry features remain admitted.

**What unfreezes it.** An owner-authorized strategy class whose friction has been *measured* and shown to be
clearable. Not a hypothesis about one — a measurement.

## 2. The 2026-08-05..07 Databento capture is STOOD DOWN

Owner decision, 2026-08-04. The capture was armed to measure the `cbbo-1m` multi-session arrival clock,
which unblocks 65 features — and per §1 those features have no consumer. The capture is at zero marginal
cost, but running it implies a queue that is now frozen.

**Owner runs this — the executing agent does not.** Run **before 09:28 ET on 2026-08-05**, or the first
launchd window fires:

```bash
launchctl bootout gui/$(id -u)/com.autoresearch.tracka.capture
launchctl bootout gui/$(id -u)/com.autoresearch.tracka.capture.midday
rm ~/Library/LaunchAgents/com.autoresearch.tracka.capture.plist
rm ~/Library/LaunchAgents/com.autoresearch.tracka.capture.midday.plist
```

These were always temporary jobs, never a standing service. The 60-second smoke capture already performed
on 2026-08-04 stands as proof the live path works end to end (`cbbo-1m` p99 **584.6 ms**), and that proof is
not lost by standing down.

## 3. Roadmap addendum

An addendum banner — **not a rewrite** — is appended to the Track-A row of
[`PATHD_PHASE1_TO_LIVE_ROADMAP_AND_STATUS_2026_08_03.md`](../research/PATHD_PHASE1_TO_LIVE_ROADMAP_AND_STATUS_2026_08_03.md)
recording that its 5-session capture declaration (2026-08-05, 06, 07, 10, 11) was superseded by the
3-session v5 declaration and is now stood down entirely.

## 4. What is explicitly NOT frozen

- **The execution plane.** IBKR paper guard and executor are proven, strategy-agnostic, and cost nothing to
  keep warm. Contract identity (`SPXW  260804C07775000`) bridges training to execution and stays closed.
- **The research machinery.** Gate, autoresearch loop, prior-art blocking, maxT, negative controls.
- **Read-only analysis on owned data.** The SPX directional-skill screen ran under this allowance.
- **The Trader Charter** and its Amendment 1.

*Drafted: Claude Opus 5 — 2026-08-04. Signed by the owner and effective 2026-08-04; signature and
stand-down verification recorded by Claude Fable 5.*
