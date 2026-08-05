# Path D Amendment Sequencing Plan (2026-08-01)

**STATUS: PLANNING — owner-agreed phasing.** Turns the Codex blast-radius audit
(7 amendment areas) into a small **Phase-1 set to do now** vs a **Phase-2 backlog**
deferred until backtest edge is proven. Not itself an authority amendment.

**Phasing principle (matches the prove-before-pay financing):** do the *minimum*
governed change to BACKTEST Path D and prove edge; defer the entire heavy live
apparatus until the model earns it. Most of the audit's blocking items are Phase-2
(live) work.

---

## Phase 1 — minimum to backtest Path D and prove edge

**Cheap in both $ and storage:** ~$208 data (12-month, `cbbo-1s` Tier-S, ~5 GB) +
ThetaData SPX. No `cmbp-1` (that's ~300 GB, Phase 2). No live subscription burn
beyond the one month already bought.

**Data plane:** Databento OPRA `cbbo-1s` (1-second exit substrate, Tier-S per A7) +
`ohlcv-1m`/`cbbo-1m` (minute entry) + `definition`/`statistics` + ES/VX context;
ThetaData SPX cash. All trailing-12-months (free under the sub).

**Governed changes (the minimal reseal set):**
1. **Path-D model-plane contract (NEW, governed, owner-signed).** Supersedes the
   *governing role* of the microstructure-mask cert + scoped-synchronization decision
   **for the Path-D path only** (old artifacts stay immutable evidence for the legacy
   IBKR-decide path). Defines: Databento OPRA + ThetaData SPX; **same-vendor train↔live
   → unmasked microstructure admissible** (sizes/spread/tick now usable); single
   decision clock; self-computed greeks (raw vendor greeks still prohibited); causal/
   provenance admission retained.
2. **FT2-60 exit contract (sign the drafted design** —
   `PROTOCOL101_ONE_SECOND_EXIT_OBJECTIVE_ARCHITECTURE_DESIGN_2026_08_01.md`, after
   Codex review). Additive **1-second exit** labels/tensor/objective (action-advantage,
   convexity) layered on the **frozen minute entry**; trained on OOF trajectories from
   the frozen entry model.
3. **Additive FT2-08 exit extension.** A 1-second exit tensor + exit label grid
   (Tier-S / `cbbo-1s`), ADDITIVE to the existing minute entry tensor/labels (entry
   side unchanged). Regenerate only the downstream hashes this touches.
4. **Provisional backtest fill model (documented, NOT the full A1 rewrite).** A
   conservative, latency-sweep-calibrated marketable-limit-through-quote exit fill for
   backtest only, explicitly marked "backtest-grade, superseded by the Phase-2 A1
   order state machine." Never presented as live-fill-parity.

**Explicitly unchanged in Phase 1** (audit safe-list): minute entry policy + the 17
synchronized entry features; SPXW identity / 42-slot ladder / causal firewalls /
self-computed greeks; the six FT2-04 label families (entry side); D48/D49 / one-account
serial accounting / affordability / one-contract / 15:55 forced-flat; simulator v5
core; IBKR paper execution guards; A7 Tier-S/Tier-T provenance.

**Phase-1 GATE:** does the Path-D trader (minute entry + learned 1s exit) show
**provable backtest edge** on historical Databento+ThetaData vs the baseline panel?
- **No edge →** stop; we spent ~$208 + design time, not a redesign campaign.
- **Edge →** proceed to Phase 2.

## Phase 2 — the heavy live apparatus (ONLY if Phase-1 edge is proven)

**Data plane:** add `cmbp-1` raw-event-path (Tier-T) for **trusted** floor/stop labels
(derive-to-1s-and-discard-raw or monthly chunks to manage the ~300 GB; free under the
sub, trailing 12mo).

**Governed changes (the deferred backlog from the audit):**
1. **Execution-policy contract + deterministic broker-facing risk governor.**
   Broker-quote-anchored bounded marketable-limit (fresh IBKR quote at submit, one
   valid tick through, IOC/short timer, bounded requotes, reconcile-before-next, exit
   no-fill escalation); partial/late-fill/disconnect handling; Databento-loss →
   forced-flat via IBKR.
2. **Full A1 rewrite** — fill law → the complete order state machine, used IDENTICALLY
   across training labels, serial replay, realized PnL, floor, and live.
3. **Machine-graph reissue** — FT2-24/25/26 cross-vendor admission → same-vendor
   feature/provenance admission; FT2-92 historical/IBKR transfer → execution-
   reconciliation certification; re-validate + re-hash.
4. **Supersede the synchronization gates** — the signed scoped-synchronization decision
   and masked-v2 cert's remaining live-gate role; recorder-first parity demoted to
   optional execution calibration.
5. **Runtime / promotion packet** — the Databento+ThetaData decision service + IBKR
   execution adapter boundary; only after its gates pass does the paper-default
   registry change (Protocol160 → Path D runtime).
6. **Coordinated fill-law resupersession** — the ~39 FT2-04/05/08/10/11 files pinning
   the old fill-law/minute hash get coordinated supersession + regenerated cross-
   contract checker; old receipts stay immutable.

**Phase-2 GATE:** paper-readiness (no-order live shadow, execution-bridge validated on
paper days) → guarded paper trading. Then, separately, any real-money discussion.

## Immutability

Old artifacts — masked-v2 cert, recorder captures, paired-diff reports, old FT2
receipts, the microstructure-mask code — remain **immutable reproducibility evidence**.
Path D **supersedes their governing role**; it does not rewrite or delete them.

## Dependency ordering

WS: buy 12mo data (done-ish) → Codex reviews FT2-60 → Phase-1 reseal set (1-4) →
build + backtest → **edge gate** → [if edge] Phase-2 backlog in the order above →
paper-readiness gate → paper.

## Why this is safe

No live subscription burn, no execution machine, no graph reissue, no 39-file
resupersession, and no ~300 GB download are required to answer "does Path D have
edge?" The audit's length becomes a *deferred backlog contingent on edge*, not a
prerequisite.
