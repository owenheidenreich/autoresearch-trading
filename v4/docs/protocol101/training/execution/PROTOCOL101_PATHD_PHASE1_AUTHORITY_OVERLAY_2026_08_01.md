# Path-D Phase-1 Authority Overlay — PROPOSAL

**STATUS: PROPOSAL (Claude writes → Codex reviews → owner signs).** NOT yet authority.
The governance-identity resolution for Path-D Phase-1: an **immutable overlay** that
composes the parent authority with the Path-D Phase-1 constituent contracts, WITHOUT
resealing or rewriting the parent authority or any legacy artifact. Scope: Tier-S
feasibility backtest only.

## Why an overlay (Codex finding A/E)

The parent authority `1d215845` hard-requires completed-minute exits (L194), the D24
completed-minute floor (L500), and the signed feature restrictions (L271); the legacy
cross-contract checker pins that hash. A canonical reseal cannot happen while Phase-2
re-pins (execution policy, A1 rewrite, graph reissue, 39-file resupersession) are
deliberately deferred. The overlay resolves this: Path-D Phase-1 work pins the OVERLAY
(which cites the parent + the new constituents); the parent authority, legacy receipts,
mask certificate, and legacy checker stay **byte-unchanged** and keep governing the legacy
IBKR-decide path. Precedence, for the Path-D path only: overlay > parent for the specific
provisions the constituents supersede (decision-plane, exit cadence/labels, fill law);
everything else inherits the parent unchanged.

## What the overlay pins

| Component | Artifact | Hash |
|---|---|---|
| Parent authority | `..._CONSOLIDATED_AUTHORITY_2026_07_28.md` | `1d215845…` (unchanged) |
| Graph topology | `PROTOCOL101_FULL_TRADER_GRAPH_V2.json` | `9955085a…` (unchanged) |
| A7 (Tier-S/Tier-T) | staged sub-minute amendment | (A7 hash) |
| Model-plane v2 | `..._PATH_D_MODEL_PLANE_CONTRACT_PROPOSAL_V2_2026_08_01.md` | (to compute at reseal) |
| FT2-60 exit v2 | `..._ONE_SECOND_EXIT_OBJECTIVE_ARCHITECTURE_DESIGN_V2_2026_08_01.md` | (to compute) |
| FT2-08 exit ext. | `..._FT2_08_PATHD_1S_EXIT_TENSOR_LABEL_CONTRACT_2026_08_01.md` | (to compute) |
| Provisional fill law | `..._PATHD_PHASE1_PROVISIONAL_FILL_LAW_2026_08_01.md` | (to compute) |

Hashes for the new constituents are computed at implementation/reseal time by Codex; the
overlay is the single object owner-signs.

## Provisions the overlay establishes (Path-D path only)

1. **Decision plane:** train + decide on Databento OPRA + ThetaData SPX; IBKR execution-
   only (model-plane v2). Three-way feature-source split; fail-closed lineage manifest.
2. **Exit cadence + labels:** 1-second learned exit (FT2-60 v2) on Tier-S `cbbo-1s`; the
   additive FT2-08 exit tensor/labels; deployable `Q(hold)`/`Q(exit)` (no oracle).
3. **Fill law:** the frozen Phase-1 provisional fill/no-fill/latency law, used identically
   across labels/replay/PnL/occupancy.
4. **Floor:** deterministic catastrophic backstop (FT2-60 §6), not the D24 minute-forecast
   floor — for the Path-D exit path only.
5. **Acceptance + calibration:** economic serial-replay ≥4/5 outer folds (authority §5.4);
   action-conditioned calibration gate (authority L449); PR/AP diagnostic-only.
6. **Claim boundary:** Tier-S feasibility (A7 L1065) — no promotion/paper/loss-control
   claim.

## What the overlay does NOT do (deferred to Phase 2)

No production execution policy / risk governor; no full A1 rewrite; no graph reissue; no
superseding the synchronization gates in the parent; no runtime/promotion/paper-default
change; no 39-file fill-law resupersession; no `cmbp-1` Tier-T acquisition. The parent
authority's live/minute-exit/floor provisions remain in force for the legacy path.

## Checker

A Path-D-namespace checker validates: overlay pins the correct parent + graph hashes
(unchanged); all constituents present + hash-consistent; the fail-closed feature-lineage
manifest (with negative fixtures); the single-shared-fill-law hash across consumers; and
that NO legacy artifact (parent authority, mask cert, legacy checker, legacy FT2 receipts)
changed. It does not modify the legacy checker.

## Implementation + verification path (for Codex)

1. Re-review the full Path-D Phase-1 set together (model-plane v2, FT2-60 v2, FT2-08 exit,
   provisional fill law, this overlay) against the A–E rubric. If any blocking defect
   remains: STOP and report.
2. If sound: compute constituent hashes, assemble the overlay, build the Path-D-namespace
   checker + negative fixtures, and confirm all legacy artifacts are byte-unchanged. Do NOT
   reseal the parent authority.
3. Claude verifies (reproduce overlay + constituent hashes; parent/graph/legacy unchanged;
   checker green incl. negative fixtures; claim boundary intact).
4. **Owner signs the overlay** (single signature covering the constituents) → Phase-1
   implementation (build tensors, backtest) is then authorized.

No training, download, broker/recorder contact, live decision, runtime change, or paper
order is authorized by this overlay.
