# Path-D Model-Plane Contract — PROPOSAL v2

**STATUS: PROPOSAL v2 (Claude writes → Codex re-reviews → owner signs).** NOT yet
authority. Supersedes v1 (`..._MODEL_PLANE_CONTRACT_PROPOSAL_2026_08_01.md`), which
Codex's design review returned NEEDS-CHANGES. Scope: **Phase 1 (Tier-S feasibility
backtest) only** — authorizes no live decisions, runtime change, or paper trading.

Governance identity: this contract is a constituent of the **Path-D Phase-1 authority
overlay** (see `..._PATH_D_PHASE1_AUTHORITY_OVERLAY_2026_08_01.md`). It does NOT reseal
the parent authority `1d215845`; that and the legacy mask cert / scoped-sync decision /
legacy checker remain byte-unchanged and keep governing the legacy IBKR-decide path.

## Why v2 (Codex findings addressed)

v1 flattened all model inputs to "Databento/ThetaData only" (breaks position-state
parity), lifted the mask open-endedly (bypasses a frozen allowlist), proposed an
unenforceable static source-purity checker, and under-specified the clock. v2 fixes
all four.

## 1. Three-way feature-source classification (replaces flat source-purity)

Every model-facing input is classified into exactly one class:

**A. Market alpha — Databento OPRA + ThetaData SPX ONLY.** All price/quote/microstructure
and index-context signal the model reasons about. Same-vendor train↔live per feed.

**B. Permitted non-alpha state.** Not market alpha, legitimately required, allowed:
- broker-confirmed **fill** price / time / quantity / contract identity (the position's
  actual entry, from IBKR execution — a *fact about our own order*, not market alpha);
- current **position + account/risk state** (open contract, premium at risk, D48/D49
  budget, occupancy);
- the deterministic **clock**, **fees**, and **eligibility masks**.

**C. Forbidden model input.**
- IBKR quote-derived **market alpha** (IBKR bids/asks/sizes/index as a signal);
- **broker outcome labels** (using realized broker results as features);
- **raw vendor greeks** (self-computed IV/delta/gamma only).

Rationale: the exit model's position-PnL path is computed from (A) live market alpha
(current Databento bid) minus (B) the broker-confirmed entry fill — so it is train/live
consistent without treating IBKR as an alpha source. This resolves the v1 conflict.

## 2. Frozen per-namespace allowlists (replaces "mask lifted, select in modeling")

- **Entry namespace: unchanged — exactly the signed 17 features.** No enrichment in
  Phase 1.
- **Path-D lifecycle (exit) namespace: a new explicit frozen allowlist.** Because the
  exit plane is same-vendor Databento, the microstructure the mask hid is admissible
  *as an enumerated allowlist*, not open-ended. The allowlist explicitly resolves each
  previously-ambiguous field — bid/ask/mid, spread, sizes, volume, open interest,
  self-computed IV/delta/gamma, breakeven, aggregated statistics, and named derived
  path features — each marked admit/forbid with a one-line reason. Multiplicity-aware:
  the allowlist is frozen before fitting and counted in the family budget.
- Legacy v1/v2 feature-contract code (`v4/live/protocol101_feature_contract.py`) is NOT
  mutated. Path D gets a **new contract + builder** in a Path-D namespace; the legacy
  mask code/artifacts stay immutable for the legacy path.
- The scoped-synchronization decision continues governing **entry** feature admission;
  only its cross-vendor-transfer role retires for Path D.

## 3. Fail-closed feature-lineage manifest (replaces the unenforceable static checker)

Static "does this feature come from IBKR" detection is not reliably inferable from code
paths. Instead, every model-facing feature MUST carry a manifest record; anything
without a valid record is rejected (fail-closed):
- component namespace (entry | exit);
- feature role (alpha | position-state | mask | label | audit);
- transitive source leaves (the raw fields it ultimately derives from);
- vendor, product, schema, historical/live mode;
- clock semantics (event vs source-receipt vs local-receipt);
- transformation + builder hashes;
- exact historical/live twin (the live builder that reproduces it);
- explicit rejection of unknown lineage and renamed proxies.

**Negative fixtures the checker must reject:** an IBKR bid renamed as `pnl`; a vendor
greek renamed as `internal_delta`; a future `ts_recv`; a feature with no live twin; an
unregistered feature. (Checker code is built at implementation/reseal time by Codex.)

## 4. Complete clock definition

Path-D features use one governed clock spec (not just "received timestamp"):
- distinct **event**, **source-receipt**, and **local-receipt** timestamps;
- **interval closure** (a second/minute is decidable only once complete);
- historical `ts_recv` vs live local-receipt correspondence (the historical/live twin);
- **cross-feed watermarking** (Databento OPRA + ThetaData SPX aligned to the slowest
  completed feed before a decision);
- allowed **latency/quote-age** bound; and **when an action may consume the completed
  second** (entry: completed minute; exit: completed second).

## 5. Tier and scope

Phase-1 sub-minute substrate = **Tier-S `cbbo-1s`** (A7) — consistent with FT2-60 v2 and
the sequencing plan. `cmbp-1` raw-event-path (Tier-T) is a Phase-2 promotion requirement,
not used here. Tier-S conclusions are feasibility-only (A7, authority L1065).

## Scope — what does NOT change

The 17 entry features + minute entry policy; SPXW identity / 42-slot ladder / causal
firewalls; D48/D49 / one-account serial accounting / affordability / one-contract /
15:55 forced-flat; the FT2-04 entry label families; simulator v5 core; the parent
authority, legacy mask cert, scoped-sync decision, and legacy checker (all immutable).
No live decisions, runtime change, execution policy, A1 rewrite, graph reissue, or paper
trading are authorized here (Phase-2 backlog).

## Implementation + verification path (for Codex)

1. Re-review against the authority, A7, the FT2 chain, and the audit: is the three-way
   split complete and leak-free? Are the exit allowlist and lineage manifest enforceable?
   Is the clock spec sufficient? Any contract still hard-requiring the mask on the Path-D
   path? If a blocking defect: STOP and report.
2. If sound: implement as a governed signed contract under the Path-D namespace; build
   the fail-closed lineage-manifest checker + negative fixtures; register this contract's
   hash in the Path-D Phase-1 overlay. Do NOT touch the parent authority or legacy checker.
3. Claude verifies (hashes; three-way split leak-free; lineage checker real + negative
   fixtures fail-closed; legacy artifacts byte-unchanged; scope excludes live).
4. Owner signs (as part of the overlay).

No training, download, broker/recorder contact, live decision, or runtime change is
authorized by this contract.
