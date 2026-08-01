# Path-D Model-Plane Contract — PROPOSAL

**STATUS: PROPOSAL (Claude writes → Codex reviews → owner signs).** NOT yet
authority. The first Phase-1 governed change: defines the Path-D model decision
plane and supersedes the *governing role* (not the existence) of the
microstructure-mask certificate and the scoped-synchronization decision, for the
Path-D path only. Prepared 2026-08-01. Scope: **Phase 1 (backtest) only** — this
authorizes no live decisions, no runtime change, no paper trading.

Precedence intent: for the Path-D path, this signed contract outranks the
masked-v2 certificate and the scoped-synchronization decision; those remain
immutable evidence governing the legacy IBKR-decide path.

## Why

Path D makes the model decision plane **same-vendor** (train and live both on
Databento for options, ThetaData for SPX). The microstructure mask
(`protocol101-live-v2-microstructure-masked`) existed solely to survive
cross-vendor drift between Databento (train) and IBKR (live). Under same-vendor
Path D that rationale is gone, so the mask must stop *governing* the Path-D model
— and the model may use the microstructure the mask previously hid.

## The contract

**1. Data sources (model plane).**
- **Databento OPRA** (`OPRA.PILLAR`): SPXW options — historical (train) and live
  (decide). Same vendor both sides.
- **ThetaData**: official **SPX cash index** — historical (train) and live
  (decide). Same vendor both sides. (Databento does not sell the SPX index.)
- **IBKR is NOT a model input.** It is execution/account/safety only; its quotes
  are read solely at order submission (Phase-2 execution policy), never fed to the
  model.

**2. Same-vendor identity requirement.** Each model feed must be the *same vendor*
in training and live. Cross-vendor feeds into the model are prohibited. (This
replaces the old Databento↔IBKR feature-parity gate; the remaining cross-vendor
concern — decide-Databento/fill-IBKR *execution* reconciliation — is a Phase-2
execution-plane item, not a model-plane gate.)

**3. Feature admission (mask lifted, discipline retained).**
- The microstructure previously masked — bid/ask/mid, spread, sizes, and
  sub-minute tick dynamics **from Databento** — is **admissible** for the model,
  because it is now consistent train↔live. Admissible ≠ mandatory; feature
  selection happens in modeling.
- **Self-computed IV/delta/gamma retained; raw vendor greeks remain prohibited**
  (computed identically from price+spot+strike+time — stability + consistency).
- **Causal + anti-leakage discipline unchanged:** every feature must be available
  at inference for its cadence; mandatory mutate-future audit before training; no
  future/path/exit label may enter a runtime feature.

**4. Decision clock.** A single documented clock convention (received-timestamp)
applied identically in training and live. Entry stays **completed-minute**; the
**exit is 1-second** (per FT2-60). Sub-minute exit substrate = `cbbo-1s` (Tier-S,
A7) for Phase-1 backtest; `cmbp-1` raw-event-path (Tier-T) is a Phase-2 promotion
requirement.

## Scope — what does NOT change

The minute entry policy and the 17 synchronized entry features; SPXW identity /
42-slot ladder / causal firewalls; D48/D49 / one-account serial accounting /
affordability / one-contract / 15:55 forced-flat; the FT2-04 entry label families;
simulator v5 core; A7 Tier-S/Tier-T provenance. This contract does not authorize
live decisions, runtime/default changes, the execution policy, the A1 rewrite, the
graph reissue, or paper trading — those are the Phase-2 backlog.

## Anti-regression checker rule (new)

Add `path_d_model_plane_source_purity`:
- every Path-D model-facing feature derives only from **Databento (options)** or
  **ThetaData (SPX)** — **no IBKR field in the model plane**;
- each model feed is the **same vendor** in the training and live manifests;
- raw vendor greeks remain absent from model inputs (self-computed only);
- IBKR references are permitted only in execution/account/safety code paths.

## Implementation + verification path (for Codex)

1. Critically review this design against the authority, A7, the FT2 chain, and the
   blast-radius audit. Is superseding the mask's governing role (for Path D only)
   sound? Any second-order conflict (e.g., a contract that still hard-requires the
   mask on the Path-D path)? Is the checker rule enforceable? If a material defect:
   STOP and report (do not implement a flawed supersession).
2. If sound: implement as a governed signed contract — record the supersession of
   the masked-v2 cert + scoped-sync decision *for the Path-D path* (old artifacts
   preserved immutable), add `path_d_model_plane_source_purity` to the consistency
   checker, and re-pin any Phase-1 specs that cite it. Keep the graph unchanged in
   Phase 1.
3. Claude verifies (reproduce hashes; confirm the checker rule real+passing;
   confirm old artifacts preserved, not rewritten; confirm scope excludes live).
4. Owner signs.

No training, download, broker/recorder contact, live decision, runtime change, or
paper order is authorized by this contract.
