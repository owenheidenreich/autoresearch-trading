## **PROMPT**:

```
Goal: PHASE-1 PATH-D GOVERNANCE REVIEW (design review only — NO implementation, NO
edits, NO reseal, NO download/broker). Read-only; verify against source; return a
written verdict. Roles: Claude wrote both docs; Codex reviews; owner signs after.

Review these two Phase-1 proposals together and against the authority (self-hash
1d215845), A7, the FT2 chain, and the blast-radius audit:
1. Path-D model-plane contract proposal:
   v4/docs/protocol101/training/execution/PROTOCOL101_PATH_D_MODEL_PLANE_CONTRACT_PROPOSAL_2026_08_01.md
2. One-second exit objective + architecture (FT2-60) design:
   v4/docs/protocol101/training/execution/PROTOCOL101_ONE_SECOND_EXIT_OBJECTIVE_ARCHITECTURE_DESIGN_2026_08_01.md

Assess and rank by severity:
 A. Is superseding the microstructure-mask cert + scoped-synchronization decision
    *for the Path-D path only* (old artifacts immutable) sound and complete? Any
    contract/spec that still HARD-REQUIRES the mask on the Path-D path and would
    conflict? Is "same-vendor model plane, IBKR execution-only" fully coherent?
 B. Does lifting the mask create any leakage/causality risk the design fails to
    guard? Is the `path_d_model_plane_source_purity` checker rule enforceable as
    written?
 C. Is the FT2-60 exit objective sound — action-advantage with slot-opportunity/
    switching/fill-latency/distributional terms, convexity, causal 1s features,
    OOF-from-frozen-entry training, GBT-baseline-then-neural, PR/AP-led eval? Any
    missing term or leakage path? Is the floor-as-catastrophic-backstop split clean?
 D. Consistency BETWEEN the two docs (feature admission, decision clock, Tier-S
    cbbo-1s for Phase-1 backtest, the provisional backtest fill model) and with A7.
 E. Is the Phase-1 scope boundary correct — i.e., these two changes are sufficient
    to BACKTEST Path D, and they correctly DEFER the execution policy / A1 rewrite /
    graph reissue / runtime to Phase 2?

Return: a verdict (sound-to-sign / needs-changes), a severity-ranked findings list,
any second-order conflicts, and the single most important thing either doc gets
wrong. If a BLOCKING defect exists, STOP and report (roles may flip). Do NOT
implement or edit anything.
```

## **ANSWER**
```
## Verdict: NEEDS CHANGES — STOP before signature

The Path‑D direction is sound, but these two documents are not jointly signable and do not yet authorize a defensible Phase‑1 backtest.

The review reproduced the governing authority hash `1d215845…` and unchanged graph hash `9955085a…`. No files, authority, runtime, data, or broker state were changed.

### Single most important defect

The exit design declares action advantage “unblocked,” but never defines `Q(hold)`.

It defines `Q(exit_now)` precisely, while leaving the other half of the equation unspecified. The existing foundation defines `Q(hold)` as the best later executable bid—a hindsight upper bound, not the value of a deployable continuation policy. [Exit design](/Users/gduby/Documents/autoresearch-trading/v4/docs/protocol101/training/execution/PROTOCOL101_ONE_SECOND_EXIT_OBJECTIVE_ARCHITECTURE_DESIGN_2026_08_01.md:31), [foundation evidence](/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_hold_exit_action_advantage_foundation_v1/summary.json:80)

Adding opportunity cost, switching cost, and uncertainty to an undefined—or oracle—`Q(hold)` does not produce a valid learned policy. This remains training-blocked.

## Severity-ranked findings

| Severity | Area | Finding |
|---|---|---|
| **Blocking** | C | `Q(hold)` has no continuation policy, horizon, Bellman target, or serial-DP definition. HOLD labels therefore remain ambiguous/oracular. |
| **Blocking** | D/E | Phase‑1’s decision-to-fill law is absent. The design values EXIT at the observed bid plus unspecified slippage, while explicitly leaving the decision→order handoff open. That handoff affects labels, occupancy, no-fills, PnL, and opportunity cost. [Exit design](/Users/gduby/Documents/autoresearch-trading/v4/docs/protocol101/training/execution/PROTOCOL101_ONE_SECOND_EXIT_OBJECTIVE_ARCHITECTURE_DESIGN_2026_08_01.md:51) |
| **Blocking** | D | The documents contradict one another on data tier: the model-plane contract says Phase‑1 uses Tier‑S `cbbo-1s`; the exit design says `cmbp-1`→1s Tier‑T. [Model-plane contract](/Users/gduby/Documents/autoresearch-trading/v4/docs/protocol101/training/execution/PROTOCOL101_PATH_D_MODEL_PLANE_CONTRACT_PROPOSAL_2026_08_01.md:51), [exit design](/Users/gduby/Documents/autoresearch-trading/v4/docs/protocol101/training/execution/PROTOCOL101_ONE_SECOND_EXIT_OBJECTIVE_ARCHITECTURE_DESIGN_2026_08_01.md:71) |
| **Blocking** | A/B | “Every model-facing feature derives only from Databento or ThetaData” conflicts with the exit’s required position-PnL path. In production, entry fill price/time/quantity and position state come from the broker. Ignoring actual fills breaks train/live state parity. [Source-purity rule](/Users/gduby/Documents/autoresearch-trading/v4/docs/protocol101/training/execution/PROTOCOL101_PATH_D_MODEL_PLANE_CONTRACT_PROPOSAL_2026_08_01.md:66), [exit features](/Users/gduby/Documents/autoresearch-trading/v4/docs/protocol101/training/execution/PROTOCOL101_ONE_SECOND_EXIT_OBJECTIVE_ARCHITECTURE_DESIGN_2026_08_01.md:71) |
| **Blocking** | A/E | The governance path is unresolved. Current authority hard-requires completed-minute exits, the D24 floor, and the signed feature restrictions. Existing validators and A7 registries pin the current authority hash. A canonical reseal cannot happen while the associated re-pins are deferred to Phase‑2. [Authority timing](/Users/gduby/Documents/autoresearch-trading/v4/docs/protocol101/training/contracts/PROTOCOL101_FULL_TRADER_GRAPH_V2_CONSOLIDATED_AUTHORITY_2026_07_28.md:194), [feature restriction](/Users/gduby/Documents/autoresearch-trading/v4/docs/protocol101/training/contracts/PROTOCOL101_FULL_TRADER_GRAPH_V2_CONSOLIDATED_AUTHORITY_2026_07_28.md:271), [checker pin](/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_scoped_final_round_5_fixes_attempt001/check_cross_contract_consistency_v3.py:47) |
| **Blocking** | E | These two documents are not sufficient by the sequencing plan’s own definition. Phase‑1 also requires the additive FT2‑08 1-second tensor/label contract and a frozen provisional fill contract. [Sequencing plan](/Users/gduby/Documents/autoresearch-trading/v4/docs/protocol101/training/execution/PROTOCOL101_PATH_D_AMENDMENT_SEQUENCING_PLAN_2026_08_01.md:24) |
| **High** | B | “Received timestamp” is not a complete clock. The repository already distinguishes event/source/receipt clocks. Phase‑1 must define interval closure, historical `ts_recv`, live local receipt, cross-feed watermarking, allowed latency, and when an action may consume the completed second. [Existing clock provenance](/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/field_semantics_manifest.json:230) |
| **High** | B | “Mask lifted; feature selection happens in modeling” is too broad. It bypasses a frozen allowlist and multiplicity-aware feature admission. Volume, OI, theta, breakeven, aggregated statistics, and derived proxies are left ambiguous. |
| **High** | C | Slot opportunity cost is not operational. WAIT→BUY conviction is not denominated in dollars and is not comparable to `Q(hold)`. The entry model also runs once per completed minute, so “current at t” cannot silently become a new 1-second entry forecast. |
| **High** | C | The switching-cost expression can double-count spread and fees if executable bid/ask prices already contain them. It also lacks the re-entry window, no-fill transition, replacement lifecycle value, and occupancy-release time. |
| **High** | C | “Frozen entry OOF trajectories” is underspecified. Each trajectory must come from a fold-specific entry model, calibrator, composer, and threshold that never saw that session—not from the final full-fit frozen model. |
| **High** | C | “Conformal calibration of EXIT reliability” is not a defined probability-calibration procedure. If uncertainty controls HOLD/EXIT, the authority requires a separate action-conditioned calibration gate. [Authority requirement](/Users/gduby/Documents/autoresearch-trading/v4/docs/protocol101/training/contracts/PROTOCOL101_FULL_TRADER_GRAPH_V2_CONSOLIDATED_AUTHORITY_2026_07_28.md:449) |
| **High** | C | PR/AP is appropriate for rare EXIT classification, but cannot lead overall candidate acceptance. Governed acceptance remains economic: identical entries, serial replay, best comparator, and at least four of five outer folds. [Lifecycle gate](/Users/gduby/Documents/autoresearch-trading/v4/docs/protocol101/training/contracts/PROTOCOL101_FULL_TRADER_GRAPH_V2_CONSOLIDATED_AUTHORITY_2026_07_28.md:524) |
| **High** | C | The floor split is not clean. A “forecast-derived” floor is still a decision policy, while a catastrophic backstop should be deterministic and independently enforceable. “Fires only if the learned exit fails” has no causal trigger definition. |

## A–E disposition

### A. Mask supersession

Sound in principle, incomplete in scope.

The safer split is:

- Entry remains on exactly the existing 17 features.
- The scoped synchronization decision continues governing entry feature admission, even though its cross-vendor-transfer role is retired for Path D.
- A new Path‑D lifecycle namespace receives an explicit unmasked exit-feature allowlist.
- Legacy mask code and artifacts remain immutable.

Current active code recognizes only the legacy v1/v2 contracts and applies the mask for v2. A separate Path‑D contract and builder are required; legacy code should not be mutated into meaning two different things. [Current implementation](/Users/gduby/Documents/autoresearch-trading/v4/live/protocol101_feature_contract.py:20)

“Same-vendor” should apply to market features, not every state input:

```text
Market alpha:
  Databento OPRA + ThetaData SPX only

Permitted non-alpha state:
  broker-confirmed fill price/time/quantity/identity
  current position and account/risk state
  deterministic clock, fees, masks

Forbidden model input:
  IBKR quote-derived market alpha
  broker outcome labels
  vendor Greeks
```

### B. Source-purity and causality

The proposed checker is not enforceable as written. Static detection of whether an arbitrary derived feature “comes from” IBKR cannot reliably be inferred from code paths.

It needs a fail-closed feature-lineage manifest recording:

- Component namespace: entry or exit.
- Feature role: alpha, position state, mask, label, or audit.
- Transitive source leaves.
- Vendor, product, schema, historical/live mode.
- Event, source-receipt, and local-receipt semantics.
- Transformation and builder hashes.
- Exact historical/live twin.
- Explicit rejection of unknown lineage and renamed proxies.

Negative fixtures should include an IBKR bid renamed as PnL, a vendor Greek renamed as internal delta, a future `ts_recv`, a missing live twin, and an unregistered feature.

### C. FT2‑60 exit objective

The architectural sequence is good:

- HGB baseline first.
- Temporal neural challenger second.
- Frozen entry before lifecycle training.
- Broad pretraining followed by OOF specialization.
- Session-clustered evidence.
- Convexity and floor ablation.
- PR/AP instead of ROC AUC for rare-action diagnostics.

But the target must first become executable. It should define either:

1. A finite-horizon continuation policy; or
2. A full-session, one-account Bellman/serial-DP value law.

For example:

```text
Q_exit(s_t)
  = value under the frozen Phase-1 exit-order simulator
  + causal flat-slot value after actual simulated occupancy release

Q_hold(s_t)
  = one-second hold transition
  + value under the same frozen continuation/flat-slot law

A_hold(s_t)
  = Q_hold(s_t) - Q_exit(s_t)
```

Every spread, fee, latency, and opportunity term must appear exactly once.

The distributional heads then need exact targets, horizons, censoring, losses, and a frozen consumption rule. “Predict downside/recovery/tail” alone is not a trainable contract.

### D. Between-document consistency

Phase‑1 must choose one coherent statement:

- Corpus: Tier‑S `cbbo-1s`.
- Purpose: skeleton/prototype feasibility.
- Floor/stop fidelity: `1-second-approximate`.
- Allowed conclusion: whether signal is promising enough to fund Tier‑T and Phase‑2.
- Forbidden conclusion: trusted loss control, proven deployable edge, paper readiness, or promotion.

A7 explicitly allows Tier‑S for feasibility and explicitly bars it from promotion/paper loss-control gates. [A7 authority](/Users/gduby/Documents/autoresearch-trading/v4/docs/protocol101/training/contracts/PROTOCOL101_FULL_TRADER_GRAPH_V2_CONSOLIDATED_AUTHORITY_2026_07_28.md:1065)

### E. Phase‑1 boundary

The full production execution policy, prospective broker calibration, live runtime, default change, and Phase‑2 graph/runtime nodes can remain deferred.

The following cannot be deferred:

- Exact 1-second tensor and label semantics.
- Exact causal clock.
- Exact Phase‑1 provisional order/fill/no-fill law.
- Serial occupancy-release semantics.
- Formal `Q(hold)` and `Q(exit)` definitions.
- OOF entry-trajectory construction.
- Tier‑S claim boundary.
- A resolved Path‑D Phase‑1 authority identity.

The cleanest governance solution is a new immutable Path‑D Phase‑1 consolidated overlay/manifest that pins:

- Parent authority `1d215845…`.
- Model-plane contract hash.
- FT2‑60 contract hash.
- Additive FT2‑08 hash.
- Provisional fill-law hash.
- A7 policy hash.
- Unchanged graph topology hash `9955085a…`.

Only Path‑D Phase‑1 artifacts should pin that resolved authority. The old authority, legacy receipts, mask certificate, and legacy checker remain unchanged.

## Minimum repair package for Claude

1. Replace “prove edge” with “produce Tier‑S feasibility evidence sufficient to decide whether to fund Tier‑T.”
2. Make both documents use `cbbo-1s` for Phase‑1 and reserve `cmbp-1` for trusted Phase‑2 evidence.
3. Define `Q(hold)`, including continuation and flat-slot value.
4. Add the exact provisional Phase‑1 decision/fill/no-fill/latency policy.
5. Split market-source purity from broker-derived position state.
6. Replace open-ended feature admissibility with separate frozen entry and exit allowlists.
7. Freeze OOF construction, primary economic metric, outer-fold law, multiplicity, calibration, event-level PR/AP, and minimum session/trajectory evidence.
8. Choose whether the floor is deterministic catastrophic safety or forecast-derived policy; it cannot be both.
9. Create a non-vacuous Path‑D authority/checker namespace without rewriting legacy evidence.

The correct disposition is therefore:

```text
PATH-D_MODEL-PLANE: concept accepted, contract repair required
FT2-60: design repair required; training remains blocked
PHASE-1 BACKTEST: not authorized
ROLE ROUTE: return to Claude for revision, then Codex re-review
```

No implementation, edits, reseal, tests, downloads, or broker actions were performed.

Goal usage: 425,675 tokens over approximately 8 minutes 24 seconds.
```