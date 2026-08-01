# One-Second Exit Objective + Architecture — DESIGN v2 (Path D, FT2-60)

**STATUS: DESIGN PROPOSAL v2 (Claude writes → Codex re-reviews → owner signs).**
Supersedes v1 (`..._ONE_SECOND_EXIT_OBJECTIVE_ARCHITECTURE_DESIGN_2026_08_01.md`), which
Codex returned NEEDS-CHANGES (chiefly: `Q(hold)` was undefined/oracle). Scope: **Phase 1
Tier-S feasibility** only. Constituent of the Path-D Phase-1 authority overlay.

## 1. Role and scope

The learned **1-second exit** is the sole position manager for the single open position
(Path D same-vendor → full admissible microstructure per the model-plane contract v2). The
protective floor is DEMOTED to a **deterministic catastrophic backstop** (see §6), not a
second decision policy. Loss control lives here (L2).

## 2. The objective — deployable `Q(hold)`/`Q(exit)` (fixes the v1 oracle defect)

v1 left `Q(hold)` as the foundation's **hindsight upper bound** (`q_hold` = "best later
executable bid" — explicitly "not a deployed policy"). v2 replaces it with **deployable
values under one frozen law**, using a **finite-horizon continuation policy** for Phase-1
(full-session serial-DP/Bellman deferred to Phase-2):

```
Q_exit(s_t)  = value of exiting now  = executable-bid PnL under the frozen Phase-1
               provisional fill law (§ fill-law contract)  +  causal flat-slot value
               of the released slot AFTER the actual simulated occupancy-release time.
Q_hold(s_t)  = value of holding      = one-second hold transition  +  value under the
               SAME frozen finite-horizon continuation policy + the SAME flat-slot law.
A_hold(s_t)  = Q_hold(s_t) - Q_exit(s_t)
```

- **Continuation policy (finite horizon H):** the value of `Q_hold` is computed by rolling
  the same learned exit policy forward up to H seconds (or forced-flat), fills valued by
  the provisional fill law — a self-consistent, deployable target, not hindsight.
- **Every economic term appears exactly once.** Spread, fees, latency, and opportunity
  cost each enter one place (fills carry spread/fees via the fill law; opportunity via the
  flat-slot term; latency via the fill law) — no double-count.

## 3. The four terms, operationalized (fixes v1 hand-waving)

- **(a) Slot opportunity cost — in dollars.** The flat-slot value = the causal expected
  dollar value of what a flat account would do with the slot from occupancy-release
  onward, denominated in the same PnL units as `Q`. Cadence-honest: the entry model runs
  once per completed minute, so the flat-slot value uses the entry model's **minute**
  forecast available at the release minute — it is NOT silently converted into a 1-second
  entry forecast.
- **(b) Switching/re-entry cost — no double-count.** If executable bid/ask already embed
  spread and fees (they do, via the fill law), the switching term adds ONLY the residual
  round-trip cost not already in the fills: the re-entry window, the no-fill transition
  probability, the replacement-lifecycle value, and the occupancy-release time. It does
  not re-charge spread/fees already counted in `Q_exit`.
- **(c) Fill/latency/quote-age uncertainty.** Sourced entirely from the frozen provisional
  fill law (latency-sweep-calibrated: median $0, tail-by-delay). Stale quotes (age > bound)
  gate the decision. No separate ad-hoc slippage.
- **(d) Distributional targets — exact contract.** Heads predict a defined set of forward
  outcome quantiles (downside, giveback-from-peak, recovery, remaining tail) with: exact
  target definitions, the finite horizon H, censoring rule at forced-flat/no-bid, the loss
  function (pinball/quantile), and a **frozen consumption rule** stating exactly how the
  distribution feeds the HOLD/EXIT decision. "Predict the distribution" alone is not a
  trainable contract; this is.

## 4. Convexity principle (Pickles)

Asymmetric: cut losers fast, let winners run. The objective rewards harvesting large
favorable excursions near their peak, tolerating small give-back on runners, and cutting
developing losses early. A pure trailing-stop (sell only after price falls to a line) is
the explicit anti-pattern.

## 5. Cadence, causality, features, training data

- 1-second decisions on the **Tier-S `cbbo-1s`** substrate (A7). (`cmbp-1` raw-event-path
  is Phase-2/Tier-T.)
- Causal 1-second features only, per the model-plane contract v2 allowlist + lineage
  manifest; self-computed greeks; mandatory mutate-future audit.
- **OOF construction (fold-specific):** each session's trajectories come from an entry
  model + calibrator + composer + threshold trained on folds that **never saw that
  session** — NOT the final full-fit frozen model. Broad eligible-position pretraining,
  then specialization to fold-specific frozen-entry trajectories. Session-clustered
  evidence throughout.

## 6. Floor = deterministic catastrophic backstop (fixes v1 "not clean")

The floor is a **deterministic, independently enforceable** hard stop with an explicit
causal trigger (e.g., executable bid ≤ a fixed catastrophic level derived once at entry;
checked every second on the live/replay bid). It is NOT forecast-derived policy and is not
a second learned decision. It fires independently of the learned exit; the two are never
blended in headline counts. (This matches the owner's earlier decision: floor = backstop,
learned exit = primary.)

## 7. Calibration + acceptance — aligned to the authority (fixes v1)

- **Calibration:** a defined conformal EXIT-reliability procedure wired into the governed
  **action-conditioned calibration gate** required by the authority (L449–456; part of the
  FT2-10 contract) — WAIT/EXIT-rate reliability, selected-contract regret vs forecast,
  exit-decision reliability, protective-floor behavior, low-sample fallback. Preregistered,
  smoke-tested, independently audited, owner-signed.
- **Acceptance is ECONOMIC (authority §5.4/D20, L524–531), not PR/AP.** The learned exit
  must beat the best honest comparator (exit-immediately; hold-to-flat; a finite
  preregistered time/stop/target set; legacy P5 lifecycle; matched-rate random) **pooled
  AND in ≥4 of 5 chronological outer folds**, under one-account serial replay, identical
  fees/stress, without violating survival controls. **PR/AP is a rare-event DIAGNOSTIC
  only** (leads nothing). Floor-on/off ablation per the FT2-74 harvest tripwire (L520–522):
  if large-win capture collapses under floor-on, route `owner_decision_required`.

## 8. Architecture (Q3)

Cheapest-first: (1) transparent **GBT baseline** on engineered 1-second path features to
validate the objective; (2) **miniature-neural temporal challenger** (sequence over the
1-second position path + market context) as the challenger, same heads. Both share the
calibration + economic acceptance above.

## 9. Tier-S claim boundary (fixes v1 overclaim)

Highest allowed conclusion: **"Tier-S feasibility evidence sufficient to decide whether to
fund Tier-T / Phase-2."** FORBIDDEN conclusions: trusted loss control, deployable/proven
edge, paper-readiness, promotion (A7, authority L1065). `cbbo-1s` floor/stop labels are
`1-second-approximate`.

## 10. Governance / no-reward-hacking

Anchors: a metric win without an economic serial-replay win is rejected; massive
improvements get immediate skepticism. Preregister objective, feature allowlist, folds,
horizon H, thresholds, calibration, and the economic primary metric + minimum
session/trajectory counts BEFORE results. This becomes the signed FT2-60 contract under the
Path-D Phase-1 overlay.

## 11. Open items for Codex re-review

1. Horizon H and the continuation-policy roll-forward cost/benefit vs a light serial-DP
   approximation — confirm finite-horizon is adequate for Tier-S feasibility.
2. The exact flat-slot value estimator (minute entry forecast → dollar slot value) and its
   leakage guards.
3. The distributional heads' exact quantile set + consumption rule.
4. Minimum session/trajectory counts (power) before fitting.
