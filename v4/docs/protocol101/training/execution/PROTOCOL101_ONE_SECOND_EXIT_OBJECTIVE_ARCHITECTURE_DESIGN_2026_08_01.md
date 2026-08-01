# One-Second Exit Objective + Architecture — DESIGN (Path D, FT2-60)

**STATUS: DESIGN PROPOSAL (Claude writes → Codex reviews → owner signs).** Not yet
an authority amendment. Specifies the learned 1-second exit for the Path D trader:
its objective (unblocking the action-advantage foundation), its causal features,
its architecture, and its evaluation. Prepared 2026-08-01.

Depends on Path D (train+decide on Databento, execute on IBKR/Schwab). Pairs with
the action-advantage foundation (currently training-blocked) and the execution
policy / A1 order-state-machine work.

---

## 1. Role and scope

The exit is the **single, fast, primary manager of the one open position**. Under
Path D (same-vendor Databento train↔live) it decides at **1-second cadence** and may
use the **full sub-minute microstructure** (the cross-vendor mask is no longer
needed). It replaces the awkward "minute-smart-exit + 1s-stop" two-layer compromise:

- **Learned 1-second exit = the only decision-maker** (HOLD vs EXIT, tick-by-tick).
- **The protective floor is DEMOTED to a catastrophic backstop** inside the
  deterministic broker-facing risk governor — it fires only if the learned exit
  fails to act and price breaches a hard, upward-only line. It is *not* a second
  decision policy and its triggers are counted separately (never blended into the
  learned-exit metrics).

Loss control belongs here (L2): the exit is the only component that sees the live
position tick-by-tick and can cut a developing loss or harvest a peak.

## 2. The objective — action-advantage, unblocked

Base signal, per the foundation:

> `A_hold(t) = Q(hold, t) − Q(exit_now, t)`, where `Q(exit_now, t) = executable
> bid(t) − fees` (no midpoint fantasy).

The foundation blocked training pending four terms. This design specifies them:

**(a) Slot opportunity cost.** Holding occupies the one position slot. Subtract the
expected value of the best alternative the entry model would take if flat:
`Q(hold,t) -= E[ value of re-entering the entry model's current top candidate ]`,
using the FROZEN entry model's forecasts at t (causal — no hindsight). Conservative
proxy if the full counterfactual is unstable: scale the penalty by the entry model's
live WAIT→BUY conviction (a strong new setup makes holding costlier).

**(b) Switching / re-entry cost.** Exiting then re-entering costs ~2× spread + 2×
fees + latency. Penalize churn: an EXIT that is followed by a re-entry within a
frozen window must clear that round-trip cost. Prevents thrash.

**(c) Fill / latency / quote-age uncertainty.** `Q(exit_now)` uses the **executable
bid** with the execution bridge's calibrated slippage (from the latency sweep:
median $0, tail grows with delay), NOT a mid. Stale quotes (age > frozen bound) gate
the decision to an abstain/hold-with-flag. The exit is valued at what it can
actually fill at.

**(d) Distributional targets, not just the mean.** Predict the *distribution* of
forward outcomes, not only `E[A_hold]`: downside tail, giveback-from-peak, recovery
probability, remaining favorable tail. This lets the exit be risk-aware — e.g., cut
when the downside tail widens even if mean `A_hold` is marginally positive.

## 3. Convexity principle (Pickles) — the shape the objective must reward

Asymmetric by design: **cut losers fast, let winners run.** The exit must
*capture the convex upside* (not sell late like a pure trailing stop) while bounding
downside. Concretely the objective rewards: harvesting large favorable excursions
near their peak; tolerating small give-back on runners; and cutting developing
losses early. A pure-stop policy (sell only after price falls to a line) is the
explicit anti-pattern — it structurally gives back every peak.

## 4. Cadence, causality, and features

- **Cadence:** 1-second decisions on the `cmbp-1`→1s substrate (D59; A7 Tier-T for
  trusted floor/stop labels — those labels inspect the raw event path before
  downsampling).
- **Causal features only** (available at 1-second inference): position PnL path,
  MFE/MAE, giveback-from-peak, quote velocities, spread, **self-computed** IV/delta/
  gamma (raw vendor greeks prohibited), time-in-trade, minutes-to-close, market
  phase, SPX path (ThetaData). Under Path D, Databento sizes/spread/tick dynamics
  are now admissible (same-vendor) — a richer input set than the masked entry contract.
- **Mandatory mutate-future audit** before any training: prove no future/path/exit
  label leaks into a runtime feature.

## 5. Training data discipline

- Train on **out-of-fold trajectories produced by the FROZEN entry model** — never
  train the exit on in-sample entry selections and then evaluate the combination as
  "unseen."
- Session-clustered evidence throughout (sessions are indivisible clusters; lead
  every table with session counts, not row counts — L4/RSD-04).
- Broad eligible-position pretraining, then specialization to frozen-entry
  trajectories (per the foundation's intended two-stage plan).

## 6. Architecture (Q3)

Two rungs, cheapest-first (owner-accepted default):

1. **Transparent baseline (validate the OBJECTIVE cheaply):** gradient-boosted model
   on engineered 1-second path features, with heads for HOLD/EXIT probability +
   continuation-value + regret + an uncertainty estimate. Purpose: prove the
   action-advantage objective produces sane, economically useful exits before
   spending on architecture.
2. **Miniature-neural challenger (the V2 model):** a temporal/sequence model over the
   position's 1-second path + market context (the "miniature but real" neural rung).
   Shared representation → the same heads. Trained identically on the OOF trajectories.

Both rungs share: conformal calibration (session-clustered) of the regret bound and
EXIT reliability; **lead evaluation with prevalence-aware PR/AP + calibration**, not
ROC AUC (L11: AUC 0.837 with AP 0.066 is not a usable exit).

## 7. Evaluation (preregistered)

- **Economic decision curves:** does it cut losses WITHOUT destroying big-win capture
  (the convexity test)? Harvest ratio, MFE-giveback, MAE, underwater duration, churn,
  time-in-trade.
- **Baselines to beat:** exit-immediately, hold-to-flat, a preregistered
  time/stop/target rule, matched-rate random exits, and (where applicable) the legacy
  lifecycle. The learned exit must beat these on risk-adjusted economics, not a proxy.
- **Floor-on/off ablation:** quantify what the catastrophic backstop adds under
  identical learned exits (report-only unless a harvest threshold is preregistered).
- **Four-bucket Pickles distribution** + equity curve + trades-on-SPX (D60), with
  natural vs backstop-triggered exits counted separately.
- Session-clustered CIs on everything; wide CIs expected early = honest.

## 8. Interaction with entry, floor, risk governor

- Entry (minute, frozen) selects the contract; the exit takes over the moment the
  position is open.
- The **floor/stop** is deterministic, upward-only, forecast-derived, and lives in
  the **broker-facing risk governor** as a catastrophic backstop; the learned exit is
  primary. No double-counting; separate accounting.
- The risk governor also owns: Databento feed-loss while holding → forced-flat via
  the broker; stale-data; daily loss breaker; 15:55 forced-flat.

## 9. Governance / no-reward-hacking

- **Anchors:** the exit must genuinely improve risk-adjusted economics vs the
  baseline panel — a metric win without an economic win is rejected. Massive
  improvements get immediate skepticism.
- Preregister the objective, feature set, folds, thresholds, and evaluation BEFORE
  results; no post-hoc tuning. Frozen metrics.
- This becomes the FT2-60 lifecycle contract once reviewed + owner-signed.

## 10. Open decisions for Codex review

1. Slot-opportunity-cost operationalization: full counterfactual re-entry value vs
   the conservative WAIT→BUY-conviction proxy — which is estimable without leakage?
2. Distributional target parameterization (quantile heads vs a small mixture) and how
   the exit consumes it (a risk-aware decision rule).
3. The exact 1-second decision → order-window handoff to the execution policy (ties
   to the A1 state machine) — where does the exit decision end and execution begin?
4. Confirm the transparent-baseline-first / neural-second sequencing and the minimum
   session/trajectory counts (power) before fitting.
