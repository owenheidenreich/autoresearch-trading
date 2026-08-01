# Protocol101 Path-D Entry + Exit Model Plan

- Date: 2026-08-01
- Status: research-grade design ready for Claude/owner review; not authority
- Scope: Tier-S offline feasibility planning only
- Corpus: `/Users/gduby/.autoresearch-trading/pathd_2025-08-01_2026-07-31`
- Highest allowed claim: **a research-grade entry+exit model plan ready for review; no model built, no edge claimed**

## 0. Decision and scope

This document is the build specification for a later, separately authorized Path-D
entry/exit research run. It reuses the acquired corpus, the validated Path-D exit-label
smoke, the signed 17-feature entry boundary, the A6/FT2-10 convexity amendment, and the
implemented `v4/path_d/` execution seam. It does not train or fit anything.

The permitted sequence is:

```text
minute entry research
  -> freeze one research entry artifact per fold
  -> generate exit trajectories only from the corresponding unseen-session entry artifact
  -> 1-second exit research
  -> four-box one-account attribution through the Path-D boundary
  -> Tier-S feasibility decision
```

The following are explicitly out of scope:

- model fitting, threshold tuning, or holdout scoring in this goal;
- broker, IBKR, live-feed, paper, or real-money activity;
- edits to Protocol158, Protocol160, the paper registry, runtime flags, launchd, or
  `v4/.env`;
- resealing the parent authority or freezing any Path-D governance proposal;
- promotion, paper-readiness, trusted loss-control, or deployable-edge claims.

The five earlier Path-D governance documents are design inputs only. Their known
round-two defects are resolved here at research-spec level, but the documents themselves
remain shelved and unsigned until edge justifies formalization.

## 1. Reused evidence and implementation boundaries

The later build must import or wrap these existing seams rather than invent parallel
ones:

| Purpose | Existing source |
|---|---|
| Corpus provenance and counts | `v4/audit/autoresearch/protocol101_pathd_data_acquisition/PATHD_DATA_ACQUISITION_HANDOFF_2026_08_01.md` |
| Non-circular 1-second label | `v4/scripts/run_pathd_exit_label_smoke.py` |
| HGB OOF/replay smoke pattern | `v4/scripts/run_pathd_exit_backtest_smoke.py` |
| Semantic execution command | `v4/path_d/contracts/execution_intent.py::ExecutionIntentV1` |
| Executor dependency boundary | `v4/path_d/contracts/executor_port.py::ExecutorPort` |
| Offline order lifecycle | `v4/path_d/execution/simulated.py::SimulatedExecutor` |
| Latency evidence | `v4/path_d/execution/latency.py` |
| Sole pre-submit authorization seam | `v4/path_d/risk/governor.py::DeterministicGovernor` |
| Signed entry alpha names | `v4/docs/protocol101/synchronization/contracts/PROTOCOL101_SCOPED_SYNCHRONIZATION_DECISION_2026_07_25.md` |
| Entry science and controls | `v4/audit/autoresearch/protocol101_ft2_10_entry_science_contract/` |
| A6 convexity correction | `v4/docs/protocol101/training/execution/PROTOCOL101_FT2_10_ENTRY_GATE_CONVEXITY_AMENDMENT_PROPOSAL_2026_07_30.md` and the owner-signed state recorded in the learnings ledger |

`ExecutionIntentV1` is the only entry/exit order-intent shape in the research replay.
Every simulated BUY or SELL must be authorized by the deterministic governor and routed
through a `SimulatedExecutor` implementing `ExecutorPort`. WAIT and HOLD create no
intent. This binds research accounting to the implemented execution seam without
claiming live-fill parity.

## 2. Data contract and firewall

### 2.1 Exact substrates

All paths below are under the immutable corpus root
`/Users/gduby/.autoresearch-trading/pathd_2025-08-01_2026-07-31`.

| Role | Exact path | Current evidence |
|---|---|---|
| Minute entry rows | `aligned/processed/minute_entry` | 251 files, 90,073 rows |
| Normalized option surface | `aligned/normalized` | 251 base plus 251 official-context Parquets, 47,707,186 rows in each view |
| 1-second exit quotes | `raw/databento/opra_spxw_cbbo_1s` | 251 files, 1,565,481,074 rows |
| Option definitions | `raw/databento/opra_spxw_definition` | 251 files, 4,548,804 rows |
| Minute CBBO / OHLCV / statistics | `raw/databento/opra_spxw_cbbo_1m`, `raw/databento/opra_spxw_ohlcv_1m`, `raw/databento/opra_spxw_statistics` | 251 paired option sessions |
| Official SPX context | `vendor/thetadata/index/spx_1m` | 251/251 paired sessions |
| Official VIX context | `vendor/thetadata/index/vix_1m` | every option session paired; four extra non-option dates excluded |
| ES context | `raw/databento/glbx_es_ohlcv_1m` | 254 nonempty sessions |
| VX context | `raw/databento/xcbf_vx_ohlcv_1m` | begins 2026-04-01; 87 nonempty sessions |
| Validated exit labels | `aligned/exit_labels/pathd_exit_labels_12m.parquet` | 37,047,600 rows, 3,012 sampled trajectories |

Entry decisions consume only completed-minute information. Exit decisions consume only
a completed 1-second interval. The decision availability time is the maximum of the
option and official-context receipt watermarks; no action may consume a bar before it is
closed. Historical `ts_recv` and the Path-D `received_timestamp_utc` decision clock are
the replay clock. Future local-live twins are deferred.

Official SPX is canonical context. Official VIX is retained for data-quality and regime
reporting but is not initial signed-17 entry alpha. ES/VX are robustness strata, not
initial model inputs: this avoids silently changing the signed entry namespace and avoids
letting the 2026-04-01 VX boundary become a missingness shortcut. Any later use as alpha
requires a separately reviewed feature change.

### 2.2 Fixed chronological split

Sort the 251 paired option sessions by date and assign the following immutable indices.
No result may be used to redraw these boundaries.

| Role | Session indices | Dates | Count |
|---|---:|---|---:|
| Initial model-fit history | 1-70 | 2025-08-01 through 2025-11-07 | 70 |
| Fold 1 embargo | 71 | 2025-11-10 | 1 |
| Fold 1 outer test | 72-99 | 2025-11-11 through 2025-12-19 | 28 |
| Fold 2 embargo | 100 | 2025-12-22 | 1 |
| Fold 2 outer test | 101-128 | 2025-12-23 through 2026-02-03 | 28 |
| Fold 3 embargo | 129 | 2026-02-04 | 1 |
| Fold 3 outer test | 130-157 | 2026-02-05 through 2026-03-17 | 28 |
| Fold 4 embargo | 158 | 2026-03-18 | 1 |
| Fold 4 outer test | 159-186 | 2026-03-19 through 2026-04-28 | 28 |
| Fold 5 embargo | 187 | 2026-04-29 | 1 |
| Fold 5 outer test | 188-215 | 2026-04-30 through 2026-06-09 | 28 |
| Final firewall block | 216-251 | 2026-06-10 through 2026-07-31 | 36 |

Outer-fold training is expanding-window: fold `k` may use only earlier, non-embargo,
non-firewalled sessions. Hyperparameters, target definitions, action rules, and the
candidate family are frozen before any of the five outer-test reports are opened.
Training-only tuning uses forward-chaining inner folds and a one-session embargo at every
inner boundary. The last 20% of each fold's training sessions is calibration-only and is
not used to fit model weights.

The six already exposed exit-smoke dates are never training, calibration, outer-test, or
holdout evidence:

```text
2026-06-30, 2026-07-01, 2026-07-02,
2026-07-10, 2026-07-13, 2026-07-14
```

Removing those six dates from indices 216-251 leaves the 30-session protected holdout:

```text
2026-06-10 through 2026-06-29,
2026-07-06 through 2026-07-09,
2026-07-15 through 2026-07-31
```

The 30 dates may undergo hashes, schemas, row counts, and causal-mutation checks only
until an entry artifact, exit artifact, composer, thresholds, and complete replay packet
are frozen. No predictions, policy PnL, threshold sweeps, or candidate comparisons may
be inspected before that freeze. The existing full-corpus label validation already
published aggregate label statistics that include these dates, so this is a protected
model/economic holdout, not an unobserved raw-data holdout; that caveat must appear in
every result.

The two OPRA vendor-degraded sessions, 2025-10-22 and 2026-07-31, are excluded from
primary fit/headline scoring and included only in a named degradation sensitivity. The
five GLBX degradation notices affect ES robustness strata only because ES is not initial
alpha.

## 3. Feature source and leakage law

Every future feature record must declare one of these three roles in a research lineage
manifest. This is a research checker, not a frozen governance contract.

| Class | Allowed |
|---|---|
| Market alpha | Databento OPRA option fields and official ThetaData context available at the decision watermark |
| Permitted non-alpha state | Simulated/broker-twin entry fill, held identity, quantity, account/position/risk state, clock, fees, eligibility masks |
| Forbidden | IBKR quote alpha, broker outcomes as features, raw vendor Greeks, future timestamps/quotes/paths, labels, oracle actions, protected-result transforms, unregistered or renamed proxies |

Self-computed IV/delta/gamma may be admitted for exit only when computed from causal
price, spot, strike, right, time, and frozen constants. Raw vendor Greeks are audit-only.
Any feature without complete transitive source leaves, clock semantics, builder hash,
and a historical/future-live twin fails closed.

Mandatory negative fixtures must reject an IBKR bid renamed `pnl`, a vendor Greek renamed
`internal_delta`, a future `ts_recv`, an unregistered feature, and a feature with no live
twin. A mutate-future audit must change every post-decision quote, label, exit time, and
future-context value in turn and prove byte-identical feature tensors and predictions at
the decision time.

## 4. Entry model

### 4.1 Action space and features

The flat-state action space is WAIT plus one exact source-neutral SPXW 0DTE contract from
the complete 42-slot ladder. Incomplete/stale ladders make every BUY action false and
leave WAIT valid. Entries stop at 15:29 ET, fill no earlier than the next completed
minute, and use their own causal cash/D48/D49 ledger.

The initial entry namespace is exactly the signed 17 alpha fields, in order:

1. `spx_vwap_gap_points`
2. `spx_vwap_gap_bps`
3. `spx_vwap_gap_over_session_range`
4. `session_range_bps`
5. `momentum_5m_bps`
6. `momentum_15m_bps`
7. `momentum_5m_over_session_range`
8. `momentum_15m_over_session_range`
9. `omar_clipped_neg3_pos3`
10. `vwap_side_alignment_flag`
11. `omar_side_alignment_flag`
12. `momentum15_side_alignment_flag`
13. `D.near_atm.straddle_mid_spot_bps`
14. `D.near_atm.put_call_mid_ratio`
15. `D.near_atm.side_smile_slope_bps_per_5pt`
16. `E.bs.delta`
17. `E.bs.gamma`

Geometry, clock, action masks, quote fields used solely for eligibility/fills, and account
state used solely for D48/D49 are not alpha. The model-plane v2 input explicitly says
"entry=17 unchanged"; therefore Path-D microstructure enrichment is admitted in the exit
namespace only for this first run. An entry enrichment cannot be smuggled in under the
new vendor topology.

### 4.2 Non-circular A6 objective

Reuse the registered FT2-04/FT2-10 MFE and profit-area labels and their dollar/return
representations. The entry gate uses horizons `h10`, `h20`, `h45`, `h90`, and
`remaining_session`; `h3` and `h5` remain reported diagnostics but do not enter the gate
because the walking skeleton showed they are structurally dead for long-premium entry.

For contract `c` at completed minute `t`, fit conditional-mean forecasts and conditional
q10 forecasts for both registered upside families at each gate horizon. On the
fold-specific, disjoint calibration sessions, compute the one-sided 90% session-block
residual correction for each mean forecast. Define:

```text
LCB_mean_upside_$(t,c)
  = equal mean over gate horizons and {MFE$, profit-area$}
    of calibrated lower confidence bounds on the conditional means

LCB_mean_upside_return(t,c)
  = the same construction in fee-adjusted return units
```

The $3 round-trip fee path is primary and the $4 path is sensitivity. A contract is
entry-eligible only when both LCB composites are strictly greater than zero after fees.
This is the A6 convexity gate: a lower confidence bound on the conditional **mean**, not
the pessimistic outcome decile.

Among gate-passers only, rank by the equal-horizon calibrated q10 upside composite,
first dollars then return, with deterministic exact-contract tie-breaking. q10 may not
gate entry, define the mean LCB, or become a disguised risk-free-entry condition. If no
contract passes, choose WAIT.

### 4.3 Architectures and OOF construction

Run the cheapest transparent model first:

1. **HGB baseline.** HistGradientBoosting mean and quantile regressors on causal summaries
   of the signed-17 histories, complete-ladder geometry/masks, and no other alpha.
2. **Neural challenger.** A small temporal/set model over the 90-minute signed-17 history
   and the current 42-action set. It uses the identical targets, masks, calibrator,
   composer, and action rule; it is a challenger, not an automatic replacement.

For every outer-test session, the model weights, scaler, conditional-mean calibrator,
q10 calibrator, composer, and action threshold must all come from strictly earlier
sessions. The final full-fit research artifact is never used to manufacture OOF entry
decisions for exit training.

### 4.4 Entry controls and acceptance

Entry is evaluated with a single frozen control exit (hold-to-flat, with the same fill
law), then repeated with the best preregistered transparent control exit. It must report:

- P5-under-cap on its own causal ledger;
- nearest-ATM at registered baseline timing;
- the eight fixed matched-random full-ladder schedules from the existing generator spec;
- constant-output, sign-reversed, and strong shuffled-target controls;
- HGB versus neural attribution and feature-family ablations that never remove masks,
  D48/D49, WAIT, or the no-screen arm.

No entry component freezes unless its one-account economic delta beats P5 and the
nonselectable aggregate matched-random control pooled and in at least four of five outer
folds, while passing the action-conditioned calibration and minimum-power rules in this
plan. Selection may not use the later exit model.

## 5. Exit model

### 5.1 Frozen entry trajectories

Exit training starts only after one entry design is frozen. For outer fold `k`, generate
positions using the fold-`k` OOF entry weights, calibrators, composer, and gate; never the
full-fit entry artifact. Each successful entry is an exact contract/fill trajectory at
1-second cadence through exit or 15:55 ET. Broad eligible-position trajectories may be
used for representation pretraining, but final specialization, calibration, and economic
evidence use only trajectories created by the frozen OOF entry policy.

### 5.2 Causal exit features

The exit namespace may use an explicit, preregistered subset of:

- current/rolling Databento bid, ask, mid, spread, bid/ask sizes, size imbalance, quote
  age, quote returns/velocities, and last causally available volume/open interest;
- official SPX state and causal returns/VWAP relations;
- self-computed IV/delta/gamma and their causal changes;
- exact held identity, entry fill/time/quantity, current net PnL, MFE/MAE to now,
  giveback to now, seconds held, seconds to 15:55, occupancy and remaining risk budget.

VIX, ES, and VX remain report/stratification fields in the first run. No future-best,
future-MFE, future-exit, oracle-advantage, label-validity, or trained-policy outcome may
enter features.

### 5.3 Frozen provisional fill and fee law

The later implementation must expose one shared fill-law object/hash to labels, targets,
replay, occupancy, and PnL:

- A valid SPXW tick is `$0.05` below `$3.00` premium and `$0.10` at or above `$3.00`.
- A BUY decision uses a limit one tick through the completed reference ask; a SELL uses a
  limit one tick through the completed reference bid.
- Headline decision-to-arrival delay is one second. Report `0, 1, 2, 5` second
  sensitivity and the existing `100, 250, 500, 1000` ms paired-quote latency bounds.
- At arrival, BUY fills only if the observed ask is at or below its limit; SELL fills
  only if the observed bid is at or above its limit. Conservative fill price is the
  submitted limit, not a favorable mid or improvement.
- A no-fill leaves the account/position state unchanged and redecides on the next closed
  second. Partial/cancel/disconnect scenarios are exercised through `SimulatedExecutor`
  but cannot be silently converted to fills.
- Charge `$1.50` on each filled side, exactly once. A completed round trip therefore
  consumes the frozen `$3.00` reserve. The entry fee is sunk and cancels when comparing
  two exits; it is not subtracted a second time from exit advantage.
- Locked/crossed/stale quotes are non-actionable. No-bid is valued at zero for adverse
  labels. At 15:55 the position is flattened from the actual boundary state, never a
  prior bid; a pending earlier exit cannot double-consume the boundary quote.
- Slippage is the adverse arrival-quote/through-limit effect under the latency sweep.
  No second ad-hoc slippage term is allowed.

If the result changes sign across the delay sensitivity band, the outcome is
`owner_decision_required`, not a pass selected from the favorable rung.

### 5.4 Non-circular target and action law

The primary reference policy is **frozen hold-to-flat**: after a successful entry it does
nothing until the mandatory 15:55 exit. The second transparent reference is a fixed
50%-premium stop followed otherwise by hold-to-flat. Neither reference uses a trained
exit or an oracle.

At every actionable second `t`, run the same frozen fill law to define:

```text
E_t = realized dollar value of deterministic EXIT-UNTIL-FILLED starting at t
H_t = realized dollar value of frozen HOLD-TO-15:55
A_ref(t) = H_t - E_t
```

`A_ref` is the primary supervised target and is the full-scale form of the validated
smoke's `a_hold`. Future-best bid and `oracle_adv` are audit-only upper bounds. They may
not be targets, features, action labels, thresholds, or selection metrics.

The action composer uses the one-sided 90% session-block calibrated lower bound on the
conditional mean plus a modest frozen downside penalty:

```text
U_hold(t) = LCB90(mean[A_ref | s_t]) + 0.25 * min(q10[A_ref | s_t], 0)

HOLD iff U_hold(t) > 0; otherwise request EXIT.
```

The `0.25` penalty is fixed before training and is not swept. EXIT no-fill means remain
open and reevaluate next second. This target is non-circular: its reference never depends
on the policy being fitted.

### 5.5 Distributional heads

Set `H = 300` seconds for local path heads. Let `e = min(t+H, 15:55)` and let `E_u`
denote the deterministic exit-until-filled value starting at second `u`.

| Head | Exact target | Loss | Censoring | Consumer |
|---|---|---|---|---|
| `A_ref_mean` | `H_t - E_t` through 15:55 | squared/Huber | always observed under bid-or-zero boundary law | primary action LCB |
| `A_ref_q10/q50/q90` | conditional quantiles of `H_t - E_t` | monotone pinball | same | q10 enters action penalty; q50/q90 calibration and attribution |
| `downside_300` | `min_{u in (t,e]}(E_u-E_t)` | q10/q50/q90 pinball | clip to 15:55 and mark `horizon_clipped` | diagnostic and catastrophic-risk attribution only |
| `recovery_300` | `max_{u in (t,e]}(E_u-E_t)` | q10/q50/q90 pinball | same | diagnostic tail-recovery/convexity report |
| `giveback_300` | `max_{u in [t,e]} E_u - E_e` | q10/q50/q90 pinball | same | floor-on/off harvest report |
| `remaining_tail_300` | `H_t - E_e` | q10/q50/q90 pinball | zero when `e=15:55` | tail-capture attribution |

Missing seconds are not forward-filled across the freshness bound. No-bid follows the
frozen zero-value law; invalid locked/crossed intervals mask action and are counted. A
trajectory ending at 15:55 is terminal, not right-censored. Quantiles are parameterized
monotonically; post-hoc sorting is forbidden.

Only `A_ref_mean` and `A_ref_q10` drive HOLD/EXIT. The other heads have the exact
diagnostic consumers shown above and cannot be added to a utility after results. This
prevents the same realized path term being charged several times under different names.

### 5.6 Model families and deterministic floor

Run an HGB baseline on engineered causal state first. Only after its labels, calibration,
and replay pass machinery checks may a small temporal neural sequence model challenge it
using the same heads and composer.

The learned exit is primary position management. The independent catastrophic backstop
is fixed once at entry and never trails a forecast or running maximum. For entry fill
`P_entry` in option-price dollars and the frozen `$3` completed-round-trip fee:

```text
floor_bid = max(0, 0.50 * P_entry + 3/100)

trigger when current executable bid <= floor_bid
equivalently: net PnL after the $3 fee <= -50% of entry premium dollars
```

Priority is: 15:55 forced flat, feed-loss safety, catastrophic floor, learned exit, hold.
The floor emits a `DETERMINISTIC_EXIT` intent with `PROTECTIVE_EXIT` urgency through the
same governor/executor seam; only feed-loss/terminal safety may use the contract's
`RISK_GOVERNOR`/`FORCED_FLAT` origin. The floor is not fitted, calibrated, blended into
the model score, or counted as a learned EXIT. Headline evaluation is floor-on;
floor-off is a mandatory ablation.

## 6. Combined evaluation

### 6.1 Four-box attribution

Run all four boxes on identical outer sessions and the same serial account law:

| Box | Entry | Exit | Question |
|---|---|---|---|
| A | P5-under-cap/control | best transparent control | honest baseline |
| B | frozen learned entry | best transparent control | incremental entry edge |
| C | P5-under-cap/control | learned exit | incremental exit edge |
| D | frozen learned entry | learned exit | integrated Path-D edge/interactions |

The exit comparator panel is fixed before results: exit-immediate; hold-to-flat;
time exits at 60, 300, and 900 seconds; every combination of stops at -25%/-50%,
targets at +25%/+50%/+100%, and those three time limits; legacy P5 lifecycle where its
inputs are valid; and eight matched-rate random exit schedules. "Best comparator" means
the maximum result across this entire honest panel, not a favorable baseline selected
after seeing the candidate.

### 6.2 One-account serial replay

Every box starts with `$10,000`, trades one contract, permits one open position, enforces
premium-plus-fee affordability, D48 5% per-trade capital-at-risk, D49 5% daily loss,
no entry after 15:29, and exact 15:55 flat. Candidate and comparator each own an
independent causal ledger after their first divergence. No overlap, unaffordable trade,
or end-of-session position may appear in headline evidence.

Each model BUY/SELL becomes `ExecutionIntentV1`, is checked by
`DeterministicGovernor`, and is submitted to `SimulatedExecutor`; order events, fill/no-
fill, occupancy release, and virtual latency are reconstructed from the executor
transcript. Direct dataframe PnL that bypasses this boundary is diagnostic-only.

### 6.3 Acceptance gates

All of the following are required for a Tier-S feasibility pass:

1. Box D beats the best honest comparator in pooled net PnL and has a strictly positive
   one-sided 95% session-block-bootstrap lower bound on the paired pooled delta.
2. Box D's paired net-PnL delta is positive in at least four of the five chronological
   outer folds. No fold may violate the survival/account controls.
3. Box B passes the entry attribution gate and Box C or a documented B/C interaction
   explains the integrated result; a Box D win supplied only by P5 timing/contract choice
   is product drift.
4. The action-conditioned calibration gate passes separately for ENTER, WAIT, HOLD, and
   EXIT. Calibration uses only each fold's disjoint earlier sessions. Each action needs
   at least 30 distinct trajectories; one-sided 90% interval coverage must be at least
   85%, and mean realized value by predicted decile must be monotone within a
   session-bootstrap uncertainty band. Insufficient action evidence falls back to the
   transparent control and returns `insufficient_evidence`, never pass.
5. The base result remains directionally positive at `$4` fees and across the latency
   rungs. A sign flip routes owner review.
6. Floor-on and floor-off results are both reported. If floor-on large-win capture or
   harvest ratio falls by more than 20% relative to floor-off, route
   `owner_decision_required` even if net PnL improves.

PR and average precision are reported for rare EXIT diagnostics only. ROC AUC, PR/AP,
accuracy, win rate, and row-level significance cannot lead acceptance.

### 6.4 Required owner-facing outputs

Every table starts with unique session count and distinct trajectory count before
1-second row count. Report paired session PnL, total/mean/median PnL, profit factor,
drawdown, worst day, daily-positive fraction, exposure/occupancy, churn, no-fill rate,
side/moneyness/time/premium/regime concentration, skipped opportunities, calibration,
and an equity curve.

The fee-adjusted four-bucket Pickles report uses return on entry premium:

| Bucket | Frozen reporting boundary |
|---|---|
| Big win | `return >= +25%` |
| Small win / scratch | `-5% < return < +25%` |
| Small loss | `-30% < return <= -5%` |
| Big loss | `return <= -30%` |

It is report/tripwire-only and may not enter loss, early stopping, architecture choice,
or threshold selection. Big-loss share above 4% or near-zero big-win count routes owner
review rather than automatic tuning.

## 7. Mandatory self-fooling guards

1. **Early-exit confound panel.** Learned exit must beat exit-immediate and matched-rate
   random exits, not merely hold-to-flat. The six-day smoke's apparent PnL improvement is
   explicitly invalid evidence because exit-immediate also improved PnL while OOF
   `A_ref` R-squared was `-0.5381`.
2. **OOF skill report.** Report session-weighted OOF R-squared for `A_ref_mean`, MAE,
   calibration by action, and policy economics. R-squared less than or equal to zero is
   a no-skill tripwire; economics cannot be called learned signal unless the candidate
   also clears the full comparator panel.
3. **Matched random.** Entry uses the existing eight full-ladder schedules. Exit uses
   eight outcome-blind schedules matched on fold, session, side, premium, time-held band,
   and learned EXIT count. No redraw or best-seed selection.
4. **Negative controls.** Constant output, globally shuffled targets, sign-reversed
   forecasts, and time-shifted features must fail to produce accepted edge.
5. **Future mutation.** Mutating all information after the decision timestamp must leave
   features and predictions identical; labels must change in at least one positive
   fixture, proving the test is live.
6. **Large-win skepticism.** A candidate improvement above two times the best comparator
   or above two session-bootstrap standard deviations triggers an automatic audit of
   fill rates, action rates, fee/slippage accounting, duplicate trajectories, temporal
   leakage, holdout access, concentration, and comparator exposure before any claim.
7. **Scale lesson.** The full-corpus `corr(pnl, a_hold)` is approximately `+0.02`, not the
   six-day `-0.42`. Neither correlation is a hypothesis or expected sign. Both appear only
   as a warning against small-sample narratives.

Minimum power is defined on clusters, not 37 million seconds:

- each outer fold must retain at least 25 non-degraded sessions;
- each model-fit fold must have at least 60 sessions and 500 distinct eligible exit
  trajectories before fitting an exit model;
- each outer fold must have at least 50 filled OOF-entry trajectories, with at least 300
  pooled across folds;
- entry economic evidence needs at least 40 executed trades per fold and 200 pooled;
- final protected scoring needs at least 25 non-degraded holdout sessions and 50 trades.

If any threshold is missed, return `insufficient_evidence`; do not lower it after seeing
results or substitute row counts for session/trajectory power.

## 8. Build sequence and feasibility gate

1. **Freeze the research preregistration.** Hash this reviewed plan, exact session lists,
   feature lineage, targets, comparators, model families, hyperparameter budgets, seeds,
   fill law, floor, and metrics.
2. **Build entry machinery.** Reuse minute substrate and FT2-10 controls; pass schemas,
   negative fixtures, mutate-future, and inner-fold tests.
3. **Fit entry HGB, then neural challenger.** Produce fold-specific OOF decisions,
   calibration, controls, and entry-only economics. If entry fails, stop.
4. **Freeze entry.** Preserve one artifact bundle per fold plus a full-fit research bundle;
   exit generation may consume only fold-specific OOF bundles for evidence sessions.
5. **Build exit trajectories/labels.** Extend the validated hold-to-flat smoke to frozen-
   entry OOF positions under the single fill-law hash.
6. **Fit exit HGB, then neural challenger.** Pass the confound panel, calibration,
   distributional diagnostics, minimum power, and exit-only economics.
7. **Run four-box combined replay** through `ExecutionIntentV1` + governor +
   `SimulatedExecutor`, then freeze the complete Tier-S packet.
8. **Open the protected holdout once.** No repair is allowed after opening; failure is
   recorded and a new data-era plan is required for another claim.

The exact result that justifies further spend/formalization is: Box D satisfies every
gate in §6.3 on the five outer folds, then remains positive on the one-shot protected
holdout with no self-fooling, survival, or claim-boundary violation. Anything less is
`no_genuine_signal`, `insufficient_evidence`, or `owner_decision_required`, not edge.

Even a pass means only: Tier-S feasibility evidence sufficient to consider the next
phase. It cannot justify promotion, paper, or real-money operation.

## 9. Deferred until the feasibility gate passes

Do not formalize, sign, reseal, or implement these five shelved Path-D governance
contracts in this plan goal:

1. Path-D model-plane v2 contract.
2. FT2-60 one-second exit objective v2.
3. Additive FT2-08 Path-D 1-second exit tensor/label contract.
4. Path-D provisional fill/no-fill/latency law.
5. Path-D Phase-1 authority overlay.

Also deferred:

- `cmbp-1` Tier-T acquisition and trusted raw-event floor/stop labels;
- full A1 execution-policy rewrite, execution reconciliation, graph reissue, and legacy
  resupersession;
- live adapters, no-order shadow, paper, registry/default changes, cutover, promotion,
  and any real-money review.

## 10. Open design decisions for Claude/owner

1. Confirm the conservative default that initial entry alpha remains exactly the signed
   17, with Path-D microstructure enrichment exit-only.
2. Confirm `H=300s`, the fixed `0.25` exit downside penalty, and the 90% calibration
   levels; changing them must occur before build and enter multiplicity.
3. Confirm the fixed net -50%-of-premium catastrophic floor rather than the current
   governor overlay's moving 80%/giveback fixture.
4. Confirm VIX/ES/VX remain diagnostics/strata in the first run rather than model alpha.
5. Accept that the final 30-session firewall is protected from model/economic inspection
   but not pristine from the already published full-corpus aggregate label statistics.

STOP_FOR_CLAUDE_VERIFICATION
