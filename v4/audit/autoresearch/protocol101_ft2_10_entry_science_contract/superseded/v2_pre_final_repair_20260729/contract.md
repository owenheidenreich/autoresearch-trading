# Protocol101 FT2-10 Entry-Science Contract — v2 Repair

Status: **PRODUCER REPAIRED**

Node: `FT2-10-ENTRY-SCIENCE-CONTRACT`

Repair goal: `FT2-10-REPAIR`, attempt 1 of 2

Outcome: `producer_repaired`

Highest allowed claim:

> FT2-10 is repaired against the FT2-20 findings under the amended authority; FT2-11 repair is unblocked.

FT2-11 is not repaired here. No model was trained or fitted, no threshold was
tuned, no session statistic was computed, and no recorder, broker, paid-data,
runtime, promotion, confirmation, or protected resource was accessed.

## 1. Scope And Owner Decisions

This contract freezes the entry component that maps causal market state and the
complete governed 42-contract SPXW 0DTE ladder to:

```text
WAIT
or BUY one exact eligible contract
```

It also freezes the serial population on which that component is evaluated.
It does not define the learned HOLD/EXIT component.

Two owner decisions recorded on 2026-07-29 are binding:

1. **Component freeze.** Entry freezes as a component only on quality evidence,
   all safety gates, and no harm versus P5 in serial dollars. `DeltaQ` is a
   diagnostic/inner-loop quantity, not a promotable dollar claim.
2. **Tail guard.** Every campaign evaluates a mandatory `alpha=0.0` no-quality-
   screen control arm. The selected setting must satisfy a relative tail-capture
   tripwire defined here and completed statistically by FT2-11.

The recorder is deliberately stopped by owner decision. Historical/IBKR
transfer can use only existing lawfully classified evidence. Insufficient
existing evidence blocks activation; this contract assumes no new paired day.

## 2. Verified Authority And Repaired Inputs

The consolidated authority was verified before repair:

```text
2363d3f986daba20bd5087ed751dc5b2d839e76cd6413aeca0bcd255eb98857a
```

| Input | SHA-256 |
|---|---|
| Consolidated product authority | `2363d3f986daba20bd5087ed751dc5b2d839e76cd6413aeca0bcd255eb98857a` |
| Graph V2 | `b06a26be59307c130da84f2dc5b6f3224c272e6c4093e83abd5bc0b280ca6d09` |
| FT2-04 v2 `label_spec.json` | `3a8518724a7c8b707007968c5dad279f70ee85919051669455b381e04e7bbf60` |
| FT2-04 v2 `oracle_rules.json` | `b435d5fd8e37aca0919196e9f8e4b2d79132e3dd8dca04cc592887e2c6b79ddb` |
| FT2-04 repaired receipt | `c763dee85293d73a4f367627e8f483a209565d3f486f82fb5b1c17f3ff611775` |
| FT2-05 v2 `census_results.json` | `3e826c9f043fdea6c252890e70f592ce0bd84d10224fc8092e7fc7b15057de0e` |
| FT2-05 repaired receipt | `0cec80e1ce029bcbec200c991f9183a9b05f1e21b28016fe944fee0ae029b33a` |
| FT2-08 v2 `tensor_schema.json` | `16e82c3ffb1612034fac817908c13fc20701ebfbad0162106db5f714eb5deca5` |
| FT2-08 v2 `label_join_spec.json` | `7ec40fa1d56e8acf3926bb214c9b4a426b954e3d77b1f29013077a9b338ed09d` |
| FT2-08 v2 `fold_roles.json` | `19710f535500c09ead3fbde32acadb002e2e0752dcd9f3a5c4c123463a19b1c4` |
| FT2-08 v2 `account_state_ledger_spec.json` | `83c854e6a8fdc9e293133d4797c2c97b05293b4cd7cd88d40e7732065146fb14` |
| FT2-08 v2 `replay_authority_v5_1_spec.json` | `a8cfd134cc7005078a8d2569d2f368c1ed3b4a870ea36ebb7af3bda9248f9ba1` |
| FT2-08 repaired receipt | `584612af00f8e04b9b903f1fb4b56396a3dc950bd21f18718fa72fd4ff8dfb9c` |
| FT2-20 joined review | `8825972e3ca99b7ee1950935e37a341699e7c78e7de6b93170270230ba50c0c3` |
| FT2-20 receipt | `b7beae089444bdec73fadec1037cbc1abd4bd6e29f1333346615a486164544f5` |
| FT2-08 repair crosswalk | `0bab5217c3e4e8b6fbf308439cddede494224a86a1a1b9b26ed583816c647627` |

The v1 FT2-10 packet is preserved byte-for-byte under:

```text
superseded/v1_pre_ft2_20_repair_20260729/
```

Its receipt SHA-256 is:

```text
ec3eff0d42947717b00d2b2055aa20e80ccfb22459b1285923e2c2bd79c682f9
```

## 3. Complete FT2-10 Finding Scope

`findings_crosswalk.json` covers every FT2-20 finding whose original anchor
names FT2-10 and every FT2-08 repair-crosswalk entry that names FT2-10 as an
owner. No severity is downgraded.

| Finding | Contract-side repair |
|---|---|
| `S1-03`, `S3-10` | Frozen universal one-account serial population, occupancy, WAIT, separate ledgers, neutral lifecycle, and exact P5-under-cap algorithm. |
| `S1-04` | Mandatory `alpha=0.0` arm and relative tail-capture tripwire. |
| `S1-05`, `S2-07` | Premium-band-and-phase nested empirical CDFs with exact weighting, ties, interpolation, scope, and fallback. |
| `S1-07` | Adjacent-strike cluster; WAIT compares selected cluster with the best outside cluster. |
| `S2-01` | Runtime horizon availability uses causal clock/model state only; stripping labels must leave actions bit-identical. |
| `S2-02` | Every census reference points to the embargo-free 45-session v2 packet. |
| `S2-03` | Component-freeze objective; `DeltaQ` demoted; serial-dollar no-harm added. |
| `S2-04` | Mandatory WAIT-probability and regret heads, losses, data path, calibration, and non-skippable gate. |
| `S2-08` | Final model uses a disjoint calibration set and is never refit after calibration. |
| `S2-10` | Eight frozen nonselectable control schedules; no redraw or best-attempt selection. |
| `S3-03` | Candidate-specific paired-source population, tolerances, strata, minimums, and p95 cluster-gap margin. |
| `S3-06` | Composer consumes FT2-08's global complete-ladder law; incomplete flat ladder means WAIT only. |

## 4. Forecast And Action-Conditioned Outputs

The six path-property families and registered horizons remain:

```text
early drawdown
time to first real profit
pre-profit adverse excursion
underwater burden
profitable-window stability
upside

horizons = {3,5,10,20,45,90,remaining-session}
```

The economic timing is inherited from FT2-04 v2:

- A BUY decided at completed minute `t` fills only at the executable ask at
  `t+1`.
- Descriptive marks begin at `t+2`.
- A learned EXIT decided at `v` would fill at `v+1`; the entry-neutral
  lifecycle below has no learned EXIT.
- A no-bid future mark is a full-loss state.
- Forced flat uses the exact 15:55 bid or zero option value when no bid exists.

Continuous heads emit monotone `q10, q25, q50, q75, q90`; time to first profit
uses discrete survival; counts use categorical heads; events use binary heads.
The exact losses and calibration routes are in `forecast_heads.json`.

### 4.1 Mandatory WAIT probability

At each governed flat decision, a global set head emits:

```text
p_wait = P(realized-label frozen composer says WAIT | causal state)
```

It consumes causal shared state, permutation-invariant contract
representations, causal masks, and path forecasts. Its target exists only after
the physically separate training label join. Masked binary cross-entropy fits
the head; same-final-model disjoint calibration uses deterministic isotonic
regression.

### 4.2 Mandatory selected-contract regret

Conditional on ENTER, the model emits:

```text
expected normalized regret
raw q90 normalized regret
```

For realized whole-path quality `Q`:

```text
r = max(0, max_Q_in_exact_risk_set - Q_selected) / 2
```

`r` lies in `[0,1]`. Expected regret uses masked squared loss. The q90 head uses
masked `tau=0.90` pinball loss and is parameterized not below the mean. The
final upper bound is:

```text
min(1, raw_q90 + conformal_q90(max(0, r - raw_q90)))
```

on the final model's disjoint calibration sessions.

Missing, untrained, uncalibrated, or failed action-conditioned outputs make a
candidate ineligible for component freeze. They cannot be skipped. Before
independent acceptance and owner signature, any runtime rehearsal is WAIT-only.

## 5. Causal Composer

### 5.1 Complete-ladder and safety law

While flat, all 42 BUY actions are false unless the FT2-08 global
`complete_ladder_mask` and every other global entry mask pass. Each contract
also needs a current quote, valid source-neutral identity, tradability,
affordability, D48, and that policy ledger's D49 mask. An incomplete flat
ladder has one legal action: WAIT.

### 5.2 Runtime horizon availability

Let the prospective fill minute be `e=t+1`.

- Finite `hN` is available only when `e+N <= 15:55 ET` and every required
  model output at `hN` is finite.
- `remaining_session` is available only when `e<=15:30` and the causal clock
  has at least one minute in `(e,15:55]`.
- A horizon crossing 15:55 is unavailable, not shortened.

The composer may not read `target_valid`, label-derived `forecast_valid`,
`label_join_present`, future quote availability, future censoring, or any label
partition field.

Acceptance invariant:

> Remove all label partitions and label-derived masks before historical
> composition. Ordered scores, horizons, actions, WAIT reasons, and tie traces
> must remain bit-identical.

### 5.3 Quality screen and v2 tail evidence

The anchor grid is:

```text
alpha in {0.00, 0.10, 0.20, 0.25, 0.30}
```

`alpha=0.00` skips learned quality thresholds but retains every safety,
positive-upside, uncertainty, and action-gate rule. It is mandatory in every
campaign evaluation.

All cited diagnostics now come from the repaired, embargo-free FT2-05 v2
packet:

| Alpha | Qualifying-minute share | Joint excluded-big-winner rate | Mean trades/session |
|---:|---:|---:|---:|
| `0.00` | `0.7843640914` | `0.0000000000` | `2.6000000000` |
| `0.10` | `0.7662517289` | `0.0645842542` | `2.5777777778` |
| `0.20` | `0.7391819798` | `0.1569053583` | `2.5777777778` |
| `0.25` | `0.7125074096` | `0.2257133736` | `2.6222222222` |
| `0.30` | `0.6731212540` | `0.2968174451` | `2.6222222222` |

These are design diagnostics from existing v2 tables, never candidate-quality
evidence.

### 5.4 Premium-conditioned dual-unit arbitration

The completed-minute executable ask at decision `t`, never the future fill
ask, assigns the contract to census-v2 bands:

| Band | Causal ask |
|---|---|
| `cheap_le_1` | `<= $1` |
| `small_1_3` | `> $1 and <= $3` |
| `medium_3_8` | `> $3 and <= $8` |
| `large_8_20` | `> $8 and <= $20` |
| `very_large_20p` | `> $20` |

For every metric, horizon, unit, band, and phase, the empirical CDF reference
contains only that model's nested fit sessions. Each contributing session has
equal weight; its rows divide that weight equally.

For a distinct value `v`, the CDF knot is:

```text
weight(y < v) + 0.5 * weight(y = v)
```

Exact ties use the knot. Values between adjacent distinct knots interpolate
linearly. Below minimum is `0`, above maximum is `1`, and a single-valued
reference maps equality to `0.5`.

Each exact band-phase stratum needs eight sessions and 2,000 finite rows. The
only fallback pools phases within the same premium band. Cross-band fallback
is forbidden; insufficient same-band evidence fails the affected rank closed.

Dollar and return favorable percentiles are therefore compared within the same
premium band before taking their minimum. No scored block or chronological
future block enters its own CDF.

### 5.5 Cluster-aware uncertainty

The selected contract's adjacent-strike cluster contains surviving contracts
with:

```text
same expiry
same right
strike within +/- 5 points
```

The WAIT gap compares the selected cluster's best score to the best surviving
contract outside that cluster. When no outside contract exists, WAIT with
score zero is the comparator. A nearly tied adjacent strike never creates a
WAIT by itself. Exact-contract uncertainty is handled by the regret head.

Model-gap conformal error and historical/IBKR source-gap error use this same
cluster-versus-outside quantity.

## 6. Same-Final-Model Calibration

Earlier-block residuals may guide nested search but may not calibrate a
different refitted final model.

For an outer-training role with `N` ordered sessions:

1. Reserve the chronologically last
   `max(20, ceil(0.20*N))` sessions as final calibration.
2. Remove the immediately preceding session as an embargo.
3. Fit the final model, transforms, CDFs, and guardrail references only on all
   earlier sessions.
4. Predict the disjoint calibration sessions once.
5. Fit every final calibrator from those predictions.
6. Never refit or update the model afterward.
7. Freeze all hashes before outer validation.

This scheme calibrates the actual final model. Outer validation, confirmation,
holdout, census-only rows outside the fit sessions, and live rows cannot fit or
change a model, CDF, threshold, or calibrator.

## 7. Frozen One-Account Serial Population

### 7.1 Universal grid and states

Every governed entry minute through 15:29 appears on one universal chronological
grid. Each policy independently occupies one of:

```text
FLAT_WAIT
FLAT_BUY_COMMIT
PENDING_TPLUS1_FILL
OCCUPIED_NEUTRAL_LIFECYCLE
SOFT_CLOSED
DAILY_STOPPED
```

A pending or occupied policy has no flat action. The row remains in paired
denominators with `Q=0` and `U=0`; no impossible BUY is scored. A true flat
WAIT also has `Q=0`, `U=0` and cannot disappear.

At decision `t`, BUY reserves the slot. At `t+1`, the policy is pending while
the executable fill resolves. The actual ask must still satisfy
affordability, D48, and policy-ledger D49. No ask or a failed recheck opens no
position, realizes no PnL, records a failed execution, and permits another
decision no earlier than `t+2`.

### 7.2 Separate causal ledgers

Each outer fold starts candidate, P5, and every control at `$10,000`; each
policy carries only its own equity between sessions inside that fold.

Every policy independently recomputes:

- D48 from 5% of its session-start equity.
- D49 from its realized session loss plus prospective actual entry cost.
- Permanent soft close below `$103` remaining budget.
- Signed realized-PnL daily stop.
- Affordability and occupancy.

The `$3` primary and `$4` stress paths are separate causal trajectories. An
extra dollar cannot be subtracted after replay because it can change later
safety state.

### 7.3 Neutral lifecycle and re-entry

Every successful entry is held to exact 15:55, then valued at the executable
bid or zero option value when no bid exists. This neutral, uncapped lifecycle
defines entry-layer serial dollars and safety without giving the entry model a
learned or fixed profit-taking exit.

After any realized exit, re-entry is legal only at the first later completed
minute when flat and before 15:30. The neutral lifecycle exits at 15:55, so a
successful fill has no same-session re-entry. A failed `t+1` fill may retry at
`t+2`.

## 8. Exact P5-Under-Cap Comparator

P5 acts at every universal-grid minute where its own ledger is flat, not
pending, before 15:30, and globally eligible.

Direction:

```text
C when finite SPX close >= finite session VWAP
P otherwise
WAIT when either input is missing/nonfinite
```

Among currently eligible contracts on that side, P5 selects the minimum:

```text
(
  abs(offset_points),
  offset_points,
  0 for C / 1 for P,
  expiry,
  strike,
  right,
  source_neutral_contract_id
)
```

P5 waits when occupied/pending, when a global safety condition fails, when its
state is missing, when no selected-side contract survives, or after its own
soft close/daily stop. It uses its own `t+1` fills, D48, D49, neutral
lifecycle, and cash ledger. It reads no model forecast, candidate action,
label, future availability, or other policy's state.

The complete algorithm lives in `objective_spec.json`; that file's receipt hash
is the algorithm hash.

## 9. Component-Freeze Objective

`DeltaQ` and `DeltaU` remain useful for inner search and diagnosis on the
universal serial grid. They cannot freeze or promote the entry component.

Entry component freeze requires all three channels:

### 9.1 Quality evidence

- Nested OOF skill for registered head families against constant and strong-
  shuffle controls under the repaired FT2-11 statistics.
- Mandatory WAIT reliability and selected-contract regret pass.
- Every quality family and upside reported; no hidden harmful family.
- Relative tail guard passes or routes owner review.

### 9.2 All safety gates

Every identity, complete-ladder, causality, no-bid, fill, D48, D49, soft-close,
daily-stop, forced-flat, fee-stress, one-position, survival, and source-
activation prerequisite remains mandatory.

### 9.3 No harm versus P5 in serial dollars

For every common governed session:

```text
DeltaSerialDollar_session
  = candidate neutral-lifecycle PnL cents
  - P5 neutral-lifecycle PnL cents
```

No-trade sessions contribute zero policy PnL. The primary path uses `$3`; the
hard stress is a separate `$4` replay. FT2-11 must freeze the noninferiority
margin, adjusted bound, multiplicity, and terminal rule before any campaign
result.

Neither positive absolute PnL nor `DeltaQ` can replace this no-harm evidence.

The entry layer may claim only that it earned a component freeze. Promotable
dollar claims belong to the combined learned-entry plus learned-lifecycle
system at FT2-80.

## 10. Tail Guard

For every serious candidate, fold, seed, ensemble, and fee path, the evaluated
alpha arm is paired with the same model and `alpha=0.0` arm. All non-alpha
state, masks, CDFs, calibration, `k`, and serial laws are identical.

A big-win path has FT2-04 v2 remaining-session fee-adjusted MFE return at least
`+40%`.

Each arm reports:

1. Big-win filled entries per governed session.
2. Positive MFE dollars from big-win filled entries per governed session.
3. Share of positive neutral-lifecycle PnL contributed by big-win paths.

The evaluated-arm value divided by the no-screen value is reported for all
three. A zero control denominator is `tail_guard_insufficient_evidence`, never
an imputed pass. FT2-11 freezes the clustered intervals, material-capture
thresholds, multiplicity, and owner-decision trigger before results.

These tail metrics cannot enter a loss, selection score, or stopping rule.

## 11. Historical-To-IBKR Transfer

Candidate-specific activation requires paired historical-versus-IBKR evidence,
not merely two implementations reading the same IBKR observation.

Only existing `development`-classified paired sessions may enter. Burned,
validation, sealed, unassigned, protected, or newly recorded sessions are
forbidden here.

Pairs use exact completed-minute and source-neutral contract identity. Identity,
slot maps, masks, actions, integer cents, and clock fields require equality.
FP32 model fields require:

```text
abs(reference-runtime) <= 1e-6 + 1e-5*abs(reference)
```

Candidate action agreement must be at least `0.99`. The battery is stratified
by phase/opening warmup, recenter events, missing/stale quotes, reconnect/reset
when present, every active extension channel, and open-state outside-ladder
rows when present.

At least three distinct development sessions and 500 paired flat decisions
are required. A missing required natural stratum means
`source_transfer_insufficient`; synthetic fixtures cannot manufacture source
evidence.

The source margin is the nearest-rank p95 absolute historical/IBKR difference
in selected-cluster-versus-best-outside-cluster score gaps:

```text
k = ceil(0.95*n), one-based, no interpolation
```

Zero is legal only when every paired gap is bit-identical. Insufficient current
evidence permits historical component research with transfer marked
unavailable, but blocks decision-shadow, live, paper, and broker activation.

## 12. Frozen Controls

Mandatory controls remain HGB, MLP, constant output, nearest ATM, exact
P5-under-cap, sign reversal, strong target shuffle, feature-family ablations,
and matched-random full-ladder controls.

Random controls use exactly eight seeds:

```text
10121001 .. 10121008
```

Each schedule is a SHA-256 ranking of canonical opportunity identities. It is
created once without realized labels, replayed on its own causal ledger, and
checked against the frozen D1 exposure table. There is no redraw and no
best-seed selection. All eight attempts—including incomplete or invalid
attempts—enter the ordered evidence ledger. If any schedule is incomplete or
fails matching, the result is `INSUFFICIENT_MATCHED_CONTROL`; another draw is
not allowed.

FT2-11 owns multiplicity across all controls and admitted extension-channel
sets. FT2-10 makes those identities nonselectable and reproducible.

## 13. Machine Authority And Stop

Top-level v2 machine contracts:

- `forecast_heads.json`
- `composer_spec.json`
- `calibration_spec.json`
- `objective_spec.json`
- `controls_spec.json`
- `findings_crosswalk.json`
- `receipt.json`

No model-quality or dollar result follows from this design repair.

Completion is `producer_repaired`. The sole next goal is the separate FT2-11
repair. FT2-20 reruns with fresh review seats only after all three contract
repairs are complete.
