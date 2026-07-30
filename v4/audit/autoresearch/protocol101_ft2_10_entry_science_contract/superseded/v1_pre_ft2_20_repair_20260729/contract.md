# Protocol101 FT2-10 Entry-Science Contract

Status: **FROZEN DESIGN**

Node: `FT2-10-ENTRY-SCIENCE-CONTRACT`

Outcome: `design_ready`

Sole next node: `FT2-11-EVIDENCE-STATISTICS-CONTRACT`

Highest allowed claim:

> The entry-science contract is frozen; FT2-11 may be drafted against it.

## 1. Scope

This document freezes how a flat-state Protocol101 model turns causal market
and full-ladder state into one of 43 actions:

```text
WAIT
BUY one exact contract from 21 strikes x 2 rights
```

It does not train a model, fit a calibrator, tune a threshold, compute a new
market statistic, evaluate outer validation, access protected or recorder
data, or authorize paper trading. It does not define the later HOLD/EXIT
model.

The learned model receives the FT2-08 90-minute market history, 42-contract
identity-aligned paths, current ladder, masks, internal Greeks, and phase/clock
state. P5 supplies none of its timing, direction, or contract choice.

## 2. Authority And Frozen Inputs

| Input | SHA-256 |
|---|---|
| Consolidated product authority | `893aa0664944680e053ffd12a4d44c8a798397cbeed6ad56cd682fd864d9f832` |
| Graph V2 | `b06a26be59307c130da84f2dc5b6f3224c272e6c4093e83abd5bc0b280ca6d09` |
| FT2-04 `label_spec.json` | `d724195999acf579a43768756b79f30ad122ec446933347b93cbbd8a7bbe9781` |
| FT2-04 `oracle_rules.json` | `23a4cab2715c925a3c4261a508e69ac2c0f70157b93083ff129f4848ae03eb81` |
| FT2-05 `census_results.json` | `8a8805b22ef04a290edf421d5229d75eebc2301bc5246bd3bc595ee616cae583` |
| FT2-05 receipt | `9085a09cbc1583973fdb372c78d78568e3fb1baa0fa1d9b42c23b05543ee24de` |
| FT2-08 `tensor_schema.json` | `890015b87c0a2636b554d4c27aea9261f91604783f70bde0a96f402e4469e585` |
| FT2-08 `storage_spec.json` | `a92645af245016490376ffff85d7e338aa37b73ea706b4040f1c159c9dc9a8c9` |
| FT2-08 `fold_roles.json` | `e5f21f566341b3d2b10baa476a6af309f31ba658a5d944a99bd6015aff032599` |
| FT2-08 `label_join_spec.json` | `8695f4bbde17c4375b67efeadffe1d8c85576544eea8b729d2a7564733651847` |
| Signed G8 v2 | `ac2cdbd9dabf8daa36aa53b8736d9691b8569b6f5b2b810d5a56707702465bf3` |
| D1 v2 | `fcaf69ae080418cf6146f08d90386665c3c9f576f7382f4de8d8c51c23cf8598` |

The Graph V2 node has one success route:
`design_ready -> FT2-11-EVIDENCE-STATISTICS-CONTRACT`.

## 3. What The Model Predicts

The entry model predicts six separate properties of each contract's future
path at the registered horizons `3, 5, 10, 20, 45, 90, remaining-session`:

1. Early drawdown.
2. Time to first real profit.
3. Adverse excursion before first profit.
4. Underwater depth and duration.
5. Stability of profitable windows.
6. Upside.

Early drawdown exists only at 3, 5, and 10 minutes because that is the FT2-04
definition. Near the close, horizons remain nominal and use masks/censoring;
they are not silently shortened.

Every continuous economic target is predicted in fee-adjusted dollars per
contract and return on entry premium. Continuous heads emit calibrated
quantiles:

```text
q10, q25, q50, q75, q90
```

They use masked pinball loss and monotone parameterization. Time to first
profit uses a right-censor-aware one-minute discrete survival head. Integer
durations and run counts use categorical distributions. Boolean events use
binary cross-entropy. Forecast calibration is chronological and training-only.

The full head list and loss definitions are frozen in
`forecast_heads.json`.

### Multi-head weights

The base loss gives each family `1/6`, each applicable horizon equal weight,
each metric equal weight, and paired dollar/return targets `1/2` each.
Autoresearch may multiply each family weight by a value from `[0.5, 2.0]` and
renormalize. It may not zero, add, delete, or redefine a family, horizon,
metric, or unit.

## 4. Deterministic Composer

The network does not directly emit an unexamined BUY command. Its path
forecasts enter one frozen composer.

### Stage 0: physical and safety mask

Only exact contracts passing the FT2-08 causal quote, identity, tradability,
affordability, D48, D49, and action masks are considered. WAIT remains
available. Missing required state fails the affected contract closed.

### Stage 1: loose quality screen

The screen excludes the predicted bottom tail of five quality families before
upside is considered. It uses these canonical components:

| Family | Horizon and component | Conservative forecast | Pass direction |
|---|---|---|---|
| Early drawdown | h10 worst | q10 in dollars and return | higher |
| Time to first profit | remaining-session event minute | q90, no-event after boundary | lower |
| Pre-profit adverse excursion | remaining-session PPAE | q10 in dollars and return | higher |
| Underwater burden | remaining-session depth-duration integral | q90 in both units | lower |
| Profitable-window stability | remaining-session positive-minute share | q10 | higher |

For an exclusion level `alpha`, a higher-good threshold is the phase-specific
training-calibration `alpha` quantile. A lower-good threshold is the
`1-alpha` quantile. A contract must pass every component and both dollar and
return units where both exist.

The frozen anchor grid is:

```text
alpha in {0.10, 0.20, 0.25, 0.30}
```

This brackets the literal worst quartile without turning the screen into an
upside ranking. It comes from the existing FT2-05 curves, not new
computation. On the 46-session census, the same anchors retained qualifying
minutes at rates `0.7754, 0.7490, 0.7222, 0.6856`; their joint excluded-big-
winner rates were `0.0676, 0.1677, 0.2365, 0.3067`. These are design
diagnostics, never model-quality evidence.

### Stage 2: conservative-upside ranking

Only screened contracts are ranked. At every valid horizon, the composer uses
calibrated q10 forecasts for:

```text
MFE
profit area
```

For each metric/horizon it converts dollar and return forecasts to favorable
training-phase percentiles and takes the smaller percentile. It then takes the
smaller of balanced MFE and balanced profit-area percentiles. The primary
contract score is the equal-horizon mean of these values.

This is the dollar/percentage arbitration rule: a cheap option cannot win only
because its percentage is large, and an expensive option cannot win only
because its dollar potential is large.

The equal-horizon mean q10 MFE and q10 profit area must each be strictly
positive after the frozen $3 round-trip fee in both units. Equality means
WAIT.

Ranking ties resolve by:

1. Higher mean balanced-upside percentile.
2. Higher worst-horizon balanced-upside percentile.
3. Higher mean q10 MFE return.
4. Higher mean q10 MFE dollars.
5. Lexically smaller exact identity tuple
   `(expiry, strike, right, contract_id)`.

Nearest-ATM, P5, iteration order, and slot index are not tie-breaks.

### Uncertainty abstention

Three independently seeded models provide epistemic dispersion. Chronological
OOF predictions provide a 90% split-conformal error for the predicted
top-versus-runner-up score gap. Candidate-specific transfer evidence later
provides p95 source score-gap drift.

```text
WAIT margin = k * (model gap error + source-transfer gap error)
k in {1.00, 1.25, 1.50, 2.00}
```

The top contract must beat the runner-up and WAIT by more than the margin.
Its conservative dollar and return edges must also exceed their conformal
absolute-error margins. Equality means WAIT. Missing live-transfer evidence
blocks live activation; it does not authorize looking at recorder evidence in
this node.

### Complete WAIT law

The action is WAIT when:

- No contract is physically/safely eligible.
- Required state is missing.
- No contract clears all five guardrails.
- Conservative upside is not strictly positive after fees in both units.
- The top action is inside calibrated model/source uncertainty.
- Confidence controls behavior before its separate gate is accepted.

## 5. Training-Only Calibration

For each outer fold, ordered training sessions are divided into five
contiguous blocks. Blocks 2-5 receive OOF predictions from models fit only on
earlier blocks after removing the last preceding session as an embargo. Block
1 is warmup and produces no calibration prediction.

All distribution calibrators, phase guardrails, uncertainty errors, and
composer settings come from those OOF training-role predictions. The final
outer model may be refit on all outer-training sessions only after these
choices freeze. Outer validation never calibrates or changes anything.

Phase-specific calibration requires at least eight distinct sessions and
2,000 valid rows. Otherwise the globally pooled training calibrator is used
and the fallback is recorded. Calibration as a whole requires at least 20
sessions, 1,000 decisions, and 10,000 valid contract rows. Less evidence
produces `calibration_insufficient_evidence`.

Every fold emits and hashes its prediction identities, calibrators, guardrail
thresholds, composer choice, and source margin.

### Joint conservatism

Guardrail alpha and uncertainty multiplier are selected together on inner OOF
evidence. Settings must fall inside this broad design target:

```text
0.6630434783 to 5.3043478261 mean trades per session
```

The FT2-05 anchor-center rate is `2.6521739130`. The band is
`[max(0.3, center/4), min(6.0, 2*center)]`. It is deliberately asymmetric:
abstention has more room than overtrading. It is a calibration target, not a
gate or daily trade cap. If no setting enters the band, the result is
`joint_calibration_insufficient`; outer evidence stays closed.

## 6. Action-Conditioned Calibration Gate

Signed G8 remains report-only. Because confidence now controls WAIT and
contract choice, this separate entry gate is preregistered:

### WAIT reliability

The predicted quantity is the probability that no contract in the exact risk
set deserves entry under the realized-label composer. The observed outcome is
one only when that frozen audit-only composer says WAIT.

Required thresholds:

- ECE at most `0.10`.
- Maximum equal-frequency-bin gap at most `0.20`.
- Brier skill versus the constant-rate forecast at least `0`.

### Selected-contract regret

Realized regret is the best realized whole-path quality `Q` available in the
same risk set minus the selected contract's `Q`, floored at zero and divided
by two because `Q` lies in `[-1,1]`. The resulting regret is in `[0,1]`. The
model reports expected normalized regret and a 90% upper bound.

Required thresholds:

- Absolute mean expected-regret calibration error at most `0.10`.
- 90% bound empirical coverage between `0.85` and `0.95`.

The gate needs at least 20 sessions, 100 flat decisions, 50 realized WAITs, 50
realized ENTERs, and 50 selected contracts. Otherwise it returns
`insufficient_calibration_evidence`.

This node preregisters the gate; it does not activate it. Before confidence may
control accepted training or runtime behavior, the gate must be smoke-tested,
independently audited, and owner-signed. FT2-60 separately defines HOLD/EXIT
reliability and floor behavior.

## 7. Entry Metric And P5

The primary entry metric does not pretend an entry is good because one fixed
exit happened to work. It scores the entire observed path.

For each registered metric, a training-phase empirical CDF maps the realized
value to `[-1,1]`, oriented so higher is always better. Paired dollar/return
metrics take the worse transformed value. Metrics average within horizons,
horizons within families, and the five quality families equally:

```text
Q = mean(
  early drawdown,
  time to first real profit,
  pre-profit adverse excursion,
  underwater burden,
  profitable-window stability
)

U = upside family score
```

A BUY receives the selected contract's realized `Q` and `U`. WAIT receives
`Q=0, U=0`. Therefore avoiding a bad trade helps, waiting through a good
opportunity hurts, and WAIT rows cannot disappear from accounting.

P5 is replayed over the same decisions and safety masks. Per session:

```text
DeltaQ = mean(Q_model - Q_P5)
DeltaU = mean(U_model - U_P5)
```

The primary comparison is paired `DeltaQ`; `DeltaU` is secondary. All five
family deltas, action rates, direction, moneyness, and missingness are also
reported. FT2-11 defines final confidence, multiplicity, and evidence
thresholds. Fixed exits, if retained, remain report-only.

## 8. Inner Autoresearch Objective

The loop uses a lexicographic objective on inner OOF evidence:

1. Maximize the one-sided 80% moving-block-bootstrap lower bound of paired
   `DeltaQ` versus P5.
2. Maximize the corresponding lower bound of paired `DeltaU`.
3. Maximize the worst of the five family lower bounds.
4. Minimize measured inference latency, then storage.

The bootstrap uses 2,000 deterministic resamples, five-session moving blocks,
and seed `101210`. These 80% bounds control search noise only; they are not
final acceptance.

A trial resets the 12-serious-trial plateau only when the candidate-versus-
incumbent `DeltaQ` lower bound is strictly positive, the `DeltaU` lower bound
is nonnegative, no family point estimate is negative, and all controls remain
valid. The graph controller owns the counter and rule. The research loop may
not edit either.

## 9. Model Lab And Controls

The lab may change architecture, width/depth, temporal and cross-ladder
representation, registered loss weights, optimizer, regularization,
mathematically equivalent calibration implementation, and efficiency.

It may not change action space, features, masks, targets, horizons, composer
semantics, folds, fees, safety, evidence gates, or stopping rules.

Mandatory controls are:

- HGB and MLP on causal summarized features.
- Constant-output model.
- Feature-independent matched-random full-ladder control under D1 v2.
- Nearest-ATM baseline.
- P5 benchmark.
- Sign-reversed forecasts.
- Strong globally shuffled targets.
- Registered feature-family ablations.

D1 v2 randomizes timing and exact contract while matching exposure after
serial replay. A contract-only randomizer is diagnostic, not hard evidence.

## 10. Prohibited Shortcuts

The following are contract violations:

- Hindsight exit minute as an entry target.
- Max future price or MFE as the complete entry label.
- Any future path, quote, Greek, context, PnL, or advantage field at runtime.
- Oracle action or oracle contract at runtime.
- P5 timing, direction, or ATM choice upstream of the learned model.
- ATM-only or reduced-ladder action space.
- Dropping WAIT rows.
- Using fixed-exit PnL to pass entry quality.
- Selecting horizons, features, thresholds, or stopping rules from outer
  validation.
- Protected data access.

MFE remains legal only as one component of the already frozen six-family path
description.

## 11. Synthetic Composer Walk-Through

These numbers are invented and are not market statistics.

Suppose three contracts survive the safety mask. The phase guardrails are:

| Component | Higher/lower | Dollar threshold | Return/other threshold |
|---|---|---:|---:|
| h10 early drawdown | higher | `-$80` | `-35%` |
| session TTFP | lower | n/a | `20 min` |
| session PPAE | higher | `-$120` | `-50%` |
| session UWI | lower | `$2,000-min` | `10 return-min` |
| positive fraction | higher | n/a | `0.30` |

Conservative forecasts:

| Contract | Early DD | TTFP | PPAE | UWI | Positive share | Screen |
|---|---|---:|---|---|---:|---|
| 6000C | `-$60`, `-25%` | 12 | `-$90`, `-40%` | `1,400`, `7` | 0.45 | pass |
| 6005C | `-$95`, `-32%` | 10 | `-$80`, `-35%` | `1,300`, `6` | 0.50 | fail: dollar drawdown |
| 5995P | `-$70`, `-30%` | 14 | `-$100`, `-45%` | `1,800`, `9` | 0.40 | pass |

The 6005C is excluded even though several properties look attractive; its
dollar drawdown sits in the rejected tail.

For the two survivors:

| Contract | Mean q10 MFE | Mean q10 profit area | Balanced score |
|---|---|---|---:|
| 6000C | `$35`, `+12%` | `$90-min`, `0.32 return-min` | 0.68 |
| 5995P | `$28`, `+16%` | `$75-min`, `0.38 return-min` | 0.66 |

The dollar/return minimum keeps the 5995P's higher percentage from erasing its
weaker dollar axis. The 6000C ranks first, but its gap is `0.02`. If the
calibrated WAIT margin is `0.03`, the action is WAIT. If the independently
frozen margin were `0.015`, the action would be BUY 6000C. The composer may
not lower the margin after seeing this decision.

## 12. Synthetic Guardrail Calibration Walk-Through

These values are also invented.

For one phase, ten OOF h10 early-drawdown-return labels sorted from worst to
best are:

```text
-0.80, -0.60, -0.40, -0.30, -0.20,
-0.10,  0.00,  0.05,  0.10,  0.20
```

At `alpha=0.25`, the linear empirical quantile is `-0.375`. Early drawdown is
higher-good, so a conservative q10 forecast must be at least `-0.375`.

For a lower-good UWI example:

```text
1, 2, 3, 4, 5, 6, 7, 8, 9, 10
```

The threshold is the `1-alpha=0.75` quantile, `7.75`; a conservative q90 UWI
forecast must be at most `7.75`.

The same operation is performed for the paired dollar threshold. If this
phase had fewer than eight sessions or 2,000 valid rows, the hashed global
training threshold would be used and the fallback recorded. Neither the
outer-validation rows nor their action rate may alter these values.

## 13. Frozen Deliverables

Machine-readable authority:

- `forecast_heads.json`
- `composer_spec.json`
- `calibration_spec.json`
- `objective_spec.json`
- `controls_spec.json`

`receipt.json` records their SHA-256 hashes and all input hashes.

No model-quality conclusion follows from this design. The only authorized
next work is the separate FT2-11 evidence/statistics contract.
