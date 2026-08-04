# Path-D Wave 1 Exit Repair — Frozen Pre-registration (2026-08-03)

**Status:** `FROZEN_BEFORE_FIT`

**Objective:** determine whether the already-validated Protocol065 recovery-penalty mechanism, the
mandatory deterministic fallback established by Protocol073/087, and a fixed class-balance repair can
turn the degenerate Path-D exit into a tail-preserving OOF policy. `NO_EDGE` is a successful terminal
answer. The protected holdout is spent and will not be opened.

## Prior-art ruling

The canonical sources are:

- `v4/docs/protocol101/training/history/PROTOCOL101_PRIOR_CAMPAIGN_DISTILLATION.md`
- `v4/docs/protocol101/training/history/PROTOCOL101_PROTOCOL_FARM_LINEAGE_2026_07_19.md`

The executable check was run before freezing this family.

| Mechanism term | Result | Ruling |
|---|---|---|
| `recovery penalty` | One non-blocking lineage hit. Protocol065 passed the screen and Protocol066 validated it over ten seeds. | Allowed. This is the known repair, applied to the current causal one-second Path-D state/label game. |
| `rebalanced exit label` | No canonical hit. | Allowed as deterministic sampling only. The `a_ref_dollars` label values and law remain unchanged. |
| `risk lower bound calibration` | No canonical hit. | Allowed as a fixed policy ablation. No threshold is selected from results. |

Protocol073 rejected removing the Protocol054 fallback, and Protocol087 again rejected removing the
upstream fallback/router. Those are not overrides: this wave retains a fixed fallback whenever the
`fallback` parameter is true. Protocol029/030/031 loss/giveback exits, H1 relaxed suffix max, H2 generic
regret weighting, H3e profitable-only targets, H3f regret regression, side-blind L3, score thresholds,
and generic early-exit knobs remain blocked.

## Frozen data and law

- Data, scratch, and artifact volume: `/Volumes/AR_TRADING_DATA`.
- Entry campaign: `/Volumes/AR_TRADING_DATA/artifacts/entry_v2/campaign.json`.
- Five chronological OOF folds and their embargoes are inherited byte-for-byte from the entry campaign.
- Target values remain `a_ref_dollars = hold_to_1555_value_dollars - exit_until_filled_value_dollars`.
- `FILL_LAW`, the causal clock, and the entry-to-exit OOF firewall are not modified.
- Headline fees remain the frozen `$1.50` per filled side. The `$0.65` rate is not used to score
  (it is the **IBKR commission line item only**, not the all-in cost; measurement on 2026-08-04 gave
  **$1.54/side all-in**, so the frozen $1.50 is very nearly correct). Original text continued:
  this wave.
- Phase-1 timing remains 2,336 ms decision emission lag and 1,000 ms order latency. This does not alter
  the separate `live_opra_training_twin` foundation, whose blockers remain unresolved.
- Model class, 52-feature order, chronological 20% calibration suffix, fit/calibration caps, and seeds
  `(301, 302, 303)` remain the existing Phase-1 HGB baseline.

## Fixed repair definitions

### Recovery penalty (`P`)

Protocol065's exact logistic constants are carried over. Its residual sign is mapped to Path-D's target:
Protocol065's negative residual (`exit now` is worse than fallback) is Path-D's positive `a_ref` (`hold`
is better than exit now). For training row `i`:

```text
recovery_strength = sigmoid((recovery_300_dollars - 250) / 200)
regret_strength   = sigmoid((a_ref_dollars - 100) / 200)
early_strength    = sigmoid((5 - seconds_held/60) / 2)
penalty           = 1[a_ref_dollars > 0] * recovery_strength * regret_strength * early_strength
weight_multiplier = 1 + 4 * penalty
```

This multiplier is combined with the existing session/trajectory-balanced sample weight and renormalized.
It is not the closed generic H2 regret weighting; regret is only one term in P065's frozen four-factor
recovery penalty.

### Class balance (`C`)

For both the fit and calibration partitions, preserve all finite positive rows up to the existing row cap
and retain an equal number of negative rows by the existing deterministic identity SHA rank. This produces
a 50% positive target without changing or synthesizing any label value. Both classes remain load-bearing;
this is not H3e profitable-only training. The original and effective positive rates are recorded before
fit for every fold. Any candidate without `C` that fails the 25%-90% `label_balance` band is rejected at
pre-flight and is not trained.

### Deterministic fallback (`F`)

The learned policy may request an exit only before the frozen reference fallback. The deterministic
policy remains:

```text
STOP_50 -> TARGET_100 -> MAX_HOLD_25M -> final 15:55 flat
```

The catastrophic `STOP_50` floor has priority and is never learned or shuffled. `TARGET_100` and
`MAX_HOLD_25M` fire if no earlier learned exit occurs. Negative controls transform only learned utility;
the deterministic fallback remains identical in every arm.

### Reduced lower-bound contribution (`D`)

The existing utility is parameterized without selecting a threshold:

```text
utility = mean + lcb_weight * (mean_lcb90 - mean) + 0.25 * min(q10, 0)
```

The baseline has `lcb_weight=1.0`; the sole fixed reduction is `0.5`. There is no result-driven grid.

## Declared family and budget

Budget and declared maxT family size are both **8**. Order is frozen.

| ID | Mechanism searched by the loop | P | C | F | D (`lcb_weight`) | Purpose |
|---|---|---:|---:|---:|---:|---|
| W1-H01 | `recovery penalty` | yes | no | no | 1.0 | Exact P065 mechanism; expected cheap pre-flight rejection if raw balance remains below 25%. |
| W1-H02 | `recovery penalty` | yes | no | yes | 1.0 | P065 plus mandatory fallback; expected cheap pre-flight rejection if raw balance remains below 25%. |
| W1-H03 | `rebalanced exit label` | no | yes | no | 1.0 | Isolate fixed 50/50 sampling. |
| W1-H04 | `risk lower bound calibration` | no | yes | no | 0.5 | Isolate the pre-declared half-strength lower-bound contribution after balance repair. |
| W1-H05 | `recovery penalty` | yes | yes | no | 1.0 | P065 plus class balance; fallback ablation. |
| W1-H06 | `rebalanced exit label` | no | yes | yes | 1.0 | Class balance plus required fallback; recovery-penalty ablation. |
| W1-H07 | `recovery penalty` | yes | yes | yes | 1.0 | Primary complete known-repair package. |
| W1-H08 | `recovery penalty` | yes | yes | yes | 0.5 | Complete package plus the sole fixed calibration ablation. |

No ninth hypothesis may be added. A pre-flight rejection still counts as a declared look. Partial models,
failed jobs, or unattractive intermediate results may not be replaced with new hypotheses.

## Replay, controls, and family correction

Every trained candidate is replayed with the existing four-box implementation. Its comparator is the
best honest comparator selected by that implementation. All economics use the frozen fee/latency law.

The three required negative controls are:

1. sign-reversed learned utility;
2. learned utility reassigned across sessions within outer fold and entry-policy role by a frozen SHA
   permutation and normalized-time interpolation;
3. constant zero learned utility.

The catastrophic floor and deterministic fallback are unchanged in controls. If any control beats the
candidate's best comparator pooled, the wave is immediately `INVALID` and all candidate results are
discarded.

Family-wise inference uses `session_blocked_max_t` with 9,999 session sign-flip permutations and seed
1065 across every trained member of the declared family. Pre-flight-rejected members remain recorded in
the eight-member family and cannot be replaced. `maxT_p_one_sided <= 0.05` is required. No family is
created per seed or component.

## Fixed acceptance and diagnostics

`TIER_A` requires all of the following:

1. pooled PnL beats the best comparator and paired delta is positive in at least four of five folds;
2. one-sided 95% session-bootstrap delta LCB is positive;
3. all negative controls fail;
4. policy p99 and top-decile PnL contribution are each at least 80% of hold-to-15:55;
5. first-step exit rate is at most 50%;
6. effective training-label positive rate is within 25%-90%;
7. exit-score decile monotonicity passes;
8. the charter big-loss share is at most 2%;
9. maxT survives;
10. no one session or weekday contributes more than 50% of positive paired delta.

Pooled improvement with another failed condition is `TIER_B`; no pooled improvement is `NO_EDGE`.
`INVALID` has precedence. The packet reports all four rejection tests and all four charter diagnostics:
four-bucket distribution, harvest ratio, underwater duration, and PnL concentration. Win rate and PnL
concentration remain report-only.

The loop stops on `INVALID`. Because maxT is defined over the complete declared matrix, a `TIER_A` can
only be certified after the eight-row family is complete; no further hypothesis is run after that family.

## Hard stop

No protected-holdout access, paid download, broker connectivity, live/paper submission, promotion,
paper-default change, runtime flag, or scheduling mutation is authorized by this pre-registration.

