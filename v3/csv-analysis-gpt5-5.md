I analyzed the 1,664 trades as an exploratory diagnostic. The biggest takeaway is that the model’s realized `hl` is **not being explained well by its own prediction columns**. The useful structure is mostly in **side × sigma_pos × IV regime × time/OMAR/volume interactions**.

## Sanity check

Overall:

| Metric                  |     Value |
| ----------------------- | --------: |
| Trades                  |     1,664 |
| Mean `hl`               | **+$188** |
| Median `hl`             |  **-$53** |
| Win rate                | **46.6%** |
| Profit factor on `hl`   |  **1.88** |
| Mean `best_forward_pnl` | **+$928** |
| Mean `regret_vs_best`   | **+$730** |

So this system is **right-tail driven**: more than half the trades lose, but a subset of large winners keeps the mean positive.

`time_stop_pnl` is entirely missing/NaN in this file, so I could not compare oracle exits against holding to session end.

---

# 1. Poor-trade patterns

## A. Calls with `orc_triggered = True` are very bad

| Rule                             |  n | Mean `hl` | Median `hl` | Win rate |       PF |
| -------------------------------- | -: | --------: | ----------: | -------: | -------: |
| `side=call & orc_triggered=True` | 43 | **-$245** |   **-$239** |    34.9% | **0.39** |

This is one of the cleanest avoid flags. The model is still entering these with average `decision_margin` around **0.60**, which is not low. This looks like a systematic L2 mistake.

**Interpretation:** ORC may be useful for puts or context, but for calls it appears toxic in this sample.

---

## B. Calls in low-sigma / high-IV regime are bad

Cell definitions from the file:

| Bucket      | Approx range             |
| ----------- | ------------------------ |
| `sigma_b=0` | `sigma_pos <= -0.506`    |
| `iv_b=2`    | `iv_percentile >= 0.547` |

| Rule                      |  n | Mean `hl` | Median `hl` | Win rate |       PF |
| ------------------------- | -: | --------: | ----------: | -------: | -------: |
| `side=call & cell=s0_iv2` | 69 |  **-$94** |   **-$176** |    39.1% | **0.68** |

This means: **calls when price is meaningfully below VWAP in high IV are poor.**

The miss: this group had average `pred_dollar_score` **+0.175** and `pred_win_prob` **0.507**, so the model was not treating it as dangerous.

---

## C. Late-day puts with high positive sigma and low/mid IV are bad

| Rule                                                     |   n | Mean `hl` | Median `hl` | Win rate |       PF |
| -------------------------------------------------------- | --: | --------: | ----------: | -------: | -------: |
| `side=put & sigma_b=2 & iv_b in {0,1} & bar_index >=106` | 100 | **-$120** |   **-$228** |    27.0% | **0.53** |

This is a strong avoid pattern.

Translated:

> Late in the session, puts are poor when price is already far above VWAP and IV is not in the high bucket.

The model is taking late puts into positive-sigma conditions where the option does not seem to have enough vol/convexity support.

---

## D. High `decision_margin` is actively bad for puts

| Rule                                  |   n | Mean `hl` | Median `hl` | Win rate |       PF |
| ------------------------------------- | --: | --------: | ----------: | -------: | -------: |
| `side=put & decision_margin >= 0.647` | 147 |  **-$65** |    **-$60** |    45.6% | **0.71** |
| `side=put & decision_margin >= 0.694` | 107 | **-$106** |   **-$117** |    42.1% | **0.53** |

This is a serious calibration problem. The model’s own confidence is backwards for a subset of puts.

This may indicate that L2 is confusing “strong put setup” with “crowded/late/exhausted put setup.”

---

## E. Very-low-IV puts are bad

| Rule                                |  n | Mean `hl` | Median `hl` | Win rate |       PF |
| ----------------------------------- | -: | --------: | ----------: | -------: | -------: |
| `side=put & iv_percentile <= 0.031` | 79 |  **-$88** |   **-$138** |    40.5% | **0.64** |

Low-IV puts appear underpowered. They may occasionally work, but the distribution is poor.

---

# 2. Unusually good-trade patterns

## A. Puts in mid-sigma / high-IV regime are the best broad cell

| Rule                     |   n | Mean `hl` | Median `hl` | Win rate |       PF |
| ------------------------ | --: | --------: | ----------: | -------: | -------: |
| `side=put & cell=s1_iv2` | 128 | **+$635** |   **+$123** |    54.7% | **4.12** |

This is one of the most exploitable-looking cells.

Cell meaning:

| Component   | Approx range                 |
| ----------- | ---------------------------- |
| `sigma_b=1` | `-0.500 < sigma_pos < 1.223` |
| `iv_b=2`    | `iv_percentile >= 0.547`     |

So the strongest put setup is not “price extremely stretched above VWAP.” It is:

> Puts in high IV when price is near/mildly above VWAP.

That makes intuitive sense: there is still room for directional movement, and option convexity is alive.

---

## B. Calls with high positive `sigma_pos` are good

| Rule                             |   n | Mean `hl` | Median `hl` | Win rate |       PF |
| -------------------------------- | --: | --------: | ----------: | -------: | -------: |
| `side=call & sigma_pos >= 1.228` | 219 | **+$363** |   **+$187** |    59.4% | **3.26** |
| `side=call & cell=s2_iv0`        | 115 | **+$322** |   **+$239** |    60.0% | **3.11** |
| `side=call & cell=s2_iv2`        |  43 | **+$610** |    **+$47** |    53.5% | **4.19** |

This is probably the cleanest call-side finding:

> Calls want positive sigma. Calls below VWAP are weak; calls above VWAP are strong.

The model partially misses this. For `call & sigma_pos >= 1.228`, average `pred_dollar_score` was basically neutral, around **-0.009**, while realized `hl` was strongly positive.

---

## C. High-IV puts are good, but the model underuses the regime information

| Rule                                |   n | Mean `hl` | Median `hl` | Win rate |       PF |
| ----------------------------------- | --: | --------: | ----------: | -------: | -------: |
| `side=put & iv_percentile >= 0.742` | 236 | **+$412** |    **+$21** |    52.1% | **2.68** |
| `side=put & atm_iv >= 0.0966`       | 242 | **+$421** |    **+$38** |    52.9% | **2.76** |

The model does recognize some of this through `pred_dollar_score`, but not cleanly enough. High-IV puts still contain avoid zones, especially late-day high-sigma low/mid-IV puts, but as a broad regime, puts like elevated IV.

---

## D. High-volume puts are good; high-volume calls are bad

| Rule                                |   n | Mean `hl` | Median `hl` | Win rate |       PF |
| ----------------------------------- | --: | --------: | ----------: | -------: | -------: |
| `side=put & volume_ratio >= 1.472`  | 243 | **+$368** |     **-$8** |    49.0% | **3.11** |
| `side=call & volume_ratio >= 1.258` | 112 |  **-$23** |   **-$125** |    37.5% | **0.91** |

This is a side-dependent volume effect.

> High volume appears to help puts and hurt calls.

That is exactly the kind of interaction a simple score may miss.

---

## E. Very small OMAR range is excellent for calls

| Rule                                     |  n | Mean `hl` | Median `hl` | Win rate |       PF |
| ---------------------------------------- | -: | --------: | ----------: | -------: | -------: |
| `side=call & omar_range_pct <= 0.000505` | 65 | **+$653** |   **+$476** |    72.3% | **8.34** |

This is one of the strongest findings, though the sample is smaller.

It was robust across all five seeds. This looks like a potentially valuable L2 entry filter or boost.

---

# 3. Features the predicted scores seem to miss

The prediction columns are weakly related to realized `hl`.

| Prediction column       | Spearman vs `hl` |
| ----------------------- | ---------------: |
| `pred_dollar_score`     |           -0.009 |
| `pred_return_score`     |           -0.001 |
| `pred_win_prob`         |           -0.025 |
| `pred_clean_entry_prob` |           -0.014 |
| `pred_stopout_risk`     |           -0.059 |
| `decision_margin`       |           +0.031 |

A linear model using all six prediction/confidence columns explained only about **0.7%** of realized `hl` variance.

The most important misses:

## Miss 1: `pred_win_prob` is almost flat

High and low `pred_win_prob` buckets do not meaningfully separate outcomes. The top `pred_win_prob` quintile had mean `hl` around **+$138**, while the bottom quintile had mean `hl` around **+$240**.

So `pred_win_prob` is not giving useful ordering here.

---

## Miss 2: `pred_clean_entry_prob` may be inverted

Overall quintiles:

| `pred_clean_entry_prob` quintile | Mean `hl` |
| -------------------------------- | --------: |
| Lowest                           | **+$310** |
| 2nd                              |     +$202 |
| Middle                           |     +$142 |
| 4th                              |     +$163 |
| Highest                          | **+$123** |

For puts specifically, low `pred_clean_entry_prob` was often good:

| Rule                                        |   n | Mean `hl` |       PF |
| ------------------------------------------- | --: | --------: | -------: |
| `side=put & pred_clean_entry_prob <= 0.270` | 151 | **+$403** | **2.80** |
| `side=put & pred_clean_entry_prob >= 0.322` | 272 |  **-$23** | **0.90** |

This suggests the “clean entry” head may be penalizing exactly the high-opportunity trades.

---

## Miss 3: Call-side `sigma_pos` is underweighted

Calls show a strong monotonic improvement as `sigma_pos` rises.

| Call `sigma_pos` bucket | Mean `hl` |  Win rate |       PF |
| ----------------------- | --------: | --------: | -------: |
| Lowest bucket           |      -$33 |     41.4% |     0.87 |
| Middle bucket           |     +$190 |     45.1% |     1.96 |
| Highest bucket          | **+$338** | **59.3%** | **2.98** |

But `pred_dollar_score` barely reflects this. The model is not giving enough credit to call entries above VWAP.

---

## Miss 4: Put-side `decision_margin` is dangerous

The model appears overconfident in bad put trades.

| Put `decision_margin` bucket | Mean `hl` |       PF |
| ---------------------------- | --------: | -------: |
| Lowest quintile              |     +$157 |     1.72 |
| Middle quintile              |     +$288 |     2.31 |
| Highest quintile             |  **+$14** | **1.07** |

At stricter thresholds, this turns negative:

| Rule                             | Mean `hl` |       PF |
| -------------------------------- | --------: | -------: |
| `put & decision_margin >= 0.647` |  **-$65** | **0.71** |
| `put & decision_margin >= 0.694` | **-$106** | **0.53** |

This is probably a L2 calibration bug or a regime-confusion problem.

---

# 4. Systematic mistakes / exploitable filters

These are the strongest candidate rules I would test out-of-sample.

## Candidate avoid filters

| Filter                                             |   n | Mean `hl` | Median |       PF | Why it matters                       |
| -------------------------------------------------- | --: | --------: | -----: | -------: | ------------------------------------ |
| `call & orc_triggered=True`                        |  43 | **-$245** |  -$239 | **0.39** | Strong call-side fail flag           |
| `call & cell=s0_iv2`                               |  69 |  **-$94** |  -$176 | **0.68** | Calls below VWAP in high IV fail     |
| `put & decision_margin >= 0.694`                   | 107 | **-$106** |  -$117 | **0.53** | Confidence inversion                 |
| `put & sigma_b=2 & iv_b in {0,1} & bar_index>=106` | 100 | **-$120** |  -$228 | **0.53** | Late high-sigma puts without high IV |
| `put & iv_percentile <= 0.031`                     |  79 |  **-$88** |  -$138 | **0.64** | Very-low-IV puts underperform        |

The union of those avoid filters captures **354 trades** with:

| Group              |     n | Mean `hl` |    Median | Win rate |       PF |
| ------------------ | ----: | --------: | --------: | -------: | -------: |
| Avoid-filter union |   354 |  **-$72** | **-$143** |    41.0% | **0.72** |
| Remaining trades   | 1,310 | **+$258** |      -$29 |    48.1% | **2.27** |

That is a large improvement from the overall PF of **1.88**.

---

## Candidate boost / preferred-entry filters

| Filter                              |   n | Mean `hl` | Median |       PF | Why it matters                        |
| ----------------------------------- | --: | --------: | -----: | -------: | ------------------------------------- |
| `put & cell=s1_iv2`                 | 128 | **+$635** |  +$123 | **4.12** | Best broad put cell                   |
| `call & sigma_pos >= 1.228`         | 219 | **+$363** |  +$187 | **3.26** | Calls like positive VWAP displacement |
| `put & iv_percentile >= 0.742`      | 236 | **+$412** |   +$21 | **2.68** | High-IV put regime                    |
| `put & volume_ratio >= 1.472`       | 243 | **+$368** |    -$8 | **3.11** | High-volume put convexity             |
| `call & omar_range_pct <= 0.000505` |  65 | **+$653** |  +$476 | **8.34** | Very strong call-side OMAR condition  |

The union of these “favor” filters captures **693 trades** with:

| Group                               |   n | Mean `hl` | Median | Win rate |       PF |
| ----------------------------------- | --: | --------: | -----: | -------: | -------: |
| Favor-filter union                  | 693 | **+$366** |   +$47 |    53.7% | **2.88** |
| Favor union excluding avoid filters | 605 | **+$430** |   +$66 |    55.4% | **3.23** |

---

# Bottom line

The model’s L2 scores are not reliably ranking entered trades. The strongest exploitable structure is:

**Good:**

* Calls with positive `sigma_pos`, especially `sigma_pos >= 1.228`
* Calls with very small `omar_range_pct <= 0.000505`
* Puts in high IV, especially `put & cell=s1_iv2`
* Puts with high `volume_ratio`

**Bad:**

* Calls with `orc_triggered=True`
* Calls below VWAP in high IV: `call & cell=s0_iv2`
* Late puts with high positive sigma but non-high IV
* High-confidence puts, especially `decision_margin >= 0.694`
* Very-low-IV puts

The cleanest next experiment would be a **post-L2 veto/boost layer** using only these causal entry-time features, then test it seed-by-seed and on a held-out date split. The most suspicious bug-like behavior is that **put `decision_margin` and `pred_clean_entry_prob` appear miscalibrated or inverted** in exactly the places where the realized outcomes differ most.
