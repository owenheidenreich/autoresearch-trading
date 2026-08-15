# §4 collapses: the decision states are worth about 1,000–7,500 observations, not 93,798

**Finding, 2026-08-14. Model-free. No policy or threshold was fitted.**

## What changed for the bot

Section 4 of the signed fit reopening charged the 20-per-parameter rule against **93,798 causal decision
states**, arguing that the unit relevant to fitting is the decision the model makes. That argument was
never measured. It has now been measured two independent ways, and **it does not hold**.

The states are heavily autocorrelated — consecutive minutes share almost all of their history, ladder and
outcome window. Measured effective sample size:

| Label | Base rate | Autocorrelation time | Effective n (autocorrelation) | Effective n (design effect) | Parameter budget at 20:1 |
|---|---:|---:|---:|---:|---:|
| `reached_30_itm_60m` | 6.62% | 10.22 | **7,557** | 1,016 | 377 |
| `reached_20_itm_60m` | 17.15% | 18.86 | 4,096 | 998 | 204 |
| `reached_10_itm_60m` | 47.34% | 26.83 | 2,879 | 916 | 143 |
| `reached_30_itm_120m` | 16.15% | 19.50 | 3,961 | 590 | 198 |

§4 assumed 77,254 usable states. The two routes return **2,879–7,557** and **590–1,016**. Both are one to
two orders of magnitude below the assumption.

## What the budget admits

Taking the **most generous** reading available — the autocorrelation route on the sparsest label, 7,557
effective observations, a budget of **377 parameters** — the architecture family must be sized to it. At
the width the project had been using (8) nothing fits; at width 3 the four comparison architectures do:

| Architecture | Width 8 | Width 3 | Budget 377 |
|---|---:|---:|---|
| `shallow_joint` | 676 | **226** | fits at 3 |
| `shallow_four_head` | 720 | **245** | fits at 3 |
| `neural_joint` | 1,252 | **322** | fits at 3 |
| `neural_four_head` | 1,296 | **341** | fits at 3 |
| `four_independent` | 4,920 | 1,250 | refused at both |

**The budget binds the shallow architectures too**, which §4's version did not — it applied the ratio only
to sequence models. Effective sample size is a property of the label and corpus, so a shallow parameter
costs the same evidence as a sequence parameter.

On the design-effect route the budget falls to 29–50 parameters and **nothing in the family fits at any
width**. The [re-ruling](../../governance/CAUSAL_DAY_FIT_RERULING_2026_08_14.md) proceeds on the generous
route as a stated judgement, and requires the conservative figure to be reported alongside any result.

### A correction to an earlier version of this finding

The first version of this document stated that every architecture fails "including both shallow controls",
citing counts of 1,604 / 1,688 / 3,780 / 3,864 / 14,952. **Those were computed at
`ArchitectureDimensions.hidden_size`'s former dataclass default of 16, which no artifact in this project
ever used** — every recorded build set width 8 explicitly. The counts above are correct. The conclusion
that the family must shrink survives; the claim that a separate parameter-count defect existed does not,
and is withdrawn. `hidden_size` now has no default so the trap cannot recur.

## Why the two routes disagree, and which to believe

They disagree by about 7x, and that gap is reported rather than resolved by preference.

The **autocorrelation** route sums the within-session correlation function with Bartlett weights, stopping
at the first non-positive pair. It measures how fast a single session's series forgets itself. The
**design-effect** route compares the observed variance of session means against what independent sampling
of the same states would produce; it captures between-session clustering the first route cannot see,
because a whole session can be uniformly high or low for reasons no within-session lag reveals.

The design effect is the more conservative and, for this question, the more relevant: the economic
inference resamples sessions, so between-session structure is exactly what the parameter budget should
respect. The autocorrelation figure is quoted as the generous bound so the conclusion cannot be accused of
resting on the harsher method — **the answer is the same under either.**

The measured autocorrelation times of 10–27 minutes are physically sensible for a 60-minute forward label
and are not an artifact: a label whose outcome window is an hour long cannot decorrelate in a minute.

## The honest unit

Neither 243 nor 93,798 is right. Sessions understate the information — a session genuinely contains more
than one usable observation. Decision states overstate it by assuming minutes are independent when they
demonstrably are not. The measured answer sits between, nearer the bottom:

> **Roughly 12 to 31 effective observations per session**, by the autocorrelation route; **2 to 4** by the
> design effect.

A re-ruling of §4 that wants to admit any current architecture must either shrink the architectures by
roughly an order of magnitude, argue the 20:1 ratio itself down, or acquire sessions. It cannot get there
by choosing a different unit, because the unit has now been measured.

## Scope and limits

- The state count here is **77,254**, not the 93,798 in §4: this collapses the candidate ladder to one
  row per session-minute and only minutes carrying an eligible contract survive. The discrepancy makes
  §4's assumption more generous, not less, so it does not affect the direction of the finding.
- Effective sample size is a property of the **label**, not of a model. A model with a different target
  would need this measured again.
- This says nothing about whether an edge exists. It says how much evidence 243 sessions carry.

## Evidence

- receipt: `v4/audit/autoresearch/effective_sample_size_2026_08_14/receipt.json`
- tool: [`ops/measure_effective_sample_size.py`](../../ops/measure_effective_sample_size.py)
- corrected parameter counts: [suspension](../../governance/CAUSAL_DAY_FIT_SUSPENSION_2026_08_14.md)
- source: `/Volumes/AR_TRADING_DATA/derived/causal_day_trader_v2/candidates.parquet`

Nothing here contacted a vendor, broker or runtime; fit a model; tuned a threshold; used post-cutoff data;
or authorized promotion.
