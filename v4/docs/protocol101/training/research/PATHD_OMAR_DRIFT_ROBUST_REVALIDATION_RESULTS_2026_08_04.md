# Path-D omar Drift-Robust Re-Validation — RESULTS (2026-08-04)

**Verdict: `OMAR_NULL_FRAGILE_CLOSED`. The last survivor of the sixty does not survive a null that
handles drift honestly. The directional programme on this feature set is closed.**

Pre-registration:
[`PATHD_OMAR_DRIFT_ROBUST_REVALIDATION_PREREGISTRATION_2026_08_04.md`](PATHD_OMAR_DRIFT_ROBUST_REVALIDATION_PREREGISTRATION_2026_08_04.md)
(frozen and committed as `55dd6de0` before the run; not edited afterwards).
Evidence: `v4/audit/autoresearch/pathd_omar_drift_robust_revalidation_2026_08_04/`
(receipt `40960ba799964470`).

213 of 215 development sessions (two half-days dropped for no fresh SPX row at the boundary),
200 surrogates per session per null, 20,000 permutations. **Zero cost: owned data, no training, no
broker, no purchase. This run did not open the protected holdout**
(`protected_holdout_opened: false`).

> **⚠ CORRECTION, same day.** This document originally stated `holdout_open_count` = 0 here and in
> §5, and called the protected holdout an "unspent" surviving asset. **That is wrong. The holdout is
> SPENT** — opened once on 2026-08-02 for the `signed18` confirmation later invalidated by a 60-second
> look-ahead (`holdout_open_count: 1` in
> `v4/audit/autoresearch/autoresearch_v2_entry_model_confirmation_2026_08_02_attempt001/holdout_access_receipt.json`).
> My error was reading this run's per-run receipt flag `protected_holdout_opened: false` — which
> means only that *this run* did not open it — as if it were the global counter. The two are easy to
> conflate and are not the same field. The corrected statement of assets is in
> [`PATHD_PROGRAMME_STAND_DOWN_RECORD_2026_08_04.md`](../contracts/PATHD_PROGRAMME_STAND_DOWN_RECORD_2026_08_04.md).
> **There is no confirmation firewall left**, which strengthens rather than weakens this document's
> conclusion: it removes the last route by which a surviving `omar` could have been confirmed.

---

## 1. The result in one table

Excess IC of `omar_clipped_neg3_pos3`, reversion sign, against each null:

| Null | 15m | 30m | 60m | maxT p (60m) | Verdict |
|---|---|---|---|---|---|
| **Reference wild** (Option 0's null, re-run unchanged) | +0.052 | +0.063 | **+0.078** | 1.0e-06 | **null is VOID** — fails its known-answer gate |
| **Zero-drift wild** (exact `\|r_t\|` placement, no drift) | +0.018 | +0.018 | **+0.019** | **0.403** | **FAILS** |
| **Permuted-time wild** (marginal `\|r\|`, placement destroyed) | +0.058 | +0.067 | **+0.075** | 0.0000 | passes |

Negative controls clean on all twelve members (min p 0.9987). Raw omar ICs reproduce the Option 0
receipt to **exactly 0.0** — same corpus, same clock, same feature, so the only thing that changed
is the null.

**Under the frozen verdict key, failure under either valid repaired null closes the survivor.**
Both repaired nulls passed their known-answer gates, so the key applies without discretion:
`OMAR_NULL_FRAGILE_CLOSED`.

## 2. The known-answer gate did its job — including on the null that produced the original result

The gate measures the pooled fixed-bin 60-minute surrogate profile, the exact statistic where the
drift defect was demonstrated on 2026-08-04. A valid null must be flat there.

| Null | surrogate top−bottom spread | Spearman ρ vs bin index | Gate |
|---|---|---|---|
| Reference wild (Option 0) | **10.121 points** | **+1.000** — perfectly monotone | **FAIL** |
| Zero-drift wild | 0.108 points | +0.406 | pass |
| Permuted-time wild | −0.007 points | −0.430 | pass |
| *(the real data, for scale)* | −0.507 points | +0.200 | pass |

The reference row is a **positive control that fired**: the null which generated the +0.078 result
manufactures a monotone 10-point slope where the real data is flat. Its p-value of 1.0e-06 is a
measurement of its own defect. This is not a re-interpretation after the fact — the defect was
demonstrated and committed (`78df5eb3`) before this study was designed, and the gate threshold was
frozen before the run.

## 3. What separates the two repaired nulls, and why the failing one is the one to believe

Both repaired nulls are true martingales. They differ in exactly one property: **whether the
intraday volatility profile stays where it actually was.** Zero-drift wild preserves `|r_t|` at
every timestamp; permuted-time wild keeps only the marginal distribution and shuffles the loud
minutes.

That single difference is worth **+0.056 IC units** (+0.019 → +0.075) — three times the residual
that remains under the tighter null. So the thing surviving under permuted-time is **intraday
volatility seasonality, not predictability.** The principle is the ordinary one: the null that
matches the real path in every respect *except* the property under test is the informative one, and
the closer the match, the less is left. Here the closest match leaves +0.019 at p = 0.40.

**The honest caveat, preregistered before the run:** zero-drift wild is *directionally
conservative* for a reversion claim. Real sessions carry drift that dilutes the real path's
bounded-path artifact, while the driftless surrogate's artifact is undiluted, which biases
`real − surrogate` toward the momentum side. A refutation resting only on a null I declared
conservative would be a weak refutation.

**It does not rest on that.** §4 is the same conclusion with no null at all.

## 4. The economics, with the null removed entirely

*Post-hoc and descriptive, in the manner of `pathd_omar_causal_sizing.py` — no test, no gate.* The
causal fade rule (long at `omar ≤ −0.6`, short at `omar ≥ +0.6`, fixed thresholds available at
decision time), pooled and count-weighted over **28,238** candidate boundaries:

| Bias correction applied | SPX points/trade | On one ES contract |
|---|---|---|
| Reference wild (Option 0's null) | **+3.610** | **+$180.48** |
| Zero-drift wild | **−0.279** | −$13.95 |
| Permuted-time wild | **−0.320** | −$16.02 |
| **None — the raw real data** | **−0.324** | **−$16.21** |

The measured friction bar is **0.358 points ($17.92)**.

Three things follow, and none of them needs a null to be adjudicated:

1. **The disputed $178/trade pooled estimate is fully explained.** It reproduces here as +3.610
   points ($180.48) under the reference wild and nowhere else. It was the defect, measured.
2. **The real fade rule loses money before costs.** −0.324 points per trade gross, −0.682 points
   (**−$34**) net of measured friction. The `NOT_IDENTIFIED` effect size from this morning now has
   an identified value, and it is negative.
3. **Flipping the sign does not rescue it.** Momentum at the same thresholds is +0.324 points
   gross — **below the 0.358-point friction bar** — so it is a loser after costs too. Both
   directions are foreclosed by one number.

## 5. What is now closed, and what is untouched

**Closed.** `omar` as a tradable directional signal on this feature set, at 15/30/60 minutes, on
SPX. With it, the last of the sixty members from the Option 0 re-screen. Ledger entry added to
[`PROTOCOL101_PRIOR_CAMPAIGN_DISTILLATION.md`](../history/PROTOCOL101_PRIOR_CAMPAIGN_DISTILLATION.md)
§4.

**Preserved, and this is the part worth keeping.** The Option 0 §1 *association* finding is not
retracted and was never wrong on its own terms: within a session, omar's rank relationship to
forward returns is real. What three studies have now established is that **the association is a
description of a bounded path's arithmetic and its volatility seasonality, not a predictability a
trader can hold.** That is a coherent, finished answer to the World A / World B question the last
two months could not isolate — and the answer is World B.

**Preserved at zero cost.** The execution plane. The research machinery. The measured ES friction
bar, which remains a durable, instrument-level number and is what made §4 decidable. The owned
corpus. The 0DTE long-premium class stays closed, structurally, and nothing here reopens it.

**Not preserved, contrary to this document's original claim — see the correction at the top.** The
protected holdout is **SPENT** (`holdout_open_count` = 1, opened 2026-08-02). No confirmation
firewall remains, so any future hypothesis needs fresh live paper or a newly reserved holdout drawn
from unused data.

## 6. Reproduction

```bash
PYTHONPATH=. python -m v4.research.pathd_omar_drift_robust_revalidation   # refuses a second run
```

`receipt.json` (self-sealed), `results.csv` (12 rows), `results.md`. Flags: `diagnostic_only: true`,
`protected_holdout_opened: false`, `model_trained: false`, `paper_order_submitted: false`.

Two design defects were caught by the known-answer gate *before* freezing, during the declared
smoke check, and are recorded in the pre-registration §4: an uncentered block-mean drift resample
(surrogate spread 10.9 points) and a centered one (5.5 points). The second is a finding in its own
right — **a 30-minute persistent drift burst is itself short-horizon momentum, so a
"drift-burst-preserving martingale" is self-contradictory at horizons that overlap the burst.**
That is why the owner's "resample the drift in bursts" prescription could not be implemented as
stated, and why N2 became the permuted-time wild instead.

*Signed: Claude Fable 5 — 2026-08-04, under the owner's "review and execute" directive on the Fable
brief.*
