# Layer 3 OOS Validation — 2026-04-21

## TL;DR

**FAIL.** On 20 trading days of genuine out-of-sample data
(2026-03-05 to 2026-04-01) — the days that exist in v2/data.pt cache
but were never included in any fold's test set — the in-sample lift
collapses dramatically:

| Metric | In-sample | OOS (20 days) | Δ |
|---|---:|---:|---:|
| Layer-2 alone PF | 1.472 (corrected) / 1.801 (fold 4) | **0.869** | **−0.603 / −0.932** |
| Layer-2 + Layer-3 PF | 2.228 / 3.492 (fold 4) | **1.028** | **−1.200 / −2.464** |
| Layer-2 alone DD | 35.6% / 32.2% (fold 4) | 25.9% | −9.7 / −6.3 pts |
| Mean $/trade alone | +234 / +322 (fold 4) | **−$68** | sign flip |
| Mean $/trade composed | +306 / +465 (fold 4) | +$10 | basically flat |

**The in-sample lift was largely selection bias.** Layer-3 still adds
+0.159 PF on OOS (0.869 → 1.028), so the exit policy itself
*directionally* generalizes — but it cannot rescue an entry policy
that crashes from PF 1.801 (fold 4) to 0.869 in the immediately
following 20 days.

Per the strategic plan's dependency graph, Option 2 (sizing) and
Option 3 (regime gating) are formally gated out. Option 1 (Layer 3
v3.1 cleanup) remains because it includes a permutation-importance
diagnostic that may explain WHAT the in-sample model was leaning on
that doesn't transfer.

## Method

This is the OOS retest the prior strategic plan called for, executed
on cached data that already exists in v2/data.pt and the per-day
sidecars (no Polygon resubscription needed).

1. **Identified 20 trading days** in V2Dataset that fall after
   fold-4's last test_day (2026-03-04) and on or before v2/data.pt's
   last day (2026-04-01).
2. **Built Layer-2 export rows** for those days using the same
   `build_export_rows_for_day` function the in-sample bundle uses
   (1817 bar-rows total).
3. **Applied fold-4 Layer-2 model** (entry + side, from
   `v3/artifacts/layer2_shared_enc_fixedq_detach/folds/4/`) — no
   retraining; pure inference.
4. **Applied fold-4 calibration thresholds** (entry=0.598,
   side=0.726) — no recalibration.
5. **per_day_choice** → 20 chosen trades (1.0 per day, slightly
   above in-sample 0.917).
6. **Trained "fold-5" Layer-3 model** on all 275 in-sample chosen
   trades (folds 0-4) for the freshest possible Layer-3 to apply.
7. **Applied Layer-3 at threshold 0.17** to OOS chosen trades.

Script:
[v3/analysis/layer3_oos_validation.py](../analysis/layer3_oos_validation.py).
Artifact:
[v3/artifacts/layer3_oos_validation/](../artifacts/layer3_oos_validation/).

This is methodologically clean OOS:
- The fold-4 Layer-2 model trained on data through ~late 2025; never
  saw 2026-03-05+.
- Calibration thresholds were frozen from in-sample val days.
- Layer-3 fold-5 model trained on chosen trades through fold-4's test
  end (2026-03-04); never saw 2026-03-05+ chosen trades.

## Results — OOS

| Metric | Value |
|---|---:|
| OOS days | 20 (2026-03-05 to 2026-04-01) |
| Chosen trades | 20 (1.00 per day) |
| Call% | 50.0% (vs fold-4 in-sample 41.7%) |
| **Layer-2 alone PF** | **0.869** |
| Layer-2 alone DD | 25.9% |
| Layer-2 alone mean $ | −$68 |
| **Layer-2 + Layer-3 PF (thr=0.17)** | **1.028** |
| Layer-2 + Layer-3 DD | 24.0% |
| Layer-2 + Layer-3 mean $ | +$10 |
| Mean bars held (Layer-3) | 107.2 |
| Early exit share | 100% |

Layer-3 lift on OOS: +0.159 PF (0.869 → 1.028) and +$78 mean trade
($-68 → $10). DD compresses from 25.9 to 24.0%.

Compare to in-sample lift on the same model+threshold:
- In-sample composed lift over Layer-2-alone: +0.756 PF (1.472 → 2.228)
- OOS lift: +0.159 PF — about 21% of the in-sample lift transfers

So Layer-3 adds *some* value on OOS, just much less than in-sample.

## Comparison to fold-4 in-sample

The OOS window is the 20 trading days immediately following fold-4's
test window (2025-12-05 to 2026-03-04). Fold-4 in-sample was the
strongest fold; OOS is the worst window observed:

| Window | Days | PF (Layer-2 alone) | PF (composed) |
|---|---:|---:|---:|
| Fold 0 | 67 | 0.860 | 0.808 (heuristic) |
| Fold 1 | 60 | 1.439 | 1.039 |
| Fold 2 | 60 | 1.038 | 1.570 |
| Fold 3 | 60 | 2.095 | 4.498 |
| Fold 4 | 60 | 1.801 | 3.492 |
| **OOS (this doc)** | **20** | **0.869** | **1.028** |

The OOS window's PF 0.869 sits between fold 0 (0.860) and fold 2
(1.038). It looks like the system entered another fold-0-like regime
right after fold 4's strong window ended. The fold-0 fragility
problem the prior research kept flagging is now manifest in OOS.

## What this changes about the prior verdicts

- **Stage 4 reality checks (PASS)** still hold AS IN-SAMPLE
  characterizations: threshold sensitivity, random-exit baseline,
  slippage robustness all hold within sample. But "robust within sample"
  is a much weaker claim than "robust out of sample," and this OOS
  result confirms the gap.
- **Layer 2's PF 1.455 (asymmetric) / 1.472 (corrected)** is best
  understood as the IN-SAMPLE ceiling on the 5 walk-forward folds.
  Forward-deployment expectation is materially lower.
- **The strategic plan's "fresh data acquisition is the gate to
  paper trading"** is vindicated. Without OOS validation, in-sample
  numbers were systematically optimistic.

## Caveats

- **20 trades is a small sample.** Standard error on PF is large.
  The 95% CI on PF=0.869 from 20 trades probably extends past 1.0
  in either direction. So technically "FAIL" by the strict criterion
  but "indeterminate" if you weight statistical power.
- **Single window.** This is one specific 20-day market regime
  (early March to early April 2026). It happens to look like a
  fold-0-style chop period. A different OOS window might give a
  different answer. The honest interpretation is that the system
  has at least one observable failure mode in the wild.
- **Call% shifted up** to 50% on OOS vs fold-4's 41.7%. The model
  thought calls were appropriate; they didn't pay off. This is
  consistent with regime mis-detection — Layer-2 sees features that
  it associates with bullish regimes during a mostly-chop period.

## Implications for the strategic plan

Per the dependency graph from the prior turn:

```
Option 4 (fresh data) -- gates everything else  [FAIL]
    ├── Option 1 (v3.1 cleanup) -- low risk, do anytime  [STILL PROCEED]
    ├── Option 2 (sizing) -- gated on Option 4 validating  [SKIP]
    └── Option 3 (regime gating) -- speculative  [SKIP]
```

Option 1 still proceeds because:
1. Permutation importance (sub-task B) is now MORE valuable, not
   less. It may reveal which features in the Layer-3 model didn't
   generalize.
2. Teacher augmentation (sub-task A) was originally about fixing
   fold 1. With OOS now showing the bigger problem is generalization,
   not in-sample fold 1, sub-task A's value drops — but running it
   provides a clean comparison point.

Option 2 (sizing) is explicitly skipped: sizing amplifies whatever
edge is underneath, and the OOS edge is essentially zero. Sizing on
PF 1.028 with high variance just makes losses bigger.

Option 3 (regime gating) is also skipped. It was explicitly conditioned
on Option 4 PASSING. With Option 4 FAILING, the regime-gating idea is
even more relevant in spirit (the OOS failure looks regime-driven) but
the framework can't be built reliably on a 20-trade OOS sample.

## What's next strategically

Given OOS FAIL, the honest options are:
- **Cold-start forward collection.** Set up an IBKR paper-account
  scraper that pulls SPX 0DTE chains daily. By early-mid May we'd
  have ~15-20 days of forward OOS to compound on top of the 20 days
  cached. Multiple windows reduce the single-window-bad-luck risk.
- **Pivot the research question.** If the system genuinely doesn't
  generalize, the project's core premise (predict per-bar
  direction on SPX 0DTE) may not have a practical edge at this
  account size. Consider alternative trade structures (spreads,
  calendars, butterflies) where the payoff geometry is different.
- **Re-examine the entry gate alone.** The random-direction PF
  was 1.116 in-sample. If THAT generalizes (Layer-2's entry-side
  was the load-bearing piece), a much simpler "entry gate +
  teacher-conditioned direction + Layer-3 exit" system might still
  have value. Worth a follow-up OOS test.

These are user-judgement calls, not in scope here.

## Verification

- [x] `python -m py_compile v3/analysis/layer3_oos_validation.py` passes
- [x] OOS validation runs end-to-end on cached data (~3 min)
- [x] Sanity: OOS days strictly post-2026-03-04 (last fold-4 test day)
- [x] No retraining; fold-4 model + calibration applied as-is
- [x] Layer-3 fold-5 model trained on all 275 in-sample chosen trades
- [x] Verdict explicitly tagged FAIL with caveats documented
- [ ] Commit
