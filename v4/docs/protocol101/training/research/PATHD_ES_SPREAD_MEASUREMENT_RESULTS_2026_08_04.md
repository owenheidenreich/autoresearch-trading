# ES Spread Measurement — RESULTS (2026-08-04)

**Status: `ES_REMAINS_CREDIBLE_BRANCH`. The assumption held — ES trades at 1.04 ticks, not 2.**

Spent **$1.478328** of a $3.00 cap. Preflight:
[`PATHD_ES_SPREAD_MEASUREMENT_PREFLIGHT_V2_2026_08_04.md`](PATHD_ES_SPREAD_MEASUREMENT_PREFLIGHT_V2_2026_08_04.md).
Evidence: `/Volumes/AR_TRADING_DATA/.../raw/databento/glbx_es_bbo_1s_measurement_2026_08_04/`.

**Authorization note for the audit trail.** The owner authorized this in conversation on 2026-08-04
("proceed to option A", then "proceed. authorized"), after I recommended against it. At their explicit
direction I supplied the `V4_PAID_DATA_APPROVAL_TEXT` string myself, which the manifest otherwise reserves
to a human. The authorization is the owner's; the keystroke was mine. Recorded here so the record is not
ambiguous about who authorized the spend.

---

## 1. What was measured

20 RTH sessions spread across the owned corpus, `GLBX.MDP3` / `ES.FUT` / `bbo-1s`, DST-aware
09:30–16:00 America/New_York. **1,095,818 quote states, 25.9 MB, 100% RTH coverage on every session.**
Each state is weighted by its duration until the next state.

| | mean spread |
|---|---|
| Unconditional | **1.0397 ticks** |
| **Elevated realized volatility** (top time-weighted quartile of causal trailing 15-minute RV) | **1.0734 ticks** |

Per-session range **1.0018 → 1.1534 ticks. No session exceeded 1.25 ticks.**

## 2. The question this was bought to answer

The Path-D ES hurdle rested on an **assumed** 1-tick spread, and that assumption was load-bearing: the
2026-08-03 adversarial review established that **at a 2-tick spread ES's advantage over passive 0DTE
options disappears entirely** (hurdle 57.38% @15m).

**The assumption held.** ES is a 1-tick market even in its noisiest quartile — the elevated-volatility
spread is 1.0734 ticks, only 3.2% wider than unconditional. The 2-tick scenario that would have closed the
branch does not occur in this sample.

## 3. Measured friction and hurdle

Friction = $4.50 commissions + elevated-RV mean spread × $12.50/tick = **$17.9176 per round trip**, against
the assumed **$17.00**. The assumption was low by 5.4%.

| Horizon | mean \|move\| | friction as % of move | mean-payoff hurdle |
|---|---|---|---|
| 15m | $300.28 | 5.97% | **52.98%** |
| 30m | $421.98 | 4.25% | **52.12%** |
| 60m | $597.09 | 3.00% | **51.50%** |

Passive 0DTE comparison: **53.9–56.6%**. **ES retains its advantage at every horizon.**

For contrast, SPXW 0DTE friction is **4.68% of a $565 premium** on a round trip that must be paid before
the position moves. ES at 60 minutes costs **3.00% of the mean move** — and the mean move is the thing the
strategy is trying to capture, not a fixed toll.

## 4. What this does and does not establish

**Established:** the denominator. ES friction is now a measurement, not an assumption, and it is durable —
it stays valid regardless of what happens to any signal question.

**Not established:** that any ES strategy is viable. This measures the *cost of trading*, not the existence
of an edge. The numerator — the size of the `omar` effect — remains
[`NOT_IDENTIFIED`](PATHD_SPX_EXCESS_SKILL_RESCREEN_RESULTS_2026_08_04.md#3b): three defensible estimators
span +0.82 / +4.54 / +3.57 points per trade, and at fixed causal omar the real forward move is flat, with
the corrected signal coming from a mis-specified surrogate drift.

## 5. What the measurement *does* give the next study: a hard target

**Any signal must be worth ≥ 0.358 ES points per trade to clear measured friction.** ($17.9176 ÷ $50/point.)

That threshold is horizon-independent in points, because friction is fixed per round trip. It converts the
open question from a vague "is the effect big enough" into a pass/fail number. Notably, **all three of the
disputed omar estimators sit above it** — the smallest, +0.82, is 2.3× the bar — which is precisely why
identifying the true value now matters more than it did this morning.

## 6. Next step

**Identify the numerator, on owned data, at zero cost.** The open question is whether a *causal*
normalization exists that captures the within-session omar association — realized-volatility scaling, or a
causal running estimate of the session's eventual range, in place of a fixed threshold or a look-ahead
within-session rank. It needs a fresh pre-registration, and it now has an explicit bar to clear: **0.358
points per trade.**

No further purchase is warranted until that resolves. Option B (longer-tenor options data) remains a hard
stop.

## 7. Verification

- 20/20 files hashed (sha256), non-empty, non-zero rows; acquisition receipt records per-file bytes, rows
  and hashes, and `guard_invoked_before_each_range_call: true`.
- Exact cost re-checked against the $3.00 cap before the first range call; the guard re-ran before **each**
  of the 20.
- `protected_holdout_opened: false`, `model_fit_executed: false`.
- `v4/tests/test_pathd_es_bbo1s_spread.py` + `test_paid_data_guard.py` — **9 passed**.

*Signed: Claude Opus 5 — 2026-08-04.*
