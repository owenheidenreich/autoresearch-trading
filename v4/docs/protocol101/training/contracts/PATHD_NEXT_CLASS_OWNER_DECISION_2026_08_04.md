# Path-D — Owner Decision: what class do we research next?

Status: **OPTION 0 OWNER-AUTHORIZED (2026-08-04) — A/B/C remain open, keyed on Option 0's result**

Date: 2026-08-04. Prepared under Deliverable 5 of the owner-approved
[`PATHD_FABLE_REVIEW_IMPLEMENTATION_PLAN_2026_08_04.md`](../execution/PATHD_FABLE_REVIEW_IMPLEMENTATION_PLAN_2026_08_04.md).

---

## The screen result this decision was supposed to key on

**It did not resolve.** Frozen verdict `DIRECTIONAL_SKILL_CANDIDATE`; interpreted verdict
**`NOT_INTERPRETABLE`**. Full write-up:
[`PATHD_SPX_DIRECTIONAL_SKILL_RESULTS_2026_08_04.md`](../research/PATHD_SPX_DIRECTIONAL_SKILL_RESULTS_2026_08_04.md).

All 27 members cleared, with a headline mean Spearman IC of **+0.522** — which is precisely why it is not
believable. Matched surrogates built from each session's own realized volatility, **with no predictability
by construction**, reproduce **78.5%** of that number, and **100–119%** of every momentum member. The
features are functions of the price level `P(t)`; the target `P(t+h) − P(t)` contains `−P(t)`; the shared
term forces the correlation for any bounded path. My pre-registered session-shuffle control could not detect
this, because shuffling destroys the pairing the artifact lives in.

**Two things did come out of it, and they are real:**

1. **Momentum carries nothing directional.** SPX 5m and 15m momentum mean-reverts *less* than chance. The
   entry model's eleven momentum/VWAP variants were not contributing mean-reversion information.
2. **A level/mean-reversion residual survives** (`omar` 60m excess `−0.112`), and `session_range_bps` — a
   monotone **time-of-session** proxy — is the least artifact-explained member at 55%. Suggestive only: the
   surrogate flattens intraday volatility seasonality, and none of it has been costed on any instrument.

So the World A / World B question is still open, and it is still the cheapest question on the table.

## The options

### Option 0 — Re-run the screen with a correct null *(recommended first; free)*

Same nine features, same clock, same corpus, same 213 sessions. Replace the null with
**intraday-volatility-matched surrogates** and apply the family correction to the **excess** IC rather than
the raw IC. Add a time-of-session member so `session_range_bps` is tested as what it actually is.

- **Cost:** none. Owned data, no training, no broker, no purchase. Runs in about a minute.
- **Prerequisite:** a fresh pre-registration (the current one is frozen and its control set is now known
  to be insufficient).
- **Buys:** a decisive answer to World A vs World B — the thing the last two months of campaigns could
  never isolate. If the excess dies, directional strategies on this feature set are closed for **every**
  instrument, and Option A becomes pointless. If it survives, Option A is worth paying for.

### Option A — Confirm ES friction from GLBX quote data

ES is the only instrument whose hurdle we have measured as clearable: **52–54% at 15–60 minutes**, versus
SPXW 0DTE where friction is 4.68% of premium.

- **Cost:** paid data. Codex probe put `bbo-1s` at ≈ **$0.073/session**. **Check existing entitlements
  first** — this may already be covered.
- **Prerequisite:** owner authorization for a paid download. Also, honestly: **Protocol 028 already tested
  stitched ES futures VWAP and rejected it** (March `−$80`, Q2 `−$2,130`, Q4 `−$2,790`). There is no
  positive evidence in the record that ES helps as a feature.
- **Buys:** a measured friction number for the one instrument with a reachable hurdle. **Only relevant if
  Option 0 finds real skill.**

### Option B — Buy longer-tenor options data

**HARD STOP. Not recommended until Option 0 or A resolves.** Longer tenor has less theta per unit time and a
better friction ratio, and we have never measured it. But the variance risk premium means long premium at
*any* tenor still needs directional or vol-timing alpha — which is exactly what has never been demonstrated.
Buying data to look for it again, before establishing it exists, repeats the pattern.

### Option C — Stand down research; keep the execution plane warm

Stop the research programme. Keep IBKR paper guard/executor and the contract-identity bridge — proven,
strategy-agnostic, **zero cost** to hold. Nothing is lost that would need rebuilding.

- **Buys:** an honest stop, with the plumbing intact for whenever a strategy hypothesis worth testing turns
  up. Per the Charter and the research standard here, `NO_EDGE` is a first-class successful outcome.

## Decision key

| If Option 0 returns | Then |
|---|---|
| **No excess skill** | Directional strategies on this feature set have no evidence on **any** instrument → **Option C**, or a genuinely non-directional class opened as a new programme. Option A becomes pointless. |
| **Excess skill survives** | **Option A** before any purchase or training question. Still not a licence to train — the holdout is SPENT, so confirmation is fresh live-paper only. |

**My recommendation:** run Option 0. It is free, it takes a minute, and it is the only thing on this list
that can turn an eight-week `NO_EDGE` streak into information rather than another negative. I will not start
it without your word, because it needs its own pre-registration.

**Independent of all four:** the 0DTE long-premium class stays closed, structurally, and none of these
options reopens it.

---

## Owner decision — 2026-08-04

**Option 0 is AUTHORIZED** (explicit owner sign-off in the Fable review session, recorded by Claude
Fable 5). Binding constraints on the rerun:

1. A fresh pre-registration is required and must declare the run **diagnostic-only**: this is a second
   look at the same 213 sessions, with the hypothesis selected after seeing run 1. No outcome of the
   rerun is confirmatory; a surviving excess feeds the Option A decision, nothing more.
2. Independent-review context for calibration (Fable, 2026-08-04): pure-random-walk simulation gives
   session-level Spearman(level, forward return) of −0.37/−0.48/−0.58 at 15/30/60 m — a *stronger*
   mechanical pull than the −0.32/−0.42/−0.52 measured, so the excess may plausibly be zero or
   slightly momentum-side. The surrogate construction (how it handles intraday vol seasonality and
   drift) is therefore load-bearing and must be specified exactly in the pre-registration.
3. Options A and B remain undecided and unauthorized. Option B stays a hard stop. The decision key
   above stands.

---

## Option 0 has now RUN — the decision key resolves to Option A

Result: [`PATHD_SPX_EXCESS_SKILL_RESCREEN_RESULTS_2026_08_04.md`](../research/PATHD_SPX_EXCESS_SKILL_RESCREEN_RESULTS_2026_08_04.md)
(pre-registration `b29ed007`, executed 2026-08-04). Verdict **`EXCESS_SKILL_CANDIDATE`** — **one survivor
of sixty members.**

**`omar_clipped_neg3_pos3`** — the position of price within the session's own realized range — survives at
all three horizons, negative sign, excess IC `+0.052 / +0.063 / +0.078` at 15/30/60 m, maxT
`p ≤ 0.0002`, 5/5 sign-stable folds, **and both independently-constructed nulls agree.** The negative
control (a held-out surrogate substituted for the real path) is clean.

**Everything else died**, including three things that had looked alive:

- `spx_vwap_gap_*` — the two nulls **disagree** (block `p 0.001–0.005`, wild `maxT p 0.21–0.99`), which is
  exactly the case the two-null rule was pre-registered to catch.
- `session_range_bps` — maxT `p 0.986–1.000`, sign-stable in 1–2 of 5 folds. Its run-1 residual was
  block-bootstrap mis-specification.
- **`minute_of_session` — dead** (`maxT p 0.87`). The time-of-session hypothesis, added specifically to be
  tested directly, **is not supported.**

**Economic size (post-hoc, descriptive).** Bias-corrected bottom-minus-top decile spread is **+1.63 SPX
points** — the raw spread is +22.92, so **93% of it was the artifact**. That is ≈ **0.82 points per trade
≈ $41 on one ES contract**, against an **assumed** $17 round-trip friction. **≈2.4× friction, and the first
number in this programme on the right side of that comparison** — carrying five caveats, of which the two
that matter most are that the corrected decile profile is **not monotone at the extremes** (decile 8 beats
decile 10) and that **ES friction is assumed, not measured.**

### ⚠ The economic size is WITHDRAWN, and Option A is NOT yet recommended

**Correction, same day.** The `$41/trade` figure above is **not causal** — within-session decile cuts are
computed from the whole session (the 10th-percentile cut ranges from −0.941 to +0.852 across 40 sessions).
Redone on **fixed absolute omar thresholds** that a trader can actually act on, three defensible
estimators give **+0.82 / +4.54 / +3.57 points per trade** ($41 / $227 / $178). **A number that moves 10×
under a change of aggregation is not a number to spend money against.**

**And the pooled table shows why.** At fixed omar the *real* forward move is **flat across every bin**
(+0.73 to −0.97, no trend). The *surrogate* runs monotonically −5.43 → +4.51. **The entire corrected signal
is the surrogate's slope**, which is a defect in the wild bootstrap: it imposes the session drift at every
minute, so "high omar" implies "still drifting up" in the surrogate but not in the real path. **Drift
handling — the exact thing flagged as load-bearing in the authorization above — is mis-specified for this
statistic.**

**Recommendation: do NOT proceed to Option A yet.** Option A measures the *denominator*. Measuring friction
precisely buys nothing while the *numerator* is unidentified and its central estimate is manufactured by a
mis-specified null. Nothing has been spent, and nothing needs to be.

**What the §1 result still supports:** the association is real *within* sessions and survives both nulls at
5/5 folds. What it does not support is a fixed-threshold rule. The open question is whether a **causal**
normalization exists that captures the within-session association — realized-volatility scaling, or a
causal running estimate of the session's eventual range. That is a new pre-registered study on **owned
data at zero cost**, and it is the natural next step ahead of any purchase.

**Still requiring owner authorization if ever taken:** Option A is a **paid-data** download and is not
authorized by the Option 0 sign-off. Option B remains a hard stop.

---

## Option A EXECUTED by owner override — 2026-08-04

The owner authorized Option A after the recommendation above ("proceed to option A", then "proceed.
authorized"). **Executed. $1.478328 spent of a $3.00 cap.** Full result:
[`PATHD_ES_SPREAD_MEASUREMENT_RESULTS_2026_08_04.md`](../research/PATHD_ES_SPREAD_MEASUREMENT_RESULTS_2026_08_04.md).

**Verdict `ES_REMAINS_CREDIBLE_BRANCH` — the assumption held.** 1,095,818 quote states over 20 RTH
sessions, 100% coverage:

| | |
|---|---|
| Unconditional spread | **1.0397 ticks** |
| Elevated-volatility spread | **1.0734 ticks** (only 3.2% wider) |
| Per-session range | 1.0018 → 1.1534; **none above 1.25** |
| **Measured friction** | **$17.9176** vs assumed $17.00 |
| Hurdle 15 / 30 / 60m | **52.98% / 52.12% / 51.50%** vs passive 0DTE 53.9–56.6% |

The load-bearing worry is dead: the 2-tick scenario that would have closed the ES branch (hurdle 57.38%
@15m) does not occur. ES is a 1-tick market even in its noisiest quartile.

**The measurement was worth having even though I advised against the timing** — it converts a standing
assumption into a durable number, and it hands the next study a hard target:

> **Any signal must be worth ≥ 0.358 ES points per trade to clear measured friction** ($17.9176 ÷ $50).

That bar is horizon-independent in points, because friction is fixed per round trip. All three disputed
omar estimators sit **above** it — the smallest, +0.82, is 2.3× — which is exactly why identifying the true
value now matters more than it did before the spend.

**Next: identify the numerator, owned data, zero cost.** No further purchase is warranted until it
resolves. Option B remains a hard stop.

*Option A result appended by Claude Opus 5 — 2026-08-04.*

*Prepared: Claude Opus 5 — 2026-08-04. Owner decision recorded by Claude Fable 5 — 2026-08-04.
Option 0 result appended by Claude Opus 5 — 2026-08-04.*
