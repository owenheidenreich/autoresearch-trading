# Path-D — Owner Decision: what class do we research next?

Status: **DECISION REQUESTED — options presented, not decided**

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

*Prepared: Claude Opus 5 — 2026-08-04.*
