# Matched Execution-Aware Feasibility Gate — PRE-REGISTRATION (2026-08-03)

**Frozen before any result is looked at.** This document fixes the hypotheses, the budget, the fill models,
the multiplicity correction, and the pass/fail gate **in advance**. Nothing below may be edited after
execution begins; amendments must be added as dated, signed appendices that leave the original text intact.

**This is a feasibility gate, not a training campaign. NO MODEL IS FIT.** Every rule is pre-stated and
parameter-free-by-construction. That is deliberate: the entry model has already been shown to have zero
ranking power on these features, and the holdout is spent, so fitting again would only manufacture a
false positive. If simple pre-stated rules cannot clear the friction hurdle, a complex model almost
certainly cannot either — and if one appears to, that is a reason for suspicion, not celebration.

## Why this gate exists

Codex's adversarial review (`321b3bbd`) refuted the claim that ES is uniquely reachable and withdrew the
"mathematically impossible" figure. The corrected position: **neither ES nor passive 0DTE is uniquely
preferred, and both need a matched comparison under executable fill assumptions before either is trained.**

The hurdle framework only establishes a *necessary* condition — how much directional accuracy is required.
It says nothing about whether that accuracy is *achievable*. This gate measures the achievable side.

## The one live lead

Codex found, and Claude independently re-verified, that the **10:30–10:59 ET** half-hour has **+$5.99 mean
gross** over 10,063 candidates across 120 sessions, positive in **4/5 folds** (+9.8/+7.6/−17.3/+30.8/+6.4).
Net is still −$20.08 under the aggressive fill.

That matters because of the arithmetic: gross **+$5.99** against friction of **$26.07**. Under the honest
one-tick-penetration passive saving (**+$12.57**) net is **≈ −$7.51** — still negative. Only under the
*optimistic* offer-touch saving (**+$22.76**) does it reach **≈ +$2.68**.

**So the single most promising configuration in the entire corpus is marginally positive only under the
most optimistic fill assumption available.** That is precisely the kind of finding that must be
pre-registered and tested honestly rather than mined further. It is hypothesis **H1** below — it is
**not** a licence to search adjacent windows.

## Frozen hypotheses — the complete budget is 9 primary tests

No hypothesis may be added, split, or re-specified after execution begins. Every test below counts toward
the family-wise correction **including any abandoned mid-run**.

| ID | Branch | Hypothesis | Horizons |
|---|---|---|---|
| **H1** | 0DTE | Long premium entered **10:30–10:59 ET**, passive entry, has net EV > 0 | 30m, 60m |
| **H2** | 0DTE | Long premium entered **any time**, passive entry, has net EV > 0 (tests whether the horizon effect converts into economics, not just hurdle) | 30m, 60m |
| **H3** | ES | Pre-stated **momentum** rule: sign of prior-`h` return, held `h` | 15m, 30m, 60m |
| **H4** | ES | Pre-stated **reversion** rule: opposite sign of prior-`h` return, held `h` | 15m, 30m (60m dropped — H3/H4 at 60m are mirror images and would double-count) |

H3 and H4 are exact mirrors by construction; they are counted as separate tests because either could be
the survivor, and reporting only the winner would be selection.

## Fill models — the honest ladder is authoritative

The **one-tick-penetration** model is the **primary** specification for every 0DTE test. The optimistic
offer-touch model may be reported as an upper bound but **cannot** satisfy the gate on its own.

| Model | Passive saving | Role |
|---|---|---|
| Offer touch (`ask ≤ L`) | +$22.76 | upper bound only — reported, never decisive |
| Touch + one-tick continuation | +$20.11 | secondary |
| **Offer penetrates L by one tick** | **+$12.57** | **PRIMARY — the gate is judged on this** |
| Offer penetrates by two ticks | +$2.31 | stress floor |

**ES friction is unmeasured.** ES results must be reported across the **full 1–4 tick band**
($17.00 / $29.50 / $42.00 / $54.50). An ES hypothesis passes only if it clears at **≥2 ticks**, since
1 tick is the most optimistic possible assumption and cannot be verified without paid data.

## Multiplicity correction

All 9 primary tests form **one family**. Apply the existing session-blocked **maxT** machinery in
`v4/research/autoresearch_v2/` (`statistics.py`, `screens.py`). Report both raw and family-adjusted
significance. **The family-adjusted number is the one that decides the gate.**

## Negative controls — mandatory, per hypothesis

1. **Sign-reversed** — invert the rule; must NOT pass.
2. **Session-shuffled** — permute rule outputs across sessions, preserving within-session structure; must NOT pass.
3. **Constant** — always-enter; must NOT pass.

If any negative control clears the gate, the entire run is `INVALID` and the result is discarded, exactly
as in the four-box replay.

## THE GATE

A hypothesis is **`FEASIBLE`** only if ALL of the following hold:

1. Net EV > 0 under the **primary** fill model (one-tick penetration for 0DTE; ≥2-tick friction for ES);
2. Positive in **≥4 of 5** folds;
3. Survives **family-wise maxT** correction across all 9 tests;
4. **All** negative controls fail;
5. Effect is not concentrated in a single session or a single day-of-week.

Anything less is **`NOT_FEASIBLE`**. If every hypothesis returns `NOT_FEASIBLE`, that is the honest,
expected, and fully acceptable outcome — **both branches close and Path-D ends.**

**A `FEASIBLE` result is NOT an edge and NOT a licence to train.** It means only that a branch is worth a
pre-registered forward test on **fresh live paper**. The holdout is spent; there is no historical
confirmation available for anything found here.

## Prior — stated in advance so it cannot be revised afterward

**Claude expects all 9 to return `NOT_FEASIBLE`.** H1 is the only one with a plausible path, and it needs a
fill assumption more generous than the primary model to reach positive. Recording this now so a surprise
must be argued for rather than accepted.

## Hard stops

- **Protected 36-session firewall is SPENT** — never reopen. `holdout_open_count` must remain 0.
- **No model fitting.** Any parameter selected from data voids the run.
- **No paid data.** ES friction stays a band; do not purchase GLBX quotes under this pre-registration.
- **Do not modify** `FILL_LAW`, the causal t−60s clock, the label law, or the OOF firewall.
- No broker/order/live/paper-submit; no promotion; no runtime-flag/launchd/plist edits.
- **No hypothesis expansion.** If an interesting adjacent window appears, record it as a *future*
  candidate — testing it inside this run voids the multiplicity correction.

---

# Codex Execution Goal

Execute the pre-registration above **exactly as frozen**. Do not optimize, extend, or improve it — its
value is entirely in having been fixed in advance.

**Inputs (all owned, read-only):**
- `/Volumes/AR_TRADING_DATA/artifacts/entry_v2/oof_scores.parquet` (156,950 candidates)
- `/Volumes/AR_TRADING_DATA/exit_features/session=*/*.parquet` (1-second paths)
- `raw/databento/opra_spxw_cbbo_1s/` for exact-quote reconstruction where needed
- `raw/databento/glbx_es_ohlcv_1m/` for ES — **exclude roll days** (Codex's A2 finding: `ES.c.0` is a
  continuous series and rolls inject artificial jumps; identify and drop them, and report how many)

**Method requirements:**
- Report each hypothesis with: n, net EV under the primary fill model, per-fold breakdown, raw and
  maxT-adjusted significance, and every negative control.
- For ES, report the full 1–4 tick friction band, not a point estimate.
- Use **mean-payoff** economics throughout, not median proxies — the median-in-an-EV-formula error is what
  produced the withdrawn 116.2% figure.
- Sanity-check that H1's population reproduces Claude's re-verification: 10,063 candidates, 120 sessions,
  +$5.99 mean gross, 4/5 folds positive.

**Deliverable:** a results document with a per-hypothesis `FEASIBLE` / `NOT_FEASIBLE` verdict, the
multiplicity-adjusted family result, all negative controls, and a single bottom line: **do either, both, or
neither of the branches warrant a forward live-paper test?** Update the roadmap status board. End with
`STOP_FOR_CLAUDE_VERIFICATION`.

**Claude verification checklist (for the gate):** reproduce every headline independently from raw
partitions; confirm no parameter was data-selected; confirm all 9 tests are in the maxT family including
abandoned ones; confirm negative controls failed; confirm firewall closed and frozen code unmodified;
apply heightened skepticism to any `FEASIBLE` result, especially H1.

*Signed: Claude Opus 5 — 2026-08-03 — FROZEN before execution.*
