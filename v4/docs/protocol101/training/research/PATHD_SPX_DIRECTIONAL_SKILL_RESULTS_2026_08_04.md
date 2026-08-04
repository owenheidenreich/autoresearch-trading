# Path-D SPX Directional-Skill Screen — RESULTS (2026-08-04)

**Status: frozen verdict `DIRECTIONAL_SKILL_CANDIDATE` — but the result is `NOT_INTERPRETABLE` as
skill. ~79% of the headline effect is reproduced by paths with no predictability whatsoever.**

Pre-registration: [`PATHD_SPX_DIRECTIONAL_SKILL_PREREGISTRATION_2026_08_04.md`](PATHD_SPX_DIRECTIONAL_SKILL_PREREGISTRATION_2026_08_04.md)
(frozen and committed as `9ce78f72` before the run; not edited).
Evidence: `v4/audit/autoresearch/pathd_spx_directional_skill_screen_2026_08_04/`.

---

## 1. What the frozen screen returned

213 of 215 development sessions (2 dropped: no fresh SPX bar at the final boundary), 5,751 session-IC
observations, 54-member family, 20,000 session-blocked permutations.

**All 27 members cleared the decision rule at their preferred sign**, 5/5 sign-stable folds, session-shuffled
control null on every member. Top of the table:

| Member | mean IC | se | maxT p | folds |
|---|---|---|---|---|
| `omar_clipped_neg3_pos3` 60m, negative sign | **+0.522** | 0.016 | 0.0000 | 5/5 |
| `spx_vwap_gap_points` 60m, negative | +0.479 | 0.017 | 0.0000 | 5/5 |
| `omar_clipped_neg3_pos3` 30m, negative | +0.419 | 0.014 | 0.0000 | 5/5 |
| `spx_vwap_gap_over_session_range` 60m, negative | +0.364 | 0.018 | 0.0000 | 5/5 |
| `momentum_15m_bps` 60m, negative | +0.130 | 0.012 | 0.0000 | 5/5 |

**The deviation declared in §2.1 of the pre-registration was load-bearing.** Every one of these is at the
**negative** sign. Under the plan outline's 27-member one-sided family they would all have scored
`maxT p ≈ 1.0000` and the screen would have reported `NO_DIRECTIONAL_SKILL`. The reported table shows both
signs, so this is directly checkable: read the bottom half.

## 2. Why the result is not skill

**A rank IC of 0.52 against SPX forward returns is not a plausible edge.** It would be among the strongest
publicly-measured signals on the most heavily arbitraged index in the world, found in nine public
trend/mean-reversion primitives. Per [`feedback_no_reward_hacks`], a number that large is a bug until
proven otherwise. So it was tested rather than reported.

**The mechanism.** Every top-ranked feature is a function of the price *level* `P(t)` — position within the
session range, gap to session VWAP. The target is `y = P(t+h) − P(t)`, which contains `−P(t)`. That shared
term forces a negative rank correlation for **any bounded path**, with or without predictability, and the
effect grows with horizon exactly as observed (0.32 at 15m → 0.52 at 60m).

**The pre-registered controls could not detect this, and that is a defect in my pre-registration.**
Session-shuffling `y` destroys the pairing between feature and target — but the artifact *lives in* that
pairing structure. Shuffling therefore returns a correct null while the artifact goes unmeasured. The
control set was sound against look-ahead and against multiplicity; it was blind to a shared-term artifact.

**The correct null is a matched surrogate** — a path carrying the session's own realized volatility and no
predictability. `v4/research/pathd_spx_screen_surrogate_diagnostic.py` (post-hoc, clearly labelled)
builds 4,260 per cell by circular block bootstrap (30-minute blocks) of each session's own one-minute log
returns, replayed from its true open.

| Feature (60m) | measured | surrogate | excess | **explained by artifact** |
|---|---|---|---|---|
| `omar_clipped_neg3_pos3` | −0.522 | −0.410 | −0.112 | **78.5%** |
| `spx_vwap_gap_points` | −0.479 | −0.408 | −0.071 | **85.2%** |
| `spx_vwap_gap_over_session_range` | −0.364 | −0.326 | −0.037 | **89.7%** |
| `momentum_15m_bps` | −0.130 | −0.155 | **+0.024** | **118.6%** |
| `momentum_5m_bps` | −0.086 | −0.102 | **+0.016** | **118.0%** |
| `session_range_bps` | +0.127 | +0.070 | +0.057 | 55.1% |

## 3. What actually survives

**Momentum carries nothing.** At every horizon the surrogate is *more* mean-reverting than the real data
(explained fraction 100–119%, excess `z` +0.05 to +2.04 in the **trend** direction). SPX 5- and 15-minute
momentum mean-reverts *less* than chance. Whatever the entry model's eleven momentum/VWAP variants were
contributing, mean reversion was not it.

**A residual survives in the level-based features**, at `z` −2.0 to −6.7 against this null — largest for
`omar` at 60m (−0.112). **Treat this as suggestive, not established.** The block-bootstrap surrogate
preserves volatility clustering but flattens **intraday volatility seasonality**: real sessions are U-shaped
(loud open, quiet midday, loud close) and the surrogate is not. Since the artifact's magnitude depends on
the path's volatility profile, some unknown part of the residual is surrogate mis-specification rather than
skill. It has not been tested with a family-corrected null.

**`session_range_bps` is the least artifact-explained member (55%)** — and it is a running max-minus-min of
the session prefix, i.e. **monotone in time of session**. It is a time-of-day proxy, which is the same
variable the [Fable brief](FABLE_REVIEW_BRIEF_2026_08_04.md) identified as the only positive signal in the
option data and the one the entry model structurally could not represent. That coincidence is worth noting
and is not worth believing yet: 213 sessions is 213 observations of one period's intraday drift shape.

## 4. Bottom line

**The screen did not answer its question.** It was built to separate World A (features carry direction, the
0DTE wrapper destroys it) from World B (features carry nothing). It returned a positive that is ~79%
mechanical, so it establishes neither.

What it does establish, firmly:

1. **Momentum features have no mean-reversion content** — they are less mean-reverting than chance. That
   closes a large part of the feature contract as a directional source.
2. **A level/mean-reversion residual exists** and is the only thing left standing. It is roughly a fifth of
   the headline and has not been converted into money at any friction on any instrument.
3. **An IC is not an edge.** Nothing here has been costed.

**Recommended next step, presented not decided:** a matched-surrogate re-screen — same features, same clock,
same corpus, with the null replaced by intraday-volatility-matched surrogates and the family correction
applied to the *excess* IC. Owned data, no purchase, no training, no broker; it needs only a fresh
pre-registration. That converts an uninterpretable result into a decisive one. See
[`PATHD_NEXT_CLASS_OWNER_DECISION_2026_08_04.md`](../contracts/PATHD_NEXT_CLASS_OWNER_DECISION_2026_08_04.md).

**Nothing here reopens the 0DTE long-premium class**, which is closed structurally and independently of
this result.

## 5. Evidence and reproduction

```bash
PYTHONPATH=. python -m v4.research.pathd_spx_directional_skill_screen        # refuses a second run
PYTHONPATH=. python -m v4.research.pathd_spx_screen_surrogate_diagnostic     # post-hoc
```

- `receipt.json` — self-sealed; pop `receipt_sha256`, recompute `stable_hash`, must match.
  Carries `protected_holdout_opened: false`, `paper_order_submitted: false`, `model_trained: false`.
- `results.csv` (54 rows), `results.md`, `surrogate_diagnostic.json`.

**One bug was caught in build, by an assertion rather than by luck.** The corpus stores `event_time` as
`datetime64[us]`; an unqualified `.astype("int64")` yields **microseconds**, a 1000× clock error. The
screen asserts at every boundary that its forward-target availability rule reproduces the frozen kernel's
`spx_close` exactly, and that assertion failed on session 1 of the first run. The fix is pinned at
[`pathd_spx_directional_skill_screen.py:118`](../../../../research/pathd_spx_directional_skill_screen.py#L118).

*Signed: Claude Opus 5 — 2026-08-04.*
