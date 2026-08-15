# Defined risk caps the tail, but the four-leg toll erases the premium

**Result:** the single preregistered five-point ATM iron fly is safe enough for the declared $10,000
account and economically unusable for an aggressive retail taker. It lost **$60.57 per session**, its
multiplicity-corrected one-sided lower bound was **-$64.66**, and **0/5** chronological blocks were
positive.

[Declaration](../../work/entry-exit-attribution/DEFINED_RISK_IRON_FLY_DECLARATION_V1.json) ·
[receipt](../../../v4/audit/autoresearch/defined_risk_iron_fly_2026_08_14_attempt001/receipt.json)

## Frozen question

Before reading outcomes, the project selected one member rather than searching a spread grid:

- enter one direction-neutral ATM iron fly at 15:00;
- short the ATM call and put at their bids and buy exact five-point wings at their asks;
- close at the first complete four-leg aggressive touch at or after 15:15 whose debit does not exceed the
  expiry width, otherwise use validated PM cash settlement;
- charge four measured $3.08 round trips;
- abstain unless entry-defined maximum loss is at most $500, or 5% of $10,000; and
- retain the 32 previously inspected two-sided cells plus this structure as a 33-member multiplicity
  family. Success required a corrected lower bound above zero and at least four of five positive blocks.

The declaration self-hash is
`e9ef53caf5ddaa63176b4ae446edcfa40d73cbc292b15d2bb0824768575bbae9`.

## Result

All 243 sessions had a complete, admissible structure. Of the exits, 239 used an executable four-leg
touch and four used validated cash settlement.

| Measure | Result |
|---|---:|
| Mean gross midpoint P&L/session | **+$1.19** |
| Mean executable net P&L/session | **-$60.57** |
| Corrected one-sided lower bound/session | **-$64.66** |
| Median executable net trade | **-$62.32** |
| Winning sessions | **0 / 243** |
| Positive chronological blocks | **0 / 5** |
| Worst realised trade | **-$127.32** |
| Largest entry-defined maximum loss | **$227.32** |

The receipt semantic hash is
`4f5bb310db7994e28a10979bd139cd3f359a0271e54a9742ca7de2e7632fbcf5`; its artifact hash and semantic
self-hash were independently reverified after the run.

## What it means

The mechanism is plain. Midpoint marks show only **+$1.19** of fifteen-minute variance-premium capture.
Crossing all four legs at entry and exit moves that to approximately **-$48.25 before fees**; the four
round-trip fees then produce **-$60.57**. Defined risk solves the naked short's catastrophic-tail problem,
but this tight structure adds too many spread crossings to harvest a very small premium.

The same-center naked short-straddle control was also negative at **-$35.83/session** under the declared
touch law. The receipt's reported `long_straddle` control is **void and must not be interpreted**: the
post-run audit found that it was constructed as the algebraic negative of the short control, which does
not substitute the long trade's actual ask entry and bid exit. This secondary-control defect does not
touch iron-fly selection, fills, P&L, inference, or the naked-short calculation, and the primary already
fails every required criterion.

## Boundary

This closes the exact 15:00, five-point, 15-minute direction-neutral iron fly. The declaration forbids a
nearby width, time or horizon retry after outcomes. It does not prove every possible spread loses, but it
removes the strongest $10,000-compatible translation supported by the existing late-session variance
evidence.

The highest-information remaining project action is still the frozen 48-parameter Q(WAIT)-versus-
Q(ENTER) fit. Its target, features, null, chronology and output path are complete, but the owner-controlled
fit gate permits only the earlier magnitude label and architecture family.
