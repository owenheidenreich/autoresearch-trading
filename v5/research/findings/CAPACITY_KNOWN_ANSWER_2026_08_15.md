# The capacity law, measured: the backfill cannot power the 120-parameter fit

**2026-08-15.** Owner-directed follow-up to the external adversarial review of the frozen
`compact_shared_lifecycle` protocol. The review claimed the 120-parameter design exceeds its per-fit
evidence budget (linear-scaled conservative budgets ~49–107 parameters against a 96-parameter entry
phase), and separately that the 20-observations-per-parameter rule this project inherited was never
measured. The owner chose to measure rather than rule. This is the measurement.

## One sentence for the bot

If we bought the proposed 794-session quote backfill today and ran the one permitted fit, the
experiment would most likely miss a real modest edge even if one existed — the known-answer rehearsal
of that experiment fails its own recovery requirement at every training size the backfill can produce,
so the purchase no longer buys what it was requested for.

## What was done

A known-answer campaign in the G1 tradition: plant an edge whose size and location we control, run the
real machinery, and ask whether it finds what is provably there — before any real data is at stake.

- **Worlds:** synthetic sessions calibrated to the measured dependence structure (20-minute
  autocorrelation time from the effective-sample-size finding; $100 value noise from the print-artifact
  finding; $15 + spread drag from the measured cost bars). In null worlds every possible entry loses;
  in edge worlds a planted signal makes correctly-sided entries pay $40 (small) or $120 (medium) per
  standard deviation of the signal.
- **Model:** the actual `CompactSharedLifecyclePolicy` entry phase — 96 trainable parameters, exit
  head frozen, exactly as the frozen protocol orders its first training stage.
- **Provably fair test:** hand-built weights inside the same architecture reach 81–87% of the oracle,
  so the planted edge is representable, and a trained fit that misses it failed to *learn*, not to
  *represent*. The recovery bar (half the oracle's value) is attainable by construction.
- **Training law:** V1 froze a 60-epoch budget and was voided by its own evidence — recovery plateaued
  at 23% while a controlled diagnostic tripling the budget moved a 404-session fit from 0% to 46% of
  oracle. V2 trains to an outcome-blind plateau of the training loss (tolerance 1e-4, patience 20,
  cap 400 epochs). Both declarations were hashed before outcomes were read; both receipts stand.
- **Grid:** training sessions 243 / 404 / 650 / 890 × worlds null / small / medium × 40 seeded trials,
  scored on 120 held-out sessions against the generator's true expected values. 480 fits per campaign.

## The measurement (V2, convergence training)

| Training sessions | Null: entries taken | Small edge: recovery (Wilson lower) | Medium edge: recovery (Wilson lower) | Small: policy vs oracle $/min | Medium: policy vs oracle $/min |
|---:|---:|---:|---:|---:|---:|
| 243 | none, 40/40 clean | 23% (14%) | 35% (24%) | 2.89 / 16.28 | 23.62 / 78.15 |
| 404 | none, 40/40 clean | 23% (14%) | 42% (31%) | 3.05 / 16.57 | 28.66 / 78.08 |
| 650 | none, 40/40 clean | 28% (18%) | 70% (57%) | 3.79 / 16.58 | 47.03 / 78.08 |
| 890 | none, 40/40 clean | 40% (28%) | 45% (33%) | 5.31 / 16.35 | 30.78 / 78.29 |

Required by the declared rule: **80% recovery** of the small edge with clean nulls.
**Measured: no tested size qualifies — the best small-edge cell reaches 40%.**

Three facts inside the table:

1. **Null discipline is perfect.** In all 160 null trials the trained policy never enters once. The
   machinery does not hallucinate edges; when it fails, it fails silent.
2. **Recovery of a modest edge is far below requirement everywhere**, including at 890 sessions — the
   largest training prefix the proposed backfill could ever produce. The medium edge (3x larger) is
   recovered more often but still never reliably (best cell 70%, Wilson lower 57%).
3. **More data helps and does not save it.** Small-edge recovery climbs 23% → 40% across a 3.7x data
   increase. Extrapolating that trend to 80% requires a corpus far beyond the ~1,037 sessions that
   exist for daily SPXW 0DTE, which began trading in mid-2022.

## What this settles

- **The referee's direction is confirmed by measurement, and the measured reality is stricter than its
  arithmetic.** The linear-scaling argument said the 96-parameter entry phase exceeds its budget at the
  first three of five outer folds. The measurement says the design misses the recovery bar at *all*
  fold sizes, including the two the arithmetic would have allowed.
- **The row-43 full-corpus budget convention is falsified for this pipeline.** A budget that admits 120
  parameters at ~1,011 sessions predicts a fit that works; the fit demonstrably does not recover a
  small planted edge at any constituent training size. Capacity claims for this pipeline must come from
  measured recovery at the actual per-fit training size, not from any observations-per-parameter ratio
  — the 20:1 rule produced an answer wrong in the optimistic direction.
- **The pending $75-capped backfill loses its stated purpose.** It was requested to enable exactly this
  fit. Under the project's own precedent (the G1 known-answer campaign closed `UNDERPOWERED` when
  recovery failed at 8% against 80%, without reading economics), an experiment whose rehearsal fails
  recovery is not run. Buying the data would fund an experiment already known to be underpowered for
  modest edges. A future request must first present a design whose known-answer rehearsal passes.
- **A silver lining that is genuinely usable:** the failure mode is one-sided. Because nulls are
  perfectly clean, a *positive* result from this pipeline would have been meaningful — the danger was
  never a fake pass but a hollow "no edge" that closed the branch wrongly. That is the referee's
  false-negative mechanism 1, now measured rather than hypothesized.

## Limits of the measurement

- Worlds are synthetic. Dependence, noise scale and drag are calibrated to measured values, but the
  planted edge is a single clean linear signal read through one feature; a real edge spread across
  correlated features could be easier to find, and a weaker or nonstationary one harder. This
  calibrates the pipeline's learning power; it says nothing about whether a market edge exists.
- The evaluation is stateless (per-minute action scoring, no position dynamics, entry phase only). The
  24-parameter exit head and trajectory-level effective sample size remain unmeasured — the referee's
  §4.2 stands open.
- One optimizer at one learning rate, frozen by declaration. A better training law could shift the
  curves; if a future design claims so, the claim is testable in this same harness for free.
- The medium-edge non-monotonicity (70% at 650, 45% at 890) is within what 40-trial noise and
  optimization variance produce; the Wilson bounds overlap. The decision rule never depended on it.

## Evidence

- V2 receipt (binding): `v4/audit/autoresearch/capacity_known_answer_2026_08_15/receipt_v2.json`
- V1 receipt (voided as a capacity measurement, preserved):
  `v4/audit/autoresearch/capacity_known_answer_2026_08_15/receipt.json`
- Declarations: `v5/work/entry-exit-attribution/KNOWN_ANSWER_CAPACITY_DECLARATION_V1.json` (`d8cd12f1…`),
  `…_V2.json` (`2cf5c2a1…`), both hashed before outcomes.
- Implementation and tests: `v5/research/capacity_campaign.py`,
  `v5/ops/run_capacity_campaign.py`, `v5/tests/test_capacity_campaign.py` (847 tests green).
- External review that prompted this:
  `v5/work/entry-exit-attribution/external-review/chatgpt-research/8-15-26/`.
