# DEVELOPMENT RESULT — the entry training law failed a synthetic harness; the capacity claim is withdrawn

**2026-08-15, corrected the same day after the second-round external review.** The first version of
this finding was titled "the backfill cannot power the 120-parameter fit." That conclusion was an
overclaim and is withdrawn. What the campaigns actually established is narrower:

> **The simplified, badly-conditioned entry training implementation used by campaigns V1 and V2
> failed to learn one dense synthetic linear signal reliably at 243–890 training sessions. This
> blocks that training law. It does not establish that the proposed backfill cannot power the
> architecture, the complete lifecycle pipeline, or the long-option strategy class.**

The operational decision is unchanged and correctly grounded either way: **no purchase and no fit**,
because the proposed experiment has not passed a faithful known-answer preflight. The branch is
**blocked, not closed**.

## Withdrawn claims

The following statements from the first version are withdrawn as unsupported by what was measured:

1. *"The backfill cannot power the 120-parameter fit"* — V2 tested a simplified entry-only harness,
   not the frozen pipeline, under a training law now shown to be defective (see below).
2. *"Every training size the backfill can produce was tested"* — the grid (243/404/650/890) omitted
   the actual fold prefixes 526, 648 and 769 and the upper-corpus endpoints (539–913 at 1,037
   sessions).
3. *"The failure mode is one-sided"* and *"a positive result would have been meaningful"* — null
   "cleanliness" was an entry-rate threshold in a world where abstention is trivially correct by
   construction, not a full-gate false-pass rate; 40 trials cannot certify a ≤5% rate even at zero
   observed failures (one-sided 95% Wilson upper bound ≈ 6.3%).
4. The extrapolation that 80% recovery requires more sessions than exist.
5. *"The 20-observations-per-parameter rule is falsified"* — the campaign showed the ratio
   insufficient to authorize this particular training law, not false in general.
6. V2's declaration claim of running V1's *identical seeds* — the seed derivation used Python's
   process-salted `hash()`, so V1 and V2 in fact ran different, unreproducible seed banks. Verified
   empirically: the same expression yields different values across processes.

## Defects found in the V1/V2 harness itself (second-round review, verified here)

- **Conditioning:** raw-dollar MSE through unscaled features (asks $100–900, moneyness −2..−32) into
  a 3-unit tanh state, single Adam setting — while the production trainer
  (`v5/ops/train_causal_day_action_value.py`) standardises and clips features, scales targets by
  $1,000, uses Smooth-L1, AdamW with weight decay and gradient clipping. Poor recovery can be an
  optimization artifact, and V1→V2 already demonstrated the result's sensitivity to the training law.
- **Checkpointing:** the V2 early stop returned the *terminal* model after 20 stale epochs rather
  than restoring the best-loss checkpoint, biasing recovery down.
- **Reproducibility:** `hash()`-derived seeds; no per-trial records; the runner never verified the
  declaration it claimed to run under.
- **Objective mismatch:** WAIT trained toward a constant zero rather than the protocol's derived
  value of preserving the slot; evaluation was stateless per-minute scoring with no occupancy,
  serial account, controls or session-level inference — so "recovery" was a representation-learning
  diagnostic, not power for the real gate, and the planted "small/medium" effects were never tied to
  the smallest effect the real gate must accept.
- **Calibration:** the $100 value-noise was taken from the print-artifact residual ($102.60) — the
  very contamination the quote purchase exists to remove — and applied as independent noise, unlike
  real option outcomes' strong cross-contract and within-session covariance.
- **Evidence publication:** the binding receipts landed in a git-ignored audit directory and were
  absent from the pushed commit that cited them. Fixed: the receipt directory now carries an explicit
  `.gitignore` exception and the receipts are tracked.

## What the V2 data still says (as a development diagnostic)

Under the defective-but-declared V2 law, with a planted edge hand-verified to be representable by
the exact architecture (reference weights reach 81–87% of oracle):

| Training sessions | Null: entries taken | Small edge recovered | Medium edge recovered |
|---:|---|---:|---:|
| 243 | none in 40 trials | 23% | 35% |
| 404 | none in 40 trials | 23% | 42% |
| 650 | none in 40 trials | 28% | 70% |
| 890 | none in 40 trials | 40% | 45% |

These numbers bind only that training law on that synthetic task. The 650→890 medium-edge drop
(28/40 → 18/40, two-sided Fisher p ≈ 0.04 uncorrected) is itself evidence of training instability
rather than a clean sample-size curve.

## What remains true and decision-relevant

- The original per-fit evidence-budget mismatch stands: the frozen chronology trains its first fit
  on a ~404-session prefix while the 122-parameter budget was projected from the full corpus, and
  the entry phase alone is 96 parameters. That mismatch — plus the exposure-contaminated score
  blocks, the unresolved serial-risk conflict, the unmeasured exit-phase evidence and the other
  protocol defects — is what blocks the experiment.
- No valid end-to-end known-answer preflight of the frozen pipeline exists. Until one passes, the
  backfill purchase buys an experiment with no demonstrated ability to answer its question, so the
  Tier-1 request stays refused on preflight grounds, not on a claimed impossibility.
- V1 and V2 are both classified **development campaigns**: their worlds and results have now
  influenced the harness design, so no future confirmation campaign may reuse their seed banks.

## V3 (in flight at time of writing)

A corrected development diagnostic mirrors the production training law (standardised clipped
features, targets scaled by $1,000, Smooth-L1, AdamW, gradient clipping, best-checkpoint
restoration, SHA-256 process-stable seeds, per-trial records, declaration verified by the runner)
on the actual fold-prefix grid 243/404/526/648/769/890. Its question is deliberately narrow: **was
the V2 learning failure a conditioning artifact of my training law, or does it persist under the
production law?** Either answer is developmental; neither can authorize a fit or purchase.

## Evidence

- V2 receipt (development): `v4/audit/autoresearch/capacity_known_answer_2026_08_15/receipt_v2.json`
- V1 receipt (development, voided training law): `…/receipt.json`
- V3 receipt (development, production-mirrored law): `…/receipt_v3.json` when complete
- Declarations: `v5/work/entry-exit-attribution/KNOWN_ANSWER_CAPACITY_DECLARATION_V{1,2,3}.json`
- Harness: `v5/research/capacity_campaign.py`, runner `v5/ops/run_capacity_campaign.py`, tests
  `v5/tests/test_capacity_campaign.py`
- Reviews: `v5/work/entry-exit-attribution/external-review/chatgpt-research/8-15-26/`
