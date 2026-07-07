# Protocol101 Stage-1 Objective & Gates — PROPOSAL (awaiting owner sign-off)

Status: DRAFT for owner review. These numbers become the preregistered law of
stage-1 hill climbing the moment the owner approves them. After approval,
changing any gate requires an explicit owner-signed revision of this document
— the autoresearch loop may never adjust its own goalposts.

## The trader being built (owner's directive, 2026-07-06)

A disciplined intraday SPXW 0DTE long-options day trader: adaptive frequency
(0 trades some days, several on others), adaptive hold (minutes to hours via
the trade-shape menu), asymmetric taste (convexity welcome, scalping not the
identity), risk scaled to calibrated confidence, single-contract plays.

## Objective (what "better" means)

Maximize **fee-adjusted net PnL through the strict one-account serial
simulator** (v2 semantics: $10k starting cash, affordability enforced,
single contract, forced flat 15:55 ET), subject to the constraints below.
Win rate is a diagnostic, never an objective.

- Fee overlay: PROPOSED $3.00 per contract round trip, with mandatory
  sensitivity reruns at $2.00 and $5.00. (Exact IBKR all-in figure to be
  pinned after external verification; the overlay constant is recorded in
  every experiment artifact.)

## Data law

- Training/CV universe: accepted (`pass`) non-holdout sessions only, loaded
  exclusively through the governed loader. Currently Oct 2024 - May 15 2025;
  Jul - Dec 2025 joins automatically once its v3.3+ acceptance lands.
- Validation scheme: chronological expanding-window cross-validation,
  5 folds, 1-session embargo between train end and test start. No shuffling
  across sessions, ever.
- Report-only sessions (early close, crisis low-liquidity, vendor context
  gap) are excluded by the loader. Protected holdout (2025-05-16 ->
  2025-06-30) is locked; opened once per promoted candidate, test role only.

## Gates (a candidate is "promotion-worthy" only if ALL hold)

| Gate | PROPOSED threshold |
|------|--------------------|
| G1 Profitability | Fee-adjusted net PnL > 0 on >= 4 of 5 CV folds, and pooled PnL > 0 |
| G2 Beats no-skill | Pooled top-selection PnL z-score >= 3.0 vs matched random-selection null (band pinned by the canary artifact) |
| G3 Beats heuristics | Pooled fee-adjusted PnL > the fixed Pickle-heuristic baseline (VWAP-side rule + best single shape) on the same folds |
| G4 Drawdown | Max simulator drawdown <= $1,500 (15% of starting cash) on every fold |
| G5 Seed robustness | >= 3 seeds; WORST seed satisfies G1 and G2 at z >= 2.0 (mean-only results are inadmissible) |
| G6 Era guard | No era with systematically negative test folds (median era test-fold PnL < 0 across folds => status `regime_bound_requires_owner_review`, implemented as code, not prose) |
| G7 Frequency band | 0.3 - 6.0 trades/day averaged per fold (the owner's day-trader band; outside it = different product, requires review) |
| G8 Calibration | Predicted-probability reliability: expected calibration error <= 0.10 on pooled CV predictions (confidence must mean something before it sizes anything) |
| G9 Confirmation run | After a candidate is selected, one fresh never-before-used seed must independently satisfy G1/G2/G4 (guards against seed-shopping) |

## Holdout protocol (the final exam)

One shot per promoted candidate, via owner override token, test role only:
fee-adjusted PnL > 0, max DD <= $1,500, and result within the 90% bootstrap
CI implied by CV (a holdout wildly ABOVE expectations is also a red flag and
triggers audit, not celebration). A failed holdout burns the candidate;
there is no second attempt without a new candidate hash and owner sign-off.

## The autoresearch loop (how hill climbing runs)

Hypothesis -> preregistered config (hashed) -> run through governed loader ->
metrics vs gates -> ledger entry -> next hypothesis. Standing rules:
- Every experiment gets a ledger entry BEFORE results are inspected.
- Nulls rerun whenever the feature contract or label policies change.
- No threshold/feature tuned on test folds; fold boundaries are frozen here.
- Model family ladder: heuristics -> gradient-boosted trees -> sequence
  models only if learning curves show data/representation, not capacity, is
  binding. GPU spend requires that evidence first.
- Stage-2 (learned exits) and stage-3 (sequential agent) each require a new
  owner-signed objective document before work begins.

## Honest expectations

With ~149 (soon ~270) sessions and a selective day trader, per-fold trade
counts will be small and confidence intervals wide. Gates G2/G5/G9 exist so
that "profitable-looking" and "statistically real" are never confused. It is
a legitimate and reportable outcome of stage-1 that NO candidate clears the
gates; that result would redirect effort (features, data, stage-2) rather
than soften the gates.

## Sign-off checklist for the owner

- [ ] Fee overlay placeholder ($3.00) acceptable pending verification?
- [ ] G4 drawdown cap: $1,500 on a $10k account — confirm or adjust.
- [ ] G7 frequency band 0.3-6.0 trades/day — confirm or adjust.
- [ ] Holdout "burns on failure" rule — confirm.
- [ ] Any gate you want stricter? (Looser requires justification in writing.)
