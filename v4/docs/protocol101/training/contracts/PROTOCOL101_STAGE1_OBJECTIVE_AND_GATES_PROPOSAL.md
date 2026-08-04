# Protocol101 Stage-1 Objective & Gates — APPROVED

Status: **APPROVED by owner, 2026-07-07, as proposed (all gates, all
checklist defaults ratified).** These numbers are now the preregistered law
of stage-1 hill climbing. Changing any gate requires an explicit owner-signed
revision of this document — the autoresearch loop may never adjust its own
goalposts.

G8 was explicitly revised and owner-signed on 2026-07-26. The binding
replacement is recorded in
`PROTOCOL101_G8_CALIBRATION_REVISION_2026_07_26.md` and in the G8 row below.

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

- Fee overlay: $3.00 per contract round trip — grounded 2026-07-19:
  IBKR fixed $0.65 + CBOE SPXW proprietary customer fee (~$0.70/side) +
  regulatory (~$0.05-0.10/side) = ~$2.80-3.00 all-in round trip.
  Sensitivity reruns at $2.60 and $4.00. Final truth-up from actual
  paper-fill commission records once they exist.
  **TRUTH-UP COMPLETE 2026-08-04: measured $1.54/side = $3.08 round trip** on a
  guarded DU paper round trip (position avgCost 81.54028 on a 0.80 fill;
  RealizedPnL -3.08 on a price-flat round trip). The $2.80-3.00 estimate above
  was correct. The $3.00 overlay slightly UNDERcharges, by $0.08. Note that the
  IBKR $0.65 line item is one of three components and is NOT the all-in cost --
  treating it as such is the error corrected across the research docs on
  2026-08-04. Evidence: v4/audit/autoresearch/
  pathd_phase0b_trackc_paper_transitions_2026_08_04/trackc_transition_evidence.json

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

## Gates

Stage-1 eligibility requires every hard gate G1-G7. G8 is a mandatory
report-only diagnostic under the signed 2026-07-26 revision. After independent
cross-hypothesis selection, the selected candidate must earn G9.

| Gate | PROPOSED threshold |
|------|--------------------|
| G1 Profitability | Fee-adjusted net PnL > 0 on >= 4 of 5 CV folds, and pooled PnL > 0 |
| G2 Beats no-skill | Pooled top-selection PnL z-score >= 3.0 vs matched random-selection null (band pinned by the canary artifact) |
| G3 Beats heuristics | Pooled fee-adjusted PnL > the fixed Pickle-heuristic baseline (VWAP-side rule + best single shape) on the same folds |
| G4 Drawdown | **v2, owner-signed 2026-07-19 (OWEN HEIDENREICH):** (a) pooled fee-adjusted net PnL / pooled max strict-serial drawdown >= 1.0, AND (b) strict-serial equity never below $5,000 on any fold. Null-relative drawdown is report-only. Calibration: forced-oracle DD = $0 at any mandated cadence and no-skill DD floor $4.7k-$38k/fold proved no absolute or equity-relative cap can separate skill from noise (see `PROTOCOL101_G4_HOLDOUT_REVISION_2026_07_19.md`, `protocol101_stage1_g4_forced_oracle_feasibility`). Supersedes the 2026-07-07 25%-of-peak rule, which was measured infeasible (0/42 ever passed). |
| G5 Seed robustness | >= 3 seeds; WORST seed satisfies G1 and G2 at z >= 2.0 (mean-only results are inadmissible) |
| G6 Era guard | No era with systematically negative test folds (median era test-fold PnL < 0 across folds => status `regime_bound_requires_owner_review`, implemented as code, not prose) |
| G7 Frequency band | 0.3 - 6.0 trades/day averaged per fold (the owner's day-trader band; outside it = different product, requires review) |
| G8 Calibration Diagnostic | **v2, owner-signed 2026-07-26 (OWEN HEIDENREICH):** Out-of-fold ECE remains required and reported for every seed using the frozen 10-bin payoff-score-to-realized-win readout. ECE <= 0.10 remains the diagnostic benchmark, not an eligibility threshold. While calibrated confidence does not control abstention, contract selection, sizing, routing, exits, or another trading behavior, G8 is report-only and cannot pass/fail Stage-1, G9, or holdout eligibility. The HGB target remains payoff / return-on-premium, never win probability. Before confidence controls behavior, a separate action-conditioned calibration gate must be preregistered, tested, independently audited, and owner-signed. See `PROTOCOL101_G8_CALIBRATION_REVISION_2026_07_26.md`. |
| G9 Confirmation run | After a candidate is selected, one fresh never-before-used seed must independently satisfy G1/G2/G4 (guards against seed-shopping) |

## Holdout protocol (the final exam)

One shot per promoted candidate, via owner override token, test role only:
fee-adjusted PnL > 0; drawdown judged by the same owner-signed G4 rule
active for this candidate generation (v2: Calmar >= 1.0 and $5,000 equity
floor), under identical fee/stress and one-account semantics; and result
within the 90% bootstrap CI implied by CV (a holdout wildly ABOVE
expectations is also a red flag and triggers audit, not celebration). A
failed holdout burns the candidate; there is no second attempt without a
new candidate hash and owner sign-off. (Stale $1,500 absolute cap removed
2026-07-19, owner-signed, with the G4 v2 revision.)

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

- [x] Fee overlay $3.00 grounded against IBKR/CBOE schedules 2026-07-19.
- [x] G4 drawdown rule: superseded by signed G4 v2 on 2026-07-19
      (pooled Calmar >= 1.0 plus the $5,000 per-fold equity floor).
- [x] G7 frequency band 0.3-6.0 trades/day.
- [x] G8 revised globally on 2026-07-26: mandatory report-only diagnostic
      until calibrated confidence controls trading behavior.
- [x] Holdout "burns on failure" rule.
- [x] No additional stricter gate requested at the 2026-07-07 approval.
