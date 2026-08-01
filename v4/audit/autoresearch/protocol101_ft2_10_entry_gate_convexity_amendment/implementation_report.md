# FT2-10 Entry-Gate Convexity Amendment — Codex Review And Implementation

Date: 2026-07-31  
Outcome: **IMPLEMENTED_AND_DELTA_REVIEW_PASS**  
Stop: **STOP_FOR_CLAUDE_VERIFICATION**

## Authority

- Verified starting authority SHA-256 before amendment: `edcbee06ebfc5ac3a26fa13da043754589ba55fbbd11b207906e19459d4103f3`.
- Amended consolidated authority SHA-256: `82d9573e120d6395825aa8a5f2d66fdac9bf32d825190737876b204dd112e2f2`.
- Full Trader graph SHA-256 remained unchanged: `9955085a31840da63057761a620a5ec2995e04f05ff2aa5f4906afd795726a08`.

## Phase A — Independent Verdict

The convexity diagnosis is correct. Across 120,229 action-eligible rows in the
clean 45-session development tensor, realized fee-adjusted MFE-return q10 was
negative at every registered horizon, while the corresponding arithmetic means
were positive. Profit-area-return q10 was exactly zero at every horizon. The old
strict all-four q10-greater-than-zero gate was therefore structurally
unsatisfiable on this slice.

The endorsed replacement is a conservative lower confidence bound on the
conditional arithmetic mean, not an individual-outcome lower prediction
quantile. The implementation fits fixed-weight squared-error support heads,
takes the median forecast across seeds 101/102/103, computes residuals on the
disjoint calibration role, averages residuals within session, gives each of the
20 sessions equal weight, and adds the empirical 0.10 quantile from 2,000
deterministic session-cluster bootstrap replicates. All causally available
horizons, including h3 and h5, remain in the gate. The fee-adjusted threshold is
strictly zero. Calibrated q10 remains the unchanged rank statistic.

## Phase B — Governed Reseal

Only the positive-after-fee entry-gate statistic changed. The RLAC targets,
labels, 20/1/20/4 roles, fees, simulator v5, D48/D49, exit design, guardrails,
action constraints, and graph topology remained unchanged. The amended
authority, FT2-10 specs, chained receipts, and checker pins were resealed. The
cross-contract checker added the regression guard
`entry_gate_uses_satisfiable_central_tendency_statistic`.

Verification:

- Tensor-contract validator: 22/22 passed.
- Cross-contract consistency checker: 32/32 passed, including the new central-tendency rule.
- Full Trader graph hash: unchanged.

## Phase C — Faithful Capped Option-D Stage 1

The A6 Stage-0 preregistration froze one HGB configuration, seeds 101/102/103,
the existing 39 primary heads, 28 gate-support conditional-mean heads, all
seven horizons, and the 2,700-second model-fit cap before fitting.

Run result on the clean 45-session development slice:

- Model-fit wall-clock: 1,361.684 seconds (within the 2,700-second cap).
- Component models: 309 total (216 signed-path models, 84 expected-gate support models, 9 action models).
- Physically eligible replay contract rows: 4,191.
- Raw expected-upside four-axis pass rows: 3,545 across 621 decisions.
- Calibrated lower-confidence expected-upside pass rows: 2,975 across 552 decisions.
- Raw q10 four-axis pass rows: 0.
- Calibrated q10 four-axis pass rows: 0.
- Headline decisions: 1,432 WAIT, 0 BUY.
- Serial trades: 0 across 0 sessions.
- Premium-band counts: none.
- Four-bucket trade distribution: none.

The entry amendment is empirically satisfiable, but the unchanged downstream
stack over-abstains on this replay. All 552 expected-upside proposer decisions
first failed the sequential uncertainty-margin check. Report-only individual
counterfactual diagnostics also found 0/552 passing the unchanged q10-MFE error
margin and 0/552 passing the q90-regret bound. The action-conditioned gate was
unavailable because it observed 19 realized WAIT outcomes versus the frozen
minimum of 50. No downstream threshold was changed.

## Final Audit

- Focused unit tests: 7/7 passed.
- Tensor-contract validator: 22/22 passed.
- Cross-contract consistency checker: 32/32 passed.
- Stage-1 delta-scoped review: 68/68 passed.
- Stage 2, Stage 3, and Stage 4: not started.
- Broker contact, paper submit, paid download, protected-resource access,
  runtime/default/promotion mutation: none.

The bounded amendment and faithful Stage-1 rerun are complete. Further work is
intentionally stopped for independent Claude verification.

`STOP_FOR_CLAUDE_VERIFICATION`
