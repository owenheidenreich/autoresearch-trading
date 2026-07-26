# Protocol101 Fair-Contract Training And Feature Recovery Plan

Generated: 2026-07-08

## Purpose

This is the current steering document for the Protocol101 fair-contract
training phase.

The objective is to build a highly profitable SPXW 0DTE long-options model
that is eligible for IBKR paper-trade validation without returning to the old
historical/live mismatch.

The project will use this certified baseline contract first:

```text
protocol101-live-v2-microstructure-masked
```

with this required model-facing transform:

```text
mask_vendor_sensitive_option_quote_greek_microstructure
```

The mask is a control, not a permanent ceiling. Greeks, IV, open interest,
volume, bid/ask, spread, and quote sizes may contain real edge. They are
currently blocked from model alpha because they showed vendor-sensitive drift,
not because the project believes they are useless.

## Current State

- Historical/live synchronization for the v2 masked contract is closed unless
  new evidence contradicts the certification packet.
- The accepted 15-month v2 training scope contains 301 pass-only sessions.
- Training must use five chronological expanding-window folds with a
  one-session embargo.
- Recorder/parity days, protected holdout sessions, failed sessions, and
  report-only sessions are not training data.
- Raw bid/ask/mid/spread/Greek/volume/OI fields remain preserved for
  tradability, affordability, labels, fills, PnL, and audit reconstruction.
- No paper-submit, broker call, paid-data download, promotion/default change,
  runtime flag edit, launchd change, or real-money path is authorized here.

## Phase 1: Masked Baseline

Run the first offline Stage-1 bounded HGB/tabular search on the v2 masked
contract as the fair control.

This phase answers:

```text
Can a useful Protocol101 trader exist using only the currently
parity-stable model-facing feature set?
```

Requirements:

- Evaluate all seven menu-v2 fixed-exit shapes.
- Train on payoff / return-on-premium, not win probability.
- Use calibrated abstention so zero-trade days are allowed.
- Select only candidates that pass the owner-approved G1-G9 gates.
- Report per-shape usage, trade frequency, drawdown, no-ruin, side/time
  exposure, concentration, skipped opportunity, feature jitter, null/canary
  comparisons, and fee/stress sensitivity.
- If entry signal is real but fixed exits cannot control drawdown, route to
  Stage-2 learned exits rather than weakening G4.

The highest allowed claim from this phase is:

```text
offline candidate eligible for paper-readiness validation
```

Do not claim paper readiness from offline training alone.

## Phase 2: Feature Recovery Ladder

After the masked baseline, recover feature groups one at a time. A feature
group may become model-facing alpha only after both gates pass:

1. **Parity gate:** live-recorded IBKR and historical Databento/ThetaData
   produce bounded, non-action-flipping behavior under the same feature
   contract and decision timestamps.
2. **Uplift gate:** the feature improves out-of-sample CV performance after
   fees, stress, null/canary checks, drawdown, concentration, and seed
   robustness.

Feature groups must not be reintroduced merely because masked-baseline PnL is
weak. PnL can motivate a hypothesis; it cannot waive parity.

Recovery order:

1. Stable index/context refinements.
2. Candidate geometry and moneyness features.
3. Internally computed Greeks/IV using identical historical/live code.
4. Normalized liquidity and spread features with bounded drift.
5. Volume and open-interest semantics, only if source timing is causal and
   live-reproducible.
6. Raw vendor quote microstructure only if proven stable enough for alpha;
   otherwise keep it limited to guards, fills, labels, and audit.

Each add-back requires:

- preregistered experiment entry before results are inspected;
- explicit feature group and feature hash;
- paired IBKR-vs-historical parity report;
- CV uplift report using the same G1-G9 gates;
- keep/reject decision.

## Phase 3: Paper-Readiness Path

Training success is necessary, not sufficient.

Before paper-submit, the selected candidate must pass:

1. strict serial validation under the selected feature contract;
2. replay against stored IBKR recorder days;
3. matching historical replay on the same days;
4. paired candidate/feature/score/action diff;
5. no-order IBKR shadow sessions with full decision reconstruction;
6. owner-approved promotion/paper-readiness review.

Only after those gates can guarded IBKR paper-submit be reviewed. Real-money
review remains a later, separate owner decision.

## Operating Rules

- The masked v2 baseline is the control, not the final ceiling.
- More features are not automatically better.
- Less drift is not automatically enough.
- A recovered feature must be both live-reproducible and useful out of sample.
- Raw market fields may always be used for mechanics such as tradability,
  affordability, fills, PnL, labels, and audit.
- Model-facing alpha must be governed by the certified feature contract.

