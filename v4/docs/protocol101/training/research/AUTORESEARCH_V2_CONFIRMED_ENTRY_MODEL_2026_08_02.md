# Autoresearch v2 confirmed entry model

Date: 2026-08-02

Status: `INVALID_EXPERIMENT` (clock-contract violation); former
`CONFIRMED_EDGE` artifact quarantined, not promoted, and not runtime-enabled

> Clock audit update (2026-08-03): the immutable model **failed** the offline
> Databento-live-OPRA decision-parity gate. More importantly, all 445,063 fitted
> rows violated the hypothesis' declared `completed_minute_plus_60s` context
> clock. The training path used the ThetaData SPX bar stamped at decision minute
> `t`; that bar represents `[t,t+60s)` and its close is available only at
> `t+60s`. The lawful SPX bar for the OPRA minute ending `t` is stamped `t-60s`.
> Therefore the old confirmation is a diagnostic result for a crossed-time
> simulation, not valid evidence for the preregistered causal experiment. The
> raw-source replay reproduced 100% only under that invalid clock and 10.34% of
> complete fixed-block outcomes under the lawful clock.

## Recorded outcome before clock invalidation

Autoresearch v2 produced and independently confirmed a fully trained entry
model. The winning policy is `signed18_model_side_nearest`:

- fit one HGB regressor on the actual frozen 18-column legacy `signed17` tuple;
- target the frozen ask-to-bid stop-50/target-100/hold-25-minute policy;
- in each of six fixed ET blocks, act on the first decision whose highest model
  score is greater than zero;
- use the side of that highest-scoring candidate;
- buy the deterministic nearest-ATM eligible contract on that side;
- use the unchanged simulator-v5 account, fee, affordability, occupancy, daily
  loss, and cutoff rules.

Learned contract ranking was removed because it did not beat time-matched
nearest ATM. Added microstructure/time features were removed because the
signed-only policy was at least as strong. Unconditional or learned 5/15-minute
delay policies failed.

## Development selection

The distinct corrected-v3.2 foundation contains 214 development sessions and
has SHA-256
`361c19a064e84b81a25db2ecc6b7d9dc37fad4b2b07396dedcaf328a5f681bfc`.
Five expanding chronological OOF folds evaluated 170 outer sessions.

Against the deterministic fixed-block/momentum-side/nearest-ATM control, the
winning policy's strict one-account delta was +$542.15/session (95% CI +$379.51
to +$704.78), positive in 5/5 folds. Against the shuffled-target signal
control, it was +$372.54/session (95% CI +$198.32 to +$546.75), also positive
in 5/5 folds. All family screens used shared session-blocked maxT correction.

The development candidate made $82,436 across 998 strict-serial trades. The
policy then emitted `PROVISIONAL_EDGE` under its frozen $100/session practical
bar and $250/session MDE80 bar.

## Model freeze

The final HGB model was fit once on 445,063 finite development rows across all
214 sessions before confirmation access.

- Model SHA-256:
  `c5d0115ba187ae8f443ac57ea813fa74cc689c03bbf546cb72d8262bc65f463e`
- Pre-holdout freeze SHA-256:
  `ff97f5a9a5ae6e3dd68489539f30d33693a23d83a11c8f61429a61c73376f954`
- Confirmation preregistration SHA-256:
  `a579adf4ca868cdbce5c544fe25495d372eaf6136618066637e562d3c43b1aa7`

The 29-session primary protected comparison had estimated 85.8% power against
zero at one-sided alpha 0.05. Exactly one candidate and one primary control
were frozen before access. No rescue candidate, threshold, feature set, exit,
or secondary winner was permitted.

## One-shot confirmation

The protected set was opened exactly once and is now spent. All 29 session
materializations preserved the corrected-v3.2 legacy fields and independently
verified their two-clock receipts.

| Confirmation result | Value |
|---|---:|
| Sessions | 29 |
| Mean paired strict-serial delta | +$540.79/session |
| 95% CI | +$90.15 to +$991.44 |
| One-sided sign-flip p | 0.0092 |
| Positive-delta sessions | 21/29 |
| Candidate PnL | +$25,559 |
| Candidate trades | 167 |
| Deterministic-control PnL | +$9,876 |
| Deterministic-control trades | 148 |

Every preregistered confirmation check passed, yielding `CONFIRMED_EDGE`.

## Claim boundary and next gate

This no longer proves an offline causal entry-policy edge. It records what the
policy did in the crossed-time simulator and is useful only for debugging and
hypothesis history. It does not authorize live shadow, paper-default, runtime,
broker, promotion, or real-money work.

The protected sessions may never be reused as confirmation data. They roll
into development history. Runtime-parity packaging for this exact immutable
model was attempted on 2026-08-03 and failed because its SPX context clock is
one minute ahead of causal live availability. No adapter-only repair is lawful:
the next research candidate would have to be a distinct generation trained and
validated on the completed-minute live-twin clock. Any such model or threshold
change cannot cite this confirmation as its own and would need a new independent
confirmation epoch; the spent holdout cannot be reopened.

## Evidence

- Engine: `v4/research/autoresearch_v2/`
- Corrected-v3.2 foundation:
  `v4/research/autoresearch_v2/foundations/development_2025-08-01_2026-06-09_corrected_v3_two_clock.json`
- Provisional selection packet:
  `v4/audit/autoresearch/autoresearch_v2_corrected_v3_signal_policy_2026_08_02_attempt001/`
- Confirmation preregistration:
  `v4/audit/autoresearch/autoresearch_v2_entry_model_confirmation_preregistration_2026_08_02.json`
- Frozen model and one-shot confirmation:
  `v4/audit/autoresearch/autoresearch_v2_entry_model_confirmation_2026_08_02_attempt001/`
- Runtime decision-parity failure packet:
  `v4/audit/autoresearch/autoresearch_v2_frozen_entry_runtime_decision_parity_2026_08_03_attempt003/`
- No-order Databento Live OPRA comparison and replacement clock:
  `v4/audit/autoresearch/databento_live_opra_training_twin_2026_08_03/comparison_attempt002/`

The clock audit accessed no broker, paper runtime, order path, default registry,
promotion state, or real-money path. A separate owner-authorized, bounded
no-order Databento Live OPRA sample was captured on 2026-08-03 to define the
replacement historical/live twin; it loaded no model and opened no holdout.
