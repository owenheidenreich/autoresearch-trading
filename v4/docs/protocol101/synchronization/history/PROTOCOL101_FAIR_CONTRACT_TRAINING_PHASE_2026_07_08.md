# Protocol101 Fair-Contract Training Phase

Generated: 2026-07-08

## Decision

Status: `protocol101_fair_contract_training_phase_defined`

Certified training/live contract:

```text
protocol101-live-v2-microstructure-masked
```

Current steering plan:

```text
v4/docs/PROTOCOL101_FAIR_CONTRACT_TRAINING_AND_FEATURE_RECOVERY_PLAN_2026_07_08.md
```

This phase authorizes offline research implementation for Protocol101
fair-contract training. It does not authorize paper-submit, promotion/default
changes, paid data downloads, broker calls, launch scheduling changes, runtime
flag edits, or real-money trading.

## Training Objective

Train a research-only SPXW 0DTE long-option entry model that:

- uses only live-reproducible information available at decision time;
- scores model-facing features under
  `mask_vendor_sensitive_option_quote_greek_microstructure`;
- preserves raw bid/ask/mid only for tradability, affordability, labels, fills,
  PnL, and audit reconstruction;
- plays the same one-account, one-position, ask-entry, bid-exit, flat-by-close
  game historically and through future IBKR no-order parity;
- beats `PAPER_DEFAULT_PROTOCOL101` only under strict one-account serial replay,
  never under overlapping diagnostic PnL.

Primary metric:

```text
plain fee-adjusted strict-serial net PnL after all hard gates pass
```

Risk adjustment lives in the hard binary gates: drawdown, no-ruin, worst-seed,
frequency, concentration, stress replay, and parity. Do not fit or tune a
weighted utility to select a winner. If any stressed or penalized scalar is ever
used as a tiebreaker, its stress amount and weights must be preregistered and
frozen before results, exactly like fees and gate thresholds.

## Trader Style

The intended trader is a restrained directional liquidity taker:

- long calls or long puts only;
- one contract, one open position, no overlapping headline trades;
- selective entries rather than high-frequency churn;
- morning/post-open and late-session opportunities allowed only when the model
  can show stable expectancy under the same live-reproducible contract;
- raw option microstructure may block bad trades but may not be used as brittle
  model alpha;
- the microstructure mask is the fair baseline control, not the final feature
  ceiling;
- Greeks, IV, OI, volume, bid/ask, spread, and quote sizes may become
  model-facing alpha only through the preregistered parity-plus-uplift feature
  recovery ladder;
- lifecycle remains frozen initially unless a separate hypothesis packet
  authorizes exit-model changes.

## Hill-Climb Strategy

Use a staged, registry-backed autoresearch loop:

1. Build a v2 preflight and manifest from canonical processed sessions.
2. Build a chronological purged split design from that manifest.
3. Run guarded dry-run plans before any training attempt.
4. Train small preregistered attempt batches only with explicit owner-approved
   training flags.
5. Score every attempt through candidate validation, selected-candidate export,
   strict replay, and feature-jitter gates.
6. Append both failures and successes to the experiment registry.
7. Promote no artifact from the loop. A pass only permits deeper research
   replay and no-order parity.

Do not weaken the mask inside the first Stage-1 baseline run because PnL is
disappointing. If the baseline is weak, write the result cleanly and then use
the feature recovery ladder to test whether additional feature groups are both
live-reproducible and useful out of sample.

Default first search family:

- histogram-gradient-boosted tabular candidate rankers first;
- all seven menu-v2 fixed-exit shape labels;
- return-on-premium/payoff target, not win probability;
- threshold selection by validation stressed PnL only inside the frozen hard
  gate framework;
- max three trades per day as the conservative first rail;
- no MLP/neural family and no learned regime classifier without a separate
  evidence note and owner approval.

## Validation Gates

Minimum gates before research freeze:

- feature contract is exactly `protocol101-live-v2-microstructure-masked`;
- model-facing transform is
  `mask_vendor_sensitive_option_quote_greek_microstructure`;
- manifest loading is required and glob loading is disabled;
- no protected holdout sessions are used for training or threshold selection;
- no recorder, paper, or parity-confirmation sessions are used for training or
  threshold selection;
- v2 nulls and canaries are recomputed on the full accepted v2 corpus before
  any uplift claim;
- any feature add-back is preregistered, parity-tested on paired
  IBKR-vs-historical evidence, and then measured for out-of-sample uplift
  before it becomes model-facing alpha;
- chronological expanding-window validation uses five folds with a one-session
  embargo, or training stops for an owner-approved split revision;
- learned candidates beat the fixed heuristic baseline on the same folds;
- fold-average trade frequency stays between 0.3 and 6.0 trades/day, while
  naturally allowing zero-trade days;
- max drawdown <=25% passes, >25% and <=35% requires owner review, and >35%
  hard-fails;
- no ruin or cash exhaustion;
- validation and diagnostic strict replay are positive after stress;
- at least three seeds are run; the worst seed must be positive, avoid ruin,
  keep drawdown <=35%, and stay in the frequency band;
- one fresh never-used confirmation seed clears core gates after selection;
- profit factor, drawdown, worst-day, concentration, side/time exposure, churn,
  and skipped-opportunity reports are produced;
- selected candidate export reconstructs every decision, score, feature hash,
  contract, entry, exit, block, and PnL input;
- feature-jitter gates show stable action presence and selected-contract
  behavior;
- no broker endpoint is called and `paper_submit_allowed` remains false.

Additional gates before paper-trade readiness:

- candidate vs `PAPER_DEFAULT_PROTOCOL101` under the same strict serial account
  game;
- untouched chronological holdout pass without tuning;
- no-order IBKR recorder/runtime parity with latency, freshness, and block
  reasons logged;
- owner-approved promotion packet. This phase does not create that packet.

## Stopping Criteria

Stop the loop and write a failure packet when any of these occurs:

- a candidate uses non-live-reproducible or future/path/exit-label features;
- the v2 feature contract or scoring transform is violated;
- validation improves only by lowering frequency below the trade-count floor;
- diagnostic or holdout PnL/PF/drawdown fails after a validation pass;
- feature-jitter or paired replay shows action/contract instability when a trade
  is possible;
- top trade/day/month concentration explains the result;
- results suggest tuning on diagnostic, holdout, recorder, or paper evidence;
- a desired next step would require paid data, broker access, paper-submit,
  promotion, default changes, runtime flag edits, or real-money trading.

## Current Implementation Scope

Allowed mutations:

- research docs;
- fair-contract preflight/design/runner/model-search scripts;
- focused tests;
- new audit outputs under `v4/audit/autoresearch/`.

Forbidden in this phase:

- changes to `v4/promotion/PAPER_TRADING_DEFAULT.json`;
- changes to `v4/runtime/` flags or launchd assets;
- broker connections or order endpoint calls;
- paid-data downloads;
- paper-submit or real-money configuration.
