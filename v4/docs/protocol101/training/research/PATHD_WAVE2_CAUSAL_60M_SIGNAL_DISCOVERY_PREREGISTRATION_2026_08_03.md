# Path-D Wave 2 — Causal 60-Minute Signal Discovery

**Status:** `FROZEN_BEFORE_EXECUTION`

## Objective and terminal vocabulary

Determine whether one of exactly three new, causal, non-fitted feature scores can select profitable
long SPXW 0DTE opportunities at a 60-minute horizon on the existing five-fold OOF development
population. No estimator is fit. The terminal verdict is exactly one of:

- `TRAINING_GATE_EARNED`
- `NO_SIGNAL`
- `INVALID`
- `BLOCKED_CLOCK_OR_DATA_CONTRACT`

A pass earns only a separately reviewed model-training proposal. It is not deployable-edge, paper, or
promotion evidence.

## Frozen population and trading game

- Read the five `fold_spec.test` lists in `/Volumes/AR_TRADING_DATA/artifacts/entry_v2/fold=*/manifest.json`.
  Their union is the only evaluated population and must match the sessions in `oof_scores.parquet`.
- Every evaluated session must be in the first 215 development sessions returned by
  `pathd_phase1_entry.development_sessions`. The spent final 36-session firewall is never decoded.
- Instrument: long one SPXW 0DTE call or put; one position; one contract.
- Account: $10,000 initial equity carried chronologically; affordability enforced; each session starts
  with a 5% realized-loss stop; every position is flat by 15:55 ET.
- A decision is eligible for the primary gate only when a fill at the end of the 60-second entry window
  can still receive a complete 60-minute hold before 15:55 ET.
- ES and VIX are causal context only and are never traded.
- Acceptance horizon: 60 minutes only. A 30-minute replay is diagnostic and cannot pass the gate.

## Causal clocks

- Candidate decision emission is `feature_boundary_ns + 2,336 ms`.
- Option CBBO-1m values stamped at or before the feature boundary may be used.
- SPX, VIX, ES, and option OHLCV minute bars are stamped at interval start and become available only at
  `interval_start + 60 seconds + 2,336 ms`.
- No score cell may be finite unless every source cell it uses is available by decision emission.
- At least 30 prior completed in-session minutes are required for every robust normalization.

## Frozen feature family — exactly three maxT members

Every raw component is converted to a robust z-score, separately by option right, using the median and
IQR of the preceding 60 in-session minute-level component medians, excluding the current minute. At
least 30 finite prior minutes are required; zero IQR is unavailable. Z-scores are clipped to
`[-3,+3]`. A member score is the
unweighted arithmetic mean of all its finite-required components; missing any component makes that row
unscorable. No sign, weight, feature, horizon, or threshold may change after this freeze.

### W2-H01 — Full-ladder surface continuation

Equal weights across:

1. exact-contract 15-minute midpoint momentum in basis points;
2. same-right breadth, centered as `2*fraction(momentum15 > 0)-1`;
3. 15-minute change in the causal near-ATM straddle-mid/spot measure; and
4. 15-minute change in the causal side-aligned near-ATM smile slope.

### W2-H02 — Option microstructure continuation

Equal weights across:

1. microprice pressure divided by half-spread;
2. displayed size imbalance `(bid_size-ask_size)/(bid_size+ask_size)`;
3. exact-contract 5-minute midpoint momentum;
4. exact-contract 15-minute midpoint momentum;
5. log completed-minute option volume, whose trailing robust normalization is the fixed volume-surprise
   transform; and
6. five-minute relative-spread compression, `relative_spread[t-5]-relative_spread[t]`.

### W2-H03 — Cross-market alignment

Let `side=+1` for calls and `-1` for puts. Equal weights across:

1. `side * SPX 15-minute close return`;
2. `side * ES 15-minute close return`;
3. `side * (ES 5-minute return - SPX 5-minute return)`; and
4. `-side * VIX 15-minute close return`.

All returns are basis points from completed bars. The ES–SPX residual has no fitted beta.
Sessions where the ES continuous-contract instrument ID changes, plus the immediately adjacent OOF
sessions, are unavailable for W2-H03 so a roll jump cannot masquerade as context alpha. An empty owned
ES minute partition is also unavailable for W2-H03 and does not block the two option-only members.

## Entry, execution, and selection

- A member may attempt entry only at score `>= +1.0`.
- Simultaneous candidates are ordered by: highest score, narrowest decision-time relative spread,
  lowest decision-time midpoint premium, then OSI symbol.
- Post a buy at the latest actionable bid at arrival. Fill at that limit only if the observed ask is at
  least one valid option tick below the limit within 60 seconds. Use actual fill time.
- Charge $1.50 at entry and $1.50 at exit. Nonfills contribute $0.
- Exit at the actionable bid after exactly 60 minutes. A filled row with no bid observation within two
  seconds of the exact 60-minute target is unavailable to the primary gate; this also removes early-close
  rows whose frozen 15:55 nominal terminal outlives the actual quote session. The separate diagnostic
  exits after 30 minutes.

## Comparators, controls, and inference

Strict serial replay is run independently for:

1. not trading;
2. passive all-time 60-minute holds; and
3. passive 10:30–10:59 ET 60-minute holds.

Constant comparator ties use relative spread, premium, and OSI. Each member is compared with the
highest-total-net comparator, with not-trading supplying the zero floor.

The three primary paired session-delta vectors form one 20,000-replicate session-blocked sign-flip maxT
family with seed 2031. Required controls per member are:

1. sign-reversed score;
2. score reassigned from a different session within outer-fold, 30-minute time, right, and exact
   moneyness strata by the frozen SHA-derived cyclic session permutation; and
3. constant `+1.0` score.

A control is accepted only if it clears the economic, fold, LCB, sample-size, concentration, big-loss,
and causal/identity gates without maxT or decile credit. Any accepted control makes the family
`INVALID`.

## Acceptance

`TRAINING_GATE_EARNED` requires at least one primary member to satisfy every item:

1. total net PnL is above the best comparator and above $0;
2. paired delta is positive in at least four of five outer folds;
3. the one-sided 95% whole-session bootstrap LCB of mean paired delta is positive;
4. family-adjusted one-sided maxT `p <= 0.05`;
5. at least 100 filled serial trades across at least 30 sessions and all five folds;
6. its top score decile is the best decile and beats its bottom decile;
7. all controls fail;
8. no session or weekday supplies more than 50% of positive paired delta;
9. filled-trade return-on-premium big-loss share (`return < -25%`) is at most 2%; and
10. causal-clock, future-mutation, identity, and deterministic-reproduction checks pass.

If multiple members pass, select highest mean session net PnL, then lower maximum drawdown, then the
declared order above. A pass drafts, but does not execute, a bounded model-training goal. Otherwise return
`NO_SIGNAL`. Do not expand the family, budget, or thresholds. A source/clock/identity inability returns
`BLOCKED_CLOCK_OR_DATA_CONTRACT` rather than an economic verdict.

## Required outputs

- immutable pre-registration freeze receipt;
- feature-lineage and availability-clock manifest;
- scored candidate and replay datasets;
- per-member and comparator results;
- family maxT packet;
- negative-control packet;
- side-effect audit;
- human-readable report; and
- only on a pass, a draft model-training goal.

Append `NO_SIGNAL` or `INVALID` to the canonical do-not-retest ledger and living roadmap. Do not update
the ledger for a blocked data/clock run.

## Hard stops

This goal authorizes only offline feature construction and fixed-score feasibility analysis using
already-owned data. It authorizes no learned estimator, hyperparameter search, paid data, broker
connection, live capture, paper submission, runtime or launch change, promotion, default replacement,
or real-money action.

STOP_FOR_CLAUDE_VERIFICATION
