# Frozen candidate protocol — shared SPXW 0DTE lifecycle after quote backfill

**Status: PRE-ACQUISITION DESIGN. NO FIT OR ECONOMICS AUTHORIZED.**

This packet prevents the same-day CBBO backfill from becoming permission for an architecture, threshold,
seed or label search. It defines one candidate pipeline for the one-long-option game.

## Trading game

- Start each session with $10,000.
- Trade only one long SPXW 0DTE call or put at a time.
- Flat actions are `WAIT` or buy one eligible contract at the ask.
- Held actions are `HOLD` or sell the same contract at the bid.
- Maximum two entries per session, maximum 120 minutes per position, forced session-close accounting under
  the existing validated settlement law.
- Ticket plus measured fee and worst observed selected loss must each stay at or below $500.
- SPX may be causal context. No other product, expiry, short option or second leg exists.

## One architecture

Use only `compact_shared_lifecycle` from
`v5/research/causal_day_compact_shared_lifecycle.py`. Its parameter count must be computed from the built
module and remain at or below the post-QC conservative evidence budget. The current design receipt records
120 parameters against a worst projected budget of 122. Any architecture change is a new family and
requires a new declaration before outcomes.

## Chronology

After acquisition and structural QC, sort all complete sessions chronologically:

1. the earliest `floor(40% × n)` sessions are the initial training prefix;
2. split the remaining sessions into five contiguous score blocks with deterministic `array_split`;
3. for each outer score block, use only strictly earlier sessions for scaling, target construction and
   fitting; and
4. never use a scored block to set a cutoff, calibration, early-stop rule or action rate.

Within every outer training prefix, generate exit-training trajectories from nested chronological
out-of-fold entries. The entry policy that creates a trajectory may not have trained on that trajectory's
session. The outer score block stays untouched by both entry and exit fitting.

## Entry and exit objectives

Entry retains the executable 120-minute `ENTER`-versus-`WAIT` dollar-value law: paying the ask and
measured fee must beat preserving the slot for a later action. There is no absolute threshold; enter only
when predicted `ENTER` strictly exceeds the feasible `WAIT` value, whose structural floor is $0.

Exit is trained only after nested out-of-fold entries exist. At every held state:

- `SELL` is the value delivered by requesting exit now under the first-later-bid/settlement law; and
- `HOLD` is the best strictly later executable sale value before the 120-minute/session-close boundary.

The shared representation and entry parameters are frozen before the exit head is trained. Exit training
therefore cannot move the entries it is supposed to manage. At inference, `SELL` fires when its predicted
value is at least `HOLD`; forced liquidation remains simulator-owned.

## Controls and outcome firewall

- Fit an identical shuffled-label path. Entry executable values are permuted within training sessions and
  `WAIT` is recomputed. Exit `(SELL,HOLD)` target pairs are permuted within training trajectories.
- Match an outcome-blind control on scored session, regime, opportunity minute, side, delta, premium and
  trade count.
- Evaluate combined entry/exit midpoint-to-midpoint gross first. If its mean is not positive, stop without
  opening bid/ask economics or selecting an operating point.
- If midpoint passes, charge ask-in, bid-out and measured fees. Require a multiplicity-corrected one-sided
  lower bound above zero absolutely and versus both controls, plus positive signs in at least four of five
  chronological folds for all three comparisons.
- Audit every feature timestamp, still-forming candles, contract identity/survivorship, whole-ladder
  equality, settlement, account state, origin routing and selected-ticket risk.

## Failure consequence

Any failed kill closes this post-backfill member. No adjacent seed, hidden width, trade cap, hold horizon,
time window, side, threshold or architecture follows. A fit still requires the acquired manifest, a fresh
self-hashed declaration, computed count, attainable selector proof and the applicable owner-controlled
gate to return `PERMITTED`.
