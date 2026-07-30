# Protocol101 FT2-08 Data/Tensor/Label Contract v3

Status: `producer_repaired`

Node: `FT2-08-DATA-TENSOR-LABEL-CONTRACT`

Product contract:
`2363d3f986daba20bd5087ed751dc5b2d839e76cd6413aeca0bcd255eb98857a`

## 1. Purpose

This contract defines the historical and live-constructible representation for:

```text
flat: WAIT or BUY one exact eligible call/put/strike from 42 slots
open: HOLD or EXIT the exact open contract
```

It is a repaired design packet. It does not build real tensors, fit a model,
select a policy, access outer evidence or the protected holdout, or modify the
runtime. The prior packets are preserved under `superseded/`, including the
direct repair parent at `superseded/v2_pre_final_repair_20260729/`.

FT2-20 rerun001 found residual causal mask, fee-path soft-close,
session-start-equity, and terminal-minute defects. This v3 packet repairs those
findings and is synchronized by exact hash with the final FT2-10/11 packets.

## 2. Authority And Repaired Inputs

The governing owner amendment freezes:

- a BUY decision from completed minute `t` filling at `t+1` ask;
- an EXIT decision from completed minute `v` filling at `v+1` bid;
- descriptive path marks beginning after the entry fill;
- no-bid path minutes at the full-loss bound;
- exact 15:55 forced flat without prior-bid substitution;
- D49 soft-close when remaining budget cannot fund a `$1` premium plus fees;
- every fold embargo session excluded from the census.

FT2-04 v2 continues to freeze the path-label definitions and 45-session role
firewall. FT2-05 v3 recomputes the opportunity census after the intent/recheck
eligibility change and supplies a v2-v3 impact note. This packet pins that final
v3 receipt before its own receipt is issued.

## 3. One Market Clock

All decisions use America/New_York completed-minute timestamps.

At flat decision minute `t`:

1. Option state is the completed state stamped `t`.
2. Index/context state is available through `t-1 minute`.
3. Entry masks screen the last known executable ask `A_t`, the policy's causal
   integer-cent ledger, and no information from `t+1`.
4. The model emits WAIT or one source-neutral contract identity.
5. A legal BUY commitment reserves the only position slot.
6. At `t+1`, only that identity is rechecked for identity/freshness, the `$1`
   premium floor, affordability, D48, and D49 using the actual ask `A_(t+1)`.
7. A passing recheck fills at `A_(t+1)`. A failure opens no position, charges
   no premium or fee, realizes no PnL, records the rejection, and permits no
   substitute contract.
8. Descriptive future marks start at `t+2` only after a successful fill.

At open decision minute `v<=15:54`, EXIT fills only from `v+1` bid. A 15:54
EXIT fills once at 15:55 and suppresses a duplicate forced-flat event. The
15:55 row is a non-model `FORCED_FLAT_TERMINAL` event with no learned action
and no t+1 execution. Otherwise the still-open option is valued once at the
exact 15:55 state; no bid means zero option value.

The last entry decision is 15:29 and fills at 15:30. From 15:30 through 15:55
the system is manage/exit-only.

Current quote age may not exceed 90 seconds. New entries require 15 consecutive
complete synchronized minutes after the open or any reconnect/delayed-feed
event. Until then, the only flat action is WAIT. History capacity remains 90
minutes and missing early history is masked, never backfilled from yesterday.

The exact causal entry law is
`intent_fill_recheck_law.json`
(`5c117d716cea3c986605faf7b58d510eedce3264a0c04f9368f6dc509dea6bd0`);
the clock/field law is machine-readable in `field_semantics_manifest.json`.

## 4. Exact Contract Identity

The source-neutral key includes:

```text
SPX underlying
SPXW trading class
same-day expiry
strike in milli-points
call/put right
100 multiplier
USD currency
```

Historical vendor symbols and IBKR `conId`/contract details are aliases to that
key. Alias records are audit data, never identity substitutes. Incorrect
trading class, expiry, strike increment, right, multiplier, currency, or an
ambiguous alias fails closed.

The canonical ATM strike uses the existing Python round-half-to-even five-point
rule. The ladder is 21 strikes from ATM-50 through ATM+50, each with call then
put, for 42 slots. Slots are coordinates, not identities. Recentring may move a
contract to another slot but cannot transfer another contract's history.

`identity_mapping_spec.json` freezes construction, aliasing, ATM rounding,
recentring, and fixtures.

## 5. Repaired 17-Feature Axis

The old v1 tensor incorrectly placed all 17 signed features on a global market
axis. Eight are contract-dependent and must follow the exact call/put/strike.

The v2 market history is `[90,11]`:

- nine contract-invariant signed context features;
- minutes since open;
- minutes to close.

The v2 contract history is `[90,42,10]`:

- offset and moneyness geometry;
- three right-dependent alignment features;
- three near-ATM D-family surface features;
- exact-slot internal delta and gamma.

Together, the alpha channels remain the signed 17. No new alpha is silently
admitted. The geometry and clock channels are the already required structural
inputs. Raw vendor Greeks are prohibited.

The D-family fields use canonical `$0.05` quantized mids. Internal Greeks use
the exact slot mid, spot, strike, right, time to same-day 16:00 ET, `r=0.05`,
and `q=0`. Every formula, unit, source timestamp, denominator rule, and
synthetic expected value is frozen in:

- `field_semantics_manifest.json`;
- `synthetic_golden_vectors.json`.

Historical and live builders must independently reproduce those vectors.

## 6. Price-Path Extension

The three direct-price extension channels remain inactive. Activation requires
all of:

1. FT2-25 producer admission pass.
2. FT2-26 independent acceptance.
3. An actually owner-signed synchronization amendment.
4. The candidate bundle pinning all three receipts/hashes.

An “owner-signable” draft is insufficient. Without the signed amendment, every
extension channel stays unavailable and ignored.

## 7. Complete-Ladder Safety

Flat-state BUY actions require:

- all 42 source-neutral definitions;
- all 42 fresh, complete current quote states;
- synchronized fresh context;
- 15-minute complete-history streak;
- exact identity/history joins;
- tradability and affordability;
- D48;
- D49;
- no soft-close or daily-stop state.

If one required ladder slot is missing or stale, all 42 BUY actions are false
and WAIT remains true. The model may not rank a partial historical ladder when
live would abstain.

An open position is different: its dedicated identity-keyed state remains
available even if the flat-entry ladder is incomplete or recentered away from
the position. A fresh executable EXIT is not blocked merely because an
unrelated ladder slot is absent.

## 8. Dedicated Open-Contract State

Open-state rows begin at 15:30 in the current late-session scaffold and
continue through 15:55. Later lifecycle machinery may extend the same contract
through all entry phases.

Every open row carries:

- exact source-neutral identity;
- entry decision and fill clocks;
- entry ask and fee reserve;
- dedicated 90-minute open-contract history;
- current bid/ask/source/receipt timestamps and freshness;
- current moneyness and internally recomputed Greeks;
- account ledger audit state.

This record is independent of the recentered 42-slot flat-entry ladder. A
contract can move outside +/-50 points and still be held or exited under the
same historical/live representation.

## 9. Account And Safety Ledger

All monetary state is integer cents.

Offline session-start equity is the policy simulator's fully reconciled cash
cents carried to the session's frozen 09:30 boundary; the first governed
session starts at `1,000,000` cents. Live paper equity is the uniquely selected
latest authenticated NetLiquidation source update from `[09:29:00,09:30:00]`
ET, received by 09:30:05, for a dedicated Protocol101 paper account with no
unexplained external state. Missing, stale, late, ambiguous, or contaminated
state forces WAIT/no new entry for the session; it is never guessed.

D48:

`A_t premium + active fee reserve <= 5% of session-start equity`

D49:

`realized session loss + A_t premium + active fee reserve <= daily budget`

At `t+1`, the actual selected-contract cost is independently rechecked against
affordability, D48, D49, and the `$1` premium floor. The active fee path computes
its own minimum floor cost: `$103` for the `$3` path and `$104` for the `$4`
path. At each flat decision, the lane permanently soft-closes when the current
exact ladder has no otherwise-eligible contract of at least `$1` whose causal
intent cost fits. A shared hard-coded threshold is forbidden. Candidate and
comparator policies maintain separate causal ledgers after divergence.

Restart recovery must reproduce the exact open position, fills, commissions,
realized loss, masks, soft-close, and daily-stop state. Any disagreement blocks
new entries.

See `account_state_ledger_spec.json`.

## 10. Replay Authority v5.1 Design

Simulator v5 source is not changed by this node. A v5.1 implementation must add
independent enforcement of:

- `t+1` fills and actual occupancy;
- causal `A_t` intent masks plus selected-contract `A_(t+1)` rechecks;
- rejected-fill accounting with no position, premium, fee, or PnL;
- D48 and D49 from ledger primitives at both stages;
- ladder-dependent `$1 + active fee` soft close (`$103`/`$104`);
- complete-ladder WAIT;
- dedicated open-contract state;
- exact 15:55 no-bid forced flat.

The `$3` and `$4` fee paths must be separate causal replays because a fee change
can alter masks, cash, and later entries. Post-hoc metrics-only fee subtraction
is forbidden.

FT2-30 must implement and test this authority; FT2-31 must independently accept
it. This design packet does not modify Simulator v5.

See `replay_authority_v5_1_spec.json`.

## 11. Labels And Future Firewall

Labels join only after the causal feature tensor is complete and the selected
entry has passed its t+1 recheck, using exact decision and source-neutral
contract identity. A rejected intent remains an explicit failed-fill population
event with no path label and zero realized policy PnL.

No-bid states are full-loss states in primary labels. A separately named
no-bid-excluded diagnostic is mandatory, and eventual candidate ranking may not
depend on that missingness convention.

The runtime loader and historical composer cannot read:

- future bids or paths;
- path-property targets;
- target-valid masks;
- label-presence masks;
- future quote availability.

Removing every label partition must leave every historical action bit-identical.
Horizon availability at runtime comes only from causal clock and model-output
state.

See `label_join_spec.json`.

## 12. Roles And Storage

The five expanding folds are adopted by immutable reference. Training may fit;
the one-session embargo may do nothing; validation is outer evidence only.

The design census is exactly 45 sessions and intersects:

- outer-test union: 0;
- protected holdout: 0;
- embargo union: 0.

The protected holdout remains closed.

Storage is identity-keyed and session-partitioned. It persists one copy of each
contract-minute path and assembles trailing windows at batch time. Atomic
checkpoints require source, builder, schema, semantics, identity, and logical
content hashes. A dedicated open-contract partition prevents recentering loss.

See `fold_roles.json` and `storage_spec.json`.

## 13. Historical/IBKR Transfer

The signed scoped synchronization decision covers the current 17-feature
boundary; it does not automatically certify new extensions or all repaired
tensor mechanics.

Before a candidate can train or shadow with an active field set, a frozen paired
source battery must cover:

- every active numeric field;
- source-neutral identities and slot maps;
- complete-ladder and per-slot masks;
- opening warmup and reconnect reset;
- `t+1` fill availability;
- open-contract state outside the ladder;
- model scores and final actions.

Every active extension also needs the owner-signed amendment. The detailed
acceptance statistics remain assigned to FT2-10/11 and FT2-92; this packet
freezes the required population and fields.

## 14. Coordinated Repair Boundary

FT2-10 consumes this exact intent-law hash and owns the composer, comparator,
CDF, regret, neutral-game, and substitute-cluster repairs. FT2-11 consumes the
same hash and owns comparator identity, studentization, graph topology,
Phase-F, and MNAR procedure repairs. The aggregate repair packet's
`findings_crosswalk.json` maps every rerun001 finding to final evidence; no
finding remains partially answered.

## 15. Acceptance Boundary

This producer packet is not independently accepted. The final re-review may
begin only after:

- the repaired FT2-05 v3 receipt is pinned;
- every machine-readable file parses;
- every synthetic vector is independently reproduced;
- all rerun001 findings are cross-walked with verdict `ANSWERED`;
- the cross-contract checker passes and its output is pinned.

## Highest Allowed Claim

> FT2-08, with its FT2-04 and FT2-05 v3 dependencies, is producer-repaired
> under the causal intent/fill-recheck, active-fee soft-close, deterministic
> session-equity, and terminal-minute laws. No implementation, model, or
> training result is accepted.
