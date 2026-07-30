# FT2-08 Repair Attempt 1 - Producer Repair Complete

Terminal outcome: `producer_repaired`

Highest allowed claim:

> FT2-08 (with its FT2-04/05 dependencies) is repaired against the FT2-20 findings under the owner's t+1 fill convention; FT2-10 repair is unblocked.

This is a producer result. It is not independent acceptance, model readiness,
training authorization, paper readiness, or promotion.

## Step 0 History

The original sub-minute fill proposal failed honestly: all 56 inspected
sessions retained one quote snapshot per contract/minute, so a first quote at
+5s or +15s could not be reconstructed. The owner subsequently authorized the
conservative next-completed-minute fill convention. The original Step-0 packet
remains byte-identical under `superseded/step0_subminute_stop_20260729/`.

## Step 1 - Authority

- Previous authority SHA-256: `893aa0664944680e053ffd12a4d44c8a798397cbeed6ad56cd682fd864d9f832`
- Amended authority SHA-256: `2363d3f986daba20bd5087ed751dc5b2d839e76cd6413aeca0bcd255eb98857a`
- Entry decisions at minute `t` fill at the completed `t+1` executable ask.
- Exit decisions at minute `v` fill at the completed `v+1` executable bid.
- No executable future bid is a full-loss state, not neutral censoring.
- D49 permanently soft-closes entries below the $1 premium-plus-fee floor.
- Census sessions exclude outer-test, protected-holdout, and embargo sessions.
- Sub-minute historical acquisition remains deferred and unauthorized.

## Step 2 - FT2-04 V2

- Census role: 45 sessions; all role intersections are empty.
- Last legal entry decision: 15:29; legal fill: 15:30.
- Last learned exit decision: 15:54; fill/forced-flat boundary: 15:55.
- Labels and runtime masks are physically separated.
- Runtime horizon availability derives from causal clock/session state only.
- The no-bid full-loss law and parallel MNAR sensitivity are frozen.
- V1 is preserved at `protocol101_ft2_04_path_label_freeze/superseded/`.

## Step 3 - FT2-05 V2 Census

- Sessions: **45**
- Governed candidate labels: **460,937**
- D48 reference labels: **159,312**
- Distinct D48 minutes: **15,183**
- Remaining-session path coverage: **1.000**
- No-bid full-loss states: **2,690,689 /
  31,579,138 (8.52%)**
- Outcome: `feasible`

The convention materially changes economics, as expected. For the
best-session oracle, pooled PnL changed from
`$377,161` to `$341,796`
(`-35,365`). The P5 diagnostic changed from
`$117,454` to `$156,830`
(`+39,376`), but v1/v2 are not directly
interchangeable because fill timing, missingness, role membership, and budget
semantics all changed. The MNAR comparison is report-only; future candidate
rankings may not depend on whether no-bid states are retained as losses or
excluded.

## Step 4 - FT2-08 V2

- Contract validation: **17/17 checks passed**.
- The tensor retains 42 governed SPXW slots plus WAIT.
- The signed 17-feature alpha scope appears exactly once across the market and
  contract axes; contract-dependent alignment, composite, delta, and gamma
  fields are no longer misrepresented as global.
- Source-neutral contract identity, IBKR alias mapping, ATM rounding/tie-break,
  recentering, and no-inheritance fixtures are frozen.
- Flat incomplete ladders force WAIT; open incomplete ladders force HOLD unless
  committed floor/forced-flat law controls the action.
- Open positions retain a dedicated identity-keyed 90-minute quote/path block
  even when outside the recentered flat ladder.
- Account state uses integer cents, a frozen paper snapshot mapping, ordered
  fills/commissions, restart reconciliation, and unavailable-state safe action.
- Replay v5.1 requirements are design-only. Simulator v5 source remains
  byte-identical at `7296a437577ed006326d2ad35ad1f3499c4925334556d64d8c5fb75e4985f548`.
- FT2-08 receipt SHA-256: `584612af00f8e04b9b903f1fb4b56396a3dc950bd21f18718fa72fd4ff8dfb9c`

## FT2-20 Findings

- Findings crosswalked: **28/28**
- Repaired in FT2-04/05/08 design: **14**
- Partially repaired with downstream gate retained: **1**
- Explicitly deferred to FT2-10/11 without severity downgrade: **13**
- Severity downgrades: **0**

## Scope And Stop

No model was trained or fitted. No threshold was tuned. No protected holdout,
outer-test, recorder-day, broker, paid-data, GPU, runtime, promotion, or paper
path was accessed. Simulator v5 source was not modified. FT2-10 and FT2-11 were
not repaired or started.

The next bounded node is `FT2-10-ENTRY-SCIENCE-CONTRACT-REPAIR`. FT2-20 reruns
with fresh review seats only after the FT2-08, FT2-10, and FT2-11 repairs all
exist.
