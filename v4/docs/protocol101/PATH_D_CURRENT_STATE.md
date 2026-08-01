# Protocol101 Path-D current state

- As of: **2026-08-01**
Status: **design repair required; Phase-1 training/backtest not authorized**

This is the current orientation page for the Path-D pivot. It does not replace
the signed consolidated authority. It resolves narrative drift among older
training, paper-runtime, and project-status documents by stating what is signed,
what the walking skeleton proved, and what remains blocked.

## Highest allowed claim

The walking skeleton exercised downstream historical plumbing through a
quarantined forced-BUY harness, but it did **not** produce a learned trader ready
for live paper trading. The real entry composer remains frozen and abstains on
the thin/weak skeleton model. The forced-BUY lifecycle results are branch-coverage
evidence only, not alpha, model-quality, promotion, or paper-readiness evidence.

## Signed state

- Current signed consolidated authority SHA-256:
  `1d215845cf7b853550c5cf27af5bafca66db2355e0f12493e2c5a8922278d4bc`.
- Current graph SHA-256:
  `9955085a31840da63057761a620a5ec2995e04f05ff2aa5f4906afd795726a08`.
- A6 is owner-signed: the entry gate uses a session-clustered lower-confidence
  bound on conditional mean upside after fees; q10 remains a ranking/risk
  quantity and cannot be the positive-entry statistic.
- A7/D59 is owner-signed: Tier-S `cbbo-1s` is allowed for quarantined
  feasibility/skeleton work; Tier-T raw-event-path evidence remains required for
  trusted loss-control and promotion/paper claims. A7 did not activate 1-second
  decisions or authorize a download.
- The graph topology is unchanged by A6/A7.

Signed evidence:

- [Consolidated authority](training/contracts/PROTOCOL101_FULL_TRADER_GRAPH_V2_CONSOLIDATED_AUTHORITY_2026_07_28.md)
- [A6 owner receipt](../../audit/autoresearch/protocol101_ft2_10_entry_gate_convexity_amendment/owner_approval_receipt.json)
- [A7 owner receipt](../../audit/autoresearch/protocol101_d59_staged_subminute_representation_amendment/owner_approval_receipt.json)

## Walking-skeleton result

The first skeleton earned useful negative and plumbing evidence:

1. It found and closed the structurally unsatisfiable q10 entry gate (L1/A6).
2. After A6, 552 proposals reached the confidence layer, but all real decisions
   still abstained because model error was roughly an order of magnitude larger
   than directional signal and action-conditioned evidence was too thin (L9).
3. A quarantined wrapper converted a deterministic subset of those proposals to
   forced BUY intents only to exercise lifecycle/simulator/chart paths. It did
   not alter the runtime composer or prove that a learned entry policy trades.
4. The downstream replay fired learned-exit, floor, and forced-flat branches,
   but those canary-shaped outcomes cannot support policy tuning or economic
   claims.
5. The run exposed real-campaign requirements: prevalence-aware exit metrics,
   native source/decision clocks, exact safety-boundary tests, rendered chart QA,
   governed data roles, and bulk acquisition.

The durable details and claim boundaries are in the
[walking-skeleton learnings ledger](training/execution/PROTOCOL101_WALKING_SKELETON_LEARNINGS_LEDGER_2026_07_30.md).

## Path-D direction

The owner-selected planning direction is:

```text
historical/live model market plane
  = Databento OPRA + ThetaData SPX

broker plane
  = IBKR execution, confirmed position/account state, and safety

candidate shape
  = frozen minute entry + learned 1-second lifecycle exit
```

This direction is recorded in the
[transition plan](training/execution/PROTOCOL101_PATH_D_TRANSITION_PLAN_2026_08_01.md)
and [sequencing plan](training/execution/PROTOCOL101_PATH_D_AMENDMENT_SEQUENCING_PLAN_2026_08_01.md).
Those documents are planning records, not signed authority amendments. Data
acquisition/subscription status is `UNKNOWN` unless a separate receipt proves it.

## Current blocker

The latest independent governance review returned **NEEDS CHANGES — STOP before
signature**. The Path-D model-plane proposal and FT2-60 exit design are not
jointly signable yet. The blocking defects are:

- `Q(hold)` lacks a deployable continuation policy/value law;
- the provisional Phase-1 decision/fill/no-fill/latency law is absent;
- Tier-S versus Tier-T use is contradictory across documents;
- market-source purity incorrectly excludes broker-confirmed position/fill state;
- exact clocks, OOF trajectory construction, feature lineage/allowlists,
  calibration, multiplicity, and economic acceptance remain underspecified; and
- Phase 1 lacks a resolved Path-D authority/checker namespace.

The full findings and minimum repair package are in the
[Path-D governance review](training/execution/path-d-governance-review.md).

## Next permitted gate

1. Repair the model-plane and FT2-60 documents using the nine-item minimum repair
   package in the governance review.
2. Add the exact additive FT2-08 one-second tensor/label contract and frozen
   provisional fill contract.
3. Re-run independent review.
4. Obtain owner signature on a non-vacuous Path-D Phase-1 authority overlay.
5. Only then may a separately authorized Tier-S feasibility skeleton/backtest be
   specified and run.

Until those gates pass: no Path-D model fitting, threshold tuning, Phase-1
backtest claim, paid-data download, broker contact, runtime/default change,
paper-submit, promotion, or real-money claim follows from this pivot record.

## Fresh-agent reading order

1. This page.
2. [Walking-skeleton learnings ledger](training/execution/PROTOCOL101_WALKING_SKELETON_LEARNINGS_LEDGER_2026_07_30.md).
3. [Path-D governance review](training/execution/path-d-governance-review.md).
4. [Path-D transition plan](training/execution/PROTOCOL101_PATH_D_TRANSITION_PLAN_2026_08_01.md).
5. [Path-D amendment sequencing](training/execution/PROTOCOL101_PATH_D_AMENDMENT_SEQUENCING_PLAN_2026_08_01.md).
6. The signed authority and A6/A7 owner receipts above.

The existing `PAPER_DEFAULT_PROTOCOL101` runtime is a separate legacy operating
surface. Its existence does not mean the new learned Path-D trader has earned
Stage 4 or paper readiness.
