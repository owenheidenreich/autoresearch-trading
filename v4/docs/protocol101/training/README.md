# Protocol101 Full Trader Program

Status: **CANONICAL PRODUCT AND GRAPH FRONT PAGE**

Last reconciled: **2026-07-28, Full Trader graph reset**

This is the one starting point for Protocol101 model research. It defines the
product that must be built, the evidence that has already been earned, and the
only workflow allowed to authorize new experiments.

Detailed experiment evidence remains under `v4/audit/autoresearch/`. Signed
historical contracts remain preserved under `contracts/`, but no old contract,
Goal prompt, baseline, or experiment may silently replace the product defined
here.

## Product Invariant

Protocol101 must learn the complete trading job:

```text
market + full governed eligible SPXW 0DTE ladder
  -> learned flat-state choice:
       WAIT
       or BUY one specific eligible call/put/strike
  -> while that position is open, learned choice:
       HOLD
       or EXIT
  -> finish the session flat
```

The flat-state model owns timing, direction, and contract selection. The
open-state model owns the causal HOLD/EXIT decision. The final object is their
combined behavior under one-account serial replay.

Hard-coded rules may enforce eligibility, affordability, one contract, one
position, fees, daily loss limits, stale-data abstention, and forced flat.
They may not decide the model's permanent timing, call/put direction, strike,
or nearest-ATM preference.

P5, nearest-ATM selection, fixed exits, and earlier numbered protocols are
baselines and historical evidence. They are not substitutes for the learned
action space.

An entry-only model, an ATM-only model, or an exit model attached to a
hard-coded entry policy is not the finished Protocol101 trader and cannot earn
IBKR paper readiness.

## Current Position

| Item | Status |
|---|---|
| Scoped historical/live synchronization | **Passed for the previously certified feature boundary** |
| Causal dataset, two-clock exits, simulator v5, identities, and serial-account machinery | **Reusable engineering assets** |
| H0-H3 entry campaign | **Closed negative/inconclusive experimental evidence; not an active roadmap** |
| M0/M1 P5-timed contract-selector pilot | **Closed negative evidence** |
| One-minute P5 HOLD/EXIT pilot | **Closed negative evidence** |
| P5 as permanent timing/direction/nearest-ATM policy | **Rejected architecture drift** |
| Accepted full-ladder flat-state model | **None** |
| Accepted learned HOLD/EXIT model | **None** |
| Complete Full Trader candidate | **None** |
| Protected holdout and final confirmation | **Unspent** |
| IBKR paper readiness | **Not earned** |
| Current graph node | **Parked before `FT-10-FULL-LADDER-MODEL-DESIGN`** |

The negative experiments are not discarded. They establish that:

- the H0-H3 fixed-exit formulation did not produce trusted selectable entry
  evidence;
- the M0/M1 formulation did not beat P5's deterministic contract choice;
- the one-step P5 lifecycle formulation failed strict serial replay;
- weak negative controls can make profitable exposure look like learned alpha;
  and
- a local experimental failure must not redefine the product.

## Active Graph

The machine-readable authority is:

- [Full Trader graph contract](execution/PROTOCOL101_FULL_TRADER_GRAPH_V1.json)
- Controller: `v4/scripts/run_protocol101_full_trader_graph.py`
- Persistent state:
  `v4/audit/autoresearch/protocol101_full_trader_graph_v1/`

The graph is a deterministic controller. It records state, validates receipts,
enforces bounded routes, and stops at owner gates. It does not train models,
invent experiments, approve its own work, access protected evidence, or call a
broker.

```text
FT-00 graph reset
  -> FT-10 full-ladder flat-state design
  -> FT-20 parallel trading/ML/live reviews
  -> FT-21 owner design approval
  -> FT-30 mechanical harness pilot
  -> FT-31 independent machinery acceptance
  -> FT-40 non-promotable feasibility pilot
  -> FT-41 independent feasibility audit
  -> FT-50 full flat-policy campaign
  -> FT-51 independent flat-policy audit
  -> FT-52 owner entry freeze
  -> FT-60 learned lifecycle design
  -> FT-61 lifecycle review and owner approval
  -> FT-69 lifecycle machinery pilot
  -> FT-70 independent lifecycle machinery acceptance
  -> FT-71 lifecycle training
  -> FT-72 combined Full Trader audit
  -> FT-80 complete-system freeze
  -> FT-81 fresh confirmation
  -> FT-82 protected holdout
  -> FT-90 IBKR decision shadow
  -> FT-91 no-order live shadow
  -> FT-91A independent live-shadow audit
  -> FT-92 owner paper authorization
  -> FT-93 guarded IBKR paper validation
  -> FT-94 independent paper-evidence review
```

Scientific failure returns to hypothesis design. Mechanical failure permits
only a bounded repair of the identified machinery defect. Product-scope
changes, inconclusive terminal evidence, protected-data access, and paper
authorization always stop for the owner.

## Drift Tripwires

The controller must stop if any proposed node or Goal:

- removes `WAIT` from the flat-state learned action space;
- prevents the model from selecting among the full governed eligible ladder;
- hard-codes call/put direction or nearest-ATM selection as the final policy;
- treats P5 or another baseline as the permanent entry architecture;
- trains HOLD/EXIT before a flat-state candidate earns its required gate;
- lets the producer independently approve its own evidence;
- weakens a scientific requirement after results are visible;
- uses positive PnL alone as proof of learned edge;
- opens confirmation, holdout, recorder, shadow, paper, or broker paths early;
  or
- claims an entry-only component is the complete trader.

Such a result is `product_drift_detected`, not a repairable experiment failure.

## Agent Roles

| Role | Responsibility |
|---|---|
| Controller | Routes receipts and state; cannot make scientific decisions |
| Research designer | Proposes one bounded hypothesis and exact experiment |
| Trading-realism reviewer | Tests whether the game resembles an executable 0DTE trader |
| ML/statistics reviewer | Tests labels, leakage, controls, inference, and multiplicity |
| Live-parity reviewer | Tests whether every runtime input and action can exist through IBKR |
| Executor | Implements or runs the owner-approved node only |
| Independent auditor | Reproduces and accepts or rejects evidence without producer authority |
| Owner | Approves product changes, major designs, protected evidence, and paper trading |

The same actor may not both produce and independently accept a node.

## Loop Rules

Every loop has a trigger, evidence requirement, retry budget, and exit:

- Design repair: at most two revisions before owner review.
- Mechanical repair: at most two revisions of the identified defect.
- Scientific redesign: at most three new hypotheses in one campaign family.
- A failed scientific hypothesis is recorded as evidence and is not rerun with
  softer gates.
- An inconclusive result stops for owner review instead of being narrated as a
  pass or failure.

No node loops on confidence. It loops only on new evidence.

## Preserved Assets

The reset keeps:

- the synchronized historical corpus and vendor provenance;
- canonical minute-state and eligible-ladder construction;
- simulator v5 and the two exit clocks;
- one-account serial replay, fees, affordability, and forced-flat behavior;
- causal feature and future-information firewalls;
- duplicate and identity rejection;
- historical/IBKR parity tools;
- immutable artifacts, hashes, registries, and resume machinery; and
- every prior result as historical evidence.

Whether a previous feature family, label, model, gate, or policy belongs in the
new trader is decided at `FT-10` and its independent review. Reuse is earned,
not assumed.

## Closed Work

The former Stage-1 graph and its 27 Goal prompts are preserved byte-for-byte
under:

- [Closed Stage-1 graph and Goals](history/closed_stage1_graph_and_goals_2026_07_28/README.md)

They are reproducibility evidence only. Their scripts and output packets may
be inspected, but their Goal prompts are not active instructions.

Important research history:

- [Original Protocol101 anatomy](research/trading_bot_engineer_strategy_audit_2026_05_24/02_PROTOCOL101_ANATOMY_AND_TRADES.md)
- [Prior campaign distillation](history/PROTOCOL101_PRIOR_CAMPAIGN_DISTILLATION.md)
- [Protocol farm lineage](history/PROTOCOL101_PROTOCOL_FARM_LINEAGE_2026_07_19.md)
- [Trader charter](contracts/PROTOCOL101_TRADER_CHARTER.md)
- [Scoped synchronization decision](../synchronization/contracts/PROTOCOL101_SCOPED_SYNCHRONIZATION_DECISION_2026_07_25.md)

These sources inform design; they do not bypass the active graph.

## Current Stop Boundary

`FT-00-GRAPH-RESET` is complete. The controller is initialized in
`parked_before_node` state with:

```text
next_node = FT-10-FULL-LADDER-MODEL-DESIGN
```

No Goal prompt for `FT-10` exists yet. No model design, fitting, replay,
selection, confirmation, holdout access, broker contact, or runtime change is
authorized by this reset.

The next owner action is to review the initialized graph packet. Only then may
the `FT-10` Goal be written and launched.

## Documentation Rule

Do not create another general index, control center, status page, or competing
roadmap.

Future Goal prompts live in `goals/` only while active and must be generated
from exactly one graph node. On terminal completion they move to history with
their receipt. This README and the machine-readable graph state are the only
current narrative and execution authorities.
