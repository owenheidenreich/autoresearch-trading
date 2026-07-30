# Protocol101 Stage-1 Regimen Repair Amendment - SIGNED FINAL

Status: **OWNER-SIGNED AND BINDING**

Effective date: 2026-07-26

This amendment adopts the corrected Stage-1 repair design in:

```text
v4/audit/autoresearch/
protocol101_stage1_regimen_repair_design_correction_attempt002/
owner_decision_packet.json
```

Signed owner-packet SHA-256:

```text
cd69707b34bf67af94b76c36443ba34cf0fa2c84e619ff03c48a8305b1703817
```

## Approved Decisions

1. Exit pricing and serial occupancy use separate clocks.
2. Stop-loss, take-profit, and no-bid exits use the event quote time for both
   pricing and occupancy.
3. Max-hold and forced-flat exits release occupancy at the policy deadline and
   use the latest causal quote at or before that deadline for pricing.
4. Frozen `labels_net_pnl` and `labels_mid_pnl` remain the bit-identical E06
   and E07 equivalence targets.
5. Quote age is required audit reporting and is not a model feature or gate.
6. Simulator v5 uses realized occupancy exit time while preserving v4 account
   continuity, fee reserve, and other signed account rules.
7. Duplicate session, decision, contract, canonical-slot, and split-role
   identities fail closed before fitting, scoring, hashing, or replay.
8. Existing models may be reused only if every E01-E13 check passes. Any
   failure requires refitting all 420 model units. Mixed reuse is forbidden.
9. The preserved D1, D5, D6, immutable rebuild, and independent-acceptance
   decisions remain binding.
10. Campaign multiplicity uses the synchronized 20,000-replicate five-session
    maxT procedure as a hard eligibility control. Adjusted
    `p_FWER <= 0.05` is required in addition to G2.

## Authorization Boundary

This signature authorizes only the bounded machinery implementation followed
by a separate independent machinery-acceptance Goal.

It does not authorize economic replay, fitting, refitting, model scoring,
gate aggregation, ranking, selection, seed 45, protected holdout access,
sealed evidence access, broker connectivity, paper submission, promotion,
runtime or launchd changes, paid downloads, or real-money work.

## Owner Sign-Off

Signed by: Owen Heidenreich

Date: 07/26/2026

Multiplicity choice: `HARD_ELIGIBILITY_CONTROL`
