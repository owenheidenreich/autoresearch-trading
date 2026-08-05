# Evidence Index

Large data and generated evidence stay outside v5. This file identifies the evidence that supports the
current claims so an agent does not have to browse the audit tree blindly.

| Claim | Evidence | Meaning |
|---|---|---|
| The project may be unable to measure a cost-scale edge | [Gate-chain audit](../research/findings/GATE_CHAIN_AUDIT_2026_08_05.md) | Reproduces session variance, power limits, gate defects, and confirmation limits |
| Five prior campaigns found no edge | [Do-not-retest ledger](../research/history/DO_NOT_RETEST.md) | Records each experiment, result, and genuinely new reopening condition |
| The protected holdout is spent | [Do-not-retest ledger](../research/history/DO_NOT_RETEST.md) and [gate-chain audit](../research/findings/GATE_CHAIN_AUDIT_2026_08_05.md) | The `signed18` look-ahead investigation opened it on 2026-08-02 |
| ES round-trip friction is 0.358 points / $17.92 | [Gate-chain audit](../research/findings/GATE_CHAIN_AUDIT_2026_08_05.md) | Measured spread, fees, and slippage define the G1 economic bar |
| Option round-trip friction is $3.08 | [Gate-chain audit](../research/findings/GATE_CHAIN_AUDIT_2026_08_05.md) | Defines the conditional G2 replay cost |
| Paper execution capability exists | [`v4` reviewer brief](../../v4/docs/protocol101/training/research/FABLE_REVIEW_BRIEF_2026_08_04.md) | One guarded paper round trip proves plumbing, not edge |
| Unattended Python can read the repository | [Track-A handoff](../../v4/docs/protocol101/training/research/HANDOFF_TRACKA_LAUNCHD_CANNOT_READ_REPO_2026_08_05.md) | The direct Python entry path was verified; real jobs remain unloaded |
| Raw generated evidence | [`v4/audit/autoresearch/`](../../v4/audit/autoresearch/) | Protected run receipts; inspect a named family, never sweep or overwrite |
| Owned market data | `~/.autoresearch-trading/` and `/Volumes/AR_TRADING_DATA` | Protected local data; never move, delete, download, or open reserved evidence casually |

An absent receipt is `UNKNOWN`. Runtime evidence wins when a document and the machine disagree.
