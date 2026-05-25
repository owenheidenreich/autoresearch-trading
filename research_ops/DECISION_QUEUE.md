# Decision Queue

This queue captures governance decisions that should be made explicitly.

Decision states:

- `queued`
- `needs_cartography`
- `needs_verification`
- `approved`
- `rejected`
- `deferred`
- `blocked`

## Queue

| ID | Decision | State | Why it matters | Required artifact | Default if no decision |
|---|---|---|---|---|---|
| D001 | Should `research_ops/` become the binding cross-agent governance layer? | queued | Agents need one operating surface outside v4. | Decision memo | Use it as advisory only. |
| D002 | Should `.github` issue templates and PR templates be added during Stage 0? | queued | GitHub workflow should enforce artifact types, but template churn should be deliberate. | Decision memo / implementation patch | Defer to Stage 1. |
| D003 | Which blocker gets the first verifier report: fills, quote age, candidate parity, or lifecycle parity? | queued | The first verifier task sets the lab's operating rhythm. | Decision memo | Start with fills/execution realism. |
| D004 | Is the `$500` IBKR reserve unavailable trading capital or only an informational reserve? | queued | Current guard does not subtract reserve, while docs call it reserve. | Verifier report / decision memo | Treat as unresolved assumption. |
| D005 | Which docs are binding when root README, v4 README, promotion packets, and ops scripts disagree? | needs_cartography | Stale-doc contradictions can mislead agents. | Cartography report | Operational source-of-truth doc wins. |
| D006 | Should new model training remain blocked until execution/parity gates close? | queued | Prevents relapse into benchmark mining. | Decision memo | Blocked. |
| D007 | Should Stage 2 `research_ops/` scaffold become binding operating law? | queued | The scaffold adds CSV registry, gates, schemas, and local iteration scripts. | Decision memo | Treat as binding for AI-agent work unless superseded. |
| D008 | Should `ASSUMPTION_REGISTRY.md` be retired in favor of `ASSUMPTION_REGISTRY.csv`? | queued | CSV is easier for scripts and dashboards, but the markdown file remains useful for narrative context. | Decision memo | CSV is binding; markdown is legacy context. |

## Decision Memo Requirements

Every decision memo must state:

```text
Decision:
Does this change PAPER_DEFAULT_PROTOCOL101:
Does this authorize model training:
Does this authorize broker/data/runtime action:
Evidence reviewed:
Risks accepted:
Reversal condition:
Owner:
```
