# Protocol101 Stage-1 G4 Feasibility V2

Offline measurement only. No model training, threshold selection, broker calls, paid downloads, paper-submit, launchd/runtime edits, promotion/default changes, real-money paths, or sealed market-data reads occurred.

## Result

- Governed sessions: `271`
- Candidate rows: `3412556`
- Current G4 nominal 10k cap: `$2,500`
- Random mean drawdown: `$20,493`
- Random max drawdown: `$38,245`
- Oracle max drawdown: `$0`
- Cheap-oracle max drawdown: `$0`

## Fold Table

| Fold | Random DD | Random PnL | Oracle DD | Oracle PnL | Cheap Oracle DD | Cheap Oracle PnL |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | $11,977 | $-7,430 | $0 | $251,442 | $0 | $51,083 |
| 1 | $20,483 | $-17,767 | $0 | $168,067 | $0 | $56,561 |
| 2 | $21,073 | $-6,826 | $0 | $190,141 | $0 | $72,998 |
| 3 | $10,689 | $13,232 | $0 | $247,573 | $0 | $65,040 |
| 4 | $38,245 | $-19,211 | $0 | $300,115 | $0 | $78,906 |

## Draft Owner Decision

The stale holdout absolute `max DD <= $1,500` line should be replaced before any holdout burn. A consistent owner-signature draft is provided in `proposed_gates_revision.md`.
