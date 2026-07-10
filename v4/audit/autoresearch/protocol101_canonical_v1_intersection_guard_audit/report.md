# Protocol101 Canonical Intersection Guard Audit

Offline burned-day audit. No runtime guard, broker, paper, paid-data, promotion, launchd, or real-money state changed.

- Paired static slots: `45234`
- Historical boundary-stable: `31246`
- IBKR boundary-stable: `29791`
- Exact intersection boundary-stable: `29008`
- Historical-only boundary-stable: `2238`
- IBKR-only boundary-stable: `783`
- Historical excess vs IBKR: `4.8840%`
- Historical acceptance haircut to exact intersection: `7.1625%`

The `vendor_only_training_guard_policy` in `summary.json` is the pessimistic guard policy Stage-1 should reference.
