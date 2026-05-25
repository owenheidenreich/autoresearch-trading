# Evidence Ladder

Last updated: 2026-05-24

This ladder defines how claims become credible. Higher levels do not erase lower
levels; they depend on them.

| Level | Name | Meaning | Example |
|---:|---|---|---|
| 0 | Assertion | A claim without reproducible evidence. | "Protocol101 seems better." |
| 1 | Cartography | The relevant code, docs, artifacts, and data paths are mapped. | Baseline inventory. |
| 2 | Diagnostic replay | A replay or diagnostic run explains behavior but may be exposed. | Known-window replay analysis. |
| 3 | Parity evidence | Replay and live semantics are matched for features, candidates, and lifecycle. | Candidate parity harness. |
| 4 | Execution evidence | Fills, non-fills, spread, quote age, latency, and cancels are observed. | Paper-submit evidence table. |
| 5 | Falsification evidence | A prewritten falsifier was tested and survived. | Fill model remains conservative after live-style evidence. |
| 6 | Untouched evaluation | A frozen candidate is evaluated once on predeclared untouched data. | Future block scored after gates pass. |
| 7 | Promotion evidence | Utility, risk, operations, and rollback conditions satisfy the promotion gate. | Approved decision memo. |

Replay profitability below Level 4 is research evidence, not tradability
evidence.
