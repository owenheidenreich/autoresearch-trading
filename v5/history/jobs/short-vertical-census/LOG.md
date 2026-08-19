# Log — defined-risk short-vertical census (job 44)

### 2026-08-15 — declaration V1, first run VOIDED before any outcome was read.

- The frozen V1 declaration (`5d37a8f5…`) launched on the 243 coverage-included sessions. Evaluation
  completed and the row table was written, then the runner crashed in aggregation: `np.array_split`
  applied to a DataFrame instead of session indices (the iron-fly runner splits indices). The crash
  preceded every economic aggregate; the only figures observed were population counts (8,748
  cell-sessions, 4,104 trades). The partial output directory was deleted.
- Runner repaired: aggregation extracted into `summarize_cells` and unit-tested end to end on
  synthetic rows (8 census tests green). Declaration V2 (`45f78cae…`) carries identical law, family,
  inputs, seeds and decision rule with the corrected runner hash and a supersedes clause.
- V2 run in progress. No purchase, vendor/broker contact, fit, or order at any point.

### 2026-08-15 — V2 run complete: `STOP_CONDITION_MET_NO_CELL_CLEARS_FEE_ONLY`. Packet closed.

- 4,104 trades over 8,748 cell-sessions. Best fee-only cell +$14.85/session (call credit, w5, ATM,
  13:00) against a family-corrected lower bound of −$30.59; best touch cell +$2.53. All ten positive
  fee-only cells are call credits; every put-credit cell is negative. No cell meets the declared pass
  at either law, so the owner's pre-committed STOP of options-strategy development fires.
- Durable records: finding `research/findings/SHORT_VERTICAL_CENSUS_2026_08_15.md`, do-not-retest
  ledger row with reopening conditions, STATUS row 44 and header notice, §17 settled. The unsigned
  trading-side charter amendment is moot and remains unsigned.
