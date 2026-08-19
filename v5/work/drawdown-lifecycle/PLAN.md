# Plan — drawdown-ordered lifecycle experiment (job 45)

**Registered 2026-08-15 on the owner's signed STOP override and scoped ledger reopening.** After
the short-vertical census fired the pre-committed STOP, the owner instructed — explicitly, after
being shown the conflict — *"continue trying to create a model that can trade 0dte options long
calls and long puts"*, with the design: enter where predicted post-entry drawdown is small (not
beyond −30%), exit trades that violate that thesis early, capture the profits of trades that run.
Authorization: [`STOP_OVERRIDE_DRAWDOWN_LIFECYCLE_2026_08_15.md`](../../governance/STOP_OVERRIDE_DRAWDOWN_LIFECYCLE_2026_08_15.md)
and [`LEDGER_REOPENING_DRAWDOWN_LIFECYCLE_2026_08_15.md`](../../governance/LEDGER_REOPENING_DRAWDOWN_LIFECYCLE_2026_08_15.md),
both signed 2026-08-15.

## Question

Does causal pre-entry state identify SPXW 0DTE long call/put entries whose option-mid path reaches
**{+30%, +50%, +100%} before ever touching −30%** within 60 minutes (Bonferroni 0.05/3), precisely
enough that the declared lifecycle — enter at the ask, stop at the bid on the first −30% touch,
otherwise hold 60 minutes, one position, $10,000 serial account — has positive executable
expectancy with a family-corrected lower bound above zero?

## Measured basis (2026-08-15, advisory; receipted recomputation under the frozen fit declaration)

32,976 random-entry 60-minute quote paths, 251 sessions: median MAE −44%; at the +50%/−30% cell
the clean-runner base rate is **31.8%**, winners under the declared law average **+$641**
(SD $1,086), losers **−$333** (SD $239 — the stop's tail control), break-even precision **34.2%**.
Calibration draws: `AR_TRADING_DATA/derived/drawdown_calibration_v1.parquet`
(sha `83384d6f…`).

## Phases

1. **Known-answer preflight (binding; kill 1).** Synthetic worlds calibrated to the measured
   draws; production training law; frozen rank+floor selector; serial walk; full gate. Declared in
   [`PREFLIGHT_DECLARATION_V1.json`](PREFLIGHT_DECLARATION_V1.json) (self-hash `596f7b5f…`):
   40 planted trials at the calibrated minimum effect, 60 nulls, PASS = recovery ≥80% with null
   Wilson upper <5%. **Sensitivity statement: the certifiable effect is ≈ +20pp of selected
   precision over the 31.8% base — a real negative rules out edges of that size and nothing
   smaller.** Outcome-blind probe grids preceded the freeze under disjoint seed prefixes; the
   campaign bank is virgin.
2. **Real entry fit (only after PREFLIGHT_PASSED).** New hashed fit declaration: label family
   {+30%, +50%, +100% before −30%}, 48-parameter `compact_interaction_entry`, production law,
   deterministic training prefix + five chronological score blocks, frozen causal selector,
   composition-matched and shuffled controls, timestamp audit, serial account, kills 2–5 of the
   signed reopening.
3. **Exit model (only if phase 2 survives every kill).** Fitted solely on surviving out-of-fold
   entries; duration-matched controls; oracle labeled unattainable.

## Boundaries

No purchase, no vendor or broker contact, no paper or live orders, no reserved (post-2026-08-05)
sessions, no promotion. One shot per the reopening: any kill spends it and the 2026-08-15 STOP
resumes.
