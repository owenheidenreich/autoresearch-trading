# Protocol101 FT2-05 Opportunity Census V5

## Owner Memo

**Outcome: `feasible`.** executable post-fee opportunities exist; FT2-05 cannot green-light training

This is a label-side map of the governed game, not evidence that a profitable
model exists. Across `45` development-only sessions, the fixed $10,000
D48 reference mask left `15,150` distinct executable minutes
with at least one contract, or about `336.7` minutes per
day. A deliberately loose 25%-tail quality screen plus positive conservative
upside left about `232.8` qualifying minutes per day. This is a
curve diagnostic, not a selected guardrail.

The hindsight `best_session` selector made `$339,686.00` under one-account
simulator-v5 semantics, D48, D49, fees, and the daily stop. P5 using the same
hindsight exit rule made `$154,008.00`. These are ceilings: they prove paths
exist, not that present-time features can identify them. The median entry
friction across premium bands was `5.3%` of premium.

The primary remaining-session paths contained `1,904,900` no-bid
minutes out of `25,853,867` marks (`7.37%`). Those minutes are
valued at full loss rather than erased. Across the preregistered MNAR
no-bid-excluded sensitivity, the minimum path-metric rank correlation was
`0.9838`. This is label-surface stability, not candidate-policy
stability; FT2-10/11 must still verify the latter.

Compared with preserved FT2-05 v2, v3 changes the realized-entry label
population by
`-37,759` rows.
It records `128,758` causal fee-3
intent opportunities and
`7,205` fee-3 t+1
rejections (plus the separately reported fee-4 path). The transparent
best-session oracle
changed by `$-1,060.00` and
`0` trades. This v2-v3 impact is attributed
only to the preregistered causal intent/fill-recheck and active-fee soft-close
repair; no new sessions or unrelated label definitions were introduced.

Compared with preserved FT2-05 v3, v4 restores
`7,205` rows to the
D48 reference/distribution population. Those are exactly the time-`t`
intent-eligible rows that did not pass the selected-only `t+1` recheck. The
reference universe is therefore the canonical time-`t` intent population;
successful realized entries and rejections remain separate outcomes.

The fee-3 fill-recheck rejection numerator is `7,205`.
It is `7,205/460,937
= 1.56%` of all governed contract-minute
rows, and `7,205/128,758
= 5.60%` conditional on reaching the
recheck population. These denominators answer different questions and are
never interchanged.

The 45-session cluster calculation estimates that a best-session improvement
would need to be roughly `$63,338.25` in total PnL to achieve conventional
95%/80% detection under the observed session variance. FT2-11 must refine this
before paid training.

Where D48-reference opportunities live:

- Market phase: `[{"count": 40809, "market_phase": "lunch", "share": 0.35186239006725295}, {"count": 31685, "market_phase": "afternoon", "share": 0.2731936540782894}, {"count": 31650, "market_phase": "primary_morning", "share": 0.2728918779099845}, {"count": 11836, "market_phase": "europe_close_transition", "share": 0.10205207794447319}]`.
- Moneyness: `[{"count": 64236, "moneyness_band": "wing", "share": 0.49888938939716365}, {"count": 42419, "moneyness_band": "near", "share": 0.3294474906413582}, {"count": 22103, "moneyness_band": "atm", "share": 0.17166311996147812}]`.
- Premium band: `[{"count": 78076, "premium_band": "small_1_3", "share": 0.6063778561332112}, {"count": 47921, "premium_band": "medium_3_8", "share": 0.37217881607356434}, {"count": 2761, "premium_band": "cheap_le_1", "share": 0.021443327793224498}]`.

**Regime warning:** this census sees only January through early March 2025.
April 2025 volatility is in fold-1 outer test, and May-June 2025 is protected
holdout. No conclusion here may be generalized to those regimes. Every headline
table is therefore also stratified by month and the governed SPX
regular-session realized-range proxy.

The only allowed interpretation of `feasible` is: the census characterizes the
opportunity surface of the governed game on the 45-session window, and FT2-08
design work may start. It does not authorize design acceptance, training,
selection, holdout access, shadow operation, or paper trading.

## Frozen Scope

- Node: `FT2-05-OPPORTUNITY-CENSUS`.
- Product-contract SHA-256: `1d215845cf7b853550c5cf27af5bafca66db2355e0f12493e2c5a8922278d4bc`.
- Sessions: exactly 45 from the repaired FT2-04 manifest.
- Horizons: `(3, 5, 10, 20, 45, 90, 'remaining_session')`.
- Entry/exit accounting: BUY masks at `t` use only `A_t`; the selected exact
  intent alone is rechecked at `t+1` and either fills at the executable ask or
  rejects with no position, premium, fee, or PnL. EXIT decision at `v` fills at
  the `v+1` executable bid.
- No-bid accounting: full-loss path state; exact 15:55 forced-flat boundary.
- D49 soft-close: the causal current ladder must contain an otherwise-eligible
  contract of at least $1 whose intent cost fits; the threshold is computed as
  $103 on fee_3 and $104 on fee_4.
- Hard safety: D48 premium cap, D49 remaining loss budget, one open position,
  15:30 entry cutoff, 15:55 forced flat.
- Volatility proxy: session SPX realized range in basis points.
- No model was fitted or scored.

## Threshold-Independent Surface

The packet contains the complete family distributions, Pareto frontier,
guardrail curves, excluded-winner curves, and oracle-assisted trade-rate curves.
Guardrail levels are percentile screens across `0%, 10%, 20%, 25%, 30%, 40%,
50%`; none is selected here.

## Oracle And P5 Ceilings

All nine transparent variants are reported: seven best-achievable-bid horizons,
hold-to-forced-flat, and first-real-profit. P5 controls direction and nearest
eligible-under-cap strike only; each comparison uses the same exit oracle to
isolate its entry choice.

## Friction, Power, And Regimes

Premium-band friction includes spread crossing plus the $3 fee and is
conditional on successful `t+1` fill rechecks. The table separately reports
the time-`t` intent-row count, fill-pass/rejection counts, friction-valid count,
and decision-time spread-valid count. MDE tables use session clusters and
separately show cluster-adjusted trade-count projections. Regime headline
tables split every replay by month and SPX range tercile.

## Compute

- Label rows: `460,937`.
- D48-reference rows: `128,758`.
- Wall time: `0.171` hours.
- CPU time: `0.170` core-hours.
- Resume checkpoints: one Parquet file and summary per session.

## Route

Terminal outcome is `feasible`. The node stops here. FT2-08 was not started.
