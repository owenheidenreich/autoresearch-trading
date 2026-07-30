# FT2-04 Path-Label Freeze — Report

Node: `FT2-04-PATH-LABEL-FREEZE` (Graph V2). Outcome: **`labels_frozen`**.
Prepared: 2026-07-28. Scope: **definitions and manifests only.** No census
statistic, distribution, ceiling, or label value was computed on real data.
The only numbers below are hand-made toy figures in the worked examples.

> **Repair note (2026-07-29 — protected-holdout firewall).** The original census
> (76 sessions) implemented the first D42 formula, which excluded only the
> outer-test union and therefore leaked all 30 protected-holdout sessions into
> the census. Caught at packet verification **before any census statistic was
> computed**. Under owner authorization, D42 was amended to also exclude
> protected resources; the census, verifier, test, proof, and receipt were
> repaired. The frozen label/oracle definitions were **not** touched.
> **Repaired census = 46 sessions.** New signed-authority SHA-256:
> `893aa0664944680e053ffd12a4d44c8a798397cbeed6ad56cd682fd864d9f832`. The
> original artifacts are preserved under `superseded/`.

## Authority verification (done before any work)

- Signed authority `PROTOCOL101_FULL_TRADER_GRAPH_V2_CONSOLIDATED_AUTHORITY_2026_07_28.md`
  SHA-256 verified **equal** to
  `e08b02b31a937bcb7c15909f66cfce5bed69678d7469c8d66c44232f34101c64` ✓
- Graph V2 JSON SHA-256 verified equal to
  `b06a26be59307c130da84f2dc5b6f3224c272e6c4093e83abd5bc0b280ca6d09` ✓
- Governing sections: §7.1 (FT2-04 spec), §7.2 (what FT2-05 consumes), §1
  (product contract incl. D48/D49), §4.2–§4.3 (economics + six families),
  §6.1 (G4 v2 / daily stop).

## 1. Economic primitives (frozen)

- Entry at the executable **ask** `A_t` at decision minute `t`; marks at the
  executable **bid** `B_u` for future minutes `u > t`.
- Contract multiplier `M = 100`; round-trip fee overlay `F = $3` (subtracted
  once from every marked-to-exit PnL — it covers both legs).
- Premium at risk (return denominator, capped loss) `= A_t · M`.
- Fee-adjusted dollar PnL: `PnL$(u) = (B_u − A_t)·M − F`.
- Fee-adjusted return on premium: `R(u) = PnL$(u) / (A_t·M)`.
- Breakeven (strict): `PnL$(u) > 0 ⇔ B_u > A_t + F/M`.
- **Dual representation:** every family target is emitted in *both* fee-adjusted
  dollars and fee-adjusted return, so cheap OTM (explosive %) and expensive
  ATM/ITM (larger $) can't each win on only one axis.

**Horizons:** {3, 5, 10, 20, 45, 90, remaining-session} minutes. Window
`W(t,h) = { u : t < u ≤ t+h }` clipped to the tradable session before forced
flat 15:55 ET; `remaining-session` runs to the last tradable mark before 15:55.

**Censoring near close:** if `t+h` runs past the last available mark, the target
is **censored** at the available boundary (mask the missing minutes, set
`censored=true`, record `available_minutes`). Never shorten the nominal horizon,
never invent or extrapolate marks. Time-to-first-profit is right-censored at the
last available minute when no profit occurs.

**Missing/stale/no-bid minute:** masked — excluded from every window
aggregation, never forward-filled, cannot be a first-profit minute, and breaks
continuity for run-length metrics.

**Tie-breaking:** argmin/argmax picks the earliest minute; the profit threshold
is strict `PnL$ > 0`.

### Shared toy path for all worked examples

`A_t = 2.00`, `M = 100`, `F = $3` ⇒ premium at risk `= $200`. Executable bids at
`t+1…t+6 = [1.90, 1.80, 2.10, 2.40, 1.95, 2.60]`; `t+7` is a no-bid (masked).

| u | B_u | PnL$(u) = (B_u−2.00)·100−3 | R(u) = PnL$/200 |
|---|---|---|---|
| t+1 | 1.90 | −13 | −0.065 |
| t+2 | 1.80 | −23 | −0.115 |
| t+3 | 2.10 | +7 | +0.035 |
| t+4 | 2.40 | +37 | +0.185 |
| t+5 | 1.95 | −8 | −0.040 |
| t+6 | 2.60 | +57 | +0.285 |
| t+7 | — | masked | masked |

## 2. The six path-property families (frozen) + worked examples

**(a) Early drawdown** — horizons {3,5,10}. Worst executable fee-adjusted move.
`EarlyDD_return(h)=min_{u∈W} R(u)`, `EarlyDD_dollars(h)=min_{u∈W} PnL$(u)`, plus
lower-tail quantiles at q∈{0.00,0.05,0.10,0.25} (q=0.00 is the min).
*Worked (h=5):* min over t+1…t+5 ⇒ **−0.115 / −$23** at t+2. (h=3 gives the same,
t+2.) Lower-tail quantile vector computed over {−0.065,−0.115,+0.035,+0.185,−0.040}.

**(b) Time to first real profit** — `TTFP(h)=min{u−t : PnL$(u)>0}`; also record the
first-profit size in both units; right-censored if none.
*Worked:* first `PnL$>0` at t+3 ⇒ **TTFP = 3 min**, first-profit size **+$7 / +0.035**.

**(c) Pre-profit adverse excursion** — worst move *before* the first profit `u*`:
`PPAE_return=min_{u<u*} R(u)`, dollars analogously; whole-window if never
profitable; `0` with `immediate_profit=true` if `u*` is the first minute.
*Worked:* before u*=t+3, over {t+1,t+2} ⇒ **−0.115 / −$23**.

**(d) Underwater burden** — a minute is underwater iff `PnL$(u)<0`.
Depth×duration integral `UWI$=Σ(−PnL$(u))·1min`; return version `UWI_ret=Σ(−R(u))`;
total underwater minutes; longest continuous underwater run.
*Worked (h=5):* underwater at t+1,t+2,t+5 ⇒ **UWI$ = 13+23+8 = $44**,
**UWI_ret = 0.065+0.115+0.040 = 0.220**, **total = 3 min**, **longest run = 2**
(t+1,t+2; t+5 isolated).

**(e) Profitable-window stability** — `frac_pos(h)=|positive|/|available|`; longest
positive run; positive-run count; `jitter(h)=max(|frac_pos(+1)−frac_pos(0)|,
|frac_pos(−1)−frac_pos(0)|)` under a ±1-minute mark shift (masked if neighbor
unavailable).
*Worked (h=5):* positive at t+3,t+4 ⇒ **frac_pos = 2/5 = 0.40**, **longest = 2**,
**count = 1**. Shift +1 ⇒ 3/5 = 0.60; shift −1 ⇒ 2/4 = 0.50 (t+1's neighbor t is
masked) ⇒ **jitter = max(0.20, 0.10) = 0.20**.

**(f) Upside** — `MFE_return(h)=max R(u)`, `MFE_dollars(h)=max PnL$(u)`; upper
quantiles q∈{0.75,0.90,0.95,1.00}; profit area `Σ_{PnL$>0} PnL$(u)·1min` (and
return version).
*Worked (h=5):* **MFE = +0.185 / +$37** at t+4; **profit area = 7+37 = $44**
(t+3,t+4); return profit area = 0.035+0.185 = 0.220. (Over h=6 the MFE rises to
+0.285 / +$57 at t+6.)

## 3. Oracle exit rules (frozen — see `oracle_rules.json`)

Three transparent hindsight rules FT2-05 will replay as opportunity ceilings
(never learned, never entry targets):

1. **best-achievable-bid by horizon** — exit at the max executable bid within the
   horizon (per-horizon ceiling).
2. **hold-to-forced-flat** — exit at the 15:55 bid (no-management baseline).
3. **first-real-profit exit** — exit at the first `PnL$>0` minute, else forced
   flat (ties to family (b); the one rule added beyond the required minimum).

**D48/D49 interaction with oracle sequencing.** Session-start equity `E`
compounds within-split (v5). The **D48 cap** (`A_t·M + 3 ≤ 0.05·E`) masks any
too-expensive contract out of the oracle's choice set. The **D49 budget**
(`L + A_t·M + 3 ≤ 0.05·E`, with `L` = realized session loss so far) gates every
new oracle entry in the serial replay; the 5% realized daily breaker halts
entries once `L ≥ 0.05·E`. **An oracle may never take a trade the rules would
mask.** The threshold-independent per-contract-minute distributions apply D48
eligibility but not D49 (a sequencing constraint); the serial one-account ceiling
applies both. FT2-05 reports both views.

## 4. Census session manifest (frozen — see `census_sessions.json`)

Derived per **D42 (amended 2026-07-29)**: census = governed corpus **minus every
session in any outer-test slice of any fold minus the protected-holdout
sessions** (minus any owner-reserved confirmation sessions once frozen).

- Governed corpus: **301** sessions (2025-01-02 → 2026-03-31), from
  `canonical_processed_session_manifest.json.included_sessions`, cross-checked
  identical to the scope-acceptance `summary.json.sessions` (status `pass`).
- Fold source: `runner_plan.json.expanding_folds`, 5 governed expanding folds;
  `fold_role_map` maps the **validation** role to **test**, so each fold's
  validation list (45 sessions) is its outer-test slice.
- Outer-test union: **225** sessions (5 × 45, disjoint).
- Protected holdout: **30** sessions (2025-05-16 → 2025-06-30), from
  `runner_plan.json.governance.protected_holdout_sessions`; **stays sealed**.
- **Census: 46 sessions** (301 − 225 − 30), range **2025-01-02 → 2025-03-11**.

**Documented observation for FT2-05 (not a defect):** under expanding-window CV
only the *earliest* sessions are never in an outer-test slice, and the repaired
rule further removes the 30 protected-holdout sessions (the entire 2025-05-16 →
06-30 tail). The 46 census sessions are therefore all Jan–early-Mar 2025. See the
regime-window section below for the caveat FT2-05 must carry.

## 5. Intersection proof (see `intersection_proof.json`)

The read-only verifier `compute_census_outer_test_intersection.py` loads the fold
manifest, governance, corpus manifest, and census manifest, and proves:

- census ∩ outer-test-union = **∅**;
- census ∩ protected-holdout = **∅** (the repaired firewall);
- census = corpus − outer-test-union − protected-holdout (completeness: 0 missing,
  0 extra);
- census ⊆ corpus; outer-test ⊆ corpus; protected-holdout ⊆ corpus.

Result: **passed = true** (`{census:46, governed_corpus:301, outer_test_union:225,
protected_holdout:30, outer_test_intersection:0, holdout_intersection:0}`). Its
focused test suite (`test_compute_census_intersection.py`, toy fixtures only)
passes **9/9** and confirms the verifier catches outer-test overlap, **protected-
holdout overlap**, incompleteness, out-of-corpus census, an ambiguous fold-role
map, and missing holdout governance.

## 6. Fold-manifest integrity check

No inconsistency found → **not** `scientific_contract_defect`. Corpus 301 =
271 fold-placed + 30 unplaced; outer-test union 225 all within corpus; 0 sessions
placed but not in the corpus. Authoritative source hashes recorded in
`receipt.json`.

## 7. Regime consequence of the repaired census window (binding on FT2-05)

The repaired census window is **2025-01-02 → ~2025-03-11 (46 sessions)** — largely
a **pre-tariff-crash regime**. The April 2025 volatility regime sits **entirely in
fold-1 outer-test** (and the May–June tail is the sealed protected holdout), so
neither the crash nor the high-vol recovery is represented in the census.

Consequences FT2-05 **must** honor:

- **Stratify** all census outputs by calendar month **and** a simple volatility
  proxy (so the single early-2025 regime cannot masquerade as the whole corpus).
- Carry a **prominent regime-window caveat on every power / MDE input** — variance
  components and detectable-edge estimates derived from a 46-session pre-crash
  window may not generalize to the fold-1 April vol regime or the sealed tail.
- Treat the census as feasibility/power *guidance*, not a regime-complete picture;
  it can only trigger the genuine-impossibility hard stop, never green-light
  training.

This is the deterministic result of the (amended) D42 firewall, not a defect —
surfaced here so it is designed around, not rediscovered.

## Highest allowed claim

> FT2-04 is repaired: the census manifest excludes the protected holdout; the
> firewall is proved against both outer-test and holdout; labels remain frozen.

No model was designed, trained, selected, or made eligible. No census statistic
was computed. No protected resource was accessed — the holdout stays sealed; it
was only *excluded by date* from the census set. No recorder day, broker path, or
paid compute was touched. FT2-05 is a separate Goal and was not begun.
