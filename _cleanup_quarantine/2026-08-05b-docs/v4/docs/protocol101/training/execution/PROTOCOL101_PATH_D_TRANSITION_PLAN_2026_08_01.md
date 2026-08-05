# Path D Transition Plan (2026-08-01)

**STATUS: PLANNING — not an authority amendment.** Records the committed Path D
direction, the exact data to buy tonight, how we use it, the project assumptions
it changes, and the IBKR/vendor changes. Binding contracts still win; conflicts
resolve before execution.

**Path D (committed):** train the model on **Databento** (OPRA options) +
**ThetaData** (official SPX cash — Databento does not sell the index), make **live
decisions on Databento + ThetaData**, and use **IBKR for execution + account +
safety only**. Financed as: prove on historical → subscribe live only to
paper-trade a backtest-proven model.

---

## 1. Vendor decisions

- **DROP `tcbbo`.** Not used anywhere as a real input (only an optional audit-slice
  alternative to `cbbo-1s`; project prefers `cmbp-1`/`cbbo-1s`). The $9–13k `get_cost`
  figure is irrelevant (and anomalous — likely a pricing quirk).
- **Databento OPRA Standard $199/mo:** subscribe ONE month, backfill, cancel (keep
  data), resubscribe for live at paper-trade time. Confirm data is retained after
  cancellation.
- **ThetaData (~$50/mo): KEEP — mandatory.** Supplies official SPX cash context
  (the `spx_*` features). Confirm ThetaData offers **live** SPX for the Path D
  decision plane (not just historical).
- **IBKR OPRA L1 $1.50/mo: KEEP** (fresh execution quote; waived at ≥$20 commissions).
- **IBKR CBOE Streaming Market Indexes $3.50/mo: CANCEL** (SPX/VIX index now from
  ThetaData; execution doesn't use it).

## 2. Download manifest (buy tonight; subscribe FIRST so L0 + trailing-12mo L1 are free)

Hard rule: **every pay-as-you-go purchase stops at 2025-08-01** — everything newer
is free under the subscription. Keep each schema a single contiguous range.

| Source | Dataset / schema | Range | Under $199 sub | Est. |
|---|---|---|---|---|
| Databento | OPRA `cmbp-1` (sub-minute exit substrate) | 2023-03-28 → now | pay older only | **~$767** |
| Databento | OPRA `cbbo-1s` (optional cross-check; `cmbp-1` is canonical) | 2025-02-20 → now | pay older only | ~$25 (or skip) |
| Databento | OPRA `cbbo-1m` (entry BBO) | 2022 → now | pay older only | ~$5 |
| Databento | OPRA `ohlcv-1m` (entry premium bars) | 2022 → now | **free (L0)** | $0 |
| Databento | OPRA `definition`, `statistics` | all | **free (L0)** | ~$0 |
| Databento | ES `GLBX.MDP3` `ohlcv-1m` (futures context) | 2022 → now | pay (not in OPRA sub) | ~$2 |
| Databento | VX `XCBF.PITCH` `ohlcv-1m` (vol context) | **2026-04 → now** (new dataset — VIX-proxy gap before) | pay | ~$8 |
| ThetaData | SPX cash (VWAP/momentum/moneyness context) | match option ranges | (ThetaData plan) | — |

**Total ≈ $199 + ~$807 ≈ ~$1,006**, and you keep the data. `cmbp-1` is the one big
line; consider skipping `cbbo-1s` (the 30-session owned pilot already certifies the
`cmbp-1`→1s downsampler per D59). Cost-estimate each request with `get_cost` before
pulling; hard-cap the run.

### 2a. STAGED OPTION — 12 months first (recommended for research)

The $199 subscription includes the **entire trailing 12 months of every schema for
free** (all L0 + all L1), so you can research on a full recent year before committing
the deep backfill:

| Item | 12-months-only (2025-08-01 → now) |
|---|---|
| OPRA `cmbp-1` / `cbbo-1s` / `cbbo-1m` / `ohlcv-1m` / `definition` / `statistics` | **FREE** under the $199 sub (~$463 of data value) |
| ES `ohlcv-1m` (12mo) | ~$0.36 pay-as-you-go |
| VX `ohlcv-1m` (only exists from ~2026-04) | ~$8.46 pay-as-you-go |
| **12-MONTHS-ONLY TOTAL** | **~$208** ( = $199 sub + ~$9 futures ) |

- **Deferred deep backfill** (older `cmbp-1` 2023-03-28→2025-08-01 ~$767 + older
  `cbbo-1s`/`cbbo-1m`/ES ~$30) ≈ **~$797** — **pay-as-you-go anytime, NO active
  subscription required** (metered historical works without a sub). Buy it only if
  the 12-month research shows the model warrants more data.
- **Path:** subscribe 1 month → grab the free 12mo (+ES/VX ~$9) → research → cancel →
  buy the ~$797 older backfill later if warranted → resubscribe for live at
  paper-trade time. So you pay $199 once now, $797 only if/when justified, and $199/mo
  again only at live deployment. **12-month research start = ~$208.**
- Caveat: VX only exists from ~2026-04, so the 12-month window has a VIX-proxy gap in
  its first ~8 months — flag it, don't silently patch.

### 2b. Coverage-table verification (Claude web-check, 2026-08-01)

The ChatGPT coverage table is **directionally right but UNVERIFIED on the costly part**;
Databento's exact per-schema/per-tier included-history is behind a JS pricing portal
not readable externally. Findings:
- **"L0/L1" in that table = a pricing tier, distinct from depth levels** (depth: L1=
  MBP-1/CMBP-1/CBBO, L2=MBP-10, L3=MBO). No authoritative public per-schema tier map found.
- **ES/VX separate from OPRA Standard — CONFIRMED.**
- **Table CONFIRMED (2026-08-01) via Databento's own source chain:** the pricing page
  states Standard = 16+ yr L0 / **1 yr L1** / 1 mo L2-L3; the schema doc classifies
  `cmbp-1`/`cbbo-1s`/`cbbo-1m`/`tcbbo` as **L1**. So ALL FOUR are trailing-12-months
  included, older pay-as-you-go. (An earlier Claude guess that `cbbo-1m` was
  under-counted was WRONG — the "10 more years" announcement referred to catalog
  AVAILABILITY, not Standard INCLUSION.)
- **Therefore the ~$797 deep `cmbp-1` backfill IS genuinely pay-as-you-go** (not free);
  the ~$208 12-month plan stands.
- **Availability caveat:** `cmbp-1`/`cbbo-1s`/`tcbbo` do NOT exist before ~2023-02-28;
  `cbbo-1m` is the finest consolidated quote schema before that date.
- **Optional belt-and-suspenders:** after subscribing, `get_cost` on an older `cmbp-1`
  range confirms true entitlement ($0 vs price) — but the doc chain already settles it.

## 3. Data-use plan (get right to work once downloaded)

1. **Normalize + align** the Databento OPRA options with ThetaData SPX and ES/VX
   context into one canonical corpus — **contiguous and paired** (no orphan patches;
   note the two natural boundaries: `cbbo-1s` starts 2025-02-20, VX starts 2026-04).
   Sub-minute substrate = `cmbp-1` → derived 1-second (per D59; A7 Tier-T).
2. **In parallel:** finish the **1-second exit objective + architecture design**
   (the long pole — Claude writes / Codex reviews).
3. **Build the exit tensor + action-advantage labels** per that design on the 1s
   substrate, from **out-of-fold trajectories of the frozen entry model**.
4. **Backtest the Path D trader** (entry minute + learned 1s exit) → gate: does it
   show provable edge? Only then does Phase 2 (subscribe live, execution bridge,
   paper-trade) begin.
5. **(Optional, later) Entry re-enrichment:** Path D removes the cross-vendor mask,
   so the entry model *could* use richer Databento microstructure. Not required;
   entry works — prioritize the new exit component first.

## 4. Baked-in assumptions Path D changes (blast radius)

| Assumption (old) | Path D change |
|---|---|
| Decisions happen on **IBKR** live | Decisions happen on **Databento + ThetaData**; IBKR = execution only |
| **Microstructure-masked** feature contract (mask bid/ask/spread/size/vendor-greeks to survive cross-vendor) | Model plane is same-vendor → **mask no longer needed**; model may use real Databento microstructure. Discipline shifts to "each feed same-vendor train↔live" + self-computed greeks |
| **Recorder-first parity / paired-diff** apparatus (Databento↔IBKR feature parity) | Model-feature parity largely **dissolves**; the remaining parity concern is **execution reconciliation** (decide Databento / fill IBKR) — a narrower, one-time bridge |
| Fill law **A1** = next-minute executable ask/bid | A1 → **intent-driven bounded marketable-limit order state machine** (used identically across labels, replay, PnL, floor, live) |
| **Minute-only** cadence (§2.4, D24 floor) | **Minute entry + 1-second exit**; sub-minute exit tensor/labels (FT2-08/FT2-60); floor demoted to catastrophic backstop in the risk governor |
| **IBKR provides SPX/VIX** index live | **ThetaData provides SPX** (both historical + live); IBKR index feed cancelable |
| **Bank IBKR recordings** (Path A) as future training source | **Dropped**; recorder role shrinks to optional execution/parity validation |
| **Execution = thin adapter** | **Execution is a second causal policy** (marketable-limit + state machine + deterministic broker-facing risk governor) |
| Sub-minute acquisition deferred / cmbp-1 for "later" (D57/D59) | Sub-minute `cmbp-1` (2023-03-28→now) becomes the **active** exit substrate now |

## 5. What survives / is deprecated / is new

- **Survives:** the entry model + its 17 synchronized features (self-computed greeks
  etc.); the label families (FT2-04); the serial simulator; D48/D49; the convexity
  entry-gate fix (A6); A7 Tier-S/Tier-T; the parity-gate *principle* (now applied to
  execution reconciliation).
- **Deprecated under Path D:** the microstructure-mask *necessity*; the recorder-first
  Databento↔IBKR feature-parity program; the Path A IBKR-recording campaign; the
  minute-only floor cadence; IBKR as a decision/context source.
- **New:** the 1-second learned exit objective + architecture (FT2-60); the execution
  policy + order state machine (A1 rewrite); the deterministic broker-facing risk
  governor; ThetaData-live SPX dependency; Databento-live decision runtime.

## 6. Immediate next steps

1. Owner: subscribe (1 month), run the download manifest (cost-estimate + hard cap),
   cancel IBKR CBOE index, confirm ThetaData live SPX + retention-after-cancel.
2. Claude: draft the **1-second exit objective + architecture** (long pole).
3. Codex (agentic): **repo-wide Path D assumption blast-radius audit** — catalog every
   file/contract that assumes decide-on-IBKR, mask-required, minute-only cadence,
   recorder-first parity, or IBKR SPX/VIX, so the full change set is known before any
   authority amendment. (Read-only; no edits.)
