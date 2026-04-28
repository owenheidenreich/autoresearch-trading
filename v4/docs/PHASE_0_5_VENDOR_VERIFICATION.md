# Phase 0.5 — Vendor Verification

**Goal**: get to a non-provisional Phase-1 budget. The earlier draft of the protocol made factual claims about vendors that turned out to be wrong on independent verification (Cboe 0.2%, Databento no Greeks, OptionsDX cutoff at 2023, VIX requires 23–37 DTE). This pass closes the remaining gaps.

**Spend ceiling for Phase 0.5**: ≤ $250 total — the cost of a one-cycle OptionsDepth subscription, which doubles as both the verification and the dealer-flow archive (protocol Section 3.4).

---

## What's already known from public sources

### OptionsDepth — DEFERRED to Phase 2B (2026-04-27)

OptionsDepth is for Block 2 (dealer flow). Block 2 is added only if Block 1 (microstructure) clears its Phase-2A gate. Buying the one-cycle now would be the "burn before edge" anti-pattern: if Block 1 fails, the $250 is dead money.

**Phase 0.5 OptionsDepth verification deferred** — re-engaged when Phase 2A delivers a Block-1 model with adversarial PF > 1.20 and LCB > 1.05 on the Databento backfill. At that point the seven public-source items below are still valid pre-purchase research; the one-cycle subscription becomes the verification + Block-2 test in one $250.

Public-source status (kept here for Phase-2B reference):

| # | Question | Public status | Source |
|---|---|---|---|
| 1 | Pro Max ($249/mo) — historical intraday snapshots exportable in machine-readable form? | **Partial** — their nav has an "API Units" page (currently 404) and they sell "raw data packages for professionals and quants" (Gamma, Charm, Positional datasets). API exists; format/scope unknown. | [optionsdepth.com](https://optionsdepth.com) |
| 2 | Exported data may be retained locally after cancellation? | **Not addressed** in T&C. Gray area. | [terms-conditions](https://www.optionsdepth.com/terms-conditions) |
| 3 | Exported data may be used to train private models (no redistribution)? | **Not addressed** in T&C; redistribution explicitly prohibited but model training is silent. | T&C |
| 4 | Historical fields point-in-time, not retrospectively revised? | **Unknown** — needs empirical check on the one-cycle archive. | — |
| 5 | Each snapshot has precise timestamp + 10-min refresh? | **✅ Confirmed**: "Updates every 10 minutes intraday" at Pro Max. | pricing page |
| 6 | Field definitions stable + documented (data dictionary)? | **Unknown** — no public dictionary. | — |
| 7 | SPX and VIX both included at Pro Max? | **✅ Confirmed**: "platform currently focuses on SPX and VIX options." | homepage |

The **only genuinely unknown** items are 1 (API-export specifics), 2 (retention), 3 (model-training), 4 (point-in-time integrity), and 6 (data dictionary).

### Databento (verified from their docs in earlier conversation rounds)

- ✅ OPRA history covers 2018+ (May-2025 expansion).
- ✅ Does NOT provide pre-calculated Greeks/IV; we compute via Black-Scholes (already built in `v4/greeks/`).
- 🟡 Cost is usage-based, billed per uncompressed GB. Provisional `$200–500` placeholder; exact number requires running their estimator with our query.

### IBKR (verified from their docs)

- ✅ Per-contract subscriptions required for option Greeks via `tickOptionComputation` (no chain-streaming).
- ✅ Initial 100 concurrent market-data lines.
- 🟡 Pacing/Greek-update cadence under our specific universe needs empirical preflight.

---

## How to close the remaining gaps

### OptionsDepth

The cheapest verification is **subscribe for one cycle, test empirically, decide before renewal**. $250 buys both the verification and the dealer-flow archive that protocol Section 3.4 wants anyway. Process:

1. Subscribe to Pro Max for one month.
2. On day 1: test the API — pull a historical intraday snapshot, confirm format (CSV/JSON/Parquet), confirm field set.
3. Throughout the month: archive every daily snapshot (and every intraday snapshot if the API permits).
4. On day 28: pull the same historical snapshot again from the API; compare to the version pulled on day 1. **Identical** → point-in-time integrity confirmed (Q4). **Different** → revisions exist; downgrade dealer flow to "noisy regime classifier."
5. Cancel before renewal.
6. After cancellation: try to re-pull yesterday's snapshot via the API. **Works** → free historical access; **fails** → archived data is what you have.
7. If anything is genuinely ambiguous (especially Q3 model-training rights), one short note via X DM or contact@optionsdepth.com is fine. Keep it to ~3 sentences. Don't write a 7-paragraph compliance email.

Expected outcome: by the end of the month you'll have empirical answers to Q1, Q2, Q4, and Q6, and a written reply (or no reply, which is also data) on Q3.

### Databento — DECIDED 2026-04-27

**Final query (recorded)**:

- Dataset: `OPRA.PILLAR`
- Symbols: `SPXW`
- Schemas: `CBBO-1m`, `statistics`, `definitions`
- Date range: 2022-05-11 (Thursday SPXW launch — full-weekday 0DTE coverage achieved) → present
- **Quoted total: $1,007.79** (CBBO-1m $844.40 + statistics $138.37 + definitions $25.42)

**Trades schema dropped**. Original quote was $8,427.50 making the total >$10k. CBBO-1m already includes last-trade-price-and-size aggregated to each minute, which is ~95% of what trades would have given the model at 1-min decision tempo. What we lose:

- Per-trade signed flow (Lee-Ready DIY proxy) — only 60–64% accurate per Grauer et al. anyway, already known to be a weak proxy
- Sweep detection — Phase 2A+ work, not Phase 1

This decision flows through the protocol: Block 1 microstructure features that previously assumed `trades` are recomputed against CBBO-1m's last-trade fields. DIY dealer-flow proxy quality is reduced (further weakening the case for trying to replicate OptionsDepth's signal in-house).

### IBKR

Run `python -m v4.scripts.ibkr_preflight --strikes-around-atm 10 --duration-seconds 300` with paper TWS on port 7497. The script's contract-discovery logic is currently commented out — you'll need to wire ~10 lines using your account's chain-discovery calls (example commented in the file). Output is JSONL audit + summary.

### VIX — DECIDED 2026-04-27

**Final query (recorded)**:

- Dataset: OPRA (per Databento estimator)
- Symbol: VIX
- Schema: OHLCV-1m
- Date range: 2022-05-11 → present
- **Quoted total: $63.96**

Reconstruction from 23–37 DTE SPX strips skipped — too cheap to bother reconstructing.

---

## Phase 0.5 exit criteria

- [x] OptionsDepth: **deferred to Phase 2B** (Block-2 dealer-flow test, only if Block 1 clears Phase-2A gate)
- [x] Databento estimator quoted total recorded: **$1,007.79** for SPXW CBBO-1m + statistics + definitions, 2022-05-11 → present (trades schema dropped — was $8,427.50)
- [ ] IBKR preflight runs end-to-end; line budget + pacing-violation count recorded
- [x] VIX path chosen: buy direct via Databento OHLCV-1m, $63.96
- [ ] Research-ledger entry written for Phase 0.5

Then Phase 1 historical pulls authorized. The OptionsDepth one-cycle has already happened by then (it WAS the verification), so Phase 1 starts with that archive in hand.

---

## What if OptionsDepth disqualifies itself?

If during the one-cycle test the API doesn't actually export, or fields turn out to be revised after-the-fact, or model-training is explicitly forbidden:

1. Record the disqualification in research-ledger.
2. Re-architect Phase 2 around DIY-only dealer flow (Lee-Ready 60–64% accuracy per Grauer et al. — known to be weaker than exchange-tagged).
3. Revise protocol Section 4.2 to mark Block 2 as DIY-only, recurring spend = $0/mo.
4. Keep the archived month's data; it's still useful as a one-time research artifact even if recurring isn't viable.

The $250 is not wasted in this case — it bought the answer.
