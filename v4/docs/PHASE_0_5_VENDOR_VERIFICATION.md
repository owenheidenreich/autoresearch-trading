# Phase 0.5 — Vendor Verification

**Goal**: get to a non-provisional Phase-1 budget. The earlier draft of the protocol made factual claims about vendors that turned out to be wrong on independent verification (Cboe 0.2%, Databento no Greeks, OptionsDX cutoff at 2023, VIX requires 23–37 DTE). This pass closes the remaining gaps.

**Spend ceiling for Phase 0.5**: ≤ $250 total — the cost of a one-cycle OptionsDepth subscription, which doubles as both the verification and the dealer-flow archive (protocol Section 3.4).

---

## What's already known from public sources

### OptionsDepth (verified from their pricing page, homepage, and T&C, 2026-04-27)

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

### Databento

Open https://databento.com/pricing, run their estimator with:

> Dataset `OPRA.PILLAR`, symbols `SPXW` (parent), schemas `trades` + `cbbo-1m` + `definitions` + `statistics`, date range 2018-01-01 → present.

Record the quoted dollar number into the protocol Section 3.3 budget. If much higher than $500, trim the start date to 2022-04 (SPXW Tuesday/Thursday launch) and re-quote — pre-2022 data is stress-test only per Phase 2A regime cohorts.

### IBKR

Run `python -m v4.scripts.ibkr_preflight --strikes-around-atm 10 --duration-seconds 300` with paper TWS on port 7497. The script's contract-discovery logic is currently commented out — you'll need to wire ~10 lines using your account's chain-discovery calls (example commented in the file). Output is JSONL audit + summary.

### VIX

Default: buy direct from CBOE/Databento. Reconstruction from 23–37 DTE strips is a research project of its own (strike selection, zero-bid rules, interpolation) and not the v4 edge. Reserve reconstruction as a one-off Phase-1 validation exercise.

---

## Phase 0.5 exit criteria

- [ ] OptionsDepth one-cycle complete; empirical answers to Q1/Q2/Q4/Q6 recorded; Q3 contacted (any reply or no reply documented)
- [ ] Databento estimator quoted total recorded; replaces provisional `$200–500`
- [ ] IBKR preflight runs end-to-end; line budget + pacing-violation count recorded
- [ ] VIX path chosen
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
