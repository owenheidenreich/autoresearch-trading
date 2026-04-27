# Phase 0.5 — Vendor Verification Checklist

**Goal**: get written answers from each vendor before any Phase-1 spend. The earlier draft of the protocol made factual claims about Databento, OptionsDepth, and OptionsDX that turned out to be wrong on independent verification — Cboe's "0.2% of SPX liquidity" finding, Databento not providing Greeks, OptionsDX coverage stopping at 2023, and VIX requiring 23–37 DTE strips. Avoid that mistake here.

**Spend**: $0 (just emails and reading).

**Gate**: written confirmation from each vendor on their side of the data contract, AND a non-provisional Phase-1 budget. Until this is green, **do not spend a dollar on Phase-1 historical pulls**.

---

## Status checklist

Update this file as answers come in. Each row is one vendor question; status flips from ⬜ → ✅ on confirmation, ❌ on disqualifying answer.

### OptionsDepth (most consequential — gates the entire dealer-flow track)

Draft email at [vendor_outreach/optionsdepth_email.md](vendor_outreach/optionsdepth_email.md). Send to whichever address their pricing/contact page lists. Each row corresponds to the 7 items in protocol Section 3.2 / 4.2.

| # | Question | Status | Their answer |
|---|---|---|---|
| 1 | Pro Max ($249/mo) — historical intraday snapshots exportable in machine-readable form? | ⬜ | |
| 2 | Exported data may be retained locally after cancellation? | ⬜ | |
| 3 | Exported data may be used to train private models (no redistribution)? | ⬜ | |
| 4 | Historical fields are point-in-time, not retrospectively revised? | ⬜ | |
| 5 | Each snapshot has a precise timestamp + known refresh cadence (10-min documented)? | ⬜ | |
| 6 | Field definitions stable + documented (data dictionary available)? | ⬜ | |
| 7 | SPX and VIX coverage both included at Pro Max tier? | ⬜ | |

**Disqualifying answers**: any "no" or "we don't allow that" makes OptionsDepth unusable as a Feature-layer source. Dealer flow becomes a manual-dashboard input only and the Phase-2B edge test cannot run as designed.

**Decision after answers**:
- All 7 ✅ → proceed to Phase 1 with OptionsDepth one-cycle ($250) budgeted
- Any ❌ → re-architect Phase 2 around DIY-only dealer flow (Lee-Ready 60-64% accuracy, per Grauer et al.) and document the loss in research-ledger

---

### Databento

Run their pricing estimator at https://databento.com/pricing for the exact pull. Query plan: [vendor_outreach/databento_query_plan.md](vendor_outreach/databento_query_plan.md).

| # | Question | Status | Result |
|---|---|---|---|
| 1 | Estimator total for SPX 0DTE (NBBO + trades + statistics + definitions), 2018–present | ⬜ | $___ |
| 2 | Estimator total for 23–37 DTE SPX strips (for VIX reconstruction), 2018–present | ⬜ | $___ |
| 3 | Confirm Greeks/IV are NOT provided (per their docs) → must compute via BS | ⬜ | (already verified in Phase 0) |
| 4 | Symbology coverage: SPXW (modern daily 0DTE) confirmed? | ⬜ | |
| 5 | Pricing-estimator quote does NOT auto-charge; explicit purchase required | ⬜ | |

**Decision after answers**:
- Provisional $200–500 holds → proceed
- Cost much higher than provisional → reduce date range (start at 2022-04 SPXW launch instead of 2018) and re-quote

---

### IBKR

No vendor email; this is a preflight against your own paper-account TWS API. Run script: [../scripts/ibkr_preflight.py](../scripts/ibkr_preflight.py).

| # | Question | Status | Result |
|---|---|---|---|
| 1 | TWS API connects from this machine | ⬜ | |
| 2 | OPRA + CBOE indices market-data subscriptions are active on the paper account | ⬜ | |
| 3 | Real-time NBBO + Greeks streamable for 50 SPX 0DTE contracts simultaneously | ⬜ | |
| 4 | No pacing-violation messages during 5-minute steady-state stream | ⬜ | |
| 5 | Greeks update at acceptable cadence during high-vol period (FOMC announcement / ES 1% move) | ⬜ | (defer until next FOMC if needed) |
| 6 | Quote stream during paper-trade matches the stream during paper-fill | ⬜ | |

**Decision after answers**:
- All ✅ → IBKR is the deploy-time data feed; document line budget in DATA_CONTRACT.md
- Pacing/line issues → two-stage scanner architecture required (cheap snapshot pass + narrow live stream); document in protocol before Phase 2A

---

### VIX (decision, not a vendor question)

Choose one:

- [ ] Buy VIX intraday history directly from Databento or CBOE (default per protocol)
- [ ] Reconstruct VIX from 23–37 DTE SPX strips (research target only)

**Recommended**: buy direct. Reconstruction is a research project of its own (strike selection, zero-bid rules, interpolation, methodology). Reserve reconstruction as a one-off Phase-1 validation exercise to confirm computed VIX matches Cboe's published value.

---

## Phase 0.5 exit criteria

Phase 0.5 is complete when:

- [ ] OptionsDepth: all 7 items answered in writing (✅ or ❌)
- [ ] Databento: pricing estimator returns a concrete number for the planned pull
- [ ] IBKR: preflight script runs end-to-end without errors and reports the line budget
- [ ] VIX strategy decided
- [ ] Phase-1 budget is non-provisional (replaces "$200–500" estimates with vendor-quoted numbers)
- [ ] Research-ledger entry for Phase 0.5 written, linking the answers + decisions

Then proceed to Phase 1 historical pulls.

---

## What if a vendor disqualifies?

Record the disqualification in the research ledger and choose:
- **Re-architect**: drop dealer flow from v4 (deploy Block-1 only forever); revise protocol Section 4.2.
- **Find alternate**: SpotGamma is the obvious-but-rejected alternative; not bot-friendly and not recommended. Bloomberg/institutional desks are above retail budget. There is no third option.
- **Stop**: accept that the v4 thesis cannot be tested as designed; document and pause the project.

The point of Phase 0.5 is to discover this *before* spending. If the answer comes back disqualifying, that is the protocol working as intended — not a failure.
