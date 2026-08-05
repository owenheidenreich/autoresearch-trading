# Codex Handoff — Options-Chain Direction Screen (2026-08-04)

**Owner decision: the target is a 0DTE SPXW long-call / long-put bot.** Everything below serves that.
**Answer §5 before writing code.**

---

## 1. The wall

**−$13.00/trade gross with ALL friction removed, negative 5/5 folds** (−18.59/−14.00/−11.63/−9.26/−11.33)
over 156,950 OOF candidates. Buying 0DTE premium at minute cadence loses to theta before any cost.
Source: [`PROTOCOL101_PRIOR_CAMPAIGN_DISTILLATION.md`](../history/PROTOCOL101_PRIOR_CAMPAIGN_DISTILLATION.md)
§4, row "Path-D 0DTE long-premium class".

Premium scaling does not fix it: at $1,500 premium the friction ratio improves 4.68% → 1.77%, but gross
scales with premium, so the dollar loss worsens. **The option leverages edge; it does not create it.**

## 2. The door — declared by the ledger itself

That same §4 row names what reopens the class:

> *"...or **demonstrated directional skill on the underlying**."*

This handoff is not a request to overturn the closure. Establish direction on SPX over a holdable horizon
and the 0DTE long option becomes the correct expression of it, reopening the class by its own rule.

## 3. What has never been tried

**Every feature this project has used is SPX price-derived** — `spx_vwap_gap_*`, `momentum_*`,
`session_range_bps`, `omar_clipped_neg3_pos3`
([`protocol101_canonical_stage1_contract.py:31`](../../../../model/protocol101_canonical_stage1_contract.py#L31))
plus `E.bs.delta`/`E.bs.gamma` on the single traded contract. All are now closed: the nine side-free
context features carry no directional content on the index, and `omar` closed on raw economics.

**The option chain has never been used as a signal.** Owned and verified today:

| | |
|---|---|
| Path | `/Volumes/AR_TRADING_DATA/vendor/pathd_2025-08-01_2026-07-31/raw/databento/opra_spxw_cbbo_1m/` |
| Sessions | **251** (215 development prefix; **36 are the spent holdout — never open**) |
| Contracts/session | **500** distinct SPXW symbols |
| Rows/session | ~178,000 |
| Columns | `ts_event, rtype, publisher_id, instrument_id, side, price, size, flags, bid_px_00, ask_px_00, bid_sz_00, ask_sz_00, bid_pb_00, ask_pb_00, symbol` |

## 4. ⚠ Most of the obvious signals are BARRED — resolve before building

I proposed four families to the owner. **Two are blocked, and I am correcting that here rather than letting
you discover it.**

`QUARANTINED_ALPHA_TOKENS`
([`protocol101_canonical_stage1_contract.py:63`](../../../../model/protocol101_canonical_stage1_contract.py#L63)),
enforced by `assert_model_alpha_firewall`, contains: `bid_size`, `ask_size`, `volume`, `open_interest`,
`raw_bid`, `raw_ask`, `raw_spread`, `quote_age`, `E.bs.iv`, `vendor_greek`.

The admission ledger
(`v4/audit/autoresearch/pathd_phase0_feature_certification_2026_08_04/feature_admission_ledger.json`)
marks **BARRED**: `intraday_open_interest`, `last_causal_open_interest`, `last_causal_minute_volume`,
`option_trade_volume_1m`.

| Family | Status |
|---|---|
| **Dealer gamma exposure (GEX)** | **DEAD.** Classic GEX needs open interest; OI is BARRED *permanently* — no live source with matching decision-time semantics. Do not propose an OI-based construction. |
| **Signed flow from trade prints** | **BLOCKED** — `volume` and `option_trade_volume_1m` are barred. |
| **Quote-size imbalance** | **BLOCKED** — `bid_size`/`ask_size` are quarantined tokens. |
| **IV-surface geometry** | **AVAILABLE.** Uses only `bid_px_00`/`ask_px_00` — the same inputs the certified path already consumes. |

**What survives is price-derived chain geometry**, and it is narrower than I told the owner:

- **Skew** — put IV minus call IV at equidistant strikes (directional, not a level).
- **Smile curvature** across the ladder.
- **Per-strike self-computed gamma**, aggregated across the chain by strike distance — a positioning
  *proxy* that uses no OI.
- **Put/call mid-price relationships** and put-call-parity implied spot vs the index.

Note `E.bs.iv` is quarantined as a *feature* while `E.bs.delta`/`E.bs.gamma` are in `FEATURE_NAMES` — IV is
lawful as an intermediate, not as model-facing alpha. **Whether skew (an IV-derived construction) is
model-facing alpha or an intermediate is an open ruling. Get it decided before fitting anything.**

## 5. Questions to answer BEFORE code

1. **Is the firewall reading in §4 correct?** Verify `assert_model_alpha_firewall` scope
   ([`protocol101_regimen_repair.py`](../../../../model/protocol101_regimen_repair.py)) and whether the
   BARRED statuses are live-parity rulings that a *research screen* on historical data must also honour, or
   only bind the trained feature contract. **This determines whether the screen is legal as designed.**
2. **Is skew model-facing alpha or an intermediate?** Cite the contract, do not infer.
3. **What is the frozen signal list?** Declared count **is** the maxT family. Keep it small.
4. **Horizon and bar.** Which holding horizons can a 0DTE long option actually survive, given the horizon
   sweep showed big-loss share rising 23%→69% from 5 to 240 minutes
   ([`PATHD_HORIZON_VS_CHARTER_SWEEP_2026_08_04.md`](../research/PATHD_HORIZON_VS_CHARTER_SWEEP_2026_08_04.md))?
5. **Where does this fail?** Adversarially.

## 6. The task, once §5 is settled

**A raw-economics screen: does the option chain predict SPX forward moves?**

Measured **in SPX points on raw data — no null, no surrogate, no model, no governance doc.** That is the
lesson that cost the most: the number that closed `omar` was computable from owned data all along while
three studies argued about an information coefficient.

Only if a signal clears does the 0DTE wrapper follow, and only then is a null (with a known-answer gate)
warranted.

## 7. Reuse — do not reimplement

| Piece | Source |
|---|---|
| Dev session list (215) | `development_sessions` — [`pathd_phase1_entry.py:67`](../../../../research/pathd_phase1_entry.py#L67) |
| Official SPX bars + provenance asserts | `_official_spx` — [`pathd_phase1_entry.py:204`](../../../../research/pathd_phase1_entry.py#L204) |
| OSI symbol → right, strike | `_OSI_RE` / `_parse_symbol` — [`pathd_phase1_entry.py:58`](../../../../research/pathd_phase1_entry.py#L58) |
| Implied vol from mid | `_implied_vol` — [`protocol101_canonical_stage1_contract.py:118`](../../../../model/protocol101_canonical_stage1_contract.py#L118) |
| Self-computed greeks | `_bs_delta_gamma` — [`protocol101_canonical_stage1_contract.py:144`](../../../../model/protocol101_canonical_stage1_contract.py#L144) |
| Availability clock, boundary grid, parity assert | [`pathd_spx_directional_skill_screen.py`](../../../../research/pathd_spx_directional_skill_screen.py) — reuse `_availability_series`, `_close_available_at`, `_boundaries_ns` |
| Folds | `entry_expanding_folds` — [`pathd_phase1_entry.py:596`](../../../../research/pathd_phase1_entry.py#L596) |
| Stats | `paired_summary`, `session_blocked_max_t` — [`autoresearch_v2/statistics.py`](../../../../research/autoresearch_v2/statistics.py) |
| Prior art, waves | `prior_art_check`, `WaveSpec`, `run_wave` — [`pathd_research_loop.py:161`](../../../../research/pathd_research_loop.py#L161) |
| Rejection tests | [`pathd_model_gate.py`](../../../../research/pathd_model_gate.py) |

**Clock gotcha, verified today:** the corpus stores `event_time` as `datetime64[us]`. An unqualified
`.astype("int64")` yields **microseconds** — a 1000× error. Pin the unit
([`pathd_spx_directional_skill_screen.py:118`](../../../../research/pathd_spx_directional_skill_screen.py#L118)).

## 8. Corrections from 2026-08-04 — do not inherit these

From the superseded mechanism-first plan
([`PATHD_MECHANISM_FIRST_MASTER_PLAN_2026_08_04.md`](PATHD_MECHANISM_FIRST_MASTER_PLAN_2026_08_04.md) §3):

- **"VIX has never entered any model" is FALSE.** `vix_level`, `vix_change_5m`, `vix_change_15m`,
  `vix_minus_spx_rv15_pct_points` are in
  [`lean_autoresearch/harness.py`](../../../../research/lean_autoresearch/harness.py); that S4/S5 campaign
  returned `NULL_NO_NEW_ENTRY_EDGE`.
- **The 10:01 boundary is a feature-warmup constraint**, not an arbitrary exclusion — `history_minutes=30`
  and `momentum_15m` needs 15 prior bars.
- **Owned index data is 251 official SPX / 251 official VIX**, not 505/338; the remainder are proxy files
  `_official_spx` rejects.
- **ES OHLCV is RTH-only** — 390 rows, 09:30–15:59, no overnight bars.

## 9. Hard stops

- **Protected holdout SPENT** (`holdout_open_count: 1`, opened 2026-08-02 for `signed18`, invalidated).
  Never reopen. There is **no confirmation firewall** — the only forward evidence is live paper.
- **Causal clock t−60s.** Verify feature-availability-clock parity, not just future-outcome guards; the
  mutate-future lint did not catch the `signed18` leak, the runtime-parity gate did.
- **Do not modify** `FILL_LAW`, the causal clock, the label law, or the OOF firewall.
- **Closed:** the nine side-free SPX context features; `omar` as a tradable directional signal; stitched ES
  VWAP as a feature (Protocol 028). **Run `prior_art_check` on every proposal** — it blocks on a
  do-not-retest row *or* a rejecting verdict anywhere in the protocol decoder.
- **No paid data, no broker, no promotion, no launchd/runtime mutation** without owner authorization.
- **Calibration:** honest best is entry PF **1.132**, composed **1.7–1.95**, both pre-parity. The one result
  that beat it was a 60-second look-ahead leak. Anything materially above is a bug until proven otherwise.
- `NO_EDGE` is a first-class successful outcome. Do not widen anything to reach a pass.

**Deliverable:** answer §5 in writing, then `STOP_FOR_CLAUDE_VERIFICATION`.

*Prepared by Claude Opus 5 — 2026-08-04.*
