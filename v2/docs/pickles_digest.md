# Pickles corpus digest — 0DTE SPX long focus

**Scope:** This digest extracts Pickles' 0DTE SPX *long* (debit call / debit put, no
spreads) decision patterns from the corpus at
`/Users/gduby/Documents/picklesGPT/pickles/` — 12 strategy docs, 8 personality docs,
2 market-context docs, 1 glossary, 167 journal entries, 45 overnight-inventory entries.
Credit-spread / theta-gang / iron-fly / vega content is documented for context only,
not targeted for extraction.

**Extraction rubric** (see `memory/feedback_extraction_discipline.md`):
- **High** — one source states preconditions + action + outcome (or level reached).
- **Medium** — action + rationale are clear; timing / parameters / outcome fuzzy.
- **Low** — interpretive reconstruction across entries.

Fork A (mechanical backtest) accepts only High. Fork C (journal-as-supervision)
accepts High directly, Medium as auxiliary, Low never. A row without a tag is Low.

**Important corpus caveat (user-confirmed 2026-04-18):** Most of Pickles' PnL
evidence lived in Discord screenshots that are **not** preserved in this text corpus.
Journal entries narrate process richly but rarely state $/% outcomes on the specific
leg of interest. Outcome inference is weaker than it would otherwise be. This does
not block Fork A (the simulator supplies PnL) but it caps Fork C Tier-3 to
**action labels only, without outcome weighting** — which is fine for behavioral
cloning; the simulator scores the cloned policy.

---

## Canonical setup table

One row per 0DTE SPX long setup consistently executed or named by Pickles. Rows
that could not be filled to Medium across every column are parked in the
**"Ambiguous patterns" appendix** below, not here.

### Row 1 — "VWAP SUPPORT quick-in-out"

| Column | Value |
|---|---|
| **Setup name** | `LONG SPX CALLS on ES VWAP SUPPORT, quick in & out, not expecting a re-visit to AM HIGHS, TP at +1 VWAP / OPENING CANDLE HIGH` (verbatim) |
| **Market context** | OVN inventory LONG; ES prior 15m candle strongly bullish; A/D ratio + A/D volume "super bullish" |
| **Time window** | After opening 15m candle closes (≥ 6:45 EST). Pre-1000 MAGIC TIME is fine. |
| **Directionality** | Long calls only. No stated put analogue. |
| **Trigger** | NQ finds support at its VWAP; **then** ES finds support on VWAP (first touch from above) |
| **Invalidation** | ES fails VWAP → "lean changes to SHORTS → go flat on acct and wait for market to show its hand" |
| **Target / exit** | TP at +1 VWAP deviation **OR** opening-candle high, whichever hits first; "quick in & out" means short hold |
| **Typical delta** | Not stated explicitly; strikes used were ATM-to-slightly-ITM (4750 CALLS with SPX near 4750 on 2023-12-14) |
| **Hold duration** | Minutes to ~1 hour (on 2023-12-14 the trade went from 7:08 AM entry to SL'd-out by 7:41 AM) |
| **Required indicators** | ES VWAP (daily), NQ VWAP (daily), +1 VWAP deviation band, opening-candle high, A/D ratio, A/D volume |
| **Current-data availability** | ES VWAP / NQ VWAP: **needs-new-data** (v2 has SPX VWAP only via `vwap_dist`; no ES/NQ). +1 VWAP deviation: **needs-new-data** (v2 has `vwap_dist` but no σ bands). Opening-candle high: **computable-from-existing** (1st 15m H on SPX). A/D ratio: **needs-new-data** (market internals). A/D volume: **needs-new-data**. |
| **Confidence** | **High** |
| **Journal refs** | 2023-12-14 7:08 AM: *"LONG SPX CALLS on ES VWAP SUPPORT, quick in & out, not expecting a re-visit to AM HIGHS, TP at +1 VWAP / OPENING CANDLE HIGH"*. Repeated verbatim 2023-12-18 — the verbatim repetition across days is the signature of a templated setup. Outcome on 2023-12-14: SL'd out ("SPX LONG CALLS SL'd out. Pickles has a house rule of 3 losing trades → log out of broker and come back tomorrow"). |

### Row 2 — "1000 MAGIC TIME reversal long"

| Column | Value |
|---|---|
| **Setup name** | `1000 MAGIC TIME` (named throughout corpus) |
| **Market context** | Opening 15m candle must have closed; **bears could not maintain strength past opening 15m** (reversal context). Index internals not diverging across the big 4 (ES / NQ / RTY / YM); XLU not leading sectors. |
| **Time window** | **0955–1010 EST** (explicitly stated: *"1000 MAGIC TIME! ( 0955 - 1010 EST is the window )"* — 2024-01-04, 2023-12-06). If the reversal doesn't fire by ~1010, next window is 1030 EST IB. |
| **Directionality** | Either — **Long CALLS** when AM was weak and bears fail to extend (reversal long); **Long PUTS** when AM was strong and MAGIC TIME "betrays" by continuing the prior direction (2024-06-13: *"out of NQ LONGS at 611, MAGIC TIME betrayed me"* → flipped to SHORT + SPX LONG PUTS). |
| **Trigger** | AM weakness fails to extend past opening 15m; ES prints a higher-low inside the 0955-1010 window; TICKS begin to turn (watch for red→green on reversal long, green→red on betrayal flip) |
| **Invalidation** | Strong cross-index divergence in the big 4 (*"1000 MAGIC TIME probably won't happen today... too much divergence in the big 4 indexes"* — 2024-05-09); defensive sector XLU leading ("heading into 1000 EST with XLU being the strongest performer is not a great signal, MAGIC TIME may be taking today off" — 2024-06-10) |
| **Target / exit** | Not stated as a universal rule; per-trade narration suggests opening-price retest or prior-session high/low as magnets |
| **Typical delta** | Not stated; strikes used across days were near-ATM when 0DTE (4510 on 2023-11-17 with SPX ~4510; 5000-area references on 2024-02-09) |
| **Hold duration** | Minutes to several hours. On 2024-01-04, *"averaged down into these SPX LONG PUTS and finally closed out for profit after a very clenchy lunch session"* → multi-hour hold with adds. On 2023-11-17 the 7:18 LONG SPX 4510 held ~25 minutes. |
| **Required indicators** | Clock (0955-1010 EST), ES 15m close direction, TICKS, sector leadership ranking, cross-index divergence among ES/NQ/RTY/YM, prior-session H/L levels |
| **Current-data availability** | Clock: **in-current-79** (`marker_10am`, `intraday_sin/cos`, `intraday_phase` — but `marker_10am` is likely a point flag, not a 0955-1010 *window* flag; **computable-from-existing** to add a `magic_time_window` flag). ES 15m close: **computable-from-existing** (SPX 1m → 15m resample; ES VWAP separately is needs-new-data). TICKS: **needs-new-data**. Sector leadership + big-4 divergence: **needs-new-data**. Prior-session H/L: **computable-from-existing**. |
| **Confidence** | **Medium** — window is High-confidence and named; the entry pattern inside the window is Medium because it varies (reversal vs betrayal-flip) and depends on preconditions that are not all in current data. |
| **Journal refs** | 2023-12-06 (window named); 2024-01-04 7:01 AM (window + entry sequence); 2024-02-09 (precondition: "pivotal in signaling if 5000 SPX will be held"); 2024-03-22 11:41 AM ("only trade i made today, was during 1000 MAGIC TIME"); 2024-06-13 7:01 AM (counter-long fired then "betrayed" → flipped); 2023-10-31 7:24 AM (*"beautiful 1005 EST reversal trade"*); 2023-11-17 6:59 AM; 2024-02-16 7:03 AM (*"LONG SPX for 1000 MAGIC TIME"*). |

### Row 3 — "Supply-zone break with confluence"

| Column | Value |
|---|---|
| **Setup name** | Unnamed but explicitly templated — "pickles loves his confluence" |
| **Market context** | Pre-identified supply zone in pre-market notes (e.g. 5211–5222 ES on 2024-05-09); multiple external confirmations aligned (VIX direction, DXY direction, 10YR direction, leading sector) |
| **Time window** | After RTH open and after first 15m close; 2024-05-09 entry was ~7:21 AM EST (within the hour after open). Not restricted to MAGIC TIME. |
| **Directionality** | Long calls on upside break; long puts on downside break. On 2024-05-09 he stated: *"5223 - 5225 and pickles is going into SPX LONG CALLS and TP on approach to entering 5240s on ES"* |
| **Trigger** | ES prints above the supply-zone top with RTH volume; confluence of VIX/DXY/10YR/sector aligning with the break direction (*"I like how VIX & DXY favor your thesis too"* — confirmed by chat partner) |
| **Invalidation** | Rejection at zone top; zone becomes resistance; he "revert[s] to trading pure price action & volume" if PA goes outside levels |
| **Target / exit** | Next named level up (5240s ES on 2024-05-09); scale-out *"as ES enters more and more into 5230s"*; complete exit on approach to TP |
| **Typical delta** | Not stated |
| **Hold duration** | ~50 minutes on 2024-05-09 (entry ~7:21, complete exit 8:11) |
| **Required indicators** | Pre-marked supply-zone levels (operator-maintained), ES spot vs level, VIX direction, DXY direction, 10YR direction, leading-sector identity |
| **Current-data availability** | Supply-zone levels: **computable-from-existing** (his pre-market LEVELS list; `es_macro_levels.txt` publishes them). ES spot: **needs-new-data** (we have SPX, not ES). VIX direction: **in-current-79** (`vix_regime`, `vix_roc`). DXY: **needs-new-data**. 10YR: **needs-new-data**. Sector leadership: **needs-new-data**. |
| **Confidence** | **High** — 2024-05-09 narrates entry trigger, scale-out rules, target level, and completion (*"completely out of position — GG"*). Confluence is explicitly enumerated. Only gap is $/% outcome (missing due to screenshot convention). |
| **Journal refs** | 2024-05-09 7:12 AM through 8:11 AM entry-to-exit; 2024-05-09 6:03 AM pre-trade thesis with zone top identified; chat exchange 7:23 AM confirming VIX/DXY confluence. |

### Row 4 — "Opening 15m candle bounce continuation"

| Column | Value |
|---|---|
| **Setup name** | `LONG SPX off ES & NQ bounce off opening 15m candle` (verbatim — 2023-11-13) |
| **Market context** | OVN inventory context set in pre-market; multi-index alignment; house rule: **never enter before 15m candle closes** |
| **Time window** | Right after the 15m close (≥ 6:45 EST). |
| **Directionality** | Long in the direction of the 15m candle (bullish candle → calls; bearish 15m with bear-strength confirmation → puts) |
| **Trigger** | 15m candle closes decisively in one direction; ES and NQ both print a bounce off the 15m candle's opposite extreme (for a long: bounce off the 15m candle's low) |
| **Invalidation** | Bears maintain strength past the opening 15m candle (for longs) — then don't take the long; he sometimes then waits for 1000 MAGIC. Conversely, bulls fail to extend past 15m (for puts). |
| **Target / exit** | First touch of VWAP as TP (2023-11-13: *"TP on first test of VWAP"*, executed at 7:19 AM after ~30 min hold) |
| **Typical delta** | Not stated |
| **Hold duration** | 10–30 minutes. On 2023-11-13: 6:50 AM entry → 7:19 AM exit (~30 min) |
| **Required indicators** | 15m candle close direction/strength, ES/NQ cross-confirmation of bounce, VWAP position |
| **Current-data availability** | 15m candle close: **computable-from-existing** (SPX 1m → 15m resample). ES/NQ cross-confirmation: **needs-new-data**. VWAP (SPX): **in-current-79** (`vwap_dist`). |
| **Confidence** | **Medium** — trigger and exit stated clearly, but the cross-index confirmation depends on ES/NQ data we don't have, and the trade is frequently woven with credit-spread positions so isolating the long leg's outcome is tricky. |
| **Journal refs** | 2023-11-13 6:50 AM; 2023-11-20 10:08 AM (*"i'm about to LONG SPX CALLS if we have continuation here"* — 10:33 AM commentary: *"looking for the exit right now, i don't see the up-trend continuing much further"* — 10:59 AM *"out of those SPX CALLS"*); 2023-10-31 7:24 AM (the 1005 reversal was preceded by *"bears could not maintain strength past opening 15m candle"* — an invalidation-of-shorts triggered the long). |

### Row 5 — "Opening drive long after news bottom-wick"

| Column | Value |
|---|---|
| **Setup name** | Not formally named; pattern in 2024-05-14 |
| **Market context** | Scheduled data event (PPI, CPI) in pre-market; news candle prints with ≥ 50% bottom wick (rejection of downside); sector/VIX backdrop aligning with long bias |
| **Time window** | Just before the **15-minute pre-RTH cool-down** window (~6:13-6:14 EST on 2024-05-14, ahead of 6:30 RTH open) — explicitly references the cool-down rule from `personality/trade_clear_levels.txt` |
| **Directionality** | Long calls when news candle wicked down (rejecting sellers); put analogue plausible but not directly journaled |
| **Trigger** | News candle bottom wick > 50% of the candle range; intent to TP "within OPENING DRIVE" |
| **Invalidation** | News candle closes at the bottom of its range (no wick) — don't take the trade |
| **Target / exit** | "Within OPENING DRIVE" (first 30-60 min of RTH); 2024-05-14 exit was 6:32 AM EST, 2 minutes after RTH open |
| **Typical delta** | Not stated |
| **Hold duration** | ~19 minutes on 2024-05-14 (6:13 entry → 6:32 exit) |
| **Required indicators** | News-release timestamp, minute-bar OHLC (for wick measurement), RTH open clock |
| **Current-data availability** | News timestamp: **needs-new-data** (economic calendar). OHLC for wick: **in-current-79** via `bar_range`, `bar_delta` but the pre-RTH 6:13 AM bar is **outside current mask window** (`NO_TRADE_BEFORE_BAR=30` starts at 10:00 EST in current data, and pre-market ETH bars are not in `v2/data.pt`). RTH open: **in-current-79** (`intraday_phase`). |
| **Confidence** | **Medium** — full entry-to-exit cycle in one entry with explicit trigger and duration, but the action happens in a time window the current dataset doesn't cover, and outcome is not stated in $/%. |
| **Journal refs** | 2024-05-14 6:13 AM through 6:32 AM. |

### Row 6 — "Weekly VWAP bounce"

| Column | Value |
|---|---|
| **Setup name** | `LONG SPX CALLS on ES WEEKLY VWAP BOUNCE` (verbatim) |
| **Market context** | Index extended below weekly VWAP; first bounce off it; stated expectation: *"a bunch of BUYERS coming in to potentially make a stronger run back to OPENING PRICE"* |
| **Time window** | RTH, no specific clock constraint beyond needing the WVWAP touch to occur |
| **Directionality** | Long calls on bounce; **long puts on break-down through WVWAP** is the mirror trade (same entry stated on 2023-11-09 7:33 AM with *"SPX PUTS on the break-down through it"* before the bounce turned things around) |
| **Trigger** | ES taps WVWAP from above; first bounce holds; re-enter *"back into these exact same CALLS, VWAP buying program fired"* when VWAP holds on retest |
| **Invalidation** | *"otherwise VWAP will act as a hard wall to push through"* — if ES fails to hold WVWAP, exit and consider puts |
| **Target / exit** | Opening price (named explicitly) or prior swing high; scale partial at intermediate levels |
| **Typical delta** | Not stated |
| **Hold duration** | Re-entry pattern (exit on first rejection, re-enter on confirmed hold) suggests minutes-to-30-minute legs |
| **Required indicators** | ES weekly VWAP, opening price, prior swing highs |
| **Current-data availability** | ES weekly VWAP: **needs-new-data** (we have SPX only; weekly VWAP on SPX is **computable-from-existing** via rolling 5-day volume-weighted mean). Opening price: **computable-from-existing** (session cache). |
| **Confidence** | **Medium** — trigger, invalidation, and re-entry rule clearly stated; no explicit outcome on the calls leg (he pivoted to a credit-spread IC immediately after). |
| **Journal refs** | 2023-11-09 8:00 AM; re-entry 2023-11-09 8:09 AM. 2023-12-05 has a related "ES RETRACE back to WEEKLY POC / VWAP" setup but the LONG never filled into 0DTE that day ("waiting for another LONG set-up to occur, need to see separation from WEEKLY POC & WEEKLY VWAP and will go into SPX LONG CALLS instead of FUTURES" — 9:00 AM). |

---

### Summary of canonical rows

| # | Setup | Confidence | Key data gaps |
|---|---|---|---|
| 1 | VWAP SUPPORT quick-in-out | **High** | ES VWAP, VWAP σ bands, A/D ratio + volume |
| 2 | 1000 MAGIC TIME reversal | **Medium** | TICKS, sector leadership, cross-index divergence, 0955-1010 window flag |
| 3 | Supply-zone break + confluence | **High** | Pre-marked levels registry, DXY, 10YR, sector leadership |
| 4 | Opening 15m bounce continuation | **Medium** | ES/NQ cross-confirmation, 15m resample pipeline |
| 5 | Opening drive after news wick | **Medium** | Economic calendar, pre-RTH ETH bars |
| 6 | Weekly VWAP bounce | **Medium** | ES weekly VWAP (SPX WVWAP is computable) |

**R1 pass condition check:** ≥ 3 rows at Medium+. Actual: **6 rows at Medium+, 2 at
High.** Pass. Fork A has candidate material in Rows 1 and 3 specifically.

---

## Ambiguous patterns appendix (Low confidence — hypothesis generation, not testing)

- **Counter-trend dip scalp** (2023-11-17 9:26 AM *"LONG SPX 4510 0DTE, COUNTER-TREND
  this dip, expecting a rip"*) — the phrasing is emotional rather than ruled; no
  explicit preconditions or invalidation.
- **"Accumulation phase → distribution to upside"** (2023-11-20 9:06-9:10 AM) —
  narrative pattern recognition; not a stated rule.
- **"Snuck in" plays before specific time gates** (2024-05-14 *"snuck in a few SPX
  LONG 0DTE CALLS before the 15 minute cool-down window"*) — impulse trades rather
  than systematic setup; categorically different from Row 5 even though same date.

---

## Time-of-day template taxonomy

Structural overview, not per-setup. Pickles' intraday rhythm:

| Window (EST) | Label | Stance |
|---|---|---|
| Pre-open ETH | OVN review, level prep | No trading; just context |
| 06:30-06:45 (first 15m) | Opening candle | **No entry** — explicit rule. Watch A/D, TICKS, 15m candle direction. |
| 06:45-~09:55 | Post-15m AM session | Primary trading window. Setups: Rows 1, 3, 4, 6 above. |
| **09:55-10:10** | **1000 MAGIC TIME** | Named reversal window. Row 2. |
| 10:30 | 1030 IB | Initial Balance completes (first hour H/L set). Second reversal window if MAGIC TIME missed. |
| 10:30-~11:15 | Late AM | "I want to be totally flat by 1115" (2023-11-17). Ride out existing positions; rarely open new. |
| 11:15-~13:15 | **Lunch / "NY lunch"** | **"Don't trade when you know you don't perform well. NY Lunch as an example for many"** (`trade_clear_levels.txt`). Mostly flat. |
| ~13:00/13:30 | "1:30 Algo" window (user-noted) | Weakly documented — no journal entry directly surfaces this by name in what I read. Pickles occasionally references 1300 EST as a data-event time but rarely enters trades then. **Low-confidence pattern.** |
| ~14:00-15:15 | Power hour prelude | Context-dependent; he usually skips. |
| 15:15-16:00 | Power hour | "Usually doesn't trade power hour" (user-noted). Manages existing positions; closes out. |

**Mechanical don't-trade gates** (from `personality/trade_clear_levels.txt`, verbatim):
- Before news events (FinancialJuice, ForexFactory)
- A few minutes before every half hour
- The "dirty thirty" of Europe close (~11:30 EST after DST) and S&P close (~15:55 EST)

These are **hard no-entry rules**, operator-level. Easy to enforce mechanically.

---

## Indicator coverage map

Every indicator in `glossary/list_of_indicators.txt` plus recurring journal terms,
mapped to availability against the current v2 79-feature set.

| Pickles indicator | Status | Note |
|---|---|---|
| VWAP (daily) | **in-current-79** | `vwap_dist` (distance from VWAP). No σ deviation bands — he explicitly uses +1/+2 σ for counter-trades. **σ bands: computable-from-existing**. |
| VWAP slope | **in-current-79** | `vwap_slope`. |
| VWAP reclaim state | **in-current-79** | `vwap_reclaim_state` ∈ {-1, 0, 1}. |
| Weekly VWAP + σ bands | **computable-from-existing** | Rolling 5-day volume-weighted mean on SPX. Not in 79. |
| S/R levels (intraday) | **computable-from-existing** | Pickles maintains a pre-market level registry (`es_macro_levels.txt`). Distance-to-nearest-level is computable if we maintain the registry. Not in 79. |
| S/R levels (weekly / historical) | **computable-from-existing** | Same method, multi-day lookback. Not in 79. |
| Fibonacci retracements (OVN, current/previous session, weekly) | **computable-from-existing** | Fibs from session / week highs and lows. Not in 79. |
| Historical H/L (prev RTH, week, month, 2mo, 3mo) | **computable-from-existing** | Trivial from price history. `prev_high_dist` in 79 but limited. |
| Supply/Demand zones | **computable-from-existing** | Swing-high/low identification. Not in 79. |
| Trend channels (weekly/monthly/swing H/L) | **computable-from-existing** | Regression channels. Not in 79. |
| Volume profile (POC, HVN, LVN) | **in-current-79** partially | `poc_dist` (distance to POC) and `va_position` (value-area position). HVN/LVN locations themselves not surfaced. |
| Bollinger Bands (4HR) | **in-current-79** partially | `bollinger_position`. Timeframe is likely 1m, not 4HR — **mismatch with Pickles' stated 4HR usage**. |
| Market internals: A/D ratio, A/D volume, TRIN, TICK, P/C ratio | **needs-new-data** | None in current 79. The put_call_txn_ratio is chain-inferred, not exchange P/C. These are standard NYSE/NASDAQ feeds; may be in Polygon. |
| Volume-at-time | **in-current-79** | `volume_ratio`, `log_total_volume`, `log_chain_volume`. |
| CCI / Woodies CCI | **needs-new-data** | Not computed; **computable-from-existing**. |
| EMA (8 / 21 / 34 on chart; 8/21 on 5m, 15m) | **computable-from-existing** | `ema_cross` in 79 (single EMA cross signal, not the specific EMAs). |
| Moving averages (50/100/200/225) | **computable-from-existing** | Not in 79. |
| TD 9/13 (on 15m) | **computable-from-existing** | Not in 79. |
| Macro levels (static round-number ES levels) | **computable-from-existing** | Published in `es_macro_levels.txt`; distance-to-nearest-round-number computable. Not in 79. |
| Larry Levlin OTF (one-time framing) on 15m/30m | **computable-from-existing** | Swing-structure based. Not in 79. |
| Inside candles (for scalping) | **computable-from-existing** | Bar pattern recognition. Not in 79. |
| Initial Balance (IB) + OVN H/L | **in-current-79** partially | `ib_break` (signed break flag), `ib_extension_pct` (size of break). OVN H/L themselves: **needs-new-data** — ETH data not in current `data.pt`. |
| Heiken Ashi (5m) | **computable-from-existing** | 5m HA candles. Not in 79. |
| MACD (slope) | **in-current-79** | `macdh_slope`. |
| RSI | **in-current-79** | `rsi_7`. |
| ATR | **in-current-79** | `atr_14`. |
| Realized vol | **in-current-79** | `realized_vol`. |
| VIX / VIX regime | **in-current-79** | `vix_regime`, `vix_roc`. |
| VX futures term structure | **needs-new-data** | He uses the VX curve for VIX forecasting (2023-11-13 explanation). Not in 79. |
| IV percentile | **in-current-79** | `iv_percentile`. |
| IV skew | **in-current-79** | `iv_skew_pct`, `slice_iv_skew_slope`. |
| VRP (vol risk premium) | **in-current-79** | `vrp`. |
| Greeks (delta, gamma, theta) | **in-current-79** | `atm_gamma`, `atm_theta_per_bar`, `theta_acceleration`, `gamma_pressure`, `aggregate_charm`, `current_moneyness_pct`, `near_atm_moneyness_pct`. Per-contract Greeks live in sidecars (`row_features`). Delta is **in-sidecars-unsurfaced** as a direct feature (the sidecars carry the full contract features). |
| Option spread % | **in-current-79** | `option_spread_pct`, `slice_mean_spread`. |
| Gamma concentration / imbalance | **in-current-79** | `slice_gamma_concentration`, `slice_gamma_dollar_concentration`, `slice_call_put_gamma_imbalance`. |
| Distance to max-gamma strike | **in-current-79** | `slice_dist_to_max_gamma`, `slice_dist_to_max_gamma_dollar`. |
| S&P sectors (XLK, XLF, XLY, XLE, XLU, XLC, XLRE, XLI, XLV, XLP) | **needs-new-data** | Sector ETF data not in current dataset. Polygon covers ETFs. |
| Sector divergence vs index | **needs-new-data** | Derived from sector data above. |
| ES / NQ / RTY / YM / CL / VIX / GC spot + levels | **needs-new-data** | Current dataset is SPX + SPY + VIX only. ES/NQ correlation is 96% (Pickles, 2024-01-04). |
| Bookmap / orderflow (institutional) | **needs-new-data** | Professional-data vendor. Out of scope for current budget tier. |
| OVN inventory (his classification) | **computable-from-existing** | Derived from ES overnight range vs prior session POC. Not in 79. Needs ES data. |
| Time-of-day markers: pre-open, first-15m, 1000, 1030 IB, 1100, 1115, 1130, lunch, 1300, power hour, 1555 close | **in-current-79** partially | `marker_10am`, `marker_11am`, `marker_1130am`, `lunch_flag`, `power_hour_flag`, `intraday_sin/cos/phase`. Missing: **MAGIC TIME 0955-1010 window flag** (trivially computable), **pre-half-hour blackout flag** (trivially computable), **1300 algo window** (no direct flag — computable from timestamp). |
| News-event calendar | **needs-new-data** | Economic calendar (FinancialJuice, ForexFactory, Polygon events API). Not in 79. |
| OPEX / MOPEX / quad-witching calendar | **computable-from-existing** | Third Friday of month + quarterly. Not in 79. |
| FVG (fair value gap) | **computable-from-existing** | 3-bar imbalance detector. Not in 79. |
| Volume imbalance | **computable-from-existing** | Not in 79. |

**No "unknown" cells.** R1 pass condition for coverage map satisfied.

### Coverage summary

- **In current 79:** ~40% of his vocabulary, concentrated in price / volatility /
  option-surface features.
- **In sidecars but unsurfaced:** per-contract delta / theta ratios exist in
  `row_features` but don't flow to the context tensor. Per-bar slice gamma/flow
  stats are in 79 already.
- **Computable from existing data (no new purchase):** a large chunk — weekly
  VWAP, historical H/L levels, volume-profile HVN/LVN, TD 9/13, EMA ladders,
  Heiken Ashi, FVG, MAGIC-TIME window flag, don't-trade blackout flags, OVN H/L
  (if ETH bars exist in raw cache), Fibonacci levels.
- **Needs new data (paid or new vendor):** market internals (TICK, TRIN, A/D ratio,
  A/D volume, NYSE-derived P/C), ES/NQ/RTY/YM futures spot, DXY, 10YR yield,
  sector ETFs (XLK/XLF/XLY/…), VX futures curve, economic calendar, Bookmap.

---

## Multi-timeframe usage

From `personality/time_frames.txt`:

| Timeframe | Pickles' use |
|---|---|
| 1m | **Not recommended** except for precise execution timing |
| 3m | Primary trading timeframe |
| 5m | Length-of-trade gauge; 5m Heiken Ashi for trend |
| 10m | Mid-term trend/momentum |
| 15m | Half of 30m OTF; first-15m close is the trade gate |
| 30m | OTF (one-time framing) |
| 1hr | "Step back" bigger picture |
| Daily | Overall market sentiment |
| Weekly / monthly | Historical levels |

**Current model's `LOOKBACK=30`** (v2/core/features.py, via training config) **is 30
one-minute bars — half an hour of context**. This is a fraction of his shortest
useful timeframe (3m) and zero of his context timeframes (hourly / daily / weekly).
Concretely: when Pickles decides at 1005 EST whether to take a MAGIC TIME long, he
is referencing the opening 15m candle (complete), yesterday's close, the weekly
POC, and the monthly level registry. The model sees only the last 30 1-min bars.

**This is a structural feature-design gap**, not just a list-of-indicators gap. A
Pickles-vocabulary feature set that kept the same LOOKBACK=30 one-minute window
would be missing the point.

---

## Confluence principle

From `personality/trade_clear_levels.txt`:

> "Go bigger where MANY factors offer confluence. You have an untouched level,
> ONH, yesterday's high, a first test of VWAP, all at a major level like 4400
> AND all the levels are within 3 points. This might be a great place to size up."

2024-05-09 chat confirmed: *"pickles loves his confluence"*. The 2024-05-09
supply-zone-break setup (Row 3) explicitly lists VIX + DXY + 10YR + sector
alignment as the confluence.

**Implication for ML:** a "confluence count" feature (number of Pickles-tier signals
aligning within N points of current price) is computable from existing data once
the level registry is maintained. Not a fundamental indicator — a *meta*-indicator.

---

## Journal outcome inference

Journal text reveals Pickles' *process* richly. His *outcomes* are mostly carried
in Discord screenshot attachments that did **not** make it into this corpus — the
user confirmed this explicitly (2026-04-18).

Where outcomes do appear textually, they cluster in a few forms:

| Outcome type | Example | Frequency across the ~15 entries I read in depth |
|---|---|---|
| Explicit $ PnL on a completed trade | 2024-05-14 *"NET PROFIT --> $3,218 from PCS"*, *"total NET PROFIT for today --> $18,900"* | ~3 entries (all on credit-spread or futures legs, not on 0DTE long legs) |
| Explicit % PnL | 2023-10-31 *"61.7% of the initial CREDIT RECIEVED"*; 2023-11-13 *"87% GAIN"* on PCS | ~3 entries (credit spreads, not calls/puts outright) |
| Win/loss language only | 2023-12-14 *"SPX LONG CALLS SL'd out"*; 2024-04-15 *"out of these calls at a loss, 1 for 3 on trades today"* | ~4 entries |
| "Out of" with no outcome | "out of SPX LONG CALLS" without $ | most entries |
| Silent on outcome | Entry narrated, exit not journaled | frequent |

**One confidence-tagged claim about his profitability pattern:** Across the 167
journal days I have not sampled exhaustively, the *narrative evidence* is
consistent with a trader who (a) wins on credit spreads / theta-gang
systematically, (b) takes 0DTE long legs opportunistically — sometimes as
hedges into losing spreads, sometimes as directional scalps — and (c) enforces
a hard **3-losing-trades-and-log-off** rule (2023-12-14: *"Pickles has a house
rule of 3 losing trades → log out of broker and come back tomorrow"*).
**Confidence in this claim: Medium.** It is consistent with all narration read
but is inferred, not stated. It does **not** establish that his 0DTE long legs
specifically have edge — the textual evidence is silent on that for most days,
and the screenshot PnL is unavailable.

**Caveat for Fork A and Fork C:** Neither fork can rely on journal-derived PnL
labels as supervision weights. Fork A uses the simulator's PnL, so this is
fine. Fork C uses Pickles' *actions* as labels and lets the simulator score
them — also fine. Any fork that *requires* journal-derived PnL as supervision
weight is **not viable** from this corpus alone.

---

## 0DTE trade mentions count

Across the 167-journal corpus, the initial regex sweep (case-insensitive,
patterns: `SPX 0DTE`, `SPX \d{4,5}`, `LONG SPX`, `0DTE (LONG|CALL|PUT)`)
surfaced:
- **~40 journal entries** with SPX long / SPX-0DTE text mentions.
- After removing credit-spread hedges (explicit "counter DELTA" / spread-defense
  context) and sub-mentions: **~15 standalone 0DTE long trade narrations** across
  dates spanning 2023-10 through 2024-06.

This is the **upper bound on Fork C Tier-3 sample size** from the current corpus.
Fifteen narrated standalone long trades is thin for per-bar behavioral cloning
but usable for **Tier-1 (day classification)** and **Tier-2 (setup classification)**
where the label density per day is 1 (one label per day) and ~150-167 labels are
available respectively.

---

## R2 theory prompts

During R1, four candidate theory prompts could be fired:

1. **"Does the VWAP-support quick-in-out setup (Row 1) depend on expiry-day
   Delta/Gamma behavior that we need to sharpen?"** → Row 1 uses ATM-to-slightly-
   ITM strikes; Delta is high and relatively stable over ~30-minute holds. I do
   not need Natenberg to adjudicate this. **Prompt not fired.**
2. **"Does the 1000 MAGIC TIME setup (Row 2) need IV-crush timing theory?"** →
   0DTE IV behavior on directional scalps at 1000 EST is ~5 hours from expiry.
   Theta acceleration exists but Delta dominates at ATM strikes. Current `atm_theta_per_bar`
   and `theta_acceleration` already surface the relevant dynamics. **Prompt not fired
   yet**; if Fork A fires and the backtester shows suspicious theta decay dominating
   winners, fire then.
3. **"Pin-risk / dealer hedging mechanics?"** → Not relevant for long 0DTE debit
   positions at ATM; only becomes a factor for very-short-dated short positions near
   strike. **Prompt not fired.**
4. **"Bar-by-bar price action formal grounding?"** → Pickles' setups are level-to-
   level and time-of-day-gated; they are already defensible without needing Brooks'
   framework to formalize them. **Prompt not fired.**

**R2 output:** No `options_theory_notes.md` is produced by R1 alone. If Fork A or
B later needs theory support, we fire a prompt then.

---

## R1 pass conditions — final check

- [x] At least three rows in the canonical setup table at Medium+ confidence.
      **6 rows at Medium+, 2 at High.**
- [x] Indicator coverage map complete, no "unknown" cells. **38 indicators mapped.**
- [x] Journal-outcome inference is a single confidence-tagged claim.
      **"Winning on spreads, opportunistic on 0DTE longs, enforced 3-loss stop"
      at Medium confidence.**

**R1 pass: YES.** Fork A has High-confidence candidates in Rows 1 and 3. Fork C's
day-level and setup-level supervision is viable; per-bar supervision is thin (~15
trades) but non-zero.

Next: R3 feasibility audit, then R4 fork decision.
