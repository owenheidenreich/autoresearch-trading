# Protocol101 Full Trader Graph V2 — Consolidated Authority

Status: **CONSOLIDATED, VALIDATED, AWAITING OWNER SIGNATURE**

Prepared: 2026-07-28 (FT2-01 Consolidated Authority goal).
Consolidation only: this document invents zero new science, changes zero
contracts, trains nothing. It merges the signed contracts, the Rev C revised
plan, and the Codex planning brief into one self-contained authority. Where two
sources disagreed, precedence resolved it and the resolution is recorded in
[Appendix A — Resolved Conflicts](#appendix-a--resolved-conflicts); nothing
substantive was resolved silently.

Machine-readable companion (same schema family as Graph V1):
`v4/docs/protocol101/training/execution/PROTOCOL101_FULL_TRADER_GRAPH_V2.json`
(SHA-256 `b06a26be59307c130da84f2dc5b6f3224c272e6c4093e83abd5bc0b280ca6d09`).

Source authority chain and precedence (highest first):

1. Signed contracts — may not be altered by consolidation:
   - `PROTOCOL101_TRADER_CHARTER.md` (signed 2026-07-25)
   - `PROTOCOL101_G4_HOLDOUT_REVISION_2026_07_19.md` (signed 2026-07-19)
   - `PROTOCOL101_G8_CALIBRATION_REVISION_2026_07_26.md` (signed 2026-07-26)
   - `PROTOCOL101_SCOPED_SYNCHRONIZATION_DECISION_2026_07_25.md` (signed 2026-07-25)
   - `PROTOCOL101_D1_NEGATIVE_CONTROL_AND_INCREMENTAL_EDGE_AMENDMENT_2026_07_28.md` (owner-authorized 2026-07-28)
2. Rev C revised plan — `PROTOCOL101_FULL_TRADER_GRAPH_V2_REVISED_PLAN_2026_07_28.md`.
3. Codex planning brief — `PROTOCOL101_FULL_TRADER_GRAPH_V2_PLANNING_AND_FABLE_REVIEW_BRIEF_2026_07_28.md`
   (SHA-256 `765591b130ccc6d1e9a12d65252808027a08c02118d9be2ace22bd2834ee2e3f`, verified 2026-07-28).

---

## 0. Owner Memo (one page, plain English)

**What this document is.** It is the single, complete, plain-and-technical
description of the Protocol101 Full Trader research program — the learned SPXW
0DTE trader that waits when nothing is worth trading, buys one exact call or put
when something is, holds while holding has value, and exits with survival
controls intact. Until now that description lived across a planning brief, three
revisions of a plan, and five signed contracts. This document folds all of it
into one, so no future step ever has to reconcile two documents. It is paired
with one machine-readable graph file that a controller can check.

**What signing it authorizes.** Signing authorizes **Phase 0 documents work
only**: freezing the path-label definitions (FT2-04) and running the label-side
opportunity census (FT2-05). Both are CPU-only, one-time, and train no model.
The census can, at most, stop the program for your decision if the game is
genuinely unplayable after fees; it can never by itself green-light training.

**What signing does NOT authorize.** It does not authorize model training,
feature admission against protected paired days, GPU spend, protected-holdout or
confirmation access, IBKR/broker contact, paper orders, promotion, paid
downloads, launchd/runtime changes, or real money. Every one of those sits
behind a later, separate owner gate in the graph (design approval FT2-21, entry
freeze FT2-52, lifecycle approval FT2-62, paper authorization FT2-95, and the
protected-resource nodes). Each node also remains a separately dispatched Goal;
this signature is not a grant to run the whole graph.

**What you are really signing off on.** Four safety/identity decisions you
already answered on 2026-07-28 (5% per-trade premium cap, budget-aware entries,
the four-bucket profile tripwire, and the loose-screen-plus-upside guardrail
philosophy) are now written into the product contract, plus eight design
decisions that still need your signature (listed in §14). If any of those eight
are wrong, this is the moment to say so — after signing, they are frozen until
you re-open them.

**Highest claim after you sign:** "Graph V2 has a consolidated, validated,
signable authority package, and Phase 0 documents work is authorized." Nothing
more.

---

## 1. Product Contract

### 1.1 Instrument and account

- Instrument: SPXW 0DTE **long** options (capped loss, uncapped gain).
- Position size: **one contract**; maximum **one** open position.
- Entry: buy to open. Exit: sell to close.
- No short premium, no scaling in/out, no averaging down, no overnight
  position, no martingale.
- Account: $10,000 starting cash; $3 round-trip fee overlay; simulator v5
  one-account serial semantics.

### 1.2 Who the trader is (the Charter, binding)

A base-hits trader with home-run capacity, protected by a survival guarantee —
the Pickles profile. Most trades end small and harmless because exit discipline
scratches anything that stops working; a real minority run to large wins; losses
are cut fast; big losses are rare. High win rate is a **byproduct** of exit
discipline, never an optimized objective (optimizing win rate teaches the system
to scratch everything and bleed fees). The optimized objective is always
fee-adjusted PnL.

The four commitments (Charter, verbatim intent):

1. **Droughts are flat, never deep.** Abstention when the market offers nothing;
   depth of drawdown matters more than duration of flatness.
2. **Floor is SPY; dream is convex (3x–10x), reached the Pickles way.** No
   hardcoded profit cap may exist in the final form; fixed exit targets are
   Stage-1 scaffolding scheduled for replacement by learned exits mandated to
   harvest the rare huge winner.
3. **No single day may wound the decision-maker.** 5%-of-session-start-equity
   daily circuit breaker; no revenge trades once hit.
4. **Survival outranks everything.** Account never below half starting capital;
   every dollar of drawdown purchased by at least a dollar of realized profit
   (inherited from signed G4 v2). Ruin is the only unrecoverable outcome.

The Pickles outcome distribution (fee-aware, directional target, report-only —
never an optimized target): big wins ~18% at ~+40%; scratches/small wins ~73%
(any exit between −5% and +5% is a scratch at our size); small losses ~8%; big
losses under ~2%.

### 1.3 Hard-coded safety (machinery may enforce)

SPXW/0DTE eligibility; data freshness; complete-ladder requirement;
affordability; one contract; one open position; ask-entry/bid-exit accounting;
fees and stress; **5%-of-session-start-equity daily realized-loss stop**;
account-survival backstop (50% of starting capital); **no new entries after
15:30 ET**; **forced flat by 15:55 ET**; stale-data abstention. Plus the two
Rev C amendments in §1.4.

Hard-coded machinery may **not** permanently decide: when to enter; call vs put;
nearest-ATM; OTM/ATM/ITM; a fixed holding period; a fixed profit target; or the
ordinary learned exit. Those are learned.

### 1.4 Rev C hard-safety product amendments (owner-answered 2026-07-28)

These amend the brief's hard-coded-safety list. They are product-contract
changes and must be implemented **identically** in the historical serial
simulator and every live/shadow guard — the **same-game law**: a cap that exists
offline but not live (or vice versa) is a synchronization defect.

- **D48 — Per-trade premium cap (T1).** Premium at risk plus the round-trip fee
  overlay **≤ 5% of session-start equity** ($500 at $10,000; scales with the
  account). Contracts above the cap are masked from the eligible action set
  exactly like unaffordable contracts — a safety mask, never an alpha feature.
  *Why:* affordability alone allowed up to ~$3,500 premium at risk on $10k; a
  single open trade could lose 8–35% of the account, and the realized-only
  breaker cannot stop an open position. This bounds the single-trade wound to
  Charter commitment 3.
- **D49 — Budget-aware entries (T2).** A new entry is legal only if
  `realized_session_loss + (new premium at risk + fee) ≤ 5% daily budget`. The
  effective ladder narrows as realized losses accumulate (natural size-down when
  losing); a day can reach forced-WAIT before the breaker fires (soft landing);
  the breaker itself keeps its **signed realized-only basis**
  (`DAILY_LOSS_BASIS = raw_realized_net_pnl_at_occupancy_exit`). Worst-day
  arithmetic under both rules is bounded ≈ −5% realized, with no open position
  able to add more than the remaining budget.

The census (FT2-05) applies both rules to its opportunity accounting; the entry
model's masks reflect them at every minute.

---

## 2. Full-Ladder And Same-Game Contract

### 2.1 Learned action spaces

When flat: `WAIT` or `BUY` one exact eligible call/put/strike. The flat model
owns timing, call-vs-put, strike, moneyness, contract choice, and abstention.
There is no upstream P5 timing rule, direction rule, or nearest-ATM selector.

When one position is open: `HOLD` or `EXIT`. The open-position model owns
continuing to hold, exiting now, and the forecast information the frozen
protective-floor composer consumes. The floor is a transparent composer output,
not a hidden third policy.

### 2.2 Action universe and masking

At each completed decision minute: `WAIT` + **42 contract slots** (21 strikes
from 50 points below through 50 points above the canonical ATM strike in 5-point
increments, times call and put). An unavailable, stale, unaffordable (incl. the
D48 cap), malformed, or D49-budget-exceeding contract is **masked**. A mask
removes an impossible action; it never becomes an alpha feature unless
explicitly admitted.

### 2.3 Contract-identity continuity

A 90-minute sequence must never blindly stack canonical slot indices. The
historical and live tensor builders must key temporal contract paths by exact
contract identity, strike, expiry, and right; separately encode each historical
cross-sectional ladder snapshot; provide current-relative moneyness/offset every
minute; mask minutes before a contract entered the governed universe; and prove
historical and live builders resolve identities identically. Failure invalidates
all model evidence.

### 2.4 Decision timing and rolling history

- The model runs once per completed minute; the same completed-minute convention
  is used historically and live. Historical labels may use future paths; runtime
  tensors contain current/past information only.
- Entry/exit fills use the causal ask/bid convention plus fees and stress.
  Latency, missing minutes, reconnects, and stale data are reproducible
  abstention conditions.
- 90-minute rolling causal history with available-history masking near the open;
  90 minutes is an owner-chosen design horizon, not a proven optimum, and does
  not cap how long a trade may remain open.

### 2.5 Clock fields and market phases (context features, not style bans)

Every model receives `minutes_since_open`, `minutes_to_close`, and a categorical
`market_phase`:

| New York time | Phase |
|---|---|
| 09:30–09:45 | Opening discovery |
| 09:45–11:15 | Primary morning session |
| 11:15–11:45 | Europe-close transition (~11:30) |
| 11:45–13:30 | Lunch |
| 13:30–15:00 | Afternoon |
| 15:00–15:30 | Power-hour entry period |
| 15:30–15:55 | Manage/exit only |

Only the no-new-entry boundary after 15:30 and forced-flat at 15:55 are hard
rules.

### 2.6 Open-state decision rows 15:31–15:55 (D40)

Verified gap: decision rows currently end 15:30 ET (359/session) while
normalized quotes reach 16:00 ET and forced-flat label coverage already passes.
The 15:31→15:55 open-state rows (≤ 25 rows/session, ≤ 7,525 minutes corpus-wide)
are a governed machinery deliverable (built and accepted at FT2-28). The gap is
the open-state decision-row contract, not raw data.

---

## 3. Feature Strategy And Admission Law

### 3.1 Mandatory feature floor

Every serious full-ladder entry candidate must receive: canonical option premium
and its 90-minute causal path; call/put identity; exact strike; current
offset/moneyness; affordability and eligibility state (incl. D48/D49 masks);
canonical SPX price action and synchronized context; **internally recomputed**
IV/delta/gamma (from canonical price, spot, strike, time-to-expiry, frozen
constants — raw vendor Greeks prohibited); clock fields and market phase; and
masks for missing history and unavailable contracts.

### 3.2 The signed synchronization conflict and how the amendment is earned

The signed synchronization decision authorizes **exactly 17 model-facing
features** and quarantines direct per-slot option-price paths (Family C
source-discriminator AUC `0.550524`, narrowly above the frozen `0.55` ceiling;
D-family `0.502582`, E-family `0.500492`). The Full Trader's mandatory floor
requires per-contract premium history. **These conflict.** Consolidation does
not silently amend the signed contract (precedence: signed > Rev C > brief). The
amendment may only be *earned* through the bounded admission node (FT2-25/26).
See [Appendix A](#appendix-a--resolved-conflicts), conflict R2.

The signed initial 17-feature alpha list (unchanged, carried for reference):
`spx_vwap_gap_points`, `spx_vwap_gap_bps`, `spx_vwap_gap_over_session_range`,
`session_range_bps`, `momentum_5m_bps`, `momentum_15m_bps`,
`momentum_5m_over_session_range`, `momentum_15m_over_session_range`,
`omar_clipped_neg3_pos3`, `vwap_side_alignment_flag`, `omar_side_alignment_flag`,
`momentum15_side_alignment_flag`, `D.near_atm.straddle_mid_spot_bps`,
`D.near_atm.put_call_mid_ratio`, `D.near_atm.side_smile_slope_bps_per_5pt`,
`E.bs.delta`, `E.bs.gamma`. Raw bid/ask/spread/sizes/quote-age/volume/OI/vendor
Greeks remain available for guards, fills, labels, PnL, audit — never silently in
the model feature matrix.

### 3.3 Admission test (D41)

- **At most 3 canonical-price transforms**, preregistered together with
  rationale before any is tested (e.g., quantized log-premium changes; premium
  normalized by straddle mid; rank-normalized within-ladder premium). Every
  attempt, pass or fail, enters the multiplicity family.
- **Harness validation first:** a positive control (a known vendor-identifying
  field) must be caught and a null control (known-clean feature) must pass, or
  the harness is rejected.
- **Evidence-aware ceiling test:** the signed 0.55 ceiling stands. A transform
  passes the discriminator component only if its **session-block bootstrap 95%
  CI upper bound is below 0.55** on leave-one-session-out evaluation. A CI
  straddling 0.55 returns `insufficient_paired_evidence` → FT2-24 collection
  plan, never pass or fail. (Rev A's arbitrary 0.53 second constant is
  withdrawn.)
- **Primary gates** (all required; AUC is one component, never a lone veto):
  decision-transfer agreement ≥ 0.99 (the signed stable-probe standard); no
  label/opportunity-correlated drift; sane boundary/stress behavior; economic
  materiality.
- **Governance (D46):** only days FT2-24 classes as available (currently
  `development`) may be used. Sealed and burned days are untouchable.
- If all 3 transforms fail → `STOP-OWNER-DECISION`: product minimum unmet, or
  collect more development-class paired days and retest under the same
  preregistration.

### 3.4 Paired-evidence manifest inventory (D46, FT2-24)

Before admission, a manifest-only inventory of every paired recorder/historical
day records sealed-day class, completeness, gate status, and governance
availability. Verified reality (sealed-day assignment run 2026-07-28T20:32Z):

| Class | Sessions |
|---|---|
| burned | 2026-06-30, 07-01, 07-02 |
| validation | 2026-07-10 |
| development | 2026-07-13, 07-14, 07-20 |
| sealed | 2026-07-28 |
| unassigned_pre_rule | 2026-06-29, 07-06, 07-07, 07-08, 07-09 |

Only the three `development` days may be used for admission CI evidence. If they
are too few for CI-based admission, the honest output is a **collection plan**,
not a weaker test.

### 3.5 Core-and-rich lanes, one lane for campaign 1 (D18)

Campaign 1 trains **parity-core + admitted canonical price history** only; the
ladder-rich challenger is deferred and may win only through frozen unseen
evidence. **Rich-lane escape (Codex Q1):** if the core lane terminates
`no_genuine_entry_signal`, the routing packet must present the owner a
preregistered, bounded rich-lane feasibility option (admission-tested rich
features, one bounded feasibility run, no silent expansion) before any conclusion
about the Full Trader product is recorded. Core-lane failure is evidence about
the core lane, not the product. In the graph this is the
`no_genuine_entry_signal` edge from FT2-51 → `STOP-OWNER-DECISION`.

---

## 4. Entry Science Defaults (FT2-10)

Defaults for the FT2-10 entry-science contract to adopt or overturn with
recorded reasons.

### 4.1 Forecast horizons (D07 — approved set retained)

Predict path properties at **{3, 5, 10, 20, 45, 90, remaining-session}**
minutes. Rev A's trim is **withdrawn** (Codex B3: short-trade resolution at 3–8
minutes is an owner emphasis). FT2-10 may prune only with pre-validation
redundancy evidence recorded in the FT2-10 packet before any pruned variant is
registered. These are observation horizons, not forced exits; near the close,
unavailable horizons are censored and masked, never shortened or invented.

### 4.2 Economic primitives and dual representation (D17)

Each path target is available in fee-adjusted **dollars per contract** and
fee-adjusted **return on entry premium**, using causal executable entry ask `A_t`,
causal executable future bid `B_u`, round-trip fee overlay `F`, and a consistent
contract multiplier. Dual representation prevents cheap OTM contracts winning
only on explosive percentages and expensive ATM/ITM contracts winning only on
larger dollar PnL.

### 4.3 Six path-property families (D06, D09)

1. **Early drawdown** — worst executable fee-adjusted return in first 3/5/10 min
   (distribution, not just mean).
2. **Time to first real profit** — time until executable bid clears entry cost +
   fees; censored when no such event before the horizon/close.
3. **Pre-profit adverse excursion** — max adverse excursion before the first
   fee-adjusted profitable minute.
4. **Underwater burden** — depth×duration integral below fee-adjusted breakeven;
   total and longest-continuous underwater minutes.
5. **Profitable-window stability** — fraction of minutes with executable positive
   PnL; longest continuous profitable interval; number of distinct windows;
   one-minute jitter sensitivity.
6. **Upside** — executable MFE / upper-return distribution by horizon;
   profit-area and tail opportunity; dollars and percentage.

### 4.4 Prohibited target shortcuts (D08)

The entry model may not receive or optimize: the perfect-hindsight exit minute;
maximum future price as a complete label; the selected future exit from the
lifecycle model; future bid/ask/spread/Greeks/context as features; oracle action
labels as runtime features; protected-holdout outcomes; or IBKR shadow/paper
outcomes. Future paths create supervised answers only.

### 4.5 Quality-first composer, guardrail philosophy (D10, D11, D51)

Two-stage decision: (1) determine which contracts have credible entry-path
quality; (2) among those, choose the best **conservative** upside; else `WAIT`.

**Guardrail philosophy (D51, owner-decided).** Guardrails are a **bottom-tail
exclusion**, not a top-tier selection. They exclude clearly-bad paths
(calibration anchored near the worst ~quartile of training-role path quality;
exact quantiles FT2-10's to set); the conservative-upside stage does the actual
ranking. *Why:* 0DTE long entries start underwater by construction (spread
crossing), and many Charter-profile big wins chop red before the move; a tight
comfort screen structurally builds the scratch-mill scalper the Charter forbids.
FT2-10 must justify its operating point against **both** census curves —
expected trade rate **and** excluded-winners rate (§7).

Guardrail calibration: quantile-based thresholds from training-role
distributions per market phase, frozen per fold before outer access, emitted with
hashes. The census (FT2-05) supplies the distributions and the code path; FT2-10
must **reuse, not reimplement**. Numeric guardrails are never selected from outer
validation, confirmation, holdout, or live results.

**Joint-conservatism control (Rev C).** Three abstention mechanisms stack
(guardrails, conservative-upside-after-fees, uncertainty-margin WAIT). FT2-10
must set them jointly against a census-derived design trade-rate band — a
calibration target, not a gate — so independently-paranoid settings cannot
silently starve the trader into `insufficient_evidence`.

### 4.6 Uncertainty-aware abstention and the G8 v2 gate

The model chooses `WAIT` when no contract clears the quality guardrails;
conservative upside does not clear zero after fees; the selected contract's edge
over alternatives is inside calibrated model/source-transfer uncertainty; or
required context/ladder state is missing.

Because confidence now controls WAIT and contract choice, the **signed G8 v2**
report-only regime does not govern this architecture as-is. Per G8 v2, before
calibrated confidence may control any trading behavior, a **separate
action-conditioned calibration gate must be preregistered, smoke-tested,
independently audited, and owner-signed**, evaluating the behavior confidence
actually controls (WAIT-rate reliability, selected-contract regret vs forecast,
exit-decision reliability, protective-floor behavior, low-sample fallback). This
gate is a required part of the FT2-10 contract. The HGB training target remains
fee-adjusted payoff / return-on-premium, never win probability. See
[Appendix A](#appendix-a--resolved-conflicts), R1.

### 4.7 Primary entry gate and P5 benchmark (D19, D20)

Whole-path entry quality is the primary gate. Fixed stops/targets/holding-times
do not decide whether an entry is good; if retained they are report-only
diagnostics. No hard trades/day maximum (historical 0.3–6/day is descriptive);
zero-trade days are allowed; a model may not pass merely by trading almost never
(→ `insufficient_evidence`, a distinct terminal). P5 (VWAP-side nearest-ATM,
run-to-flat) is a **mandatory benchmark, never final architecture**; the entry
model must beat P5 on matched whole-path entry quality with WAIT decisions kept
in the accounting.

---

## 5. Lifecycle Defaults (FT2-60 series)

### 5.1 Training order (D21, D27)

Freeze the entry model first; pretrain lifecycle broadly on eligible causal
contract paths; specialize/evaluate on out-of-fold trajectories actually selected
by the frozen entry; keep entry and exit separately auditable; **no** joint
fine-tuning in campaign 1. Two lifecycle families compete on identical evidence:
transparent HGB baseline and temporal position-aware neural challenger (D26).

### 5.2 Runtime inputs and shadow evaluation (D22, D23)

The open-state model may receive complete causal market/ladder history; open
contract identity/state; entry ask and elapsed hold; current bid/ask/mid and
admitted spread; fee-adjusted unrealized PnL (dollars and percent); MFE/MAE
through the current minute; giveback and time since MFE; recent price/PnL
velocity; current moneyness and internal Greeks; account and daily-stop state
(incl. D49 budget); current protective floor; market phase and minutes to forced
flat; the frozen entry model's **no-action shadow scores**; and a causal forecast
of qualified replacement opportunities. Actual future opportunities are labels,
never runtime inputs.

Multi-horizon hold advantage: value of exiting now; continuation value over
horizons; downside if holding; recovery chance/magnitude; giveback risk;
remaining tail; opportunity cost of the single occupied slot; and forecast
uncertainty. No one-minute oracle HOLD/EXIT label as a runtime feature.

### 5.3 Protective floor (D24) and Rev C additions (D43, D52)

Forecast-derived, upward-only floor, checked at completed-minute executable bids;
it may stay or rise, never fall. Causal order at minute t: read bid/state; if the
floor committed at t−1 is crossed, exit; else evaluate new HOLD/EXIT forecasts;
if HOLD, compute the next floor and commit it for t+1.

- **Floor-slippage realism (D43):** every lifecycle packet reports realized exit
  price vs committed floor (gap-loss distribution). No intraminute protection is
  claimed; the evidence shows what crossing actually cost.
- **Floor ablation (D52):** the exit comparator set must include
  **learned-exit-without-floor**. The floor must earn its keep; if floor-on
  materially degrades tail capture vs floor-off, that is a finding, not an
  implementation detail.
- **Floor breathing room (D52):** the floor equation may be explicitly time- and
  profit-conditional (loose early, tighter late / deep in profit). An
  upward-only ratchet on noisy minute bids otherwise converts every post-MFE
  retrace into a forced exit and chokes the Charter's big-win column, which
  requires enduring −20/−30% retraces from peak mid-trade.
- **Harvest tripwire (D52):** if the harvest ratio (realized / peak available
  PnL) collapses under floor-on relative to floor-off, route
  `owner_decision_required` — protection is being bought with the dream. In the
  graph: FT2-74 `harvest_tripwire_owner_decision` → `STOP-OWNER-DECISION`.

### 5.4 Exit evaluation (D20, four-box)

The learned exit is evaluated on identical frozen entries against: exit
immediately; hold to forced flat; a finite preregistered set of transparent
time/stop/target comparators; legacy P5 lifecycle where relevant; matched-rate
random exits. It must beat the best honest comparator pooled, in ≥ 4 of 5
chronological outer folds, after identical fees/stress, under one-account serial
replay, without violating survival controls. Comparators do not limit learned
trade duration. Diagnostics include the four-bucket distribution, harvest ratio,
loss truncation, tail-win capture, MFE giveback, MAE, underwater duration, churn,
time-in-trade, skipped opportunities while occupied, exposure by
side/moneyness/phase/premium/regime, and protective-floor activation/boundary.

**Four-box attribution (D32):** Box A (P5/controls entry × best control exit);
Box B (learned entry × control exit); Box C (P5/control entry × learned exit);
Box D (learned × learned). If B fails, entry not accepted; if D fails while C
fails, exit is the leading problem; if B and C pass but D fails, interaction
attribution; if D passes only because P5 supplied timing/contract, product drift.

---

## 6. Evidence And Statistics Law (FT2-11)

- **Session-level effective sample size** and all bootstraps at session level;
  session-block bootstrap throughout.
- **Multiplicity family (D-per-B3):** admission attempts, inner-loop shortlist
  candidates, both lifecycle families, and registered composer/guardrail
  variants — **not per-head** (heads of one jointly trained model are
  shared-parameter outputs, not independent hypotheses; what multiplies is
  selectable variants). Preregistered before Phase C.
- **Minimum detectable improvement (MDE) before spend (Codex Q4):** using census
  variance components, compute the smallest incremental edge over P5/matched-
  random detectable at the required confidence given projected trade counts. If
  the MDE exceeds any plausible edge, that is a `resource_owner_decision_required`
  finding **before** the first GPU tranche.
- **`insufficient_evidence` is a distinct terminal** from `no_genuine_signal`
  (STOP-INSUFFICIENT-EVIDENCE in the graph). Rare-trade candidates whose CI is
  too wide return insufficient evidence, not pass and not automatic rejection.
- **Four-bucket profile tripwire (D50, owner-decided):** every candidate packet
  reports the Charter's four-bucket distribution. A grossly-off profile —
  big-loss share above 2× the Charter's ~2%, or a big-win column near zero —
  routes `owner_decision_required` at the freeze gate. Never auto-pass, never
  auto-fail, and **never an optimizable target** (a gated profile invites
  scratch-harvesting; the Charter warns against optimizing the profile directly).
  In the graph: `profile_tripwire_owner_decision` edges from FT2-51 and FT2-74.
- **Full-ladder D1 V2 extension (D44):** incremental-edge controls extend the
  signed D1 V2 matching table **unchanged** — entry-intent exact; executed count
  ≤ 5%; call/put TV ≤ 5%; moneyness TV ≤ 6% on ATM/NEAR/WING from slot 10;
  premium ≤ 10%; holding ≤ 10%; occupancy ≤ 10%; outcome-blind; post-replay
  verification fails closed; timing **and** contract choice randomized
  feature-free. A real row is selection-eligible only when its paired 95% lower
  confidence bound is strictly greater than zero and the multiplicity-adjusted
  one-sided p ≤ 0.05, with all 28 preregistered row identities and the canonical
  SHA-256 of the ordered multiplicity family present. The $3-per-trade
  equivalence remains a precision diagnostic, never a gate.

### 6.1 Charter gates retained (denominator-independent, carry unchanged)

G4 v2 pooled Calmar ≥ 1.0 **and** per-fold equity never below $5,000; daily
realized-loss stop 5% of session-start equity; one contract / one open position;
fee-adjusted net PnL is primary economics; win rate diagnostic only; SPY is the
live first-year opportunity-cost benchmark; tail capture and survival both
matter. Only G1/G2/G3/G5/G6/G7/G8 need re-derivation for the new action space
(the entry-quality denominators changed); G4 v2, the four-bucket reporting, the
5% daily stop, and D1 V2's matching law carry into the Full Trader unchanged.

### 6.2 Balanced evidence standard (D31)

Strict integrity controls; bounded downside in every fold; multiplicity-adjusted
95% improvement interval above zero; positive evidence in ≥ 4 of 5 chronological
outer folds with a bounded, explained fifth; no dependency on one
side/phase/month/regime; realistic fees/stress; no single-seed story; no
single-metric promotion. Reward-hacking controls (brief §18) are mandatory in
every promotable campaign; positive absolute PnL is never evidence of learning.

---

## 7. Census Specification (FT2-04 / FT2-05)

### 7.1 FT2-04 — Path-label freeze (before any census statistic)

Freeze: exact path-label definitions for all six families (dollar +
return-on-premium); the oracle-selector semantics (entry at executable ask; exits
at executable bids under a small preregistered set of transparent oracle exit
rules — e.g., best-achievable bid by horizon, hold-to-flat — with serial
one-account replay and the daily stop applied through simulator v5); and the
census session set. **Census sessions = governed corpus MINUS every session
appearing in any outer-test slice of any fold MINUS the protected-holdout
sessions MINUS any owner-reserved confirmation sessions once frozen (D42, as
amended 2026-07-29). The FT2-04 packet must prove the census set is disjoint
from BOTH the outer-test union AND the protected holdout, from the fold and
governance manifests** (this replaces Rev A's leaky "outer-train only" rule and
corrects the original D42 formula, which omitted protected resources).

### 7.2 FT2-05 — Opportunity census (label-side only, no model)

On the frozen census sessions, report **threshold-independent** evidence:

- distributions of each path-quality family by phase, premium band, moneyness;
- the Pareto frontier of quality-vs-upside;
- qualifying-minute share as a **curve** across candidate guardrail levels (not
  one number);
- ceiling PnL of the preregistered oracle rules under serial replay with fees,
  the daily stop, **and the Rev C hard-safety rules (D48 premium cap + D49
  budget-aware entries)**;
- P5's share of each ceiling;
- measured label-build compute;
- statistical-power inputs FT2-11 needs (session-level variance components, MDE
  curves vs trade count);
- **excluded-winners rate (D51):** of all oracle big-win paths (fee-adjusted
  return ≥ +40%, the Charter's big-win column), the fraction excluded at each
  candidate guardrail level — the direct measurement of "is the comfort screen
  eating the home runs";
- **friction by premium band:** round-trip cost (spread crossing + fees) as a
  fraction of premium, per band — where the game is even playable (cheap wings
  can carry ~25%+ friction inside current tradability guards);
- **expected trade-rate curves** at each guardrail level under the D48/D49 rules,
  feeding §4.5's joint-conservatism control.

**Hard stop (`STOP-OWNER-DECISION`) only for genuine impossibility:** unusable
labels, no executable post-fee opportunities, or ceilings indistinguishable from
fee drag under session-block CIs. Otherwise the census cannot kill the program;
it informs FT2-21 owner approval and calibrates FT2-11. The no-outer-test-session
firewall (D42) is proved by manifest intersection.

---

## 8. Graph Topology, Loop Policy, Budgets, Hardware

The machine-readable graph (`PROTOCOL101_FULL_TRADER_GRAPH_V2.json`) is authoritative
for routing; this section is its plain-English mirror. Graph V1's controller
schema (receipts, `loop_budgets`, terminal states, `gate_dominance_requirements`,
drift tripwires) is retained; V2 is a topology amendment in the same schema
family. `(NEW)` = not in the Codex brief; `(SPLIT)` from a brief mega-node;
`(REV B/C)` = changed by review.

### 8.1 Phases and nodes

**Phase 0 — consolidation, label freeze, feasibility**
- `FT2-01-CONSOLIDATED-AUTHORITY` (NEW) — this document + the graph JSON; owner
  signs; nothing executes off a delta chain. **Complete** (this goal).
- `FT2-04-PATH-LABEL-FREEZE` (NEW) — §7.1.
- `FT2-05-OPPORTUNITY-CENSUS` (REV B) — §7.2; impossibility-only hard stop.

**Phase A — scientific contracts (SPLIT from the brief's FT2-10 mega-node)**
- `FT2-08-DATA-TENSOR-LABEL-CONTRACT` — corpus roles, fold manifests,
  identity-keyed storage, tensor schema/masking, storage precision law (D47),
  open-state row contract; inherits FT2-04 labels.
- `FT2-10-ENTRY-SCIENCE-CONTRACT` — §4, incl. the G8 v2 action-conditioned gate.
- `FT2-11-EVIDENCE-STATISTICS-CONTRACT` — §6.
- `FT2-20-PARALLEL-DESIGN-REVIEW` / `FT2-21-OWNER-DESIGN-APPROVAL` — three
  contracts reviewed as three documents; a defect in one does not reopen the
  other two (**per-contract `design_repair` budgets**).

**Phase B — feature admission and machinery**
- `FT2-24-PAIRED-EVIDENCE-MANIFEST-INVENTORY` (NEW) — §3.4.
- `FT2-25-FULL-LADDER-FEATURE-ADMISSION` (REV B) — §3.3.
- `FT2-26-INDEPENDENT-FEATURE-ADMISSION-AUDIT`.
- `FT2-28-LIFECYCLE-ROW-BUILD` (NEW) — §2.6.
- `FT2-30-ENTRY-HARNESS-PILOT` — publish actuals for every §8.4 estimate; run the
  fp16 equivalence test; and if GPU is authorized, **prove checkpoint/resume with
  a ≤ $10 dry run before any paid loop**.
- `FT2-31-INDEPENDENT-ENTRY-MACHINERY-ACCEPTANCE`.
- `FT2-32-BOUNDED-MECHANICAL-REPAIR`.

**Phase C — entry autoresearch** — `FT2-40-ENTRY-INNER-AUTORESEARCH` (12-serious-
trial plateau; tranche/MDE/14-day pauses) → `FT2-41-ENTRY-SHORTLIST-FREEZE`
(≤ 3) → `FT2-50-ENTRY-OUTER-EVALUATION` → `FT2-51-INDEPENDENT-ENTRY-AUDIT`
(rich-lane escape on `no_genuine_entry_signal`; profile tripwire;
insufficient-evidence terminal) → `FT2-52-OWNER-ENTRY-FREEZE`.

**Phase D — lifecycle** — `FT2-60-LIFECYCLE-DESIGN-FOR-FROZEN-ENTRY` →
`FT2-61-PARALLEL-LIFECYCLE-REVIEW` → `FT2-62-OWNER-LIFECYCLE-APPROVAL` →
`FT2-69-LIFECYCLE-HARNESS-PILOT` → `FT2-70-INDEPENDENT-LIFECYCLE-MACHINERY-
ACCEPTANCE` → `FT2-71-LIFECYCLE-INNER-AUTORESEARCH` → `FT2-72-LIFECYCLE-
SHORTLIST-FREEZE` → `FT2-73-LIFECYCLE-OUTER-EVALUATION` → `FT2-74-INDEPENDENT-
LIFECYCLE-AUDIT` (harvest + profile tripwires). `FT2-68-BOUNDED-LIFECYCLE-
MECHANICAL-REPAIR` serves the Phase-D machinery.

**Phase E — integration** — `FT2-80-FOUR-BOX-COMBINED-AUDIT` →
`FT2-81-FAILURE-ATTRIBUTION` (on interaction/invalid) / `FT2-82-COMPLETE-SYSTEM-
FREEZE` (on accept).

**Phase F — protected and live** — `FT2-90-FRESH-CONFIRMATION` →
`FT2-91-PROTECTED-HOLDOUT` → `FT2-92-IBKR-DECISION-SHADOW` → `FT2-93-NO-ORDER-
LIVE-SHADOW` → `FT2-94-INDEPENDENT-LIVE-SHADOW-AUDIT` → `FT2-95-OWNER-PAPER-
AUTHORIZATION` → `FT2-96-GUARDED-IBKR-PAPER-VALIDATION` → `FT2-97-INDEPENDENT-
PAPER-EVIDENCE-REVIEW` → `STOP-PAPER-EVIDENCE-COMPLETE`. Paper completion is not
real-money authorization.

### 8.2 Terminals, loop policy, tripwires

Terminal set (matches the JSON `terminal_states`): `STOP-OWNER-DECISION`,
`STOP-REDESIGN-REQUIRED`, `STOP-CANDIDATE-REJECTED`, `STOP-INSUFFICIENT-EVIDENCE`,
`STOP-PAPER-EVIDENCE-COMPLETE`. None may be narrated as success.

The loop may change weights, architecture, loss (within the frozen target
family), optimizer, regularization, and calibration implementation — nothing
else. The plateau counter, tranche sizes, and evidence contract are **graph
state, not loop state**; the loop cannot touch its own stopping rule. A "serious
trial" is declared in the registry with its hypothesis before results are seen.

Drift tripwires carried from V1 (flat-model-cannot-WAIT; cannot-choose-full-
ladder; direction/strike permanently hard-coded; P5/nearest-ATM becomes final;
HOLD/EXIT before entry gate; producer grades own evidence; entry-only claimed as
complete trader) **plus Rev C additions**: four-bucket profile grossly-off at
freeze; harvest-ratio collapse under floor-on.

### 8.3 Loop budgets and compute tranches (D29)

Loop budgets (JSON `controller_policy.loop_budgets`): `design_repair_data_tensor_
label` 2, `design_repair_entry_science` 2, `design_repair_evidence_statistics` 2,
`design_repair_lifecycle` 2, `mechanical_repair` 2, `scientific_redesign` 3. The
12-serious-trial plateau survives; unbounded spend does not.

**Tranche authorization (amends D29):** GPU spend in owner-authorized tranches
(**$20/tranche**; owner-amended from $150 at signing, 2026-07-28). First-tranche
prerequisites: the ≤ $10
checkpoint/resume dry-run proof **and** the FT2-11 MDE computation showing a
plausible edge is detectable. Tranche exhaustion → `resource_owner_decision_
required`. **14-day owner check-in** regardless of tranche state. No local
training during recorder hours (~05:30–13:15 local on collection days). Every GPU
run checkpoints durably so a killed lease loses ≤ 1 trial.

### 8.4 Hardware plan and the arithmetic the brief was missing

| Workload | Where |
|---|---|
| Opportunity census, label builds | M5 (CPU-parallel), hours-scale, checkpointable |
| Machinery pilots, smoke trainings, fp16 equivalence test | M5 (MPS), overnight/weekend |
| Entry + lifecycle inner autoresearch loops | Leased GPU (~100–300 GPU-hours) |
| All inference, shadow, live | M5 (forward pass is milliseconds; the live risk is feature-build latency — FT2-30 measures it) |

Decision surface: 301 pass-only sessions × 359 flat rows = **108,059 flat-state
minutes**; open-state extension ≤ 7,525 minutes; ladder 42 slots + WAIT;
~1.6–3.2M contract-minute path labels (one-time, CPU-parallel, checkpoint per
session, M5-feasible). **Trades are the statistical bottleneck, not minutes:** at
1–3 trades/day, ~300–900 trades corpus-wide; an outer-fold test slice (~60
sessions) holds 60–180 trades — hence session-level ESS, session-block bootstrap,
`insufficient_evidence`, and MDE-before-spend. Naive fp32 materialization ≈ 14 GB
exceeds 16 GB headroom → identity-keyed storage with per-batch assembly and the
§8.5 precision law. Primary model ~1–5M params; M5 ~5–15 min/epoch (~3–15 h/run,
one pilot fine; campaign not); campaign runs leased. FT2-30 must publish actuals;
deviation > 2× any estimate is a mechanical finding.

### 8.5 Storage precision law (D47)

Exact types for exact things: contract identities, strikes, session dates,
timestamps as integers; option prices as integer ticks (SPXW 0.05/0.10) or
fixed-point. **fp32** for precision-sensitive market/context fields (SPX levels,
Greeks, path differences). **fp16 only** for normalized model-input tensors, and
only after FT2-30 demonstrates quantified prediction-and-decision equivalence
(identical actions on a reference slice; score drift below a preregistered bound)
vs the fp32 reference. Contract-identity continuity must never silently re-map a
slot's history.

---

## 9. Documented-Limitations Register (D53)

Known, accepted limitations of campaign 1 — part of the signed authority.
Removing a limitation mid-campaign is a scientific change requiring owner review.

1. **Minute cadence:** decisions fire on completed minutes; the trader is
   structurally ~1 minute late to breakouts. This biases learnable edge toward
   structural moves and away from scalps — consistent with the anti-scalper
   identity, but breakout-scalp alpha is out of scope by construction.
2. **Myopic entry:** WAIT triggers when nothing qualifies *now*; there is no
   "a better setup is likely at 10:30" forecast. Phase-calibrated selectivity is
   the patience proxy for campaign 1. A look-ahead opportunity head is Stage-3
   (sequential agent) territory.
3. **No intraminute protection:** the floor and all exits evaluate at
   completed-minute bids; intraminute gaps are absorbed as slippage and measured
   (§5.3), not prevented.
4. **Entry model does not see own recent PnL:** flat-state features are
   market/ladder state; no tilt, no hot-hand — by design. Account state affects
   only masks (affordability, D48 cap, D49 budget).

---

## 10. Paper-Readiness Meaning And Freeze Requirements

Offline success earns only: "Frozen Full Trader candidate eligible for
paper-readiness validation." Paper readiness additionally requires
candidate-specific historical/IBKR tensor+action transfer; no-order live
operation on the M5 with Gateway; reconstructable WAIT/entry/HOLD/floor/EXIT
decisions; stale-data/reconnect/incomplete-ladder/forced-flat behavior;
guard/rollback readiness; independent live-shadow acceptance; and owner
authorization. An entry-only model is never paper-ready. A frozen bundle includes
weights, architecture, feature names/order, tensor schema, 90-min semantics,
contract-identity rules, phase definitions, target definitions, calibration
transforms, composer thresholds, action masks, fees/fill assumptions, simulator
version/config, fold/role manifests, source/dataset hashes, training-code commit,
model/artifact hashes, latency/memory measurements, and the independent
acceptance receipt. Any dependency change creates a new candidate identity.

---

## 11. Graph V2 Validation (recorded output)

The read-only validator confirms the machine-readable graph is well-formed. It
never trains, touches protected evidence, or contacts a broker. Output:

```json
{
  "edge_count": 103,
  "errors": [],
  "graph_id": "protocol101-full-trader-v2",
  "node_count": 47,
  "reachability_root": "FT2-00-GRAPH-RESET",
  "reachable_from_root": 47,
  "schema_version": "Protocol101FullTraderGraphV2",
  "terminal_states": [
    "STOP-CANDIDATE-REJECTED",
    "STOP-INSUFFICIENT-EVIDENCE",
    "STOP-OWNER-DECISION",
    "STOP-PAPER-EVIDENCE-COMPLETE",
    "STOP-REDESIGN-REQUIRED"
  ],
  "model_training_nodes": [
    "FT2-40-ENTRY-INNER-AUTORESEARCH",
    "FT2-71-LIFECYCLE-INNER-AUTORESEARCH"
  ],
  "valid": true
}
```

Checks passed: JSON parses; schema is `Protocol101FullTraderGraphV2`; 47 unique
node ids; all 103 edges reference defined nodes with unique `(from, outcome)`
routes; every non-terminal node has ≥ 1 outgoing edge; every terminal node has 0
outgoing edges; all 47 nodes reachable from the reset root; the `kind==terminal`
set equals the declared `terminal_states`; both `model_training` nodes carry a
valid `action_space`; no `independent_acceptance` node is run by the executor
role; all 8 `gate_dominance_requirements` hold (each gate removed makes its target
unreachable); every edge `loop_budget` names a defined budget; and `initial_state`
matches the consolidation contract (FT2-01 complete, current_node FT2-04, parked,
owner_start_required true).

---

## 12. Decision Ledger D01–D53

Every decision appears exactly once, with status and provenance (which
revision/answer settled it). "Brief §27" = 2026-07-28 planning session.

| ID | Decision | Status | Settled by |
|---|---|---|---|
| D01 | Product is learned `WAIT or exact contract`, then learned `HOLD or EXIT` | RETAINED | Brief §27 |
| D02 | Full governed ladder is 42 nominal contracts, not three near-ATM | RETAINED | Brief §27 |
| D03 | Entry timing, direction, strike are learned | RETAINED | Brief §27 |
| D04 | 90-minute rolling causal history + masks + clock fields | RETAINED | Brief §27 |
| D05 | Seven explicit Pickles-style market phases | RETAINED | Brief §27 |
| D06 | Forecast several entry path properties separately | RETAINED | Brief §27 |
| D07 | Forecast at {3,5,10,20,45,90,session}; prune only with pre-validation redundancy evidence | RETAINED (Rev C reverts Rev A trim) | Rev C §9 / Codex B3 |
| D08 | Whole-path entry quality is the primary entry gate | RETAINED | Brief §27 |
| D09 | Early drawdown, pre-profit MAE, underwater burden, profit-window stability are primary evidence | RETAINED | Brief §27 |
| D10 | Quality-first selection, then conservative upside | RETAINED | Brief §27 |
| D11 | Calibrate quality guardrails on training roles, freeze before outer validation | RETAINED | Brief §27 |
| D12 | Uncertainty-aware `WAIT` | RETAINED | Brief §27 |
| D13 | No hard trades/day maximum; frequency judged economically | RETAINED | Brief §27 |
| D14 | Rare-trade candidates require evidence sufficiency, not a fixed minimum | RETAINED | Brief §27 |
| D15 | Joint temporal full-ladder neural primary with simpler controls | RETAINED | Brief §27 |
| D16 | Require canonical price history, geometry, moneyness, internal Greeks | RETAINED | Brief §27 |
| D17 | Predict both dollars per contract and percentage economics | RETAINED | Brief §27 |
| D18 | Core and ladder-rich challengers; campaign 1 = one lane + rich-lane escape route | AMENDED | Rev B §9 (Codex Q1) |
| D19 | P5 is a mandatory benchmark, never final architecture | RETAINED | Brief §27 |
| D20 | Final trader must earn more profit than P5 with bounded risk | RETAINED | Brief §27 |
| D21 | Broad causal lifecycle pretraining precedes specialization to frozen OOF entries | RETAINED | Brief §27 |
| D22 | Exit predicts multi-horizon hold advantage, downside, giveback, slot opportunity cost | RETAINED | Brief §27 |
| D23 | Entry model runs no-action shadow while a position is open | RETAINED | Brief §27 |
| D24 | Forecast-derived completed-minute protective floor that only ratchets upward | RETAINED | Brief §27 |
| D25 | Re-entry may occur next completed minute; no arbitrary cooldown | RETAINED | Brief §27 |
| D26 | Neural and HGB lifecycle candidates compete on identical evidence | RETAINED | Brief §27 |
| D27 | Keep entry and exit separate in campaign 1; no joint fine-tuning | RETAINED | Brief §27 |
| D28 | Nested chronological cross-fitting | RETAINED | Brief §27 |
| D29 | Stop after 12 serious non-improving trials; plus tranches, MDE prerequisite, 14-day check-in | AMENDED | Rev B §9 (Codex Q4) |
| D30 | Freeze at most three candidates for one-shot outer evaluation | RETAINED | Brief §27 |
| D31 | Balanced evidence with strict integrity and bounded fold downside | RETAINED | Brief §27 |
| D32 | Four-box attribution before complete-system freeze | RETAINED | Brief §27 |
| D33 | Automatically repair only mechanical defects | RETAINED | Brief §27 |
| D34 | Preserve owner pauses after entry selection and before lifecycle training | RETAINED | Brief §27 |
| D35 | One consolidated live dashboard | RETAINED | Brief §27 |
| D36 | No early Mac model-size caps; §8.4 estimates published now, FT2-30 measures actuals | CLARIFIED | Rev B §9 |
| D37 | Preserve protected holdout and separate paper authorization | RETAINED | Brief §27 |
| D38 | Create Graph V2, preserve Graph V1 unchanged | RETAINED | Brief §27 |
| D39 | Census is a feasibility/power diagnostic after label freeze; hard-stop only on genuine impossibility; otherwise informs owner approval | REVISED | Rev B §9 (from Rev A) |
| D40 | Open-state rows 15:31→15:55 as governed deliverable | NEW | Rev B §9 |
| D41 | Admission: 3 preregistered transforms; positive/null controls; session-block 95% CI upper bound < 0.55; transfer ≥ 0.99; bias/boundary primary; `insufficient_paired_evidence` legal | REVISED | Rev B §9 (Codex B2) |
| D42 | Census = governed corpus MINUS all outer-test slices MINUS protected-holdout sessions MINUS owner-reserved confirmation sessions once frozen; FT2-04 packet proves disjointness from BOTH outer-test union AND protected holdout | REVISED; owner-amended 2026-07-29 | Rev B §9 (Codex B1); 2026-07-29 protected-resource correction |
| D43 | Floor-slippage is first-class lifecycle evidence | NEW | Rev C §9 |
| D44 | Full-ladder controls extend signed D1 V2 matching law unchanged | NEW | Rev B §9 |
| D45 | One consolidated authority document + machine-readable Graph V2 JSON before signature (FT2-01) | NEW | Rev B §9 (Codex Q3) — **satisfied by this document** |
| D46 | Paired-evidence manifest inventory (FT2-24) precedes admission; sealed-day governance binding | NEW | Rev B §9 (Codex Q2) |
| D47 | Storage precision law: ints/ticks exact, fp32 sensitive, fp16 only after equivalence proof | NEW | Rev B §9 (Codex B4) |
| D48 | Hard per-trade premium cap: premium + round-trip fee ≤ 5% of session-start equity; identical offline and live | NEW, OWNER-ANSWERED 2026-07-28 | Rev C §0.5 T1 |
| D49 | Budget-aware entries: realized session loss + new premium at risk ≤ 5% daily budget; breaker keeps signed realized-only basis | NEW, OWNER-ANSWERED 2026-07-28 | Rev C §0.5 T2 |
| D50 | Four-bucket profile tripwire at freeze gates: grossly-off → `owner_decision_required`; never a gate, never an optimizable target | NEW, OWNER-ANSWERED 2026-07-28 | Rev C §0.5 T3 |
| D51 | Guardrails are bottom-tail exclusion; conservative upside ranks; FT2-10 justifies operating point against census trade-rate AND excluded-winners curves | NEW, OWNER-ANSWERED 2026-07-28 | Rev C §0.5 T4 |
| D52 | Floor ablation (learned-exit-without-floor comparator), time/profit-conditional floor, harvest-ratio tripwire | NEW | Rev C §9 |
| D53 | Documented-limitations register (§9) is part of the signed authority; removing a limitation mid-campaign requires owner review | NEW | Rev C §9 |

---

## 13. Highest Allowed Claim

> Protocol101 Full Trader Graph V2 has a consolidated, validated, signable
> authority package awaiting owner signature.

No model is designed, trained, selected, or eligible for anything by this
document. No protected resource, broker path, paid download, paid compute, or
runtime change is authorized by this document. Signing authorizes Phase 0
documents work only.

---

## 14. Owner Signature Section

Signing this document (a) adopts the merged authority above as the single plan of
record, and (b) authorizes **Phase 0 documents work only** (FT2-04 label freeze
and FT2-05 census). All later phases remain behind their own owner gates.

**Twelve decisions carried to signature.** The four owner-answered items are
recorded with their 2026-07-28 provenance and re-appear here for formal
signature; the eight open items are the substantive decisions from Rev C §10 that
still need your explicit sign-off.

Owner-answered 2026-07-28 (recorded provenance, formalized by signature):

- [x] **D48** Per-trade premium cap 5% — owner-answered 2026-07-28 (Rev C §0.5 T1).
- [x] **D49** Budget-aware entries — owner-answered 2026-07-28 (Rev C §0.5 T2).
- [x] **D50** Four-bucket profile tripwire — owner-answered 2026-07-28 (Rev C §0.5 T3).
- [x] **D51** Loose screen + upside ranking — owner-answered 2026-07-28 (Rev C §0.5 T4).

Open sign-off items (Rev C §10):

- [x] Census as feasibility/power diagnostic after label freeze (D39).
- [x] One-lane campaign 1 with rich-lane escape route (D18).
- [x] Approved horizons {3,5,10,20,45,90,session} retained (D07).
- [x] Admission law: caps + CI-based ceiling + transfer/bias primary (D41).
- [x] Compute tranches at $20 (owner-amended from $150 at signing) + MDE prerequisite + 14-day check-in (D29).
- [x] Open-state row extension 15:31→15:55 (D40).
- [x] Sealed-day governance binding on admission evidence (D46) — **owner-confirmed 2026-07-28** (admission testing restricted to development-class days; burned/validation/sealed days untouchable for that purpose).
- [x] Storage precision law (D47).

Post-signature amendments:

- Owner amendment 2026-07-29: D42 corrected to exclude protected resources from
  the census set (original formula omitted the protected holdout; caught at
  FT2-04 packet verification before any census statistic was computed). Approved
  in the Fable planning session; FT2-04 repaired accordingly.

Every owner-gate packet in Graph V2 must open with a one-page plain-English memo:
the question, the options, the evidence in trader terms, the recommendation, and
what happens next under each choice. No owner decision may require reading model
internals.

Owner signature: Owen Heidenreich  Date: july-28-2026

---

## Appendix A — Resolved Conflicts

Conflicts between sources, resolved by precedence (signed contracts > Rev C >
Codex brief). Nothing substantive was resolved silently.

| # | Source A | Source B | Winner / resolution | Why |
|---|---|---|---|---|
| R1 | Codex brief §17.2: "G8 can no longer remain report-only for this architecture" | Signed G8 v2: "G8 is report-only… before confidence controls behavior a separate action-conditioned gate must be preregistered, smoke-tested, independently audited, owner-signed" | **Harmonized (signed contract governs).** G8 stays report-only; the brief's need is met by adding the required *separate* action-conditioned calibration gate to the FT2-10 contract (§4.6). Not a contradiction — a dependency. | Signed contract is highest precedence and already anticipates this architecture |
| R2 | Codex brief §7.1: mandatory feature floor requires per-contract canonical price history | Signed synchronization decision: exactly 17 features; Family C per-slot option-price paths quarantined at AUC 0.550524 > 0.55 | **Signed contract wins; amendment must be earned.** No silent amendment; per-contract price history may enter only through the bounded admission node FT2-25/26 (§3.3) producing an owner-signable synchronization amendment; else `STOP-OWNER-DECISION` | Signed contract outranks brief; Rev C §5 codifies the earn-it path |
| R3 | Rev A: census "outer-train only" firewall | Rev C/B (D42): census sessions in NO outer-test slice of any fold, manifest-proved | **Rev C/B wins** (higher precedence than superseded Rev A) | Codex B1 showed the outer-train rule leaks across folds |
| R4 | Rev A: recorder "collecting through the present" | Rev B verification: sealed-day classes (burned/validation/development/sealed); only 3 development days | **Rev B verification wins** (§3.4) | Fresh 2026-07-28 check; Rev A overstated availability |
| R5 | Codex brief §5.3 hard-coded safety (affordability only, no per-trade cap) | Rev C D48/D49 + Charter commitment 3 (worst day 3–5%) | **Rev C wins (owner-answered).** 5% per-trade premium cap + budget-aware entries added to the product contract (§1.4) | Owner decided 2026-07-28; brings the trader back to the Charter |
| R6 | Codex brief §21: `FT2-10-FULL-TRADER-DESIGN` single mega design node; brief was itself a delta over prior text | Rev C §3–§4: split into FT2-08/10/11; single consolidated authority (D45) | **Rev C wins.** FT2-10 is repurposed as the entry-science contract; the design contract is split three ways; this document is the single authority | Rev C outranks the brief; Codex Q3 required one authority |
| R7 | Rev A: horizon trim | Codex B3 / Rev C (D07): approved set {3,5,10,20,45,90,session} restored | **Rev C wins** (§4.1) | Short-trade resolution is an owner emphasis; pruning needs pre-validation redundancy evidence |

---

## Appendix B — Owner Questions (unresolved conflicts)

**None.** Consolidation surfaced no substantive conflict that precedence could
not resolve and none whose resolution would require inventing science. The four
trader-identity conflicts Rev C raised (T1–T4) were already put to the owner and
answered on 2026-07-28 (now D48–D51). The seven cross-source conflicts in
Appendix A were all resolved cleanly by precedence. The eight open items in §14
are **decisions awaiting signature**, not unresolved conflicts — they are carried
to the owner as sign-off items, exactly as Rev C §10 left them. If, on reading,
the owner disagrees with any Appendix A resolution, that item converts to an
owner question at signing.

---

## Appendix C — Consolidation-Completeness Checklist

**Decision ledger.** All 53 decisions D01–D53 appear exactly once in §12
(verified by enumeration).

**Rev C amendments present.** D48 §1.4; D49 §1.4; D50 §6; D51 §4.5; D52 §5.3; D53
§9; same-game law §1.4; profile tripwire §6; harvest tripwire §5.3; floor
ablation/breathing §5.3; census additions (excluded-winners, friction,
trade-rate) §7.2; rich-lane escape §3.5; per-contract design_repair budgets §8.1;
tranche/MDE/14-day §8.3; storage precision §8.5; documented-limitations §9. All
present.

**Codex brief section map** (every brief section → consolidated section or
"superseded by" note):

| Brief § | Consolidated location / disposition |
|---|---|
| §1 What this is | §0 memo; superseded framing (now a signable authority, not a review brief) |
| §2 Why a new pass | §8 (topology rationale); historical, folded into §8.1 |
| §3 Harness/loops/graph | §8.2 (loop policy) |
| §4 Existing authority + conflict | §3.2, Appendix A (R1, R2) |
| §5 Product contract | §1 |
| §6 Full ladder / same-game | §2 |
| §7 Feature strategy | §3 |
| §8 Flat-state entry model | §4 |
| §9 Entry targets | §4.2–§4.4 |
| §10 Composer and WAIT | §4.5–§4.6 |
| §11 Entry evaluation | §4.7 |
| §12 Open-state HOLD/EXIT | §5.1–§5.2 |
| §13 Protective floor | §5.3 |
| §14 Exit evaluation | §5.4 |
| §15 Four-box | §5.4 |
| §16 Data roles / anti-overfitting | §6 (nested cross-fitting, plateau, shortlist) |
| §17 Evidence standard | §6 |
| §18 Reward-hacking controls | §6.2 |
| §19 Hardware | §8.4 |
| §20 Graph is coordination | §8 |
| §21 Topology | §8.1 (SPLIT/REV per Rev C) |
| §22 Loop/failure policy | §8.2 |
| §23 Dashboard | §8.2 note + §14 (plain-English memos); D35 |
| §24 Freeze requirements | §10 |
| §25 Paper-readiness | §10 |
| §26 Drift tripwires | §8.2 |
| §27 Settled decisions | §12 (D01–D38) |
| §28 Not-yet-frozen items | §4–§6 defaults + §8.1 (resolved into the split FT2-08/10/11 contracts) |
| §29 Requested Fable review | Superseded — the review produced Rev A→C; dispositions in Appendix A |
| §30 Highest allowed claim | §13 (updated to the FT2-01 claim) |

**Deliverable hashes.**

- Graph V2 JSON: SHA-256 `b06a26be59307c130da84f2dc5b6f3224c272e6c4093e83abd5bc0b280ca6d09`.
- This narrative authority: SHA-256 recorded in the FT2-01 completion note
  (a file cannot contain its own final hash); recompute with
  `shasum -a 256 <this file>`.

**Validation.** Graph V2 validator: `valid: true`, 0 errors (§11).
