# The learning content of the SPXW 0DTE lifecycle model

**Job 46, Phase 4 design. Written 2026-08-16 in answer to
[`DESIGN_BRIEF_FOR_FABLE.md`](DESIGN_BRIEF_FOR_FABLE.md).** Controlling state is
[`STATUS.md`](../../STATUS.md); authorization is
[`DEVELOPMENT_CHARTER_2026_08.md`](../../governance/DEVELOPMENT_CHARTER_2026_08.md); the phase
ordering and the exit-scoring law are in [`PLAN.md`](PLAN.md).

This document decides **what the model looks at and what it learns**. It writes no governance,
reopens no signed law, and authorizes no fit. Where it needs a rule the owner has not made, it says
so in one line and stops (§7).

---

## 1. What this design is aimed at

The last fitted model was attributed and the finding was structural, not statistical: **V5 became an
additive option-geometry sensor.** Its out-of-fold scores decompose into a minute-common state
offset plus a contract-only term, reconstructed to `2.38e-7`; within a minute its node term
correlated `+0.851` with vega, `+0.788` with absolute delta, `+0.718` with moneyness; and its
magnitude loss never used the WAIT or EXIT outputs, so those heads received no gradient at all.
Because the state contributed the *same* offset to every contract in a minute, **the tape could not
reorder the ladder** — the model chose the same contract it would have chosen with the chart
deleted. Sources: `v4/audit/autoresearch/causal_day_v5_instability_attribution_2026_08_14_attempt002/`
and cycles 1 and 4 of [`../entry-exit-attribution/LOG.md`](../entry-exit-attribution/LOG.md).

`compact_shared_lifecycle` fixed the *architecture* — `direction(state)×is_call` and
`depth(state)×moneyness` mean state now **can** reorder — but nobody asked whether the *features*
carry information capable of doing so. Feeding the same inputs to more parameters buys a larger
geometry sensor.

**The owner's test, applied to every field below:** *could this feature change which strike the
model prefers, or does it only describe the contract?* Minute-common fields can only reorder through
the two interaction channels; per-contract fields reorder directly. A contract in which nothing can
reorder is V5 again.

## 2. The honest prior, and the one gap worth designing into

Every feature family that has been fitted here is measured dead:

- **Chart, clock and geometry state** — the 375-cell conditional drift census (ledger row 338) found
  no positive cell and 77 significantly negative ones. A rule is a function of state; a census of
  states with no positive cell admits no rule built on them.
- **Price and volume, with free contract choice** — rows 339 and 340. On quotes, trained out-of-time
  and priced at the mid with the spread removed *entirely*, the selective policy averages **−$3.5
  per trade**. The edge is absent before costs, not eaten by them.
- **Greeks** — row 336. Recomputed from price and added to the entry features, they bought nothing;
  the apparent correlation with dollar excursion was largely contract size.
- **Exit rules of every formulation** — row 341. The fitted stopping rule loses to a five-minute
  stopwatch at both the bid and the mid.

Row 338 names its own reopening conditions — the information sources no tested feature set
contained. Checked one at a time against what this project may actually use:

| Row 338's named source | Status for this design |
|---|---|
| Cross-asset context | **Unavailable.** The owner's 2026-08-14 instruction is SPXW and SPX only; ES may remain in historical evidence but may not enter active policy inputs. |
| Term structure | **Cannot exist.** The corpus is audited 0DTE-only — every quoted instrument expires the same session (251-session scope audit, cycle 6). |
| Event calendar | **Blocked by scope, not by evidence.** Charter §2 authorizes exactly one purchase and no other data source. See §7.1. |
| Order-flow imbalance | **Partly available today, and never used.** |
| Chain-wide skew | **Available today, and never used as state.** |

That last pair is the whole opening. **The ladder tensor already carries `bid_size`, `ask_size` and
a whole-chain IV surface, and no fitted model has ever read them as anything but geometry.**
`surface_summaries` in [`build_causal_day_dataset.py`](../../ops/build_causal_day_dataset.py)
already computes `displayed_size_imbalance` and `call_minus_put_iv` per minute — they are computed,
receipted, and have never reached a model.

So this design does two things:

1. **Spends the whole feature budget on the option market's own internals**, by capacity-neutral
   substitution — census-dead chart channels out, chain-internal channels in.
2. **Makes "can these features close the gap?" a measurement taken before the fit** (§5.2), with a
   pre-declared verdict rule, so the brief's invited answer — *the features cannot close it* — can be
   delivered with a receipt instead of after a third failed fit.

**The bar is roughly twice the largest entry effect this project has ever measured.** Random-entry
baseline **31.65%**; survival at a $10,000 account needs **45–50%**; that is **≈+14 points** against
a best-ever **+7.1pp** that itself failed chronological stability. Stated here, before any outcome,
so a mediocre result cannot be talked into a good one afterwards.

The narrow reason to proceed is unchanged: **large winners demonstrably exist** — 36.7% of random
entries exceed +50% excursion and perfect foresight keeps **+$623/trade at the mid** — and no design
has yet been given features capable of separating them before entry.

---

## 3. Deliverable A — the feature contract, field by field

### 3.1 State vector (minute-common): 24 fields

Reordering power is **indirect**, through `direction(state)×is_call` and `depth(state)×moneyness`.

**Tape — 4 fields** (from the underlying candle series; source ruling in §7.2)

| Field | Reorders? | Reason |
|---|---|---|
| `move_from_open_points` | via direction | Signed position in the day; the primary input to a side preference. |
| `move_15m_rel` | via direction | Recent momentum; the single representative of the seven-lookback family. |
| `range_position` | via depth | Where price sits in the developing range; a depth-preference input. |
| `realised_vol_15m` | via depth | Realised leg of the vol state; pairs with `atm_iv_change_5m` to express realised-vs-implied. |

**Barred from the state**, each for a stated reason: `body_points`, `upper_wick_points`,
`lower_wick_points` (one-minute candle anatomy — census-dead and the noisiest channels V5 had);
`return_1m` (dominated by the 15-minute term); `close_from_running_high_points`,
`close_from_running_low_points` (redundant with `range_position`); `realised_vol_5/30/60m`
(collinear with the retained 15-minute term); `log_volume`, `cumulative_volume`,
`volume_vs_expanding_median` (see §7.2 — the retained underlying source carries no volume, and the
volume that existed was censused).

**Clock — 5 fields, unchanged:** elapsed fraction, remaining fraction, morning flag, `sin`, `cos`.
Cheap, era-symmetric, and they carry the theta regime the afternoon depends on (job 40 measured
afternoon theta running 2.4× faster with 1.75× the gamma on a contract costing 40% less).

**Chain internals — 7 fields. This is the substance of the design; none has ever been in a fitted
feature set here.**

| Field | Definition (causal: snapshot at or before the decision minute) | Why it can reorder |
|---|---|---|
| `call_minus_put_iv_median` | Median call IV − median put IV across the live chain (a risk reversal). Already computed as `call_minus_put_iv` by `surface_summaries`. | Signed demand skew. Through `direction×is_call` it can **flip the preferred side** while every geometric quantity is unchanged. Row 338's "chain-wide skew". |
| `skew_change_15m` | 15-minute change in that risk reversal. | Skew **dynamics** — repositioning in progress, not a level. A level is geometry; a change is behaviour. |
| `atm_iv_change_5m` | 5-minute change in near-ATM IV. | Vol-regime shift at the decision minute. The *level* stays out of state deliberately (censused, and per-contract `self_iv` already carries it); the *change* is new. |
| `chain_depth_imbalance` | (Σ`bid_size` − Σ`ask_size`) / (Σ`bid_size` + Σ`ask_size`) over the live chain. Already computed as `displayed_size_imbalance`. | The closest object this corpus owns to **order-flow imbalance** — row 338's first named source. Displayed size is in the tensor today and has never been read. |
| `put_call_depth_ratio` | Displayed depth on puts against calls. | Side-resolved leaning; feeds the side interaction independently of IV. |
| `smile_curvature` | Quadratic term of the per-side IV smile fit. | How the chain prices tails against the body — a depth-preference input that is not the contract's own geometry. |
| `implied_spot_dispersion_bps` | Cross-strike dispersion of the parity spot `S = K + C − P` near the money. | Microstructure stress; doubles as the QC channel for the parity spot the backfill era depends on (§4.3). |

**Checked against the issued G3 ledger, not assumed.** Five of these seven are **exact ADMITTED
rows** — `chain_depth_imbalance`, `put_call_depth_ratio`, `smile_curvature`, `opra_atm_iv_change_5m`,
`opra_implied_spot_dispersion_bps`. The remaining two (`call_minus_put_iv_median`,
`skew_change_15m`) are **derived from admitted parents** (`opra_call_skew`, `opra_put_skew`,
`side_smile_slope`) and are flagged as needing their own v5 ledger row before any parity, shadow or
paper claim. The certification expires **2026-11-10**; a fit after that date needs a reissued ledger,
which `assert_features_admitted` enforces fail-closed.

**Ladder context (mean and max over the live chain) — 6 fields**, reduced from ten: mean and max of
`spread`, `self_iv`, `self_theta_per_minute`.

Two removals, both load-bearing:

- `self_gamma` aggregates — deterministic given moneyness, IV and time-to-expiry for a 0DTE chain.
  Pure geometry, and the same family as V5's strongest correlate.
- **`moneyness_itm_points` aggregates — barred as an era detector.** The max of signed moneyness over
  the whole live chain *is* the ladder's edge, and 2022 ladders are roughly a third the width of 2025
  ones. Any raw width, count or extent field lets the model read the calendar instead of the market.
  **This bar generalises: no field whose value is a monotone function of ladder size may enter the
  state.** Verified by diagnostic D7.

**Account — 2 fields, unchanged:** cash fraction and remaining trade-cap fraction, required by the
serial law. **Flagged:** the G3 family `entry.causal_account_state.v1` is **BARRED** in the admission
ledger for missing historical/live transition receipts. That is legal inside development (the
charter authorizes construction on owned data), but it must be re-certified before any parity,
shadow or paper claim. Recorded here so it is not discovered at Phase 6.

### 3.2 Per-contract fields: contract base widens 5 → 6

These reorder **directly** — they are the only channel by which one contract outranks another
without the state's help.

| Field | Status | Reason |
|---|---|---|
| `ask` | keep | The executable cost the label charges; the anchor of the ENTER-versus-WAIT trade. |
| `spread` | keep | The toll, and the tradeability ordering. |
| `self_iv` | keep | Price of this contract's volatility. Per-contract, so it reorders. |
| `self_theta_per_minute` | keep | The decay the WAIT decision is priced against. |
| **`smile_residual`** | **add** | `self_iv` minus the fitted same-side smile at this contract's moneyness. **Relative value against its own neighbours.** The smile fit is geometry; the *deviation from it* is the market disagreeing with its own surface — the one per-contract quantity here that is informational rather than descriptive. |
| **`contract_depth_imbalance`** | **add** | (`bid_size` − `ask_size`)/(`bid_size` + `ask_size`) at this strike — who is leaning on this contract specifically. A per-contract order-flow proxy, and the field most likely to reorder a ladder the chart cannot. |
| `self_gamma` | **bar** | Deterministic geometry for 0DTE; V5's failure mode in one column. |
| `microprice` / microprice tilt | **bar** | Algebraically near-collinear with depth imbalance; one representative kept. |
| per-contract `volume`, `open_interest` | **bar** | The G3 family `entry.opra_ohlcv1m_sparse.v1` is barred (sparse zero-fill, no multi-session receipt-latency distribution); coverage is era-asymmetric, which makes it an era shortcut; and rows 339/340 measured price-plus-volume selection at zero. Revisit only if that family's admission is repaired. |
| `bid`, `mid` | bar from base | Spanned by `ask` and `spread`. |
| `minutes_to_expiry_fraction` | bar from base | A clock duplicate on a 0DTE chain; the state's clock already carries it. |

**Interactions — unchanged, and deliberately so.** `direction(state)×is_call` and
`depth(state)×moneyness` are the architecture's only reordering channels. This design does not add a
channel; it changes what is multiplied through them. One optional third channel,
`relative_value(state)×smile_residual`, is declared as a **Tier-1 extension only if the measured
budget allows it** (§3.3) — it would let the chain's state modulate how much relative cheapness
matters.

### 3.3 Capacity — derived by building the module, never transcribed

The 2026-08-14 ruling bars asserted parameter counts. These were produced by constructing each
variant in torch and summing trainable weights; the current member reproduces its known 120 as the
control:

| Variant | State | Base | **Total** | Entry phase | Exit head |
|---|---:|---:|---:|---:|---:|
| Current `compact_shared_lifecycle` (control) | 25 | 5 | **120** | 96 | 24 |
| **Proposed (rung 0)** | 24 | 6 | **118** | 94 | 24 |
| Proposed + third interaction (Tier-1) | 24 | 6 | 122 | 98 | 24 |

**The redesign is capacity-neutral: 118 against 120, and 94 against 96 in the entry phase.** It buys
seven chain-internal state channels and two per-contract informational channels by spending
census-dead chart anatomy and redundant geometry. That matters because the external review's
confirmed finding stands: the per-fit budget binds on the *training prefix*, not the whole corpus,
and the first outer fit sees only ~40% of the chronology.

**The parameter budget is re-measured, never inherited** (brief fact 2). Re-run
[`measure_effective_sample_size.py`](../../ops/measure_effective_sample_size.py) on the built
two-era corpus and take the per-fit budgets from it. The 122–216 projection was an extrapolation
from the 243-session corpus and must not be carried forward.

**Pre-declared shrink ladder** — mechanical, so a budget shortfall is not resolved by taste after the
number is known. Drop state fields in **ascending order of the information they carry in the
Phase-4a probe** (§5.2), which is measured before any fit:

| Rung | State fields | Total | Entry phase |
|---:|---:|---:|---:|
| 0 | 24 | 118 | 94 |
| 1 | 21 | 109 | 85 |
| 2 | 18 | 100 | 76 |
| 3 | 15 | 91 | 67 |
| 4 | 12 | 82 | 58 |
| 5 | 9 | 73 | 49 |

Two constraints on the shrink: the seven chain-internal fields are the hypothesis and are dropped
**last**; and shrinking operates on **feature count, never on hidden width**, because
`causal_day_hidden_size` is a frozen knob at 3 and moving it is a knob edit, not a design choice.

**The exit head's 24 parameters need their own budget.** The external review confirmed its
trajectory-level effective sample size was never measured. The exit budget must be measured on
**trajectories**, not sessions, before the exit phase runs.

### 3.4 Admission status — checked against the issued ledger, not assumed

Every proposed field was resolved against the G3 ledger issued 2026-08-12 (51 of 73 scoped features
admitted; [`feature_admission.py`](../../research/feature_admission.py) fails closed on anything
unknown, barred or stale). The result is better than the design needed, and one row of it is a
problem to fix later rather than discover later.

| Proposed field | Ledger status |
|---|---|
| `chain_depth_imbalance`, `put_call_depth_ratio`, `smile_curvature` | **ADMITTED** — exact rows (`entry.opra_cbbo1m_cross_section.v1`) |
| `atm_iv_change_5m` | **ADMITTED** — `opra_atm_iv_change_5m` (`entry.opra_implied_volatility.v1`) |
| `implied_spot_dispersion_bps` | **ADMITTED** — `opra_implied_spot_dispersion_bps` (`entry.opra_implied_spot.v1`) |
| `contract_depth_imbalance` | **ADMITTED** — `size_imbalance` (`entry.opra_cbbo1m_native.v1`) |
| `ask`, `spread`, `self_iv`, `self_theta_per_minute` | **ADMITTED** — `option_ask`, `option_spread`, `self_iv`, `bs_theta` |
| `move_15m_rel`; the five clock fields | **ADMITTED** — `opra_spot_return_15m_bps`; `entry.contract_clock.v1` (8/8) |
| `call_minus_put_iv_median`, `skew_change_15m` | **derived** from `opra_call_skew` / `opra_put_skew`, both admitted |
| `smile_residual` | **derived** from `self_iv`, `side_smile_slope`, `smile_curvature`, all admitted |
| `move_from_open_points`, `range_position`, `realised_vol_15m` | **derived** from `entry.opra_implied_spot.v1` (admitted 8/8) |
| the two account fields | **BARRED** — `entry.causal_account_state.v1`, missing historical/live transition receipts |
| per-contract `volume`, `open_interest` | **BARRED** — `entry.opra_ohlcv1m_sparse.v1` (already barred in §3.2 on three independent grounds) |

Three consequences worth stating rather than leaving implicit:

1. **Five of the seven chain-internal state fields are exact admitted rows**, and the other two derive
   from admitted parents. The hypothesis of this design is not resting on features that have no live
   arrival evidence.
2. **`derived` is not `admitted`.** A causal function of an admitted parent inherits that parent's
   arrival evidence but is **not itself a ledgered row**. Every derived field above needs a v5 ledger
   row before any parity, shadow or paper claim. This is a Phase-6 obligation, recorded now.
3. **The whole certification expires 2026-11-10** and the account-state family is barred today, so a
   fresh v5 ledger is required before runtime parity regardless of what this design chooses. The
   charter authorizes development fitting without it; it does not authorize a parity claim without it.

This check also settled §7.2's source question on evidence rather than preference: the entire
`entry.opra_implied_spot.v1` family is admitted 8/8, while ES-derived chart channels appear **nowhere**
in the option-feature ledger.

---

## 4. Deliverable B — the label at corpus scale, across two eras

### 4.1 Member P (primary) — first-touch stop-or-hold, in executable dollars

The label family is **+G before −30% within 60 minutes**, already implemented as
`first_touch_{30,50,100}pct_before_loss_30pct_60m` by `_first_touch_order` in the dataset builder,
with the declared member at **+50%**.

The builder defines first touch on quote **mids**; the entry target this design trains on is the
**executable dollar value** of the resulting lifecycle: enter at the ask, exit at the first
executable bid on or after the touch minute under the existing first-later-bid law, charge measured
fees, stop at −30% the same way, otherwise the 60-minute clock. Mid-defined ordering with
ask-in/bid-out accounting is the convention already in the builder and it is retained unchanged.

Why this label leads:

- It is the **best-conditioned target this project has priced**: base rate 31.8% at +50%/−30%,
  break-even precision **34.2%** — a **+2.4pp** gap, the smallest ever required here.
  *(Two figures for this cell exist on disk and the conflict is reported rather than resolved
  silently: STATUS row 45 and ledger row 345 carry 34.2%/+2.4pp from job 45's frozen calibration
  export, winners +$641 against losers −$333; the entry-exit LOG's 34.7%/+2.9pp came from winners
  +$636 / losers −$338 and is labelled there as advisory, to be recomputed under the frozen
  declaration. The frozen figure is used. Both reduce to `L/(W+L)` exactly, so the difference is the
  calibration, not an arithmetic error — and it must be re-measured per era on the built corpus
  regardless, per §4.3.)*
- The −30% stop is **real tail control**, not a decoration: it cuts loser dispersion from $613 to
  $239 (job 45 calibration).
- It is the owner's stated design — entry selects for least post-entry drawdown, exit cuts
  thesis-violating trades and rides runners.

**NaN discipline.** `_first_touch_order` returns NaN when the path becomes unobservable before either
threshold, and when gain and loss land in the same minute (unresolvable at minute cadence). Those
rows are **masked in the loss and never dropped**. Dropping rows by their later path is exactly the
row-331 look-ahead and the row-346 survivorship defect class; the label-coverage rate is reported per
era and per block instead.

### 4.2 Member Q (second declared member) — ENTER versus WAIT at 120 minutes

The existing action-value law
([`causal_day_action_value_targets.py`](../../research/causal_day_action_value_targets.py)):
for one $10,000 trade, does buying this exact contract at the ask and liquidating through the
120-minute bid/settlement/fee law beat preserving the slot for every later action? WAIT's floor is
$0.

Its sparsity is stated in advance rather than discovered: on the owned year only **0.72%** of
contract-actions beat waiting (median Q(enter) −$173 against median Q(wait) +$1,437). It is a
rare-event regression and is expected to be the harder member. **P leads; Q may not be promoted over
P on point estimates** — declared now, before either is fitted.

### 4.3 The era law — the settlement finding made binding

The measurement that changes the corpus build: the owned official, non-derived SPX 16:00 source
covers **251 sessions (2025-08 → 2026-07) and nothing else.** There are zero official files for
2022-06 → 2025-07. **About 76% of the finished corpus has no official settlement source**, so the
approved plan's fallback — parity spot plus the validated-cash-settlement law with a paired
zero-recovery sensitivity — **governs the majority of the corpus, not a remainder.** Five
consequences, all binding on the builder:

1. **Settlement source is a first-class per-session column**, `official_1600` or `parity_close`, and
   it appears in every receipt. A cash settlement is never called a fill.
2. **The zero-recovery twin is computed for every terminal-dependent number**, and it flows through
   **member Q's targets as a declared evaluation variant**, not merely through the reporting layer.
   *Any result whose sign differs between the settled and zero-recovery twins is not bankable* —
   stated before outcomes exist.
3. **The session grid is read from the data, never assumed.** Backfill-era sessions end at **15:58**,
   not 16:00, and only **42–53%** of rows in the closing minutes are two-sided (47–58 contracts still
   quoting). `LAST_QUOTE_MINUTE` becomes per-session. The share of exits resolving as
   `executable_bid` / `delayed_first_later_bid` / `validated_cash_settlement` / `blocked` is a
   **per-era QC gate reported before any fit** — the two eras are not interchangeable for label
   construction and must not be averaged over.
4. **Base rates are re-measured per era.** 31.65% is an owned-year measurement; 2022's volatility
   regime and one-third-width ladders will move it. The declaration carries the fixed target *and*
   the per-era baselines. The **target itself does not move** — it is account arithmetic, and it is
   signed.
5. **Era is confounded with chronology, and the design says so plainly.** The earliest-40% training
   prefix is entirely parity-settled narrow-ladder era; the late score blocks hold the owned year. A
   model can therefore "learn" the era instead of the market. Two defences: no field that is a
   monotone function of ladder size may enter the state (§3.1), and diagnostic D7 tests the state
   vector for era-classifiability directly. Composition-matched controls are drawn **within era**.

---

## 5. Deliverable C — the curriculum

### 5.1 Order of operations

1. **Corpus, then budget.** Build both eras with the §4.3 law, then re-measure effective sample size
   and set per-fit budgets by measurement.
2. **Phase 4a — the feature-information preflight (§5.2). The decision point.**
3. **Entry phase, member P.** Shared representation and entry heads train; the exit head is frozen
   and verified bitwise afterwards. Scaling and imputation are fitted per fit from strictly earlier
   sessions — the tensorizer defers these deliberately and the fit must not shortcut it.
4. **Out-of-fold trajectories, then the exit head.** Nested inner folds inside the training prefix; a
   generator may never have trained on the session whose trajectory it produces; entry weights frozen
   and verified bitwise. Both requirements are already structural in
   [`lifecycle_trainer.py`](../../research/lifecycle_trainer.py).
5. **Member Q** through the identical curriculum.
6. **Economics last:** midpoint gross first, then ask-in/bid-out with measured fees, through the
   serial $10,000 simulator under the signed 2026-08-16 risk law — $2,000 fixed ticket, the
   independently load-bearing `moneyness_band`, 20% breaker on session-starting equity, and a
   simulator-enforced stop no tighter than −40%. Surviving-path per-trade economics **and** ruin
   probability with time-to-ruin as headline numbers, never a composite (row 346).

The exit is scored on the two skills the owner ruled binding — loss averted on entries that did not
develop, and excursion captured on entries that did — each against a **duration-matched** control,
never "beat holding". That law is already written into [`PLAN.md`](PLAN.md) phase 5 and is not
restated here.

### 5.2 Phase 4a — the feature-information preflight

**This is the deliverable that decides whether the fit is worth running, and it is the cheapest
honest answer available.**

Run on the **training prefix only**, with economics unread, charged to the alpha ledger, under a
self-hashed declaration verified by its runner:

- A **≤25-parameter linear/logistic probe** of the §3 feature set against member P's label, nested
  chronological folds, session-clustered intervals. Small enough that a failure is a statement about
  the *features*, not about optimisation.
- A **known-answer twin**: the identical probe on synthetic data carrying a planted ≈+20pp precision
  lift, to establish that the probe can find an edge of the size that matters at this geometry. Job
  45 closed because a gate could not certify an edge the models had demonstrably learned; a preflight
  that cannot recover its own plant proves nothing about the features.

**Pre-declared verdict rule** (stated before the probe runs):

| Plant recovered? | Real-feature lift (Wilson upper, at the declared ~2 trades/day rate) | Outcome |
|---|---|---|
| Yes | **< +4.0pp** | **STOP and report the negative**: the available features cannot close the +14-point gap on this corpus. Brief §7's invited answer, with a receipt. |
| Yes | ≥ +4.0pp | Proceed to the full fit; the measured per-field information sets the shrink-ladder order. |
| No | any | Report **advisory — underpowered**; the go/no-go is an owner decision, not an agent's. |

The +4.0pp bar is roughly half the largest entry effect ever measured here (+7.1pp) and well under
the +14 the target needs. A feature set that cannot clear even half of a historical best in a
low-variance probe will not clear twice it in a 118-parameter fit.

### 5.3 What the model is asked to learn that is not already priced

The brief calls this the hard one, and it deserves a direct answer.

**Already priced, and therefore not the target.** Magnitude is predictable and *fully* priced: row
332 selected the busiest third, raised the realised move 51% and the premium 42%, and moved P&L by
**thirty cents**. Direction from chart state is censused dead (row 338). Timing from price and volume
is zero at the mid (row 340). Any design that rediscovers magnitude has lost before it starts.

**Not priced, and therefore the hypothesis.** The premium is *one number per contract*. It cannot
separately express two things:

1. **Cross-sectional ordering.** Which of the same minute's contracts will realise a path outcome
   better than its *own* premium implies. The premium prices each contract against its own risk; it
   does not rank them against each other on a bracketed path outcome.
2. **Chain-internal state before the tape confirms it.** When the option market's own behaviour — a
   skew shift, a depth lean, a smile deviation — indicates the coming hour will favour holders of one
   side or depth, *before* that shows up in the chart channels every prior study measured.

Member P's first-touch bracket is precisely the payoff that an ordering claim monetises: it pays for
reaching +50% before −30%, which is a statement about *path order*, not about magnitude or terminal
direction. **That is the entire hypothesis.** If Phase 4a finds no information in those channels,
nothing downstream can conjure it — which is why the preflight, and not the fit, is the decision
point.

### 5.4 How the two members differ, concretely

| | Member P | Member Q |
|---|---|---|
| Label | +50% before −30% within 60m, executable dollars | ENTER vs WAIT action value, 120m |
| Density | Dense — base ≈32% | Sparse — 0.72% of actions beat WAIT |
| Loss | Smooth-L1 on scaled dollars, NaN-masked | Smooth-L1 on the aligned joint action surface |
| Risk shape | Stop-bounded losers; the tail is cut by construction | Unbounded within the ticket cap |
| Role | **The certifiable claim** | The diagnostic claim: what the model thinks a slot is worth |

---

## 6. Deliverable D — the diagnostic suite

**V5's failure was visible only after the fit. These make it visible during.** Every diagnostic below
is computed from fits, out-of-fold scores and labels **before any bid economics are opened**, with
pass bars written into the fit declaration in advance.

| # | Diagnostic | What it answers | Rule |
|---|---|---|---|
| **D1** | **Additive decomposition.** Fit OOF scores to minute-offset + contract-term; report the interaction share of within-minute score variance. | Is this V5 again? V5 reconstructed at `2.38e-7`, interaction share ≈ 0. | Measurement; feeds D2. |
| **D2** | **Tape counterfactual.** Hold a minute's ladder fixed and substitute state vectors from other minutes of the same session; measure how often the preferred **side** flips and the preferred **depth** shifts. | Can the tape actually reorder the ladder? | **Pass/fail before economics.** ≈0 flips ⇒ geometry sensor ⇒ stop. |
| **D3** | **Channel-permutation null.** Permute the seven chain-internal state fields across minutes within a session, ladder and labels intact. | Are the new channels used, or decorative? | Unchanged OOF loss ⇒ the hypothesis is not being learned. |
| **D4** | **Geometry twin.** Same architecture, state channels zeroed, same law and folds. | Does state contribute anything at all? | The full model must beat its twin by a declared OOF margin; a tie is V5 with more parameters. |
| **D5** | **Head-gradient audit.** Per-head gradient norms logged every epoch. | Are WAIT and EXIT actually learning? | Structural test — V5's received no gradient. A zero-gradient head fails the run. |
| **D6** | **Timing/selection split.** Same-contract-across-minutes rank correlation (pure timing) against within-minute demeaned across-contract rank correlation (pure selection). | *Which* skill is any lift? | Reported per era; names the skill rather than averaging it. |
| **D7** | **Era honesty.** Skill decomposed by era and settlement source, plus a linear probe attempting to classify era from the state vector. | Is the model reading the calendar instead of the market? | Era-classifiability above a declared bar ⇒ the offending field is removed (§3.1's width bar, verified rather than asserted). |
| **D8** | **Suite calibration.** Synthetic corpus carrying (i) a planted state-dependent reordering edge and (ii) a planted geometry-only offset. | Does the suite work? | Must flag (i) as timing/selection and must **not** flag (ii). **A suite that cannot pass its own positive control is not run on a real fit.** |

D3 generalises row 337's lesson: the shuffled-**label** null structurally cannot detect an unused or
leaked *feature*, because permuting the label removes the thing the feature would relate to. To test
whether an input matters, permute **that input**.

---

## 7. Boundary items — one line each, no governance written

1. **Event calendar.** Row 338's remaining in-scope-in-principle source is excluded because charter
   §2 authorizes no data source beyond the one purchase; admitting even a free public macro calendar
   is an owner authorization, and it is the highest-value feature this design cannot use.
2. **Underlying source for the tape channels.** Designed on the **SPX parity-spot series in both
   eras** for era symmetry (official SPX 1m exists only for the owned 251 sessions, and the owner's
   2026-08-14 ruling is SPXW/SPX only, which the corpus's existing ES candles predate). Using ES
   candles instead is a one-line owner boundary ruling; nothing in the contract changes except those
   four fields' source.
3. **Trainer contract amendment.** `SessionEpisode.sell_paths` is keyed by decision-minute alone, but
   a sell path depends on **which contract** the entry model chose (§8). The key must become
   `(decision_index, ladder_column)`. Mechanical, declared, and no change to the frozen training law.

---

## 8. Deliverable E — the wiring spec

**This is the literal gap between "built" and "runnable":** `lifecycle_trainer` consumes
`SessionEpisode` records and no adapter produces them. New module
`v5/research/lifecycle_episode_adapter.py`, with builder extensions.

**CorpusIndex** (built once, receipted, hashed into the fit declaration): session →
`{root, era, settlement_source, settlement_spx, quote_grid, ladder_width_stats}`. Two data roots —
the owned tree and the SSD backfill tree — resolved here so nothing downstream knows about roots.
Session identifiers sort lexicographically, which is chronological, and `ChronologySplit` consumes
that list.

**Lazy iteration.** ~1,045 sessions × ~390 minutes × up to 1,048 contracts cannot be resident. The
adapter yields one `SessionEpisode` per session from per-session parquet, deterministically and
content-hashed. Precomputed once per session at build time: the wide **exit-value matrix**
(minute × contract) with the first-later-bid rule and the settlement branch already applied and
source-labelled — sell paths and clock exits are both slices of it.

**`entry_batch`** — every causal decision minute of the session as one padded `CausalPolicyBatch`:
candle prefixes padded under `candle_mask`, ladder padded per session under `ladder_mask`, and
`entry_action_mask` from the rebuilt `eligible_entry`. **The mask must be rebuilt, not reused:** the
ceiling moved from an equity-derived $1,300 to a fixed **$2,000 premium plus fees** on 2026-08-16, so
every mask and candidate artifact built before that date is stale and pre- and post-correction tables
may never be mixed. Both guards are independently load-bearing — the dollar cap **and** the
`moneyness_band` — because 37.3% of cap-eligible contracts are in the money. The training-time
account vector is the declared session-start state; real account evolution belongs to the serial
simulator at evaluation.

**`entry_value_usd`** — a minute × ladder-column tensor aligned to the padded ladder, member-specific
(P's stop-or-hold dollars, Q's action value), NaN off-mask and wherever the outcome is unknown, with
alignment validated at load in the style of `validate_alignment`.

**`sell_paths`** — keyed `(decision_index, ladder_column)` per §7.3, sliced from the exit-value
matrix; `generate_oof_trajectories` looks up the column the entry model actually chose.

**`held_batch_builder`** — per-held-minute batches: the same state pipeline at each held minute plus
the position vector (side, strike−spot, entry ask, unrealised P&L, MFE and MAE from mids observed so
far, minutes held, origin regime). Causal by construction — every term reads the path up to the
current minute only.

**Tests the adapter must carry:**

- **Mutate-the-future invariance** — perturb any minute after *t*; the batch at *t* is bitwise
  unchanged. The project's standing causality control.
- **Per-feature knowability audit** for every new §3 field — the minute each value is knowable, which
  is the only control that caught row 337's one-minute leak.
- Alignment of `entry_value_usd` and `sell_paths` to the padded ladder.
- A **15:58-era session fixture** and an official-settlement fixture, proving both eras build.
- Settlement-source labelling and the zero-recovery twin.
- Determinism: the same session yields a byte-identical episode hash.

---

## 9. Compliance with the do-not-retest ledger

Every row that governs this design, and how it is respected rather than skirted:

| Row | What it closed | How this design complies |
|---|---|---|
| **331** | Look-ahead filter: conditioning slot inclusion on anything measured after entry | No candidate is ever dropped for its later path; unknown outcomes are NaN-masked (§4.1). |
| **335** | Percentage-excursion labels | Both members are in **dollars**, net of the measured round trip. |
| **336** | Exit formulations; greeks bought nothing | Greeks are **reduced**, not added; `self_gamma` is barred both per-contract and in context. |
| **337** | The shuffled-label null cannot catch feature leakage | Per-feature knowability audit (§8) plus D3, which permutes the **input** rather than the label. |
| **338** | 375-cell census: no state pays | Every censused observable is dropped or demoted; the design spends its budget on the row's **own named unexplored source** — order-flow imbalance and chain-wide skew. |
| **339/340** | Selective long entry on prints and on quotes | New information family, not a new architecture on the same inputs; and the pre-fit verdict rule (§5.2) refuses to spend a fit re-establishing a closed measurement. |
| **341** | The exit is a stopwatch; measured only on random entries | The exit trains on **out-of-fold entries it will actually manage**, scored on two skills against duration-matched controls. |
| **344** | Short-vertical census | Untouched — this is long single-leg only. |
| **345** | Drawdown experiment closed at its own preflight | The label family is reused under the development charter, and the lesson is honoured: **a preflight that cannot recover its own plant proves nothing** (§5.2). |
| **346** | Survivorship in a compounding path | Surviving-path economics labelled as such; ruin and time-to-ruin as headline numbers (§5.1). |

The development charter permits construction, fitting and diagnostic evaluation on owned pre-cutoff
data; it does not turn any result here into a confirmatory claim against these rows, and this design
makes none.

## 10. What this design does not claim

It does not claim an edge exists. Every economic screen this project has run on long single-leg 0DTE
has come back negative or underpowered, and the bar here is about twice the largest entry effect ever
measured. What it claims is narrower and testable: **there is exactly one information family in this
corpus that no fitted model has ever been given, the architecture can now express it, and it can be
measured for information before a fit is spent on it.**

If Phase 4a says the information is not there, that is the answer, and it is a better one than a
third rediscovery of V5's failure.
