# Protocol101 Full Trader Graph V2 Planning And Fable Review Brief

Status: **DETAILED NON-EXECUTABLE PLANNING DRAFT FOR EXTERNAL REVIEW**

Prepared: **2026-07-28**

Audience: **Owen Heidenreich, Fable, future Protocol101 designers, and future
Codex agents**

## 1. What This Document Is

This document records the complete planning discussion for the proposed
Protocol101 Full Trader Graph V2. It is intended to be handed to Fable for
adversarial analysis before Graph V2, its scientific contracts, or any Goal
prompts are written.

This document is **not**:

- an executable program;
- a model architecture contract;
- an owner-signed scientific preregistration;
- a training authorization;
- a Goal prompt;
- a protected-evidence authorization;
- a broker or paper-trading authorization; or
- proof that a profitable model can be built.

The graph is a way to manage a large agentic research program. It describes
which research node may run next, what evidence that node must produce, who
may judge it, and where human decisions are required. The graph does not train
models by itself and cannot guarantee that the research will discover an edge.

The desired outcome is still ambitious but simple:

> Build one learned SPXW 0DTE trader that sees the synchronized market and full
> eligible option ladder, waits when no trade is justified, buys one specific
> call or put contract when justified, holds the open position while doing so
> has positive value, and exits it profitably when possible while respecting
> survival controls.

## 2. Why A New Planning Pass Was Required

The project drifted during the previous Stage-1 program.

The intended product was a learned full-ladder trader. The experiments became
progressively narrower:

1. H0-H3 trained entry scores against seven fixed exit-label shapes.
2. D1 exposed that profitable P5 exposure and a weak shuffle could make
   nonsense models look profitable.
3. M0/M1 then tested only contract selection after P5 supplied timing and
   direction.
4. The failed M0/M1 pilot was interpreted as support for retaining P5 and
   nearest-ATM selection.
5. A narrow one-minute HOLD/EXIT pilot was then attached to P5.

Those experiments are useful negative or inconclusive evidence. They did not
prove that the intended full-ladder learned trader is impossible. They tested
different and narrower scientific questions.

The reset therefore preserves the engineering and evidence while rejecting the
architectural compromise:

```text
market + full governed eligible ladder
  -> learned WAIT or BUY one exact contract
  -> learned HOLD or EXIT while open
```

The current canonical front page already records this reset:

- [Protocol101 Full Trader Program](../README.md)
- [Full Trader Graph V1](PROTOCOL101_FULL_TRADER_GRAPH_V1.json)

Graph V1 is parked before `FT-10-FULL-LADDER-MODEL-DESIGN`. No new full-ladder
training has begun.

## 3. Relationship Between Harnesses, Loops, And The Graph

The planning model follows:

- [Agents, loops, and graphs - July 28](<../../../agents, loops, and graphs - july 28.rtf>)

The three layers must not be confused.

### 3.1 Harness

The harness is the machinery around the research agents and models:

- files and data access;
- model-training environments;
- resumable state and checkpoints;
- immutable receipts and hashes;
- role and permission boundaries;
- subagent dispatch;
- test and audit interfaces;
- protected-resource controls; and
- progress reporting.

### 3.2 Loops

The entry and exit autoresearch programs are bounded improvement loops. Each
loop has:

- a frozen scientific question;
- a fixed data arena;
- allowed model changes;
- external evaluation;
- durable experiment state;
- a plateau rule;
- an independent audit; and
- a terminal result.

The loops may change model architecture, loss, optimizer, regularization, and
training settings. They may not change the product, action space, labels,
features, folds, fees, simulator, or gates after seeing disappointing results.

### 3.3 Graph

The graph coordinates the larger program:

- design;
- independent review;
- owner decisions;
- machinery construction;
- entry autoresearch;
- entry freeze;
- lifecycle design;
- lifecycle autoresearch;
- combined attribution;
- confirmation;
- holdout;
- live transfer;
- paper authorization; and
- paper evidence.

The graph's job is controlled routing and durable state, not model fitting.

## 4. Existing Authority And A Newly Exposed Conflict

### 4.1 Existing signed authority

The following remain binding unless explicitly amended:

- [Trader Charter](../contracts/PROTOCOL101_TRADER_CHARTER.md)
- [G4 and holdout revision](../contracts/PROTOCOL101_G4_HOLDOUT_REVISION_2026_07_19.md)
- [G8 calibration revision](../contracts/PROTOCOL101_G8_CALIBRATION_REVISION_2026_07_26.md)
- [Scoped synchronization decision](../../synchronization/contracts/PROTOCOL101_SCOPED_SYNCHRONIZATION_DECISION_2026_07_25.md)
- [D1 negative-control amendment](../contracts/PROTOCOL101_D1_NEGATIVE_CONTROL_AND_INCREMENTAL_EDGE_AMENDMENT_2026_07_28.md)

The current learned-lifecycle proposal is unsigned and will need substantial
revision if this plan survives review:

- [Current Stage-2 draft](../contracts/PROTOCOL101_STAGE2_OBJECTIVE_AND_GATES_PROPOSAL.md)

That draft still records a temporary P5 standardized-entry exception and a
runtime entry action described only as buy-call or buy-put. Those statements
are historical scaffolding, not authority for the newly approved learned exact
contract action.

The old Stage-1 G1-G9 language was also written for H0-H3 and seven fixed exit
labels. Graph V2 may preserve controls whose meanings still match, including
causality, fees, serial replay, G4 survival, robustness, confirmation, and
holdout discipline. It must not import old gates whose denominators or
semantics no longer match whole-path entry forecasts and learned lifecycle
actions. A new Full Trader gate amendment is required before training.

### 4.2 Synchronization conflict

The signed synchronization decision authorizes exactly 17 model-facing
features. It explicitly quarantines direct per-slot option-price paths because
the clean-window Family C source-discriminator AUC was `0.550524`, narrowly
above the frozen `0.55` ceiling.

The owner has now approved a stronger product minimum:

> Every serious full-ladder model must see canonical option premium and its
> history, contract identity, strike and moneyness, and internally recomputed
> Greeks.

These two statements conflict. This planning approval does not silently amend
the signed synchronization contract.

Graph V2 must therefore include a pre-training feature-admission node that:

1. Defines a causal canonical per-contract price representation.
2. Tests it on the existing historical/IBKR paired evidence.
3. Tests source leakage, decision transfer, boundary behavior, and economic
   materiality.
4. Freezes the accepted transform before model training.
5. Produces an owner-signable synchronization amendment.

If canonical contract-price history cannot pass that bounded admission test,
the planned product requirement is not met. The graph must stop for an owner
decision. It may not quietly train a "full-ladder" model that cannot see option
prices.

### 4.3 Evidence map for review

Fable should inspect at least these packets before accepting or revising this
plan:

- [Canonical L0/L2 design audit](<../../../../audit/autoresearch/protocol101_canonical_v1_l0_l2_design_audit_attempt001/report.md>)
- [Canonical probe rejection reconciliation](<../../../../audit/autoresearch/protocol101_canonical_v1_probe_rejection_reconciliation_attempt001/report.md>)
- [Clean-window synchronization readiness](<../../../../audit/autoresearch/protocol101_clean_window_hill_climb_readiness_2026_07_25/report.md>)
- [Stage-1 regimen adversarial audit](<../../../../audit/autoresearch/protocol101_stage1_training_regimen_adversarial_audit_attempt001/report.md>)
- [Simulator-v5 repair machinery producer packet](<../../../../audit/autoresearch/protocol101_stage1_regimen_repair_machinery_attempt001/report.md>)
- [Fresh H0-H3 entry-campaign independent audit](<../../../../audit/autoresearch/protocol101_full_trader_stage1_entry_campaign_independent_audit_selection_attempt001/report.md>)
- [Corrected D1 causal attribution](<../../../../audit/autoresearch/protocol101_d1_shuffled_profit_causal_attribution_inference_correction_attempt011/report.md>)
- [M0/M1 Stage-0 contract-selector result](<../../../../audit/autoresearch/protocol101_policy_neutral_contract_selector_stage0_feasibility_attempt002/report.md>)
- [Exact proposed-model architecture diagnosis](<../../../../audit/autoresearch/protocol101_exact_proposed_model_architecture.md>)

Important implementation assets that are preserved but not automatically
accepted for the new scientific question include:

- `v4/dataset/spxw_0dte_neural.py`
- `v4/model/protocol101_serial_simulator_v5.py`
- `v4/model/protocol101_canonical_stage1_contract.py`
- `v4/scripts/run_protocol101_full_trader_graph.py`

## 5. Product Contract

### 5.1 Instrument and account

- Instrument: SPXW 0DTE long options.
- Position size: one contract.
- Maximum open positions: one.
- Entry: buy to open.
- Exit: sell to close.
- No short premium.
- No scaling in or out.
- No averaging down.
- No overnight position.
- No martingale.

### 5.2 Learned action spaces

When flat:

```text
WAIT
BUY exact eligible call/put/strike
```

When one position is open:

```text
HOLD
EXIT
```

The flat model owns:

- timing;
- call versus put;
- strike;
- moneyness;
- contract choice; and
- abstention.

The open-position model owns:

- continuing to hold;
- exiting now; and
- the forecast information used by the frozen protective-floor composer.

### 5.3 Hard-coded safety only

Hard-coded machinery may enforce:

- SPXW and 0DTE eligibility;
- data freshness;
- complete-ladder requirements;
- affordability;
- one contract;
- one open position;
- ask-entry and bid-exit accounting;
- fees and stress;
- 5% of session-starting-equity daily stop;
- account-survival backstop;
- no new entries after `15:30 ET`; and
- forced flat by `15:55 ET`.

Hard-coded machinery may not permanently decide:

- when to enter;
- call versus put;
- nearest ATM;
- OTM versus ATM versus ITM;
- a fixed holding period;
- a fixed profit target; or
- the ordinary learned exit.

## 6. Full Ladder And Same-Game Contract

### 6.1 Action universe

At each completed decision minute, the nominal action universe is:

- `WAIT`; plus
- 21 strikes from 50 points below through 50 points above the canonical ATM
  strike in 5-point increments; times
- call and put rights.

That is `WAIT + 42 contract slots`.

An unavailable, stale, unaffordable, malformed, or otherwise ineligible
contract is masked. A mask removes an impossible action; it does not become an
alpha feature unless explicitly admitted.

### 6.2 Contract identity continuity

A 90-minute sequence must never blindly stack canonical slot indices.
Recentring the ladder can cause one offset slot to represent different strikes
at different times.

The historical and live tensor builders must therefore:

- key temporal contract paths by exact contract identity, strike, expiry, and
  right;
- separately encode each historical cross-sectional ladder snapshot;
- provide current-relative moneyness and offset at every minute;
- mask minutes before a contract entered the governed universe; and
- prove that historical and live builders resolve identities identically.

Failure to preserve identity invalidates all model evidence.

### 6.3 Decision timing

- The model runs once per completed minute.
- The same completed-minute convention must be used historically and live.
- Historical labels may use future paths.
- Runtime tensors may contain current and past information only.
- Entry and exit fills use the accepted causal ask/bid convention plus fees and
  stress.
- Latency, missing minutes, reconnects, and stale data must be reproducible
  abstention conditions.

### 6.4 Rolling history

The owner approved:

- a 90-minute rolling history;
- available-history masking near the open; and
- explicit clock fields.

Ninety minutes is an owner-chosen design horizon, not a claim that measurement
proved it optimal. It does not cap how long a trade may remain open.

### 6.5 Clock fields and market phases

Every model receives:

- `minutes_since_open`;
- `minutes_to_close`; and
- a categorical `market_phase`.

The approved phases are:

| New York time | Phase |
|---|---|
| 09:30-09:45 | Opening discovery |
| 09:45-11:15 | Primary morning session |
| 11:15-11:45 | Europe-close transition around 11:30 |
| 11:45-13:30 | Lunch |
| 13:30-15:00 | Afternoon |
| 15:00-15:30 | Power-hour entry period |
| 15:30-15:55 | Manage/exit only |

These are context features, not style bans. Only the existing no-new-entry
boundary after `15:30` and forced-flat boundary at `15:55` are hard rules.

## 7. Feature Strategy

### 7.1 Mandatory feature floor

Every serious full-ladder entry candidate must receive:

- canonical option premium and its 90-minute causal path;
- call or put identity;
- exact strike;
- current strike offset and moneyness;
- affordability and eligibility state;
- canonical SPX price action and synchronized context;
- internally recomputed IV, delta, and gamma;
- clock fields and market phase; and
- masks describing missing history and unavailable contracts.

Internally recomputed Greeks must come from canonical price, spot, strike, time
to expiry, and frozen assumptions. Raw vendor Greeks are prohibited.

### 7.2 Core and rich challenger

Two feature lanes are planned:

**Parity-core lane**

- Contains only features whose historical/live construction and transfer have
  passed the amended admission contract.
- Must still satisfy the mandatory feature floor.

**Ladder-rich challenger**

- May add canonical spread behavior, richer bid/ask structure, option-price
  path summaries, cross-strike surface structure, and additional internally
  recomputed Greeks.
- Every added field must pass field-level bias, source-discriminator,
  decision-transfer, and stress checks before training.

The rich lane may win only through frozen unseen evidence. It does not receive
special preference.

### 7.3 Guard, fill, and audit-only fields

Unless separately admitted, these remain unavailable as alpha:

- raw vendor bid and ask signatures;
- raw spread signatures;
- quote age;
- update counts;
- sizes or depth;
- volume;
- open interest;
- vendor IV or Greeks;
- sub-minute fields; and
- distance to guard boundaries.

They may still be used for eligibility, fills, stress, safety, and audit.

## 8. Flat-State Entry Model

### 8.1 Model responsibility

At every completed minute while flat, the model evaluates the market and all
eligible contracts jointly. It must choose:

```text
WAIT
or
BUY one exact contract
```

There is no upstream P5 timing rule, direction rule, or nearest-ATM selector.

### 8.2 Primary architecture

The primary candidate is a temporal full-ladder neural model:

1. A market-history encoder processes 90 minutes of synchronized SPX and
   context state.
2. A shared contract-history encoder processes each exact contract's causal
   option path.
3. A cross-ladder component compares calls, puts, strikes, relative prices,
   moneyness, Greeks, and surface relationships.
4. Separate prediction heads produce interpretable path forecasts for each
   eligible contract.
5. A frozen quality-first composer converts forecasts into `WAIT` or one
   contract.

The exact neural family is not frozen by this planning document. Temporal
attention, a set-aware ladder encoder, recurrent or temporal-convolution
components may compete inside the bounded model lab, provided the input,
output, labels, and gates remain unchanged.

### 8.3 Required controls

- HGB or MLP using the same authorized information in summarized form.
- Constant-output model.
- Matched-random full-ladder selector.
- Nearest-ATM selector.
- Legacy P5 baseline.
- Sign-reversed selector.
- Strong shuffled-target fits.
- Feature-family ablations.

Controls are comparisons, not fallback architectures that may silently replace
the intended product.

## 9. Entry Targets: Predict Path Properties Separately

The owner explicitly rejected a single eventual-profit label as the primary
definition of entry quality.

A trade that falls deeply red immediately, remains red for most of its life,
and finally touches a small profit is a poor entry even if it technically ends
green.

### 9.1 Forecast horizons

For each eligible contract, predict path properties at:

- 3 minutes;
- 5 minutes;
- 10 minutes;
- 20 minutes;
- 45 minutes;
- 90 minutes; and
- the remaining tradable session.

These are observation horizons, not forced exits. Near the close, unavailable
horizons are censored and masked rather than shortened or invented.

### 9.2 Economic primitives

Let:

- `A_t` be the causal executable entry ask;
- `B_u` be the causal executable future bid;
- `F` be the round-trip fee overlay;
- one SPXW contract multiplier be applied consistently.

Each path target must be available in:

- fee-adjusted dollars per contract; and
- fee-adjusted return on entry premium.

This dual representation prevents:

- cheap OTM contracts winning only because percentage returns are explosive;
  and
- expensive ATM/ITM contracts winning only because dollar PnL is larger.

### 9.3 Required forecast families

**Early drawdown**

- Worst executable fee-adjusted return during the first 3, 5, and 10 minutes.
- Quantiles or calibrated distribution, not only the mean.

**Time to first real profit**

- Time until the executable bid first clears entry cost plus fees.
- Censored when no such event occurs before the relevant horizon or close.

**Pre-profit adverse excursion**

- Maximum adverse excursion before the first fee-adjusted profitable minute.
- A path that must endure a deep loss before a small gain is penalized here.

**Underwater burden**

- Depth-times-duration integral while the trade is below fee-adjusted
  breakeven.
- Total underwater minutes.
- Longest continuous underwater interval.

**Profitable-window stability**

- Fraction of observed minutes with executable positive PnL.
- Longest continuous profitable interval.
- Number of distinct profitable windows.
- Sensitivity to one-minute timing jitter.

**Upside**

- Executable MFE or upper return distribution by horizon.
- Profit-area and tail opportunity.
- Dollar and percentage versions.

### 9.4 Prohibited target shortcuts

The entry model may not receive or optimize:

- the perfect hindsight exit minute;
- maximum future price as a complete label;
- the selected future exit from the eventual lifecycle model;
- future bid, ask, spread, Greeks, or context as features;
- oracle action labels as runtime features;
- protected-holdout outcomes; or
- IBKR shadow or paper outcomes.

Future paths create supervised answers only.

## 10. Entry Composer And WAIT

### 10.1 Quality first

The owner approved a two-stage quality-first decision:

1. Determine which contracts have credible entry-path quality.
2. Among those contracts, choose the best conservative upside.

If no contract qualifies, choose `WAIT`.

### 10.2 Training-calibrated guardrails

Numeric guardrails are not selected from outer validation, confirmation,
holdout, or live results.

The scientific design must preregister a deterministic calibration procedure
that:

- uses training-side distributions and fee-aware outcomes only;
- calibrates separately where justified by phase or contract regime;
- freezes all numeric thresholds before the corresponding outer fold;
- applies the same method in every fold;
- accounts for all threshold variants in campaign multiplicity; and
- emits exact calibration identities and hashes.

The intended constraints include:

- conservative early-drawdown forecast;
- bounded underwater burden;
- credible time to first profit;
- a stable profitable window;
- positive conservative dollar and percentage upside; and
- adequate model and source-transfer margin.

The exact numeric calibration algorithm remains an item for the FT-10
scientific contract and Fable's review. It is deliberately not being invented
inside a later training Goal.

### 10.3 Uncertainty-aware abstention

The model must choose `WAIT` when:

- no contract clears the quality guardrails;
- conservative upside does not clear zero after fees;
- the selected contract's edge over alternatives is inside calibrated model or
  source-transfer uncertainty; or
- required context or ladder state is missing.

Because confidence controls `WAIT` and contract choice, the signed G8
amendment requires a new action-conditioned calibration gate. G8 can no longer
remain report-only for this architecture.

## 11. Entry Evaluation

### 11.1 Primary entry gate

Whole-path entry evaluation is primary.

Fixed stops, fixed targets, and fixed holding times do not decide whether an
entry is good. If retained, they are report-only diagnostics for the entry
layer.

### 11.2 Frequency

- There is no hard maximum trades per day.
- The historical `0.3-6` trades/day range is descriptive, not gating.
- A few seven-trade sessions are not automatically a failure.
- The model may take zero trades when no opportunity qualifies.
- A model may not pass merely by taking almost no trades.

Evidence sufficiency replaces an arbitrary minimum count. If the confidence
interval is too wide because the model trades rarely, the result is
`insufficient_evidence`, not a pass and not an automatic no-signal rejection.

### 11.3 P5 benchmark

P5 is a legacy, hand-coded comparison:

- if SPX is at or above VWAP, choose a call;
- if SPX is below VWAP, choose a put;
- choose the nearest-ATM eligible contract on that side; and
- under the historical policy-5 shape, normally let the trade run toward
  forced flat unless premium is effectively lost or the very large target is
  reached.

P5 does not learn timing, direction, strike, or exits. It remains a mandatory
benchmark only.

The entry model must beat P5 on matched whole-path entry quality. `WAIT`
decisions must remain in the accounting rather than being removed to flatter
the learned model.

## 12. Open-State HOLD/EXIT Model

### 12.1 Training order

The owner approved:

1. Accept and freeze the entry model first.
2. Pretrain lifecycle behavior broadly on eligible causal contract paths.
3. Specialize and evaluate it on out-of-fold trajectories actually selected by
   the frozen entry model.
4. Keep entry and exit separately auditable.
5. Do not jointly fine-tune them in the first Full Trader campaign.

### 12.2 Candidate families

- Transparent HGB lifecycle baseline.
- Temporal position-aware neural challenger.

Both receive identical causal features, folds, entries, fees, stress, and
serial replay. Unseen evidence chooses the winner.

### 12.3 Runtime inputs

The open-state model may receive:

- complete causal market and ladder history available at the current minute;
- exact open contract identity and current contract state;
- entry ask and elapsed holding time;
- current executable bid, ask, mid, and admitted spread state;
- current fee-adjusted unrealized PnL in dollars and percent;
- MFE and MAE accumulated only through the current minute;
- giveback from MFE and time since MFE;
- recent option-price and PnL velocity;
- current moneyness and approved internal Greeks;
- current account and daily-stop state;
- current protective floor;
- current market phase and minutes to forced flat;
- the frozen entry model's no-action shadow scores; and
- a causal forecast of qualified replacement opportunities.

### 12.4 Shadow entry evaluation while occupied

While a position is open, the frozen entry model continues to score the market
without taking actions. This tells the exit model what the flat-state policy
would consider now.

The lifecycle model may also forecast the probability and quality of a
replacement opportunity appearing soon. It must do so from current and past
data only. Actual future opportunities are labels, never runtime inputs.

### 12.5 Multi-horizon hold advantage

At each completed minute, forecast:

- value of exiting now at the executable bid;
- continuation value over multiple future horizons;
- downside risk if holding;
- chance and magnitude of recovery;
- giveback risk;
- remaining tail opportunity;
- opportunity cost of keeping the only slot occupied; and
- uncertainty in those forecasts.

The model does not receive a one-minute oracle HOLD/EXIT label as a runtime
feature. The frozen composer uses the forecast distribution to choose.

## 13. Protective Floor

The owner approved a forecast-derived, upward-only protective floor.

The model still has only two learned lifecycle actions: `HOLD` and `EXIT`.
The floor is a transparent risk-control output of the frozen composer, not a
third hidden policy.

### 13.1 Causal order

At completed minute `t`:

1. Read the executable bid and state at `t`.
2. Compare the bid with the floor committed at `t-1`.
3. If the committed floor is crossed, exit.
4. Otherwise evaluate the new HOLD/EXIT forecasts.
5. If HOLD is chosen, compute the next floor from current causal forecasts.
6. Commit that floor for `t+1`.

The floor may stay unchanged or rise. It may never fall.

### 13.2 What informs the floor

- current net PnL;
- MFE and giveback;
- calibrated downside;
- continuation upside;
- forecast uncertainty;
- option velocity;
- admitted spread state;
- elapsed time;
- market phase;
- time to forced flat; and
- replacement-opportunity pressure.

### 13.3 Same-game limitation

The synchronized historical paths are minute-level. Therefore the initial
validated floor is checked at completed-minute executable bids.

The plan does not claim intraminute protection. Native IBKR stop orders or
sub-minute triggers would create a historical/live mismatch unless separately
supported by compatible historical data and an independently accepted fill
contract.

## 14. Exit Evaluation

The learned exit is evaluated on exactly the same frozen entries as:

- exit immediately;
- hold to forced flat;
- a finite preregistered set of transparent time/stop/target comparators;
- the legacy P5 lifecycle where relevant; and
- matched-rate random exits.

The learned exit must beat the best honest comparator:

- pooled;
- in at least four of five chronological outer folds;
- after identical fees and stress;
- under one-account serial replay; and
- without violating the charter's survival controls.

The comparators are not the entry-quality target and do not limit learned trade
duration.

Required lifecycle diagnostics include:

- fee-adjusted PnL;
- loss truncation;
- tail-win capture;
- four-bucket charter outcome distribution;
- harvest ratio;
- MFE giveback;
- MAE;
- underwater duration;
- churn;
- time in trade;
- skipped opportunities while occupied;
- call/put, moneyness, phase, premium, and regime exposure; and
- protective-floor activation and boundary behavior.

## 15. Four-Box Attribution

The four-box method is applied after entry and exit are separately accepted and
again whenever their interaction materially changes.

| Box | Entry | Exit | Main question |
|---|---|---|---|
| A | P5 and other fixed controls | Best honest fixed/control exit | What can the non-learned baseline do? |
| B | Learned full-ladder entry | Same fixed/control exit | Did learned entry improve opportunity quality? |
| C | P5/control entry | Learned exit | Does learned lifecycle add value independent of learned entry? |
| D | Learned full-ladder entry | Learned exit | Does the intended complete trader work? |

Interpretation:

- If Box B fails, the entry layer is not accepted.
- If Box B passes but D fails while C also fails, the exit layer is the leading
  problem.
- If B and C pass separately but D fails, route to interaction attribution.
- If D passes only because P5 supplies timing or contract choice, product drift
  has occurred.

The full trader must earn more fee-adjusted profit than the complete P5
benchmark while also satisfying bounded-risk requirements. More profit
obtained through unacceptable drawdown, ruin risk, or instability does not
count as a win.

## 16. Data Roles And Anti-Overfitting Design

### 16.1 Nested chronological cross-fitting

The 15-month corpus is shared without allowing either layer to train on its
test trades.

For each outer fold:

- the entry model trains only on earlier sessions;
- entry tuning uses nested chronological development roles;
- frozen entry predictions create out-of-fold trajectories;
- lifecycle training uses only earlier out-of-fold trajectories;
- lifecycle tuning uses nested chronological development roles; and
- combined entry-plus-exit evaluation occurs on a later period unseen by both
  fitted components.

No random row split is permitted.

### 16.2 Inner autoresearch arena

The autoresearch agent may inspect only inner development evidence while
iterating.

Allowed changes:

- neural architecture;
- width, depth, and embeddings;
- temporal or cross-ladder representation;
- loss weighting within the frozen target family;
- optimizer and schedule;
- regularization;
- calibration implementation under the frozen rule; and
- training efficiency.

Forbidden changes after the campaign begins:

- action space;
- product definition;
- authorized features;
- path targets;
- forecast horizons;
- composer semantics;
- fees or fills;
- folds or roles;
- primary gates;
- P5/control definitions;
- simulator semantics; and
- protected data.

### 16.3 Plateau

There is no wall-clock limit chosen in advance.

The loop stops after 12 consecutive serious valid trials fail to improve the
frozen inner objective beyond measured resampling noise.

A serious trial must be:

- materially different;
- mechanically valid;
- fully reproducible;
- evaluated on the same arena; and
- attributable to a declared architecture, loss, or regularization change.

Duplicate configurations, failed jobs, cosmetic changes, and seed-only
reruns do not count toward the 12.

### 16.4 Outer shortlist

- Freeze no more than three meaningfully different learned candidates.
- Evaluate the shortlist once on outer folds.
- Apply multiplicity adjustment across every candidate and registered variant.
- Do not tune against outer-fold failures.
- Outer failure terminates that campaign version.

Baselines do not consume learned-shortlist slots.

## 17. Evidence Standard

The owner approved balanced evidence:

- strict integrity controls;
- bounded downside in every fold;
- adjusted 95% improvement interval above zero;
- positive evidence in at least four of five chronological outer folds;
- a bounded and explained fifth fold;
- no dependency on one side, phase, month, or regime;
- realistic fees and stress;
- no single-seed story; and
- no single-metric promotion.

No fixed trades-per-day gate is used. Rare-trade candidates must produce enough
independent evidence for their interval to clear the required comparison.

### 17.1 Charter gates retained

- Fee-adjusted net PnL is primary economics.
- Win rate is diagnostic, never optimized directly.
- G4 Calmar floor: pooled net PnL / pooled maximum drawdown `>= 1.0`.
- Per-fold equity never below `$5,000` from `$10,000` starting cash.
- Daily realized-loss stop: 5% of session-starting equity.
- One contract and one open position.
- SPY remains the live first-year opportunity-cost benchmark.
- Tail capture and survival both matter.

### 17.2 Confidence gate

The old report-only G8 does not govern this architecture because confidence
now changes actions.

Before training, Graph V2 must define and owner-sign an action-conditioned
calibration amendment covering:

- entry `WAIT`;
- contract selection;
- quality guardrails;
- exit `HOLD/EXIT`;
- protective-floor behavior; and
- low-sample fallback behavior.

Calibration is evaluated on the behavior confidence actually controls, not on
an unrelated post-selection win probability.

## 18. Reward-Hacking And Integrity Controls

Every promotable campaign must include:

- strong group-aware shuffled targets;
- matched-random full-ladder actions;
- constant-output selectors;
- reversed selectors;
- source-discriminator checks;
- future-field and alias firewall tests;
- target recomputation from raw paths;
- decision, contract, slot, session, fold, and path identity uniqueness;
- exact contract-risk-set membership;
- chronological role separation;
- fees and ask/bid accounting;
- timing jitter;
- feature-noise injection based on measured transfer drift;
- spread and fill stress;
- stale and missing-minute stress;
- complete one-account serial replay;
- model and artifact hash reproduction;
- independent producer/auditor separation; and
- campaign-wide multiplicity control.

Positive absolute PnL is not evidence of learning. The model must add credible
incremental value beyond exposure-matched controls.

## 19. Hardware And Execution Environment

### 19.1 Training

- Training may run locally or on leased Akash GPU resources.
- Paid compute requires owner authorization when a real cost is known.
- The graph records model size, training time, peak memory, storage, and cost.
- No local model training runs concurrently with live IB Gateway collection or
  shadow execution.

### 19.2 Live inference

The final frozen trader must run on the owner's Apple M5 MacBook Pro with 16 GB
memory while IB Gateway and required monitoring are active.

The owner rejected arbitrary early limits on parameter count, memory, or
latency. The graph therefore measures:

- artifact size;
- resident memory;
- inference latency;
- feature-build latency;
- sustained CPU/GPU/Neural Engine load where applicable;
- thermal behavior;
- Gateway and recorder interference; and
- missed decision deadlines.

If the selected model cannot run reliably on the exact laptop, the graph stops
for an owner decision. It may propose quantization, distillation, a smaller
challenger, remote inference, or different hardware. It may not silently weaken
the model and call the problem solved.

## 20. Graph V2 Is Coordination, Not An Executable Trader

Graph V2 should be represented as a machine-readable state contract plus a
human-readable dashboard, but it is not a monolithic executable that builds a
profitable model on demand.

Each node is executed by an authorized agent, deterministic script, auditor, or
owner. The controller:

- checks receipts;
- validates legal transitions;
- resumes interrupted work;
- dispatches the next written node;
- records state; and
- stops at required boundaries.

The controller cannot:

- invent a new scientific question;
- train without a written node contract;
- approve its own output;
- access protected evidence early;
- guarantee a profitable result; or
- authorize broker orders.

## 21. Proposed Graph V2 Topology

Graph V1 is preserved unchanged as reset history. Graph V2 becomes canonical
only after Fable review, revision, owner approval, implementation, and
independent controller validation.

### Phase A: Full scientific contract

1. `FT2-10-FULL-TRADER-DESIGN`
   - Freeze entry, lifecycle, feature, label, composer, evidence, and failure
     contracts.
2. `FT2-20-PARALLEL-DESIGN-REVIEW`
   - Trading-realism, ML/statistics, and live-parity reviewers analyze the same
     frozen design independently.
3. `FT2-21-OWNER-DESIGN-APPROVAL`
   - Owner accepts or rejects the complete plan.

### Phase B: Feature admission and machinery

4. `FT2-25-FULL-LADDER-FEATURE-ADMISSION`
   - Resolve the signed synchronization conflict for canonical contract-price
     history and any rich challenger fields.
5. `FT2-26-INDEPENDENT-FEATURE-ADMISSION-AUDIT`
   - Reproduce transfer, bias, and source-leakage evidence.
6. `FT2-30-ENTRY-HARNESS-PILOT`
   - Validate tensors, identities, labels, masks, composer plumbing, memory,
     and runtime on a small sample.
7. `FT2-31-INDEPENDENT-ENTRY-MACHINERY-ACCEPTANCE`
   - Independently accept machinery.

A small pilot cannot issue a scientific `no_signal` verdict. It may stop only
for a mechanical defect, scientific-contract defect, or resource blocker.

### Phase C: Entry autoresearch

8. `FT2-40-ENTRY-INNER-AUTORESEARCH`
   - Run the bounded neural entry loop until a champion frontier or 12-trial
     plateau.
9. `FT2-41-ENTRY-SHORTLIST-FREEZE`
   - Freeze at most three candidates before outer access.
10. `FT2-50-ENTRY-OUTER-EVALUATION`
    - One-shot whole-path evaluation against P5 and controls.
11. `FT2-51-INDEPENDENT-ENTRY-AUDIT`
    - Reproduce models, predictions, controls, multiplicity, and gates.
12. `FT2-52-OWNER-ENTRY-FREEZE`
    - Required owner pause, as explicitly selected in planning.

If no entry candidate passes, the graph stops with a precise no-signal,
insufficient-evidence, invalid-evidence, or contract-defect result. It does not
begin lifecycle training.

### Phase D: Lifecycle design and autoresearch

13. `FT2-60-LIFECYCLE-DESIGN-FOR-FROZEN-ENTRY`
    - Instantiate hold-advantage, opportunity-cost, and protective-floor
      contracts for the selected entry.
14. `FT2-61-PARALLEL-LIFECYCLE-REVIEW`
    - Trading, ML, and live reviewers analyze it.
15. `FT2-62-OWNER-LIFECYCLE-APPROVAL`
    - Required second owner pause, as explicitly selected.
16. `FT2-69-LIFECYCLE-HARNESS-PILOT`
    - Validate broad pretraining, OOF specialization, labels, floor timing, and
      serial accounting.
17. `FT2-70-INDEPENDENT-LIFECYCLE-MACHINERY-ACCEPTANCE`
    - Independently accept machinery.
18. `FT2-71-LIFECYCLE-INNER-AUTORESEARCH`
    - Run neural and HGB loops under the same frozen target.
19. `FT2-72-LIFECYCLE-SHORTLIST-FREEZE`
    - Freeze at most three candidates.
20. `FT2-73-LIFECYCLE-OUTER-EVALUATION`
    - Compare identical entries across learned and control exits.
21. `FT2-74-INDEPENDENT-LIFECYCLE-AUDIT`
    - Reproduce exit-specific gates.

### Phase E: Integration and attribution

22. `FT2-80-FOUR-BOX-COMBINED-AUDIT`
    - Run Boxes A-D under strict serial economics.
23. `FT2-81-FAILURE-ATTRIBUTION`
    - Route failures to entry, lifecycle, interaction, machinery, or
      insufficient evidence without changing the contract.
24. `FT2-82-COMPLETE-SYSTEM-FREEZE`
    - Freeze all models, features, calibration, thresholds, transforms,
      simulator, hashes, and runtime interface.

### Phase F: Protected and live evidence

25. `FT2-90-FRESH-CONFIRMATION`
    - Owner-approved one-shot confirmation.
26. `FT2-91-PROTECTED-HOLDOUT`
    - Owner-approved one-shot protected holdout.
27. `FT2-92-IBKR-DECISION-SHADOW`
    - Candidate-specific historical/IBKR transfer.
28. `FT2-93-NO-ORDER-LIVE-SHADOW`
    - Complete trader runs live without orders.
29. `FT2-94-INDEPENDENT-LIVE-SHADOW-AUDIT`
    - Reconstruct every state and action.
30. `FT2-95-OWNER-PAPER-AUTHORIZATION`
    - Owner decides whether guarded paper orders may begin.
31. `FT2-96-GUARDED-IBKR-PAPER-VALIDATION`
    - Collect actual paper evidence.
32. `FT2-97-INDEPENDENT-PAPER-EVIDENCE-REVIEW`
    - Review fills, behavior, safety, and divergence.

Paper completion is not real-money authorization.

## 22. Loop And Failure Policy

### 22.1 Automatic repairs

The controller may automatically repair:

- serialization;
- checkpointing;
- identity plumbing;
- reproducibility;
- tensor shape or masking;
- deterministic replay mismatches;
- stale progress state; and
- other independently identified mechanical defects.

Each mechanical node has at most two repair attempts and must return to
independent acceptance.

### 22.2 Forbidden automatic repairs

The controller may not automatically change:

- the trader's action space;
- the full-ladder requirement;
- mandatory features;
- entry targets;
- forecast horizons;
- exit target;
- primary gates;
- P5 comparison;
- owner risk rules;
- outer data roles;
- protected-evidence rules; or
- paper boundaries.

These are scientific or product changes and require owner review.

### 22.3 Terminal meanings

- `mechanical_defect`: evidence is invalid until repaired and reaccepted.
- `no_genuine_entry_signal`: the entry hypothesis failed under the frozen
  contract.
- `no_genuine_exit_signal`: lifecycle failed on accepted identical entries.
- `interaction_failure`: components passed separately but not together.
- `insufficient_evidence`: sample uncertainty prevents a valid conclusion.
- `scientific_contract_defect`: the planned test cannot answer its own
  question.
- `product_drift_detected`: a node changed the intended trader.
- `resource_owner_decision_required`: hardware, storage, time, or paid compute
  requires owner choice.

None of these may be narrated as success.

## 23. Progress And Owner Visibility

The owner approved one consolidated live dashboard.

It must display:

- current graph and node;
- plain-English node purpose;
- current executor and independent reviewer;
- completed gates;
- current entry and exit model status;
- leading candidate behavior, not only IDs;
- current evidence and uncertainty;
- P5 and control comparisons;
- active blocker;
- compute time, storage, and estimated cost;
- retry and plateau counts;
- protected resources spent or still closed;
- next automatic action;
- next owner decision; and
- whether any broker or paper path has been touched.

Detailed packets remain in `v4/audit/autoresearch`, but the owner must not need
to reconstruct program state from them.

## 24. Freeze Requirements

An accepted component bundle includes:

- model weights;
- architecture definition;
- feature names and ordering;
- input tensor schema;
- 90-minute history semantics;
- contract-identity rules;
- market-phase definitions;
- target definitions;
- calibration transforms;
- composer thresholds;
- action masks;
- fees and fill assumptions;
- simulator version and configuration;
- fold and role manifests;
- source and dataset hashes;
- training code commit;
- model and artifact hashes;
- latency and memory measurements; and
- independent acceptance receipt.

Any dependency change creates a new candidate identity.

## 25. Paper-Readiness Meaning

Offline success earns only:

> Frozen Full Trader candidate eligible for paper-readiness validation.

Paper readiness additionally requires:

- candidate-specific historical/IBKR tensor and action transfer;
- no-order live operation on the M5 with Gateway;
- reconstructable `WAIT`, entry, HOLD, floor, and EXIT decisions;
- stale-data, reconnect, incomplete-ladder, and forced-flat behavior;
- guard and rollback readiness;
- independent live-shadow acceptance; and
- owner authorization.

The graph must never describe an entry-only model as paper-ready.

## 26. Explicit Drift Tripwires

Stop immediately if any node:

- replaces full-ladder choice with ATM-only selection;
- makes P5 permanent;
- hard-codes call/put direction;
- removes `WAIT`;
- judges entry only by eventual profit;
- uses perfect hindsight exit as entry truth;
- starts lifecycle training before entry acceptance;
- trains lifecycle only on P5 and calls it the final trader;
- lets low trade count pass without adequate evidence;
- changes gates after outer results;
- uses outer folds as iterative training feedback;
- exposes future paths to runtime features;
- uses vendor-identifying fields without admission;
- ignores same-game differences;
- silently simplifies for the Mac;
- lets the producer independently approve itself; or
- accesses holdout, shadow, paper, or broker paths early.

## 27. Decisions Settled In The Planning Session

| ID | Settled decision |
|---|---|
| D01 | Product is learned `WAIT or exact contract`, then learned `HOLD or EXIT`. |
| D02 | Full governed ladder is 42 nominal contracts, not three near-ATM contracts. |
| D03 | Entry timing, direction, and strike are learned. |
| D04 | Use 90-minute rolling causal history plus masks and clock fields. |
| D05 | Use the seven explicit Pickles-style market phases. |
| D06 | Forecast several entry path properties separately. |
| D07 | Forecast at 3/5/10/20/45/90 minutes and through close. |
| D08 | Use whole-path entry quality as the primary entry gate. |
| D09 | Early drawdown, pre-profit MAE, underwater burden, and profit-window stability are primary evidence. |
| D10 | Use quality-first selection, then conservative upside. |
| D11 | Calibrate quality guardrails on training roles and freeze before outer validation. |
| D12 | Use uncertainty-aware `WAIT`. |
| D13 | No hard trades/day maximum; frequency is judged economically. |
| D14 | Rare-trade candidates require evidence sufficiency, not a fixed minimum. |
| D15 | Use a joint temporal full-ladder neural primary with simpler controls. |
| D16 | Require canonical price history, geometry, moneyness, and internal Greeks. |
| D17 | Predict both dollars per contract and percentage economics. |
| D18 | Use core and ladder-rich feature challengers, with transfer admission. |
| D19 | P5 is a mandatory benchmark, never final architecture. |
| D20 | Final trader must earn more profit than P5 with bounded risk. |
| D21 | Broad causal lifecycle pretraining precedes specialization to frozen OOF entries. |
| D22 | Exit predicts multi-horizon hold advantage, downside, giveback, and slot opportunity cost. |
| D23 | Entry model runs in no-action shadow while a position is open. |
| D24 | Use a forecast-derived completed-minute protective floor that only ratchets upward. |
| D25 | Re-entry may occur on the next completed minute; no arbitrary cooldown. |
| D26 | Neural and HGB lifecycle candidates compete on identical evidence. |
| D27 | Keep entry and exit separate in the first campaign; no joint fine-tuning. |
| D28 | Use nested chronological cross-fitting. |
| D29 | Stop after 12 serious non-improving inner trials; no wall-clock limit. |
| D30 | Freeze at most three candidates for one-shot outer evaluation. |
| D31 | Use balanced evidence with strict integrity and bounded fold downside. |
| D32 | Run four-box attribution before complete-system freeze. |
| D33 | Automatically repair only mechanical defects. |
| D34 | Preserve owner pauses after entry selection and before lifecycle training. |
| D35 | Maintain one consolidated live dashboard. |
| D36 | Do not impose early Mac model-size limits; measure and escalate real limits. |
| D37 | Preserve protected holdout and separate paper authorization. |
| D38 | Create Graph V2 and preserve Graph V1 unchanged. |

## 28. Items Deliberately Not Yet Frozen

These are not permission for downstream improvisation. They must be resolved in
the reviewed FT2-10 design before training:

1. Exact canonical transformation and admission thresholds for per-contract
   price history.
2. Exact parity-core and ladder-rich feature lists.
3. Exact neural tensor schema and masking representation.
4. Exact multi-head losses and distribution-calibration method.
5. Exact deterministic training-side guardrail-calibration algorithm.
6. Exact scalar or lexicographic inner-loop objective.
7. Exact action-conditioned confidence gate.
8. Exact set of transparent exit comparators.
9. Exact forecast-to-protective-floor equation.
10. Exact treatment of split-confidence and floor-boundary cases.
11. Exact block-bootstrap, dependence, multiplicity, and effective-sample
    implementation.
12. Exact no-order live-shadow evidence-sufficiency rule.
13. Exact protected confirmation and holdout identities.
14. Exact Akash provisioning, spending, and checkpoint policy if GPU training
    is required.

Every item must be decided before its relevant results exist.

## 29. Requested Fable Review

Fable is asked to analyze this plan deeply rather than merely approve its
direction.

Please answer:

1. Does the entry target measure timing quality without smuggling in a perfect
   exit oracle?
2. Are the path-property heads sufficient, redundant, or missing a critical
   outcome?
3. Can training-calibrated guardrails be defined without turning threshold
   search into another overfit model?
4. Is the mandatory option-price feature floor compatible with the existing
   synchronization evidence, and what exact bounded admission test is needed?
5. Is a joint 42-contract temporal model the correct inductive bias?
6. How should exact contract continuity and recentered ladder context be
   represented?
7. Is nested cross-fitting sufficient to train exits from learned-entry
   trajectories without contamination?
8. Is the replacement-opportunity target causal and decision-relevant?
9. Can the completed-minute protective floor be trained and evaluated without
   hindsight or hidden fixed-exit behavior?
10. Are the P5 and matched controls exposure-fair?
11. Does the balanced evidence standard reject noise without suppressing real
    0DTE alpha?
12. Are the 12-trial plateau and three-candidate outer shortlist defensible?
13. Are the graph node boundaries large enough to be useful but small enough
    to attribute failures?
14. Which nodes can safely run automatically, and where is owner judgment
    indispensable?
15. What reward-hacking or source-transfer path remains unaddressed?
16. What would most likely make the plan computationally infeasible on the
    available corpus or operationally infeasible on the M5?
17. What minimum changes are required before this can become an owner-signable
    scientific contract?

Fable should return:

- a verdict on the product architecture;
- a verdict on entry target validity;
- a verdict on lifecycle target validity;
- a verdict on evidence and multiplicity;
- a verdict on same-game/live transfer;
- concrete corrections;
- any fatal blocker;
- the smallest defensible FT2-10 design scope; and
- a recommended first non-training Goal.

## 30. Highest Allowed Claim

After this document is written and reviewed, the highest allowed claim is:

> Protocol101 Full Trader Graph V2 has a detailed planning and external-review
> brief.

No model design is yet owner-signed. No Graph V2 controller is yet canonical.
No new full-ladder model has been trained. No entry or lifecycle model has been
accepted. No protected evidence or broker path is authorized by this document.
