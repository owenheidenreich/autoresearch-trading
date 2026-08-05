# ML Engineer Handoff: Protocol101 Versus Protocol265

Last updated: 2026-05-23

## Purpose Of This Document

This document is written for an external machine learning / AI engineer who needs
to understand the current v4 SPXW 0DTE trading-bot research stack. It explains
the project goal, the data assumptions, the replay environment, the current
paper-trading default model, and the strongest recent research challenger.

The two competing model stacks covered here are:

- `PAPER_DEFAULT_PROTOCOL101`: the current operational paper-trading default.
- `CHALLENGER_SOURCE_PENALTY_BASELINE_ANCHORED_CONTINUATION_V1`: historically
  `Protocol265`, the strongest recent lifecycle challenger that beat
  Protocol101 on the common lifecycle-harness splits but is still research-only.

The important frame: v4 is not only an audit system. The intended final product
is a neural SPXW 0DTE long-options trading bot that can be tested in live paper
conditions and, eventually, real-money conditions only after strong evidence.
The research/audit system exists to prevent self-deception.

## Current Project Goal

The goal is to train a neural trader for SPXW 0DTE options that can make
realistic long-only decisions:

- do nothing,
- buy one SPXW call,
- buy one SPXW put,
- hold an existing position,
- exit an existing position.

The model must learn an executable edge, not a fake midpoint backtest edge. The
training and replay environment should match the future live/paper trading
environment as closely as possible:

- SPXW PM-settled contracts only,
- 0DTE contracts,
- `$5` strike increments,
- long calls and long puts only,
- executable ask entry,
- executable bid exit,
- one paper account,
- one open position max for the current phase,
- one contract max for the current operational phase,
- affordability enforced from a `$10,000` paper account,
- no simultaneous overlapping positions in source-of-truth equity,
- no lookahead features,
- mandatory flat-before-close,
- no paid data expansion without explicit approval.

Older diagnostic artifacts may contain overlapping independent-candidate PnL.
Those remain useful for research, but they are not promotion evidence. Promotion
must use strict one-account serial replay.

## Data And Market Reconstruction

The v4 stack moved away from v2/v3-style flawed data because the earlier systems
had no reliable bid/ask spreads and could not prove executable fills.

The intended source-of-truth historical market reconstruction is:

- Databento `OPRA.PILLAR` definitions for SPXW contract identity.
- Databento `OPRA.PILLAR` `cbbo-1m` for consolidated option bid/ask.
- Databento `OPRA.PILLAR` `ohlcv-1m` for option trade bars where available.
- Databento `OPRA.PILLAR` statistics for open interest where available.
- ThetaData SPX and VIX 1-minute index bars for official index context.
- Existing targeted higher-resolution slices for selected timing audits.

Important rules:

- `cbbo-1m` is quote data, not full trade-flow volume.
- The model must not treat forward-filled last-sale size as volume.
- Greeks are computed/repaired internally when possible.
- Missing or invalid Greeks should be repaired before candidates are skipped.
- Historical 1-minute replay is still not identical to live fills; live/no-order
  shadow and paper logs are needed to validate timing and routing.

## Evaluation Philosophy

The current promotion standard is skeptical:

- A model must beat `PAPER_DEFAULT_PROTOCOL101` under strict serial replay.
- It must use ask-entry / bid-exit accounting.
- It must have no overlaps, no unaffordable trades, and no non-flat sessions.
- It must survive slippage stress, especially `$0.10` and `$0.25` per side.
- It must not win by a single day/trade/month concentration.
- It must not tune directly on protected holdouts.
- It must pass runtime/no-order parity before replacing the paper default.

The system currently distinguishes:

- paper default: operationally allowed paper-trading candidate,
- research challenger: historically promising model not yet operational,
- diagnostic/audit: explains model behavior but is not a model,
- runtime/parity harness: proves historical/live feature and candidate parity,
- promotion decision: explicit status change.

## Model A: PAPER_DEFAULT_PROTOCOL101

### Plain-English Summary

Protocol101 is the current paper-trading default. It is a neural event policy
with causal short-history features. It decides whether a candidate trade is worth
taking while the account is flat. Once it enters, the exit path comes from the
older frozen lifecycle/exit stack associated with the candidate. It is more
conservative than later research challengers: fewer trades, higher win rate, and
stronger operational/live-paper plumbing.

Trader interpretation:

Protocol101 looks for relatively clean, high-confidence 0DTE opportunities,
usually near the model's preferred time windows, then exits using a frozen
lifecycle path. It is not trying to hold every large directional move. It was
selected because it survived the stricter one-account serial gate and has the
most live/paper infrastructure around it.

### Primary Files And Artifacts

Code:

- `v4/scripts/run_protocol101_event_history_policy.py`
- `v4/model/serial_opportunity.py`
- `v4/scripts/run_protocol092_serial_opportunity_policy.py`
- `v4/scripts/run_protocol097_sequential_event_policy.py`
- `v4/live/protocol101_live_entry.py`
- `v4/scripts/run_protocol158_protocol101_live_entry_paper_bridge.py`
- `v4/scripts/run_protocol160_protocol101_persistent_paper_trader.py`

Freeze and promotion artifacts:

- `v4/promotion/PROTOCOL_101_FREEZE.json`
- `v4/promotion/PROTOCOL_101_PROMOTION_READINESS_PACKET.md`
- `v4/audit/autoresearch/v4_aplus_hypothesis_101_event_history_policy/summary.json`
- `v4/audit/autoresearch/v4_aplus_hypothesis_101_event_history_policy/serial_policy_trades.json`
- `v4/audit/autoresearch/v4_aplus_hypothesis_101_event_history_policy/model_artifacts/`

Status nuance: some older Protocol101 promotion documents use the phrase
"not paper/live approved" because they were written before the later IBKR paper
operations work. In the current project vocabulary, Protocol101 is the paper
default for guarded paper testing only. It is not real-money approved.

Live operations:

- `v4/docs/PROTOCOL101_DAILY_PAPER_TRADING.md`
- `v4/ops/ibkr/run_protocol101_paper_session.sh`
- `v4/audit/autoresearch/v4_aplus_hypothesis_157_protocol101_daily_ops_monitor/`

### Upstream Protocols Supporting Protocol101

Protocol101 did not appear in isolation. It is built on a sequence of earlier
research steps:

- `Protocol077`: lifecycle sequence dataset around earlier Protocol054 exits.
- `Protocol081`: frozen candidate exit behavior used by later entry policies.
- `Protocol092`: serial opportunity-cost entry policy. It created a candidate
  table and trained an entry scorer while enforcing one open position max.
- `Protocol097`: sequential event policy. It shifted from isolated candidate
  scoring to event-level decisions.
- `Protocol101`: added causal short-history features to Protocol097 after Q4
  seed-level failures suggested the model was too memoryless.
- `Protocol102`: readiness/freeze diagnostics for Protocol101.
- `Protocol114`: skeptical falsification audit.
- `Protocol157/158/160`: daily monitor, live entry bridge, and persistent paper
  trader infrastructure.
- `Protocol161/162`: May 2026 live/historical parity diagnostic slices.

### Candidate Universe

Protocol101's training event summary reports:

- `23,530` decision events,
- max candidates per event: `10`,
- mean candidates per event: about `2.22`,
- splits: Q1 2025, Q2 2025, Q3 2025, Q4 2025, Q1 2026.

This is not the entire raw SPXW option chain. It is a candidate stream derived
from prior frozen selected opportunities and lifecycle exits. That makes the
model more operationally constrained and easier to validate, but it also means
it is not yet a fully general "see every valid contract and decide" neural bot.

### Feature Set

Protocol101 uses Protocol092 entry-only candidate features plus causal
short-history event summaries.

Entry features include:

- time since open,
- time to forced flat,
- day progress and sinusoidal time encodings,
- time bucket flags,
- call/put side flags,
- strike offset and absolute offset,
- surface edge,
- bid, ask, mid,
- spread and spread fractions,
- bid/ask sizes,
- underlying price,
- IV,
- delta, gamma, theta,
- gamma/theta ratio,
- theta over mid,
- theta burden,
- gamma per premium,
- premium over underlying,
- spread over mid,
- size imbalance,
- signed call/put delta fields.

Causal short-history features include:

- number of prior events seen in the session,
- minutes since previous event,
- previous event candidate count,
- previous event max/mean edge,
- previous event max gamma,
- previous event mean theta burden,
- previous event minimum spread over mid,
- previous event call/put counts,
- previous call-minus-put edge,
- rolling-3 candidate count mean,
- rolling-3 max/mean edge,
- rolling-3 max gamma,
- rolling-3 mean theta burden,
- rolling-3 minimum spread over mid,
- rolling-3 call-minus-put edge.

The important design choice is causal history only. Features are created from
events at or before the current decision time. No future path, exit, MFE/MAE, or
realized PnL columns are allowed into the model input.

### Model Architecture

Protocol101 uses the same small MLP family as Protocol092/097:

- input: entry features plus history features,
- hidden dimension: `96`,
- GELU nonlinearities,
- layer normalization,
- dropout,
- scalar score output.

The model is trained over chronological folds with seeds `[1,2,3,4,5]`.
Thresholds are selected only on validation splits. Test and March are reported
after threshold selection.

Protocol101 persists deployable artifacts:

- `model.pt`,
- `scaler.json`,
- `manifest.json`,
- feature columns,
- freeze manifest with hashes.

This persistence is one major reason Protocol101 is the operational paper
default rather than merely a research screen.

### Training Objective And Labels

Protocol101 inherits the serial opportunity framing:

- each event contains candidate contracts,
- the model scores candidates,
- validation selects a margin/threshold,
- simulation enters only when a candidate clears threshold,
- while in a position, later entries are skipped until exit,
- PnL uses the frozen candidate exit path.

The model is not a full hold/exit policy. It is mainly an entry/event-selection
model over a candidate stream whose exits are frozen from previous lifecycle
protocols.

### Historical Performance

From the Protocol101 freeze/readiness artifacts:

| Split | Median PnL | Profit Factor | Median Trades | Positive Seeds | `$0.25`/side Stress |
|---|---:|---:|---:|---:|---:|
| Q3 2025 | `$59,130` | `3.779` | `255` | `1.00` | `$46,380` |
| Q4 2025 | `$92,460` | `5.899` | `270` | `1.00` | `$77,760` |
| Q1 2026 | `$95,010` | `3.714` | `249` | `1.00` | `$82,560` |
| March 2026 | `$41,790` | `3.420` | `110` | `1.00` | `$36,290` |
| Recent 2026 | `$7,450` | `1.278` | `90` | `1.00` | reported in later comparisons |

The Recent 2026 number comes from later common-split comparisons, not from the
original Protocol101 freeze packet.

### Strengths

- Current operational paper default.
- Best live/paper infrastructure.
- Persisted model/scaler manifests.
- Higher win-rate style than broad challengers.
- Lower trade count, cleaner behavior.
- Strong historical PF in the original promotion gates.
- Tested under strict one-account serial replay.
- Uses causal event-history state.

### Weaknesses

- Candidate universe is not the full valid SPXW action surface.
- Entry model depends on prior candidate-generation/frozen exit artifacts.
- Exits are not a unified learned hold/exit policy.
- It can be too conservative and miss broader directional moves.
- It may behave scalpy in some replay cases.
- Recent 2026 evidence is weaker than newer research challengers.
- Live paper evidence is still immature; logs exist, but this is not live-real
  confidence yet.

### Live/Paper Trading Stack

Protocol101 currently has the operating stack:

- IB Gateway paper mode scheduled around market open.
- Live entry bridge in guarded `paper-submit` mode.
- Paper-order guards:
  - paper account only,
  - environment flag required,
  - fresh SPX/VIX context,
  - fresh SPXW NBBO,
  - PM-settled SPXW only,
  - max quantity 1,
  - max concurrency 1,
  - affordability enforced,
  - buy limit at current ask,
  - entry cancel timeout around 15 seconds,
  - exits from frozen lifecycle decision,
  - no real-money endpoint.
- Daily monitor for startup, blocks, decisions, orders, fills, PnL, and
  contracts.
- Append-only JSONL/CSV logs as source of truth.

The live/paper system is currently designed around Protocol101, not Protocol265.

## Model B: CHALLENGER_SOURCE_PENALTY_BASELINE_ANCHORED_CONTINUATION_V1

### Plain-English Summary

Protocol265 is the strongest recent lifecycle challenger. It is not the paper
default. It was created after multiple lifecycle experiments showed a specific
failure mode: models that were free to exit early could improve some blocks but
damage recent 2026 by leaving or blocking valuable Protocol261 base-stream
opportunities.

Protocol265 keeps the Protocol261 entry stream and original baseline exit as a
safety anchor. The neural lifecycle model can only extend the trade beyond that
baseline exit. It cannot exit earlier.

Trader interpretation:

Protocol265 says: "Take the Protocol261 research entry. Do not scalp out before
the baseline exit. At the baseline exit point, only keep holding if the
post-entry path still looks like it has continuation value."

This is closer to the user's stated intuition about not forcing arbitrary
minimum holds while also not teaching churn. It preserves the base trade and
only tests continuation.

### Primary Files And Artifacts

Code:

- `v4/scripts/run_protocol265_source_penalty_baseline_anchored_continuation.py`
- `v4/scripts/run_protocol200_lifecycle_continuation_policy.py`
- `v4/scripts/run_protocol251_premium_blend_slot_aware_lifecycle.py`
- `v4/scripts/run_protocol261_router_source_penalty_calibration.py`
- `v4/scripts/run_protocol263_source_penalty_lifecycle_attribution.py`
- `v4/scripts/run_protocol264_source_penalty_unified_entry_lifecycle.py`

Artifacts:

- `v4/audit/autoresearch/v4_aplus_hypothesis_265_source_penalty_baseline_anchored_continuation/summary.json`
- `v4/audit/autoresearch/v4_aplus_hypothesis_265_source_penalty_baseline_anchored_continuation/report.md`
- `v4/audit/autoresearch/v4_aplus_hypothesis_265_source_penalty_baseline_anchored_continuation/source_penalty_baseline_anchored_continuation_trades.csv`
- `v4/audit/autoresearch/v4_aplus_hypothesis_265_source_penalty_baseline_anchored_continuation/source_penalty_baseline_serial_trades.csv`

Important implementation caveat:

Protocol265 currently writes replay outputs, but it does not yet persist
deployable model/scaler artifacts the way Protocol101 does. That must be fixed
before any runtime parity or paper-default replacement work.

### Upstream Protocols Supporting Protocol265

Protocol265 rests on a newer research lineage:

- `Protocol240`: premium-leaning blended utility challenger, a broad
  research candidate with higher PnL but lower win rate and more churn.
- `Protocol248`: challenger failure-surface audit, created to stop mixing
  misleading metric scopes and to compare challengers apples-to-apples.
- `Protocol249`: entry-quality calibrator.
- `Protocol250`: contract-value selection head.
- `Protocol257`: complete router proposal stream, collecting proposal streams
  from Protocol101 and stronger full-action challengers.
- `Protocol258`: policy router over the complete stream with history.
- `Protocol260`: reliability priors for router decisions.
- `Protocol261`: source-penalty router calibration. This is the actual frozen
  entry stream used by Protocol265.
- `Protocol262`: slot-aware lifecycle over Protocol261. It beat Protocol101
  but trailed Protocol261 on recent 2026.
- `Protocol263`: attribution audit showing the recent gap was lifecycle/slot
  opportunity loss, not merely entry failure.
- `Protocol264`: unified entry-plus-lifecycle sequence model. It beat
  Protocol101 but under-traded recent 2026 versus Protocol261.
- `Protocol265`: baseline-anchored continuation, preserving Protocol261 exits
  while allowing learned extension.

### Protocol261 Entry Stream

Protocol261 is a source-penalty calibrated router over proposal streams. It
uses a complete router proposal stream from Protocol257 and chooses between
Protocol101-like proposals and stronger challenger proposals. Its purpose is to
penalize unreliable source switching while still allowing the broad challenger
stream when validation says it is useful.

Protocol261 aggregate metrics:

| Split | Median PnL | PF | Median Trades | Positive Seeds | `$0.25`/side Stress |
|---|---:|---:|---:|---:|---:|
| Q3 2025 | `$94,430` | `2.245` | `628` | `1.00` | `$60,490` |
| Q4 2025 | `$169,450` | `2.223` | `789` | `1.00` | `$130,000` |
| Q1 2026 | `$167,120` | `2.187` | `772` | `1.00` | `$130,720` |
| March 2026 | `$63,645` | `2.033` | `291` | `1.00` | `$47,545` |
| Recent 2026 | `$51,625` | `2.091` | `281` | `1.00` | `$37,525` |

Protocol261 is research-only, but it is the stronger entry source that
Protocol265 inherits.

### Protocol265 Candidate Universe

Protocol265 starts from Protocol261 model trades:

- source file: `v4/audit/autoresearch/v4_aplus_hypothesis_261_router_source_penalty_calibration/model_trades.csv`
- candidate entries loaded: `12,095`
- path records built: `11,518`
- path skips: `577`
- baseline trade rows: `10,248`
- model trade rows: `19,454`

These are not arbitrary raw-chain candidates. They are Protocol261 selected
research entries with path reconstruction from normalized official-context quote
data.

### Feature Set

Protocol265 uses the lifecycle path-record features from
`run_protocol200_lifecycle_continuation_policy.py`, not Protocol101's entry-only
feature set.

The lifecycle model sees causal post-entry state at each quote/path step:

- minutes since entry,
- minutes to forced flat,
- day progress,
- option bid, ask, mid,
- spread and spread fraction,
- bid/ask size,
- quote gap,
- underlying price,
- directional underlying move from entry,
- IV,
- delta,
- gamma,
- theta,
- vega,
- current path PnL,
- running MFE,
- running MAE,
- giveback,
- giveback fraction,
- time since MFE,
- 1/3/5-step PnL velocity,
- rolling 5/10-step PnL volatility,
- bid over entry ask,
- mid over entry ask,
- theta over mid,
- gamma/theta,
- time-theta burden,
- entry score,
- entry offset,
- entry premium,
- entry premium over starting cash,
- call/put side flags.

These are holding-state features. They are meant to answer: "Given we are
already in this contract, is it still worth occupying the single position slot?"

### Model Architecture

Protocol265 uses the `ContinuationMLP` from Protocol200:

- input: lifecycle path features per step,
- hidden dimension: `128`,
- GELU nonlinearities,
- layer normalization,
- dropout,
- scalar continuation-value output.

Training config in the current run:

- seeds: `[11, 22, 33]`,
- epochs: `7`,
- batch size: `8192`,
- max train steps: `650,000`,
- learning rate: `1e-3`,
- threshold candidates inherited from Protocol200:
  `[-300, -150, -50, 0, 50, 100, 200, 350, 500, 750, 1000, 1500, 2000]`.

### Training Objective

Protocol265 reuses the continuation target from Protocol200. For each path step,
the model is trained to predict future continuation utility:

```text
future_best_pnl - current_pnl - risk_penalty * adverse_excursion
```

The value is clipped and scaled during training. This is a supervised hindsight
label, but model inputs during evaluation are causal.

The key difference is in the simulator:

- the baseline Protocol261 exit is treated as an anchor,
- model exit is not allowed before the anchor,
- after the anchor, if predicted continuation falls below the selected
  threshold, the model exits,
- if it never falls below threshold, it can continue until forced flat,
- entry stream is unchanged,
- one-account serial replay is enforced.

This is deliberately conservative. It is not a fully learned entry/hold/exit
policy. It is a learned continuation overlay on top of Protocol261.

### Historical Performance

Protocol265 versus Protocol101:

| Split | Protocol265 Median PnL | Protocol101 Median PnL | Delta | Protocol265 PF | Trades |
|---|---:|---:|---:|---:|---:|
| March 2026 | `$63,645` | `$41,790` | `+$21,855` | `2.011` | `289` |
| Q1 2026 | `$159,345` | `$95,010` | `+$64,335` | `2.144` | `741` |
| Recent 2026 | `$53,055` | `$7,450` | `+$45,605` | `2.222` | `265` |

Protocol265 versus Protocol261:

| Split | Protocol265 | Protocol261 | Delta |
|---|---:|---:|---:|
| March 2026 | `$63,645` | `$63,645` | about `$0` |
| Q1 2026 | `$159,345` | `$162,575` | `-$3,230` |
| Recent 2026 | `$53,055` | `$48,905` | `+$4,150` |

Stress:

| Split | `$0.10`/side Stress | `$0.25`/side Stress |
|---|---:|---:|
| March 2026 | `$57,205` | `$47,545` |
| Q1 2026 | `$144,505` | `$122,245` |
| Recent 2026 | `$47,695` | `$39,655` |

Invariants:

- overlap violations: `0`,
- unaffordable violations: `0`,
- NaN time rows: `0`.

### Extension Behavior

Protocol265's extension behavior is the most important diagnostic.

| Split | Exit Reason | Rows | PnL | Baseline PnL | Delta | Median Extra Steps |
|---|---|---:|---:|---:|---:|---:|
| March 2026 | baseline anchor exit | `4,133` | `$1,046,285` | `$1,046,285` | `$0` | `0` |
| March 2026 | model extended exit | `95` | `-$79,980` | `-$46,820` | `-$33,160` | `4` |
| Q1 2026 | baseline anchor exit | `10,950` | `$2,453,430` | `$2,453,430` | `$0` | `0` |
| Q1 2026 | model extended exit | `133` | `-$111,420` | `-$77,360` | `-$34,060` | `5` |
| Recent 2026 | baseline anchor exit | `4,097` | `$700,905` | `$700,905` | `$0` | `0` |
| Recent 2026 | model extended exit | `46` | `$126,380` | `$105,350` | `+$21,030` | `8` |

Interpretation:

- The anchor preserved recent 2026 better than free early-exit lifecycle models.
- Learned extension helped recent 2026.
- Learned extension hurt Q1/March.
- Therefore, "hold longer" is not universally good; the model needs better
  context for when extension is worth it.

### Strengths

- Beats Protocol101 on all common lifecycle-harness splits.
- Fixes the recent-2026 degradation seen in Protocol262 and Protocol264.
- Preserves one-account serial replay.
- Preserves affordability and no-overlap invariants.
- Uses causal post-entry lifecycle features.
- Directly addresses the churn / early-exit / re-entry failure mode.
- Keeps the base trade intact instead of teaching arbitrary fast exits.

### Weaknesses

- Research-only; not operational paper default.
- Does not yet persist deployable model/scaler artifacts.
- Not integrated into live/paper runtime.
- Depends on Protocol261 selected entries rather than full raw action-space
  candidate generation.
- It is not a unified policy over `wait / enter call / enter put / hold / exit`.
- Baseline anchor is a designed constraint. It may be useful, but it is still a
  human-imposed structural prior.
- Extension logic is mixed: helpful in recent 2026, harmful in Q1/March.
- Has not yet passed no-order runtime parity.
- Has not yet been tested as a paper replacement.

## Direct Comparison: Protocol101 Versus Protocol265

### Conceptual Difference

Protocol101:

- primarily an entry/event-selection model,
- uses causal short-history features,
- exits are inherited from frozen candidate lifecycle artifacts,
- conservative paper-default behavior,
- operational runtime exists.

Protocol265:

- primarily a post-entry lifecycle continuation overlay,
- entries come from Protocol261,
- baseline exit is preserved as an anchor,
- model can only extend beyond the baseline exit,
- broader research behavior,
- no operational runtime yet.

### What Each Model Is Trying To Learn

Protocol101 asks:

```text
While flat, is this candidate trade worth entering now?
```

Protocol265 asks:

```text
After entering a Protocol261 trade and reaching the baseline exit point,
is this position still worth holding?
```

Neither model is yet the final desired neural trader. The final target should
eventually ask both questions in one coherent position-state policy:

```text
Flat: wait / enter call / enter put.
Holding: hold / exit.
Future phase: scale in / scale out only after one-contract behavior is proven.
```

### Action Space Difference

Protocol101:

- flat-state decision over a candidate event stream,
- enter selected candidate or wait,
- exits are external/frozen.

Protocol265:

- entry decision already happened in Protocol261,
- holding-state decision only,
- baseline exit or extend beyond baseline,
- cannot exit earlier than baseline.

### Runtime Readiness Difference

Protocol101:

- has saved model artifacts,
- has a daily paper-trading runbook,
- has IBKR paper-session plumbing,
- has monitor/logging infrastructure,
- is the current operational default.

Protocol265:

- has historical replay outputs,
- has no persisted model/scaler artifacts yet,
- has no live runtime adapter yet,
- has no no-order shadow parity yet,
- should not replace Protocol101 without additional work.

### Risk Profile Difference

Protocol101:

- fewer trades,
- higher PF in original gates,
- cleaner conservative profile,
- may undertrade or miss larger move capture.

Protocol265:

- more trades,
- lower PF than Protocol101 on March/Q1 but much higher recent PnL,
- captures more Protocol261-style opportunities,
- continuation overlay helps recent but hurts some earlier blocks.

## Where The Project Is Still Off From The Final Goal

The project is closer than v2/v3 because v4 uses executable bid/ask data and
strict serial replay. But it is not yet the final bot for these reasons:

1. The best operational model and the best research challenger are not the same.
   Protocol101 is operational; Protocol265 is research-only.

2. Protocol265 does not yet persist deployable artifacts. It needs model/scaler
   persistence, manifesting, hash/freeze support, and a runtime adapter.

3. The entry and lifecycle systems are still split. Protocol101 handles entry
   better operationally; Protocol265 handles one kind of lifecycle continuation.
   The final desired bot should learn flat and holding decisions in a unified
   train/live-compatible environment.

4. Candidate generation is still lineage-dependent. Protocol101 uses a capped
   event/candidate stream; Protocol265 uses Protocol261 selected entries. The
   desired final model should eventually see the full valid live action surface
   and choose intelligently.

5. Live parity is incomplete for the challengers. Protocol101 has live/paper
   plumbing; Protocol265 does not.

6. Historical replay remains 1-minute based for most coverage. That is useful,
   but 0DTE option prices can move violently intraminute. Live/paper logs and
   targeted 1s/tick audits are still important before replacement decisions.

7. Multi-contract sizing is intentionally not part of the operational model yet.
   Account-aware sizing and scale-in/scale-out remain research topics after
   one-contract live/paper behavior is reliable.

## Suggested Review Questions For The ML Engineer

### Data And Leakage

- Are all Protocol101 and Protocol265 input features genuinely causal?
- Are repaired Greeks being computed consistently across historical and live
  contexts?
- Is the Protocol261 entry stream creating hidden selection bias that makes
  Protocol265's lifecycle results less general?
- Are quote timestamps and SPX/VIX timestamps aligned closely enough for 0DTE
  execution?
- Are we overfitting to the currently downloaded blocks because the later
  research loop iterates heavily on them?

### Modeling

- Should Protocol265's anchor be treated as a legitimate structural prior or as
  a crutch that prevents true lifecycle learning?
- Should the continuation label include next-slot opportunity value, or should
  that be learned by a separate state/action model?
- Should the final model be:
  - two-stage entry then lifecycle,
  - unified entry/lifecycle policy,
  - sequence model / transformer over candidate surface and position state,
  - offline RL / imitation from a DP oracle,
  - or a hybrid supervised policy with conservative runtime gates?
- Should ranking loss be used for same-minute candidate arbitration?
- Should the model optimize dollar PnL, return on premium, risk-adjusted PnL, or
  a multi-objective utility?

### Simulation

- Is the current serial simulator close enough to live paper trading?
- Does starting cash affect selection enough that training should explicitly
  include account-state features?
- Should daily drawdown and premium exposure be part of the objective, not just
  reported?
- Are same-side re-entry chains being treated correctly?
- Should skipped future opportunities be part of the training target?

### Runtime

- What is the minimum no-order shadow evidence needed before Protocol265 can
  be considered as a paper replacement?
- How should Protocol265's lifecycle continuation model be plugged into the
  existing Protocol101 paper runner?
- Should the runtime evaluate every second or every minute, given the historical
  model's 1-minute training cadence?
- How should limit-order fill uncertainty be modeled for live/paper parity?

## Recommended Next Technical Work

Before a promotion decision:

1. Persist Protocol265 model artifacts:
   - `model.pt`,
   - `scaler.json`,
   - `manifest.json`,
   - feature columns,
   - thresholds,
   - hashes.

2. Add Protocol265 attribution:
   - where extensions help,
   - where extensions hurt,
   - side/time/moneyness/premium buckets,
   - whether bad extensions are mostly calls, puts, high-premium contracts, or
     late-day decay.

3. Build a no-order runtime parity harness for Protocol265:
   - same candidate inputs,
   - same lifecycle features,
   - same anchor behavior,
   - no broker order endpoint,
   - JSONL logs for every decision.

4. Compare Protocol265 and Protocol101 on the same live/paper shadow days before
   any replacement:
   - candidate availability,
   - predicted action,
   - quote freshness,
   - expected ask/bid,
   - actual paper fill if Protocol101 trades,
   - hypothetical Protocol265 decision.

5. Only after runtime parity passes, create a formal replacement decision packet.

Longer term:

1. Build a deployable unified position-state model:
   - flat: wait / enter call / enter put,
   - holding: hold / exit,
   - no overlapping positions,
   - same exact features in training and live.

2. Move from lineage-selected candidates toward full valid action-surface
   candidate generation.

3. Add account-state features and eventually research multi-contract sizing only
   after one-contract live/paper parity is strong.

4. Stage broader paid data purchases only after the model architecture and
   train/live contract are stable.

## Bottom-Line Status

`PAPER_DEFAULT_PROTOCOL101` is still the operational paper-trading default
because it has frozen deployable artifacts and live/paper infrastructure.

`CHALLENGER_SOURCE_PENALTY_BASELINE_ANCHORED_CONTINUATION_V1` is the stronger
recent lifecycle research clue. It beats Protocol101 on the common recent
lifecycle-harness splits and fixes a recent-2026 weakness, but it is not ready
to replace Protocol101 because it lacks deployment artifacts, runtime parity,
and a formal promotion packet.

The main ML hypothesis now is:

```text
The next edge is likely in position-state lifecycle intelligence:
preserve good entries, avoid premature churn, and learn when continuation is
worth occupying the single position slot.
```

That is the current bridge from a promising research system toward the actual
0DTE neural trading bot.
