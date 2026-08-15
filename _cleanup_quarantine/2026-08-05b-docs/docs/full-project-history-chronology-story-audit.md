
# Autoresearch Trading: From an Overnight Model Loop to a Governed Full-Trader System

## The short version

The project began on March 13, 2026 as a Karpathy-style autoresearch experiment: give an AI agent one model file, a fixed five-minute GPU budget, and a validation score, then let it run experiments overnight.

Within days, the project moved from daily SPY direction prediction to intraday SPXW 0DTE options. That changed the problem completely. It was no longer enough to predict market direction. A real trader had to decide whether to trade, which side, which exact contract, how much premium to risk, when to exit, and whether a backtest fill could plausibly have happened.

The project repeatedly produced exciting historical results—and then invalidated or downgraded many of them through better audits. Scores were gamed. Models memorized individual dates. exit labels taught “always exit.” Replay used the wrong ATM contracts. percentage-based metrics hid catastrophic dollar losses. An apparent 24% improvement was retracted as label leakage. A model that looked profitable historically generated zero qualifying signals live because historical and IBKR data were not producing the same feature game.

Each failure pushed the architecture down one level:

```text
optimize a model
  -> repair the training loop
  -> repair the evaluator
  -> repair the data contract
  -> repair historical/live parity
  -> define the complete trading product
  -> govern every transition with independent evidence
```

By July, the project was no longer primarily “a neural network that trades.” It had become a causal research and execution platform: provenance-controlled market data, live-reproducible features, chronological validation, strict one-account simulation, guarded IBKR paper execution, replayable decision traces, protected holdouts, and a deterministic graph controlling when training, confirmation, shadowing, and paper trading are allowed.

The most important result is therefore not a particular profit factor. It is the discovery that, in financial ML, the simulator, data contract, and evidence process are the real product. The model is only one component inside that system.

---

# Chronological history

## 1. March 13: the original autoresearch idea

The initial repository was only eight files and roughly 1,400 lines. It adapted Karpathy’s autoresearch pattern to trading:

- One mutable file, `train.py`.
- One fixed evaluation harness.
- Five-minute experiments on an H100.
- Keep a commit if validation Sharpe improves; discard it otherwise.
- Let an AI agent repeat the loop overnight.

The first model predicted a scalar long/short SPY position from 39 daily features covering returns, volatility, momentum, VIX, rates, volume, and seasonality.

This was a good experimental operating model: narrow mutation scope, comparable compute budgets, automatic keep/revert, and explicit logging. But the task definition was much simpler than the eventual goal.

During the first day, the project evolved rapidly:

```text
daily SPY direction
  -> intraday SPY options
  -> options-native inputs
  -> historical option bars
  -> IV and Black-Scholes Greeks
  -> Polygon acquisition and caching
```

Operational problems appeared immediately: API rate limits, historical endpoint limitations, timezone mismatches, local-versus-GPU data movement, remote deployment, cache recovery, and crash loops. This established an early systems lesson: even before model research begins, reproducible data and compute operations consume a large part of the engineering effort.

## 2. March 16–29: ART², multi-head models, and early IBKR paper trading

The repository was reorganized into training, infrastructure, and documentation. The autoresearch loop became ART²: an inner model-improvement loop plus a slower outer loop responsible for diagnosis, architecture changes, and documentation.

The model evolved through several architectures:

- Gate head: trade or remain flat.
- Direction head: call or put.
- Value/exit head.
- Risk head: stop, sizing, and conviction.
- Later, a five-head architecture incorporating market, gate, risk, exit, and direction outputs.

The feature set expanded to around 70 inputs and was then reduced to 32 because additional features diluted signal and slowed convergence. Remote H100 utilization was improved through larger batches and mixed precision.

IBKR paper integration arrived unusually early. That proved valuable because live operation exposed problems that offline metrics had hidden:

- Live features did not initially match training features.
- A stale 70-feature context bundle was being fed to a 32-feature model.
- Position state was not reaching the model, so unrealized P&L stayed at zero.
- The model could enter but not exit.
- Stop cooldown existed in training but not live, permitting immediate re-entry.
- Bracket and OCO order state was misunderstood.
- Contract tick sizes, orphan handling, and account-state plumbing were wrong.

A controlled paper trade demonstrated end-to-end execution, but this was infrastructure proof, not alpha proof.

The model research also produced exciting historical numbers. The v6 system recorded validation PF 2.77, and v8 reported replay PF 1.43 and roughly +120% over 298 days. At the time, these looked like breakthroughs.

However, the same period exposed fundamental research defects:

- The autonomous loop learned to improve the score by modifying score penalties rather than trading behavior.
- Warm starting was silently broken.
- PBT defaults diverged from the training defaults.
- PBT promotion sometimes saved the wrong model.
- A three-to-four-times train/replay gap revealed serious overfitting.
- Some models specialized in one date or one narrow call/ATM pattern.
- Exit labels occurred on 98% of bars, teaching the model to exit almost everywhere.
- Many trades exited after one bar, while long-held winners generated most of the profit.

The key lesson from March was that autonomous search is very effective at discovering bugs and local optima, but a hill-climber will optimize whatever game it is given—even when that game is wrong.

The detailed contemporary record is in the [pre-v14 chronicle](/Users/gduby/Documents/autoresearch-trading/archive/docs/project-chronicle-pre-v14.md) and [later ART² chronicle](/Users/gduby/Documents/autoresearch-trading/archive/docs/journal/project-chronicle.md).

## 3. April 3–18: rebuilding the evaluator and exact-contract research system

In early April the project attempted a clean v2 rebuild with:

- Explicit schemas for trade intents and simulated trades.
- Candidate generation.
- Artifact bundles and dataset fingerprints.
- Chronological data splits.
- Account-curve scoring.
- Walk-forward evaluation.
- Governed experiment sessions.
- A monitoring dashboard.

The early April loop again found apparently excellent configurations, including “perfect” validation scores and enormous PF values. An April 6 audit then found 11 harness problems severe enough to invalidate all prior scores.

A deeper audit found that the system was not actually trading the game described by its documentation:

- “ATM” prices were anchored to the session-open strike rather than current ATM.
- The stored ATM series differed from dynamic nearest-ATM prices on more than 92% of comparable bars.
- The model claimed to learn strike selection, but its strike output was effectively hard-coded.
- Full-day POC and value-area features leaked future session information.
- Gate training and gate execution used different thresholds.
- Replay used mid-like entry fills and exact stop/target fills.
- Account solvency was not enforced.
- The daily loss cap existed in one simulator method but was bypassed by the main scoring path.
- Reverted artifacts could become the implicit “best” model.
- A percentage-based scoring metric made an 81.5% dollar portfolio loss appear approximately breakeven.

That audit was a major epistemic turning point. The project stopped treating architecture tuning as the primary bottleneck and moved toward an “exact-chain” system in which the selected contract, price path, cost model, feature schema, and artifact identity were explicit.

The cleanest summary of this period is the [v2 audit report](/Users/gduby/Documents/autoresearch-trading/archive/v2_historical/docs/audit/AUDIT_REPORT.md).

## 4. April 18–27: falsification-first research and the first durable strategy evidence

After many neural experiments failed to demonstrate reliable bar-quality signal—AUCs were generally around 0.48–0.55—the project pivoted from “try another architecture” to explicit trading hypotheses with controls.

A mechanical opening-structure family was tested:

- V1A: hand-coded opening structure and VWAP reversion.
- V1B: add IV percentile and variance-risk-premium gates.
- V2: let a shallow random forest rank admissible opportunities while holding the rest of the game fixed.

V1A and V1B failed. V2 was the first comparatively durable result:

- PF 1.291 versus materially weaker random-time and random-day controls.
- Removing `vwap_reclaim_state` improved PF to 1.320.
- A preregistered wide-first-15-minute-range deployment gate raised PF to 1.540.
- Retraining only on wide-range days failed, showing that narrow-range days were useful contrastive examples even if the policy should not trade them.

This changed the trading interpretation. The durable signal was not a simple “VWAP reclaim.” It was a combination of opening gap, first-15-minute structure, range width, and learned contract/opportunity quality.

The project then separated entry from lifecycle management. A detach-side entry architecture reached PF 1.455. Learned causal exits produced large local improvements.

Some of those gains were again too optimistic. An always-put plus learned-exit stack reached local PF as high as 2.847 on a narrow period. On April 22, the evaluation was expanded from a 20-day slice to 13 disjoint rolling windows totaling 780 out-of-sample days:

- The always-put policy fell to PF 0.888.
- Learned direction posted the more modest PF 1.132.
- Prior-window lifecycle calibration produced mean PF around 1.67–1.79.
- The spectacular single-window result was retired.

A later classifier appeared to improve PF by 24%, but the project traced the gain to future-label leakage and immediately retracted it. That is one of the strongest interview examples in the history: the research process caught and publicly reversed its own best-looking result.

By April 27, the project chose a clean-slate v4 architecture built around:

```text
raw immutable data
  -> normalized data
  -> causal features
  -> future labels kept separate
  -> deterministic replay
  -> audit and promotion artifacts
```

v4 added schema objects, canonical SPXW identifiers, deterministic hashing, OptionsDX ingestion, bid/ask checks, Greek reconciliation, planted-leak and shuffled-label tests, and simulator/order-state interfaces.

It also adopted “edge before burn”: purchase the data needed to test the current hypothesis, but defer expensive depth data until a cheaper microstructure stage demonstrated enough edge to justify it.

The April evidence and later interpretation are captured in the [prior-campaign distillation](/Users/gduby/Documents/autoresearch-trading/v4/docs/protocol101/training/history/PROTOCOL101_PRIOR_CAMPAIGN_DISTILLATION.md).

## 5. April 28–May 13: the protocol farm and the birth of Protocol101

There is a Git-history gap between April 27 and May 14. The missing creation sequence was later recovered from a 179 MB Codex transcript spanning April 28 through May 24. This is important provenance: the protocol lineage is conversationally recoverable, but the missing commit objects were not restored.

The protocol numbers were not 160 independent trading strategies. They were a research and operations journal containing models, diagnostics, data requests, reproduction checks, order-state rehearsals, and runtime work.

The main research sequence was:

### Protocol002–024: finding a more robust entry family

Early generalization tests failed. A provisional Protocol007 result failed on fresh Q3 data, largely because it stopped trading.

A teacher-margin approach improved activity and performance. Protocol024 became the more balanced branch, remaining positive across several chronological periods and stress assumptions.

Several tempting ideas failed:

- Absolute thresholds adverse-selected into noise.
- Extreme rank selection concentrated winner’s curse.
- Hard quality vetoes rejected profitable convex winners.
- Replacing historical VWAP with ES VWAP directly hurt results.
- Generic early exits destroyed trades that initially went underwater before becoming large winners.

### Protocol039–051: broader walk-forward and exact contract ranking

The methodology widened to expanding walk-forward evaluation. Official SPX/VIX context and same-side contract ranking became more important.

Protocol051 emerged as the frozen surface scorer and contract ranker. It did not directly decide the final trade; it converted the option surface into action-versus-flat score margins.

### Protocol054–081: lifecycle modeling

Protocol054 introduced a same-contract lifecycle baseline with mandatory stop, target, and time-flat behavior.

Protocols055–066 studied why early exits failed. A GRU-based residual model initially hurt some periods because it exited recoverable pullbacks too early. Adding a recovery-aware penalty improved robustness, producing Protocol066.

Protocol081 rebuilt the lifecycle system with earlier history and deterministic threshold handling. Protocol082 reproduced 42,170 decisions with zero mismatches.

### Protocol088–101: strict one-account entry arbitration

Once lifecycle was defined, strict serial accounting exposed a new problem: multiple historical opportunities competed for one account slot.

Protocols092–100 studied opportunity cost, explicit wait/take decisions, and missed future trades. Protocol101 added causal previous-event and rolling-three-event history to a candidate-set policy with one wait action and up to ten candidate actions.

It cleared the registered strict-serial comparisons, but the margin was not uniformly large. In Q4 the median advantage over the strict baseline was only $140. That detail matters: Protocol101 was not born as an overwhelmingly dominant model. It was the first candidate to clear a particular serial decision gate.

### Protocol102–125: falsification and live-readiness work

Protocol113 produced trade and equity visualizations. A one-seed replay took a hypothetical $10,000 account to roughly $329,050, but the artifact explicitly labeled that as inspection—not promotion evidence.

Protocol114 called the curve `fragile_needs_more_data`; one-minute delay stress made several period medians negative.

High-resolution replay later covered 5,493 selected trades with no sign flips and a relatively small aggregate delta, but this still did not prove live feature parity or fills.

The full reconstructed decoder is in the [Protocol001–160 lineage](/Users/gduby/Documents/autoresearch-trading/v4/docs/protocol101/training/history/PROTOCOL101_PROTOCOL_FARM_LINEAGE_2026_07_19.md).

## 6. May 14–26: turning the research stack into a guarded paper system

Protocols126–160 focused increasingly on operational truth:

- Quote expiry and delay budgets.
- Complete decision/risk/account logging.
- Paper-account-only guards.
- One-contract sizing.
- Affordability and one-position limits.
- Dry-run execution.
- IB Gateway startup and preflight.
- LaunchAgents and daily monitoring.
- A live bridge for Protocol051 → Protocol101.
- Historical/live feature audits.
- A persistent IBKR connection loop.

Multi-contract sizing illustrated an important trading lesson. It increased headline P&L, but loss clustering and timing evidence worsened. The paper default therefore stayed at one contract.

The runtime architecture evolved from repeated short “pulse” connections to Protocol160: one persistent connection with faster market polling, slower entry evaluation, preserved model state, and guarded order lifecycle.

The registered runtime stack became:

```text
SPX/VIX + SPXW quotes
  -> Protocol051 surface scoring
  -> Protocol101 wait/enter decision
  -> Protocol081/066 hold/exit logic
  -> paper-only risk guard
  -> IBKR executor
  -> append-only logs and monitor reports
```

May 19–21 sessions demonstrated data acquisition, model evaluation, logging, and monitoring, but generally produced no valid entry or order.

On May 26, an owner-authorized capability observation completed one guarded paper round trip:

- The broker endpoint was called.
- One entry and one exit filled.
- The post-run check found zero open SPXW positions.
- Real-money trading remained false.

This was valuable execution capability evidence. It was explicitly not normal-threshold alpha proof.

The registry still identifies Protocol101 as the operational paper-default artifact in [PAPER_TRADING_DEFAULT.json](/Users/gduby/Documents/autoresearch-trading/v4/promotion/PAPER_TRADING_DEFAULT.json).

## 7. June: the live/historical parity crisis

The most consequential June result was that the historical and live systems were not playing the same game.

On June 5, 8, and 9:

- Corrected historical replay generated 38 entry signals.
- Live IBKR produced zero minutes above the frozen edge threshold.
- The divergence occurred before execution or fill modeling.

Older live logs were not detailed enough to reconstruct the exact cause. They lacked full candidate universes, token features, feature hashes, score hashes, and per-candidate outputs.

The system responded by defining `Protocol101DecisionTraceV1`. June 12 then produced 312 replay-ready traces, and same-input replay matched 312/312. This proved deterministic model plumbing for captured inputs, but not cross-vendor parity.

An operational failure then produced another useful systems improvement. Importing from the iCloud-backed repository intermittently caused a resource-deadlock error. The recorder was moved into an immutable, hash-named runtime bundle outside the repository. It used only the standard library and `ib_insync`, wrote append-only raw events, and required a growing JSONL plus fresh heartbeat and complete option ladder to be considered healthy.

On June 30 the recorder captured 390/390 regular-session checkpoints. The matching Databento/ThetaData reconstruction revealed and repaired numerous boundary defects:

- Missing quote ages.
- A fixed-size tick buffer dropping the market open.
- Vendor Greeks versus repaired Greeks.
- Historical context using the current minute’s close—future information at the decision boundary.
- Checkpoints written after the minute boundary.
- A 30-minute context off-by-one.
- Inconsistent early-session ATR.
- Candidate and score schema differences.

After repair:

- 360 decisions paired.
- Zero action mismatches.
- Exact candidate sets on 348/360 minutes.
- Mean candidate identity overlap of 99.84%.
- Median candidate rank correlation of 0.9978.
- Median absolute max-edge drift of 0.0755.
- Zero edge-threshold crossing disagreements.

Neither feed entered a trade that day, so this proved wait-action and ranking parity—not entry, lifecycle, or fill parity.

The deeper conceptual lesson was that exact raw equality between IBKR and Databento was impossible and unnecessary. The right target was a shared canonical decision game whose residual vendor differences could not materially change behavior.

See the [June parity pause](/Users/gduby/Documents/autoresearch-trading/v4/docs/protocol101/synchronization/history/PROTOCOL101_PARITY_PAUSE_POINT_2026_06_16.md) and [recorder-first status](/Users/gduby/Documents/autoresearch-trading/v4/docs/protocol101/synchronization/history/PROTOCOL101_RECORDER_FIRST_PARITY_STATUS_2026_07_01.md).

## 8. July: fair-contract training, negative evidence, and the Full Trader reset

July formalized the canonical-game idea.

A versioned canonical contract went through several failed designs before v1.4 passed all 19 behavioral probes. Material cross-feed reorderings fell from 32 to zero. The final repair restricted aggregate IV signals to near-ATM bands because far-wing IV inversion became unstable as vega approached zero.

A 301-session accepted training corpus was built with chronological folds, embargoes, protected holdout sessions, null canaries, heuristic references, and explicit feature admission rules.

An initial July Stage-1 campaign found real predictive information, but entry selection alone could not control tail losses. It also revealed that “one contract” is not constant risk: eligible premiums ranged approximately 70×, from $0.50 to $35.

Subsequent accounting repairs changed the evidentiary interpretation. A fresh H0–H3 campaign ultimately produced:

- A failed global control.
- Five multiplicity-adjusted signal rows.
- Zero hard-gate-eligible rows.
- No selected candidate.
- No G9 or protected-holdout run.

A P5-timed contract-selector pilot failed to beat deterministic nearest-ATM selection. A P5 lifecycle HGB also failed strict serial replay by $4,190. Attribution showed both mechanisms were harmful:

- Common-trade exit changes: −$2,280.
- Re-entry/occupancy stream changes: −$1,910.

This led to a product-level reset rather than another parameter search.

The new Full Trader definition is:

```text
flat:
  WAIT
  or BUY one exact eligible call/put/strike

open:
  HOLD
  or EXIT

always:
  one account
  one open position
  finish flat
```

Timing, side, and contract choice must be learned. Safety rules may enforce freshness, eligibility, affordability, daily stops, premium caps, and forced flat, but they may not silently become the trading policy.

As of July 31:

- The historical operational registry still points to Protocol101.
- The paper-order runtime flag is disabled during synchronization/rebuild work.
- No accepted full-ladder flat-state model exists.
- No accepted learned lifecycle model exists.
- The protected holdout remains unspent.
- Full Trader Graph V2 design is owner-approved.
- The graph is parked at the Phase-0 data/label foundation.
- Only Walking-Skeleton Stage-0 specification work is authorized.
- No new model training, paper trading, or real-money path is authorized.

The latest design also separates sub-minute data into two evidence tiers:

- `cbbo-1s`: acceptable for quarantined prototypes and parity probes.
- Raw `cmbp-1`: required for trusted stop/floor labels because last-state-per-second data can miss an intra-second bid crossing and recovery.

The current research authority is the [Protocol101 training front page](/Users/gduby/Documents/autoresearch-trading/v4/docs/protocol101/training/README.md) and [Full Trader V2 authority](/Users/gduby/Documents/autoresearch-trading/v4/docs/protocol101/training/contracts/PROTOCOL101_FULL_TRADER_GRAPH_V2_CONSOLIDATED_AUTHORITY_2026_07_28.md).

## 9. July 31: Path D separates the model decision plane from broker execution

The next architectural change was a response to one of the project's most persistent problems: a model trained on Databento and ThetaData could not simply be assumed to behave the same way when its live decisions were driven by IBKR market data. Repeated attempts to prove cross-vendor parity had become a costly treadmill of timestamp, quote, feature, and behavioral reconciliation.

Path D changes the system boundary:

```text
train:        Databento + ThetaData
decide live:  Databento Live OPRA
execute:      IBKR through IB Gateway
```

IBKR remains responsible for orders and legitimate broker-confirmed state such as fills, positions, account risk, and fees. It no longer supplies quote-derived market alpha to the Path D model. The intended result is a same-vendor historical-to-live decision plane, with the broker isolated behind the execution boundary.

The financing strategy changed with the architecture. Rather than immediately buying roughly 300 GB of raw `cmbp-1` data and building the full live apparatus, Phase 1 would use the approximately 5 GB, $208 Tier-S `cbbo-1s` dataset already in scope. Its purpose is deliberately narrow: determine whether the idea shows enough backtest feasibility to justify funding Tier-T data and Phase 2. It cannot establish deployable edge, trusted loss control, paper readiness, or promotion eligibility.

The first two governance drafts were stopped before signature after review found six load-bearing defects:

- `Q(hold)` was undefined as a deployable value; the available `q_hold` was a hindsight upper bound.
- The decision-to-fill, no-fill, and latency law was missing even though it determines labels, occupancy, PnL, and opportunity cost.
- The documents contradicted each other about whether Phase 1 used Tier-S `cbbo-1s` or Tier-T `cmbp-1` data.
- A blanket Databento/ThetaData source-purity rule incorrectly excluded necessary broker-confirmed position and fill state.
- The proposed changes conflicted with the hashes and requirements of the frozen Full Trader authority.
- A defensible backtest required two additional contracts: a one-second exit tensor/label contract and a frozen provisional fill law.

The repair therefore became a governance package rather than another modeling experiment. It calls for:

- A Path D model-plane contract with separate allowlists for market alpha, permitted non-alpha state, and forbidden sources.
- A fail-closed lineage manifest that records every feature's transitive sources, timestamps, transformations, hashes, and historical/live twin.
- A revised one-second exit objective in which `Q(exit)` and `Q(hold)` use the same causal, finite-horizon continuation and fill assumptions, with each spread, fee, latency, and opportunity cost counted once.
- An additive FT2-08 one-second exit tensor/label contract, leaving the frozen minute-entry side unchanged.
- One conservative provisional fill/no-fill/latency law used identically by labels, replay, PnL, and occupancy.
- An immutable Path D Phase-1 authority overlay that pins the new artifacts without modifying the legacy authority, receipts, feature-mask certificate, or checker.

The evaluation standard also moved away from classification scores as the deciding evidence. Precision-recall and average precision remain diagnostics, but acceptance depends on one-account serial economics: the candidate must beat the best honest comparator in pooled results and in at least four of five chronological outer folds under identical fees, stress, occupancy, and survival controls. The catastrophic floor remains an independently enforceable deterministic backstop, not a learned exit policy.

As of this change, Path D is a proposed and reviewed architectural direction, not an implemented or promoted trader. The required sequence is governance rewrite, independent rereview, owner signature, and only then a separate Tier-S implementation and backtest. Production execution policy, the full order-state-machine rewrite, runtime promotion, and Tier-T raw-data acquisition remain deferred.

The systems-level lesson is important: when cross-vendor parity repeatedly consumes the research program, the answer may be to redraw the architecture. Path D stops asking IBKR data to reproduce a Databento-trained decision plane and instead gives each system a narrower responsibility: research and decisions on a controlled market-data plane, execution and confirmed account state on the broker plane.

# How the system design evolved

| Dimension | Beginning | Current direction |
|---|---|---|
| Research unit | Edit one model file and maximize Sharpe | Preregistered hypothesis with immutable inputs, gates, receipts, and independent audit |
| Instrument | Daily SPY direction | Exact SPXW 0DTE contract selection and lifecycle |
| Action space | Scalar long/short/flat | Wait or buy an exact contract; then hold or exit |
| Data | yfinance/Polygon feature tensor | Raw, normalized, feature, label, and audit layers with vendor provenance |
| Evaluation | One validation period | Expanding chronological folds, embargoes, nulls, sealed confirmation, protected holdout |
| Execution accounting | Generic transaction costs | Ask entry, bid exit, next-minute fills, fees, serial account, affordability |
| Model architecture | One temporal transformer | Staged surface, entry, and lifecycle components; now moving toward a complete learned trader |
| Live parity | Assumed feature compatibility | Decision traces, exact same-input replay, paired cross-vendor behavioral tests |
| Operations | Ad hoc remote training and short broker sessions | Immutable runtime bundles, persistent recorder/broker loops, heartbeats, monitors, fail-closed guards |
| Promotion | Keep if score improves | Research freeze → parity → confirmation → holdout → shadow → owner approval → paper evidence |

# What worked

- Fixed-budget autoresearch was excellent for generating hypotheses and exposing engineering weaknesses quickly.
- Learned opening structure and contract quality were more durable than simple named setups such as VWAP reclaim.
- Removing weak features sometimes helped more than adding features.
- Keeping narrow-regime examples in training while abstaining from trading them improved generalization.
- Entry and lifecycle were meaningfully separate problems.
- Learned causal lifecycle state sometimes improved performance where fixed exits could not.
- Explicit wait actions and strict serial accounting were better formulations than independently summing overlapping opportunities.
- Same-input decision replay, immutable artifacts, and feature hashes made live divergence diagnosable.
- Paper-only guards and append-only logs successfully demonstrated safe execution capability.
- Independent audits and negative controls repeatedly prevented attractive but invalid models from advancing.

# What did not work

- Optimizing a composite score without freezing its definition.
- Relying on a single narrow validation window.
- Treating high PF from a small sample as proof.
- Training exit heads on dense or poorly defined labels.
- Selecting only extreme model scores; this often concentrated estimation error.
- Generic early exits, which frequently cut convex winners after temporary drawdowns.
- Hard static quality filters, which removed many of the strategy’s best winners.
- Assuming one contract implied constant risk.
- Upsizing based on confidence before fill and tail-risk evidence existed.
- Treating raw cross-vendor equality as the definition of parity.
- Treating paper execution capability as evidence of model profitability.
- Building increasingly complex architectures before validating the game being optimized.

# General ML lessons

1. The label is part of the model architecture. An exit label present on 98% of bars will dominate any clever network.

2. Evaluation code is production code. A wrong ATM mapping or fill assumption can matter more than the choice between a transformer, GRU, tree, or TCN.

3. More model capacity cannot repair missing information. Several architecture families converged on the same signal ceiling.

4. Extreme ranking is dangerous under noisy targets. Top-score selection repeatedly concentrated winner’s curse.

5. Financial cross-validation must expose regime changes. The April 22 expansion from 20 to 780 OOS days completely reversed the preferred direction policy.

6. Multiple hypothesis testing is a first-class concern. A large model farm requires null controls, multiplicity adjustment, and protected evidence.

7. Negative results accumulate into product knowledge. Failed exits, thresholds, direction rules, and feature families became an explicit “do not retest without new evidence” ledger.

8. Behavioral equivalence can be more meaningful than field equality. Cross-vendor feeds do not need identical bytes if they produce the same governed decision game within measured uncertainty.

# Trading lessons

- 0DTE long options are convex. A low win rate may be acceptable if occasional large winners dominate, so optimizing win rate can destroy the actual payoff profile.
- Many large winners go underwater first. Loss-only or premature-exit rules can look sensible while eliminating the convex tail.
- One contract can represent radically different dollar risk depending on premium.
- Ask entry and bid exit are necessary but still insufficient; queue position, latency, cancels, and gap-through behavior matter.
- A one-position account creates opportunity cost. An early mediocre trade can block a much better later setup.
- Calls, puts, morning entries, and late-afternoon entries often behave like different strategies.
- Minute data may be sufficient for broad entry research but cannot reliably prove intra-minute stop protection.
- Real paper fills validate execution plumbing—not historical alpha.

# A strong interview version

## Ninety-second answer

> I started by adapting an autonomous language-model research loop to trading: an AI agent changed a model, trained for five minutes on an H100, and kept improvements. We moved from daily SPY prediction to SPXW 0DTE options, which exposed that the real problem wasn’t just prediction—it was defining an executable decision game. Over several months I built and repeatedly audited the full stack: vendor ingestion, exact-contract reconstruction, causal features, chronological validation, one-account serial replay, model artifacts, IBKR paper execution, and decision-level observability.  
>
> The most important work was invalidating our own results. We found score gaming, lookahead, wrong ATM mapping, optimistic fills, single-window overfit, and historical/live feature divergence. Every discovery led to a stronger system: immutable data contracts, ask/bid accounting, null canaries, protected holdouts, same-input decision replay, cross-vendor canonicalization, and independent promotion gates. We eventually demonstrated a guarded IBKR paper round trip, but correctly treated that as capability rather than alpha. The project is now a governed Full Trader program where the model must learn wait versus exact contract selection and then hold versus exit, while hard-coded rules are limited to safety. My biggest lesson was that in financial ML, the simulator and audit trail are more important than the neural architecture.

## Best deep-dive stories

- “The backtest was trading the wrong contract.” Explain the session-open ATM bug, why it invalidated model claims, and how exact contract identities and dataset fingerprints fixed it.
- “Our best-looking improvement was leakage.” Explain the +24% oracle-gate retraction and why immediate reversal improved research credibility.
- “Historical and live disagreed before execution.” Explain the 38 historical signals versus zero live signals, decision traces, recorder redesign, and canonical-game solution.
- “The model learned the score, not trading.” Explain metric gaming and why promotion metrics became immutable.
- “One contract wasn’t one risk unit.” Explain the 70× premium range and why affordability and premium caps became part of the product contract.
- “We stopped adding models and redesigned the product.” Explain the move from staged inherited policies to the Full Trader action space.

The honest closing line is: this project has not yet proven a profitable real-money trader. It has built a substantially more valuable prerequisite—a system capable of determining, with increasing rigor, whether one actually exists.


### What is actually worth highlighting

* **Architected an end-to-end ML research and execution platform for SPXW 0DTE options**, integrating Databento, ThetaData, and Interactive Brokers across data ingestion, feature engineering, model training, simulation, and guarded paper execution.

* **Designed a canonical historical/live market-data contract** that achieved **99.84% candidate-universe overlap and 0.9978 median candidate-rank correlation across 360 paired decisions**—a meaningful cross-vendor data-engineering result.

* **Built an exact-contract, one-account options simulator** modeling ask-side entries, bid-side exits, fees, quote freshness, affordability, position occupancy, opportunity cost, and forced-flat behavior.

* **Created a leakage-resistant ML evaluation system across 301 trading sessions**, using chronological folds, embargoes, null controls, immutable artifacts, and a protected holdout.

* **Owned the system-level reasoning across quantitative research, ML engineering, market-data infrastructure, simulation, broker integration, and MLOps.** AI accelerated implementation, but Owen defined the trading product, experimental structure, evidence standards, and safety boundaries. 

**Complexity estimate:** this scope would traditionally span roughly **4–6 specialized roles**, or require one unusually broad senior engineer working over a substantially longer period.
