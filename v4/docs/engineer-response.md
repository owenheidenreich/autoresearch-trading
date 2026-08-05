> **SUPERSEDED 2026-08-05.** This document treats a **2026-05-23** handoff as the project source of truth. It is not. Current status is [`STATUS.md`](../../STATUS.md).

I’m treating the attached handoff from **2026-05-23** as the project source of truth. The core project is a v4 SPXW 0DTE neural trading system whose intended final action space is: do nothing, buy one call, buy one put, hold, or exit, with ask-entry/bid-exit accounting, one account, one open position, one contract, affordability constraints, no lookahead, and mandatory flat-before-close.  

## 1. Plain-English restatement

This project is trying to build a **neural day-trading decision engine for same-day SPXW options**. The desired bot should sit through the trading day, observe the current market, observe the available 0DTE SPXW option contracts, and decide:

> “Should I do nothing, buy a call, buy a put, keep holding my current option, or exit?”

The important part is that this is not meant to be a pretty backtest over mid-prices. It is meant to approximate the future live/paper game: enter at executable ask, exit at executable bid, trade only contracts the live system could identify and afford, use only causal features, avoid overlapping trades, and end the day flat. The handoff is unusually clear that older overlapping candidate PnL may remain useful for diagnostics, but promotion evidence must come from **strict one-account serial replay**. 

The present system has two main model families:

**Protocol101** is the operational paper default. It is mostly an **entry/event-selection model**: while flat, it scores candidate trades using entry features and causal short-history features. Once it enters, the exit behavior is inherited from older frozen lifecycle logic, so Protocol101 is not a complete learned entry/hold/exit policy. 

**Protocol265** is a stronger recent research clue but not operational. It starts from Protocol261 entries and uses a lifecycle continuation model. Its key design is: do not let the model exit before the baseline Protocol261 exit; only let it decide whether to extend beyond that baseline exit. That directly attacks premature churn, but it is still a continuation overlay, not a unified trading policy. 

My diagnosis: the project is closest to the right problem when it treats trading as **sequential decision-making under execution frictions**, not as “predict whether the next trade is profitable.”

---

## 2. Diagnosis of the core ML problem

### Best framing

The true problem is a **finite-horizon, partially observed, constrained sequential decision problem**. In ML terms, it is a hybrid of:

* **Contextual ranking / supervised learning** for choosing among candidate contracts while flat.
* **Optimal stopping / lifecycle policy learning** for hold versus exit.
* **Offline RL / conservative policy improvement** because the system learns from historical logged market trajectories rather than exploring online.
* **Sequence modeling** because the decision quality depends on the path of the session, recent candidate flow, volatility regime, quote dynamics, position state, and opportunity cost of occupying the single slot.
* **Imitation learning from an oracle or baseline** if you construct dynamic-programming labels from historical replay.

I would not frame the project as ordinary supervised learning. Supervised learning is currently being used as a training tool, but the deployed object is a **policy**. The actions change the future state distribution: entering a trade blocks later entries; exiting frees the slot; holding can preserve a convex winner or trap capital in decaying premium. That violates the usual i.i.d. assumption behind standard supervised examples, a problem explicitly highlighted in imitation-learning literature for sequential prediction settings where future observations depend on prior actions. ([Proceedings of Machine Learning Research][1])

I also would not jump straight to unconstrained online RL. Financial online exploration is dangerous, sample-inefficient, and likely unacceptable here. The more appropriate family is **offline policy learning with strong conservative constraints**. Offline RL research emphasizes both the promise of learning policies from fixed datasets and the practical challenge that learned policies can exploit distribution shift in actions not well-covered by the historical data. ([arXiv][2]) Conservative Q-Learning and Implicit Q-Learning are relevant because they explicitly try to improve a policy from offline data while controlling overestimation and behavior-policy deviation. ([arXiv][3]) ([arXiv][4])

### What the model is actually being asked to learn

The desired model is not merely learning “direction” or “edge.” It must learn something closer to:

[
Q(s_t, a_t) =
\text{expected future executable utility from taking action } a_t
\text{ in state } s_t
]

where the state includes:

* current market context,
* option surface context,
* candidate set,
* bid/ask and quote freshness,
* time to forced flat,
* current position state,
* account state,
* recent session history,
* current unrealized PnL, MFE/MAE, giveback, theta burden, and volatility,
* and the opportunity cost of using the one available position slot.

The key latent question is:

> “Is this action worth more than the best realistic alternative I give up by taking it?”

That applies to both entry and lifecycle. A good entry is not just an option that later goes up. It is an option that was better than waiting, better than the other available contracts, and still profitable after spread, slippage, fill uncertainty, and the chance of blocking a better future trade. A good hold is not just “the position might go higher.” It is “continuing to hold has higher utility than exiting now at bid and potentially using the freed slot later.”

### Where the current formulation is too weak or misaligned

The current system is strong on audit discipline, but the ML formulation is still fragmented.

Protocol101 asks, essentially:

> “While flat, is this candidate worth entering?”

But it does so over a lineage-derived candidate stream, not the full valid SPXW action surface, and its PnL target depends on an externally frozen lifecycle. That means it may learn “which historical candidate generator plus frozen exit path worked” rather than a general trading policy. 

Protocol265 asks:

> “After the Protocol261 trade reaches the baseline exit, is continuation worthwhile?”

That is a valuable lifecycle experiment, but it cannot cut losers before the baseline exit, and its continuation target is a hindsight utility of `future_best_pnl - current_pnl - risk_penalty * adverse_excursion`. The inputs are causal, but the label is an oracle-style future-best label. That can be useful supervision, but it is not the same as learning the action value of hold versus exit under the deployed policy. 

The strongest evidence that the lifecycle model is not yet generally solved is the Protocol265 extension attribution: extensions helped Recent 2026 but hurt March/Q1. The handoff itself notes that “hold longer” is not universally good; the model needs better context for when extension is actually worth occupying the slot. 

---

## 3. Critique of the current approach

### Entry prediction

Protocol101’s entry model is valuable because it is conservative, causal, persisted, operationally integrated, and evaluated under strict serial replay. Those are real strengths. But as an ML formulation, entry prediction is currently too dependent on a curated upstream candidate universe. Protocol101’s candidate stream is explicitly not the full SPXW option chain; it is derived from prior selected opportunities and lifecycle artifacts. 

That creates several risks:

1. **Selection bias.** The model may not learn what makes a contract good; it may learn what makes a pre-filtered candidate good.
2. **Missing hard negatives.** A full action surface includes many contracts that should be rejected. The model needs to see those in training, not merely easy candidates.
3. **Weak abstention training.** “Do nothing” is a first-class action. It should not be represented only as “no candidate crossed threshold.”
4. **Insufficient candidate arbitration.** When multiple contracts are available, the model should learn relative ranking: this call versus that call, call versus put, near-ATM versus OTM, enter now versus wait.
5. **Frozen-exit coupling.** The entry label is only meaningful relative to the exit policy used to score it. If the lifecycle policy changes, the entry labels may become stale.

The right next formulation is not merely a better binary classifier. It is a **candidate-set ranking model with abstention**:

[
A_{\text{enter}}(s_t, c_i) =
Q(s_t, \text{enter contract } c_i) - Q(s_t, \text{wait})
]

The entry model should learn whether each available contract has positive advantage over waiting, and then rank contracts by expected utility under the same lifecycle policy that will be deployed.

### Hold/exit lifecycle modeling

Protocol265 is directionally more interesting than Protocol101 because it attacks the lifecycle problem directly. Its lifecycle feature set includes causal post-entry variables such as current path PnL, running MFE/MAE, giveback, PnL velocity, rolling volatility, theta burden, gamma/theta, bid over entry ask, and time since entry. Those are the right kinds of features for a hold/exit policy. 

The problem is the objective. A target based on future-best PnL is not the same as the deployed decision problem. Future-best labels tend to teach the model about the best point that will occur later, but at live time the policy must choose whether to hold one more step under uncertainty. A better lifecycle target is the **advantage of holding over exiting now**:

[
A_{\text{hold}}(s_t) =
Q(s_t, \text{hold}) - Q(s_t, \text{exit now at bid})
]

This target should include:

* spread cost already paid,
* current executable exit bid,
* remaining time decay,
* realized volatility regime,
* probability of further upside,
* probability of catastrophic giveback,
* opportunity cost of keeping the position slot occupied,
* and the distribution, not just mean, of possible outcomes.

Protocol265’s baseline anchor is a useful structural prior, but it is also a crutch. It prevents one failure mode—premature early exits—but creates another: inability to cut losers before the anchor. The correct long-term goal is not “never exit early” or “hold at least N minutes.” The correct goal is:

> Exit when the learned continuation advantage turns negative, and hold when the learned continuation advantage remains positive.

That lets winners run without hardcoded hold rules and cuts losers without hardcoded percentage stops.

### Trade churn

The churn problem is not solved by adding minimum hold times. Minimum hold rules are usually symptoms of a model that has not internalized transaction costs, slot opportunity cost, or state transition costs.

A policy that exits and immediately re-enters should pay an explicit **switching cost** in training:

[
\text{switching cost}
=====================

\text{spread tax}
+
\text{fill uncertainty}
+
\text{latency risk}
+
\text{opportunity cost}
+
\text{model uncertainty penalty}
]

The key idea is to make churn economically unattractive when it is unattractive in reality, rather than forbidding it by rule. Some re-entries may be valid; many are just the model repeatedly paying the spread because the labels were not account-state-aware.

### Train/live parity

The handoff is right to preserve train/live parity as a hard requirement. Protocol101’s major advantage is not necessarily that it is the best research policy; it is that it has deployable artifacts, live/paper plumbing, logging, guards, and persistent manifests. Protocol265 currently lacks persisted model/scaler artifacts, runtime adapter, and no-order parity, so it should not replace Protocol101 yet. 

The train/live parity standard should be stricter than “same feature names.” It should require:

* same candidate generator code path,
* same contract filters,
* same quote freshness rules,
* same Greek repair logic,
* same scaling and missing-value handling,
* same action masks,
* same time-zone/session logic,
* same account-state transitions,
* same lifecycle state machine,
* and logs for every decision opportunity, including rejected candidates.

The 1-minute replay versus live runtime cadence is a major unresolved issue. Historical 1-minute CBBO can support broad research, but 0DTE options can move violently intraminute. The handoff already notes that live/no-order shadow and paper logs are needed to validate timing and routing. 

### Simulator realism

The project has already made the most important move: abandon midpoint fantasy and use ask-entry/bid-exit accounting. That is necessary, but not sufficient.

Databento’s own schema documentation distinguishes full order-level MBO, top-of-book update streams, interval BBO/CBBO, trades, aggregate bars, instrument definitions, and statistics. In particular, interval CBBO provides consolidated best bid, best offer, and sale at 1-second or 1-minute intervals; it is not the same as a full order book or full order event stream. ([Databento][5])

That matters because a live option order at the displayed ask is not automatically filled. The simulator needs a calibrated fill model:

* quote age at decision time,
* bid/ask size,
* spread width,
* recent quote movement,
* underlying movement since quote,
* option premium bucket,
* time of day,
* side/call/put,
* whether the ask was lifted in trades shortly after,
* cancel timeout,
* and latency between decision, order submission, and broker acknowledgment.

For SPXW specifically, the official product specs state that SPXW options ordinarily cease trading on expiration day at 4:00 p.m. ET, with earlier cessation on half-day holidays. That supports the flat-before-close design, but it also means the final hour has special microstructure and forced-liquidation risk. ([Cboe Global Markets][6])

ABIDES is relevant as a research comparison because it was built as a high-fidelity market simulation environment for AI agents, with message-based simulation, exchange interaction, and latency modeling. Your project does not need a full ABIDES-style multi-agent simulator immediately, but it does need the same mindset: the market simulator is part of the ML problem, not an afterthought. ([arXiv][7])

### Data sufficiency

The data is probably insufficient for a high-capacity unified sequence model, but more data is not automatically the answer.

Protocol101 has 23,530 decision events and a mean of about 2.22 candidates per event; Protocol265 has about 11,518 built path records from Protocol261 selected entries. Those are useful, but small relative to the non-stationarity, action sparsity, class imbalance, and heavy-tailed outcome distribution of 0DTE options.  

More data is likely to help when:

* the simulator is already realistic enough that additional labels mean the same thing as future live labels,
* candidate generation is stable and live-compatible,
* the target is not dominated by hindsight artifacts,
* the model is variance-limited rather than bias-limited,
* the added period contains regimes not already represented,
* the evaluation holdout remains untouched,
* and added data improves truly embargoed periods, not just the validation blocks repeatedly used during research.

More data may not help when:

* the action space remains lineage-selected,
* labels are misaligned with the deployed policy,
* 1-minute replay hides the fill failure that kills live performance,
* the strategy has been repeatedly tuned against the same protected periods,
* old regimes differ structurally from the current 0DTE market,
* or the model learns stale microstructure patterns that no longer exist.

Before expensive broad acquisition, I would require a **data value test**: add a small, well-chosen, previously untouched slice and verify that it improves a frozen protocol on an even later untouched slice or live-shadow sample. If added data only improves in-sample or near-validation metrics, acquisition is not justified.

### Label construction

The biggest ML weakness is not feature count; it is label semantics.

Current labels appear to mix three things:

1. Entry quality under a frozen lifecycle.
2. Continuation value under a baseline anchor.
3. Historical best future PnL adjusted by adverse excursion.

These are useful diagnostics, but they are not yet the action-value labels of a coherent policy.

Better labels should be constructed around action advantages:

[
A_{\text{enter}} = Q(\text{enter candidate}) - Q(\text{wait})
]

[
A_{\text{exit}} = Q(\text{exit now}) - Q(\text{hold})
]

[
A_{\text{switch}} = Q(\text{exit and later re-enter}) - Q(\text{continue current position})
]

The label should be computed under the same constraints as deployment: one account, one position, one contract, ask-entry, bid-exit, no overlaps, forced flat, and live-feasible candidate availability.

### Objective functions

Accuracy, win rate, and raw median PnL are all incomplete. Profit factor is useful but still insufficient. The actual objective should be a **serial account utility**:

[
U =
\text{net executable PnL}
-------------------------

## \lambda_1 \cdot \text{drawdown}

## \lambda_2 \cdot \text{tail loss}

## \lambda_3 \cdot \text{spread/slippage sensitivity}

## \lambda_4 \cdot \text{trade concentration}

\lambda_5 \cdot \text{policy instability}
]

For training, I would use multi-task heads:

* candidate utility / advantage,
* pairwise or listwise candidate ranking,
* abstain probability,
* hold-versus-exit advantage,
* quantiles of future return,
* probability of catastrophic loss,
* model uncertainty / OOD score.

For evaluation, keep the current strict serial replay discipline, but add distributional metrics: per-day PnL distribution, worst-day loss, largest giveback, concentration by day/time/side/premium bucket, sensitivity to spread and fill assumptions, and performance under embargoed splits.

### Protected holdout methodology

The current skeptical promotion standard is good: strict serial replay, ask/bid accounting, no overlaps, no unaffordable trades, slippage stress, no protected-holdout tuning, and runtime/no-order parity before replacing the paper default. 

But there is still a meta-overfitting risk. The project has many protocols, many thresholds, many feature variants, and repeated comparisons on Recent 2026. Even when each individual experiment is “proper,” the research process can overfit through repeated model selection. Bailey, Borwein, López de Prado, and Zhu specifically argue that standard holdouts can be unreliable for investment backtests and propose estimating probability of backtest overfitting through combinatorially symmetric cross-validation. ([SSRN][8])

I would introduce:

* an experiment ledger with every attempted protocol,
* a frozen untouched “sacred” period,
* embargo gaps between train/validation/test,
* day-level block bootstrap confidence intervals,
* PBO/CSCV-style false discovery checks,
* deflated Sharpe or equivalent multiple-test adjustment,
* and live-shadow days that are not used for model selection.

---

## 4. Comparison to similar technical problems and research areas

### Neural trading agents

This resembles neural trading-agent research, but with more operational realism than most papers. Many academic trading agents optimize stylized rewards with simplified execution. Your project’s ask-entry/bid-exit, one-account serial replay, affordability, and flat-before-close constraints are closer to a deployable research environment.

The main missing piece is that the policy is still decomposed into separate historical artifacts. The final agent should have a unified state machine:

> flat → wait / enter call / enter put
> holding → hold / exit

The handoff explicitly identifies this as the long-term target. 

### Offline RL for financial markets

Offline RL is relevant, but dangerous if used naively. The dataset contains actions selected by earlier protocols, not all possible live actions. Offline RL methods are vulnerable to overestimating actions outside the data distribution; CQL directly addresses this by learning conservative value estimates, while IQL avoids directly querying values for unseen actions and extracts policies via advantage-weighted behavioral cloning. ([arXiv][3]) ([arXiv][4])

For this project, offline RL should be used conservatively:

* action masks should restrict to live-feasible candidates,
* values should be pessimistic under uncertainty,
* policy improvement should be incremental over Protocol101/261 baselines,
* and every offline improvement must survive serial replay plus no-order live shadow.

### Execution-aware trading systems

Execution-aware trading systems treat execution cost, latency, and market impact as part of the optimization. Almgren-Chriss is a classic reference for optimizing execution by balancing risk and transaction costs, although it targets portfolio execution rather than 0DTE option entry/exit. ([그대안의작은호수 | 살아온 날의 흔적, 살아갈 날의 기록][9])

Your project has minimal market impact because it is one contract, but it still has meaningful execution risk: spread crossing, stale quotes, fill uncertainty, cancel timing, and adverse selection. So the relevant lesson is not “market impact model”; it is “execution cost belongs inside the objective.”

### Options market microstructure

0DTE SPXW options are especially challenging because the premium path is dominated by:

* underlying movement,
* implied volatility movement,
* gamma exposure,
* theta decay,
* spread dynamics,
* quote staleness,
* and time-to-close constraints.

The current feature sets already include many of these ingredients: spread fractions, bid/ask sizes, IV, delta/gamma/theta, gamma/theta, theta burden, giveback, MFE/MAE, and PnL velocity. The next step is to model them as a **path-dependent decision state**, not merely as tabular features at isolated events.

### Sequence models for decision policies

Decision Transformer is relevant because it frames RL as conditional sequence modeling over past states, actions, and returns using a causal transformer. ([arXiv][10]) That idea maps naturally to a trading session: each day is a sequence of market states, candidate sets, actions, fills, and rewards.

But I would not start with a large Decision Transformer. The dataset is probably too small and too biased. A better progression is:

1. MLP baselines for entry and lifecycle.
2. GRU/TCN over recent session state.
3. Candidate-set encoder plus small temporal transformer.
4. Return-conditioned sequence model only after the simulator and labels are stable.

### Noisy, non-stationary, sparse-reward systems

This problem has all the hard ingredients:

* rare high-convexity winners,
* frequent noisy small losses,
* non-stationary regimes,
* sparse terminal rewards,
* heavy-tailed PnL,
* high sensitivity to transaction costs,
* and policy-induced distribution shift.

That makes ordinary train/validation accuracy nearly useless. The most important evidence is not classification score; it is stable, executable, serial replay performance across time, followed by no-order shadow parity and paper evidence.

---

## 5. Research-backed takeaways

The relevant established lessons are:

1. **Offline policy learning is plausible but distribution-shift constrained.** Offline RL can learn from previously collected datasets, but learned policies can exploit unsupported actions unless conservative methods or behavior constraints are used. ([arXiv][2]) ([arXiv][3])

2. **Sequential imitation is not i.i.d. classification.** When actions change future observations, pure supervised behavior cloning can fail under the induced state distribution. ([Proceedings of Machine Learning Research][1])

3. **Sequence models are promising only if the sequence contract is correct.** Decision Transformer-style approaches are conceptually relevant, but they need enough high-quality trajectories and consistent causal state/action/reward encoding. ([arXiv][10])

4. **Financial backtests require special overfitting controls.** Ordinary holdouts are weaker in investment simulations because the research process itself selects among many variants. ([SSRN][8])

5. **Friction-aware objectives are essential.** Deep Hedging is not the same task, but it is relevant because it explicitly frames derivatives trading under transaction costs, liquidity constraints, and risk limits as a sequential learning problem. ([arXiv][11])

---

## 6. Novel directions for this project

### A. Build a unified position-state model

The next serious architecture should represent the whole live decision state:

```text
session_state_t:
  market context
  volatility context
  candidate option set
  current position, if any
  account state
  time-to-close
  recent decision/action/fill history
```

Then use action masks:

```text
if flat:
  actions = wait, enter candidate_1, ..., enter candidate_k

if holding:
  actions = hold, exit
```

A good architecture would be:

* **candidate encoder**: shared MLP over each option candidate;
* **candidate-set pooling / attention**: learns relative opportunity quality across contracts;
* **temporal encoder**: GRU, TCN, or small causal transformer over recent session states;
* **position encoder**: separate features for current trade lifecycle;
* **policy heads**:

  * enter ranking,
  * abstain,
  * hold/exit,
  * value estimate,
  * risk quantiles,
  * uncertainty/OOD score.

This would unify Protocol101’s entry strength and Protocol265’s lifecycle insight.

### B. Train from a dynamic-programming oracle, not arbitrary hold rules

For each historical day, construct a DP oracle under the exact deployment constraints:

* one account,
* one open position,
* one contract,
* ask-entry,
* bid-exit,
* no overlaps,
* affordability,
* forced flat,
* live-feasible candidate set,
* realistic fill/slippage assumptions.

The oracle computes the best action value at each state. Then train the neural policy to imitate the oracle’s **action advantages**, not just profitable trades.

This directly solves the “let winners run / cut losers” problem without arbitrary rules. A winner runs when the oracle says hold has positive continuation advantage. A loser is cut when exit has higher value than hold. The decision boundary is learned from context, not from a fixed percentage stop.

### C. Replace future-best lifecycle labels with hold/exit advantage labels

Protocol265’s continuation target should be replaced or augmented with:

[
A_{\text{hold}} =
V_{\text{continue}}(s_t) - V_{\text{exit now}}(s_t)
]

Train the lifecycle model to predict:

* mean continuation advantage,
* downside quantile,
* probability of giving back more than X dollars,
* probability of recovering to new MFE,
* expected value of freeing the slot.

The model should not merely know “there will be a better price later.” It should know whether holding one more decision interval is worth the risk and opportunity cost.

### D. Add opportunity-cost-aware lifecycle targets

The single-position constraint is central. Holding a trade has a hidden cost: it blocks all later entries. Current lifecycle labels appear too focused on the current path. They need to include:

```text
value_of_holding_current_position
minus
value_of_exiting_and_reopening_opportunity_stream
```

This is likely the missing ingredient behind churn and under/over-extension. A model that does not price the next-slot opportunity will either overhold decaying positions or exit too quickly to chase noisy new candidates.

### E. Use distributional prediction, not only scalar scores

0DTE options have asymmetric, heavy-tailed outcomes. A scalar expected PnL can be misleading. Add quantile or distributional heads:

* 10th percentile continuation PnL,
* median continuation PnL,
* 90th percentile continuation PnL,
* probability of near-total premium loss,
* probability of 2x/3x premium expansion,
* expected giveback from current MFE.

Then the policy can distinguish:

* low-mean but high-convexity continuation,
* high-mean but catastrophic downside,
* small scalps that are not worth spread,
* and true runaway winners.

### F. Use richer historical context, but only if live-feasible

Useful causal context likely includes:

* SPX returns over 1/3/5/15/30/60 minutes,
* realized intraday volatility,
* opening gap and overnight range,
* distance from VWAP or intraday high/low,
* VIX level/change,
* option surface skew/slope around ATM,
* ATM straddle price and intraday change,
* liquidity regime: spread, size, quote update frequency,
* time-of-day regime,
* economic calendar / FOMC/CPI/NFP flags,
* previous day realized range,
* current day trend versus mean-reversion features.

Every feature must be reproducible live. Historical-only features that cannot be computed in paper/no-order mode should be banned.

### G. Introduce conservative policy improvement

Instead of asking, “Can a new model beat Protocol101 in backtest?” ask:

> “Can this model improve Protocol101 or Protocol261 only in states where the evidence is strong, and defer to the baseline elsewhere?”

This can be implemented as:

* baseline action value,
* challenger action value,
* uncertainty penalty,
* OOD penalty,
* minimum advantage margin,
* and source reliability prior.

This is philosophically close to Protocol261/265, but should be made explicit at the policy level.

### H. Calibrate a fill model from no-order and paper evidence

Before broad data expansion, collect targeted evidence on the live execution gap:

* predicted ask at decision time,
* actual ask when order sent,
* fill or no fill,
* time to fill,
* cancel reason,
* spread at submit,
* quote age,
* underlying movement during latency,
* exit bid at decision,
* actual exit fill,
* paper/live discrepancy.

Then fit a stochastic fill model and replay historical decisions through that model. A strategy that survives deterministic ask/bid but fails calibrated fill replay is not ready.

---

## 7. More historical data: when it helps and when it does not

The hypothesis “more historical data may produce a cleaner and more generalizable model” is plausible, but only conditionally.

### More data is likely to help if the project is variance-limited

More data should help when:

* the same feature/label contract is stable,
* the simulator approximates live enough,
* candidate generation is no longer lineage-biased,
* the model is underfitting rare regimes due to too few examples,
* extension behavior differs by identifiable regime,
* and performance improves on untouched periods after training on broader data.

This is especially true for lifecycle learning. Rare continuation winners and catastrophic givebacks need many examples. A small lifecycle dataset can easily learn the wrong boundary.

### More data will not fix a biased problem

More data will not solve:

* wrong labels,
* midpoint or stale-quote fantasy,
* hidden lookahead,
* non-live candidate construction,
* repeated tuning on the same holdouts,
* missing fill uncertainty,
* or a fragmented entry/lifecycle objective.

In those cases, more data can make the model more confidently wrong.

### Conditions before expensive broad acquisition

I would require the following before broad paid expansion:

1. A frozen unified replay spec.
2. A live-compatible candidate generator.
3. A stable feature manifest.
4. A clear label definition tied to policy action values.
5. Protocol265 or successor artifact persistence.
6. No-order parity logs showing historical/live feature agreement.
7. A small-slice acquisition test showing marginal value.
8. A protected evaluation plan that the new data cannot contaminate.

Data acquisition should be staged:

* **Stage 1:** targeted high-resolution slices around actual trades, rejected high-score candidates, and churn episodes.
* **Stage 2:** broader 1-minute CBBO history after the policy/label contract is frozen.
* **Stage 3:** tick/order-book-level data only if fill-model error is proven to dominate strategy uncertainty.

---

## 8. Bottom-line assessment

Protocol101 is the correct operational default for now because it has deployable artifacts and paper infrastructure. Protocol265 is the more interesting ML research direction because it identifies lifecycle intelligence as the likely next edge, but it is not ready for replacement because it lacks deployment artifacts, runtime integration, and no-order parity. The handoff’s own bottom line matches this: Protocol101 remains the paper default, while Protocol265 is the stronger recent lifecycle clue but not yet a paper replacement. 

My strongest recommendation is to stop thinking of the next model as “better entry” or “better exit.” The next model should be a **unified constrained policy** trained on **action advantages** from a realistic serial replay game.

---

# Prioritized research agenda



| Priority | What to try next                                                                                                                          | Why it is likely to help                                                                                                                                 | Evidence that would confirm it                                                                                                  | Evidence that would falsify it                                                                              |
| -------: | ----------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
|        1 | **Persist Protocol265 artifacts and reproduce exact replay from saved model/scaler/manifest.**                                            | It closes the gap between research result and deployable candidate.                                                                                      | Replayed metrics match current Protocol265 from frozen artifacts; feature hashes and thresholds are stable.                     | Reproduction drift, missing feature ambiguity, or threshold dependence that cannot be frozen.               |
|        2 | **Run Protocol265 extension attribution by side, time, moneyness, premium, spread, IV, delta/gamma/theta, trend, and volatility regime.** | The current extension result is mixed: helpful in Recent 2026, harmful in March/Q1. Attribution should reveal whether extension has learnable structure. | Bad extensions cluster in identifiable regimes; good extensions cluster elsewhere.                                              | Extension deltas look random, unstable, or dominated by a few trades/days.                                  |
|        3 | **Build a no-order runtime parity harness for Protocol265.**                                                                              | It tests whether the lifecycle features, candidate stream, and anchor behavior exist live exactly as in replay.                                          | High agreement between historical-style and live no-order features; no unexplained missing candidates or timestamp drift.       | Live candidate/feature mismatches, stale quotes, inconsistent Greeks, or unavailable path features.         |
|        4 | **Construct a unified serial simulator with full live-feasible candidate surface.**                                                       | Current models are lineage-selected. A full candidate surface is required for a general neural trader.                                                   | DP oracle and baseline policies can be evaluated under identical constraints; rejected candidates become useful hard negatives. | Full-surface replay is too noisy, too incomplete, or destroys apparent edge after realistic filters.        |
|        5 | **Create DP oracle action-advantage labels for wait/enter/hold/exit.**                                                                    | This directly aligns labels with the real sequential objective and prices opportunity cost.                                                              | A supervised clone of oracle advantages beats Protocol101/261 on untouched serial replay without increasing churn.              | Oracle advantage disappears under slippage/fill stress, or learned clone fails outside training blocks.     |
|        6 | **Train a lifecycle model on `hold vs exit now` advantage, including next-slot opportunity value.**                                       | This is the cleanest path to letting winners run and cutting losers without arbitrary stops.                                                             | Fewer damaging extensions in March/Q1 while preserving Recent 2026 lift.                                                        | The model either exits too early again or extends losers despite opportunity-cost labels.                   |
|        7 | **Replace scalar entry scoring with candidate-set ranking plus abstention.**                                                              | Entry is a relative choice among contracts plus wait, not independent binary classification.                                                             | Better same-minute candidate selection, lower churn, improved calibration, and less source dependence.                          | Performance still depends mainly on upstream proposal source rather than learned candidate quality.         |
|        8 | **Add distributional heads for lifecycle risk and convex upside.**                                                                        | 0DTE outcomes are heavy-tailed; mean PnL alone hides tail behavior.                                                                                      | The policy avoids low-quality tail-risk trades while preserving rare convex winners.                                            | Quantile heads are unstable, uncalibrated, or provide no out-of-sample decision lift.                       |
|        9 | **Calibrate a stochastic fill model from paper/no-order logs and targeted high-resolution historical slices.**                            | Ask/bid replay is necessary but not sufficient; fill uncertainty may be the live edge killer.                                                            | Strategy survives calibrated fill replay and observed paper fills match simulator assumptions.                                  | Edge vanishes under realistic fill probability, quote-age, or cancel-timeout assumptions.                   |
|       10 | **Redesign protected validation with experiment ledger, embargoed holdout, block bootstrap, and PBO-style checks.**                       | Many protocols create meta-overfitting risk even if each protocol uses a holdout.                                                                        | Lift survives untouched periods and multiple-testing adjustments.                                                               | Performance is concentrated in repeatedly examined periods or fails once the evaluation protocol is frozen. |
|       11 | **Only then stage broader historical data acquisition.**                                                                                  | More data is valuable after the game, labels, and simulator are stable.                                                                                  | Added data improves a frozen protocol on later untouched data and live shadow.                                                  | Added data improves validation but not embargoed/live-shadow performance, or reveals simulator/label bias.  |

[1]: https://proceedings.mlr.press/v15/ross11a.html "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning"
[2]: https://arxiv.org/abs/2005.01643 "[2005.01643] Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems"
[3]: https://arxiv.org/abs/2006.04779 "[2006.04779] Conservative Q-Learning for Offline Reinforcement Learning"
[4]: https://arxiv.org/abs/2110.06169 "[2110.06169] Offline Reinforcement Learning with Implicit Q-Learning"
[5]: https://databento.com/docs/schemas-and-data-formats/whats-a-schema "What's a schema? | Databento schemas & data formats"
[6]: https://www.cboe.com/en/tradable-products/sp-500/spx-options/spx-specifications/ "S&P 500 Index Options Product Specifications | Cboe"
[7]: https://arxiv.org/abs/1904.12066 "[1904.12066] ABIDES: Towards High-Fidelity Market Simulation for AI Research"
[8]: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253 "The Probability of Backtest Overfitting by David H. Bailey, Jonathan Borwein, Marcos Lopez de Prado, Qiji Jim Zhu :: SSRN"
[9]: https://www.smallake.kr/wp-content/uploads/2016/03/optliq.pdf?utm_source=chatgpt.com "Optimal Execution of Portfolio Transactions∗"
[10]: https://arxiv.org/abs/2106.01345 "[2106.01345] Decision Transformer: Reinforcement Learning via Sequence Modeling"
[11]: https://arxiv.org/abs/1802.03042 "[1802.03042] Deep Hedging"
