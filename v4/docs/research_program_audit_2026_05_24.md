# Institutional Research Program Audit - SPXW 0DTE System

Audit date: 2026-05-24

Scope: current repository code, docs, audit artifacts, model reports, local paper logs, runtime scripts, replay infrastructure, and validation governance.

Bottom line: the project is much more rigorous than a typical retail trading system, but it has not yet proven an executable trading edge. The current system is best understood as a constrained sequential imitation-learning stack that distills a deterministic replay oracle over a learned candidate stream and frozen lifecycle exits. That is a defensible research bridge, not a final formulation. The largest remaining risk is not "model quality." It is whether the replay world, especially quote timing and fills, is close enough to the executable SPXW 0DTE market for the learned policy to mean anything.

## 0. Evidence Map

Most important local evidence:

- Current default: `docs/CURRENT_TRADING_BOT_SINGLE_SOURCE_OF_TRUTH.md:8-24` identifies `PAPER_DEFAULT_PROTOCOL101` as the current paper default and says current paper logs show zero broker calls/orders/fills.
- Current game definition: `v4/docs/MODEL_IMPROVEMENT_GUIDELINES.md:61-78` defines `$10,000`, SPXW PM 0DTE, long calls/puts/no trade, one open position, ask-entry/bid-exit, affordability, flat by close, causal inputs, and runtime-reproducible candidate generation.
- Data contract: `v4/docs/DATA_CONTRACT.md:18-35`, `v4/docs/DATA_CONTRACT.md:40-50`, and `v4/docs/DATA_CONTRACT.md:112-124` define raw/normalized/feature/label/audit layers, `decision_time`, and live-reproducibility.
- Historical dataset builder: `v4/dataset/spxw_0dte_neural.py:42-68`, `v4/dataset/spxw_0dte_neural.py:236-273`, and `v4/dataset/spxw_0dte_neural.py:346-383`.
- Databento normalization: `v4/ingest/databento_opra.py:178-211`, `v4/ingest/databento_opra.py:257-271`, `v4/ingest/databento_opra.py:338-357`, and `v4/ingest/databento_opra.py:455-550`.
- Protocol101 event policy: `v4/scripts/run_protocol097_sequential_event_policy.py:236-275`, `v4/scripts/run_protocol101_event_history_policy.py:68-76`, and `v4/scripts/run_protocol101_event_history_policy.py:209-232`.
- Protocol101 live inference: `v4/live/protocol101_entry.py:122-173`, `v4/live/protocol101_entry.py:271-311`, and `v4/live/protocol101_live_entry.py:214-274`.
- Lifecycle training/live mismatch: `v4/live/protocol066_inference.py:1-5`, `v4/scripts/build_lifecycle_sequence_dataset.py:60-118`, `v4/scripts/build_lifecycle_sequence_dataset.py:352-482`, and `v4/scripts/run_protocol158_protocol101_live_entry_paper_bridge.py:871-922`.
- Execution guard/executor: `v4/live/ibkr_paper_guard.py:18-29`, `v4/live/ibkr_paper_guard.py:81-143`, `v4/live/ibkr_paper_executor.py:29-129`, and `v4/live/ibkr_paper_executor.py:195-256`.
- Simulator state: `v4/sim/simulator.py:1-16`, `v4/sim/simulator.py:66-85`, and `v4/sim/simulator.py:114-145`.
- Replay accounting: `v4/sim/paper_replay.py:1-8`, `v4/sim/paper_replay.py:111-180`, and `v4/sim/paper_replay.py:188-195`.
- Protocol101 report: `v4/audit/autoresearch/v4_aplus_hypothesis_101_event_history_policy/report.md:1-15`.
- Overfit/holdout governance: `v4/audit/autoresearch/v4_aplus_hypothesis_273_model_selection_overfit_risk/report.md:1-26`, `v4/audit/autoresearch/untouched_holdout_availability/report.md:1-20`, and `v4/audit/autoresearch/formal_validation_governance/report.md:1-33`.
- Strategy forensics: `v4/audit/autoresearch/protocol101_strategy_forensics_packet_v1/report.md:11-21`, `v4/audit/autoresearch/protocol101_strategy_forensics_packet_v1/report.md:75-95`, and `v4/audit/autoresearch/protocol101_internal_slot_cost_counterfactual_v1/report.md:11-25`.

External anchors used:

- Ross, Gordon, and Bagnell, DAgger / imitation learning covariate shift: https://proceedings.mlr.press/v15/ross11a.html
- Kumar et al., Conservative Q-Learning / offline RL distribution shift: https://proceedings.neurips.cc/paper/2020/hash/0d2b2061826a5df3221116a5085a6052-Abstract.html
- Bailey, Borwein, Lopez de Prado, Zhu, probability of backtest overfitting: https://carmamaths.org/jon/backtest2.pdf
- White, "A Reality Check for Data Snooping": https://econpapers.repec.org/RePEc:ecm:emetrp:v:68:y:2000:i:5:p:1097-1126
- Hansen, Superior Predictive Ability test: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=264569
- Cboe SPX/SPXW product specifications and hours: https://www.cboe.com/en/tradable-products/sp-500/spx-options/spx-specifications/ and https://www.cboe.com/about/hours/us-options
- Cboe SPXW Weeklys overview: https://www.cboe.com/tradable_products/sp_500/spx_weekly_options/specifications/
- IBKR option Greeks/subscription requirement: https://interactivebrokers.github.io/tws-api/option_computations.html and https://www.interactivebrokers.com/campus/ibkr-api-page/market-data-subscriptions/
- Almgren-Chriss execution-cost framing: https://docslib.org/doc/1384720/optimal-execution-of-portfolio-transactions
- Avellaneda-Stoikov and Cont-Stoikov-Talreja limit-order-book execution/microstructure framing: https://www.tandfonline.com/doi/abs/10.1080/14697680701381228 and https://pubsonline.informs.org/doi/10.1287/opre.1090.0780

## 1. Is This The Correct Formulation?

### Verdict

The correct problem is a partially observed, constrained sequential decision problem with stochastic execution, not a static supervised classification problem. The current `wait vs candidate` Protocol101 formulation is a reasonable intermediate imitation-learning formulation, but it is incomplete and can overfit the replay oracle. The final production formulation should model at least:

- no-entry / enter candidate,
- hold / exit / forced-flat,
- one-position slot opportunity cost,
- affordability and account state,
- execution uncertainty,
- quote freshness and latency,
- partial observability of the option book and Greeks.

Protocol101 is not "just a classifier." It is a candidate-set policy with an explicit wait action, trained from a backward dynamic-programming oracle that maximizes frozen candidate PnL under a one-open-position constraint (`v4/scripts/run_protocol097_sequential_event_policy.py:236-275`). Protocol101 then adds short causal history (`v4/scripts/run_protocol101_event_history_policy.py:33-53`, `v4/scripts/run_protocol101_event_history_policy.py:209-232`) and selects entries using a wait-logit margin (`v4/live/protocol101_entry.py:271-311`).

That is closer to the economics than independent entry classification. But it still learns an oracle over deterministic fills, deterministic exits, and an upstream candidate stream. In imitation-learning terms, the expert is not a real trader and not the market; the expert is the simulator. Ross et al. warn that imitation learning in sequential settings is distribution-sensitive because actions change the future observation distribution. Here the action distribution changes which later candidates are reachable because one open position blocks the account.

### Formulation alternatives

Supervised classification: inadequate as the top-level formulation. It can train components such as "candidate likely positive after current lifecycle," but it cannot price opportunity cost, holding-state risk, or action-dependent future availability. The repo correctly moved beyond pure classification.

Contextual bandit: inadequate for headline trading. A contextual bandit assumes each decision's reward is independent after action. Protocol101's one-slot constraint (`v4/model/serial_opportunity.py:623-706`) means entering now censors future entries until exit. The internal slot-cost audit found 3,723 blocked model-approved entries and `$399,510` best-blocked-minus-open counterfactual value (`v4/audit/autoresearch/protocol101_internal_slot_cost_counterfactual_v1/report.md:17-25`). That is not bandit structure.

Optimal stopping: necessary but not sufficient. Lifecycle is an optimal-stopping-like problem after entry. The lifecycle data builds causal path-so-far features and future path labels (`v4/scripts/build_lifecycle_sequence_dataset.py:60-118`, `v4/scripts/build_lifecycle_sequence_dataset.py:352-482`). But entry and exit are coupled through slot cost and execution.

Imitation learning: this is the current closest description. Protocol101 imitates a DP oracle whose reward is frozen candidate PnL plus future value after exit (`v4/scripts/run_protocol097_sequential_event_policy.py:247-261`). This is appropriate only if the oracle's reward, fills, and candidate set are executable.

Offline RL: conceptually relevant later, premature now. Offline RL is dangerous when the learned policy takes actions outside the support of logged behavior. CQL and related methods exist because value overestimation under distribution shift is a core failure mode. This repo lacks the logged behavior coverage and calibrated execution model needed for credible offline RL. Running CQL on deterministic replay candidates would mostly learn simulator artifacts.

Market making: not the right top-level formulation. The system is a directional long-option liquidity taker, not a two-sided inventory market maker. But microstructure models from market making and limit-order-book literature are still relevant to fill probability, queue priority, quote staleness, and adverse selection.

Trader-rule distillation: a fair current description. Protocol051 creates an edge-ranked surface; Protocol101 gates and orders entries; Protocol066/081 manage exits. The stack is distilling a set of research heuristics and replay oracles into neural modules. That is acceptable if treated as hypothesis generation, not as proof.

### The one-position constraint changes everything

The one-position constraint is not a minor risk knob. It transforms the problem from ranking independently good entries into allocating a scarce intraday option on attention/capital. The strict serial replay and paper guard correctly recognize this (`v4/model/serial_opportunity.py:623-706`, `v4/live/ibkr_paper_guard.py:107-108`). The unresolved question is whether the model has enough state to decide when to spend the slot and when to preserve it. Current Protocol101 history features are shallow aggregate summaries of prior candidate sets, not a belief state over current regime or latent opportunity arrival intensity.

### Partial observability

The project is treating parts of a partially observable environment as if they were fully observable minute rows. Live IBKR data is per-contract, sparse, and subscription-limited. Historical `cbbo-1m` is not full tick/queue state. Greeks can be live only when option and underlying subscriptions are present according to IBKR documentation. The correct mental model is a POMDP with noisy observations and censored execution, not a fully observed MDP.

## 2. Is The Data Correct?

### Strongest data decisions

The data contract is unusually strong. It explicitly separates raw, normalized, feature, label, and audit layers (`v4/docs/DATA_CONTRACT.md:18-35`), makes `decision_time` central (`v4/docs/DATA_CONTRACT.md:40-50`), and treats `is_live_reproducible` as a leak-prevention contract (`v4/docs/DATA_CONTRACT.md:112-124`). The repo also encodes SPXW PM settlement at ingestion (`v4/ingest/databento_opra.py:178-211`), uses ask-entry/bid-exit labels rather than mid fills (`v4/dataset/spxw_0dte_neural.py:346-383`), and applies historical tradability filters for premium, spread, quote age, and sizes (`v4/dataset/spxw_0dte_neural.py:42-68`, `v4/dataset/spxw_0dte_neural.py:236-273`).

Those choices put the project above most option-strategy research code.

### Weakest data assumptions

1. Historical quote age is not the same object as live quote age. Databento normalization sets `quote_age_ms` to zero when a quote timestamp exists (`v4/ingest/databento_opra.py:491-542`), while the dataset builder later computes age as `decision_time - quote_time` and allows up to 90 seconds (`v4/dataset/spxw_0dte_neural.py:236-249`). Live runtime now computes quote freshness from ticker timestamps where available (`v4/scripts/run_protocol158_protocol101_live_entry_paper_bridge.py:600-635`), but missing timestamps become `None` and must be blocked by the guard (`v4/live/ibkr_paper_guard.py:121-124`). The semantics are improving, but historical, normalized, and live quote age are still not one unified concept.

2. The live candidate filter is materially looser than the historical tradability filter. Historical candidates require mid bounds, max absolute spread, max spread fraction, and min sizes (`v4/dataset/spxw_0dte_neural.py:252-273`). Live `_valid_quote` only requires SPXW/right/finite strike and positive non-crossed bid/ask (`v4/live/protocol101_live_entry.py:420-428`). The selected order is later guarded for quote age and some price movement, but Protocol051/101 can score a live candidate surface with candidates that would never have existed in training.

3. Live market-window features are not identical to historical context features. `LiveIndexState.market_window` forward/back-fills sparse SPX/VIX rows (`v4/live/protocol101_live_entry.py:69-83`), and `_market_features_from_live` computes VWAP as a simple mean of observed SPX closes (`v4/live/protocol101_live_entry.py:404-417`). Historical context is built from one-minute bars and nominal market features (`v4/dataset/spxw_0dte_neural.py:89-96`). That is replay/live feature semantic drift.

4. Labels are carried in the same row artifact as features. The builder outputs `option_ladder`, `candidate_mask`, and future labels together (`v4/dataset/spxw_0dte_neural.py:508-524`). Protocol051 training consumes `labels_net_pnl` and masks tokens by finite labels (`v4/model/hypothesis_protocol.py:1306-1318`, `v4/model/hypothesis_protocol.py:1418-1427`). Live rows fill labels with zeros (`v4/live/protocol101_live_entry.py:257-270`). This is not direct leakage into live inference, but it creates a training/live action-mask mismatch risk if finite-label availability is correlated with liquidity or future data coverage.

5. Lifecycle labels are necessarily future-path labels. That is acceptable for supervised lifecycle training, but it raises the bar for leakage controls. Future max/min/final PnL and recovery/decay labels are computed from `future = pnls[idx:]` (`v4/scripts/build_lifecycle_sequence_dataset.py:372-480`). The model input columns are mostly causal path-so-far features (`v4/scripts/build_lifecycle_sequence_dataset.py:60-100`), but live must reproduce the same sequence state.

6. Live lifecycle currently violates the inference contract. The inference module says live callers must feed the full causal path since entry, not a single isolated row (`v4/live/protocol066_inference.py:1-5`). The paper bridge builds one row for the current position and calls `predict_protocol066_sequence` on that single-row frame (`v4/scripts/run_protocol158_protocol101_live_entry_paper_bridge.py:737-756`, `v4/scripts/run_protocol158_protocol101_live_entry_paper_bridge.py:871-922`). That is one of the most important replay/live parity gaps.

7. Protocol051 has a stale v4 purity contradiction. `v4/README.md:1-4` and `v4/README.md:47-53` claim no v2/v3 imports, but `v4/model/hypothesis_protocol.py:33` imports `v2.core.market_structure`. That matters because hidden legacy context can carry stale assumptions even if it is not a direct label leak.

8. Fees are zero in the main label config. `NeuralDatasetConfig.fee_per_contract` is `0.00` (`v4/dataset/spxw_0dte_neural.py:58-60`). For one-contract SPXW this may be small relative to some winners, but it is not zero, and it matters for narrow edge, churn, and small premium candidates.

### Protocol051: edge or quote-mechanics artifact?

Protocol051 may be learning true opportunity quality, but it may also be learning quote mechanics and liquidity proxies. The evidence:

- Its targets are future ask-to-bid PnL clipped and scaled (`v4/model/hypothesis_protocol.py:1418-1427`).
- Protocol101 uses Protocol051's flat-vs-token score difference as `edge` and filters to `edge >= 25` (`v4/live/protocol101_entry.py:122-173`).
- The Protocol101 forensics show the profitable archetype is narrow: high-premium, ITM, post-open, often sequence residual exits (`v4/audit/autoresearch/protocol101_strategy_forensics_packet_v1/report.md:26-41`).

That can be real. It can also be a stable artifact of how one-minute CBBO, spreads, and labels interact. The falsification test is not "another model." It is matched controls by spread, premium, quote age, side, time, and moneyness, with tick/1-second or paper-observed fill stress.

### Assumption ranking

| Assumption | Importance | Fragility | Falsification risk | Audit view |
|---|---:|---:|---:|---|
| Historical ask-entry/bid-exit labels approximate executable fills | Critical | High | High | Needs paper/live fill table and tick/latency replay. |
| `cbbo-1m` timestamp semantics preserve decision causality | Critical | Medium-high | High | Needs ts_event/ts_recv audit and high-res cross-check. |
| Live candidate filtering matches historical candidate filtering | Critical | High | High | Current code says no. Fix or prove harmless. |
| Lifecycle live row equals replay sequence state | Critical | High | High | Current code says no. Build parity harness. |
| Protocol051 score is opportunity quality, not liquidity artifact | High | Medium-high | High | Needs matched controls and cross-vendor/high-res replay. |
| Protocol101 short-history state is sufficient | High | Medium | Medium-high | Needs ablation, slot-cost, and regime-state diagnostics. |
| Labels do not contaminate feature/mask semantics | High | Medium | Medium | Needs feature/label separation tests on actual artifacts. |
| IBKR live Greeks and historical Greeks are semantically comparable | Medium-high | Medium-high | Medium-high | IBKR requires option+underlying subscriptions; compare distributions. |
| $10k account and reserve semantics match replay | Medium-high | Medium | Medium | Guard does not subtract the documented `$500` reserve. |
| Fees/commissions negligible | Medium | Medium | Medium | Add fee sensitivity. |

## 3. Is The Validation Methodology Correct?

### Verdict

The validation methodology is credible as research governance, not as institutional proof of deployable edge. The repo deserves credit for explicitly recognizing this. It has chronological folds, seed medians, stress scenarios, strict serial baselines, and formal holdout governance. But current Protocol101 profitability cannot be treated as final out-of-sample evidence because the major splits have been repeatedly mined, the simulator has not been calibrated to live fills, and no protected untouched block is available.

### What is good

Protocol101 reports split-level medians, profit factor, trade counts, slippage stress, and strict serial baselines (`v4/audit/autoresearch/v4_aplus_hypothesis_101_event_history_policy/report.md:8-15`). It also clearly says no paid data, live broker data, or order endpoint was used (`v4/audit/autoresearch/v4_aplus_hypothesis_101_event_history_policy/report.md:1-3`).

The model-improvement rulebook explicitly bans treating overlapping candidate PnL as headline equity and requires the live-like serial game (`v4/docs/MODEL_IMPROVEMENT_GUIDELINES.md:61-78`). It also requires runtime parity views (`v4/docs/MODEL_IMPROVEMENT_GUIDELINES.md:216-227`).

The overfit-risk audit is unusually honest: q3_2025, q4_2025, q1_2026, march_2026, and recent_2026 are classified as repeated-research tests, not sacred holdouts (`v4/audit/autoresearch/v4_aplus_hypothesis_273_model_selection_overfit_risk/report.md:10-26`). The untouched holdout is explicitly pending future collection (`v4/audit/autoresearch/untouched_holdout_availability/report.md:10-20`).

### What blocks institutional credibility

Repeated validation reuse: the audit reports 135 summaries / 118 model protocols on q3_2025, 143 / 126 on q4_2025, 152 / 134 on q1_2026, and 112 / 101 on march_2026 (`v4/audit/autoresearch/v4_aplus_hypothesis_273_model_selection_overfit_risk/report.md:12-21`). That is exactly the multiple-hypothesis environment addressed by White's Reality Check, Hansen's SPA test, and Bailey/Lopez de Prado PBO.

Threshold selection is validation-PnL driven. Protocol092 selects thresholds using validation stress total PnL, PF, base PnL, and trades (`v4/model/serial_opportunity.py:544-620`). Protocol097/101 select a margin threshold by validation stress delta versus strict serial baseline (`v4/scripts/run_protocol097_sequential_event_policy.py:389-430`). This is not wrong, but it makes protected test separation non-negotiable.

Protocol101 margins over baseline are not uniformly large. The official report shows q4_2025 beats strict serial by only `$140` and q3_2025 by `$1,340` in the promotion checks (`v4/audit/autoresearch/v4_aplus_hypothesis_101_event_history_policy/report.md:17-32`). Against Protocol097, Protocol101 is slightly worse on q3, q1, and march (`v4/audit/autoresearch/v4_aplus_hypothesis_101_event_history_policy/report.md:10-15`). That does not invalidate it, but it means the claimed improvement is narrow relative to the upstream candidate stream.

Model training history shows classic overfit shape in at least the active fold3 seed1 artifact: best validation cross-entropy at epoch 1, then train loss improves while validation worsens through epoch 16. The artifact does keep the best epoch, but the pattern supports strict freeze discipline.

The formal validation governance packet is a control, not a proof. It has 21 comparable strategy/slippage variants over 4 exposed splits and a CSCV proxy PBO of 0.0 (`v4/audit/autoresearch/formal_validation_governance/report.md:10-18`), but the same report says the splits are research-exposed diagnostics and the reserved untouched block should be scored once only after fill/parity/promotion gates pass (`v4/audit/autoresearch/formal_validation_governance/report.md:19-33`).

### Metrics

PnL and PF are necessary but not sufficient. For 0DTE SPXW, the validation metric needs to be fill-adjusted utility under:

- missed fills and cancels,
- latency distribution,
- stale quote blocks,
- fees,
- daily loss and drawdown constraints,
- trade concentration,
- regime/bucket robustness,
- affordability and reserve treatment,
- realized broker execution logs.

Without those, PF/DD can be a simulator-fitted statistic.

## 4. Is The Execution Model Realistic?

### Verdict

Not yet. This is the main blocker. Ask-entry/bid-exit replay is more realistic than mid-price replay, but it is not evidence of executable fills. The repo has a good guard and a thin paper executor, but it has no calibrated fill model and recent inspected logs have zero submitted orders/fills.

### Why ask-entry/bid-exit is still optimistic

Buying at the displayed ask and selling at the displayed bid assumes the displayed NBBO is current, available to you, and fillable at the size and timing of the decision. In SPXW 0DTE, the option value can move meaningfully between quote observation, model inference, order submission, broker routing, and order acknowledgment. A limit buy at the ask can miss when the ask lifts; a sell at the bid can miss when the bid fades; partial fills and cancels matter even for one lot if the market is moving and the quote is stale.

The timing-fragility evidence is already severe. Protocol126 shows large PnL deterioration under delays: for q3_2025, 15 seconds reduces PnL by `$139,700`, 30 seconds by `$214,090`, and 60 seconds turns it negative; similar degradation appears in q4_2025 and q1_2026 (`v4/audit/autoresearch/v4_aplus_hypothesis_126_protocol101_timing_fragility_hardening/report.md:9-36`). That is not a small execution nuisance. It is core to the edge.

### Current execution stack

Good:

- Paper guard requires paper flags, DU account prefix, one contract, one open position, bid/ask validity, quote age <= 1500 ms, context age <= 5000 ms, and affordability (`v4/live/ibkr_paper_guard.py:18-29`, `v4/live/ibkr_paper_guard.py:81-143`).
- Paper executor calls `placeOrder` only after permission and validation pass (`v4/live/ibkr_paper_executor.py:29-129`).
- Executor waits for order resolution, records status/fill count/avg fill price, and can cancel unfilled orders (`v4/live/ibkr_paper_executor.py:195-256`).
- Live bridge logs option quote timestamps/ages where available (`v4/scripts/run_protocol158_protocol101_live_entry_paper_bridge.py:600-681`).

Not good enough:

- `v4/sim/simulator.py` is still an interface and `NullSimulator`; no calibrated fill model exists (`v4/sim/simulator.py:1-16`, `v4/sim/simulator.py:114-145`).
- `v4/sim/paper_replay.py` explicitly says it is not a live broker simulator (`v4/sim/paper_replay.py:1-8`), and its replay frame deterministically uses entry ask and exit bid (`v4/sim/paper_replay.py:111-180`).
- Recent inspected paper logs through 2026-05-21 had zero broker endpoint calls, zero paper orders submitted, and zero fills. `docs/CURRENT_TRADING_BOT_SINGLE_SOURCE_OF_TRUTH.md:21-24` says the latest inspected Protocol101 paper log had 2,737 rows but no broker order endpoint rows.
- The account guard defines a `$500` IBKR access reserve but affordability checks only `premium_required > account_cash` and does not subtract reserve (`v4/live/ibkr_paper_guard.py:18-29`, `v4/live/ibkr_paper_guard.py:133-141`).

### Market-structure implications

Cboe official specs confirm SPXW is the correct PM-settled weekly/daily product and expiring SPXW options ordinarily cease trading at 4:00 p.m. ET / 3:00 p.m. CT. A 15:55 ET forced-flat buffer is directionally sensible. But the final hour has high gamma, fast delta changes, and quote-fade risk. Post-open has fast repricing and spread normalization. These are exactly the windows Protocol101 is allowed to trade (`v4/live/protocol101_entry.py:30`, `v4/live/protocol101_entry.py:138-139`, `v4/live/protocol101_entry.py:351-354`).

IBKR documentation says live option Greeks require market data subscriptions for both option and underlying. This matters because live Greeks are not merely feature values; they are entitlement-, subscription-, and timestamp-dependent observations. Historical Black-Scholes Greeks and IBKR model Greeks are not automatically identical features.

The execution literature's core lesson applies even at one lot: execution cost is a stochastic control problem, not a constant slippage haircut. Almgren-Chriss is about the risk/cost tradeoff in execution; Avellaneda-Stoikov and order-book models emphasize order arrival, quote placement, and fill probability. This project does not need a full market-making engine, but it does need a calibrated taker/limit-fill model.

## 5. Is The Model Architecture Appropriate?

### Protocol051

Protocol051/A+ is a reasonable surface scorer. It produces flat plus token scores, uses candidate masks, and supplies a compact "edge" input to Protocol101. The danger is that it is upstream of everything: Protocol101 can only choose candidates Protocol051 and the candidate gates expose. If Protocol051 learns liquidity artifacts or misses a second playbook, Protocol101 cannot recover.

The v2 dependency contradiction (`v4/model/hypothesis_protocol.py:33` versus `v4/README.md:1-4`, `v4/README.md:47-53`) should be treated as a governance defect, not necessarily a performance defect.

### Protocol101

Protocol101 is the most appropriate current entry architecture in the repo. A permutation-aware candidate-set model with a wait logit (`v4/scripts/run_protocol097_sequential_event_policy.py:49-82`) matches the "multiple candidates at a decision time" structure. The short history features are causal (`v4/scripts/run_protocol101_event_history_policy.py:209-232`) and operationally reproducible through `Protocol101HistoryState` (`v4/live/protocol101_entry.py:46-86`).

Weaknesses:

- The oracle is deterministic replay PnL, not executed return (`v4/scripts/run_protocol097_sequential_event_policy.py:236-275`).
- Candidate order/action space may differ between training and live: `build_events` takes the first 10 sorted by seed/contract/candidate UID (`v4/scripts/run_protocol097_sequential_event_policy.py:218-233`), while live Protocol101 sorts by surface edge and contract id (`v4/live/protocol101_entry.py:171-172`). This may be benign if the offline candidate dataset was already pre-ranked, but it should be proven.
- Score calibration is weak. Strategy forensics report selected-trade Spearman score-margin vs PnL of `-0.0571` and vs MFE of `-0.1595` (`v4/audit/autoresearch/protocol101_strategy_forensics_packet_v1/report.md:75-80`).
- The live `entry_minutes_to_forced_flat` feature actually counts to 15:30 no-new-entry, not the 15:55 forced-flat deadline (`v4/live/protocol101_entry.py:321-347`). That may match training but the name is semantically misleading.

### Lifecycle systems

The lifecycle idea is right: exits are sequence problems. But the current live implementation is not yet trustworthy because the GRU was trained on sequences (`v4/scripts/run_protocol061_sequence_lifecycle_model.py:107-126`) and live currently supplies one-row frames. This is not a minor code hygiene issue; a recurrent model's hidden state is part of its semantics.

The lifecycle target is also complex: the value target is future upside minus a downside penalty (`v4/scripts/run_protocol061_sequence_lifecycle_model.py:191-198`), while live action exits when `predicted_continuation_value > threshold` after mandatory stop/target/time-flat checks (`v4/live/protocol066_inference.py:126-149`). This may be a residual-exit design inherited from Protocol066/081, but it needs a plain-English invariant: "higher value means exit now" or "higher value means continue" cannot remain ambiguous.

### What architecture should come next?

Do not jump directly to transformers or offline RL. The next architecture should be determined by falsification gates:

1. Unified decision-state/action-advantage labels after parity and fill gates. The docs already point toward `UnifiedDecisionStateV1`, `ExecutionModelV1`, and `ActionAdvantageLabelV1` (`v4/docs/MODEL_IMPROVEMENT_GUIDELINES.md:18-26`). That is the right direction because it forces entry, hold, exit, defer, affordability, and execution to share one state/action contract.

2. Hierarchical policy only if playbooks are empirically distinct. If call/put, post-open/late-day, scalp/runner archetypes have different causal signatures, use a hierarchy: playbook classifier/gate, then specialized action policy. If not, hierarchy just adds selection degrees of freedom.

3. Sequence or transformer models only after high-resolution causal sequences exist. The current short history is intentionally simple. A transformer could model regime and candidate-arrival context, but without high-res validated data it will overfit exposed replay windows.

4. Offline RL only after behavior coverage, execution model, and OPE are credible. Offline RL needs logged actions, missed actions, rewards after execution, and coverage of the learned action space. Otherwise conservative algorithms will either collapse to the behavior policy or overestimate out-of-support trades.

5. Hybrid rule-neural remains appropriate. Hard safety rules for instrument, time, quote age, affordability, forced flat, and max position should remain non-neural. Neural modules should propose utility-ranked actions inside that guardrail.

## 6. What Did The Project Do Right?

The project did many important things right:

- It treats the audit trail as first-class. The v4 README says the model is subordinate to simulator, data, and falsification (`v4/README.md:7-13`).
- It uses ask-entry/bid-exit labels instead of midpoint fantasy (`v4/dataset/spxw_0dte_neural.py:346-383`).
- It encodes PM-settled SPXW 0DTE filtering in ingestion and dataset construction (`v4/ingest/databento_opra.py:178-211`, `v4/dataset/spxw_0dte_neural.py:386-410`).
- It has a strong data contract with explicit causality and live-reproducibility fields (`v4/docs/DATA_CONTRACT.md:18-35`, `v4/docs/DATA_CONTRACT.md:112-124`).
- It moved from independent candidate scoring to explicit wait-vs-candidate sequential event policy (`v4/scripts/run_protocol097_sequential_event_policy.py:1-82`).
- It enforces one-position serial replay and has a strict baseline (`v4/model/serial_opportunity.py:623-706`, `v4/model/serial_opportunity.py:768-780`).
- It has a broker guard that is meaningfully conservative for paper orders (`v4/live/ibkr_paper_guard.py:18-29`, `v4/live/ibkr_paper_guard.py:81-143`).
- It refuses to call research challengers paper defaults without decision packets (`v4/docs/MODEL_IMPROVEMENT_GUIDELINES.md:9-17`).
- It explicitly recognizes model-selection overfit and reserves a future untouched block (`v4/audit/autoresearch/v4_aplus_hypothesis_273_model_selection_overfit_risk/report.md:1-26`, `v4/audit/autoresearch/untouched_holdout_availability/report.md:1-20`).
- It has strategy forensics that ask trader-level questions instead of only optimizing model metrics (`v4/audit/autoresearch/protocol101_strategy_forensics_packet_v1/report.md:82-95`).

This is not a toy system. The danger is that the rigor of the research scaffolding can create false comfort if execution realism remains unproven.

## 7. Most Dangerous Assumptions

| Rank | Assumption | Why dangerous | How to falsify | Confidence increases if | Confidence collapses if |
|---:|---|---|---|---|---|
| 1 | Ask-entry/bid-exit replay is executable | Main PnL can be quote timing, not edge | Paper-submit/live fill table by spread, premium, side, quote age, time | Filled trades retain bucketed replay expectancy after misses/cancels | Selected asks rarely fill or filled trades underperform replay |
| 2 | Exposed splits still predict future | 100+ protocols/summaries touched major splits | Freeze candidate, score one new untouched block once | Untouched block matches preregistered bucket behavior | Untouched performance vanishes or reverses |
| 3 | One-minute CBBO is enough for 0DTE execution | Intraminute path and quote fades dominate 0DTE | Tick/1s replay around selected times | Delay/fill-adjusted PnL robust to observed latency | 1-15 second delays erase edge |
| 4 | Live features equal training features | Live context, candidate filtering, Greeks, quote age differ | Feature parity harness over no-order logs | Scaled live features match historical reconstruction | Live candidate distribution shifts out of training support |
| 5 | Lifecycle live behavior matches replay | GRU trained on sequence, live sends one row | Replay a known trade and compare live-row actions per step | Actions and scaled features match step-by-step | Live exits differ materially or hidden state matters |
| 6 | Protocol051 edge is causal alpha | It may rank liquidity/label artifacts | Matched controls and cross-vendor/high-res validation | Edge remains monotonic after cost/fill controls | Edge decays after spread/quote-age matching |
| 7 | Protocol101 margin is calibrated utility | Forensics show weak/negative selected-trade correlation | Candidate-level margin bins including rejected candidates | Margin monotonic within side/time/premium buckets | Higher margin does not improve realized utility |
| 8 | One-slot opportunity cost is handled | Current model can enter early and block better later signals | Slot-cost replay with mutually exclusive switch/defer actions | Weak open states predict blocked winners causally | Blocked winners are hindsight-only or unfillable |
| 9 | `$10k` account semantics are consistent | Reserve not subtracted; high premium matters | Guard-identical account replay | Most winners affordable under exact guard/reserve | PnL depends on trades guard would block |
| 10 | IBKR paper evidence will transfer live | Paper data/fills can differ from live | Compare paper fills to live small-size observations only after approval | Paper and live fill/cancel distributions align | Paper fills are systematically optimistic or delayed |
| 11 | Fees are negligible | Narrow trades and churn can lose edge | Add commission/fee sensitivity | Results stable after realistic fees | Edge concentrated in low expectancy trades |
| 12 | Current hard stops are unavoidable | If avoidable, model is wasting tail risk | Matched hard-stop autopsy using causal pre-entry fields | Hard stops indistinguishable pre-entry | Hard stops share clear pre-entry avoidable signatures |

## 8. Research Roadmap From Here

Prioritization principle: maximize falsification power and reduce replay-only edge risk before adding model capacity.

### P0 - Freeze and classify evidence

Why it matters: Without freeze discipline, every diagnostic becomes another researcher degree of freedom.

Do:

- Freeze Protocol101 artifact, threshold, candidate generator, lifecycle artifact, and current paper guard as the control.
- Mark q3/q4/q1/march/recent as exposed diagnostics only.
- Keep no new model training until execution, parity, and untouched-holdout gates pass.

Validates if: future reports can state exactly which artifact and rules were frozen before evidence collection.

Falsifies if: thresholds, filters, feature sets, or artifacts keep changing while diagnostics are being interpreted.

### P1 - Execution realism packet

Why it matters: This is the fastest way to falsify the whole project.

Do:

- Collect no-order and human-approved one-contract paper-submit observations.
- Log all selected and rejected candidate quote timestamps, received timestamps, decision timestamps, submission timestamps, broker ack timestamps, fill/cancel status, fill price, and order age.
- Stratify by side, premium, spread, moneyness, time bucket, quote age, candidate edge, and latency.
- Build a non-parametric fill/miss/cancel table before fitting any model.

Evidence that validates: the replay-dominant archetypes fill with high probability and retain positive post-fill PnL after realistic miss/cancel handling.

Evidence that falsifies: profitability concentrates in stale, fast-moving, wide-spread, or low-fill buckets.

Prerequisites: paper order approval and strict logging; no real-money trading needed.

Operational risk: paper orders can lose simulated money and can create false comfort if paper/live differ.

Information gain: extreme.

### P2 - Replay/live parity harness

Why it matters: A model cannot be validated if its live features are not the training features.

Do:

- Build a feature-by-feature diff harness for `build_live_surface_row` versus historical row construction.
- Make live candidate filtering identical to historical `_candidate_is_tradable`, or explicitly version the difference.
- Compare Protocol051 scores, Protocol101 candidate frames, scaled feature vectors, logits, margins, and selected action for replayed no-order live snapshots.
- Build a lifecycle parity harness that feeds full historical paths and live-style row construction at every timestamp.

Validates if: candidates, scaled features, logits, lifecycle actions, and guard decisions match within tolerance.

Falsifies if: live rows produce different masks, edge distributions, context features, or lifecycle actions.

Information gain: extreme.

### P3 - Data-quality and timestamp audit

Why it matters: 0DTE edge can be a timestamp artifact.

Do:

- Audit `ts_event`, `ts_recv`, `quote_time`, `receive_time`, `decision_time`, and `quote_age_ms` across normalized, feature, and live layers.
- Cross-check a sample of selected Protocol101 trades with higher-resolution or tick-level data.
- Rebuild quote-age and delay-stress metrics using the same timestamp definition live and historical.

Validates if: selected trades remain profitable under true actionable timestamps.

Falsifies if: labels used quotes not actually available at decision time.

Information gain: high.

### P4 - Guard-identical account replay

Why it matters: The paper account is part of the trading problem.

Do:

- Replay Protocol101 through the exact `validate_order_intent` affordability, reserve, max-position, quote-age, and ask-move rules.
- Decide whether the `$500` reserve is trading capital or unavailable capital, then encode it consistently.
- Include fees and commission sensitivity.

Validates if: headline Protocol101 trades are mostly unchanged under exact guard semantics.

Falsifies if: profitable trades are unaffordable, stale, or reserve-dependent.

Information gain: high.

### P5 - Strategy archetype and hard-stop autopsy

Why it matters: Architecture should follow the actual playbook.

Do:

- Expand the existing trade atlas into stable archetypes: call/put, post-open/late-day, premium, ITM/ATM/OTM, spread, VIX/SPX context, exit reason, duration, MFE/MAE.
- Compare hard-stop losers against matched winners using only entry-time fields.
- Separate "unavoidable loss" from "avoidable pre-entry regime" and "lifecycle management failure."

Validates if: a small number of causal, fillable archetypes explain most robust PnL.

Falsifies if: PnL is diffuse, period-specific, or dominated by one-off days.

Information gain: high.

### P6 - Slot-cost and lifecycle research

Why it matters: Entry quality may not be the bottleneck.

Do:

- Convert the internal slot-cost audit into a mutually exclusive replay with defer/switch/exit choices.
- Attach current open-position causal state to every blocked candidate.
- Test runner/giveback alternatives using next-step hold distributions, not future best highs.
- Validate live lifecycle sequence parity before changing exit models.

Validates if: blocked winners or runner opportunities are identifiable from causal state.

Falsifies if: gains require hindsight or unfillable alternatives.

Information gain: medium-high.

### P7 - Validation hardening

Why it matters: The project has many protocols and exposed splits.

Do:

- Pre-register the next frozen candidate and baseline before scoring any new data.
- Use a single protected future block collected after 2026-05-24; score once.
- Report White/SPA-style family selection checks and CSCV/PBO on comparable strategy families, but do not use them to tune.
- Use daily/block bootstrap and concentration metrics.

Validates if: the candidate survives the new untouched block with preregistered metrics and archetype behavior.

Falsifies if: exposed split performance does not transfer.

Information gain: high, but only after P1-P4.

### P8 - Long-term architecture

Why it matters: Once truth gates are closed, model capacity can target the right bottleneck.

Do:

- Implement unified state/action/action-advantage data contracts.
- Add stochastic execution model outputs to labels and replay.
- Consider hierarchical playbook policies only if archetypes prove distinct.
- Consider sequence/transformer models only after high-res causal trajectories are available.
- Consider conservative offline RL only after logged behavior/action coverage, execution rewards, and OPE are credible.
- Add uncertainty/OOD gates for live feature drift.

Validates if: new architecture improves fill-adjusted utility on protected data and explains which bottleneck it solves.

Falsifies if: improvements appear only on exposed replay windows or broaden action space without live-feasible evidence.

Information gain: medium now, high later.

## Final Assessment

The project is approaching the right class of problem, but it is not yet solving the full problem. Its current formulation is an intelligent research compromise: Protocol051 narrows the option surface, Protocol101 learns wait-vs-enter under one-slot serial constraints, and Protocol066/081 supply lifecycle behavior. That is substantially better than naive supervised entry classification.

The critical unresolved question is whether the simulator's executable world is real. Until fill probability, quote age, latency, candidate parity, lifecycle parity, and untouched validation are closed, replay profitability should be treated as a promising hypothesis, not tradable edge.

The most productive next move is not a larger network. It is an execution-and-parity falsification campaign.
