# Trinity Audit Report

The system is not close to a live-ready bot that "knows how to trade" SPX 0DTE options.

The two biggest blockers are contract dishonesty and simulation optimism:

1. The active dataset and replay path are not scoring the contract the docs say they are scoring. The current `data.pt` label/replay path is tied to session-open ATM ladders, not the dynamic current ATM / strike-selection problem the system claims to solve.
2. The simulator is materially easier than live trading. It uses mid-like entry fills, exact stop/target fills, no stop slippage, and no implemented live execution stack to prove parity.

Runtime spot-checks make this worse, not better:

- `v2/data.pt` currently reports 47 features and Tier 3 labels, but its metadata is stale after relabeling. It claims `total_trades=160267` and `gate_true_rate=0.6773`; the actual label tensors contain `197041` trade bars and an `0.8327` trade rate on signal bars.
- The stored replay "ATM" prices are not current ATM prices. In `v2/data.pt`, `atm_call_prices` differs from `nearest_call_close` on 92.4% of comparable bars, with median relative difference 91.1%. `atm_put_prices` differs from `nearest_put_close` on 93.7% of bars, with median relative difference 75.8%.
- The current best artifact (`exp_066`) replays at 54 trades over 60 promote days, trades on only 29 days, and shows +$30.7k on a $10k account with one contract. That is evidence of a favorable harness, not evidence of live readiness.

## Part 1: Truthfulness Audit

### 1. Market Data

| Finding | File:Line | Doc claim | Code reality | Impact |
|---------|-----------|-----------|-------------|--------|
| Feature/docs contract is stale | `v2/docs/feature_schema.md:5-7`, `v2/docs/feature_schema.md:82-136`, `v2/docs/data_contract.md:45-86`, `v2/core/features.py:43-61`, `v2/pipeline/build_v2_dataset.py:307-318` | The project uses 39 features and per-day z-score normalization. | The live code and `data.pt` use 47 features and a 60-day rolling z-score. | Anyone reasoning from the docs is reasoning about the wrong input space and wrong normalization regime. |
| The documented multi-strike oracle is not the active labeling path | `v2/docs/labeling.md:36-61`, `v2/core/labels.py:133-284`, `v2/pipeline/build_v2_dataset.py:394-510`, `v2/pipeline/relabel_tier3.py:106-210` | The oracle searches up to 26 candidates per bar and stores the best executable trade. | The active relabel path grid-searches only `atm_call_prices` and `atm_put_prices`. `core/labels.py` is not used by the current `build_v2_dataset.py` / `relabel_tier3.py` pipeline. | The system says it is teaching strike selection and full trade choice. The active dataset is teaching ATM-only direction/risk on pre-baked price arrays. |
| Replay and current labels are keyed to session-open ATM ladders, not current ATM | `v2/pipeline/build_v2_dataset.py:209-251`, `v2/pipeline/relabel_tier3.py:106-107`, `v2/replay.py:340-364` | `atm_*` / `otm*` arrays represent the current ATM +/- offset universe the evaluator trades. | The arrays are built from `wide['atm_strike']` and fixed offsets from that opening strike. Replay later maps current intents back into those arrays by coarse offset bucket. Runtime inspection shows the stored ATM series differs from the separately stored dynamic nearest series on 92%+ of bars. | This is the single most damaging data-contract bug in the repo. The model is trained and evaluated on the wrong contracts once spot moves away from the open. |
| Active label window disagrees with policy/oracle window | `v2/core/labels.py:205`, `v2/core/policy.py:38-39`, `v2/pipeline/build_v2_dataset.py:378`, `v2/pipeline/relabel_tier3.py:148`, `v2/replay.py:206` | The system trades from bar 30 until bar 330. | The current dataset builders label only bars `< 270`, while replay and policy allow entries until bar 330. | The model can be evaluated on the last hour even though the active dataset never supervised that hour. |
| POC / value area leak the rest of the day | `v2/pipeline/compute_features.py:234-274` | Market-structure features are valid intraday decision features. | POC and value area are computed once from the full day and copied onto every bar in that day. | The model sees future session structure on every intraday decision. That is plain look-ahead bias. |
| Tier 3 relabeling is incomplete and internally inconsistent | `v2/docs/labeling.md:48-55`, `v2/core/labels.py:40-42`, `v2/pipeline/relabel_tier3.py:35-39`, `v2/pipeline/relabel_tier3.py:81-93` | Tier 3 includes holds `[30, 60, 120, 240, 390]` and matches evaluator cost logic. | `relabel_tier3.py` omits `390`, and its exit spread uses the configured `hold` instead of the actual exit bar. | The active `data.pt` can never label end-of-day holds and systematically misprices early exits. |
| `data.pt` metadata is not truthful after relabeling | `v2/pipeline/build_v2_dataset.py:620-639`, `v2/pipeline/relabel_tier3.py:261-269`, `v2/data.pt` (runtime inspection) | Metadata records the active dataset's true label regime and cost model. | After relabeling, `label_scheme` is updated, but `total_trades`, `gate_true_rate`, `mean_pnl_trade`, and `cost_model` remain stale or misleading. The current artifact still says flat `spread_rt=0.3`. | The primary artifact people inspect for truth is lying about both label density and friction. |

### 2. Training Runs

| Finding | File:Line | Doc claim | Code reality | Impact |
|---------|-----------|-----------|-------------|--------|
| The model does not emit TradeIntents and does not score dynamic candidates | `v2/program.md:3`, `v2/docs/goal.md:20-22`, `v2/docs/how_training_works.md:46`, `v2/train.py:179-209`, `v2/replay.py:72-142` | v2 removed translation layers and the model learns to emit trades. | The model predicts call/put P&L plus raw risk. Replay then reconstructs a `TradeIntent` through a translation layer. | The codebase still has the exact translation-layer problem the docs claim v2 fixed. |
| Strike selection is fake | `v2/program.md:53-57`, `v2/docs/goal.md:53-55`, `v2/train.py:195-197`, `v2/replay.py:98-100`, `v2/core/candidates.py:48-181` | Strike selection is dynamic from real candidates. | The strike head is hardcoded to ATM. `v2/core/candidates.py` exists but has no call sites under `v2/`. | Contract selection is non-functional. The model cannot learn or express the most important choice in 0DTE trading after direction. |
| Risk decoding contract is self-contradictory | `v2/core/policy.py:27`, `v2/train.py:160-163`, `v2/replay.py:106-115` | Risk outputs are sigmoid-squashed into policy ranges. | The model emits unbounded raw risk values and replay hard-clamps them. | Capacity is wasted on learning bounds, policy comments are false, and out-of-range predictions do not mean what the docs say they mean. |
| Gate supervision is misaligned with inference | `v2/pipeline/build_v2_dataset.py:509-510`, `v2/pipeline/relabel_tier3.py:209-210`, `v2/train.py:330-334`, `v2/core/policy.py:25` | Trade/no-trade is learned consistently from the same decision boundary used in replay. | Labels set `trade=True` only when best P&L exceeds 4%, while training/replay gate logic treats `max(pred_call_pnl, pred_put_pnl) > 0` as the threshold behind `gate_threshold=0.50`. | The model is trained to one gate and executed with another. That makes gate metrics and research sweeps misleading. |
| Confidence and direction "probabilities" are not probabilities | `v2/train.py:192-200`, `v2/replay.py:93-99`, `v2/replay.py:121` | Replay consumes meaningful confidence and direction outputs. | Replay applies `softmax` to raw P&L regression outputs and `sigmoid` to an arbitrary `3x` margin scalar. | If confidence or direction scores are ever used for sizing or filtering, they are numerically cosmetic, not calibrated signals. |
| Feature schema is not stored or validated where it matters | `v2/train.py:31`, `v2/train.py:126`, `v2/train.py:490-504`, `v2/ops/artifact.py:88-105`, `v2/ops/artifact.py:166-175` | Model artifacts are self-describing and safe to reload against the current data. | Checkpoints/artifacts store hyperparameters but not feature names or feature count. `NUM_FEATURES` comes from process env/defaults, not from the artifact. | Feature-pipeline changes can silently break reloads or produce opaque dimension failures. |

### 3. Validation

| Finding | File:Line | Doc claim | Code reality | Impact |
|---------|-----------|-----------|-------------|--------|
| Fill model is materially more optimistic than documented | `v2/docs/evaluator.md:34-37`, `v2/core/simulator.py:87-88`, `v2/core/simulator.py:151-176` | Market buys fill at ask and sells at bid; stops/targets follow evaluator rules. | Simulator fills entry from next-bar mid series and exits at exact stop/target/trailing prices with no gap-through slippage. | Replay systematically overstates live P&L on 0DTE contracts where spreads and jumps dominate outcomes. |
| Replay computes stop/target off the wrong premium | `v2/replay.py:287-295`, `v2/replay.py:118-119`, `v2/core/simulator.py:97-103` | Risk is referenced to the chosen contract. | Replay uses `option_mid = max(atm_call_px, atm_put_px)` before it even knows direction. For the cheaper side, that can generate invalid stop/target levels and skip otherwise valid trades. | This distorts side selection and risk evaluation, especially when call/put prices diverge. |
| Replay cannot honestly evaluate strike choice | `v2/replay.py:340-364`, `v2/pipeline/build_v2_dataset.py:209-251` | The evaluator scores current ATM +/- offsets. | Replay bins intents into coarse `atm/otm5/...` arrays that are themselves anchored to the session-open ATM. | Even if the model learned strike selection tomorrow, replay would still not be scoring the intended contract. |
| Daily loss cap is dead in the main scoring path | `v2/core/policy.py:46`, `v2/core/simulator.py:228-315`, `v2/replay.py:309-316` | The policy's daily loss cap is part of simulator safety. | `simulate_day()` enforces it, but replay never calls `simulate_day()`. It calls `simulate_trade()` directly. | Capital-preservation rules that look implemented in docs and policy are not part of the score the loop actually optimizes. |
| Baseline definitions in docs and code do not match | `v2/docs/baselines.md:43-49`, `v2/docs/baselines.md:61-67`, `v2/replay.py:427-520`, `v2/replay.py:637-643`, `v2/replay.py:573-580`, `v2/replay.py:737-746` | Random baseline is a 50/50 trade coin flip averaged over 100 seeds; simple-rules uses raw 5-bar momentum; ATM baselines are honest comparators. | Random baseline trades at 2% probability over 5 pooled seeds, simple-rules thresholds a z-scored feature at `0.005`, and both ATM baselines trade calls only. | The promotion bar is not the one the docs describe, and half the directional problem is not baseline-tested at all. |
| Replay has no account solvency constraint | `v2/replay.py:251-327`, `v2/core/metrics.py:224-249` | The model is evaluated on a $10k account curve. | Replay never stops trading after equity is depleted. Runtime spot-check: the random baseline reached `AcctDD=510.2%` and `Equity=-$32,902.96`. | The score can be computed on a bankrupt account. That is not a live-tradable risk model. |
| One documented exit policy does not exist | `v2/core/schema.py:23`, `v2/docs/evaluator.md:126-132`, `v2/core/simulator.py:165-191` | `MODEL_EXIT` is a supported exit policy. | The simulator only handles stop/target/trailing/max-hold/EOD. There is no model-exit branch. | Exit intelligence is overstated in docs and absent in the evaluator. |

### 4. ART2 Pipeline

| Finding | File:Line | Doc claim | Code reality | Impact |
|---------|-----------|-----------|-------------|--------|
| Session limits have no single source of truth | `v2/program.md:152-160`, `CLAUDE.md:17`, `v2/ops/inner_loop.py:33-38`, `v2/docs/baselines.md:98-107` | The session limits are defined and enforced consistently. | `program.md` / `CLAUDE.md` say 20 experiments, 10 hours, 6 reverts, 4-hour plateau. `inner_loop.py` / `baselines.md` say 50 experiments, 6 hours, 8 reverts, 3-hour plateau. | Depending on which file the operator follows, the same "protocol" means different behavior. |
| The pipeline switched to walk-forward, but docs and stop logic still talk to the old runner | `v2/program.md:19`, `v2/program.md:136`, `CLAUDE.md:23`, `v2/ops/deploy.sh:823`, `v2/ops/deploy.sh:853-863`, `v2/ops/inner_loop.py:270-275` | `run_experiment.py` is the canonical runner. | `deploy.sh run_one` runs `run_experiment_wf.py`, while `cmd_stop` kills `run_experiment.py` and `inner_loop.py` still shells into the old runner. | The orchestration layer is not internally consistent. Stopping or reproducing runs can target the wrong process. |
| Artifacts are saved before keep/revert, and replay defaults to the highest-score artifact whether it was kept or not | `v2/ops/run_experiment_wf.py:69-85`, `v2/ops/artifact.py:186-208`, `v2/replay.py:58-69` | Artifact loading reflects the promoted model lineage. | Every finished run can create an artifact. `get_best_artifact()` ignores keep/revert state and just picks the highest manifest score. Replay default loads that artifact. | A reverted experiment can become the implicit "best model" for replay and analysis. That breaks research history integrity. |
| `results.tsv` experiment IDs are not unique | `v2/results.tsv:2-20`, `v2/ops/inner_loop.py:346-347` | Experiment IDs identify a unique run. | IDs restart from `exp_001` each session and the log already contains duplicates. | You cannot reconstruct the research path from the log without manual archaeology. |
| `deploy.sh start` still advertises warm-starting | `v2/ops/deploy.sh:428-434`, `v2/train.py:415`, `CLAUDE.md:20`, `v2/program.md:129-140` | Every experiment trains from scratch. | `train.py` does train from scratch, but `deploy.sh start` still uploads `v2/model.pt` "for warm-start". | The code path and the operator story disagree on whether prior weights matter. |
| Pre-GPU gates are documented as hard requirements but are not actually enforced | `v2/docs/baselines.md:5-18`, `v2/docs/baselines.md:106-121`, `v2/docs/migration.md:129-147`, `v2/ops/deploy.sh:803-823` | Tests, determinism checks, and local baseline gates block bad GPU runs. | There is no `tests/` or `tests/v2/` tree in the repo root, and `run_one` does not run any local gate before spending GPU time. | The pipeline claims discipline it does not actually enforce. |

### 5. IBKR Paper Trading

| Finding | File:Line | Doc claim | Code reality | Impact |
|---------|-----------|-----------|-------------|--------|
| The live system does not exist yet | `v2/docs/goal.md:65-71`, `v2/live/service.py:8-12`, `v2/live/market.py:11-14`, `v2/live/decision.py:12-15`, `v2/live/execution.py:12-17` | The roadmap is headed toward a bot that can run a full RTH paper session autonomously. | Every live module is still a TODO stub. | There is no path from the current best artifact to actual paper trading. |
| Feature parity is unimplemented and under-specified where it matters most | `v2/core/features.py:43-90`, `v2/pipeline/build_v2_dataset.py:189-191`, `v2/live/market.py:11-14` | Live will compute the same 47 features. | The live feature engine is not implemented, and the required 60-day normalization bootstrap is still a TODO. | Even a "working" execution engine would feed the model a different distribution than training/replay. |
| Execution-state safety is documentation only | `v2/docs/execution.md:71-125`, `v2/docs/execution.md:177-220`, `v2/live/execution.py:12-17`, `v2/live/service.py:10-12` | The system has reconnect recovery, orphan handling, kill switch, timeouts, and audit logs. | The code implementing those states does not exist. | The stated completion gate in `goal.md` is not even partially satisfied. |
| The contract-resolution spec is wrong for SPXW weeklies | `v2/docs/contracts.md:183-192` | `resolve_to_ibkr()` is a correct broker resolution template. | The spec routes to `exchange="SMART"` and does not specify `tradingClass="SPXW"` / explicit CBOE weekly handling. | The live spec can resolve the wrong instrument family before the implementation even starts. |
| Quote provenance and mid-trade risk-adjustment contracts are defined but unused | `v2/core/schema.py:71-73`, `v2/core/schema.py:156-165`, `v2/replay.py:123-142`, `v2/live/execution.py:12-17` | The shared trade contract already supports quote audit and risk adjustment. | Replay does not populate `bid_at_decision` / `ask_at_decision`, and no live code consumes `RiskAdjustment`. | The supposedly shared contract is only partially exercised. The audit trail needed for live slippage analysis is missing from day zero. |

## Part 2: Simulation-to-Live Gap Analysis

The P&L distortion estimates below are inferences from the code paths and the observed contract drift in `v2/data.pt`, not from a live fill study. They are still directionally reliable enough to make planning decisions.

| Gap | File:Line | Estimated distortion | Bias |
|-----|-----------|----------------------|------|
| Entry fills use next-bar mid-like prices instead of ask; exits fill exactly at stop/target/trailing prices | `v2/docs/evaluator.md:34-37`, `v2/core/simulator.py:87-88`, `v2/core/simulator.py:151-176` | Usually 5% to 20% of premium per trade on cheap 0DTE contracts once half-spread and stop gaps are included. Larger on fast moves. | Optimistic |
| Replay and active relabeling use session-open ATM arrays as if they were current ATM/current strike arrays | `v2/pipeline/build_v2_dataset.py:209-251`, `v2/pipeline/relabel_tier3.py:106-107`, `v2/replay.py:340-364`, `v2/data.pt` runtime inspection | On trend days this can change the contract economics by tens to hundreds of percent. The median relative difference between stored "ATM" and dynamic nearest ATM is 91.1% for calls and 75.8% for puts. | Mostly optimistic in directional moves; structurally invalid either way |
| Risk is parameterized off `max(atm_call_px, atm_put_px)` instead of the chosen side | `v2/replay.py:287-295`, `v2/core/simulator.py:97-103` | Distorts stop/target geometry on the cheaper side and can skip valid trades entirely. The effect can be the full difference between call and put premiums at entry. | Usually pessimistic for the cheaper side, but inconsistent |
| Current labels are ATM-only and current replay cannot score real strike choice | `v2/docs/labeling.md:36-61`, `v2/train.py:195-197`, `v2/core/candidates.py:48-181`, `v2/replay.py:340-364` | Not a small spread miss. It removes an entire decision dimension that often dominates 0DTE P&L. | Optimistic about "model competence"; pessimistic about what live deployment will reveal |
| Live feature engine has no implemented 60-day normalization bootstrap | `v2/core/features.py:97-142`, `v2/live/market.py:11-14` | Early live sessions will run out-of-distribution. Feature shifts can be multiple z-score units on volatility and flow features. | Unstable; likely pessimistic at startup |
| Historical option features use full-chain Polygon bars and per-bar BS solves; live parity path is missing | `v2/pipeline/compute_features.py:625-709`, `v2/live/market.py:11-14` | Feature drift is unbounded until a real parity layer exists. `gamma_pressure`, `atm_iv`, `iv_percentile`, and flow features are the obvious failure points. | Unstable; direction unknown |
| No stop-gap/slippage model for 0DTE gamma moves | `v2/core/simulator.py:152-176`, `v2/docs/execution.md:102-110` | Worst on stops. A stop meant to lose 20% to 30% can lose much more in live conditions. | Optimistic |
| No implemented live state machine, kill switch, reconnect, or orphan handling | `v2/live/execution.py:12-17`, `v2/live/service.py:8-12`, `v2/docs/execution.md:129-220` | Not a "small friction miss". It turns a profitable replay into a deployment failure mode. | Optimistic about deployability |

## Part 3: "Does the Model Know How to Trade?" Assessment

Runtime reference point: replaying the current best artifact (`exp_066`) on `promote_mask` produced 54 trades across 60 days, traded on only 29 days, and scored 5.5862 under the current harness. That is evidence the model can exploit this replay setup. It is not evidence the system can run a full day live bot.

| Dimension | Rating | Evidence | Assessment |
|-----------|--------|----------|------------|
| Entry quality | Partially functional | `v2/train.py:179-209`, `v2/train.py:330-334`, `v2/pipeline/relabel_tier3.py:209-210`, runtime replay of `exp_066` | The model can gate trades and avoid constant firing in replay, but the gate is trained against a different threshold than it uses at inference, and the active labels say "trade" on 83.3% of signal bars. The current entry logic is not clean enough to trust live. |
| Contract selection | Non-functional | `v2/train.py:195-197`, `v2/replay.py:98-100`, `v2/core/candidates.py:48-181` | The model always emits ATM. The candidate engine is dead code. The system does not know how to choose strikes. |
| Risk management | Partially functional | `v2/train.py:160-163`, `v2/replay.py:106-119`, `v2/pipeline/relabel_tier3.py:176-218` | Stop/target/hold are predicted and replayed, but the head is unbounded then clamped, current labels omit hold-to-EOD, and replay sometimes computes risk from the wrong reference premium. There is some functionality here, but it is not trustworthy. |
| Exit intelligence | Non-functional | `v2/core/schema.py:23`, `v2/core/simulator.py:165-191`, `v2/docs/evaluator.md:126-132` | There is no model-driven exit. Exits are mechanical stop/target/trailing/time rules. The system does not know when to get out because of new information. |
| Regime awareness | Partially functional | `v2/train.py:141-149`, `v2/train.py:168`, `v2/core/features.py:97-142`, `v2/live/market.py:11-14` | There is a regime encoder and regime-heavy feature set, but the encoder only sees the last bar, and live parity for regime-sensitive features is not implemented. The idea exists. The end-to-end behavior does not. |
| Capital preservation | Non-functional | `v2/core/policy.py:46`, `v2/core/simulator.py:228-315`, `v2/replay.py:309-316`, `v2/live/service.py:10-12`, runtime random baseline | The documented daily loss cap is dead in replay, there is no insolvency stop, and live kill-switch / flatten logic is still a TODO. The system does not currently have real capital-preservation machinery. |

Bottom line: the model can rank ATM call vs ATM put inside an optimistic replay harness. It does not yet know how to choose contracts, manage live exits, preserve capital, or run a full-day autonomous trading process.

## Part 4: Action Plan

### Direct Improvements

1. Fix the contract series the system is actually labeling and replaying.
Files: `v2/pipeline/build_v2_dataset.py`, `v2/pipeline/relabel_tier3.py`, `v2/replay.py`
Expected impact: Removes the biggest source of contract dishonesty. Makes labels, replay, and future strike work talk about the same instrument.
Priority: P0

2. Make the simulator at least directionally honest about fills.
Files: `v2/core/simulator.py`, `v2/docs/evaluator.md`, `v2/pipeline/relabel_tier3.py`, `v2/pipeline/build_v2_dataset.py`
Expected impact: Reduces the largest optimism in the system. Scores will drop, but they will mean more.
Priority: P0

3. Align the active label window with the policy/replay window, and align gate labeling with inference.
Files: `v2/pipeline/build_v2_dataset.py`, `v2/pipeline/relabel_tier3.py`, `v2/train.py`, `v2/core/policy.py`
Expected impact: Removes unsupervised last-hour trading and stops the model from learning one gate while being executed with another.
Priority: P0

4. Repair `data.pt` metadata so it describes the active artifact after relabeling.
Files: `v2/pipeline/relabel_tier3.py`
Expected impact: Prevents every downstream analysis from reading stale trade counts, stale gate rate, and fake friction assumptions.
Priority: P1

5. Stop replay from default-loading reverted artifacts.
Files: `v2/ops/run_experiment_wf.py`, `v2/ops/artifact.py`, `v2/replay.py`
Expected impact: Restores a single source of truth for "best model" and makes analysis reproducible.
Priority: P1

6. Remove stale protocol lies or update them immediately.
Files: `v2/program.md`, `CLAUDE.md`, `v2/docs/feature_schema.md`, `v2/docs/data_contract.md`, `v2/docs/baselines.md`, `v2/docs/how_training_works.md`, `v2/docs/goal.md`
Expected impact: Prevents humans and future sessions from optimizing against a fictional system.
Priority: P2

### Deeper Planning

1. Build a real candidate-scoring path end to end.
Problem: The system claims dynamic strike selection but only trains ATM. The candidate engine exists as dead code.
Target state: Labels search real candidate contracts, the model scores candidate contracts, replay simulates the chosen contract, and live uses the same contract identity.
Dependencies and risks: Requires changing dataset storage, replay mapping, model outputs, and live decision logic together. The current open-ATM ladder design cannot be patched into honesty.
Suggested sequence: dataset storage first, replay second, model/output redesign third.

2. Build a real live feature-parity layer before any paper-order work.
Problem: The 47-feature training schema depends on rolling history, option-chain volume, and BS-derived Greeks. Live parity is not implemented.
Target state: A live feature engine that can reproduce every training feature or explicitly drop/replace the ones that cannot be reproduced.
Dependencies and risks: Requires historical bootstrap, feature persistence, and a per-feature parity test harness. Without this, live inference will be out-of-distribution from the first minute.
Suggested sequence: bootstrap buffer, per-feature parity report, then inference integration.

3. Build the live execution state machine as a real system, not a doc set.
Problem: The repo has specs for kill switch, reconnect recovery, orphan handling, and audit logs, but no code.
Target state: `v2/live/service.py` + `v2/live/execution.py` manage a full RTH session with JSONL audit trail, bracket reconciliation, and hard flatten.
Dependencies and risks: Requires careful reuse or porting from the v1 live stack, plus mock IBKR tests before any paper session.
Suggested sequence: contract resolver, entry/exit lifecycle, reconnect/orphan logic, then kill switch / EOD flatten.

4. Re-open the "immutable harness" rule until the harness is actually correct.
Problem: The research loop is currently locked to `train.py` / `policy.py`, but the biggest blockers are in dataset, replay, simulator, and live parity.
Target state: A short, explicit harness-repair phase with versioned evaluator/data fingerprints and parity tests, followed by a return to a narrow mutable surface.
Dependencies and risks: Requires process discipline. Without it, the team will keep tuning a model against broken labels and broken replay contracts.
Suggested sequence: fix contract storage, fix simulator honesty, add parity tests, then re-freeze.

5. Replace the current hill-climbing objective with one that can survive structural change.
Problem: The current loop rejects any paradigm shift that temporarily lowers score, and the score is already dominated by consistency over participation.
Target state: A protocol that can evaluate harness fixes and structural model changes without forcing immediate monotonic score improvement against the old benchmark.
Dependencies and risks: Requires explicit branching of "research score" vs "parity score" and probably a temporary reset of the leaderboard.
Suggested sequence: finish harness repair first, then redefine the score target and promotion protocol.

## Part 5: Recommended Next Session

Do not spend the next session on another `train.py` hyperparameter sweep. The next session should be a harness-repair session. If you keep tuning the model now, you will optimize harder against the wrong contract and the wrong fill assumptions.

Recommended next 10 experiments, in order:

1. Measure the contract drift on a small audit slice.
Files: add a one-off script under `v2/analysis/`
Hypothesis: the stored replay ATM ladder is not the current ATM ladder.
Success criteria: produce a report showing mismatch frequency and relative price error by day. The 92%+ mismatch seen in the global spot-check should be reproducible on a small slice.

2. Patch `build_v2_dataset.py` so replay price tensors are keyed to current ATM / current offset, not session-open ATM.
Files: `v2/pipeline/build_v2_dataset.py`
Hypothesis: replay contract dishonesty is coming from how price tensors are stored, not just how they are decoded.
Success criteria: `atm_*` and `nearest_*` become equivalent by construction, and offset tensors represent current offsets, not open-of-day offsets.

3. Patch `relabel_tier3.py` to relabel against the corrected contract tensors, include hold `390`, and price exit spread off the actual exit bar.
Files: `v2/pipeline/relabel_tier3.py`
Hypothesis: the current active labels are teaching the wrong contract and the wrong hold grid.
Success criteria: `label_grid` includes `390`, metadata matches actual label counts, and the relabeler no longer uses stale open-ATM arrays.

4. Rebuild a small debug `data.pt` and verify metadata truthfulness.
Files: `v2/pipeline/build_v2_dataset.py`, `v2/pipeline/relabel_tier3.py`
Hypothesis: once relabeling is fixed, metadata and tensors should agree exactly.
Success criteria: `metadata.total_trades`, `metadata.gate_true_rate`, and actual tensor counts match exactly.

5. Patch `core/simulator.py` to use an explicitly harsher live-proxy fill model.
Files: `v2/core/simulator.py`, `v2/docs/evaluator.md`
Hypothesis: current scores are overstated by mid entry and exact stop fills.
Success criteria: replay P&L drops on both model and baselines, but rank ordering stays stable and the code path matches the written evaluator.

6. Align trade-window supervision.
Files: `v2/pipeline/build_v2_dataset.py`, `v2/pipeline/relabel_tier3.py`, `v2/core/policy.py`
Hypothesis: the model should not be allowed to trade where it was never labeled.
Success criteria: either labels extend to bar 330 or replay/policy clamp to bar 270. No remaining mismatch.

7. Align gate training with gate execution.
Files: `v2/train.py`, `v2/pipeline/relabel_tier3.py`, possibly `v2/core/policy.py`
Hypothesis: the current 4% label gate and 0% inference gate are fighting each other.
Success criteria: one threshold definition exists, it is policy-linked, and both label generation and replay use it.

8. Fix artifact lineage.
Files: `v2/ops/run_experiment_wf.py`, `v2/ops/artifact.py`, `v2/replay.py`
Hypothesis: replay should never silently load a reverted model.
Success criteria: artifacts are either saved only on KEEP or tagged with keep/revert, and `load_best_model()` only loads promoted artifacts while enforcing dataset fingerprint compatibility.

9. Run one full walk-forward retrain on the repaired dataset/replay harness.
Files: same as above plus the normal experiment runner
Hypothesis: the score will likely drop, but it will finally correspond to the contract and fill model the system says it uses.
Success criteria: one honest baseline score for the repaired harness, plus per-fold metrics and promote replay on the corrected contract series.

10. Only after the repaired replay is stable, start the live-parity sprint.
Files: `v2/live/market.py`, `v2/live/decision.py`, `v2/live/execution.py`
Hypothesis: paper-trading work before replay/data repair will just port the wrong assumptions into IBKR.
Success criteria: a concrete feature-parity checklist, bootstrap plan, and v1-port map for market data and execution before any paper orders are sent.

If the team wants the shortest path toward the goal, the next session should treat this as a data/replay integrity repair, not as a model-improvement session.
