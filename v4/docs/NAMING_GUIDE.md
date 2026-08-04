# v4 Naming Guide

Protocol numbers are historical IDs, not the main project language. Keep them in file names and reports for backward compatibility, but every new artifact must lead with a plain-English role label.

For model-improvement work, naming is not enough. Use [MODEL_IMPROVEMENT_GUIDELINES.md](MODEL_IMPROVEMENT_GUIDELINES.md) before calling a challenger "better" or discussing paper-default replacement. Use [HYPOTHESIS_TO_PROMOTION_PROCESS.md](HYPOTHESIS_TO_PROMOTION_PROCESS.md) for the required stage-gate path from hypothesis to experiment to validation to promotion.

## Role Labels

| Role | Meaning | Name pattern | Example |
|---|---|---|---|
| Paper Default | The only candidate currently allowed to drive paper/live trading. | `PAPER_DEFAULT_<candidate>` | `PAPER_DEFAULT_PROTOCOL101` |
| Research Challenger | A model or strategy trying to beat the paper default. It is not allowed to submit paper/live orders. | `CHALLENGER_<idea>_V#` | `CHALLENGER_LIFECYCLE_SLOT_AWARE_V1` |
| Experiment | A training run, model change, objective change, feature change, or replay test. | `EXP_<date>_<idea>_V#` | `EXP_2026_05_22_LIFECYCLE_CONTEXT_CALIBRATION_V1` |
| Diagnostic / Audit | A test that explains behavior but is not itself a model. | `AUDIT_<behavior>_V#` | `AUDIT_CHURN_REENTRY_V1` |
| Runtime / Parity Harness | A live-safe or no-order infrastructure test. | `RUNTIME_<purpose>_V#` | `RUNTIME_NO_ORDER_SHADOW_V1` |
| Freeze / Promotion Decision | A decision packet that changes or freezes a candidate status. | `DECISION_<status>` | `DECISION_FREEZE_CHALLENGER_RESEARCH_ONLY` |

## Required Report Header

Every new report must start with these fields:

```text
What is this:
Does it change the paper-trading default:
Candidate being tested:
Paper default baseline:
Other baseline:
Data used:
Paid data downloaded:
Broker endpoint called:
Next experiment:
```

## Status Rules

- A `Research Challenger` can be profitable and still not be paper-ready.
- A `Diagnostic / Audit` can explain a failure mode but must not be described as a model.
- A `Runtime / Parity Harness` can prove safe plumbing but not historical edge.
- A `Freeze / Promotion Decision` must explicitly say whether the paper default changes.
- `PAPER_DEFAULT_PROTOCOL101` remains the paper/live default until a decision packet says otherwise.

## Historical Protocol Map

| Historical ID | Plain-English role | Current status |
|---|---|---|
| Protocol051 | Surface-edge feature/model component used by the Protocol101 stack. | Legacy component |
| Protocol054 | Fallback lifecycle component used by the Protocol101 stack. | Legacy component |
| Protocol081 | Frozen lifecycle/exit replay component used by several challengers. | Baseline lifecycle component |
| Protocol101 | `PAPER_DEFAULT_PROTOCOL101`: current paper-trading default. | Paper default |
| Protocol113 | `DIAGNOSTIC_TRADE_AND_EQUITY_VISUALS`: serial replay visual inspection artifacts. | Diagnostic |
| Protocol114 | `AUDIT_PROTOCOL101_SKEPTICAL_FALSIFICATION_V1`: falsification audit for the equity curve. | Audit |
| Protocol126 | `AUDIT_TIMING_FRAGILITY_V1`: high-resolution delay and quote-freshness timing audit. | Audit |
| Protocol127 | `RUNTIME_LIVE_SHADOW_SCHEMA_V1`: no-order JSONL schema hardening. | Runtime harness |
| Protocol128 | `RUNTIME_PAPER_ACCOUNT_RISK_GATE_V1`: deterministic paper-account risk gate. | Runtime harness |
| Protocol129 | `EXP_OFFLINE_POSITION_SIZING_V1`: offline multi-contract sizing simulator. | Research experiment |
| Protocol130 | `RUNTIME_TUESDAY_NO_ORDER_RUNBOOK_V1`: no-order live-shadow runbook. | Runtime harness |
| Protocol155 | `RUNTIME_ONE_CONTRACT_LIVE_TIMING_EVIDENCE_V1`: one-contract paper/live evidence plan. | Runtime harness |
| Protocol157 | `RUNTIME_DAILY_OPS_MONITOR_V1`: daily monitor for startup, decisions, orders, and logs. | Runtime harness |
| Protocol160 | `RUNTIME_PERSISTENT_PAPER_TRADER_V1`: persistent IBKR paper-trader loop. | Runtime harness |
| Protocol161 | `DIAGNOSTIC_MAY2026_HISTORICAL_REPLAY_V1`: May 19-20 historical replay. | Diagnostic |
| Protocol162 | `AUDIT_MAY2026_SERIAL_LIFECYCLE_REPLAY_V1`: proved independent entries overstated May 20. | Audit |
| Protocol163 | `EXP_SERIAL_ONE_ACCOUNT_ENTRY_POLICY_V1`: serial account-aware entry training from Protocol101-exposed candidates. | Rejected / attribution only |
| Protocol164 | `EXP_FULL_ACTION_SPACE_DATASET_V1`: full SPXW ladder candidate universe. | Dataset experiment |
| Protocol165 | `EXP_FULL_ACTION_SPACE_NEURAL_POLICY_V1`: full-action neural policy. | Research experiment |
| Protocol166 | `RUNTIME_TRAIN_LIVE_PARITY_CONTRACT_V1`: shared historical/live candidate contract. | Runtime parity contract |
| Protocol194 | `CHALLENGER_FULL_ACTION_SURFACE_EDGE_V1`: strongest entry-side challenger; beat Protocol101 historically but needed timing/runtime validation. | Research challenger |
| Protocol197 | `RUNTIME_CHALLENGER_PARITY_LATENCY_V1`: no-order runtime/latency harness for Protocol194. | Runtime harness |
| Protocol198 | `AUDIT_CHURN_REENTRY_V1`: same-side exit/re-entry churn counterfactual. | Audit |
| Protocol199 | `AUDIT_LIFECYCLE_FULL_PATH_ORACLE_V1`: lifecycle continuation label-surface audit. | Audit |
| Protocol200 | `EXP_LIFECYCLE_CONTINUATION_V1`: first causal hold/exit continuation model. | Rejected |
| Protocol201 | `AUDIT_LIFECYCLE_CONTINUATION_FAILURE_V1`: overholding and blocked-slot attribution. | Audit |
| Protocol202 | `CHALLENGER_LIFECYCLE_SLOT_AWARE_V1`: slot-aware lifecycle challenger. | Research challenger only |
| Protocol203 | `AUDIT_SLOT_AWARE_MIXED_RESULT_V1`: attribution for Protocol202 mixed result. | Audit |
| Protocol204 | `EXP_RECURRENT_SLOT_AWARE_LIFECYCLE_V1`: recurrent lifecycle model. | Rejected as replacement |
| Protocol205 | `AUDIT_RECENT_GAP_SLOT_AWARE_V1`: recent_2026 gap attribution for Protocol202. | Audit |
| Protocol206 | `DECISION_FREEZE_CHALLENGER_RESEARCH_ONLY`: freezes Protocol202 as research-only. | Decision packet |
| Protocol207 | `EXP_2026_05_22_LIFECYCLE_CONTEXT_CALIBRATION_V1`: validation-only side/time lifecycle calibration test. | Research experiment |
| Protocol208 | `AUDIT_CONTEXT_CALIBRATED_RECENT_GAP_V1`: attribution for the context-calibrated challenger's recent lifecycle miss. | Audit |
| Protocol209 | `EXP_2026_05_22_UNIFIED_ENTRY_LIFECYCLE_SEQUENCE_V1`: shared entry-slot and hold/exit sequence model. | Rejected research experiment |
| Protocol210 | `AUDIT_FULL_ACTION_FEATURE_PARITY_V1`: checks whether the full-action surface-edge dataset has the Protocol101-style causal inputs needed for fair train/live parity. | Audit |
| Protocol211 | `EXP_2026_05_22_FULL_ACTION_HISTORY_FEATURE_REPAIR_V1`: adds missing causal short-history features to the full-action surface-edge dataset. | Dataset repair experiment |
| Protocol212 | `EXP_2026_05_22_FULL_ACTION_SURFACE_EDGE_HISTORY_POLICY_SMOKE_V1`: smoke test proving the repaired full-action feature set can train/evaluate through the two-stage policy runner. | Smoke experiment only |
| Protocol213 | `EXP_2026_05_22_FULL_ACTION_SURFACE_EDGE_HISTORY_POLICY_SCREEN_V1`: one-seed all-session screen for the repaired full-action surface-edge/history challenger. | Research screen; requires multi-seed confirmation |
| Protocol214 | `EXP_2026_05_22_FULL_ACTION_SURFACE_EDGE_HISTORY_ADDITIONAL_SEEDS_V1`: additional seed confirmation run for seeds 2-5. | Supporting confirmation run |
| Protocol215 | `EXP_2026_05_22_FULL_ACTION_SURFACE_EDGE_HISTORY_5SEED_CONFIRMATION_V1`: combined five-seed confirmation for the repaired full-action surface-edge/history challenger. | Confirmed research challenger; not paper default |
| Protocol216 | `AUDIT_CHALLENGER_FULL_ACTION_HISTORY_VS_PROTOCOL101_V1`: attribution of the confirmed challenger versus Protocol101 across churn, side/time exposure, exact trade overlap, and directional move capture. | Audit |
| Protocol217 | `RUNTIME_FULL_ACTION_HISTORY_NO_ORDER_PARITY_V1`: no-order runtime parity and latency harness for the confirmed challenger feature set. | Runtime harness; paper default unchanged |
| Protocol218 | `DIAGNOSTIC_CHALLENGER_FULL_ACTION_HISTORY_TRADE_CHARTS_V1`: serial one-account visual inspection charts for the confirmed challenger. | Diagnostic; paper default unchanged |
| Protocol219 | `RUNTIME_FULL_ACTION_HISTORY_FEATURE_BUILDER_PARITY_V1`: live-style causal history feature-builder parity against the Protocol211 parquet build. | Runtime harness; paper default unchanged |
| Protocol220 | `AUDIT_CHALLENGER_MONEYNESS_AND_PROMOTION_BLOCKERS_V1`: ITM/ATM/OTM and premium-profile audit plus promotion blocker summary for the confirmed challenger. | Audit; paper default unchanged |
| Protocol221 | `EXP_2026_05_22_RETURN_ON_PREMIUM_FULL_ACTION_POLICY_V1`: pure return-on-premium objective for the full-action/history challenger architecture. | Rejected as replacement; useful direction |
| Protocol222 | `EXP_2026_05_22_CONFIDENCE_SCALED_PREMIUM_SIZING_V1`: offline confidence/premium-at-risk sizing overlay for the return-on-premium challenger. | Research-only sizing candidate; paper default unchanged |
| Protocol223 | `EXP_2026_05_22_ACCOUNT_AWARE_CONFIDENCE_SIZING_V2`: account-balance-aware confidence sizing with hard premium caps, liquidity caps, drawdown throttling, and multi-balance stress tests. | Research-only sizing challenger; paper default unchanged |
| Protocol224 | `AUDIT_2026_05_22_SCALE_IN_OUT_PATH_OPPORTUNITY_V1`: path audit for scale-in/scale-out opportunity, continuation value, and averaging-down risk. | Audit; paper default unchanged |
| Protocol225 | `EXP_2026_05_22_ACCOUNT_AWARE_LIFECYCLE_EXIT_POLICY_V1`: causal hold/exit model over the account-aware sized trade stream. | Rejected; exited too early and reduced PnL |
| Protocol226 | `EXP_2026_05_22_BASELINE_ANCHORED_CONTINUATION_V1`: baseline-exit-anchored continuation test that could extend but not exit earlier. | Rejected; no material improvement |
| Protocol227 | `EXP_2026_05_22_SCALE_OUT_RUNNER_SCREEN_V1`: simple scale-out runner screen using frozen baseline exits plus forced-flat runner. | Rejected; validation selected no runner |
| Protocol228 | `EXP_2026_05_22_BASELINE_RELATIVE_EARLY_EXIT_V1`: early-exit model only when current bid was predicted better than baseline exit. | Rejected; no material improvement |
| Protocol229 | `AUDIT_2026_05_22_POSITION_ACTION_DP_ORACLE_V1`: hindsight hold/reduce/exit oracle over intratrade paths. | Audit; shows huge exit-timing upper bound but mostly hold/exit-all labels |
| Protocol230 | `EXP_2026_05_22_ORACLE_EXIT_ACTION_CLASSIFIER_V1`: class-weighted causal classifier for sparse oracle exit moments. | Rejected as replacement; improved PF/DD but gave up PnL |
| Protocol231 | `AUDIT_2026_05_22_ACCOUNT_AWARE_SIZING_RISK_SCREENS_V1`: conservative and balanced risk-parameter screens for account-aware sizing. | Audit; original Protocol223 sizing remained best of tested overlays |
| Protocol232 | `EXP_2026_05_22_BLENDED_DOLLAR_PREMIUM_POLICY_V1`: blended dollar-plus-premium full-action objective. | Research experiment; led to premium-leaning challenger |
| Protocol233 | `AUDIT_2026_05_22_BLENDED_POLICY_ACCOUNT_AWARE_SIZING_SCREEN_V1`: account-aware sizing screen on the 65/35 blended policy trade stream. | Rejected sizing overlay; paper default unchanged |
| Protocol234 | `EXP_2026_05_22_PREMIUM_LEANING_BLEND_SCREEN_V1`: 45/55 dollar/premium blended objective screen. | Promising research challenger; lower premium and beat Protocol101 |
| Protocol235 | `AUDIT_2026_05_22_PREMIUM_LEANING_ACCOUNT_AWARE_SIZING_SCREEN_V1`: account-aware sizing screen on the 45/55 challenger. | Rejected sizing overlay; stress fragile |
| Protocol236 | `EXP_2026_05_22_STRONG_PREMIUM_BLEND_SCREEN_V1`: 25/75 dollar/premium blended objective screen. | Research screen; more OTM/capital-efficient but lower dollar PnL |
| Protocol237 | `AUDIT_2026_05_22_STRONG_PREMIUM_ACCOUNT_AWARE_SIZING_SCREEN_V1`: account-aware sizing screen on the 25/75 challenger. | Rejected sizing overlay; Q3/Q4 stress fragile |
| Protocol238 | `EXP_2026_05_22_PREMIUM_LEANING_BLEND_SEED_STABILITY_V1`: 3-seed stability run for the 45/55 premium-leaning challenger. | Passed research gate; paper default unchanged |
| Protocol239 | `EXP_2026_05_22_PREMIUM_LEANING_BLEND_SEED_HOLDOUT_4_5_V1`: seed 4-5 stability run for the same 45/55 challenger. | Passed holdout seed gate; paper default unchanged |
| Protocol240 | `DECISION_2026_05_22_PREMIUM_LEANING_BLEND_FIVE_SEED_V1`: combined five-seed freeze packet for the premium-leaning blended utility challenger. | Frozen research challenger; paper default unchanged |
| Protocol241 | `RUNTIME_PREMIUM_LEANING_BLEND_NO_ORDER_PARITY_V1`: no-order runtime parity and latency harness for the frozen premium-leaning challenger. | Runtime parity passed; paper default unchanged |
| Protocol242 | `AUDIT_PREMIUM_LEANING_BLEND_VS_PROTOCOL101_V1`: attribution of the premium-leaning challenger versus Protocol101 across side/time exposure, exact overlap, churn, and directional move capture. | Attribution supports challenger; paper default unchanged |
| Protocol243 | `RUNTIME_PREMIUM_BLEND_OFFHOURS_STACK_BRIDGE_V1`: off-hours construction/stack bridge proving the challenger can rebuild trained full-action features from live-style quote objects, while showing recorded Protocol101 logs are too narrow for replacement promotion. | Runtime bridge passed; paper default unchanged |
| Protocol244 | `RUNTIME_PREMIUM_BLEND_PAPER_RUNTIME_SHELL_V1`: paper-style no-broker runtime shell for the premium-leaning challenger, emitting monitorable JSONL/CSV decision logs and validating hypothetical order intents. | Runtime shell passed; paper default unchanged |
| Protocol245 | `RUNTIME_PREMIUM_BLEND_LIVE_SURFACE_AUTOTEST_V1`: Tuesday broker-connected no-order live surface breadth/freshness check for the premium-leaning challenger using the full SPXW ATM +/- $50 ladder. | Scheduled runtime harness; paper default unchanged |
| Protocol247 | `AUDIT_PREMIUM_BLEND_METRIC_RECONCILIATION_V1`: corrected the premium-blend versus Protocol101 comparison by separating five-seed attribution, five-seed medians, and single-seed equity-chart metrics. | Audit; paper default unchanged |
| Protocol248 | `AUDIT_CHALLENGER_FAILURE_SURFACE_V1`: apples-to-apples strict-serial scorecard for challenger weakness analysis. | Audit; paper default unchanged |
| Protocol249 | `EXP_ENTRY_QUALITY_CALIBRATOR_V1`: second-stage gate over challenger entries. | Research experiment; paper default unchanged |
| Protocol250 | `EXP_CONTRACT_VALUE_SELECTION_HEAD_V1`: contract-value selection head over the full action surface. | Research experiment; paper default unchanged |
| Protocol261 | `CHALLENGER_ROUTER_SOURCE_PENALTY_CALIBRATED_V1`: source-penalty router stream used as the base for later continuation work. | Research challenger; paper default unchanged |
| Protocol265 | `CHALLENGER_SOURCE_PENALTY_BASELINE_ANCHORED_CONTINUATION_V1`: baseline-anchored continuation challenger with persisted artifacts. | Research-only challenger; paper default unchanged |
| Protocol266 | `AUDIT_PROTOCOL265_ARTIFACT_REPRODUCTION_V1`: exact replay reproduction of Protocol265 saved artifacts. | Artifact audit |
| Protocol267 | `DECISION_FREEZE_PROTOCOL265_RESEARCH_ONLY_V1`: freezes Protocol265 as reproducible research-only and keeps Protocol101 as paper default. | Decision packet |
| Protocol268 | `AUDIT_PROTOCOL265_EXTENSION_REGIME_ATTRIBUTION_V1`: explains where Protocol265 lifecycle extensions help or hurt by regime, side, time, moneyness, Greeks, and churn. | Audit |
| Protocol269 | `RUNTIME_PROTOCOL265_NO_ORDER_PARITY_V1`: no-order runtime parity proxy for Protocol265 saved artifacts. | Runtime harness; paper default unchanged |
| Protocol270 | `DATASET_FULL_SURFACE_ACTION_ADVANTAGE_V1`: full-surface serial wait/enter action-advantage labels under the unified game contract. | Dataset; paper default unchanged |
| Protocol271 | `CHALLENGER_UNIFIED_ACTION_ADVANTAGE_POLICY_V1`: candidate-set neural challenger trained on full-surface action-advantage labels. | Research challenger; paper default unchanged |
| Protocol272 | `AUDIT_FILL_MODEL_READINESS_V1`: checks whether live/paper logs contain enough fills to calibrate execution instead of deterministic ask/bid plus stress. | Audit |
| Protocol273 | `AUDIT_MODEL_SELECTION_OVERFIT_RISK_V1`: inventories repeated split exposure and requires a future untouched block before final promotion claims. | Meta-validation audit |
| Protocol274 | `DATASET_POSITION_STATE_ACTION_ADVANTAGE_V1`: holding-state hold-vs-exit labels using executable quote paths plus future flat-slot opportunity value. | Dataset; paper default unchanged |
| Protocol275 | `CHALLENGER_POSITION_STATE_LIFECYCLE_POLICY_V1`: research-only neural hold/exit model trained on Protocol274 position-state action advantages. | Research challenger component; paper default unchanged |
| Protocol276 | `CHALLENGER_INTEGRATED_ENTRY_LIFECYCLE_SERIAL_REPLAY_V1`: strict one-account replay integrating the Protocol271 full-action entry policy with the Protocol275 learned lifecycle policy. | Research integration replay; paper default unchanged |
| Foundation unified conservative offline policy | `FOUNDATION_UNIFIED_CONSERVATIVE_OFFLINE_POLICY_V1`: frozen state/action/execution/label contract for the next best-foundation ML direction after abandoning Protocol276 as a candidate. | Foundation contract; paper default unchanged |

## Writing Rule

Use this style in conversation and reports:

```text
CHALLENGER_LIFECYCLE_SLOT_AWARE_V1, historically Protocol202, ...
```

Do not write this as the main explanation:

```text
Protocol202 did X.
```
