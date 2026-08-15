# V4 Baseline Inventory

Audit date: 2026-05-24
Repository root: `/Users/gduby/Documents/autoresearch-trading`
Purpose: read-only baseline inventory of the current v4 control before the `research_ops` transition.

This report is a cartography artifact, not a strategy recommendation. It verifies the current operational surface against code, docs, artifacts, logs, and local git state. No trading logic, runtime flags, launchd assets, broker scripts, paid-data scripts, training jobs, threshold tuning jobs, or challenger promotion jobs were run.

## 0. Git Freeze Status

Stage 1.1 and 1.2 are only partially complete because the repository is not clean.

- Current checked-out branch: `v4/phase-0`.
- Current `HEAD`: `87ded0ba30fa236ab9c1b90fce7e82ff7f25a478`.
- Local `main`: `8ca3f362ff1e731d6f8a7936b094aa650fce52a7`.
- Created branch ref: `research-ops-transition` at local `main` commit `8ca3f362ff1e731d6f8a7936b094aa650fce52a7`.
- Working tree state before this inventory: `36` tracked dirty entries with untracked entries excluded; `1107` total porcelain entries with untracked entries included.
- Frozen tag `v4-protocol101-control-2026-05-24`: not created. The Stage 1 instruction says not to tag until the working tree is clean.
- `git pull --rebase origin main`: not completed. Network fetch/checkout attempts hung and were terminated without changing the checked-out branch.
- `git checkout research-ops-transition`: not completed. The repo remains on `v4/phase-0`.
- Pushes: not performed.

Operational implication: the control is not yet a clean immutable git object. Before using the tag as governance evidence, the team must decide whether the intended control is local `main`, current `v4/phase-0`, or the dirty working tree including local artifacts and edits.

## 1. Current Operational Default

The current operational trading bot is the v4 `Protocol101` SPXW 0DTE stack. The source-of-truth doc identifies `PAPER_DEFAULT_PROTOCOL101` as the paper default and explicitly keeps Protocol194, Protocol240, Protocol265, Protocol276, and unified-conservative systems as research-only unless a later decision packet changes that default.

Primary evidence:

- `docs/CURRENT_TRADING_BOT_SINGLE_SOURCE_OF_TRUTH.md:10` names v4 `Protocol101` as the current operational bot and `PAPER_DEFAULT_PROTOCOL101` as the current paper default.
- `docs/CURRENT_TRADING_BOT_SINGLE_SOURCE_OF_TRUTH.md:24` says guarded IBKR paper runtime exists and is scheduled, but latest inspected sessions remain effectively no-entry/no-order.
- `v4/docs/NAMING_GUIDE.md` is cited by the source-of-truth as the role map for `PAPER_DEFAULT_PROTOCOL101`.
- `v4/ops/ibkr/run_protocol101_paper_session.sh:13-15` defaults to `PROTOCOL101_SESSION_MODE=no-order-shadow`, `PROTOCOL101_ENTRY_BRIDGE_MODE=paper-submit`, and `PROTOCOL101_SESSION_KIND=persistent`.
- `v4/ops/ibkr/run_protocol101_paper_session.sh:43-58` executes `python -m v4.scripts.run_protocol160_protocol101_persistent_paper_trader` for the persistent default.
- `v4/scripts/run_protocol160_protocol101_persistent_paper_trader.py:1-7` states that Protocol160 preserves the frozen Protocol101 entry model, Protocol066/081 lifecycle exit model, and paper-order guardrails.
- `v4/scripts/run_protocol160_protocol101_persistent_paper_trader.py:95` defaults `--mode` to `paper-submit`.

Trading scope documented as current truth:

- Underlying context: SPX index context with SPXW option contracts.
- Instrument: same-day PM-settled SPXW calls and puts.
- Direction: long options only.
- Current paper guard size: one contract.
- Entry price convention: buy at current ask in live paper executor; ask-entry in replay.
- Exit price convention: sell at current bid in live paper executor; bid-exit in replay.
- Account assumption: `$10,000` paper cash baseline; `$500` reserve is documented but not subtracted by `validate_order_intent`.
- Max concurrency: one open position.
- Not in current default: non-SPX underlyings, AM-settled SPX, multi-leg spreads, futures, equities, crypto, short options, real-money orders, default multi-contract paper trading.

## 2. Current Model Artifacts

Current entry model:

- Runtime loader: `v4/live/protocol101_entry.py`.
- Artifact loader: `load_protocol101_entry_artifact()` in `v4/live/protocol101_entry.py:89-119`.
- History state: `Protocol101HistoryState` in `v4/live/protocol101_entry.py:46-87`.
- Candidate builder: `protocol101_candidate_frame_from_surface()` in `v4/live/protocol101_entry.py:122-149`.
- Default Protocol101 directory: `v4/audit/autoresearch/v4_aplus_hypothesis_101_event_history_policy`.
- Default Protocol101 manifest: `v4/audit/autoresearch/v4_aplus_hypothesis_101_event_history_policy/model_artifacts/fold3_train_q1_q2_q3_validate_q4_test_q1_2026/seed_1/manifest.json`.
- Default Protocol101 summary: `v4/audit/autoresearch/v4_aplus_hypothesis_101_event_history_policy/summary.json`.
- Artifact files under the Protocol101 directory include `manifest.json`, `model.pt`, and `scaler.json` for fold1, fold2, and fold3 seeds 1 through 5.

Current upstream surface scorer:

- Default surface manifest: `v4/audit/autoresearch/v4_aplus_hypothesis_075_protocol054_frozen_stack_artifacts/model_artifacts/train_through_q4_2025_test_q1_2026/seed_11/manifest.json`.
- Runtime import: `load_surface_edge_artifact()` and `score_surface_decisions()` are imported by Protocol160 at `v4/scripts/run_protocol160_protocol101_persistent_paper_trader.py:30`.
- Protocol101 therefore remains a two-model stack: Protocol051/054-style surface scores feed Protocol101 entry features.

Current lifecycle model:

- Runtime lifecycle loader: `load_protocol066_artifact()` imported at `v4/scripts/run_protocol160_protocol101_persistent_paper_trader.py:31`.
- Runtime lifecycle default: `DEFAULT_PROTOCOL081_MANIFEST` imported by Protocol160 at `v4/scripts/run_protocol160_protocol101_persistent_paper_trader.py:46-58`.
- Default Protocol081 manifest path is defined in `v4/scripts/run_protocol081_live_shadow_router.py`.
- `v4/live/protocol066_inference.py` describes frozen Protocol066 lifecycle inference and maps model output plus hard lifecycle constraints into hold/exit/stop/forced-flat actions.

Important artifact caveat:

- The latest source-of-truth says the Protocol101 score margin is weakly calibrated: strategy forensics reported Spearman score-margin versus PnL of `-0.0571` in `v4/audit/autoresearch/protocol101_strategy_forensics_packet_v1/report.md:77`.
- Protocol101 remains the control because it is the guarded default, not because executable edge has been proven.

## 3. Current Paper Runtime Entrypoints

Primary daily paper runtime:

- `v4/ops/ibkr/run_protocol101_paper_session.sh`
  - Defaults `PROTOCOL101_ENTRY_BRIDGE_MODE=paper-submit`.
  - Defaults `PROTOCOL101_SESSION_KIND=persistent`.
  - Defaults `PROTOCOL101_ENABLE_PAPER_ORDERS=YES`.
  - Defaults `PROTOCOL101_ACKNOWLEDGE_PAPER_LOSS=YES`.
  - Runs `v4.scripts.run_protocol160_protocol101_persistent_paper_trader` when persistent.
  - Falls back to `v4.scripts.run_protocol147_protocol101_morning_session` for non-persistent/cycled mode.

Persistent runtime:

- `v4/scripts/run_protocol160_protocol101_persistent_paper_trader.py`
  - `parse_args()` defaults to `--mode paper-submit`.
  - Uses default manifests for surface, Protocol101, Protocol101 summary, runtime flag, runtime state, and live index context log at `v4/scripts/run_protocol160_protocol101_persistent_paper_trader.py:93-105`.
  - Loads surface, Protocol101, and lifecycle artifacts at `v4/scripts/run_protocol160_protocol101_persistent_paper_trader.py:171-173`.
  - Calls `execute_guarded_paper_order()` only after intent, mode, runtime flag, and guard conditions at `v4/scripts/run_protocol160_protocol101_persistent_paper_trader.py:683-725`.
  - Writes runtime state only if an entry result reports a fill at `v4/scripts/run_protocol160_protocol101_persistent_paper_trader.py:736-737`.

Bridge and legacy runtime:

- `v4/scripts/run_protocol158_protocol101_live_entry_paper_bridge.py`
  - One-shot/live-entry paper bridge used by earlier readiness flows and by Protocol160 for shared helpers.
  - Contains `live_option_quotes()` and quote-freshness helpers.
  - Checks runtime flag before paper-submit entry and exit paths.
- `v4/scripts/run_protocol147_protocol101_morning_session.py`
  - Legacy/cycled morning harness used by the shell when `PROTOCOL101_SESSION_KIND` is not persistent.

Operational monitor and gate:

- `v4/scripts/run_protocol150_protocol101_paper_order_enablement_gate.py`
  - Writes `v4/runtime/protocol101_paper_order_enablement.json` only when called with `--write-runtime-flag` and gate passes.
- `v4/scripts/run_protocol157_protocol101_daily_ops_monitor.py`
  - Summarizes paper activity, launchd state, and runtime flag state.

Broker-facing paper files:

- `v4/live/ibkr_paper_guard.py`
- `v4/live/ibkr_paper_executor.py`
- `v4/live/paper_trade_log.py`
- `v4/live/protocol101_live_entry.py`
- `v4/scripts/ibkr_preflight.py`
- `v4/scripts/check_ibkr_live_data_entitlements.py`

Launch assets:

- `v4/ops/launchd/com.autoresearch.protocol101.paper-session.plist`
- `v4/ops/launchd/com.autoresearch.protocol101.paper-preflight.plist`
- `v4/ops/launchd/com.autoresearch.protocol101.daily-monitor.plist`
- `v4/ops/launchd/com.autoresearch.ibgateway.paper.plist`
- `v4/ops/launchd/com.autoresearch.ibgateway.paper-shutdown.plist`
- `v4/ops/launchd/install_ibkr_paper_autostart.sh`
- `v4/ops/launchd/uninstall_ibkr_paper_autostart.sh`

## 4. Current Replay Entrypoints

Core replay/simulation modules:

- `v4/sim/paper_replay.py`
- `v4/sim/shadow_paper.py`
- `v4/sim/protocol101_position_sizing.py`
- `v4/sim/simulator.py`

Protocol101 and lifecycle replay scripts:

- `v4/scripts/run_protocol090_strict_shadow_lifecycle_replay.py`
- `v4/scripts/run_protocol101_event_history_policy.py`
- `v4/scripts/run_protocol118_protocol101_shadow_rehearsal.py`
- `v4/scripts/run_protocol161_may2026_historical_replay.py`
- `v4/scripts/run_protocol162_may2026_serial_lifecycle_replay.py`
- `v4/scripts/run_protocol168_threshold_replay.py`
- `v4/scripts/run_protocol178_protocol175_recall_threshold_replay.py`
- `v4/scripts/run_protocol276_integrated_entry_lifecycle_serial_replay.py`
- `v4/scripts/run_unified_conservative_neural_policy_strict_replay.py`

Replay evidence/artifact directories include:

- `v4/audit/autoresearch/v4_aplus_hypothesis_162_may2026_serial_lifecycle_replay`
- `v4/audit/autoresearch/v4_aplus_hypothesis_163_recent_protocol101_serial_lifecycle_replay`
- `v4/audit/autoresearch/v4_aplus_hypothesis_168_protocol163_threshold_replay`
- `v4/audit/autoresearch/unified_conservative_neural_policy_strict_replay_v1`
- `v4/audit/autoresearch/unified_conservative_neural_policy_flat_calibrated_strict_replay_v1`

Replay caveat:

- `v4/sim/simulator.py:115` still includes `NullSimulator`; `v4/sim/simulator.py:138` states it rejects all intents in Phase 0. Current research claims therefore depend on specialized replay scripts and paper logs, not on a single production-grade calibrated execution simulator.

## 5. Current Logs And Artifact Directories

Paper trading logs:

- Root: `v4/logs/paper_trading`.
- Recent inspected session: `v4/logs/paper_trading/2026-05-21/protocol101_persistent-paper_2026-05-21.jsonl`.
- CSV companion: `v4/logs/paper_trading/2026-05-21/protocol101_persistent-paper_2026-05-21.csv`.
- Latest inspected JSONL row count: `2737`.
- The first row confirms `mode=paper-submit`, `paper_trading=true`, `real_money_trading=false`, `live_orders_enabled=true`, and runtime flag `paper_orders_enabled=true`.
- The source-of-truth reports this session had `broker_order_endpoint_called = 0` and `paper_orders_submitted = 0`.

Operational runtime artifacts:

- Protocol160 output root: `v4/audit/autoresearch/v4_aplus_hypothesis_160_protocol101_persistent_paper_trader`.
- Paper enablement gate artifacts: `v4/audit/autoresearch/v4_aplus_hypothesis_150_protocol101_paper_order_enablement_gate`.
- Live timing evidence: `v4/audit/autoresearch/v4_aplus_hypothesis_155_protocol101_live_timing_evidence`.
- Live entry paper bridge evidence: `v4/audit/autoresearch/v4_aplus_hypothesis_158_protocol101_live_entry_paper_bridge`.

Research audit directories that are relevant to the control:

- `v4/audit/autoresearch/v4_aplus_hypothesis_101_event_history_policy`
- `v4/audit/autoresearch/protocol101_strategy_forensics_packet_v1`
- `v4/audit/autoresearch/protocol101_internal_slot_cost_counterfactual_v1`
- `v4/audit/autoresearch/truth_grounded_replacement_program_v1`
- `v4/audit/autoresearch/v4_aplus_hypothesis_272_fill_model_readiness`
- `v4/audit/autoresearch/v4_aplus_hypothesis_273_model_selection_overfit_risk`
- `v4/audit/autoresearch/formal_validation_governance`
- `v4/audit/autoresearch/untouched_holdout_availability`
- `v4/audit/autoresearch/unified_untouched_holdout_reservation`

Data artifact roots to treat as expensive or source-like:

- `data/processed`
- `data/cache`
- `v4/normalized_official_context`
- `v4/normalized_official_context_smoke`
- `v4/normalized_official_context_fix_smoke`
- `v4/raw`
- `v4/normalized`
- `v4/feature`
- `v4/label`

## 6. Mutable Runtime State

Current files under `v4/runtime`:

- `v4/runtime/protocol101_live_index_context.jsonl`
- `v4/runtime/protocol101_paper_order_enablement.json`

Runtime state referenced by current paper trader:

- `v4/runtime/protocol101_live_paper_state.json`
  - Referenced by `v4/scripts/run_protocol160_protocol101_persistent_paper_trader.py:104`.
  - Not present in the current `find v4/runtime -maxdepth 2 -type f` output.
  - Written only after a fill by Protocol160.

Mutable operational semantics:

- `protocol101_paper_order_enablement.json` controls whether paper-submit mode may call the guarded paper executor.
- `protocol101_live_index_context.jsonl` is rolling live context used by the persistent trader.
- Paper logs under `v4/logs/paper_trading` are append-only operational evidence, not hand-editable research scratch files.
- Protocol160 writes output directories under `v4/audit/autoresearch/v4_aplus_hypothesis_160_protocol101_persistent_paper_trader`.

Do not hand-edit these files unless an explicit decision memo authorizes the mutation.

## 7. Broker-Risk Files

The following files can connect to IBKR, manage IB Gateway, touch paper credentials/config, submit/cancel paper orders, inspect account state, or activate scheduled jobs. They must be treated as broker-risk.

Paper executor/guard/logging:

- `v4/live/ibkr_paper_executor.py`
  - File docstring says `placeOrder` is called only after the guard passes.
  - `ib.placeOrder(contract, order)` appears at `v4/live/ibkr_paper_executor.py:108`.
- `v4/live/ibkr_paper_guard.py`
  - `validate_order_intent()` begins at `v4/live/ibkr_paper_guard.py:81`.
  - Max option quote age default is `1500` ms at `v4/live/ibkr_paper_guard.py:27`.
- `v4/live/paper_trade_log.py`
- `v4/live/protocol101_live_entry.py`

Paper runtime and bridge scripts:

- `v4/scripts/run_protocol147_protocol101_morning_session.py`
- `v4/scripts/run_protocol150_protocol101_paper_order_enablement_gate.py`
- `v4/scripts/run_protocol158_protocol101_live_entry_paper_bridge.py`
- `v4/scripts/run_protocol160_protocol101_persistent_paper_trader.py`
- `v4/scripts/run_tuesday_protocol101_paper_fill_observation.py`
- `v4/scripts/check_ibkr_live_data_entitlements.py`
- `v4/scripts/ibkr_preflight.py`

IBKR ops scripts:

- `v4/ops/ibkr/probe_ibkr_api.py`
- `v4/ops/ibkr/start_ib_gateway_paper.sh`
- `v4/ops/ibkr/start_ib_gateway_paper_ibc.sh`
- `v4/ops/ibkr/shutdown_ibkr_paper_stack.sh`
- `v4/ops/ibkr/store_ibkr_paper_credentials.sh`
- `v4/ops/ibkr/write_ibc_runtime_config.py`
- `v4/ops/ibkr/wait_for_ibkr_api.py`
- `v4/ops/ibkr/run_protocol101_paper_preflight.sh`
- `v4/ops/ibkr/run_protocol101_paper_session.sh`
- `v4/ops/ibkr/run_protocol101_daily_monitor.sh`
- `v4/ops/ibkr/run_tuesday_no_order_evidence_collection.sh`
- `v4/ops/ibkr/run_tuesday_paper_fill_observation_collection.sh`
- `v4/ops/ibkr/run_tuesday_post_session_truth_audit.sh`

Launchd files:

- Everything under `v4/ops/launchd`, especially the Protocol101 paper session, paper preflight, daily monitor, IB Gateway paper, and Tuesday fill-observation plists plus install/uninstall scripts.

Control rule:

- A cartography/verifier agent may read these files.
- An implementation agent may not edit or run them without an explicit decision memo and a broker-risk checklist.

## 8. Paid-Data-Risk Files

Paid-data-risk files are any files that can call paid vendors, trigger downloads, mutate normalized market data roots, or rebuild derived datasets from purchased data.

Paid data guard and ingest:

- `v4/checks/paid_data_guard.py`
- `v4/ingest/databento_opra.py`
- `v4/scripts/build_databento_neural_dataset.py`

Download scripts:

- `v4/scripts/download_databento_cbbo_1s_audit.py`
- `v4/scripts/download_databento_cbbo_1s_selected.py`
- `v4/scripts/download_databento_context_proxies.py`
- `v4/scripts/download_databento_es_vwap.py`
- `v4/scripts/download_databento_pilot.py`
- `v4/scripts/download_protocol101_targeted_highres.py`
- `v4/scripts/download_thetadata_index_bars.py`

Data entitlement and validation scripts:

- `v4/scripts/check_ibkr_live_data_entitlements.py`
- `v4/scripts/run_protocol117_protocol101_targeted_highres_validation.py`

Paid or purchased data evidence/artifacts:

- `v4/audit/databento_*`
- `v4/audit/thetadata_*`
- `v4/audit/autoresearch/v4_aplus_hypothesis_111_paid_data_guardrail`
- `v4/audit/autoresearch/v4_aplus_hypothesis_116_protocol101_targeted_1s_download`
- `v4/audit/autoresearch/v4_aplus_hypothesis_117_protocol101_targeted_highres_path_audit`
- `v4/audit/autoresearch/v4_aplus_hypothesis_117_protocol101_targeted_highres_validation`
- `v4/promotion/PROTOCOL_101_TARGETED_CBBO_1S_DOWNLOAD_MANIFEST.json`

Control rule:

- No paid data script should run from a Stage 1 cartography session.
- Dataset rebuilds are research mutations and require an experiment RFC or verifier memo, not ad hoc exploration.

## 9. Current Known Blockers From The Audit

These blockers come from the source-of-truth and are verified against current code/artifacts where possible.

1. Execution/fill realism is not calibrated.
   - `docs/CURRENT_TRADING_BOT_SINGLE_SOURCE_OF_TRUTH.md:53` calls out `NullSimulator` and zero/insufficient fill observations.
   - `v4/sim/simulator.py:115` defines `NullSimulator`.
   - `v4/audit/autoresearch/truth_grounded_replacement_program_v1/report.md:52` marks fill observations as `blocked_insufficient_fill_observations`.
   - This is the highest-risk blocker because ask/bid replay profitability is not the same as executable PnL.

2. Latest inspected paper session had no actual paper orders/fills.
   - `v4/logs/paper_trading/2026-05-21/protocol101_persistent-paper_2026-05-21.jsonl` contains `2737` rows.
   - Source-of-truth reports `broker_order_endpoint_called = 0` and `paper_orders_submitted = 0`.
   - Paper-submit plumbing exists, but live fill evidence is absent.

3. Quote freshness and timestamp semantics remain unresolved.
   - `docs/CURRENT_TRADING_BOT_SINGLE_SOURCE_OF_TRUTH.md:54` says the older bridge path set `quote_age_ms=0`.
   - Current code has `quote_freshness_from_ticker()` at `v4/scripts/run_protocol158_protocol101_live_entry_paper_bridge.py:600` and `live_option_quotes()` at `v4/scripts/run_protocol158_protocol101_live_entry_paper_bridge.py:653`.
   - Protocol160 passes selected quote age into the executor at `v4/scripts/run_protocol160_protocol101_persistent_paper_trader.py:709-714`.
   - This is doc/code drift in the right direction, but parity and falsification evidence remain open.

4. Lifecycle live/replay mismatch risk remains.
   - Source-of-truth says `protocol066_action()` expects causal sequence state, while current live bridge builds a current-position row.
   - `v4/live/protocol066_inference.py` is the frozen lifecycle inference helper; Protocol160 imports `handle_open_position` from Protocol158 rather than showing a single unified replay/runtime lifecycle path.
   - This matters because exit behavior is often economically more important than entry selection in 0DTE options.

5. Protocol101 is a two-model stack.
   - Protocol160 loads the surface artifact and Protocol101 artifact separately.
   - Protocol101 entry features depend on upstream Protocol051/054-style surface scores.
   - Replay/live parity can fail if the surface scorer, live row builder, or candidate filter semantics differ from historical rows.

6. Score calibration is weak.
   - `v4/audit/autoresearch/protocol101_strategy_forensics_packet_v1/report.md:77` reports Spearman score-margin versus PnL of `-0.0571`.
   - A ranking score that does not correlate with realized utility is dangerous for threshold tuning, candidate priority, and future position sizing.

7. One-position slot cost is unresolved.
   - `v4/audit/autoresearch/protocol101_internal_slot_cost_counterfactual_v1/report.md:21-24` reports `2291` actual open trades with at least one blocked entry, `1109` where best blocked entry beats actual open-trade PnL, and `$399,510` best-blocked-minus-open total.
   - This means the one-position rule may change the problem from entry classification into optimal stopping/resource allocation.

8. Validation overfit and researcher degrees-of-freedom risk remain.
   - `v4/audit/autoresearch/truth_grounded_replacement_program_v1/report.md:42` marks validation overfit as blocked pending a strategy matrix/PBO/CSCV path.
   - Multiple protocol families and replay scripts have been explored. The control must be protected from benchmark mining.

9. Replacement policy training is explicitly blocked.
   - `v4/audit/autoresearch/truth_grounded_replacement_program_v1/report.md:14` says Protocol101 stays the control and no generic replacement model should be trained until diagnostics, data gates, execution evidence, validation controls, and untouched scoring exist.

10. Capital/reserve semantics are not fully aligned.
    - Source-of-truth documents `$10,000` paper cash and a `$500` reserve not subtracted by `validate_order_intent`.
    - This matters for account realism, sizing, and late-day option affordability.

## 10. Stale Or Contradictory Docs

The following docs should not be treated as operational truth without cross-checking current code and source-of-truth.

1. Root `README.md`
   - `README.md:10` says live execution is intentionally deferred.
   - `README.md:20-28` still frames `v2` as the active system and `v4` as clean-slate protocol work.
   - Current source-of-truth says v4 Protocol101 has guarded IBKR paper runtime and scheduled paper-submit mode. The README remains directionally true for real-money trading, but stale for current paper ops.

2. `v4/README.md`
   - `v4/README.md:3` says v4 has no shared imports with v2 or v3.
   - `v4/README.md:49` repeats "No imports from v2 or v3."
   - `v4/model/hypothesis_protocol.py:33` imports `v2.core.market_structure`.
   - Therefore the "clean-slate/no v2 imports" claim is false for current code.

3. `v4/promotion/PROTOCOL_101_PROMOTION_READINESS_PACKET.md`
   - `v4/promotion/PROTOCOL_101_PROMOTION_READINESS_PACKET.md:6` says Protocol101 is a research promotion candidate, not paper/live approved.
   - `v4/promotion/PROTOCOL_101_PROMOTION_READINESS_PACKET.md:38` says Protocol101 is not ready for broker-connected paper trading.
   - Current Protocol160/158 plus launchd/runtime flag implement guarded IBKR paper submission. Treat this packet as older blocker evidence, not current operational default.

4. Quote freshness statement drift
   - `docs/CURRENT_TRADING_BOT_SINGLE_SOURCE_OF_TRUTH.md:54` says the bridge sets `quote_age_ms=0`.
   - Current Protocol158 code has quote freshness helpers and Protocol160 passes quote age to the executor.
   - The right interpretation is not "quote freshness solved"; it is "older doc statement partially superseded, but replay/live timestamp parity still needs verifier evidence."

5. Root quickstart
   - `README.md:52-60` still uses v2 health/replay/plot commands.
   - These commands are not the current Protocol101 paper control entrypoints.

## 11. Files That Must Not Be Touched Without Explicit Approval

This list is intentionally conservative. "Touch" means edit, delete, regenerate, run in a mode that mutates state, or run in a mode that can reach broker/vendor endpoints.

Runtime state:

- `v4/runtime/protocol101_paper_order_enablement.json`
- `v4/runtime/protocol101_live_index_context.jsonl`
- `v4/runtime/protocol101_live_paper_state.json`
- `v4/logs/paper_trading/**`

Broker and paper runtime:

- `v4/live/ibkr_paper_executor.py`
- `v4/live/ibkr_paper_guard.py`
- `v4/live/paper_trade_log.py`
- `v4/live/protocol101_live_entry.py`
- `v4/scripts/run_protocol147_protocol101_morning_session.py`
- `v4/scripts/run_protocol150_protocol101_paper_order_enablement_gate.py`
- `v4/scripts/run_protocol158_protocol101_live_entry_paper_bridge.py`
- `v4/scripts/run_protocol160_protocol101_persistent_paper_trader.py`
- `v4/scripts/run_tuesday_protocol101_paper_fill_observation.py`
- `v4/scripts/check_ibkr_live_data_entitlements.py`
- `v4/scripts/ibkr_preflight.py`
- `v4/ops/ibkr/**`
- `v4/ops/launchd/**`

Current control model artifacts:

- `v4/audit/autoresearch/v4_aplus_hypothesis_101_event_history_policy/model_artifacts/**`
- `v4/audit/autoresearch/v4_aplus_hypothesis_101_event_history_policy/summary.json`
- `v4/audit/autoresearch/v4_aplus_hypothesis_101_event_history_policy/report.md`
- `v4/audit/autoresearch/v4_aplus_hypothesis_075_protocol054_frozen_stack_artifacts/model_artifacts/**`
- `v4/audit/autoresearch/v4_aplus_hypothesis_081_q4start_residual_sequence_deterministic_artifacts/model_artifacts/**`

Promotion/control docs:

- `v4/promotion/PROTOCOL_101_PROMOTION_READINESS_PACKET.md`
- Any `v4/promotion/**FREEZE**` file.
- Any promotion packet declaring paper/live readiness or changing `PAPER_DEFAULT_PROTOCOL101`.

Paid-data and dataset mutation:

- `v4/checks/paid_data_guard.py`
- `v4/ingest/databento_opra.py`
- `v4/scripts/download_databento_cbbo_1s_audit.py`
- `v4/scripts/download_databento_cbbo_1s_selected.py`
- `v4/scripts/download_databento_context_proxies.py`
- `v4/scripts/download_databento_es_vwap.py`
- `v4/scripts/download_databento_pilot.py`
- `v4/scripts/download_protocol101_targeted_highres.py`
- `v4/scripts/download_thetadata_index_bars.py`
- `v4/scripts/build_databento_neural_dataset.py`
- `data/processed/**`
- `data/cache/**`
- `v4/raw/**`
- `v4/normalized/**`
- `v4/normalized_official_context/**`
- `v4/feature/**`
- `v4/label/**`

Training, tuning, and promotion:

- `v4/scripts/run_protocol101_event_history_policy.py`
- `v4/scripts/run_protocol097_sequential_event_policy.py`
- Any `v4/scripts/run_protocol*` file that trains, retunes thresholds, writes model artifacts, writes runtime flags, or promotes challengers.
- Any challenger artifact directory under `v4/audit/autoresearch` unless an RFC says the session owns that research slice.

## Control Summary

The current control is operationally Protocol101 guarded paper, but it is not yet a clean git-frozen control. The engineering layer exists: runtime shell, launchd assets, runtime flag, guard, executor, trade log schema, live row builder, Protocol101 entry loader, surface artifact, lifecycle artifact, and paper logs. The scientific control is weaker than the engineering control: no meaningful fill evidence, unresolved execution realism, unresolved lifecycle parity, weak score calibration, one-position opportunity-cost evidence, and validation overfit risk.

Until the dirty repo state is resolved and the frozen tag is created, `research_ops` should treat this inventory as the Stage 1 baseline map, not as immutable provenance.
