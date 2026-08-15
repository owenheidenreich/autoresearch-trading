from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


ROLE_LABEL = "FOUNDATION_TRUTH_GROUNDED_REPLACEMENT_PROGRAM_V1"
DEFAULT_OUTPUT_DIR = Path("v4/audit/autoresearch/truth_grounded_replacement_program_v1")
DEFAULT_DOC = Path("v4/docs/TRUTH_GROUNDED_REPLACEMENT_PROGRAM_V1.md")
DEFAULT_FORENSICS_DIR = Path("v4/audit/autoresearch/protocol101_strategy_forensics_packet_v1")
DEFAULT_FILL_SUMMARY = Path("v4/audit/autoresearch/v4_aplus_hypothesis_272_fill_model_readiness/summary.json")
DEFAULT_OVERFIT_SUMMARY = Path("v4/audit/autoresearch/v4_aplus_hypothesis_273_model_selection_overfit_risk/summary.json")
DEFAULT_HOLDOUT_SUMMARY = Path("v4/audit/autoresearch/unified_untouched_holdout_reservation/summary.json")
DEFAULT_LEARNED_DEFER_SUMMARY = Path("v4/audit/autoresearch/learned_defer_challenger_research_packet_v1/summary.json")


REGISTRY_COLUMNS = [
    "playbook_id",
    "track",
    "status",
    "thesis",
    "market_mechanism",
    "entry_context",
    "entry_trigger",
    "invalidation",
    "hold_exit_logic",
    "forbidden_states",
    "data_needed",
    "falsification_test",
    "expected_edge",
    "expected_failure_mode",
    "protocol101_comparison",
    "required_diagnostic_artifact",
    "next_action",
    "model_training_allowed",
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _exists(path: str | Path) -> bool:
    return Path(path).exists()


def build_strategy_hypothesis_registry() -> pd.DataFrame:
    rows = [
        {
            "playbook_id": "PROTOCOL101_HARD_STOP_AVOIDANCE_GATE_V1",
            "track": "A_protocol101_forensics",
            "status": "diagnostic_only",
            "thesis": "Some Protocol101 hard-stop losers are identifiable before entry or shortly after early MFE appears.",
            "market_mechanism": "Late/exhausted directional entries or adverse quote/spread conditions turn premium-rich ITM options into fast full-premium losses.",
            "entry_context": "Protocol101 candidate that otherwise passes the current event policy.",
            "entry_trigger": "No new entry trigger; this is a rejection or early-risk gate over Protocol101 entries.",
            "invalidation": "Hard-stop losers are not separable from winners with causal features and early path state.",
            "hold_exit_logic": "Reject pre-entry if avoidable; otherwise switch to early giveback/loss guard after first favorable excursion.",
            "forbidden_states": "No rejection based on realized PnL, path MFE/MAE, future exit reason, or hindsight day outcome.",
            "data_needed": "Hard-stop autopsy, losing-day autopsy, quote freshness, spread, entry score, early path evidence.",
            "falsification_test": "Matched hard-stop candidates cannot be separated from winners without using future/path labels.",
            "expected_edge": "Reduce left-tail losses without materially reducing target/sequence-residual winners.",
            "expected_failure_mode": "Overfitting rare hard-stop rows and rejecting convex winners.",
            "protocol101_comparison": "Must improve Protocol101 drawdown and worst-day loss in the declared hard-stop regime.",
            "required_diagnostic_artifact": "PROTOCOL101_HARD_STOP_AUTOPSY_V1",
            "next_action": "Manually classify current hard_stop_autopsy.csv rows into pre-entry avoidable, path-management avoidable, execution artifact, or unavoidable.",
            "model_training_allowed": False,
        },
        {
            "playbook_id": "PROTOCOL101_CONFIRMED_MFE_RUNNER_OVERLAY_V1",
            "track": "A_protocol101_forensics",
            "status": "blocked_missing_post_exit_paths",
            "thesis": "Some Protocol101 winners should transition from scalp capture into runner management after confirmed MFE.",
            "market_mechanism": "0DTE directional continuation can persist after the inherited exit, but only after the trade proves itself.",
            "entry_context": "Existing Protocol101 entry with early favorable movement.",
            "entry_trigger": "Post-entry state transition after causal MFE/velocity/quote conditions confirm continuation.",
            "invalidation": "Post-exit bid paths show extension mostly gives back gains or blocks better opportunities.",
            "hold_exit_logic": "Keep baseline exit for normal trades; allow runner transition only in confirmed-MFE archetypes with giveback guard.",
            "forbidden_states": "No blanket hold-longer rule; no runner transition on fallback losers or stale/wide quotes.",
            "data_needed": "Post-exit bid path from actual exit to forced flat, MFE/MAE, giveback, later slot opportunities.",
            "falsification_test": "Runner extension loses after slippage/stress or benefit concentrates in a tiny number of exposed days.",
            "expected_edge": "Capture fuller moves without degrading Protocol101's high hit-rate scalp behavior.",
            "expected_failure_mode": "Recreates Protocol200/276 overholding and slot blocking.",
            "protocol101_comparison": "Must beat Protocol101 only within declared confirmed-runner regime after slot-cost charge.",
            "required_diagnostic_artifact": "PROTOCOL101_CONFIRMED_MFE_RUNNER_AUDIT_V1",
            "next_action": "Build post-exit path attachment for Protocol101 trades before any runner model.",
            "model_training_allowed": False,
        },
        {
            "playbook_id": "PROTOCOL101_INTERNAL_SLOT_DEFER_V1",
            "track": "A_protocol101_forensics",
            "status": "blocked_missing_counterfactual_flat_actions",
            "thesis": "Some valid Protocol101 entries are not worth taking because they block better later Protocol101 opportunities.",
            "market_mechanism": "The one open-position slot has option value during high-signal 0DTE sessions.",
            "entry_context": "Protocol101 entry when account is flat.",
            "entry_trigger": "Defer if current expected edge is below counterfactual next-slot value plus uncertainty charge.",
            "invalidation": "Counterfactual-flat replay shows blocked Protocol101 opportunities are rare or lower value than taken entries.",
            "hold_exit_logic": "This is entry/defer logic, not lifecycle replacement.",
            "forbidden_states": "No use of deployed holding labels as proxy for counterfactual-flat entries.",
            "data_needed": "Counterfactual-flat Protocol101 event replay that scores every event as if no position were open.",
            "falsification_test": "Taken entries have nonnegative net slot value after later feasible opportunities are charged.",
            "expected_edge": "Avoid low-margin early trades that consume the only slot before stronger same-session opportunities.",
            "expected_failure_mode": "Over-defers and removes Protocol101's core compounding edge.",
            "protocol101_comparison": "Must improve baseline-relative utility while preserving trade count in high-confidence regimes.",
            "required_diagnostic_artifact": "PROTOCOL101_INTERNAL_SLOT_COST_V1",
            "next_action": "Build counterfactual-flat Protocol101 action table.",
            "model_training_allowed": False,
        },
        {
            "playbook_id": "PROTOCOL101_MISSED_WINNER_ABSTENTION_EXPANSION_V1",
            "track": "B_new_alpha_playbooks",
            "status": "diagnostic_only",
            "thesis": "Protocol101 may reject Protocol101-like candidates that are causally distinguishable from rejected losers.",
            "market_mechanism": "The current curated candidate stream and wait threshold may be too narrow around certain side/time/premium regimes.",
            "entry_context": "Full-surface candidate similar to Protocol101 trades but rejected or not selected.",
            "entry_trigger": "Only after matched-control tests show rejected winners are separable from rejected losers using causal features.",
            "invalidation": "Missed winners disappear under causal matching and are only visible through oracle/path sorting.",
            "hold_exit_logic": "Use Protocol101-compatible exit unless a separate runner audit authorizes lifecycle changes.",
            "forbidden_states": "No candidate chosen because of future PnL, oracle action, path_max_pnl, or q_enter label.",
            "data_needed": "Full-surface action-advantage rows, Protocol101 baseline actions, causal feature similarity, rejected-candidate controls.",
            "falsification_test": "Matched rejected candidates do not outperform selected Protocol101 trades after costs and slot charge.",
            "expected_edge": "Carefully widen Protocol101 without inheriting broad challenger churn.",
            "expected_failure_mode": "Hindsight bait: broader entry raises PnL in exposed replay but worsens win rate, PF, drawdown, or timing fragility.",
            "protocol101_comparison": "Must beat Protocol101 in a predeclared rejected-candidate regime under strict serial replay.",
            "required_diagnostic_artifact": "PROTOCOL101_MISSED_WINNER_ABSTENTION_AUDIT_V1",
            "next_action": "Join full_surface_action_advantage.parquet to Protocol101 baseline actions and build matched controls.",
            "model_training_allowed": False,
        },
        {
            "playbook_id": "SIDE_SPECIFIC_PUT_REGIME_PLAYBOOK_V1",
            "track": "B_new_alpha_playbooks",
            "status": "diagnostic_only",
            "thesis": "Protocol101 puts may require a separate volatility/exhaustion playbook from calls.",
            "market_mechanism": "Put trades can monetize volatility expansion but may fail differently around exhaustion, spread, and IV regimes.",
            "entry_context": "Put candidates in Protocol101 or full-surface candidate sets.",
            "entry_trigger": "Side-specific context confirms put edge beyond shared Protocol101 score.",
            "invalidation": "Put/call differences are unstable across splits or disappear after timing/fill stress.",
            "hold_exit_logic": "Use side-specific stop/giveback/runner rules only after path audit.",
            "forbidden_states": "No global put penalty or put boost without regime evidence.",
            "data_needed": "Side-specific trade atlas, IV/VIX context, spread, moneyness, score, MFE/MAE, losing-day labels.",
            "falsification_test": "Put regime rules do not improve out-of-sample Protocol101 put performance after costs.",
            "expected_edge": "Reduce put hard-stop losses while preserving higher average put PnL.",
            "expected_failure_mode": "Overfits the small hard-stop sample and misses crash/volatility winners.",
            "protocol101_comparison": "Must improve put-only risk-adjusted performance without degrading call behavior.",
            "required_diagnostic_artifact": "PROTOCOL101_SIDE_SPECIFIC_STRATEGY_AUDIT_V1",
            "next_action": "Build call/put asymmetry report by time, premium, moneyness, exit reason, MFE/MAE, and delay stress.",
            "model_training_allowed": False,
        },
        {
            "playbook_id": "SIDE_SPECIFIC_CALL_CONTINUATION_PLAYBOOK_V1",
            "track": "B_new_alpha_playbooks",
            "status": "diagnostic_only",
            "thesis": "Call continuation may be a cleaner trend-following playbook than shared call/put Protocol101 logic.",
            "market_mechanism": "Calls may benefit from trend continuation and lower panic-spread behavior than puts in specific windows.",
            "entry_context": "Call candidates in post-open or late-afternoon continuation regimes.",
            "entry_trigger": "Causal trend/continuation context supports call follow-through beyond current Protocol101 score.",
            "invalidation": "Call continuation winners are not separable from late/exhausted call failures.",
            "hold_exit_logic": "Baseline Protocol101 exit unless confirmed-MFE runner audit authorizes extension.",
            "forbidden_states": "No late continuation after evidence of exhaustion or wide/stale quotes.",
            "data_needed": "Call-specific path taxonomy, underlying momentum, spread, moneyness, time bucket, MFE capture.",
            "falsification_test": "Call-specific rule fails to improve Protocol101 calls under matched controls and delay stress.",
            "expected_edge": "Capture cleaner directional call continuation without broadening noisy OTM exposure.",
            "expected_failure_mode": "Buys late highs and increases hard-stop/MAE risk.",
            "protocol101_comparison": "Must improve call-only incremental utility over Protocol101.",
            "required_diagnostic_artifact": "PROTOCOL101_SIDE_SPECIFIC_STRATEGY_AUDIT_V1",
            "next_action": "Analyze call-only winners/losers by trend and exit archetype.",
            "model_training_allowed": False,
        },
        {
            "playbook_id": "LATE_AFTERNOON_CONTINUATION_PLAYBOOK_V1",
            "track": "B_new_alpha_playbooks",
            "status": "diagnostic_only",
            "thesis": "Late-afternoon continuation may deserve its own entry and lifecycle rules.",
            "market_mechanism": "0DTE late-day gamma can produce fast continuation or compression; edge may differ from post-open morning.",
            "entry_context": "Late-afternoon SPXW candidates within live-feasible quote freshness and spread bounds.",
            "entry_trigger": "Late-day continuation context confirms directional move with acceptable spread and time-to-close.",
            "invalidation": "Late-afternoon edge is dominated by a few exposed days or vanishes after delay/fill stress.",
            "hold_exit_logic": "Shorter time-to-close aware exit, forced-flat guard, no stale quote extension.",
            "forbidden_states": "No late-day entry when spread/freshness or time-to-close constraints fail.",
            "data_needed": "Late-afternoon trade atlas, timing fragility, quote freshness, time-to-close, post-exit path.",
            "falsification_test": "Late-afternoon playbook fails after slippage/delay and day-block concentration checks.",
            "expected_edge": "Find a separate high-convexity window without broad all-day trading.",
            "expected_failure_mode": "Compression/chop and fill slippage erase replay edge.",
            "protocol101_comparison": "Must beat Protocol101 late-afternoon trades in declared conditions.",
            "required_diagnostic_artifact": "PROTOCOL101_TIME_BUCKET_PLAYBOOK_AUDIT_V1",
            "next_action": "Separate post-open and late-afternoon archetypes with delay/fill stress.",
            "model_training_allowed": False,
        },
        {
            "playbook_id": "LOWER_PREMIUM_CONVEX_ADDON_PLAYBOOK_V1",
            "track": "B_new_alpha_playbooks",
            "status": "high_risk_diagnostic_only",
            "thesis": "Lower-premium convex trades may be valid only under narrow, causally identifiable regimes.",
            "market_mechanism": "Cheaper ATM/OTM contracts provide convexity but are vulnerable to churn, spread, and wipeout.",
            "entry_context": "Full-surface lower-premium candidates not normally favored by Protocol101.",
            "entry_trigger": "Only if causal regime filters distinguish winners from cheap-option noise.",
            "invalidation": "Higher PnL comes with worse PF, drawdown, churn, or timing sensitivity.",
            "hold_exit_logic": "Likely separate exit logic with tight invalidation; do not reuse blindly.",
            "forbidden_states": "No pure return-on-premium objective and no broad OTM expansion.",
            "data_needed": "Full-surface candidates, premium/moneyness buckets, spread, delta/gamma/theta, delay/fill stress.",
            "falsification_test": "Matched lower-premium regime fails Protocol101 quality bars or loses under delay/fill stress.",
            "expected_edge": "Capture rare convex add-on without corrupting Protocol101's high-quality core.",
            "expected_failure_mode": "Recreates Protocol240/248 higher-PnL lower-quality profile.",
            "protocol101_comparison": "Must beat Protocol101 quality-adjusted utility, not just gross PnL.",
            "required_diagnostic_artifact": "LOWER_PREMIUM_CONVEX_REGIME_AUDIT_V1",
            "next_action": "Use Protocol248/270 evidence to separate valid lower-premium regimes from hindsight bait.",
            "model_training_allowed": False,
        },
        {
            "playbook_id": "PLAYBOOK_AWARE_REPLACEMENT_POLICY_V1",
            "track": "C_replacement_ml_stack",
            "status": "blocked_until_track_a_b_complete",
            "thesis": "A replacement model should choose among wait, Protocol101, and named playbook actions using baseline-relative utility.",
            "market_mechanism": "The model should allocate the single slot only when a named playbook beats Protocol101/defer after uncertainty and execution penalties.",
            "entry_context": "Candidate-set decision state with Protocol101 action and playbook candidate actions present.",
            "entry_trigger": "Playbook action allowed only when its estimated incremental utility clears risk, slot, and fill charges.",
            "invalidation": "Named playbooks fail diagnostics or model cannot defer safely to Protocol101.",
            "hold_exit_logic": "Separate playbook-aware lifecycle states: hold, exit, runner transition, giveback guard.",
            "forbidden_states": "No anonymous rowwise return maximizer; no raw-PnL-only objective; no future/path/oracle input columns.",
            "data_needed": "Completed Track A/B diagnostics, strategy registry labels, baseline actions, candidate sets, fill/timing evidence, strategy matrix.",
            "falsification_test": "Fails to beat Protocol101 on untouched data under strict serial replay and stress assumptions.",
            "expected_edge": "Replace Protocol101 only where named playbooks prove incremental utility.",
            "expected_failure_mode": "Overfits exposed splits or learns broad challenger behavior with worse quality.",
            "protocol101_comparison": "Protocol101 is always available as the defer/control action.",
            "required_diagnostic_artifact": "TRACK_A_B_COMPLETION_PACKET_V1",
            "next_action": "Do not train until required playbook diagnostics and validation gates pass.",
            "model_training_allowed": False,
        },
    ]
    return pd.DataFrame(rows, columns=REGISTRY_COLUMNS)


def validate_strategy_registry(registry: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    missing = [col for col in REGISTRY_COLUMNS if col not in registry.columns]
    if missing:
        errors.append(f"missing_registry_columns:{','.join(missing)}")
        return errors
    if registry.empty:
        errors.append("empty_strategy_registry")
    for col in REGISTRY_COLUMNS:
        if registry[col].isna().any() or registry[col].astype(str).str.strip().eq("").any():
            errors.append(f"blank_registry_column:{col}")
    if registry["playbook_id"].duplicated().any():
        errors.append("duplicate_playbook_id")
    if registry["model_training_allowed"].astype(bool).any():
        errors.append("model_training_allowed_before_diagnostics")
    return errors


def build_protocol101_weakness_matrix(forensics_summary: dict[str, Any]) -> pd.DataFrame:
    headline = forensics_summary.get("headline", {})
    score = forensics_summary.get("score_calibration", {})
    slot = forensics_summary.get("slot_opportunity_status", {})
    rows = [
        {
            "weakness_id": "execution_realism",
            "priority": 1,
            "question": "Does Protocol101 survive latency, quote freshness, spread, and fill probability?",
            "current_evidence": "Protocol101 forensics and prior timing audits show severe delay fragility; fill readiness has zero fills.",
            "status": "blocked_missing_fill_and_latency_distribution",
            "source_artifact": "v4/audit/autoresearch/protocol101_strategy_forensics_packet_v1/execution_fragility_by_archetype.csv",
            "next_diagnostic": "PROTOCOL101_EXECUTION_REALISM_BY_ARCHETYPE_V1",
            "success_criteria": "Positive Protocol101 edge by archetype under observed latency/fill assumptions.",
            "replacement_implication": "No replacement model can be trusted until execution realism is calibrated.",
        },
        {
            "weakness_id": "hard_stop_losses",
            "priority": 2,
            "question": "Are hard-stop losses avoidable before entry or after early MFE appears?",
            "current_evidence": f"{headline.get('hard_stop_trades', 0)} hard-stop trades for ${headline.get('hard_stop_pnl', 0):,.0f}; many have early MFE.",
            "status": "ready_for_manual_classification",
            "source_artifact": "v4/audit/autoresearch/protocol101_strategy_forensics_packet_v1/hard_stop_autopsy.csv",
            "next_diagnostic": "PROTOCOL101_HARD_STOP_AUTOPSY_V1",
            "success_criteria": "Classify every hard stop into pre-entry avoidable, path-management avoidable, execution artifact, or unavoidable.",
            "replacement_implication": "May justify a narrow rejection gate or early lifecycle guard.",
        },
        {
            "weakness_id": "losing_days",
            "priority": 3,
            "question": "Are worst days caused by hard stops, many small losses, execution artifacts, or regime failures?",
            "current_evidence": "Worst losing days include multi-small-loss days without hard stops.",
            "status": "ready_for_manual_classification",
            "source_artifact": "v4/audit/autoresearch/protocol101_strategy_forensics_packet_v1/losing_day_autopsy.csv",
            "next_diagnostic": "PROTOCOL101_LOSING_DAY_AUTOPSY_V1",
            "success_criteria": "Identify recurring avoidable loss regimes without using future day outcome as input.",
            "replacement_implication": "May define forbidden states or day-regime defer rules.",
        },
        {
            "weakness_id": "runner_giveback",
            "priority": 4,
            "question": "Which Protocol101 exits are early, late, or accidentally right?",
            "current_evidence": "In-trade MFE/giveback proxy exists, but post-exit path is missing.",
            "status": "blocked_missing_post_exit_path_attachment",
            "source_artifact": "v4/audit/autoresearch/protocol101_strategy_forensics_packet_v1/runner_giveback_proxy.csv",
            "next_diagnostic": "PROTOCOL101_CONFIRMED_MFE_RUNNER_AUDIT_V1",
            "success_criteria": "Runner candidates improve after post-exit path, slippage, and slot-cost charges.",
            "replacement_implication": "May justify a runner overlay instead of new entry model.",
        },
        {
            "weakness_id": "score_reliability",
            "priority": 5,
            "question": "Does Protocol101 score margin calibrate realized value?",
            "current_evidence": f"Selected-trade Spearman score-margin vs PnL {score.get('spearman_score_margin_vs_pnl', 0):.4f}; vs MFE {score.get('spearman_score_margin_vs_mfe', 0):.4f}.",
            "status": "proxy_only_rejected_candidates_missing",
            "source_artifact": "v4/audit/autoresearch/protocol101_strategy_forensics_packet_v1/score_calibration.csv",
            "next_diagnostic": "PROTOCOL101_SCORE_RELIABILITY_V1",
            "success_criteria": "Score/margin reliability is measured across selected and rejected same-event candidates.",
            "replacement_implication": "Determines whether score can support defer/risk logic.",
        },
        {
            "weakness_id": "internal_slot_cost",
            "priority": 6,
            "question": "Do Protocol101 entries block better later Protocol101 opportunities?",
            "current_evidence": slot.get("reason", "Baseline attachment records deployed state only."),
            "status": slot.get("status", "blocked_missing_counterfactual_flat_protocol101_actions"),
            "source_artifact": "v4/audit/autoresearch/unified_protocol101_baseline_attachment/protocol101_baseline_event_actions_training_scope.parquet",
            "next_diagnostic": "PROTOCOL101_INTERNAL_SLOT_COST_V1",
            "success_criteria": "Every Protocol101 open interval has counterfactual-flat later-entry opportunity cost.",
            "replacement_implication": "May justify a Protocol101 defer overlay or playbook slot-pricing model.",
        },
        {
            "weakness_id": "missed_winner_abstention",
            "priority": 7,
            "question": "Is Protocol101 too narrow or correctly abstaining?",
            "current_evidence": "Broader challengers found more PnL but lower quality; full-surface rows exist for matched controls.",
            "status": "partially_ready_requires_matched_controls",
            "source_artifact": "v4/audit/autoresearch/v4_aplus_hypothesis_270_full_surface_action_advantage_dataset/full_surface_action_advantage.parquet",
            "next_diagnostic": "PROTOCOL101_MISSED_WINNER_ABSTENTION_AUDIT_V1",
            "success_criteria": "Rejected winners remain separable from rejected losers using only causal features.",
            "replacement_implication": "Decides whether to widen Protocol101 or define a second playbook.",
        },
        {
            "weakness_id": "side_asymmetry",
            "priority": 8,
            "question": "Do calls and puts require separate playbooks?",
            "current_evidence": "Puts have higher average PnL in seed-1 replay but dominate hard-stop losses.",
            "status": "ready_for_side_specific_audit",
            "source_artifact": "v4/audit/autoresearch/protocol101_strategy_forensics_packet_v1/trade_archetype_cube.csv",
            "next_diagnostic": "PROTOCOL101_SIDE_SPECIFIC_STRATEGY_AUDIT_V1",
            "success_criteria": "Side-specific rules improve one side without degrading the other across diagnostic splits.",
            "replacement_implication": "May require side-conditioned playbook labels or model heads.",
        },
        {
            "weakness_id": "validation_overfit",
            "priority": 9,
            "question": "Are exposed split wins real or selected from repeated experiments?",
            "current_evidence": "Protocol273 classifies Q3/Q4/Q1/March/recent as repeated-research diagnostics, not sacred holdouts.",
            "status": "blocked_missing_strategy_matrix_pbo_cscv",
            "source_artifact": "v4/audit/autoresearch/v4_aplus_hypothesis_273_model_selection_overfit_risk/report.md",
            "next_diagnostic": "FORMAL_STRATEGY_MATRIX_PBO_CSCV_V1",
            "success_criteria": "Candidate remains strong after strategy-matrix, bootstrap, concentration, and untouched block scoring.",
            "replacement_implication": "Blocks all paper-default replacement claims.",
        },
    ]
    return pd.DataFrame(rows)


def build_research_data_layer(fill_summary: dict[str, Any], holdout_summary: dict[str, Any]) -> pd.DataFrame:
    fill_readiness = fill_summary.get("readiness", {})
    holdout = holdout_summary.get("reservation", {})
    rows = [
        {
            "data_domain": "protocol101_selected_trades",
            "current_source": "v4/audit/autoresearch/v4_aplus_hypothesis_113_protocol101_trade_charts/trades.csv",
            "current_status": "available",
            "coverage_notes": "Seed-1 paper-account selected trades with MFE/MAE, exit reason, premium, score, and bid/ask replay fields.",
            "required_for_playbooks": "hard-stop, losing-day, side-specific, score proxy, scalp taxonomy",
            "missing_data_or_blocker": "Does not include rejected candidates or post-exit forced-flat path.",
            "next_data_action": "Keep as current Protocol101 trade-atlas source of truth.",
            "broad_purchase_allowed": False,
        },
        {
            "data_domain": "full_surface_candidates",
            "current_source": "v4/audit/autoresearch/v4_aplus_hypothesis_270_full_surface_action_advantage_dataset/full_surface_action_advantage.parquet",
            "current_status": "available_for_diagnostics",
            "coverage_notes": "Large ATM +/- $50 SPXW candidate surface with action-advantage labels.",
            "required_for_playbooks": "missed-winner, lower-premium convex add-on, candidate-set replacement model",
            "missing_data_or_blocker": "Oracle/path labels must not define causal similarity or model inputs.",
            "next_data_action": "Join to Protocol101 baseline actions and build matched rejected-candidate controls.",
            "broad_purchase_allowed": False,
        },
        {
            "data_domain": "protocol101_baseline_actions",
            "current_source": "v4/audit/autoresearch/unified_protocol101_baseline_attachment/protocol101_baseline_event_actions_training_scope.parquet",
            "current_status": "available_but_deployed_state_only",
            "coverage_notes": "Contains wait/enter/holding/exit_then_wait state under deployed Protocol101 account state.",
            "required_for_playbooks": "baseline-relative utility, missed-winner matching, slot-cost diagnostics",
            "missing_data_or_blocker": "Does not provide counterfactual-flat Protocol101 entries during holding intervals.",
            "next_data_action": "Build counterfactual-flat Protocol101 action replay.",
            "broad_purchase_allowed": False,
        },
        {
            "data_domain": "timing_delay_stress",
            "current_source": "v4/audit/autoresearch/v4_aplus_hypothesis_114_protocol101_skeptical_falsification/delay_stress_rows.csv",
            "current_status": "available_historical",
            "coverage_notes": "Historical one-minute delay stress rows; Protocol126/117 provide additional timing evidence.",
            "required_for_playbooks": "execution realism, timing fragility, forbidden-action map",
            "missing_data_or_blocker": "Needs observed latency distribution and live/paper fill evidence.",
            "next_data_action": "Build latency-distribution replay once live/no-order timing logs are sufficient.",
            "broad_purchase_allowed": False,
        },
        {
            "data_domain": "fill_observations",
            "current_source": "v4/audit + v4/logs/paper_trading",
            "current_status": fill_readiness.get("status", "blocked_insufficient_fill_observations"),
            "coverage_notes": f"{fill_readiness.get('fill_observations', 0)} fill observations; required {fill_readiness.get('required_fill_observations', 30)}.",
            "required_for_playbooks": "execution realism, stochastic fill model, promotion-grade replay",
            "missing_data_or_blocker": fill_readiness.get("reason", "no paper/live fill rows found"),
            "next_data_action": "Collect stratified no-order/paper fill evidence by side, premium, spread, quote age, and time bucket.",
            "broad_purchase_allowed": False,
        },
        {
            "data_domain": "untouched_evaluation_block",
            "current_source": "v4/audit/autoresearch/unified_untouched_holdout_reservation/summary.json",
            "current_status": holdout.get("status", "reserved_pending_collection"),
            "coverage_notes": "Reserved for final evaluation only after policy, labels, metrics, and thresholds are frozen.",
            "required_for_playbooks": "final replacement claim",
            "missing_data_or_blocker": "Data pending collection; forbidden for feature, threshold, objective, sizing, and model selection.",
            "next_data_action": "Do not score until candidate is frozen and all promotion blockers are closed.",
            "broad_purchase_allowed": False,
        },
        {
            "data_domain": "formal_strategy_matrix",
            "current_source": "v4/audit/autoresearch/v4_aplus_hypothesis_273_model_selection_overfit_risk",
            "current_status": "blocked_missing_pbo_cscv_matrix",
            "coverage_notes": "Split exposure inventory exists; cross-strategy PBO/CSCV matrix does not.",
            "required_for_playbooks": "promotion-grade validation and false-discovery control",
            "missing_data_or_blocker": "No common daily PnL matrix across frozen strategies/seeds/overlays.",
            "next_data_action": "Build strategy matrix before any replacement packet.",
            "broad_purchase_allowed": False,
        },
        {
            "data_domain": "trader_research_inputs",
            "current_source": "user trading experience, trading books, ChatGPT research mode",
            "current_status": "available_unstructured",
            "coverage_notes": "Useful for hypothesis generation, not direct evidence.",
            "required_for_playbooks": "new alpha playbook ideation and mechanism naming",
            "missing_data_or_blocker": "Must be converted into falsifiable registry rows before experiments.",
            "next_data_action": "Create playbook notes that map each idea to entry, invalidation, exit, risk, and falsification fields.",
            "broad_purchase_allowed": False,
        },
    ]
    return pd.DataFrame(rows)


def build_stage_gate_rulebook() -> pd.DataFrame:
    rows = [
        {
            "gate_id": "G1_named_playbook",
            "stage": "hypothesis",
            "requirement": "Every experiment names a playbook and trader mechanism.",
            "status": "pass_registry_seeded",
            "blocks": "anonymous_model_experiments",
        },
        {
            "gate_id": "G2_protocol101_forensics",
            "stage": "diagnostic",
            "requirement": "Protocol101 weakness matrix must identify which decision changes are justified.",
            "status": "partial_packet_exists_more_diagnostics_required",
            "blocks": "replacement_training",
        },
        {
            "gate_id": "G3_no_future_inputs",
            "stage": "dataset",
            "requirement": "No model inputs may include future/path/oracle columns.",
            "status": "required_for_future_model",
            "blocks": "model_training",
        },
        {
            "gate_id": "G4_execution_realism",
            "stage": "replay",
            "requirement": "Timing/fill assumptions must be calibrated or explicitly stress-tested.",
            "status": "blocked_zero_fill_observations",
            "blocks": "promotion_and_replacement_claims",
        },
        {
            "gate_id": "G5_strategy_matrix",
            "stage": "validation",
            "requirement": "Formal strategy matrix with daily returns/deltas and PBO/CSCV-style checks.",
            "status": "blocked_missing_strategy_matrix",
            "blocks": "untouched_scoring",
        },
        {
            "gate_id": "G6_untouched_holdout",
            "stage": "final_evaluation",
            "requirement": "Frozen candidate scored once on newly reserved untouched data.",
            "status": "blocked_pending_new_data_collection",
            "blocks": "paper_default_replacement",
        },
        {
            "gate_id": "G7_protocol101_default",
            "stage": "operations",
            "requirement": "Protocol101 remains paper default until explicit replacement decision packet.",
            "status": "pass_default_preserved",
            "blocks": "unapproved_paper_default_change",
        },
    ]
    return pd.DataFrame(rows)


def build_replacement_candidate_protocol_spec(registry: pd.DataFrame, rulebook: pd.DataFrame) -> dict[str, Any]:
    blocked_gates = rulebook[rulebook["status"].astype(str).str.startswith("blocked")]
    return {
        "protocol_id": "PLAYBOOK_AWARE_REPLACEMENT_CANDIDATE_PROTOCOL_V1",
        "candidate_label": "CHALLENGER_PLAYBOOK_AWARE_REPLACEMENT_POLICY_V1",
        "status": "not_training_allowed",
        "paper_default_baseline": "PAPER_DEFAULT_PROTOCOL101",
        "goal": "eventually replace Protocol101 only where named playbooks prove incremental utility under the same live-like serial game",
        "actions": [
            "wait",
            "take_protocol101_action",
            "enter_named_playbook_candidate",
            "hold",
            "exit",
            "runner_transition",
            "defer_to_protocol101",
        ],
        "objective": "baseline_relative_incremental_utility_after_slot_timing_fill_drawdown_tail_and_uncertainty_charges",
        "optimization_target": "incremental_utility_over_protocol101_not_raw_pnl",
        "must_include": [
            "slot_opportunity_cost",
            "timing_and_fill_penalty",
            "drawdown_and_tail_penalty",
            "uncertainty_defer_behavior",
            "strict_one_account_serial_replay",
            "named_playbook_eligibility",
        ],
        "model_shape": {
            "candidate_set_encoder": "required",
            "playbook_conditioning": "required",
            "rowwise_candidate_scorer_only": "forbidden",
            "distributional_risk_heads": [
                "downside_quantile",
                "giveback_probability",
                "timing_fill_sensitivity",
                "catastrophic_loss_probability",
                "uncertainty_or_ood",
            ],
        },
        "required_inputs": [
            "Protocol101 proposed action",
            "candidate set with live-feasible masks",
            "named playbook eligibility flags",
            "account state and slot occupancy",
            "quote freshness and spread",
            "side/time/premium/moneyness/Greeks/context features",
        ],
        "forbidden_inputs": [
            "future PnL",
            "path MFE/MAE as pre-entry model input",
            "oracle action",
            "future best exit",
            "realized exit reason",
            "untouched holdout-derived feature or threshold",
        ],
        "training_allowed": False,
        "blocked_by_gates": blocked_gates["gate_id"].tolist(),
        "allowed_work_now": [
            "diagnostics",
            "registry refinement",
            "data inventory",
            "counterfactual table construction",
            "manual playbook notes",
            "non-training replay/audit scaffolds",
        ],
        "playbooks_seeded": registry["playbook_id"].tolist(),
    }


def validate_replacement_program(
    registry: pd.DataFrame,
    weakness: pd.DataFrame,
    data_layer: pd.DataFrame,
    rulebook: pd.DataFrame,
    spec: dict[str, Any],
) -> list[str]:
    errors = validate_strategy_registry(registry)
    if weakness.empty:
        errors.append("empty_weakness_matrix")
    if data_layer.empty:
        errors.append("empty_research_data_layer")
    if rulebook.empty:
        errors.append("empty_stage_gate_rulebook")
    if spec.get("training_allowed") is not False:
        errors.append("replacement_spec_training_allowed")
    if "PAPER_DEFAULT_PROTOCOL101" != spec.get("paper_default_baseline"):
        errors.append("protocol101_not_preserved_as_baseline")
    if not any(rulebook["status"].astype(str).str.startswith("blocked")):
        errors.append("no_blocked_gates_detected")
    forbidden = set(spec.get("forbidden_inputs", []))
    for required in {"future PnL", "oracle action", "future best exit"}:
        if required not in forbidden:
            errors.append(f"missing_forbidden_input:{required}")
    return errors


def write_report(
    output_dir: Path,
    summary: dict[str, Any],
    registry: pd.DataFrame,
    weakness: pd.DataFrame,
    data_layer: pd.DataFrame,
    rulebook: pd.DataFrame,
) -> str:
    lines = [
        f"# {ROLE_LABEL}",
        "",
        "What is this: foundation / truth-grounded Protocol101 replacement research program",
        "Does it change the paper-trading default: no",
        "Paper default baseline: `PAPER_DEFAULT_PROTOCOL101`",
        "Paid data downloaded: no",
        "Broker endpoint called: no",
        "Model training: no",
        "Untouched holdout scored: no",
        f"Decision: `{summary['decision']}`",
        "",
        "## Bottom Line",
        "",
        "Do not try another generic model. Protocol101 stays the control strategy while the replacement program converts trader ideas into named, falsifiable playbooks. A replacement model is explicitly blocked until the required diagnostics, data gates, execution evidence, validation controls, and untouched scoring path exist.",
        "",
        "## Seeded Strategy Hypotheses",
        "",
        "| playbook | track | status | next action |",
        "|---|---|---|---|",
    ]
    for _, row in registry.iterrows():
        lines.append(f"| `{row['playbook_id']}` | {row['track']} | `{row['status']}` | {row['next_action']} |")
    lines.extend(
        [
            "",
            "## Protocol101 Weakness Matrix",
            "",
            "| priority | weakness | status | next diagnostic |",
            "|---:|---|---|---|",
        ]
    )
    for _, row in weakness.sort_values("priority").iterrows():
        lines.append(f"| {int(row['priority'])} | {row['weakness_id']} | `{row['status']}` | {row['next_diagnostic']} |")
    lines.extend(
        [
            "",
            "## Research Data Layer",
            "",
            "| data domain | status | next data action |",
            "|---|---|---|",
        ]
    )
    for _, row in data_layer.iterrows():
        lines.append(f"| {row['data_domain']} | `{row['current_status']}` | {row['next_data_action']} |")
    lines.extend(
        [
            "",
            "## Stage Gates",
            "",
            "| gate | status | blocks |",
            "|---|---|---|",
        ]
    )
    for _, row in rulebook.iterrows():
        lines.append(f"| {row['gate_id']} | `{row['status']}` | {row['blocks']} |")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Summary: `{output_dir / 'summary.json'}`",
            f"- Strategy hypothesis registry: `{output_dir / 'strategy_hypothesis_registry.csv'}`",
            f"- Protocol101 weakness matrix: `{output_dir / 'protocol101_weakness_matrix.csv'}`",
            f"- Research data layer: `{output_dir / 'research_data_layer.csv'}`",
            f"- Stage-gate rulebook: `{output_dir / 'stage_gate_rulebook.csv'}`",
            f"- Replacement candidate protocol spec: `{output_dir / 'replacement_candidate_protocol_spec.json'}`",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    forensics_summary = read_json(Path(args.forensics_dir) / "summary.json")
    fill_summary = read_json(args.fill_summary)
    overfit_summary = read_json(args.overfit_summary)
    holdout_summary = read_json(args.holdout_summary)
    learned_defer_summary = read_json(args.learned_defer_summary)

    registry = build_strategy_hypothesis_registry()
    weakness = build_protocol101_weakness_matrix(forensics_summary)
    data_layer = build_research_data_layer(fill_summary, holdout_summary)
    rulebook = build_stage_gate_rulebook()
    spec = build_replacement_candidate_protocol_spec(registry, rulebook)
    validation_errors = validate_replacement_program(registry, weakness, data_layer, rulebook, spec)

    registry.to_csv(output_dir / "strategy_hypothesis_registry.csv", index=False)
    weakness.to_csv(output_dir / "protocol101_weakness_matrix.csv", index=False)
    data_layer.to_csv(output_dir / "research_data_layer.csv", index=False)
    rulebook.to_csv(output_dir / "stage_gate_rulebook.csv", index=False)
    (output_dir / "replacement_candidate_protocol_spec.json").write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")

    blocked_gates = rulebook[rulebook["status"].astype(str).str.startswith("blocked")]["gate_id"].tolist()
    summary = {
        "role_label": ROLE_LABEL,
        "what_is_this": "foundation / truth-grounded Protocol101 replacement research program",
        "changes_paper_default": False,
        "paper_default_baseline": "PAPER_DEFAULT_PROTOCOL101",
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "model_training": False,
        "untouched_holdout_scored": False,
        "decision": "truth_grounded_replacement_program_created_training_and_replacement_blocked",
        "program_tracks": {
            "track_a": "Protocol101 forensics",
            "track_b": "new alpha playbooks",
            "track_c": "playbook-aware replacement ML stack",
        },
        "counts": {
            "playbooks": int(len(registry)),
            "weaknesses": int(len(weakness)),
            "data_domains": int(len(data_layer)),
            "blocked_gates": int(len(blocked_gates)),
        },
        "blocked_gates": blocked_gates,
        "validation_errors": validation_errors,
        "replacement_candidate_protocol_status": spec["status"],
        "source_status": {
            "forensics_decision": forensics_summary.get("decision", "missing"),
            "fill_model_decision": fill_summary.get("decision", "missing"),
            "overfit_decision": overfit_summary.get("decision", "missing"),
            "holdout_decision": holdout_summary.get("decision", "missing"),
            "learned_defer_decision": learned_defer_summary.get("decision", "missing"),
        },
        "outputs": {
            "summary": str(output_dir / "summary.json"),
            "report": str(output_dir / "report.md"),
            "doc": str(args.doc),
            "strategy_hypothesis_registry": str(output_dir / "strategy_hypothesis_registry.csv"),
            "protocol101_weakness_matrix": str(output_dir / "protocol101_weakness_matrix.csv"),
            "research_data_layer": str(output_dir / "research_data_layer.csv"),
            "stage_gate_rulebook": str(output_dir / "stage_gate_rulebook.csv"),
            "replacement_candidate_protocol_spec": str(output_dir / "replacement_candidate_protocol_spec.json"),
        },
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    report = write_report(output_dir, summary, registry, weakness, data_layer, rulebook)
    (output_dir / "report.md").write_text(report)
    Path(args.doc).write_text(report)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    parser.add_argument("--forensics-dir", type=Path, default=DEFAULT_FORENSICS_DIR)
    parser.add_argument("--fill-summary", type=Path, default=DEFAULT_FILL_SUMMARY)
    parser.add_argument("--overfit-summary", type=Path, default=DEFAULT_OVERFIT_SUMMARY)
    parser.add_argument("--holdout-summary", type=Path, default=DEFAULT_HOLDOUT_SUMMARY)
    parser.add_argument("--learned-defer-summary", type=Path, default=DEFAULT_LEARNED_DEFER_SUMMARY)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
