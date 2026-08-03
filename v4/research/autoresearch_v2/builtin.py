"""Built-in first-wave hypotheses emitted as typed JSON before execution."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .dataset import FEATURE_INVENTORY, POLICIES, SIGNED17
from .schema import SCHEMA_VERSION, TERMINAL_STATUSES


REFERENCE_EXIT = POLICIES[4]
SHORT25_EXIT = POLICIES[1]


def _features() -> list[dict[str, str]]:
    return [
        {"name": name, **FEATURE_INVENTORY[name]}
        for name in SIGNED17
    ]


def _base(hypothesis_id: str, claim: str, component: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": hypothesis_id,
        "claim": claim,
        "component": component,
        "feature_families": {"add": ["signed17"], "remove": []},
        "features": _features(),
        "target": {
            "kind": "single_policy_pnl",
            "fields": [REFERENCE_EXIT],
            "weights": [1.0],
            "frozen_reference_exit": REFERENCE_EXIT,
            "required_materialized_fields": [
                "policy_net_pnl",
                "policy_mid_pnl",
                "realized_exit_time",
                "source_exit_quote_time",
                "exit_quote_age",
                "exit_reason",
                "executable_exit_bid",
                "policy_deadline",
            ],
        },
        "arms": ["candidate", "paired_baseline"],
        "primary_contrast": "candidate",
        "threshold": {"kind": "fixed", "value": 20.0, "fit_role": "none"},
        "paired_baseline": {"name": "component_specific", "exposure_matching": []},
        "model": {
            "family": "hist_gradient_boosting",
            "hyperparameters": {
                "loss": "squared_error",
                "learning_rate": 0.05,
                "max_iter": 100,
                "max_leaf_nodes": 31,
                "max_depth": 3,
                "min_samples_leaf": 80,
                "l2_regularization": 1.0,
                "max_bins": 255,
                "early_stopping": False,
            },
            "seeds": [101],
            "max_fits": 5,
        },
        "development_epoch": {
            "epoch_id": "development_2025-08-01_2026-03-31_two_clock",
            "session_manifest": (
                "v4/audit/autoresearch/"
                "protocol101_pathd_entry_exit_model_research_corrected_v3_2_2026_08_01/"
                "session_assignments.json"
            ),
            "allowed_role": "development",
            "confirmation_rule": "fresh_epoch_once_then_roll_to_development",
        },
        "power": {
            "alpha": 0.05,
            "power": 0.80,
            "practical_effect_dollars_per_session": 10.0,
            "maximum_mde_dollars_per_session": 25.0,
        },
        "compute_budget": {"max_permutations": 4999, "max_minutes": 45},
        "terminal_statuses": list(TERMINAL_STATUSES),
    }


def builtin_hypotheses() -> dict[str, dict[str, Any]]:
    exit_spec = _base(
        "exit_short25_vs_frozen120_v1",
        "The 25-minute exit improves matched strict-serial development economics over the frozen 120-minute reference exit.",
        "exit",
    )
    exit_spec["arms"] = [SHORT25_EXIT, REFERENCE_EXIT]
    exit_spec["primary_contrast"] = SHORT25_EXIT
    exit_spec["paired_baseline"] = {
        "name": REFERENCE_EXIT,
        "exposure_matching": ["entry_identity", "entry_count", "side", "premium", "moneyness"],
    }
    exit_spec["power"]["practical_effect_dollars_per_session"] = 9.0

    timing = _base(
        "entry_timing_now_wait5_wait15_v1",
        "Waiting 5 or 15 minutes after a causal entry signal improves frozen-exit value for the same contract.",
        "timing",
    )
    timing["arms"] = ["enter_now", "wait_5m", "wait_15m"]
    timing["primary_contrast"] = "wait_5m"
    timing["paired_baseline"] = {
        "name": "enter_now",
        "exposure_matching": ["signal_identity", "contract_id", "side", "moneyness"],
    }

    direction = _base(
        "entry_direction_call_vs_put_v1",
        "OOF signed features select the better call-versus-put side within a symmetric premium-matched pair.",
        "direction",
    )
    direction["arms"] = ["model_selected_side", "opposite_side"]
    direction["primary_contrast"] = "model_selected_side"
    direction["paired_baseline"] = {
        "name": "opposite_side",
        "exposure_matching": ["decision_time", "premium", "absolute_moneyness"],
    }

    contract = _base(
        "entry_contract_ranking_full_epoch_v1",
        "OOF within-minute contract ranking improves frozen-exit value beyond nearest ATM over the full two-clock development epoch.",
        "contract_choice",
    )
    contract["arms"] = ["model_ranked_contract", "nearest_atm"]
    contract["primary_contrast"] = "model_ranked_contract"
    contract["paired_baseline"] = {
        "name": "nearest_atm",
        "exposure_matching": ["decision_time", "side", "premium_caliper"],
    }

    target = _base(
        "entry_multi_policy_path_target_v1",
        "A multi-policy path-value target improves frozen-reference exit selection beyond a single fixed-exit target.",
        "target",
    )
    target["target"] = {
        **target["target"],
        "kind": "multi_policy_path_composite",
        "fields": list(POLICIES[:5]),
        "weights": [0.2] * 5,
    }
    target["arms"] = ["multi_policy_target", "single_reference_target"]
    target["primary_contrast"] = "multi_policy_target"
    target["paired_baseline"] = {
        "name": "single_reference_target",
        "exposure_matching": ["decision_time", "entry_count", "side", "premium", "moneyness"],
    }

    unavailable = _base(
        "entry_mfe_mae_ttb_heads_v1",
        "MFE, MAE, and time-to-breakeven heads improve a frozen composer.",
        "target",
    )
    unavailable["target"] = {
        **unavailable["target"],
        "kind": "mfe_mae_time_to_breakeven_heads",
        "fields": ["mfe", "mae", "time_to_breakeven"],
        "weights": [1.0, -1.0, -1.0],
        "required_materialized_fields": ["mfe", "mae", "time_to_breakeven"],
    }
    unavailable["arms"] = ["path_heads", "single_reference_target"]
    unavailable["primary_contrast"] = "path_heads"
    unavailable["paired_baseline"] = target["paired_baseline"]

    canonical_names = (
        "spx_vwap_gap_points",
        "spx_vwap_gap_bps",
        "spx_vwap_gap_over_session_range",
        "session_range_bps",
        "momentum_5m_bps",
        "momentum_15m_bps",
        "momentum_5m_over_session_range",
        "momentum_15m_over_session_range",
        "omar_clipped_neg3_pos3",
        "vwap_side_alignment_flag",
        "omar_side_alignment_flag",
        "momentum15_side_alignment_flag",
        "D.near_atm.straddle_mid_spot_bps",
        "D.near_atm.put_call_mid_ratio",
        "D.near_atm.side_smile_slope_bps_per_5pt",
        "E.bs.delta",
        "E.bs.gamma",
    )
    exact_path = _base(
        "entry_policy_neutral_m0_m1_full_epoch_v1",
        "The exact 15-minute multi-horizon/MFE/MAE/breakeven/barrier utility lets canonical M1 rank contracts better than delta-gamma M0 across the full two-clock development epoch.",
        "contract_choice",
    )
    exact_path["feature_families"] = {
        "add": ["canonical_signed17"],
        "remove": [],
    }
    exact_path["features"] = [
        {
            "name": name,
            "family": "canonical_signed17",
            "available_at": "completed_minute_plus_60s",
            "live_twin": (
                "Protocol101 live option ladder"
                if name.startswith("E.")
                else "Protocol101 completed-minute context and decision-local ladder"
            ),
        }
        for name in canonical_names
    ]
    exact_path["target"] = {
        "kind": "policy_neutral_15m_path_utility",
        "fields": ["primary_utility"],
        "weights": [1.0],
        "frozen_reference_exit": REFERENCE_EXIT,
        "required_materialized_fields": ["policy_neutral_primary_utility"],
    }
    exact_path["arms"] = ["M1", "M0", "nearest_atm", "exact_random"]
    exact_path["primary_contrast"] = "M1"
    exact_path["paired_baseline"] = {
        "name": "M0",
        "exposure_matching": ["decision_time", "side", "premium_caliper"],
    }
    exact_path["model"]["max_fits"] = 10
    # These two legacy schema fields carry decision-local utility units here.
    exact_path["power"]["practical_effect_dollars_per_session"] = 0.01
    exact_path["power"]["maximum_mde_dollars_per_session"] = 0.01

    return {
        item["hypothesis_id"]: item
        for item in (
            exit_spec,
            timing,
            direction,
            contract,
            target,
            unavailable,
            exact_path,
        )
    }


def write_builtin_hypotheses(output_dir: str | Path) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, payload in builtin_hypotheses().items():
        path = output / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        paths.append(path)
    return paths
